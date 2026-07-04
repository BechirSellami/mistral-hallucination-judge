# QLoRA Training — Judgment About the Knobs (hands-on lessons)

Learned by hitting each wall on a real run (Mistral-7B QLoRA, FIB judge, Colab).
These are the "owning the loop" details that separate running a cookbook from
understanding the pipeline.

## Precision: bf16 vs fp16 (the one that cost the most)
- **Modern models (Mistral/Llama/Qwen) are bf16-native. Train with `bf16=True`, not `fp16=True`.**
- Setting `fp16=True` on bf16 weights crashes: `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16` — the fp16 GradScaler can't unscale bf16 grads.
- **T4 trap:** `is_bf16_supported()` returns True on a T4, but bf16 runs via *slow emulation* (Turing, cc 7.5). fp16 is the T4's fast path — but crashes with bf16 weights. So a T4 has no good option: fast-but-crashes vs safe-but-slow.
- **Gate on compute capability, not `is_bf16_supported()`:** bf16 is only hardware-fast on Ampere+ (cc ≥ 8.0: A100/L4/RTX30+).
- **Takeaway:** the L4 (cc 8.9) makes bf16 native+fast and removes the whole dilemma. Don't fight a T4 on precision — upgrade.

## Memory levers (the OOM ladder)
Ordered from most memory-hungry/fastest to most frugal/slowest:
1. `gradient_checkpointing=False` + large batch — fastest, most memory. (OOM'd batch 4 on 24GB.)
2. `gradient_checkpointing=False` + `batch_size=2` — the stable sweet spot found here.
3. `gradient_checkpointing=False` + `batch_size=1`.
4. `gradient_checkpointing=True` + `batch_size=2` — most frugal, ~30% slower (recompute tax).
- **Gradient checkpointing** trades compute for memory by recomputing activations in the backward pass. A *necessity* on 15GB (T4); *optional* on 24GB (L4) — turn it off for speed when you have headroom.
- **Effective batch = `batch_size × grad_accum`.** Keep it constant (e.g. 16) while tuning batch_size for the memory/speed tradeoff: bs=1/accum=16 and bs=4/accum=4 train the same, at different speed/memory.
- **The OOM was in the loss, not the model:** TRL casts logits to fp32 over `seq_len × vocab (~32k)`. That fp32 matmul is the memory high-water mark and a big chunk of step time — near-total waste for a judge task where only a few answer tokens matter.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** reduces fragmentation. Set it *before* launch. When OOM is marginal (we were over by 256MB with 547MB reserved-but-unallocated = fragmentation), this alone can rescue a bigger batch.

## Sequence length (data-driven, not guessed)
- **Profile token lengths, then set `max_len` at ~p95–p99.** For FIB: p99=1001, max=1218 → `max_len=1024` truncates only 0.9% of rows. 2048 was pure waste (~830 padding tokens/example, quadratic attention cost).
- **Truncation on a judge task = label noise:** cutting the document may remove the evidence a summary claim depends on → the label becomes wrong, not just lossy. Keep truncation in single-digit %.
- The default `max_len` is a real memory *and* quality lever, not a formality.

## Correctness checks that catch silent failures
- **Overfit-10 first, always.** If loss won't → ~0 on 10 examples, the pipeline is broken (masking/LR/freeze). Catches it in 2 min vs after a multi-hour run. (Ours hit 0.0003 → pipeline validated.)
- **`print_trainable_parameters()` before training.** Expect ~0.1–1% trainable (we saw 0.1877%). If 0% → the freeze bug; if ~100% → LoRA not applied.
- **TRL + PEFT freeze bug (#3926):** pass the *quantized base* + `peft_config` and let SFTTrainer wrap LoRA once. Pre-wrapping with `get_peft_model` AND passing it can re-freeze adapters silently (loss flat, no error).
- **Loss curve read:** on the real set, want *decreasing then leveling* — not collapse to 0 (that's memorization). Overfit-10 is the only place you *want* 0.
- **Merge/reload is a separate step:** save adapter → reload in a *fresh process* → generate. Doing it in the same process hides silent save/load bugs.

## Prompt-loss masking (own it, don't rely on magic)
- Pre-tokenize with prompt tokens → `-100` (ignore index); tell TRL `skip_prepare_dataset=True`. More robust than `assistant_only_loss`, which needs `{% generation %}` template markers and only auto-patches known families.
- Collator pads labels with `-100` too, so padding is ignored by the loss — same convention as prompt masking.

## Colab / environment friction (real time sinks)
- **Train to LOCAL disk (`/content`), sync checkpoints to Drive on an interval.** Drive is a FUSE network mount — training directly to it is I/O-bound and ~50–100x slower. Point `WANDB_DIR` local too.
- **Interactive prompts (W&B/HF login) don't work under `!python script.py`.** Pre-set via env vars / secrets, or log in from a normal cell first.
- **Checkpoints survive session death only if synced.** Local `/content` is wiped on disconnect; lower `--sync-every-min` (e.g. 5) so you lose minutes, not hours. On resume, copy the Drive checkpoint back to local *before* `--resume`.
- **Cache the base model on Drive** (`HF_HOME`) so the 14.5GB base doesn't re-download every session.

## Data integrity — audit your labels, don't trust "balanced"
- **A "balanced" benchmark can be skewed after your own preprocessing.** FIB is
  balanced at the PAIR level (1 consistent + 1 inconsistent summary per doc). But
  FIB generates the inconsistent summary from MULTIPLE distractor models, so one
  document appears across several examples: same reference (consistent) summary,
  different inconsistent one each time. Unpacking to rows + deduping the identical
  consistent summaries left ~1 consistent : N inconsistent per doc → an 85/15 skew
  the source data never had.
- **The tell was in the metrics, not an error:** F1 = 0.0 with accuracy ≈ 0.85.
  High accuracy + zero F1/precision/recall = the model collapsed to always
  predicting the majority class. When accuracy ≈ the majority-class fraction,
  suspect class collapse, not "bad model." (r32/r64 collapsed; r16 survived.)
- **Diagnose before "fixing":** checked #choices/example (all 2 — ruled out
  multi-distractor-per-row) and per-doc (n_consistent, n_inconsistent) shape
  (mostly 1 : 5–7 — confirmed the asymmetric-dedup cause). Never patch a data bug
  you haven't located; you'll just move the skew.
- **Dedup vs balance are different jobs, run in order.** Dedup = remove exact
  duplicate rows (correctness). Balance = fix the class ratio. Dedup FIRST so you
  sample the balanced pair from genuinely-distinct summaries.
- **Fix (Option A): balance per document, before splitting.** Keep 1 consistent +
  1 randomly-sampled inconsistent per doc → clean 50/50, the truest form of the
  paired design; random pick preserves distractor variety across the set. Doing it
  BEFORE a document-level split means every split inherits balance for free —
  label-stratification comes automatically.
- **Alternatives considered:**
  - *Oversample the consistent row* — repeats an identical string N× → overfit risk.
  - *Keep the skew + class-weighted loss* — implementable via a `compute_loss` override
    that up-weights per-token CE at the position whose target is the minority class
    token. But it's next-token prediction, not classification: the "class" is a single
    `0`/`1` token inside the generated JSON, so weighting depends on resolving that
    token id exactly (fragile — `"1"` tokenizes differently in isolation vs. inside
    `": 1}"`). And it only rebalances the *gradient* — the eval set stays skewed, so
    metrics remain misleading.
  - *Downsample to one balanced pair per doc* (chosen) — fixes the imbalance at the
    source: clean train AND eval, honest metrics, no fragile token targeting. Simplest
    and most defensible.
- **Cost/benefit:** ~7k skewed rows → ~1.3k balanced. Smaller, but better signal —
  and "my splits are balanced, stratified, contamination-checked" is table stakes
  for a piece an interviewer might audit. Worth the re-run.

## The meta-lesson
The knobs interact: precision ↔ hardware, batch ↔ memory ↔ checkpointing ↔ speed,
max_len ↔ memory ↔ label noise. "Owning the loop" means knowing which lever to
pull for which symptom, and that the fast path and the safe path are often
different — the engineering is choosing deliberately, and recording why.

And it extends past the training loop to the DATA: a benchmark labeled "balanced"
was 85/15 after my own unpacking+dedup, and the only symptom was F1=0 hiding behind
85% accuracy. The lesson: read the confusion counts, not just the headline metric;
trace a skew to its exact cause before fixing; and audit your data with the same
rigor as your training config — it moves results more than any hyperparameter.
