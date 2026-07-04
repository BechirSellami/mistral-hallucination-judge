# Data Card — FIB Hallucination Judge

## Source
Factual Inconsistency Benchmark (FIB), `r-three/fib` on the Hugging Face Hub.
Tam et al. 2022, "Evaluating the Factual Consistency of LLMs Through Summarization"
(arXiv:2211.08412). Documents/summaries from XSum and CNN/DM.

## Task framing
Native FIB is paired-choice (each document has one consistent + one inconsistent
summary). We UNPACK each pair into two single-summary classification rows:
- label 1 = factually consistent (every claim supported by the document)
- label 0 = factually inconsistent (some claim unsupported/contradicted)
This yields a class-balanced binary classification dataset by construction.

## Splitting — THREE-WAY (train / val / test)
FIB ships test-only, so we created the split ourselves.
- **train** (70%) — fits the model.
- **val** (15%) — guides Week-4 tuning (rank/LR/target_modules, FIB vs FIB+USB).
- **test** (15%) — **SEALED. Touched ONCE in Week-5 for the headline number.**
  Do not look at test while tuning; reporting on a set you tuned against inflates results.
- DOCUMENT-LEVEL: both summaries of a document stay in ONE split (no article leakage).
- SOURCE-STRATIFIED: XSum and CNN/DM each split independently → all buckets balanced.
- Verified pairwise-disjoint document sets. seed=42.

## Statistics
| split | rows | documents | label 1 | label 0 | XSum | CNN/DM | other |
|-------|------|-----------|---------|---------|------|--------|-------|
| train | 836 | 418 | 418 | 418 | 700 | 136 | 0 |
| val | 178 | 89 | 89 | 89 | 150 | 28 | 0 |
| test | 178 | 89 | 89 | 89 | 150 | 28 | 0 |

## Formatting
- Mistral chat template via `tokenizer.apply_chat_template`.
- Prompt-loss masking: only the assistant JSON answer contributes to the loss
  (prompt tokens set to label -100). Base model for the template: `mistralai/Mistral-7B-Instruct-v0.3`.

## Evaluation protocol (Week 5)
- Report F1 / precision / recall on **test** (never on val or train).
- Report SEPARATELY per source (XSum vs CNN/DM) — FIB README is explicit that the
  two sources have different inconsistency characteristics.
- Also evaluate on **USB (out-of-domain, Wikipedia)** via `load_usb.py` for a
  generalization claim, not just in-domain held-out performance.
- Compare: fine-tuned vs base (zero/few-shot) vs a strong prompted baseline.

## Known limitations / caveats
- FIB inconsistent summaries are model-GENERATED for XSum but model-EXTRACTED for
  CNN/DM — different inconsistency characteristics; hence per-source reporting.
- Small benchmark (~3.5k pairs → ~7k rows); good for a focused judge, not a
  general-purpose factuality model.
- Long news documents — watch max_len during training; truncation that cuts the
  text a claim depends on injects label noise. See sequence-length section below.

## Sequence length (train split, tokenized) — drives max_len
Mean 551 tokens. Percentiles: p50=550 | p75=658 | p90=736 | p95=810 | p99=1026 | p100=1120.

| max_len | rows truncated |
|---------|----------------|
| 512 | 57.7% |
| 768 | 6.7% |
| 1024 | 1.1% |
| 1536 | 0.0% |
| 2048 | 0.0% |

**Recommended max_len = 1536** (smallest cap truncating <1% of rows). Truncation on this task = label noise, so keep it low.
