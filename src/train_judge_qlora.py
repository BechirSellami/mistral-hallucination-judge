"""
train_judge_qlora.py — QLoRA SFT for the hallucination judge
====================================================================

Colab-ready. Consumes train.jsonl / eval.jsonl from prepare_fib.py
(pre-tokenized: input_ids, attention_mask, labels with prompt tokens
already masked to -100).

Design decisions baked in:
- PRE-MASKED LABELS path (we do NOT rely on TRL's assistant_only_loss):
  the data already carries correct -100 masking, which is explicit and
  robust across TRL versions.
- Overfit-10 SANITY CHECK runs FIRST. If loss won't collapse on 10
  examples, the pipeline is broken — stop and fix before the real run.
- T4-SAFE: SDPA attention (NOT flash-attn-2, unsupported on Turing),
  gradient checkpointing, 4-bit NF4, bs=1 + grad-accum, fp16 compute.
- DRIVE CHECKPOINTING: output_dir on mounted Drive so a killed Colab
  session costs minutes, not the whole run. resume_from_checkpoint aware.
- W&B tracking wired (set WANDB_API_KEY or pass --no-wandb).
- Avoids the SFTTrainer+PeftModel double-prepare freeze bug (see TRL
  issue #3926): we pass the QUANTIZED BASE + peft_config and let the
  trainer wrap it ONCE. We never pre-wrap with get_peft_model.

Colab quickstart (in a cell):
    !pip install -q -U "transformers>=4.45" "trl>=0.12" "peft>=0.13" \
        "bitsandbytes>=0.44" accelerate datasets wandb
    from google.colab import drive; drive.mount('/content/drive')
    !python train_judge_qlora.py \
        --train ./data/train.jsonl --eval ./data/eval.jsonl \
        --base-model mistralai/Mistral-7B-Instruct-v0.3 \
        --output-dir /content/drive/MyDrive/judge_ckpts/run1

Pin versions if anything below errors — TRL's SFTConfig arg names move.
"""

import argparse
import glob
import json
import os
import shutil
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Drive-sync callback — fast local training, periodic safe copy to Drive
# ---------------------------------------------------------------------------
class DriveSyncCallback(TrainerCallback):
    """Training writes to FAST local disk (/content); this callback copies new
    checkpoints to (slow, network-mounted) Google Drive on a time interval, so a
    killed Colab session loses at most `sync_every_min` minutes of progress.

    WHY THIS EXISTS: Drive in Colab is a FUSE network mount — every write is a
    remote call. Training/W&B do many tiny writes, so pointing output_dir straight
    at Drive makes each step ~50-100x slower (I/O-bound, GPU idle). We keep the hot
    path local and pay the Drive cost only occasionally, in the background-ish.
    """
    def __init__(self, local_dir: str, drive_dir: str, sync_every_min: float = 15.0):
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.sync_every = sync_every_min * 60.0
        self._last = 0.0

    def _sync(self, tag=""):
        if not self.drive_dir:
            return
        os.makedirs(self.drive_dir, exist_ok=True)
        # copy each checkpoint-* dir that isn't already on Drive (skip if present)
        for ckpt in sorted(glob.glob(os.path.join(self.local_dir, "checkpoint-*"))):
            dest = os.path.join(self.drive_dir, os.path.basename(ckpt))
            if os.path.exists(dest):
                continue
            try:
                shutil.copytree(ckpt, dest)
                print(f"[drive-sync{tag}] copied {os.path.basename(ckpt)} → Drive")
            except Exception as e:
                print(f"[drive-sync{tag}] WARN failed to copy {ckpt}: {e}")
        self._last = time.time()

    def on_save(self, args, state, control, **kwargs):
        # A checkpoint was just written locally. Sync if enough time elapsed
        # (rate-limited so back-to-back saves don't each trigger a slow Drive copy).
        if time.time() - self._last >= self.sync_every:
            self._sync()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Always do a final sync so the last checkpoint reaches Drive.
        self._sync(tag="-final")
        return control


# ---------------------------------------------------------------------------
# Model + tokenizer (4-bit NF4, T4-safe)
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(base_model: str):
    # PRECISION: bf16 is only FAST on Ampere+ (A100/L4/RTX30+, compute capability
    # >= 8.0). On a T4 (Turing, cc 7.5) is_bf16_supported() returns True but bf16
    # runs via SLOW emulation — fp16 is the T4's native fast path. So gate on the
    # actual compute capability, not just is_bf16_supported(), or the T4 silently
    # takes the slow route.
    major_cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    use_bf16 = torch.cuda.is_bf16_supported() and major_cc >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[precision] device cc major={major_cc} → "
          f"{'bf16' if use_bf16 else 'fp16'} compute")

    # 4-bit QLoRA quantization of the FROZEN base. This is what lets a 7B fit a
    # 15GB T4: weights drop to ~4-5GB, leaving room for activations + LoRA state.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4: the QLoRA-paper quant type
        bnb_4bit_use_double_quant=True,      # double quant: quantizes the quant constants too — extra memory savings
        bnb_4bit_compute_dtype=compute_dtype,  # matmuls run in fp16/bf16, not 4-bit
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        # T4-SAFE: SDPA, NOT flash_attention_2 — flash-attn-2 is unsupported on the
        # T4's Turing architecture and will error. SDPA is the portable fallback.
        attn_implementation="sdpa",
        torch_dtype=compute_dtype,
    )
    # use_cache (the KV cache) is for INFERENCE and is incompatible with gradient
    # checkpointing — leaving it on triggers a warning and wastes memory. Off for training.
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token        # many base models ship without a pad token
    tok.padding_side = "right"               # right-pad for TRAINING (left-pad is for batched generation)
    return model, tok, compute_dtype


def make_peft_config(r, alpha, dropout, target_modules):
    # The LoRA adapter spec. Only these low-rank A/B matrices train; the 4-bit
    # base stays frozen. target_modules defaults to the attention projections;
    # adding the MLP projections (gate/up/down) is a deliberate ablation knob.
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,       # attn-only by default; +MLP is a Wk4 knob
    )


# ---------------------------------------------------------------------------
# Data — already tokenized + masked by prepare_fib.py
# ---------------------------------------------------------------------------
def load_prepared(train_path, eval_path):
    # Each line is ALREADY tokenized and masked by prepare_fib.py:
    #   {"input_ids": [...], "attention_mask": [...], "labels": [...], "meta": {...}}
    # labels already have prompt tokens set to -100, so we do NOT re-template here.
    ds = load_dataset(
        "json",
        data_files={"train": train_path, "eval": eval_path},
    )
    # The collator/trainer should only see the three tensor columns. Drop "meta"
    # (and anything else) so it doesn't get fed into the model as a batch field.
    keep = {"input_ids", "attention_mask", "labels"}
    for split in ds:
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)
    return ds["train"], ds["eval"]


def compute_eval_save_steps(n_train, batch_size, grad_accum, epochs,
                            evals_per_epoch, floor=5):
    """Dynamically size eval_steps/save_steps so we get ~`evals_per_epoch`
    evaluations regardless of dataset size.

    Fixed eval_steps breaks when data shrinks: 836 rows @ eff-batch 16 ≈ 52
    steps/epoch, so eval_steps=100 would evaluate ONCE at the very end — no
    trajectory visibility exactly when (small data) overfitting shows up.

    steps_per_epoch = ceil(n_train / (batch_size * grad_accum))
    eval_steps = max(floor, steps_per_epoch // evals_per_epoch)
    """
    import math
    eff_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(n_train / eff_batch)
    total_steps = steps_per_epoch * epochs
    ev = max(floor, steps_per_epoch // max(1, evals_per_epoch))
    print(f"[schedule] n_train={n_train} eff_batch={eff_batch} "
          f"steps/epoch={steps_per_epoch} total={total_steps} → eval/save every {ev} steps "
          f"(~{max(1, steps_per_epoch // ev)} evals/epoch)")
    return ev


def build_sft_config(args, fp16: bool, max_steps=None, output_dir=None, run_name=None,
                     eval_save_steps=None):
    """One place to build SFTConfig so the overfit check and real run share settings.

    VERSION CAVEAT: TRL's SFTConfig argument names move between releases
    (e.g. `max_seq_length` -> `max_length`, `evaluation_strategy` -> `eval_strategy`).
    These target current TRL (>=0.12). If an arg errors on your install, it's almost
    always a renamed kwarg, not a logic bug — check the installed SFTConfig signature
    and pin versions via the install line in the docstring.
    """
    # dynamic cadence if provided, else fall back to the CLI fixed value
    steps_cadence = eval_save_steps if eval_save_steps is not None else args.eval_steps
    return SFTConfig(
        output_dir=output_dir or args.output_dir,
        run_name=run_name,
        # BATCH (T4-safe): 15GB can't hold a real batch of a 7B, so use bs=1 and
        # accumulate gradients over `grad_accum` micro-batches → effective batch
        # = batch_size * grad_accum (default 1 * 16 = 16) without the memory cost.
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        # MEMORY LEVERS (T4-safe):
        # gradient checkpointing trades compute for memory by recomputing
        # activations in the backward pass instead of storing them. use_reentrant=False
        # is the modern, less buggy variant (the old default warns/misbehaves).
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # PRECISION: fp16 on T4, bf16 on L4/A100 (decided in load_model_and_tokenizer).
        fp16=fp16,
        bf16=not fp16,
        # paged 8-bit Adam: keeps optimizer state in 8-bit and pages it to CPU on
        # memory spikes — another lever that makes a 7B fit a T4.
        optim="paged_adamw_8bit",
        learning_rate=args.lr,               # ~2e-4: LoRA adapters want ~10x a full-FT LR
        lr_scheduler_type="constant",
        warmup_ratio=0.03,
        max_grad_norm=1.0,                    # gradient clipping for stability
        # LENGTH (memory/quality lever — TUNE FOR YOUR DATA): news documents can
        # exceed 2048 tokens. Truncation that cuts the text a summary's claim depends
        # on injects LABEL NOISE — the model is told "inconsistent" but never saw the
        # supporting sentence. Check the doc-length distribution in the data card and
        # raise max_len if memory allows, or filter long docs in prepare_fib.py.
        max_length=args.max_len,
        # packing concatenates examples to fill sequences — but it would BREAK our
        # per-example -100 masking, so it must stay OFF for the pre-masked path.
        packing=False,
        # LOGGING / EVAL / CHECKPOINT cadence
        logging_steps=10,
        eval_strategy="steps" if max_steps is None else "no",
        eval_steps=steps_cadence,
        # DRIVE CHECKPOINTING: output_dir is on mounted Drive (see main). Saving every
        # `save_steps` with save_total_limit=3 means a killed Colab session costs
        # minutes, not the whole run — resume with --resume. Directly addresses the
        # Colab idle-timeout / 12h-session hazard.
        save_strategy="steps",
        save_steps=steps_cadence,
        save_total_limit=3,                  # keep last 3 checkpoints on Drive
        num_train_epochs=args.epochs if max_steps is None else 1,
        max_steps=max_steps if max_steps is not None else -1,
        report_to=("wandb" if not args.no_wandb else "none"),
        seed=args.seed,
        # PRE-MASKED-LABELS PATH: our data is already tokenized with -100 masking,
        # so tell TRL to skip its own tokenization/templating. We deliberately do NOT
        # use assistant_only_loss (which needs {% generation %} template markers and
        # only auto-patches known model families — a version-fragile dependency).
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def make_trainer(model, tok, train_ds, eval_ds, peft_config, sft_config, callbacks=None):
    # COLLATOR: pads variable-length sequences in a batch. Crucially it pads the
    # LABELS with -100 (label_pad_token_id), so padding positions are ignored by
    # the loss — same -100 convention as our prompt masking.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tok, label_pad_token_id=-100, padding=True
    )
    # FREEZE-BUG AVOIDANCE (TRL issue #3926): we pass the QUANTIZED BASE model plus
    # peft_config and let SFTTrainer wrap it with LoRA exactly ONCE. If we instead
    # pre-wrapped with get_peft_model and ALSO passed it in, the trainer's internal
    # prepare_model_for_kbit_training can re-freeze everything — including the
    # adapters — so nothing trains (loss flat, no error). Let the trainer own the wrap.
    return SFTTrainer(
        model=model,                 # quantized BASE — NOT a pre-wrapped PeftModel
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,     # trainer applies LoRA once, the safe way
        data_collator=collator,
        processing_class=tok,
        callbacks=callbacks,         # e.g. DriveSyncCallback on the real run
    )


# ---------------------------------------------------------------------------
# STEP 1 — overfit-10 sanity check (runs FIRST)
# ---------------------------------------------------------------------------
def overfit_ten(model, tok, train_ds, peft_config, args, fp16):
    # WHY THIS RUNS FIRST: overfitting 10 examples is the fastest possible test that
    # the WHOLE pipeline is wired correctly — masking, LR, LoRA actually training,
    # data shape. A model that can train MUST be able to memorize 10 examples and
    # drive loss toward ~0 in a few dozen steps. If it can't, something is broken,
    # and you've found it in ~2 minutes instead of after a multi-hour full run.
    print("\n" + "=" * 70)
    print("OVERFIT-10 SANITY CHECK — loss MUST collapse toward ~0 on 10 examples.")
    print("If it doesn't, the pipeline is broken (masking/LR/config). Fix before real run.")
    print("=" * 70)
    tiny = train_ds.select(range(min(10, len(train_ds))))
    cfg = build_sft_config(
        args, fp16=fp16, max_steps=60,       # short probe: 60 steps is plenty to memorize 10 examples
        output_dir="/tmp/overfit10", run_name="overfit10",
    )
    # frequent logging, no eval/checkpoint noise for the probe
    cfg.logging_steps = 5
    cfg.save_strategy = "no"
    trainer = make_trainer(model, tok, tiny, tiny, peft_config, cfg)
    trainer.train()
    final_loss = trainer.state.log_history[-1].get(
        "train_loss", trainer.state.log_history[-2].get("loss", None)
    )
    print(f"\n[overfit10] final train loss ≈ {final_loss}")
    print("[overfit10] EXPECT this to be small (< ~0.1). If it's flat/high, STOP and debug.\n")
    # DELIBERATELY does NOT auto-proceed on a bad result. The human reads the loss
    # and decides — a failing probe should halt you, not be silently passed through.
    return final_loss


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="./data/train.jsonl")
    ap.add_argument("--eval", default="./data/val.jsonl")
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--run-name", default="judge_qlora_full", help="W&B run name")
    # OUTPUT DIR: default is FAST LOCAL disk (/content on Colab), NOT Drive.
    # Training to Drive directly is I/O-bound and ~50-100x slower. Checkpoints are
    # copied to --drive-dir periodically by DriveSyncCallback for crash-safety.
    ap.add_argument("--output-dir", default="/content/judge_ckpts/run1",
                    help="LOCAL working dir for fast training (wiped when session ends)")
    ap.add_argument("--drive-dir", default="/content/drive/MyDrive/judge_ckpts/run1",
                    help="Drive path to periodically copy checkpoints to (crash-safety). "
                         "Set to '' to disable Drive sync.")
    ap.add_argument("--sync-every-min", type=float, default=15.0,
                    help="minutes between checkpoint copies to Drive")
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"])  # +MLP = Wk4 ablation
    # training
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)      # T4-safe: 1 sequence at a time
    ap.add_argument("--grad-accum", type=int, default=16)     # effective batch = batch_size * grad_accum = 16
    ap.add_argument("--lr", type=float, default=2e-4)         # LoRA LR (~10x a full-FT LR)
    ap.add_argument("--max-len", type=int, default=2048)      # memory/quality lever — see build_sft_config note
    ap.add_argument("--eval-steps", type=int, default=100,
                    help="fixed fallback if --evals-per-epoch logic is bypassed")
    ap.add_argument("--save-steps", type=int, default=100,
                    help="fixed fallback (dynamic cadence overrides on the real run)")
    ap.add_argument("--evals-per-epoch", type=int, default=4,
                    help="target evaluations per epoch; eval/save steps computed from "
                         "dataset size so cadence adapts as data grows/shrinks")
    ap.add_argument("--seed", type=int, default=42)
    # flow control
    ap.add_argument("--skip-overfit-check", action="store_true",
                    help="skip the overfit-10 probe (NOT recommended)")
    ap.add_argument("--overfit-only", action="store_true",
                    help="run ONLY the overfit-10 probe and exit")
    ap.add_argument("--resume", action="store_true",
                    help="resume_from_checkpoint from output-dir")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", "hallucination-judge")
        # Keep W&B's local run files off Drive — same I/O reason as output-dir.
        os.environ.setdefault("WANDB_DIR", "/content/wandb")

    model, tok, compute_dtype = load_model_and_tokenizer(args.base_model)
    fp16 = compute_dtype == torch.float16
    train_ds, eval_ds = load_prepared(args.train, args.eval)
    print(f"[data] train={len(train_ds)}  eval={len(eval_ds)}  "
          f"compute_dtype={'fp16' if fp16 else 'bf16'}")

    peft_config = make_peft_config(
        args.lora_r, args.lora_alpha, args.lora_dropout, args.target_modules
    )

    # --- STEP 1: overfit-10 FIRST (unless explicitly skipped) ---
    if not args.skip_overfit_check:
        overfit_ten(model, tok, train_ds, peft_config, args, fp16)
        if args.overfit_only:
            print("[flow] --overfit-only set; exiting after probe.")
            return
        # RELOAD A CLEAN MODEL: the probe just trained LoRA adapters on 10 examples.
        # If we reused this model for the real run, that tiny overfit would leak in
        # as a warm start. Tear it down and load fresh so the full run starts clean.
        del model
        torch.cuda.empty_cache()
        model, tok, compute_dtype = load_model_and_tokenizer(args.base_model)

    # --- STEP 2: real run ---
    os.makedirs(args.output_dir, exist_ok=True)
    # DYNAMIC EVAL CADENCE: size eval/save steps from the actual dataset so we get
    # ~evals_per_epoch evaluations regardless of dataset size (fixed eval_steps=100
    # would evaluate ~once on the small balanced set).
    eval_save = compute_eval_save_steps(
        len(train_ds), args.batch_size, args.grad_accum, args.epochs,
        args.evals_per_epoch)
    sft_config = build_sft_config(args, fp16=fp16, run_name=args.run_name,
                                  eval_save_steps=eval_save)
    # Attach the Drive-sync callback so local checkpoints get copied to Drive every
    # --sync-every-min minutes (crash-safety) WITHOUT slowing the hot training path.
    callbacks = []
    if args.drive_dir:
        callbacks.append(DriveSyncCallback(
            local_dir=args.output_dir,
            drive_dir=args.drive_dir,
            sync_every_min=args.sync_every_min,
        ))
    trainer = make_trainer(model, tok, train_ds, eval_ds, peft_config, sft_config,
                           callbacks=callbacks)
    # SANITY: confirm ONLY the LoRA adapters are trainable. You want to see roughly
    # 0.1-1% of params trainable. If it prints 0 (or 100%), the freeze-bug or a
    # config error has occurred — do not let a flat-lining run waste hours.
    trainer.model.print_trainable_parameters()

    print(f"\n[train] training to LOCAL {args.output_dir} (fast); "
          f"syncing checkpoints → {args.drive_dir or '(disabled)'} every {args.sync_every_min} min.")
    # resume_from_checkpoint picks up the latest checkpoint after a killed Colab
    # session (pass --resume). NOTE: on resume you must first copy the last Drive
    # checkpoint back into --output-dir, since local /content was wiped.
    trainer.train(resume_from_checkpoint=args.resume)

    # --- save ADAPTER ONLY (small: just the LoRA weights, not the 4-bit base) ---
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"[save] adapter saved → {adapter_dir}")
    # Copy the final adapter to Drive too (tiny — seconds) so it survives the session.
    if args.drive_dir:
        drive_adapter = os.path.join(args.drive_dir, "final_adapter")
        try:
            if os.path.exists(drive_adapter):
                shutil.rmtree(drive_adapter)
            shutil.copytree(adapter_dir, drive_adapter)
            print(f"[save] final adapter copied → {drive_adapter}")
        except Exception as e:
            print(f"[save] WARN could not copy adapter to Drive: {e}")
    # MERGE IS A SEPARATE STEP ON PURPOSE: loading base+adapter, merge_and_unload,
    # saving, then RELOADING IN A FRESH PROCESS and generating is the test that
    # catches silent save/load bugs. Doing it here, in the same process that still
    # holds the trained model in memory, would hide exactly those bugs. Keep it in
    # merge_judge.py and verify the artifact loads clean before trusting it.
    print("[next] merge + reload-in-fresh-process is a SEPARATE step (merge_judge.py)")
    print("       — verify the saved artifact loads clean before trusting it.")


if __name__ == "__main__":
    main()
