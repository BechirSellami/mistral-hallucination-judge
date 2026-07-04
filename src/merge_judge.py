"""
merge_judge.py — merge LoRA adapter, save, verify by cold reload
=============================================================================

Two responsibilities, deliberately separated:
  1. MERGE: fuse the LoRA adapter (B·A) into the base weights -> a standalone
     model with no PEFT dependency and no inference overhead. Save it.
  2. VERIFY: reload the saved model IN A FRESH SUBPROCESS and generate on a few
     eval examples. The subprocess is the point — it proves the artifact loads
     from disk cleanly, catching silent save bugs that an in-process reload
     (with cached state) would hide.

Two modes:
  python merge_judge.py --merge   ...   # do the merge + save (process 1)
  python merge_judge.py --verify  ...   # cold-reload + generate (process 2)
  python merge_judge.py            ...   # merge, then auto-spawn --verify subprocess

Colab:
  !python merge_judge.py \
      --base-model mistralai/Mistral-7B-Instruct-v0.3 \
      --adapter /content/judge_ckpts/run1/final_adapter \
      --merged-dir /content/judge_merged \
      --eval ./val.jsonl
"""

import argparse
import json
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# 1. MERGE — fuse adapter into base, save standalone
# ---------------------------------------------------------------------------
def do_merge(base_model: str, adapter: str, merged_dir: str):
    from peft import PeftModel

    print(f"[merge] loading base {base_model} in bf16 (NOT 4-bit — we want mergeable weights)")
    # IMPORTANT: load the base in bf16/fp16, NOT 4-bit. merge_and_unload on a
    # 4-bit base is lossy/unsupported for a clean standalone export — you want
    # full-precision weights so W + B·A is exact and the result loads anywhere.
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"[merge] attaching adapter {adapter}")
    model = PeftModel.from_pretrained(base, adapter)

    print("[merge] merge_and_unload() — folding B·A into W")
    model = model.merge_and_unload()   # returns the base model with weights updated in place

    print(f"[merge] saving standalone model -> {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(adapter)  # adapter dir has the tokenizer we trained with
    tok.save_pretrained(merged_dir)
    print("[merge] done. Adapter is now baked into the weights; no PEFT needed to load.")


# ---------------------------------------------------------------------------
# 2. VERIFY — cold reload in this (fresh) process, generate on eval examples
# ---------------------------------------------------------------------------
def parse_label_from_text(text: str):
    """Pull the predicted 0/1 out of the model's generated JSON-ish answer.
    Robust to minor format drift: look for consistency: 0/1 or a bare 0/1."""
    import re
    m = re.search(r'consistency"?\s*:\s*([01])', text)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([01])\b', text)
    return int(m.group(1)) if m else None


def do_verify(merged_dir: str, eval_path: str, n: int, base_model_for_prompt: str):
    print(f"[verify] COLD-LOADING merged model from disk: {merged_dir}")
    # This load happening in a fresh process is the actual test. If the save was
    # broken, THIS is where it fails — not silently later.
    tok = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("[verify] loaded OK. Running generations on eval examples...")

    # Read a few eval rows. prepare_fib.py wrote pre-tokenized rows with input_ids
    # + labels + meta. We reconstruct the PROMPT (everything before the answer) by
    # cutting input_ids at the first supervised (non -100) label position.
    rows = []
    with open(eval_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break

    correct, total = 0, 0
    per_source = {}
    for r in rows:
        input_ids = r["input_ids"]
        labels = r["labels"]
        gold = r["meta"]["label"]
        source = r["meta"].get("source", "unknown")

        # prompt = tokens up to the first non-masked label (the answer boundary)
        ans_start = next((i for i, l in enumerate(labels) if l != IGNORE_INDEX), len(labels))
        prompt_ids = torch.tensor([input_ids[:ans_start]], device=model.device)

        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=10,          # answer is tiny: {"consistency": 0/1}
                do_sample=False,            # greedy — deterministic judge
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen_text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_label_from_text(gen_text)

        ok = (pred == gold)
        correct += int(ok)
        total += 1
        d = per_source.setdefault(source, {"c": 0, "n": 0})
        d["n"] += 1; d["c"] += int(ok)
        print(f"  [{source}] gold={gold} pred={pred} {'✓' if ok else '✗'}  raw={gen_text!r}")

    print(f"\n[verify] quick accuracy on {total} eval examples: {correct}/{total} = {correct/total:.2%}")
    for src, d in per_source.items():
        print(f"[verify]   {src}: {d['c']}/{d['n']} = {d['c']/d['n']:.2%}")
    print("[verify] NOTE: this is a SANITY generation check, not the final eval "
          "(full held-out F1 vs base vs prompted baseline). It proves the merged "
          "artifact loads and behaves; it does not certify performance.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--adapter", default="/content/judge_ckpts/run1/final_adapter")
    ap.add_argument("--merged-dir", default="/content/judge_merged")
    ap.add_argument("--eval", default="./data/val.jsonl")
    ap.add_argument("--n", type=int, default=8, help="eval examples for the sanity check")
    ap.add_argument("--merge", action="store_true", help="only merge + save")
    ap.add_argument("--verify", action="store_true", help="only cold-reload + generate")
    args = ap.parse_args()

    # Explicit single-phase modes
    if args.merge:
        do_merge(args.base_model, args.adapter, args.merged_dir)
        return
    if args.verify:
        do_verify(args.merged_dir, args.eval, args.n, args.base_model)
        return

    # Default: merge in THIS process, then spawn a SEPARATE process for verify.
    # The separate process is essential — it guarantees a cold load from disk,
    # with none of this process's cached model/tokenizer state.
    do_merge(args.base_model, args.adapter, args.merged_dir)
    print("\n[main] merge complete. Spawning a FRESH process to cold-reload + verify...\n")
    cmd = [
        sys.executable, __file__, "--verify",
        "--merged-dir", args.merged_dir,
        "--eval", args.eval,
        "--n", str(args.n),
        "--base-model", args.base_model,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[main] VERIFY SUBPROCESS FAILED — the merged artifact did not load/run "
              "cleanly. Investigate before trusting it.")
        sys.exit(result.returncode)
    print("[main] merge + cold-reload verification complete. Artifact is trustworthy.")


if __name__ == "__main__":
    main()
