"""
eval_judge.py — score a hallucination-judge model on a jsonl set
================================================================

ONE scorer, TWO uses:
  - score a candidate model on val.jsonl to pick sweep configs.
  - score the FROZEN model on the sealed test sets (FIB test, USB).
Same code path both times → how you tune == how you report (no divergence).

Model-agnostic + set-agnostic:
    python eval_judge.py --model <merged_dir_or_hf_id> --data ./data/val.jsonl
    python eval_judge.py --model <merged_dir> --data ./data/test.jsonl --out results_test.json
    python eval_judge.py --model <merged_dir> --data ./data/usb_test.jsonl --out results_usb.json

Reports F1/precision/recall/accuracy overall AND per-source (XSum/CNN-DM/USB),
since FIB requires per-source reporting. Convention: 1 = consistent, 0 = inconsistent.
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# metrics (no sklearn dependency — keep it self-contained)
# ---------------------------------------------------------------------------
def binary_metrics(golds, preds, positive=1):
    """F1/precision/recall/accuracy treating `positive` as the positive class.
    preds may contain None (unparseable) — counted as wrong."""
    tp = fp = fn = tn = 0
    correct = 0
    n = len(golds)
    for g, p in zip(golds, preds):
        if p is None:
            # unparseable → wrong; attribute to the class it should have been
            if g == positive: fn += 1
            else: fp += 1
            continue
        correct += int(p == g)
        if p == positive and g == positive: tp += 1
        elif p == positive and g != positive: fp += 1
        elif p != positive and g == positive: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = correct / n if n else 0.0
    return {"n": n, "f1": round(f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "accuracy": round(acc, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def parse_label(text):
    """Extract predicted 0/1 from the model's generated answer."""
    m = re.search(r'consistency"?\s*:\s*([01])', text)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([01])\b', text)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# data — pre-tokenized rows from prepare_fib / load_usb
# ---------------------------------------------------------------------------
def load_rows(path):
    rows = [json.loads(l) for l in open(path)]
    # split each row into prompt_ids (up to the answer) + gold + source
    out = []
    for r in rows:
        labels = r["labels"]
        ans_start = next((i for i, l in enumerate(labels) if l != IGNORE_INDEX), len(labels))
        out.append({
            "prompt_ids": r["input_ids"][:ans_start],
            "gold": r["meta"]["label"],
            "source": r["meta"].get("source", "unknown"),
        })
    return out


# ---------------------------------------------------------------------------
# generation — batched greedy decode of the 0/1 answer
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_model(model, tok, rows, batch_size=16, max_new_tokens=8):
    preds = []
    model.eval()
    pad_id = tok.pad_token_id or tok.eos_token_id
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        # left-pad for batched generation so all prompts end at the same position
        tok.padding_side = "left"
        enc = tok.pad({"input_ids": [b["prompt_ids"] for b in batch]},
                      return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=pad_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for g in gen:
            preds.append(parse_label(tok.decode(g, skip_special_tokens=True)))
        print(f"  scored {min(i+batch_size, len(rows))}/{len(rows)}", end="\r")
    print()
    return preds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir or HF id (base for baseline)")
    ap.add_argument("--data", required=True, help="jsonl: val.jsonl / test.jsonl / usb_test.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None, help="optional JSON results path")
    ap.add_argument("--limit", type=int, default=None, help="score only first N (quick check)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[eval] {args.model} on {args.data} ({len(rows)} rows)")

    preds = run_model(model, tok, rows, batch_size=args.batch_size)
    golds = [r["gold"] for r in rows]

    overall = binary_metrics(golds, preds)
    print(f"\n[overall] F1={overall['f1']}  P={overall['precision']}  "
          f"R={overall['recall']}  acc={overall['accuracy']}  (n={overall['n']})")

    # per-source breakdown (FIB requires XSum vs CNN-DM separately; USB is its own)
    by_source = defaultdict(lambda: ([], []))
    for r, p in zip(rows, preds):
        g_list, p_list = by_source[r["source"]]
        g_list.append(r["gold"]); p_list.append(p)
    per_source = {}
    for src, (g, p) in by_source.items():
        m = binary_metrics(g, p)
        per_source[src] = m
        print(f"[{src}] F1={m['f1']}  P={m['precision']}  R={m['recall']}  "
              f"acc={m['accuracy']}  (n={m['n']})")

    results = {"model": args.model, "data": args.data,
               "overall": overall, "per_source": per_source}
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
