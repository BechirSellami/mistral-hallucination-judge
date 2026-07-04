"""
load_usb.py — Out-of-domain test set (USB, Wikipedia) for the hallucination judge
=================================================================================

Purpose: FIB is news-domain (XSum/CNN-DM). USB is Wikipedia-derived across 6
domains. Evaluating the FIB-trained judge on USB measures OUT-OF-DOMAIN
generalization — a stronger, more honest claim than in-domain held-out F1 alone.

Source: `kundank/usb` (Krishna et al., EMNLP Findings 2023). The relevant task is
FAC (Factuality Classification): predict whether a summary sentence is factually
accurate w.r.t. the document. Maps to our binary judge:
  label 1 = factually consistent, label 0 = inconsistent.

This is a TEST set only — never train on it. Emits usb_test.jsonl in the same
pre-tokenized+masked format as prepare_fib.py so the eval harness treats
it identically to the FIB test set.

NOTE: USB's exact FAC schema (config name, field names, label encoding) can vary.
This loader PRINTS the schema on load and resolves fields defensively — if it
raises, adjust the FIELD_* / CONFIG constants to match what it prints. Fail loud.

Usage:
    python load_usb.py --base-model mistralai/Mistral-7B-Instruct-v0.3 \
                       --out-dir ./data --max-doc-chars 6000
"""

import argparse
import json
import hashlib
from pathlib import Path

# Reuse the exact formatting/masking from prepare_fib so USB rows are identical
# in structure to FIB rows (same prompt, same -100 masking, same chat template).
from prepare_fib import PROMPT, ANSWER_TEMPLATE, IGNORE_INDEX, to_chat_masked, write_jsonl

# USB config for the factuality-classification task. If load fails, print configs
# via: load_dataset("kundank/usb") and inspect available configs/splits.
USB_CONFIG_CANDIDATES = ["fac", "factuality", "factuality_classification", "FAC"]


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def load_usb_fac():
    """Load the USB factuality-classification task. Tries known config names,
    prints the schema of the first example so field names can be verified."""
    from datasets import load_dataset, get_dataset_config_names

    configs = []
    try:
        configs = get_dataset_config_names("kundank/usb")
        print(f"[load] available USB configs: {configs}")
    except Exception as e:
        print(f"[load] could not list configs ({e}); trying candidates directly")

    # pick a factuality-classification config
    chosen = None
    for cand in USB_CONFIG_CANDIDATES:
        if not configs or cand in configs:
            chosen = cand
            break
    if chosen is None and configs:
        # fall back: any config with 'fac' in the name
        for c in configs:
            if "fac" in c.lower():
                chosen = c
                break
    if chosen is None:
        raise RuntimeError(
            f"Could not find a factuality-classification config in USB. "
            f"Available: {configs}. Inspect and set USB_CONFIG_CANDIDATES."
        )

    print(f"[load] using USB config: {chosen}")
    ds = load_dataset("kundank/usb", chosen)
    # prefer a test split; fall back to whatever exists
    split = "test" if "test" in ds else list(ds.keys())[0]
    print(f"[load] using split: {split} ({len(ds[split])} examples)")
    ex0 = ds[split][0]
    print(f"[load] first example keys: {list(ex0.keys())}")
    print(f"[load] first example (truncated): " + json.dumps(
        {k: (str(v)[:120] + '...' if len(str(v)) > 120 else v) for k, v in ex0.items()},
        indent=2))
    return [dict(e) for e in ds[split]]


def _resolve_fields(ex: dict):
    """Pull (document, summary, label) from a USB FAC example.

    Real USB factuality_classification schema (verified):
      input_lines  -> the document, as a STRINGIFIED python list of sentences
      summary_sent -> the summary sentence being judged
      label        -> 0/1 (0 = inconsistent, 1 = consistent)
    """
    # document: input_lines is a str like "['sent one', 'sent two', ...]"
    raw_doc = ex.get("input_lines")
    if raw_doc is None:
        raise KeyError(f"No input_lines field. Keys: {list(ex.keys())}")
    document = _join_input_lines(raw_doc)

    summary = ex.get("summary_sent")
    if summary is None:
        raise KeyError(f"No summary_sent field. Keys: {list(ex.keys())}")

    if "label" not in ex or ex["label"] is None:
        raise KeyError(f"No label field. Keys: {list(ex.keys())}")
    label = _normalize_label(ex["label"])

    return str(document), str(summary), label


def _join_input_lines(raw):
    """input_lines is a stringified python list of sentences. Parse it safely and
    join into a single document string. Falls back to the raw string if parsing
    fails (so we never lose an example to a formatting quirk)."""
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw)
    s = str(raw)
    try:
        import ast
        parsed = ast.literal_eval(s)   # safe: only evaluates literals, not code
        if isinstance(parsed, (list, tuple)):
            return " ".join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return s


def _normalize_label(raw):
    """Map various encodings to 1 (consistent) / 0 (inconsistent)."""
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)  # assumes 1=consistent, 0=inconsistent
    s = str(raw).strip().lower()
    consistent = {"1", "true", "correct", "factual", "consistent", "supported", "yes"}
    inconsistent = {"0", "false", "incorrect", "non-factual", "nonfactual",
                    "inconsistent", "unsupported", "no"}
    if s in consistent:
        return 1
    if s in inconsistent:
        return 0
    raise ValueError(f"Cannot normalize label value: {raw!r}")


def extract_rows(raw):
    rows, skipped = [], 0
    for ex in raw:
        try:
            document, summary, label = _resolve_fields(ex)
        except (KeyError, ValueError) as e:
            skipped += 1
            if skipped <= 3:
                print(f"[extract] skipping malformed example: {e}")
            continue
        rows.append({
            "doc_id": doc_hash(document),
            "document": document,
            "summary": summary,
            "label": label,
            "source": "usb_wikipedia",   # tag so the eval harness can label OOD results
        })
    print(f"[extract] produced {len(rows)} USB rows ({skipped} skipped)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--max-doc-chars", type=int, default=None,
                    help="optional: drop rows whose document exceeds this many chars")
    ap.add_argument("--out-name", default="usb_test.jsonl")
    ap.add_argument("--flip-labels", action="store_true",
                    help="invert 0/1 labels. USB FAC polarity must be verified: if their "
                         "label=1 means 'contains factual error' (not 'consistent'), pass "
                         "this so it matches our convention (1=consistent, 0=inconsistent).")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = load_usb_fac()
    rows = extract_rows(raw)

    if args.flip_labels:
        for r in rows:
            r["label"] = 1 - r["label"]
        print("[usb] flipped labels (1↔0) per --flip-labels")

    if args.max_doc_chars:
        before = len(rows)
        rows = [r for r in rows if len(r["document"]) <= args.max_doc_chars]
        print(f"[filter] dropped {before - len(rows)} rows over {args.max_doc_chars} chars")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    recs = [to_chat_masked(r, tok) for r in rows]
    write_jsonl(out / args.out_name, recs)

    # class balance report — USB FAC may be IMBALANCED (unlike FIB's forced 50/50).
    # This matters for reading F1 vs accuracy.
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    print(f"[usb] label balance: consistent(1)={c.get(1,0)}  inconsistent(0)={c.get(0,0)}")
    if c.get(1, 0) and c.get(0, 0):
        ratio = c.get(1, 0) / (c.get(0, 0) or 1)
        print(f"[usb] pos/neg ratio = {ratio:.2f} — if far from 1.0, prefer F1 over accuracy")
    print(f"[usb] wrote {len(recs)} OOD test rows → {out / args.out_name}")
    print("[usb] NOTE: TEST ONLY. Never train on this. Report as the out-of-domain "
          "number, separate from the in-domain FIB test result.")


if __name__ == "__main__":
    main()
