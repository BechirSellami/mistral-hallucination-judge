"""
make_fib_usb_mix.py — build train_fib_usb.jsonl for the FIB→FIB+USB staging
===========================================================================

Concatenates FIB train.jsonl with a training slice of USB (Wikipedia) so the
judge sees both domains during training.

CONTAMINATION GUARD: pulls USB from its TRAIN + VALIDATION splits ONLY, never
the TEST split (which load_usb.py reserves for the sealed OOD evaluation).
Same tokenization/masking/polarity handling as load_usb.py.

Usage:
    python make_fib_usb_mix.py --base-model mistralai/Mistral-7B-Instruct-v0.3 \
        --fib-train ./train.jsonl --out ./train_fib_usb.jsonl \
        --max-doc-chars 6000 --flip-labels --usb-cap 5000
"""

import argparse
import json
import random
from pathlib import Path

# reuse everything from load_usb (fields, join, normalize, formatting)
from load_usb import (
    _resolve_fields, doc_hash, extract_rows as _usb_extract,
    USB_CONFIG_CANDIDATES,
)
from prepare_fib import to_chat_masked, write_jsonl


def load_usb_train_splits():
    """Load USB FAC TRAIN + VALIDATION only (never test). Returns raw examples."""
    from datasets import load_dataset, get_dataset_config_names

    try:
        configs = get_dataset_config_names("kundank/usb", trust_remote_code=True)
    except Exception:
        configs = []
    chosen = next((c for c in USB_CONFIG_CANDIDATES if not configs or c in configs), None)
    if chosen is None:
        chosen = next((c for c in configs if "fac" in c.lower()), None)
    if chosen is None:
        raise RuntimeError(f"No FAC config found. configs={configs}")

    ds = load_dataset("kundank/usb", chosen, trust_remote_code=True)
    raw = []
    for split in ("train", "validation"):
        if split in ds:
            print(f"[usb-mix] using USB {split}: {len(ds[split])} examples")
            raw.extend(dict(e) for e in ds[split])
    if "test" in ds:
        print(f"[usb-mix] EXCLUDING USB test ({len(ds['test'])}) — reserved for OOD eval")
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--fib-train", default="./data/train.jsonl")
    ap.add_argument("--out", default="./data/train_fib_usb.jsonl")
    ap.add_argument("--max-doc-chars", type=int, default=6000)
    ap.add_argument("--flip-labels", action="store_true",
                    help="match the polarity you used in load_usb (USB native is inverted → flip)")
    ap.add_argument("--usb-cap", type=int, default=None,
                    help="cap USB rows to avoid swamping FIB (optional)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. USB train-slice rows → tokenized
    raw = load_usb_train_splits()
    usb_rows = _usb_extract(raw)
    if args.flip_labels:
        for r in usb_rows:
            r["label"] = 1 - r["label"]
        print("[usb-mix] flipped USB labels to match convention (1=consistent)")
    if args.max_doc_chars:
        before = len(usb_rows)
        usb_rows = [r for r in usb_rows if len(r["document"]) <= args.max_doc_chars]
        print(f"[usb-mix] dropped {before-len(usb_rows)} rows over {args.max_doc_chars} chars")
    if args.usb_cap and len(usb_rows) > args.usb_cap:
        random.Random(args.seed).shuffle(usb_rows)
        usb_rows = usb_rows[:args.usb_cap]
        print(f"[usb-mix] capped USB to {args.usb_cap} rows")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    usb_recs = [to_chat_masked(r, tok) for r in usb_rows]

    # 2. FIB train rows (already tokenized) — read as-is
    fib_recs = [json.loads(l) for l in open(args.fib_train)]
    print(f"[usb-mix] FIB train: {len(fib_recs)}  |  USB train-slice: {len(usb_recs)}")

    # 3. concat + shuffle so batches mix domains
    mixed = fib_recs + usb_recs
    random.Random(args.seed).shuffle(mixed)
    write_jsonl(Path(args.out), mixed)
    print(f"[usb-mix] wrote {len(mixed)} rows → {args.out} "
          f"(FIB {len(fib_recs)} + USB {len(usb_recs)})")
    print("[usb-mix] NOTE: USB test split was NOT included — OOD eval stays clean.")


if __name__ == "__main__":
    main()
