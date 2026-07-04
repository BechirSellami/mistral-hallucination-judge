"""
prepare_fib.py — Data prep for the hallucination-judge anchor project
=====================================================================

Loads the FIB benchmark, converts its paired-summary format into single
(document, summary, label) classification rows, formats them with the
Mistral chat template + prompt-loss masking, splits DOCUMENT-LEVEL to avoid
contamination, and writes train.jsonl / eval.jsonl + a data card.

KEY FACTS about FIB that drive this script:
- FIB is test-only: 3,579 examples, each = 1 document + 2 summaries
  (one factually consistent, one inconsistent). We create our own split.
- Because each doc yields one positive + one negative, the data is
  class-balanced by construction.
- Examples come from two sources (XSum, CNN/DM) whose behavior differs;
  we tag source and the eval harness should report them separately.
- Contamination guard MUST be document-level: both summaries of a document
  go to the SAME split, else the article leaks across train/eval.

Usage:
    pip install datasets transformers
    python prepare_fib.py --base-model mistralai/Mistral-7B-Instruct-v0.3 \
                          --out-dir ./data --eval-frac 0.2 --seed 42

Output: data/train.jsonl, data/eval.jsonl, data/data_card.md

NOTE: inspect a few raw examples first (this script prints schema on load) —
FIB field names can vary by mirror. If keys differ from what's assumed in
extract_pairs(), adjust the FIELD_* constants below. Fail loud, don't guess.
"""

import argparse
import json
import hashlib
import random
from collections import Counter
from pathlib import Path


# --- Field names in the r-three/fib schema (adjust if your mirror differs) ---
# The FIB repo describes each example with keys like:
#   {id, input (the document), correct_choice, list_choices, lbl}
# where list_choices holds the two summaries and correct_choice/lbl marks the
# factually-consistent one. We resolve these defensively at runtime.
PROMPT = (
    "You are an expert at detecting factual inconsistencies and hallucinations. "
    "You will be given a document and a summary. Decide whether every statement "
    "in the summary is supported by the document.\n"
    "- Label 1 (consistent): all statements in the summary are supported by the document.\n"
    "- Label 0 (inconsistent): at least one statement is unsupported by or contradicts the document.\n"
    "A statement that is true in the world but NOT inferable from the document is inconsistent.\n"
    'Respond ONLY with JSON: {{"consistency": 0}} or {{"consistency": 1}}.\n\n'
    "Document:\n{document}\n\nSummary:\n{summary}"
)
ANSWER_TEMPLATE = '{{"consistency": {label}}}'
IGNORE_INDEX = -100


def load_fib():
    """Load FIB from the Hub. Returns a list of raw example dicts and prints
    the schema of the first example so you can verify field names."""
    from datasets import load_dataset
    ds = load_dataset("r-three/fib", split="test")
    print(f"[load] FIB loaded: {len(ds)} examples")
    print(f"[load] first example keys: {list(ds[0].keys())}")
    print(f"[load] first example (truncated): "
          f"{json.dumps({k: (str(v)[:120] + '...' if len(str(v)) > 120 else v) for k, v in ds[0].items()}, indent=2)}")
    return [dict(ex) for ex in ds]


def _resolve_fields(ex: dict):
    """Defensively pull (document, consistent_summary, inconsistent_summary, source)
    out of a raw FIB example, tolerating schema variations across mirrors.
    Raises if it can't — we fail loud rather than mislabel."""
    # document
    document = ex.get("input") or ex.get("document") or ex.get("article")
    if document is None:
        raise KeyError(f"No document field found. Keys: {list(ex.keys())}")

    # the two candidate summaries
    choices = ex.get("list_choices") or ex.get("choices") or ex.get("summaries")
    if not choices or len(choices) < 2:
        raise KeyError(f"No 2-summary choice list found. Keys: {list(ex.keys())}")

    # which one is factually consistent
    # 'correct_choice' may be the text, 'lbl' may be an index — handle both.
    correct = ex.get("correct_choice")
    lbl = ex.get("lbl")
    if correct in choices:
        consistent = correct
        inconsistent = next(c for c in choices if c != correct)
    elif isinstance(lbl, int) and 0 <= lbl < len(choices):
        consistent = choices[lbl]
        inconsistent = choices[1 - lbl] if len(choices) == 2 else \
            next(c for i, c in enumerate(choices) if i != lbl)
    else:
        raise KeyError(
            f"Cannot resolve which summary is consistent. "
            f"correct_choice={correct!r}, lbl={lbl!r}, keys={list(ex.keys())}"
        )

    # source: XSum vs CNN/DM (reported separately per the FIB README)
    source = ex.get("source") or ex.get("dataset") or "unknown"
    return document, consistent, inconsistent, source


def doc_hash(document: str) -> str:
    """Stable id for a document, for document-level splitting + dedup."""
    return hashlib.sha256(document.strip().encode("utf-8")).hexdigest()[:16]


def extract_rows(raw):
    """Unpack each paired FIB example into TWO classification rows.
    Returns list of dicts: {doc_id, document, summary, label, source}."""
    rows = []
    skipped = 0
    for ex in raw:
        try:
            document, consistent, inconsistent, source = _resolve_fields(ex)
        except KeyError as e:
            skipped += 1
            if skipped <= 3:
                print(f"[extract] skipping malformed example: {e}")
            continue
        did = doc_hash(document)
        rows.append({"doc_id": did, "document": document,
                     "summary": consistent, "label": 1, "source": source})
        rows.append({"doc_id": did, "document": document,
                     "summary": inconsistent, "label": 0, "source": source})
    print(f"[extract] produced {len(rows)} rows from {len(raw)} examples "
          f"({skipped} skipped)")
    return rows


def dedup_rows(rows):
    """Remove exact-duplicate (document, summary, label) triples."""
    seen = set()
    out = []
    for r in rows:
        key = (r["doc_id"], hashlib.sha256(r["summary"].strip().encode()).hexdigest()[:16], r["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if len(out) != len(rows):
        print(f"[dedup] removed {len(rows) - len(out)} duplicate rows")
    return out


def balance_per_doc(rows, seed):
    """Force class balance per document (Option A): FIB pairs each document's
    single consistent summary against MULTIPLE distractor-generated inconsistent
    summaries, so after unpacking each doc has 1 label-1 and N label-0 rows —
    an ~85/15 skew. We keep, per document, min(#consistent, #inconsistent) of
    EACH class (here 1 + 1), sampling the inconsistent one randomly (seeded).

    Result: exactly one consistent + one inconsistent per document → clean 50/50,
    the truest form of FIB's paired design. Randomly picking among the distractors
    also preserves distractor-model variety across the dataset.
    """
    rng = random.Random(seed)
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], {0: [], 1: []})[r["label"]].append(r)

    balanced = []
    for did, cls in by_doc.items():
        k = min(len(cls[0]), len(cls[1]))   # keep equal numbers of each class
        if k == 0:
            continue                        # doc missing one class entirely → drop
        balanced.extend(rng.sample(cls[1], k))   # consistent
        balanced.extend(rng.sample(cls[0], k))   # inconsistent
    rng.shuffle(balanced)

    from collections import Counter
    before = Counter(r["label"] for r in rows)
    after = Counter(r["label"] for r in balanced)
    print(f"[balance] per-doc balanced: {dict(before)} → {dict(after)} "
          f"({len(rows)}→{len(balanced)} rows)")
    return balanced


def split_by_document(rows, val_frac, test_frac, seed):
    """THREE-WAY split (train / val / test), DOCUMENT-LEVEL + SOURCE-STRATIFIED.

    Why three splits (not two):
      - train  → fits the model weights
      - val    → guides tuning choices (rank/LR/target_modules/ablations)
      - test   → SEALED; touched once in for the unbiased headline number.
    Reporting on the set you tuned against inflates the result. Because we run a
    real hyperparameter sweep, val and test MUST be distinct.

    Guards (unchanged, extended to 3 buckets):
      - document-level: both summaries of a document stay in ONE split (no article
        leaks across splits).
      - source-stratified: XSum and CNN/DM each split independently, so all three
        buckets contain both sources.
    """
    rng = random.Random(seed)

    # doc_id -> source (all rows of a doc share a source)
    doc_source = {r["doc_id"]: r["source"] for r in rows}

    # group doc_ids by source: {"xsum": [...], "cnn_dailymail": [...]}
    docs_by_source = {}
    for did, src in doc_source.items():
        docs_by_source.setdefault(src, []).append(did)

    val_docs, test_docs = set(), set()

    # split each source independently so all three buckets are source-balanced
    for src, dids in docs_by_source.items():
        dids = sorted(set(dids))     # deterministic pre-shuffle order
        rng.shuffle(dids)
        n = len(dids)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # carve test first, then val, remainder = train
        test_docs.update(dids[:n_test])
        val_docs.update(dids[n_test:n_test + n_val])

    def bucket(r):
        d = r["doc_id"]
        if d in test_docs:
            return "test"
        if d in val_docs:
            return "val"
        return "train"

    train = [r for r in rows if bucket(r) == "train"]
    val   = [r for r in rows if bucket(r) == "val"]
    test  = [r for r in rows if bucket(r) == "test"]

    # CONTAMINATION ASSERTION — the three document sets must be pairwise disjoint.
    d_train = {r["doc_id"] for r in train}
    d_val   = {r["doc_id"] for r in val}
    d_test  = {r["doc_id"] for r in test}
    assert not (d_train & d_val),  f"CONTAMINATION train∩val: {len(d_train & d_val)} docs"
    assert not (d_train & d_test), f"CONTAMINATION train∩test: {len(d_train & d_test)} docs"
    assert not (d_val & d_test),   f"CONTAMINATION val∩test: {len(d_val & d_test)} docs"

    print(f"[split] train: {len(train)} rows / {len(d_train)} docs | "
          f"val: {len(val)} rows / {len(d_val)} docs | "
          f"test: {len(test)} rows / {len(d_test)} docs | overlaps: 0 ✓")

    return train, val, test


def to_chat_masked(row, tokenizer):
    """Format one row with the Mistral chat template and build labels with
    prompt-loss masking (-100 on everything except the assistant answer).

    Returns a dict with input_ids, attention_mask, labels — ready for SFT.
    We render the prompt and the full conversation separately so we know
    exactly where the assistant answer begins, then mask everything before it.    
    """
    user_content = PROMPT.format(document=row["document"], summary=row["summary"])
    answer = ANSWER_TEMPLATE.format(label=row["label"])

    # Prompt-only render (add_generation_prompt=True puts the template up to
    # where the assistant should start generating).
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=True,
    )
    # Newer transformers may return BatchEncoding instead of a plain list.
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]
    # Full render (user + assistant answer).
    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content},
         {"role": "assistant", "content": answer}],
        add_generation_prompt=False,
        tokenize=True,
    )
    if not isinstance(full_ids, list):
        full_ids = full_ids["input_ids"]

    # Mask: everything in the prompt prefix is IGNORE_INDEX; the assistant
    # answer tokens (the suffix beyond the prompt length) keep their ids.
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # length guard — labels must align 1:1 with input_ids
    assert len(labels) == len(full_ids), \
        f"label/length mismatch: {len(labels)} vs {len(full_ids)}"

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        # keep raw fields for eval-time prompting / inspection
        "meta": {"doc_id": row["doc_id"], "label": row["label"], "source": row["source"]},
    }


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {path}: {len(records)} records")


def _len_section(s):
    if not s:
        return "_(not computed)_"
    p = s["pctiles"]
    pct_rows = " | ".join(f"p{k}={v}" for k, v in p.items())
    # recommend the smallest max_len that truncates < ~1% of rows
    rec = next((ml for ml in (512, 768, 1024, 1536, 2048) if s["trunc"][ml] < 1.0), 2048)
    trunc_rows = "\n".join(f"| {ml} | {s['trunc'][ml]}% |"
                           for ml in (512, 768, 1024, 1536, 2048))
    return (f"Mean {s['mean']} tokens. Percentiles: {pct_rows}.\n\n"
            f"| max_len | rows truncated |\n|---------|----------------|\n{trunc_rows}\n\n"
            f"**Recommended max_len = {rec}** (smallest cap truncating <1% of rows). "
            f"Truncation on this task = label noise, so keep it low.")


def length_stats(records):
    """Token-length distribution + truncation table, computed from the tokenized
    records (len of input_ids). Returns a dict the data card renders. This makes
    the max_len choice data-driven and reproducible instead of a hardcoded note."""
    import numpy as np
    lengths = np.array([len(r["input_ids"]) for r in records])
    pctiles = {p: int(np.percentile(lengths, p)) for p in (50, 75, 90, 95, 99, 100)}
    trunc = {ml: round(float((lengths > ml).mean()) * 100, 1)
             for ml in (512, 768, 1024, 1536, 2048)}
    return {"mean": int(lengths.mean()), "pctiles": pctiles, "trunc": trunc}


def write_data_card(path, train, val, test, base_model, val_frac, test_frac, seed,
                    len_stats=None):
    def stats(rows):
        labels = Counter(r["label"] for r in rows)
        sources = Counter(r["source"] for r in rows)
        return (len(rows), len({r["doc_id"] for r in rows}), labels, sources)

    def row(name, rows):
        n, d, l, s = stats(rows)
        xsum = s.get("xsum", s.get("XSum", 0))
        cnndm = s.get("cnndm", s.get("cnn_dm", 0))
        other = sum(v for k, v in s.items() if k.lower() not in ("xsum", "cnndm", "cnn_dm"))
        return f"| {name} | {n} | {d} | {l.get(1,0)} | {l.get(0,0)} | {xsum} | {cnndm} | {other} |"

    train_frac = round(1 - val_frac - test_frac, 3)
    card = f"""# Data Card — FIB Hallucination Judge

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
- **train** ({train_frac:.0%}) — fits the model.
- **val** ({val_frac:.0%}) — guides tuning (rank/LR/target_modules, FIB vs FIB+USB).
- **test** ({test_frac:.0%}) — **SEALED. Touched ONCE for the headline number.**
  Do not look at test while tuning; reporting on a set you tuned against inflates results.
- DOCUMENT-LEVEL: both summaries of a document stay in ONE split (no article leakage).
- SOURCE-STRATIFIED: XSum and CNN/DM each split independently → all buckets balanced.
- Verified pairwise-disjoint document sets. seed={seed}.

## Statistics
| split | rows | documents | label 1 | label 0 | XSum | CNN/DM | other |
|-------|------|-----------|---------|---------|------|--------|-------|
{row("train", train)}
{row("val", val)}
{row("test", test)}

## Formatting
- Mistral chat template via `tokenizer.apply_chat_template`.
- Prompt-loss masking: only the assistant JSON answer contributes to the loss
  (prompt tokens set to label -100). Base model for the template: `{base_model}`.

## Evaluation protocol
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
{_len_section(len_stats)}
"""
    Path(path).write_text(card)
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of documents for the validation split (Wk-4 tuning)")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="fraction for the SEALED test split (Wk-5 only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-doc-chars", type=int, default=None,
                    help="optional: drop rows whose document exceeds this many chars")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)  # create output dir if it doesn't exist
    # Load dataset from datasets package
    raw = load_fib()
    rows = extract_rows(raw)
    rows = dedup_rows(rows)

    if args.max_doc_chars:
        
        before = len(rows)
        rows = [r for r in rows if len(r["document"]) <= args.max_doc_chars]
        print(f"[filter] dropped {before - len(rows)} rows over {args.max_doc_chars} chars")

    # Balance classes per document (Option A) BEFORE splitting, so every split
    # inherits a clean 50/50. Done after length-filtering so we balance the rows
    # that actually survive to training.
    rows = balance_per_doc(rows, args.seed)

    train_rows, val_rows, test_rows = split_by_document(
        rows, args.val_frac, args.test_frac, args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_recs = [to_chat_masked(r, tok) for r in train_rows]
    val_recs   = [to_chat_masked(r, tok) for r in val_rows]
    test_recs  = [to_chat_masked(r, tok) for r in test_rows]

    write_jsonl(out / "train.jsonl", train_recs)
    write_jsonl(out / "val.jsonl", val_recs)     # Wk-4 tuning
    write_jsonl(out / "test.jsonl", test_recs)   # SEALED — Wk-5 only
    write_data_card(out / "data_card.md", train_rows, val_rows, test_rows,
                    args.base_model, args.val_frac, args.test_frac, args.seed,
                    len_stats=length_stats(train_recs))

    # quick masking sanity print on one record
    ex = train_recs[0]
    n_supervised = sum(1 for l in ex["labels"] if l != IGNORE_INDEX)
    print(f"\n[sanity] first train record: {len(ex['input_ids'])} tokens, "
          f"{n_supervised} supervised (answer) tokens, "
          f"{len(ex['input_ids']) - n_supervised} masked (prompt) tokens")
    print("[sanity] supervised token ids (the answer):",
          [l for l in ex["labels"] if l != IGNORE_INDEX])


if __name__ == "__main__":
    main()
