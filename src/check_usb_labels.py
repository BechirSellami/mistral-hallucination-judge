"""
check_usb_labels.py — decode usb_test.jsonl rows + guard against polarity inversion
===================================================================================

Two parts:
  1. INSPECT: decode a few rows to text so you can eyeball document/summary/label.
  2. GUARD:  known-answer probes — sentinel summaries whose correct label is
             unambiguous. If the file's polarity is inverted, these assertions
             FAIL loudly instead of silently poisoning the OOD eval.

Our convention: label 1 = consistent (supported), 0 = inconsistent (unsupported).

Usage:
    python check_usb_labels.py --file ./data/usb_test.jsonl \
                               --base-model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import json
import sys

IGNORE_INDEX = -100

# Sentinel probes drawn from the verified USB pairs. After --flip-labels, these
# are the labels we EXPECT in usb_test.jsonl. If reality disagrees, polarity is
# wrong. Match is substring-based on the decoded summary text.
EXPECTED = [
    ("taught at Northwestern",              1),   # supported -> consistent
    ("teaches at Northwestern",             0),   # present tense contradicts "until 1991"
    ("In January 2019, floods",             1),   # supported (vague but correct)
    ("On 22 January 2019, floods",          0),   # fabricated specific date
    ("interactionist, although he does not align", 1),  # matches doc
    ("or social constructionist",           0),   # added claim not in doc
]


def decode_row(row, tok):
    """Reconstruct (prompt_text, answer_text, gold_label) from a tokenized row."""
    input_ids = row["input_ids"]
    labels = row["labels"]
    gold = row["meta"]["label"]
    ans_start = next((i for i, l in enumerate(labels) if l != IGNORE_INDEX), len(labels))
    prompt_text = tok.decode(input_ids[:ans_start], skip_special_tokens=True)
    answer_text = tok.decode(input_ids[ans_start:], skip_special_tokens=True)
    return prompt_text, answer_text, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="./data/usb_test.jsonl")
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--n", type=int, default=4, help="rows to print for eyeballing")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)

    rows = [json.loads(l) for l in open(args.file)]
    print(f"[check] loaded {len(rows)} rows from {args.file}\n")

    # --- 1. INSPECT: print a few decoded rows ---
    for row in rows[:args.n]:
        prompt, answer, gold = decode_row(row, tok)
        # show only the tail of the prompt (the doc+summary part is long)
        print("=" * 70)
        print(f"gold label = {gold}  ({'consistent' if gold == 1 else 'inconsistent'})")
        print(f"answer tokens decode to: {answer!r}")
        print(f"prompt tail: ...{prompt[-400:]}")
        print()

    # --- 2. GUARD: known-answer probes ---
    print("=" * 70)
    print("[guard] checking sentinel probes (fail = polarity inverted)\n")
    # index rows by decoded summary text for lookup
    decoded = [(decode_row(r, tok)[0], r["meta"]["label"]) for r in rows]

    failures, checked = [], 0
    for needle, expected_label in EXPECTED:
        hits = [(txt, lbl) for txt, lbl in decoded if needle in txt]
        if not hits:
            print(f"  [skip] probe not found: {needle!r}")
            continue
        checked += 1
        # if any matching row has the wrong label, that's a failure
        actual = hits[0][1]
        ok = (actual == expected_label)
        print(f"  [{'ok' if ok else 'FAIL'}] {needle!r}: expected {expected_label}, got {actual}")
        if not ok:
            failures.append((needle, expected_label, actual))

    print()
    if not checked:
        print("[guard] WARNING: no probes matched — cannot verify polarity. "
              "Check that --max-doc-chars didn't drop these examples.")
        sys.exit(2)
    if failures:
        print(f"[guard] FAILED: {len(failures)} probe(s) inverted. "
              f"usb_test.jsonl polarity is WRONG — regenerate load_usb.py with the "
              f"opposite --flip-labels setting before using this file.")
        sys.exit(1)
    print(f"[guard] PASSED: all {checked} probes match. Polarity is correct "
          f"(1=consistent, 0=inconsistent). Safe to use as OOD test set.")


if __name__ == "__main__":
    main()
