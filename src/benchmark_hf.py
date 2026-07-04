"""
benchmark_hf.py — serving benchmark on the HF/transformers stack (vLLM fallback)
================================================================================

Same metrics as benchmark_vllm.py (throughput tok/s, p50/p95 latency, TTFT,
per-batch sweep) but using plain transformers batched generation — no vLLM, no
CUDA-version matching. Runs on the exact stack you already trained/merged with.

For a judge with ~8-token outputs, the workload is REQUEST-bound, not
generation-bound, so the throughput story is close to vLLM's. Reports both
tok/s AND documents-judged/sec (the metric that actually matters for a judge
scoring a dataset).

Usage:
    python benchmark_hf.py --model /content/work/judge_merged_fibusb \
        --data ./data/test.jsonl --batch-sizes 1 8 16 --num-prompts 128
"""

import argparse
import json
import time
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


def load_prompts(path, tok, n):
    rows = [json.loads(l) for l in open(path)][:n]
    prompts = []
    for r in rows:
        labels = r["labels"]
        ans_start = next((i for i, l in enumerate(labels) if l != IGNORE_INDEX), len(labels))
        prompts.append(tok.decode(r["input_ids"][:ans_start], skip_special_tokens=True))
    return prompts


@torch.no_grad()
def gen_batch(model, tok, prompts, max_new_tokens):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=1280).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    total_out = int((new_tokens != (tok.pad_token_id or tok.eos_token_id)).sum())
    return total_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="./data/test.jsonl")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32])
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default="hf_benchmark.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompts = load_prompts(args.data, tok, args.num_prompts)
    print(f"[bench] {len(prompts)} prompts from {args.data}")

    # warmup (first call compiles/allocates — don't time it)
    gen_batch(model, tok, prompts[:2], args.max_new_tokens)

    # TTFT proxy: single request, 1 new token
    t0 = time.perf_counter()
    gen_batch(model, tok, prompts[:1], 1)
    ttft_ms = round((time.perf_counter() - t0) * 1000, 1)

    results = {"model": args.model, "ttft_ms": ttft_ms, "by_batch": {}}
    print(f"\n[bench] TTFT (single req, 1 tok): {ttft_ms} ms\n")
    print(f"{'batch':>6}{'tok/s':>10}{'docs/s':>9}{'p50 ms':>9}{'p95 ms':>9}")

    for bs in args.batch_sizes:
        # throughput: run all prompts in batches of bs
        total_out = 0
        t0 = time.perf_counter()
        for i in range(0, len(prompts), bs):
            total_out += gen_batch(model, tok, prompts[i:i+bs], args.max_new_tokens)
        wall = time.perf_counter() - t0
        tok_s = total_out / wall
        docs_s = len(prompts) / wall

        # per-request latency at this batch size (time each batch, normalize)
        lats = []
        for i in range(0, min(len(prompts), bs*6), bs):
            t1 = time.perf_counter()
            gen_batch(model, tok, prompts[i:i+bs], args.max_new_tokens)
            lats.append((time.perf_counter() - t1) * 1000)
        lats.sort()
        p50 = round(statistics.median(lats), 1)
        p95 = round(lats[int(len(lats)*0.95)-1], 1) if len(lats) > 1 else p50

        results["by_batch"][bs] = {"tok_s": round(tok_s, 1), "docs_s": round(docs_s, 1),
                                   "p50_ms": p50, "p95_ms": p95}
        print(f"{bs:>6}{tok_s:>10.1f}{docs_s:>9.1f}{p50:>9}{p95:>9}")

    json.dump(results, open(args.out, "w"), indent=2)
    best = max(results["by_batch"].items(), key=lambda kv: kv[1]["tok_s"])
    print(f"\n[bench] peak: {best[1]['tok_s']} tok/s ({best[1]['docs_s']} docs/s) "
          f"at batch {best[0]}")
    print(f"[bench] wrote {args.out}")
    print("[bench] NOTE: HF batched generation (vLLM fallback). For a judge, "
          "docs/s is the meaningful metric; tok/s is low only because outputs are ~8 tokens.")


if __name__ == "__main__":
    main()
