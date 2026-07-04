"""
benchmark_vllm.py — offline serving benchmark for the merged judge model
========================================================================

OFFLINE benchmark (not an online server): we batch-score judge prompts, which is
exactly the judge's real workload (score a dataset), and the primary metric is
throughput. We also report latency percentiles + TTFT for completeness.

Measures, per batch size:
  - throughput: total output tokens / wall time  (tok/s)  ← headline number
  - per-request latency: p50 / p95 (ms)
  - TTFT: time to first token (ms), single-request
Sweeps batch sizes and (optionally) dtype so you can report the tradeoff.

Usage:
    pip install vllm
    python benchmark_vllm.py --model /content/judge_merged \
        --data ./test.jsonl --dtype bfloat16 \
        --batch-sizes 1 8 16 32 --num-prompts 128
"""

import argparse
import json
import time
import statistics


IGNORE_INDEX = -100


def load_prompts(path, tok, n):
    """Reconstruct judge PROMPT strings from pre-tokenized rows (cut at answer)."""
    rows = [json.loads(l) for l in open(path)][:n]
    prompts = []
    for r in rows:
        labels = r["labels"]
        ans_start = next((i for i, l in enumerate(labels) if l != IGNORE_INDEX), len(labels))
        prompts.append(tok.decode(r["input_ids"][:ans_start], skip_special_tokens=True))
    return prompts


def bench_throughput(llm, sampling_params, prompts, batch_size):
    """Batched offline generation. Returns (tok_s, total_out_tokens, wall_s)."""
    total_out = 0
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outs = llm.generate(batch, sampling_params)   # vLLM batches internally
        for o in outs:
            total_out += len(o.outputs[0].token_ids)
    wall = time.perf_counter() - t0
    return total_out / wall, total_out, wall


def bench_latency(llm, sampling_params, prompts, n=50):
    """Per-request latency (one prompt at a time) → p50/p95 ms."""
    lats = []
    for p in prompts[:n]:
        t0 = time.perf_counter()
        llm.generate([p], sampling_params)
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    p50 = statistics.median(lats)
    p95 = lats[int(len(lats) * 0.95) - 1]
    return round(p50, 1), round(p95, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir")
    ap.add_argument("--data", default="./test.jsonl", help="jsonl to draw prompts from")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--quantization", default=None,
                    help="e.g. awq / gptq — leave unset for bf16/fp16")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32])
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--max-output-tokens", type=int, default=8,  # judge answer is tiny
                    help="judge emits {\"consistency\": N} — a handful of tokens")
    ap.add_argument("--max-model-len", type=int, default=1280)   # ~1024 prompt + margin
    ap.add_argument("--out", default="vllm_benchmark.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = load_prompts(args.data, tok, args.num_prompts)
    print(f"[bench] {len(prompts)} prompts from {args.data}")

    # deterministic decode (judge is greedy); short output
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens)

    print(f"[bench] loading vLLM: {args.model} dtype={args.dtype} "
          f"quant={args.quantization or 'none'}")
    llm = LLM(model=args.model, dtype=args.dtype, quantization=args.quantization,
              max_model_len=args.max_model_len, seed=2025)

    # TTFT (single request, first-token time) — approximate via a 1-token generation
    ttft_params = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.perf_counter()
    llm.generate([prompts[0]], ttft_params)
    ttft_ms = round((time.perf_counter() - t0) * 1000, 1)

    results = {"model": args.model, "dtype": args.dtype,
               "quantization": args.quantization, "ttft_ms": ttft_ms,
               "by_batch": {}}
    print(f"\n[bench] TTFT (single req, 1 tok): {ttft_ms} ms\n")
    print(f"{'batch':>6}{'tok/s':>12}{'p50 ms':>10}{'p95 ms':>10}")

    for bs in args.batch_sizes:
        tok_s, total_out, wall = bench_throughput(llm, sampling_params, prompts, bs)
        p50, p95 = bench_latency(llm, sampling_params, prompts)
        results["by_batch"][bs] = {"tok_s": round(tok_s, 1), "p50_ms": p50,
                                   "p95_ms": p95, "total_out_tokens": total_out,
                                   "wall_s": round(wall, 2)}
        print(f"{bs:>6}{tok_s:>12.1f}{p50:>10}{p95:>10}")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[bench] wrote {args.out}")
    best = max(results["by_batch"].items(), key=lambda kv: kv[1]["tok_s"])
    print(f"[bench] peak throughput: {best[1]['tok_s']} tok/s at batch {best[0]} ")
    print("[bench] TIP: re-run with --quantization awq (needs an AWQ build of the "
          "model) to report the bf16-vs-quantized tradeoff.")


if __name__ == "__main__":
    main()
