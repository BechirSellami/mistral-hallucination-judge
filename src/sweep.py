"""
sweep.py — hyperparameter sweep for the hallucination judge
==================================================================

Orchestrates: for each config -> train (train_judge_qlora) -> merge (merge_judge)
-> eval on val.jsonl (eval_judge) -> append val F1 to a results table.

Philosophy: ONE-AXIS-AT-A-TIME from a baseline, not a full grid. You learn what
moves val F1 and what doesn't — that's the deliverable, not just a winner.
Everything scores on val.jsonl; test.jsonl stays SEALED.

Colab/resume design:
- Train on fast local disk under --work-dir.
- If --drive-root is set, train_judge_qlora.py periodically syncs checkpoint-*
  and final_adapter to Drive under --drive-root/<config_name>.
- sweep_results.json is also mirrored to Drive, so completed configs are skipped
  after a Colab restart.
- With --resume, this script copies Drive checkpoint-* folders back to local
  before calling train_judge_qlora.py --resume.

Typical usage:
    python sweep.py \
      --base-model mistralai/Mistral-7B-Instruct-v0.3 \
      --data-dir . \
      --work-dir /content/work/sweep \
      --drive-root /content/drive/MyDrive/judge_sweep \
      --val ./val.jsonl \
      --only baseline r32 r64 attn_mlp lr1e4

Resume after Colab crash:
    python sweep.py \
      --base-model mistralai/Mistral-7B-Instruct-v0.3 \
      --data-dir . \
      --work-dir /content/work/sweep \
      --drive-root /content/drive/MyDrive/judge_sweep \
      --val ./val.jsonl \
      --resume \
      --only baseline r32 r64 attn_mlp lr1e4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_MLP = ATTN + ["gate_proj", "up_proj", "down_proj"]

# Baseline + one-axis-at-a-time variations. Each config overrides only what differs.
BASELINE = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lr": 2e-4,
    "target_modules": ATTN,
    "train_file": "train.jsonl",
}

SWEEP = [
    {"name": "baseline"},                                  # r16 a32 attn lr2e-4 FIB
    {"name": "r8",       "lora_r": 8,  "lora_alpha": 16},
    {"name": "r32",      "lora_r": 32, "lora_alpha": 64},
    {"name": "r64",      "lora_r": 64, "lora_alpha": 128},
    {"name": "attn_mlp", "target_modules": ATTN_MLP},      # capacity via MLP layers
    {"name": "lr1e4",    "lr": 1e-4},
    {"name": "fib_usb",  "train_file": "train_fib_usb.jsonl"},  # data-mix ablation
]


def cfg_for(entry):
    c = dict(BASELINE)
    c.update({k: v for k, v in entry.items() if k != "name"})
    c["name"] = entry["name"]
    return c


def run(cmd, env_updates=None):
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    print(f"\n$ {' '.join(str(x) for x in cmd)}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"step failed ({r.returncode}): {cmd[:3]}...")


def copy_missing_dirs(src_root: Path, dst_root: Path, pattern: str):
    """Copy matching directories from src_root to dst_root if missing locally."""
    if not src_root.exists():
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_root.glob(pattern)):
        if not src.is_dir():
            continue
        dst = dst_root / src.name
        if dst.exists():
            continue
        print(f"[resume] copying {src} -> {dst}")
        shutil.copytree(src, dst)


def load_results(local_results_path: Path, drive_results_path: Path | None):
    """Prefer local results; after Colab restart, recover from Drive if available."""
    if local_results_path.exists():
        return json.load(open(local_results_path))
    if drive_results_path and drive_results_path.exists():
        local_results_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_results_path, local_results_path)
        print(f"[resume] restored sweep results from Drive -> {local_results_path}")
        return json.load(open(local_results_path))
    return []


def save_results(results, local_results_path: Path, drive_results_path: Path | None):
    local_results_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(local_results_path, "w"), indent=2)
    if drive_results_path:
        drive_results_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_results_path, drive_results_path)
        print(f"[save] mirrored results -> {drive_results_path}")


def one_config(c, args):
    name = c["name"]
    work = Path(args.work_dir) / name
    adapter = work / "final_adapter"
    merged = work / "merged"

    drive_dir = Path(args.drive_root) / name if args.drive_root else None
    drive_dir_arg = str(drive_dir) if drive_dir else ""

    print(
        f"\n{'=' * 70}\n"
        f"CONFIG: {name}  {json.dumps({k: v for k, v in c.items() if k != 'name'})}\n"
        f"{'=' * 70}"
    )

    # After a Colab crash, local /content is wiped. Rehydrate local checkpoints
    # from Drive before asking Trainer to resume.
    if args.resume and drive_dir:
        copy_missing_dirs(drive_dir, work, "checkpoint-*")

    # 1. train
    cmd = [
        sys.executable,
        "train_judge_qlora.py",
        "--skip-overfit-check",
        "--train", str(Path(args.data_dir) / c["train_file"]),
        "--eval", args.val,
        "--base-model", args.base_model,
        "--output-dir", str(work),
        "--drive-dir", drive_dir_arg,
        "--sync-every-min", str(args.sync_every_min),
        "--run-name", f"sweep_{name}",
        "--max-len", str(args.max_len),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--lora-r", str(c["lora_r"]),
        "--lora-alpha", str(c["lora_alpha"]),
        "--lr", str(c["lr"]),
        "--target-modules", *c["target_modules"],
    ]

    if args.resume:
        cmd.append("--resume")

    # Fixes the old bug where "--no-wandb 999" was accidentally produced.
    if args.no_wandb:
        cmd.append("--no-wandb")

    # W&B environment defaults. The CLI --run-name above is the source of truth
    # for the visible run name; WANDB_NAME is kept as a fallback for older setups.
    env_updates = {
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_NAME": f"sweep_{name}",
        "WANDB_DIR": args.wandb_dir,
    }

    run(cmd, env_updates=env_updates)

    # 2. merge (adapter lives in output-dir/final_adapter)
    run([
        sys.executable,
        "merge_judge.py",
        "--merge",
        "--base-model", args.base_model,
        "--adapter", str(adapter),
        "--merged-dir", str(merged),
    ])

    # 3. eval on val
    out_json = work / "val_results.json"
    run([
        sys.executable,
        "eval_judge.py",
        "--model", str(merged),
        "--data", args.val,
        "--out", str(out_json),
        "--batch-size", str(args.eval_batch),
    ])

    res = json.load(open(out_json))
    return {
        "name": name,
        **{k: v for k, v in c.items() if k != "name"},
        "val_f1": res["overall"]["f1"],
        "val_acc": res["overall"]["accuracy"],
        "per_source": {s: m["f1"] for s, m in res["per_source"].items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--val", default="./val.jsonl")
    ap.add_argument("--work-dir", default="/content/sweep")

    # Drive root for persistence across Colab restarts. Each config gets its own
    # directory: <drive-root>/<config_name>/checkpoint-* and final_adapter.
    ap.add_argument("--drive-root", default="",
                    help="Drive root for checkpoint persistence, e.g. /content/drive/MyDrive/judge_sweep")
    ap.add_argument("--sync-every-min", type=float, default=15.0,
                    help="minutes between checkpoint copies to Drive")
    ap.add_argument("--resume", action="store_true",
                    help="copy Drive checkpoint-* back to local and pass --resume to train_judge_qlora.py")

    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)

    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="hallucination-judge")
    ap.add_argument("--wandb-dir", default="/content/wandb",
                    help="local W&B cache; online W&B still preserves synced metrics")

    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these config names (e.g. --only baseline r32)")
    args = ap.parse_args()

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    local_results_path = Path(args.work_dir) / "sweep_results.json"
    drive_results_path = (Path(args.drive_root) / "sweep_results.json") if args.drive_root else None
    results = load_results(local_results_path, drive_results_path)
    done = {r["name"] for r in results}

    for entry in SWEEP:
        if args.only and entry["name"] not in args.only:
            continue
        if entry["name"] in done:
            print(f"[skip] {entry['name']} already in results")
            continue

        c = cfg_for(entry)
        t0 = time.time()
        try:
            row = one_config(c, args)
            row["minutes"] = round((time.time() - t0) / 60, 1)
            results.append(row)
            done.add(row["name"])
            save_results(results, local_results_path, drive_results_path)
            print(f"[done] {c['name']}: val_f1={row['val_f1']} ({row['minutes']}m)")
        except Exception as e:
            print(f"[FAIL] {c['name']}: {e}  — continuing to next config")

    # final comparison table
    print(f"\n{'=' * 70}\nSWEEP RESULTS (val F1)\n{'=' * 70}")
    print(f"{'config':<12}{'r':>4}{'alpha':>7}{'lr':>10}{'target':>12}{'val_f1':>10}{'val_acc':>10}")
    for r in sorted(results, key=lambda x: -x["val_f1"]):
        tgt = "attn+mlp" if len(r.get("target_modules", ATTN)) > 4 else "attn"
        print(
            f"{r['name']:<12}{r.get('lora_r', ''):>4}{r.get('lora_alpha', ''):>7}"
            f"{r.get('lr', ''):>10}{tgt:>12}{r['val_f1']:>10.4f}{r['val_acc']:>10.4f}"
        )

    if results:
        best = max(results, key=lambda x: x["val_f1"])
        print(f"\nBest: {best['name']} — freeze this config, then run the eval on test.jsonl + usb_test.jsonl.")
    else:
        print("\nNo successful results yet.")


if __name__ == "__main__":
    main()
