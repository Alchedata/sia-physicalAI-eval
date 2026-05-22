#!/usr/bin/env python3
"""Aggregate per-shard eval JSONs into a single summary.

Each shard JSON (produced by `eval_libero.py --results_json shard_X.json`)
contains a `per_task_sr` dict for the subset of tasks that shard ran.
This script merges them into one summary with all per-task SRs and the
overall SR (mean across tasks).

Usage:
    python AlphaBrain/training/reinforcement_learning/eval/aggregate_shards.py \\
        --out_dir   <DIR containing shard_*.json> \\
        --action_token_ckpt  <ckpt path used> \\
        --vla_ckpt  <vla ckpt path used> \\
        --suite     libero_goal \\
        --n_eps     50
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True,
                   help="Directory containing shard_*.json; summary.json is written here.")
    p.add_argument("--action_token_ckpt", required=True)
    p.add_argument("--vla_ckpt", required=True)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--n_eps", type=int, required=True,
                   help="Episodes per task (recorded in the summary; not re-computed).")
    args = p.parse_args()

    per_task = {}
    for f in sorted(glob.glob(os.path.join(args.out_dir, "shard_*.json"))):
        with open(f) as fp:
            data = json.load(fp)
        if isinstance(data, list):
            data = data[-1]   # eval_libero.py appends to a list
        for tid, sr in data["per_task_sr"].items():
            per_task[int(tid)] = float(sr)

    if not per_task:
        raise SystemExit("ERROR: no per-task SR found in shard JSONs")

    overall = sum(per_task.values()) / len(per_task)
    print(f"\n{'=' * 60}")
    print(f"Eval — {len(per_task)} tasks, {args.n_eps} eps/task")
    print('=' * 60)
    for tid in sorted(per_task):
        print(f"  task_{tid:02d}: {per_task[tid]:.2%}")
    print(f"  Overall: {overall:.2%}")
    print('=' * 60)

    summary = {
        "action_token_ckpt": args.action_token_ckpt,
        "vla_ckpt": args.vla_ckpt,
        "suite": args.suite,
        "n_eps_per_task": args.n_eps,
        "per_task_sr": per_task,
        "overall_sr": overall,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()
