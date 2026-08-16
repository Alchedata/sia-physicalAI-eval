#!/usr/bin/env python
"""Head-only fine-tuning of QwenOFT (L1RegressionActionHead) on LIBERO-Spatial LeRobot data.

Pragmatic macOS/MPS pipeline (official train_alphabrain.py needs DeepSpeed -> not usable here):
  stage=stats : compute q01/q99 action normalization stats from parquet files
  stage=cache : frozen Qwen2.5-VL-3B forward -> cache action-token hidden states (B,8,2048) + normalized action chunks
  stage=train : train the 42M-param MLP action head on cached features (L1 loss), save self-contained checkpoint dir

Usage (alphabrain env):
  MPLBACKEND=Agg python train_action_head.py --stage stats
  MPLBACKEND=Agg python train_action_head.py --stage cache --episodes 120 --stride 4
  MPLBACKEND=Agg python train_action_head.py --stage train --steps 3000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/fei/Code/Alchedata/sea-physicalAI-eval")
DATA = ROOT / "lerobot_data/libero_spatial_no_noops_1.0.0_lerobot"
CKPT_DIR = ROOT / "checkpoints/qwenoft_spatial_headonly"
CACHE_DIR = ROOT / "checkpoints/feature_cache_spatial"
BASE_VLM = str(ROOT / "pretrained_models/Qwen2.5-VL-3B-Instruct")
CHUNK = 8
ACTION_DIM = 7
DATASET_KEY = "libero_spatial_no_noops"

os.chdir(ROOT / "AlphaBrain")
sys.path.insert(0, ".")


def framework_cfg_dict():
    return {
        "framework": {
            "name": "QwenOFT",
            "qwenvl": {"base_vlm": BASE_VLM, "attn_implementation": "eager"},
            "action_model": {
                "action_model_type": "L1RegressionActionHead",
                "action_dim": ACTION_DIM,
                "future_action_window_size": CHUNK - 1,
                "past_action_window_size": 0,
            },
        },
        "datasets": {"vla_data": {"image_size": [224, 224]}},
        "trainer": {"pretrained_checkpoint": None},
    }


def load_meta():
    tasks = {}
    with open(DATA / "meta/tasks.jsonl") as f:
        for line in f:
            row = json.loads(line)
            tasks[row["task_index"]] = row["task"]
    info = json.load(open(DATA / "meta/info.json"))
    return tasks, info


def episode_paths(info, ep_idx):
    chunk_size = info.get("chunks_size", 1000)
    chunk = ep_idx // chunk_size
    pq = DATA / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
    vids = {}
    for key in VIDEO_KEYS:
        vids[key] = DATA / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep_idx)
    return pq, vids


VIDEO_KEYS = []  # filled in main from info.json


def compute_stats(args):
    import pandas as pd

    files = sorted((DATA / "data/chunk-000").glob("episode_*.parquet"))
    acts = []
    for f in files:
        df = pd.read_parquet(f, columns=[ACTION_COL])
        acts.append(np.stack(df[ACTION_COL].to_numpy()))
    a = np.concatenate(acts, 0)
    q01 = np.quantile(a, 0.01, axis=0)
    q99 = np.quantile(a, 0.99, axis=0)
    stats = {
        DATASET_KEY: {
            "action": {
                "q01": q01.tolist(),
                "q99": q99.tolist(),
                "min": a.min(0).tolist(),
                "max": a.max(0).tolist(),
                "mean": a.mean(0).tolist(),
                "std": a.std(0).tolist(),
                "mask": [True] * 6 + [False],
                "norm_mode": "q99",
            }
        }
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("frames:", a.shape, "q01:", np.round(q01, 4), "q99:", np.round(q99, 4))
    print("gripper uniques:", np.unique(a[:, 6])[:10])
    print("STATS_DONE")


def normalize_actions(a, st):
    q01 = np.array(st["q01"])
    q99 = np.array(st["q99"])
    out = np.clip(2 * (a - q01) / np.maximum(q99 - q01, 1e-8) - 1, -1, 1)
    # gripper dim: dataset values in [-1,1] (or {-1,1}); target the {0,1} convention used by
    # BaseFramework.unnormalize_actions (threshold 0.5, mask[6]=False)
    out[:, 6] = (a[:, 6] > 0).astype(np.float64)
    return out


def decode_video(path):
    import av

    frames = []
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    return frames


def build_model():
    import torch
    from omegaconf import OmegaConf

    from AlphaBrain.model.framework.QwenOFT import Qwenvl_OFT

    cfg = OmegaConf.create(framework_cfg_dict())
    model = Qwenvl_OFT(cfg)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(dev).eval(), dev


def cache_features(args):
    import pandas as pd
    import torch
    from PIL import Image

    st = json.load(open(CKPT_DIR / "dataset_statistics.json"))[DATASET_KEY]["action"]
    tasks, info = load_meta()
    model, dev = build_model()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_eps = min(args.episodes, info["total_episodes"])
    ep_indices = np.linspace(0, info["total_episodes"] - 1, n_eps).astype(int)
    t_start = time.time()
    done = 0
    for ep in ep_indices:
        out_path = CACHE_DIR / f"ep_{ep:06d}.npz"
        if out_path.exists():
            done += 1
            continue
        pq, vids = episode_paths(info, int(ep))
        df = pd.read_parquet(pq)
        actions = np.stack(df[ACTION_COL].to_numpy())  # [T,7]
        task_idx = int(df["task_index"].iloc[0])
        instruction = tasks[task_idx]
        T = len(df)
        frame_ids = list(range(0, T, args.stride))
        video_frames = {k: decode_video(v) for k, v in vids.items()}
        feats, labels = [], []
        for i in range(0, len(frame_ids), args.batch):
            batch_fids = frame_ids[i : i + args.batch]
            batch_images = [
                [Image.fromarray(video_frames[k][t]) for k in VIDEO_KEYS] for t in batch_fids
            ]
            instrs = [instruction] * len(batch_fids)
            with torch.no_grad():
                q = model.get_action_queries(batch_images=batch_images, instructions=instrs)
            feats.append(q.float().cpu().numpy().astype(np.float16))
            for t in batch_fids:
                chunk = actions[t : t + CHUNK]
                if len(chunk) < CHUNK:
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], CHUNK - len(chunk), 0)], 0)
                labels.append(normalize_actions(chunk, st))
        np.savez_compressed(
            out_path,
            feats=np.concatenate(feats, 0),
            labels=np.stack(labels).astype(np.float32),
            task_index=task_idx,
        )
        done += 1
        el = time.time() - t_start
        print(f"[cache] ep {ep} ({done}/{n_eps}) frames={len(frame_ids)} elapsed={el/60:.1f}min", flush=True)
    print("CACHE_DONE")


def save_checkpoint(head, step, losses):
    import torch
    from omegaconf import OmegaConf

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    sd = {f"action_model.{k}": v.cpu() for k, v in head.state_dict().items()}
    torch.save(sd, CKPT_DIR / "pytorch_model.pt")
    OmegaConf.save(OmegaConf.create(framework_cfg_dict()), CKPT_DIR / "framework_config.yaml")
    with open(CKPT_DIR / "train_log.json", "w") as f:
        json.dump({"step": step, "losses": losses}, f)


def train(args):
    import torch

    from AlphaBrain.model.modules.action_model.mlp_action_header import L1RegressionActionHead

    files = sorted(CACHE_DIR.glob("ep_*.npz"))
    feats = np.concatenate([np.load(f)["feats"] for f in files], 0)
    labels = np.concatenate([np.load(f)["labels"] for f in files], 0)
    print(f"train set: feats {feats.shape} labels {labels.shape} from {len(files)} episodes")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    X = torch.tensor(feats, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32)
    n = len(X)
    rng = np.random.default_rng(0)
    val_idx = rng.choice(n, size=max(64, n // 20), replace=False)
    val_mask = np.zeros(n, bool)
    val_mask[val_idx] = True
    Xtr, Ytr, Xva, Yva = X[~val_mask], Y[~val_mask], X[val_mask].to(dev), Y[val_mask].to(dev)

    head = L1RegressionActionHead(
        input_dim=2048, hidden_dim=4096, action_dim=ACTION_DIM, NUM_ACTIONS_CHUNK=CHUNK
    ).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    lossf = torch.nn.L1Loss()
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.choice(len(Xtr), size=args.batch, replace=False))
        xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
        pred = head.predict_action(xb)
        loss = lossf(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            head.eval()
            with torch.no_grad():
                vl = lossf(head.predict_action(Xva), Yva).item()
            head.train()
            losses.append({"step": step, "train_l1": round(loss.item(), 5), "val_l1": round(vl, 5)})
            print(f"step {step} train_l1 {loss.item():.5f} val_l1 {vl:.5f} elapsed {time.time()-t0:.0f}s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(head, step, losses)
    print("TRAIN_DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["stats", "cache", "train"])
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=500)
    args = p.parse_args()

    tasks, info = load_meta()
    ACTION_COL = "actions" if "actions" in info["features"] else "action"
    # order must match eval client: primary agentview first, then wrist
    VIDEO_KEYS[:] = sorted(
        (k for k, v in info["features"].items() if v.get("dtype") == "video"),
        key=lambda k: ("wrist" in k, k),
    )
    print("action col:", ACTION_COL, "video keys:", VIDEO_KEYS)
    if args.stage == "stats":
        compute_stats(args)
    elif args.stage == "cache":
        cache_features(args)
    else:
        if args.batch == 8:
            args.batch = 256
        train(args)
