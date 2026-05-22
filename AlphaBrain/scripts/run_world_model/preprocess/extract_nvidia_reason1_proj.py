# Copyright 2026 VLA-Engine. All rights reserved.
#
# Licensed under the VLA-Engine License.

"""One-time extractor for NVIDIA's pretrained Reason1 cross-attn projection.

Reads the NVIDIA Cosmos-Predict2.5 action-cond EMA checkpoint
(~4.25 GB), pulls out the two tensors that initialize the Reason1
28-layer concat -> 1024 projection head, validates their shape/dtype,
and writes a ~200 MB companion file that can be loaded in milliseconds
at model init time.

Source ckpt key path:
    net.crossattn_proj.0.weight  -> (1024, 100352)
    net.crossattn_proj.0.bias    -> (1024,)

Output file payload:
    {"weight": Tensor(1024, 100352, bf16),
     "bias":   Tensor(1024, bf16)}

Usage:
    python scripts/extract_nvidia_reason1_proj.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

DEFAULT_SRC = (
    "data/pretrained_models/Cosmos-Predict2.5-2B/robot/action-cond/"
    "38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
)
DEFAULT_DST = "data/pretrained_models/reason1_proj_pretrained.pt"

KEY_W = "net.crossattn_proj.0.weight"
KEY_B = "net.crossattn_proj.0.bias"
EXPECTED_W_SHAPE = (1024, 100352)
EXPECTED_B_SHAPE = (1024,)


def _unwrap(sd):
    """Allow {state_dict: {...}}, {model: {...}}, {ema: {...}} wrappers."""
    if isinstance(sd, dict) and KEY_W in sd:
        return sd
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "ema"):
            inner = sd.get(k)
            if isinstance(inner, dict) and KEY_W in inner:
                return inner
    raise KeyError(
        f"Key {KEY_W!r} not found in checkpoint (top-level or in "
        "state_dict/model/ema wrappers)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", default=DEFAULT_DST)
    args = parser.parse_args()

    src = args.src
    dst = args.dst

    if not os.path.isfile(src):
        print(f"[extract] ERROR: source ckpt not found: {src}", file=sys.stderr)
        sys.exit(1)

    src_size_gb = os.path.getsize(src) / (1024**3)
    print(f"[extract] Loading NVIDIA ckpt: {src}  ({src_size_gb:.2f} GB)")
    sd = torch.load(src, map_location="cpu", weights_only=False)
    sd = _unwrap(sd)

    W = sd[KEY_W]
    b = sd[KEY_B]

    # Shape / dtype assertions
    assert tuple(W.shape) == EXPECTED_W_SHAPE, (
        f"weight shape mismatch: got {tuple(W.shape)}, expected {EXPECTED_W_SHAPE}"
    )
    assert tuple(b.shape) == EXPECTED_B_SHAPE, (
        f"bias shape mismatch: got {tuple(b.shape)}, expected {EXPECTED_B_SHAPE}"
    )
    assert W.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"unexpected weight dtype: {W.dtype}"
    )

    W_bf16 = W.detach().to(torch.bfloat16).contiguous()
    b_bf16 = b.detach().to(torch.bfloat16).contiguous()

    # Stats (cast to float32 for a stable mean/std)
    W_f32 = W_bf16.float()
    b_f32 = b_bf16.float()
    print("[extract] --- tensor summary ---")
    print(
        f"  weight: shape={tuple(W_bf16.shape)} dtype={W_bf16.dtype} "
        f"mean={W_f32.mean().item():+.4e} std={W_f32.std().item():.4e} "
        f"min={W_f32.min().item():+.4e} max={W_f32.max().item():+.4e}"
    )
    print(
        f"  bias  : shape={tuple(b_bf16.shape)} dtype={b_bf16.dtype} "
        f"mean={b_f32.mean().item():+.4e} std={b_f32.std().item():.4e} "
        f"min={b_f32.min().item():+.4e} max={b_f32.max().item():+.4e}"
    )

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    payload = {"weight": W_bf16, "bias": b_bf16}
    torch.save(payload, dst)

    dst_size_mb = os.path.getsize(dst) / (1024**2)
    # Theoretical: 1024 * 100352 * 2 bytes + 1024 * 2 bytes ~= 195.97 MB
    print(f"[extract] Wrote: {dst}  ({dst_size_mb:.2f} MB)")
    print("[extract] done.")


if __name__ == "__main__":
    main()
