# World Model Quick Start

Train and evaluate world-model visual backbones on LIBERO.

This directory covers two related setups:

1. **World Model + GR00T** — Cosmos 2.0 / Cosmos 2.5 / WAN 2.2 / V-JEPA 2 as the visual backbone, paired with a GR00T FlowMatching DiT action decoder.
2. **Cosmos Policy** — full-DiT finetune of NVIDIA Cosmos-Predict2-2B-Video2World as a direct policy (single-model alternative; eval against the official `Cosmos-Policy-LIBERO-Predict2-2B` checkpoint).

Currently only the `QwenGR00T` action decoder is wired up here. PI and OFT decoders are planned but not configured yet.

---

## 1. Prerequisites

Assumes the **base AlphaBrain environment is already set up** (see the top-level project README for that). Below are the extras this WM / Cosmos Policy pipeline expects on top.

### Extra Python packages

`cosmos2-diffusers` backend pins `diffusers==0.36.0` — newer releases change the VAE API and break the cross-attn / text-encoder shims silently. If the base env ships a newer version, downgrade it for this pipeline:

```bash
pip install 'diffusers==0.36.0'
```

Everything else (`transformers`, `accelerate`, `deepspeed`, `torch`, `omegaconf`, `imageio`, `av`, ...) is already part of the base environment.

### LIBERO simulator

Clone and install the LIBERO repo separately (the simulator, not a Python package we ship):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
cd LIBERO && pip install -e .
export LIBERO_HOME=/abs/path/to/LIBERO
```

All eval scripts expect `LIBERO_HOME` to be set.

---

## 2. Required checkpoints and data

All paths are **relative to the project root**. Default layout:

```
data/
├── pretrained_models/
│   ├── Cosmos-Predict2-2B-Video2World/       # Cos 2.0 backbone + T5 text encoder + VAE
│   ├── Cosmos-Predict2.5-2B-diffusers/       # Cos 2.5 backbone
│   ├── Cosmos-Reason1-7B/                    # Reason1 text encoder (for Cos 2.5)
│   ├── Cosmos-Policy-LIBERO-Predict2-2B/     # Official CP eval ckpt (HF)
│   ├── Wan2.2-TI2V-5B/                       # WAN 2.2 backbone + bundled UMT5-XXL
│   ├── vjepa2/                               # V-JEPA 2.1 ViT-Gigantic
│   ├── t5-small/                             # small text encoder used by some recipes
│   └── text_embeddings/                      # cached outputs of the precompute scripts
└── datasets/
    └── libero_datasets/                      # LeRobot-format LIBERO suites
        ├── libero_goal_no_noops_1.0.0_lerobot/
        ├── libero_spatial_no_noops_1.0.0_lerobot/
        ├── libero_object_no_noops_1.0.0_lerobot/
        └── libero_10_no_noops_1.0.0_lerobot/
```

Source for each (download yourself via HuggingFace or the original release):

| Purpose                         | Name                                  | Used by           |
|---------------------------------|---------------------------------------|-------------------|
| Cos 2.0 backbone + T5 + VAE     | `Cosmos-Predict2-2B-Video2World`      | `MODEL=cos2`, CP  |
| Cos 2.5 backbone                | `Cosmos-Predict2.5-2B-diffusers`      | `MODEL=cos25_4gpu`|
| Cos 2.5 text encoder            | `Cosmos-Reason1-7B`                   | `MODEL=cos25_4gpu`|
| Cosmos Policy eval ckpt         | `Cosmos-Policy-LIBERO-Predict2-2B`    | `eval_cosmos_policy.sh` |
| WAN 2.2 backbone (+ UMT5-XXL)   | `Wan2.2-TI2V-5B`                      | `MODEL=wan22`     |
| V-JEPA 2.1 ViT-G                | `vjepa2` (`vjepa2_1_vitG_384.pt`)     | `MODEL=vjepa`     |
| T5 small                        | `t5-small`                            | V-JEPA text cond. |
| LIBERO LeRobot datasets         | `libero_{goal,spatial,object,10}_no_noops_1.0.0_lerobot` | training + eval |

---

## 3. Precompute text embeddings

These scripts dedupe instructions across suites and cache encoder outputs as pkl files that training and eval load directly.

**Required for Cosmos 2.0 / 2.5.** Without the pkl the backbone silently falls back to dummy zero conditioning — training and inference will appear to run but produce no meaningful instruction-following behavior.

**Recommended for WAN 2.2 and V-JEPA 2.1.** Both backbones have online-encoder fallbacks (UMT5-XXL 5.7B and T5-small 60M respectively), but precomputing saves GPU memory at startup and a small amount of per-step overhead.

Run each one-liner from project root:

| Script                                                                 | Output                                                          | Used by            |
|------------------------------------------------------------------------|-----------------------------------------------------------------|--------------------|
| `preprocess/precompute_text_embeddings/precompute_t5.py`               | `data/datasets/libero_datasets/t5_text_embeddings.pkl`          | Cos 2.0, V-JEPA    |
| `preprocess/precompute_text_embeddings/precompute_reason1.py`          | `data/pretrained_models/text_embeddings/reason1_28layer_text_embeddings.pkl` | Cos 2.5 |
| `preprocess/precompute_text_embeddings/precompute_umt5.py`             | `data/datasets/libero_datasets/umt5_text_embeddings.pkl`        | WAN 2.2            |
| `preprocess/extract_nvidia_reason1_proj.py`                            | `data/pretrained_models/reason1_proj_pretrained.pt` (~200 MB)   | Cos 2.5 (init)     |

Each script takes `DATA_ROOT`, output path, etc. as env vars — see the docstring at the top.

---

## 4. Train — World Model + GR00T

One consolidated script, parameterized by `MODEL`:

```bash
MODEL=cos2       bash scripts/run_world_model/train/run_world_model.sh
MODEL=cos25_4gpu bash scripts/run_world_model/train/run_world_model.sh
MODEL=vjepa      bash scripts/run_world_model/train/run_world_model.sh
MODEL=wan22      bash scripts/run_world_model/train/run_world_model.sh

# Resume from latest checkpoint
MODEL=cos2 RESUME=true bash scripts/run_world_model/train/run_world_model.sh
```

| `MODEL` value | Backbone            | Default port | Config                               |
|---------------|---------------------|--------------|--------------------------------------|
| `cos2`        | Cosmos Predict 2 DiT 2B | 29500    | `configs/models/config_cos2.yaml`    |
| `cos25_4gpu`  | Cosmos Predict 2.5 DiT 2B (4-GPU only)| 29501 | `configs/models/config_cos25_4gpu.yaml` |
| `vjepa`       | V-JEPA 2.1 ViT-Gigantic (1.8 B) | 29502 | `configs/models/config_vjepa.yaml` |
| `wan22`       | WAN 2.2 TI2V-5B     | 29503        | `configs/models/config_wan22.yaml`   |

Defaults: 4 GPUs, ZeRO-2. Override via `NUM_GPUS=...`, `MASTER_PORT=...`, `CONFIG_YAML=...`.

Checkpoints land in `results/training/<run_id>/checkpoints/steps_<N>/`. The run id is read from the config `run_id:` field.

---

## 5. Train — Cosmos Policy

```bash
bash scripts/run_world_model/train/run_cosmos_policy.sh
```

This does a full DiT 2B finetune and is memory-heavy (requires 2+ GPUs). It targets effective batch ~1920 via grad accumulation. Tunable via env vars:

```bash
NUM_GPUS=8 PER_DEVICE_BATCH=4 GRAD_ACCUM=60 \
MAX_STEPS=40000 SAVE_INTERVAL=5000 \
  bash scripts/run_world_model/train/run_cosmos_policy.sh
```

See the header of `train/run_cosmos_policy.sh` for the full list of knobs.

---

## 6. Eval — World Model

```bash
CKPT=results/training/<run_id>/checkpoints/steps_30000 \
  bash scripts/run_world_model/eval/eval_world_model.sh
```

This starts `deployment/model_server/server_policy.py` with the ckpt, waits for it to come up, then runs `benchmarks/LIBERO/eval/eval_libero.py` against it. Default: `libero_goal`, 3 trials per task.

```bash
# 50 trials on libero_spatial
CKPT=... TASK_SUITE=libero_spatial NUM_TRIALS=50 \
  bash scripts/run_world_model/eval/eval_world_model.sh

# Side-by-side predicted-vs-rollout video
CKPT=... PREDICT_VIDEO=true \
  bash scripts/run_world_model/eval/eval_world_model.sh
```

With `PREDICT_VIDEO=true` the eval script saves one **816×512 side-by-side mp4 per episode** (left: next-frame prediction sampled from the backbone's `denoise_future_frame`; right: actual LIBERO rollout; caption strip on top showing the task instruction). Output path:

```
results/evaluation/<suite>/<ckpt_name>-<timestamp>/videos/rollout_<task>_episode<n>_<success|failure>.mp4
```

Only diffusion-based WM backbones (Cos 2.0, Cos 2.5, WAN 2.2) actually render a prediction panel. V-JEPA 2.1 and any VLM+decoder checkpoint (QwenOFT / PI / Pi0 / CP) silently ignore the flag — the client still sends `return_predicted_frame=True`, the server's `predict_action(**kwargs)` absorbs it, no `predicted_frame` comes back, and the eval falls through to a normal single-panel mp4.

Results land under `results/evaluation/<task_suite>/<ckpt_tag>-<timestamp>/`.

---

## 7. Eval — Cosmos Policy

Server-client mode (wrapper script):

```bash
# Quick smoke test vs the official pretrained ckpt (defaults to it)
bash scripts/run_world_model/eval/eval_cosmos_policy.sh

# Full 50-trial eval
NUM_TRIALS=50 bash scripts/run_world_model/eval/eval_cosmos_policy.sh

# Custom checkpoint
CKPT_DIR=results/training/<run>/checkpoints/steps_40000 \
  bash scripts/run_world_model/eval/eval_cosmos_policy.sh
```


Known baseline on the official `Cosmos-Policy-LIBERO-Predict2-2B` checkpoint: **LIBERO-Goal 30/30 (100%)** at 3 trials per task.

### Reproduction reference

Training the 8-GPU recipe in `train/run_cosmos_policy.sh` on an internal **B20Z** node (8 × A800-80GB) for only 10K steps reaches within 2.5 points of the official release — **~1/5 of the training budget matches the official SOTA**.

| Checkpoint | libero_goal | libero_spatial | libero_object | libero_10 | **Avg** |
|-----------|:-----------:|:--------------:|:-------------:|:---------:|:-------:|
| Official `Cosmos-Policy-LIBERO-Predict2-2B` | 98.0% | 98.0% | 100.0% | 98.0% | **98.0%** |
| Ours, 10K steps (~1/5 budget) | 96.0% | 96.0% | 96.0% | 94.0% | **95.5%** |

*Eval config: 5 trials/task × 10 tasks × 4 suites = 200 episodes per checkpoint.*

---

## 8. Environment variables reference

| Variable      | Used by           | Meaning                                                       |
|---------------|-------------------|---------------------------------------------------------------|
| `LIBERO_HOME` | all eval scripts  | Absolute path to your LIBERO repo checkout (required)         |
| `PYTHON`      | all eval scripts  | Python interpreter (default: `python`)                        |
| `MODEL`       | `run_world_model.sh` | `cos2` / `cos25_4gpu` / `vjepa` / `wan22`                 |
| `NUM_GPUS`    | train scripts     | Process count for `accelerate launch` (default: 4 for WM, 2 for CP) |
| `MASTER_PORT` | train scripts     | Distributed port (defaults per MODEL)                         |
| `RESUME`      | train scripts     | `true` to resume from latest checkpoint                       |
| `CKPT`        | `eval_world_model.sh` | Checkpoint dir to serve (required)                        |
| `CKPT_DIR`    | `eval_cosmos_policy.sh` | Cosmos Policy ckpt dir                                  |
| `TASK_SUITE`  | eval scripts      | `libero_goal` / `libero_spatial` / `libero_object` / `libero_10` |
| `NUM_TRIALS`  | eval scripts      | Trials per task                                               |
| `GPU_ID`      | eval scripts      | GPU index for the server                                      |
| `PORT`        | eval scripts      | Websocket port                                                |
| `PREDICT_VIDEO` | `eval_world_model.sh` | `true` to render side-by-side visualizations            |

---

## 9. Troubleshooting

- **Wrong `AlphaBrain` loaded.** If you have multiple clones, `pip install -e .` only points to one. Verify with `python -c 'import AlphaBrain; print(AlphaBrain.__file__)'` — the path must be inside the checkout you are running from. Re-run `pip install -e . --no-deps` from the correct root if not.

- **Stale bytecode after `git pull`.** Clear `__pycache__/` globally (`find . -name __pycache__ -exec rm -rf {} +`) if you see old class signatures or AttributeErrors. The training scripts do this automatically before launch.

- **`diffusers` version mismatch.** Cos 2.0 / Cos 2.5 require `diffusers==0.36.0`. Newer versions change the VAE API and will break silently.

---

## Version History

[0421] This release takes the direct approach: a single forward pass that predicts both the next video frame and future actions. There's still plenty of room for improvement. In upcoming versions we'll introduce a range of optimizations and common WM-pipeline techniques such as inverse dynamics. If you have a better solution, PRs are very welcome — let's make progress together and grow the open-source WM community.
