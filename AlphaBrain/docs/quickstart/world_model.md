# World Model

Train and evaluate world-model visual backbones on LIBERO. Two setups:

1. **World Model + GR00T** — Cosmos 2.0 / Cosmos 2.5 / WAN 2.2 / V-JEPA 2 paired with a GR00T FlowMatching DiT action decoder.
2. **Cosmos Policy** — full-DiT finetune of NVIDIA `Cosmos-Predict2-2B-Video2World` as a direct policy.

---

## Prerequisites

Base AlphaBrain env already set up (see [Installation](installation.md)). Extras:

```bash
pip install 'diffusers==0.36.0'          # Cos 2.0/2.5 pin
git clone https://github.com/Lifelong-Robot-Learning/LIBERO && cd LIBERO && pip install -e .
export LIBERO_HOME=/abs/path/to/LIBERO
```

Download the backbones you need under `data/pretrained_models/` (Cosmos-Predict2-2B-Video2World, Cosmos-Predict2.5-2B-diffusers, Cosmos-Reason1-7B, Wan2.2-TI2V-5B, vjepa2, t5-small, …) and LIBERO LeRobot datasets under `data/datasets/libero_datasets/`.

## Precompute text embeddings (required for Cos 2.0 / 2.5)

```bash
python preprocess/precompute_text_embeddings/precompute_t5.py        # Cos 2.0, V-JEPA
python preprocess/precompute_text_embeddings/precompute_reason1.py   # Cos 2.5
python preprocess/precompute_text_embeddings/precompute_umt5.py      # WAN 2.2
python preprocess/extract_nvidia_reason1_proj.py                     # Cos 2.5 init
```

---

## Train — World Model + GR00T

```bash
MODEL=cos2       bash scripts/run_world_model/train/run_world_model.sh
MODEL=cos25_4gpu bash scripts/run_world_model/train/run_world_model.sh
MODEL=vjepa      bash scripts/run_world_model/train/run_world_model.sh
MODEL=wan22      bash scripts/run_world_model/train/run_world_model.sh

# resume
MODEL=cos2 RESUME=true bash scripts/run_world_model/train/run_world_model.sh
```

## Train — Cosmos Policy

```bash
bash scripts/run_world_model/train/run_cosmos_policy.sh
```

---

## Evaluate — World Model

```bash
CKPT=results/training/<run_id>/checkpoints/steps_30000 \
  bash scripts/run_world_model/eval/eval_world_model.sh

# side-by-side predicted-vs-rollout video
CKPT=... PREDICT_VIDEO=true \
  bash scripts/run_world_model/eval/eval_world_model.sh
```

## Evaluate — Cosmos Policy

```bash
bash scripts/run_world_model/eval/eval_cosmos_policy.sh                      # vs official ckpt
CKPT_DIR=results/training/<run>/checkpoints/steps_40000 \
  bash scripts/run_world_model/eval/eval_cosmos_policy.sh
```

Results: `results/evaluation/<suite>/<ckpt_tag>-<timestamp>/`.

---

Full env-var reference, required-checkpoint table, reproduction numbers, and troubleshooting: [`scripts/run_world_model/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_world_model/README.md).
