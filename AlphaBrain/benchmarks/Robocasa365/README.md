# RoboCasa365

Evaluation + training for the RoboCasa **365-task** suite (atomic + composite pick-and-place, Panda-Omron). Entry points live in this directory; training & eval launchers live at repo root.

---

## 1. Environment Setup

Two conda environments are involved:

- **AlphaBrain** — Install per the top-level repo README.
- **robocasa365** — Install by following the [official RoboCasa365 repository](https://github.com/robocasa/robocasa).

---

## 2. Training

### 2.1 Dataset preparation

Download the RoboCasa365 dataset following the [robocasa.ai dataset docs](https://robocasa.ai/docs/build/html/datasets/using_datasets.html).

After download, arrange files so `${ROBOCASA365_DATA_ROOT}` matches the layout below — this is the shape expected by the `robocasa365_*` mixtures defined in `AlphaBrain/dataloader/gr00t_lerobot/mixtures.py`:

```
${ROBOCASA365_DATA_ROOT}/
├── pretrain/
│   ├── atomic/<task>/<date>/     
│   │   ├── README.md
│   │   └── lerobot/              
│   └── composite/<task>/<date>/
└── target/
    ├── atomic/<task>/<date>/      
    └── composite/<task>/<date>/
```

Available mixtures (`dataset_mix` names in the training config). RoboCasa365 ships data in two trees — `pretrain/` (broad multi-task data used for pretraining) and `target/` (the specific tasks you evaluate on) — and the mixture name tells you which tree it draws from:

**Pretrain dataset** — draws from `pretrain/`:

| Mixture | Glob |
|---|---|
| `robocasa365_pretrain300` | `pretrain/*/*/*/lerobot` |

**Target-task dataset** — draws from `target/`:

| Mixture | Glob / contents |
|---|---|
| `robocasa365_target_atomic` | `target/atomic/*/*/lerobot` |
| `robocasa365_target_composite_seen` | 16 composite tasks — seen split (DeliverStraw, KettleBoiling, PrepareCoffee, …) |
| `robocasa365_target_composite_unseen` | 16 composite tasks — unseen split (ArrangeBreadBasket, ArrangeTea, BreadSelection, …) |

### 2.2 `.env` setup

Append to repo-root `.env`:

```bash
# RoboCasa365
ROBOCASA365_PYTHON=<your_path_to_robocasa_conda_env>/bin/python
ROBOCASA365_DATA_ROOT=<your_path_to_robocasa365_dataset>
```

### 2.3 Run training

```bash
bash scripts/run_finetune.sh robocasa365_train
```

Two model frameworks are currently validated on RoboCasa365 — **QwenGR00T** (Qwen3-VL-4B + GR00T-style diffusion action head) and **QwenOFT** (Qwen3-VL-4B + OFT action head). The example below uses QwenGR00T via `modes.robocasa365_train` in `configs/finetune_config.yaml`.

**Knobs you will likely change:**

| Setting | Default | Note |
|---|---|---|
| `dataset_mix` | `robocasa365_target_composite_unseen` | The split to train on (see §2.1 mixture table). **Edit in two places:** top-level `dataset_mix` *and* `datasets.vla_data.dataset_mix`. |
| `base_vlm` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` | Path to your local Qwen3-VL-4B weights. |
| `run_id` | `qwen_gr00t_robocasa365_composite_unseen` | Output directory name. |

**Recipes:**

**A. Pretrain.**
1. First pass — set `dataset_mix: robocasa365_pretrain300` and run `bash scripts/run_finetune.sh robocasa365_train`.
2. When the pretrain run finishes, edit the config: under `trainer:` add `pretrained_checkpoint: <path/to/pretrain/checkpoint>`.
3. Switch `dataset_mix` to the target mixture you want — one of `robocasa365_target_atomic`, `robocasa365_target_composite_seen`, or `robocasa365_target_composite_unseen` — and re-launch.

**B. Target-only.** Skip steps 1–2; set `dataset_mix` directly to a target mixture.

---

## 3. Evaluation

Eval runs as two processes communicating over WebSocket:
- a **policy server** (AlphaBrain env) serving actions;
- a **simulation client** (robocasa365 env) stepping MuJoCo envs.

```bash
bash scripts/run_eval.sh robocasa365_eval
```

Reads `modes.robocasa365_eval` in `configs/finetune_config.yaml`. Edit there to change:

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | `results/training/.../final_model` | Path to trained model |
| `task_set` / `task_suite` | `composite_seen` | Options: `composite_seen` / `composite_unseen` / `atomic_seen`. |
| `n_episodes` | 50 | Rollouts per task |
| `n_action_steps` | 16 | Action chunk size per policy query |
| `port` | 5694 | WebSocket port |
| `gpu_id` | 0 | CUDA device for server |

Results + aggregate stats land under `results/evaluation/robocasa365/<split>/<task_slug>/`.

---

## 4. Results

| Suite | GR00T w/o pretrain | GR00T w/ pretrain | QwenGR00T w/o state | QwenOFT |
|-------|:---------------------:|:--------------------:|:---------:|:-------:|
| Atomic-Seen (18) | 59.1% | 68.6% | 75.1% | 66.4% |
| Composite-Seen (16) | 34.6% | 40.6% | 36.1% | 14.4% |
| Composite-Unseen (16) | 30.8% | 42.1% | 37.6% | 17.8% |
| **Average** | **43.7%** | **51.1%** | **50.6%** | **34.2%** |

> **QwenGR00T w/o state** and **QwenOFT** columns report **Target Task Learning Only** (no pretrain stage; trained directly on the target mixture).

> **GR00T w/o pretrain** and **GR00T w/ pretrain** numbers are taken from the official source.
