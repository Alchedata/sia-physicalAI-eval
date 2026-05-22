# RoboCasa Tabletop

Evaluation + training for the **RoboCasa-GR1-Tabletop-Tasks** benchmark — pick-and-place tasks on a Fourier GR1 humanoid (arms + waist + two Fourier hands). Entry points live in this directory; training & eval launchers live at repo root.

---

## 1. Environment Setup

Two conda environments are involved:

- **AlphaBrain** —  Install per the top-level repo README.
- **robocasa_tabletop** — Install by following the [official robocasa-gr1-tabletop-tasks repository](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) 

---

## 2. Training

### 2.1 Dataset preparation

Download the Isaac GR00T X-Embodiment-Sim dataset from HuggingFace: [`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim). Only the `gr1_unified.*_1000` subfolders are needed for this benchmark — 24 tasks × 1000 episodes each.



### 2.2 `.env` setup

Append to repo-root `.env`:

```bash
# RoboCasa Tabletop
ROBOCASA_TABLETOP_PYTHON=<your_path_to_robocasa_tabletop_conda_env>/bin/python
ROBOCASA_TABLETOP_DATA_ROOT=<your_path_to_PhysicalAI-Robotics-GR00T-X-Embodiment-Sim>
```

### 2.3 Run training

```bash
bash scripts/run_finetune.sh qwen_oft_robocasa_tabletop
```

The validated framework for this benchmark is **QwenOFT** (Qwen3-VL-4B + OFT action head). All settings come from `modes.qwen_oft_robocasa_tabletop` in `configs/finetune_config.yaml`.

**Knobs you will likely change:**

| Setting | Default | Note |
|---|---|---|
| `dataset_mix` | `fourier_gr1_unified_1000` | The mixture to train on. **Edit in two places:** top-level `dataset_mix` *and* `datasets.vla_data.dataset_mix`. |
| `base_vlm` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` | Path to your local Qwen3-VL-4B weights. |
| `run_id` | `qwen_oft_robocasa_tabletop` | Output directory name. |

---

## 3. Evaluation

```bash
bash scripts/run_eval.sh robocasa_tabletop_eval
```

Reads `modes.robocasa_tabletop_eval` in `configs/finetune_config.yaml`. Edit there to change:

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | `./results/training/qwen_oft_robocasa_tabletop/checkpoints/steps_80000` | Path to trained model |
| `n_episodes` | 50 | Rollouts per task |
| `n_action_steps` | 12 | Action chunk size per policy query |
| `port` | 5694 | WebSocket port |
| `gpu_id` | 0 | CUDA device for server |

Results + aggregate stats land under `results/evaluation/robocasa_tabletop/<env_slug>/`.
