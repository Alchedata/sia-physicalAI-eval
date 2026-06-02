# Environment Setup Guide

This guide walks through the complete local environment setup for SEPA-Eval, including the AlphaBrain policy server, LIBERO simulation, and both RoboCasa benchmarks.

## Overview: Three Python Environments

SEPA-Eval separates concerns across three environments because the simulation clients have conflicting system-level dependencies:

| Environment | Purpose | Configured as |
|---|---|---|
| `alphabrain` | Policy server, SEPA-Eval package, training | conda env |
| `robocasa_tabletop` | RoboCasa Tabletop simulation client | conda env |
| `robocasa365` | RoboCasa 365-task simulation client | conda env |

The `.env` file wires them together by pointing each `*_PYTHON` variable at the right interpreter.

---

## Step 1 — AlphaBrain environment

```bash
conda create -n alphabrain python=3.10 -y
conda activate alphabrain

cd sia-physicalAI-eval/AlphaBrain
pip install -e .
pip install -e ".[dev]"

# SEPA-Eval optional deps
pip install msgpack scikit-learn jinja2 pyyaml openai pyarrow pytest
```

---

## Step 2 — LIBERO simulator

LIBERO must be cloned as source and added to `PYTHONPATH` — it is not on PyPI.

```bash
# Clone into the default location expected by AlphaBrain/scripts/run_eval.sh
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    sia-physicalAI-eval/LIBERO

# Install into the alphabrain env (--no-deps to avoid disturbing the robotics stack)
conda activate alphabrain
pip install -e sia-physicalAI-eval/LIBERO/ --no-deps

# Verify
python -c "import libero; print('LIBERO OK')"
```

Set in `.env`:
```bash
LIBERO_HOME=/absolute/path/to/sia-physicalAI-eval/LIBERO
LIBERO_PYTHON=/path/to/anaconda3/envs/alphabrain/bin/python
```

> **LIBERO_HOME is the simulator source directory, not the data directory.**
> `run_eval.sh` adds `$LIBERO_HOME` to `PYTHONPATH` and reads task configs from
> `$LIBERO_HOME/libero/`.

---

## Step 3 — LIBERO datasets

SEPA-Eval and AlphaBrain training use the LeRobot-format LIBERO datasets hosted on
HuggingFace under the `IPEC-COMMUNITY` organisation. The `data_preparation.sh` script
downloads all four suites and symlinks them into `AlphaBrain/data/datasets/`.

```bash
# Run from AlphaBrain/
conda activate alphabrain
export DEST=/path/to/store/libero_data   # e.g. /Users/fei/data/libero_lerobot

bash benchmarks/LIBERO/data_preparation.sh "$DEST"
```

This downloads:
- `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot`
- `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot`
- `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`
- `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`

Set in `.env`:
```bash
# LeRobot format (training + SEPA-Eval export)
LEROBOT_LIBERO_DATA_DIR=/path/to/store/libero_data/libero

# RLDS/HDF5 format (IPEC / openvla eval pipelines)
# If you have the raw HDF5 demos, point here; otherwise can duplicate the same path.
LIBERO_DATA_ROOT=/path/to/store/libero_data/libero
```

> **Common mistake**: `LEROBOT_LIBERO_DATA_DIR` expects subdirectories named
> `libero_spatial_no_noops_1.0.0_lerobot/`, `libero_goal_no_noops_1.0.0_lerobot/`,
> etc. Raw HDF5 files directly in `libero_goal/` are the *original* LIBERO format
> and will not work here — run `data_preparation.sh` first.

---

## Step 4 — RoboCasa Tabletop environment

### 4a. Create env and install robosuite 1.5.1

The tabletop fork requires **robosuite exactly 1.5.1**. Install it from source so you
can pin the git tag:

```bash
conda create -n robocasa_tabletop python=3.10 -y
conda activate robocasa_tabletop

# Clone robosuite and pin to v1.5.1 — do NOT install from PyPI (latest is 1.5.2 which breaks the fork)
git clone https://github.com/ARISE-Initiative/robosuite.git
git -C robosuite checkout v1.5.1
pip install -e robosuite/

# Set up the private macro file (suppresses warning, required before first import)
python robosuite/robosuite/scripts/setup_macros.py
```

> **Why not `pip install robosuite==1.5.1`?** PyPI installs don't let you pin to a
> git tag after the fact. Checking out `v1.5.1` on a source clone is the only
> reliable way to ensure compatibility when the upstream releases a breaking patch.

### 4b. Install the tabletop robocasa fork

The tabletop benchmark uses the `robocasa-gr1-tabletop-tasks` fork, **not** the
standard kitchen `robocasa` package. Installing both in the same env will break things.

```bash
conda activate robocasa_tabletop

# If the kitchen robocasa is somehow already installed, remove it first:
pip uninstall robocasa -y 2>/dev/null || true

git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
pip install -e robocasa-gr1-tabletop-tasks/

# The tabletop fork has no setup_macros.py — create macros_private.py manually:
cp robocasa-gr1-tabletop-tasks/robocasa/macros.py \
   robocasa-gr1-tabletop-tasks/robocasa/macros_private.py
```

### 4c. Download tabletop assets

```bash
conda activate robocasa_tabletop
cd robocasa-gr1-tabletop-tasks

python robocasa/scripts/download_tabletop_assets.py -y
python robocasa/scripts/download_groot_assets.py -y
```

### 4d. Verify

```bash
conda activate robocasa_tabletop
python -c "
import robosuite, robocasa
from robocasa.utils.gym_utils import GrootRoboCasaEnv
print('robosuite:', robosuite.__version__)   # must be 1.5.1
cfg = robosuite.load_part_controller_config(default_controller='OSC_POSE')
print('OSC_POSE controller config OK')
print('GrootRoboCasaEnv:', GrootRoboCasaEnv)
"
```

### 4e. Download the training/eval dataset

The tabletop eval reads from the `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` HuggingFace
dataset. Only the `gr1_unified.*_1000` task folders are needed:

```bash
conda activate alphabrain
cd AlphaBrain
python benchmarks/Robocasa_tabletop/train/download_gr00t_ft_data.py
```

Set in `.env`:
```bash
ROBOCASA_TABLETOP_PYTHON=/path/to/anaconda3/envs/robocasa_tabletop/bin/python
ROBOCASA_TABLETOP_DATA_ROOT=/path/to/data/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
```

---

## Step 5 — RoboCasa 365 environment

```bash
conda create -n robocasa365 python=3.10 -y
conda activate robocasa365

git clone https://github.com/robocasa/robocasa.git
pip install -e robocasa/
pip install mujoco gymnasium ffmpeg-python
```

Download the dataset following the
[robocasa.ai dataset docs](https://robocasa.ai/docs/build/html/datasets/using_datasets.html).
The expected directory layout is:

```
$ROBOCASA365_DATA_ROOT/
├── pretrain/
│   └── atomic/<task>/<date>/lerobot/
└── target/
    ├── atomic/<task>/<date>/lerobot/
    └── composite/<task>/<date>/lerobot/
```

Set in `.env`:
```bash
ROBOCASA365_PYTHON=/path/to/anaconda3/envs/robocasa365/bin/python
ROBOCASA365_DATA_ROOT=/path/to/robocasa365_dataset
```

---

## Step 6 — Configure `.env`

Copy the template and fill in your paths:

```bash
cp AlphaBrain/.env.example AlphaBrain/.env
# or at project root:
cp .env.example .env
```

### Full variable reference

```bash
# ── Pretrained model weights ──────────────────────────────────────────────
# Parent directory that CONTAINS model subdirectories.
# e.g.: /Users/you/models/Qwen2.5-VL-3B-Instruct/  lives inside this dir.
# Do NOT point at the model subdirectory itself.
PRETRAINED_MODELS_DIR=/path/to/models/

# ── LIBERO ────────────────────────────────────────────────────────────────
# LeRobot-format data (from data_preparation.sh). Contains subdirs named
# libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot/
LEROBOT_LIBERO_DATA_DIR=/path/to/libero_lerobot_data/libero

# RLDS/HDF5-format data root (used by IPEC / openvla eval pipelines).
# Usually the same parent as above if you used data_preparation.sh.
LIBERO_DATA_ROOT=/path/to/libero_lerobot_data/libero

# LIBERO simulator SOURCE directory (git clone root, not data).
# run_eval.sh adds this to PYTHONPATH and reads configs from $LIBERO_HOME/libero/
LIBERO_HOME=/path/to/sia-physicalAI-eval/LIBERO

# Python interpreter in the alphabrain conda env
LIBERO_PYTHON=/path/to/anaconda3/envs/alphabrain/bin/python

# ── RoboCasa Tabletop ─────────────────────────────────────────────────────
ROBOCASA_TABLETOP_PYTHON=/path/to/anaconda3/envs/robocasa_tabletop/bin/python
ROBOCASA_TABLETOP_DATA_ROOT=/path/to/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

# ── RoboCasa 365 ──────────────────────────────────────────────────────────
ROBOCASA365_PYTHON=/path/to/anaconda3/envs/robocasa365/bin/python
ROBOCASA365_DATA_ROOT=/path/to/robocasa365_dataset

# ── Optional ──────────────────────────────────────────────────────────────
# WANDB_API_KEY=your_key
# OPENAI_API_KEY=your_key   # required for InstructionParaphrase mutation operator
```

---

## Step 7 — Verify the full stack

```bash
# AlphaBrain + SEPA-Eval
conda activate alphabrain
cd sia-physicalAI-eval/AlphaBrain
python -m sepa_eval status
pytest sepa_eval/tests -q

# LIBERO simulator
python -c "import libero; print('LIBERO OK:', libero.__file__)"

# RoboCasa Tabletop
conda activate robocasa_tabletop
python -c "
import robosuite, robocasa
from robocasa.utils.gym_utils import GrootRoboCasaEnv
assert robosuite.__version__ == '1.5.1', f'need 1.5.1, got {robosuite.__version__}'
print('All tabletop checks passed')
"

# RoboCasa 365
conda activate robocasa365
python -c "import robocasa; print('robocasa365 OK')"
```

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: robosuite version must be 1.5.{0,1}` | PyPI robosuite 1.5.2 installed | Clone from source and `git checkout v1.5.1` |
| `ImportError: get_elements from mjcf_utils` | Kitchen robocasa (1.0.1) installed alongside 1.5.x robosuite | `pip uninstall robocasa -y` then install the `robocasa-gr1-tabletop-tasks` fork |
| `AttributeError: module 'robosuite' has no attribute 'load_controller_config'` | API renamed in 1.5.x | Use `load_part_controller_config` (for `OSC_POSE`) or `load_composite_controller_config` (for `BASIC`, `WHOLE_BODY_IK`, etc.) |
| `AttributeError: module 'robocasa' has no attribute 'models'` | Wrong robocasa package or import-time crash masking the real error | Fix the robosuite version mismatch first; the `models` subpackage is then importable |
| `[robocasa WARNING] No private macro file found` | `macros_private.py` missing (tabletop fork has no `setup_macros.py`) | `cp robocasa/macros.py robocasa/macros_private.py` |
| Training fails to find data | `LEROBOT_LIBERO_DATA_DIR` points at raw HDF5 files | Run `data_preparation.sh` to download the LeRobot-format datasets from HuggingFace |
| `LIBERO_HOME` errors / missing task configs | `LIBERO_HOME` set to data directory instead of simulator source | Point to the LIBERO git clone root (the one containing `setup.py` and `libero/`) |
| `PRETRAINED_MODELS_DIR` not found | Path points to a model subdirectory | Set to the **parent** folder that contains `Qwen2.5-VL-3B-Instruct/`, `Llama-3.2-11B-Vision-Instruct/`, etc. |
