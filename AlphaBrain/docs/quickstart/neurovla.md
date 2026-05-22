# NeuroVLA

Brain-inspired VLA: pretrain a Spiking Neural Network action head with backprop, fine-tune with hybrid R-STDP, evaluate on LIBERO.

---

## Prerequisites

```bash
conda activate alphabrain
# Same .env fields as Baseline VLA
```

Hardware: 4 × A800 80 GB default. Smaller setups work with `--gpus N`.

---

## Pipeline

### 1. Pretrain

```bash
bash scripts/run_brain_inspired_scripts/run_neurovla_pretrain.sh \
    --steps 50000 --run-id my_pretrain
```

### 2. R-STDP Fine-tune

```bash
bash scripts/run_brain_inspired_scripts/run_stdp_finetune.sh \
    --pretrained results/training/my_pretrain/checkpoints/steps_50000 \
    --steps 10000 --run-id my_stdp_ft
```

### 3. Evaluate

```bash
# libero_goal, 10 trials/task
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained results/training/my_stdp_ft/checkpoints/steps_10000

# all 4 suites, 50 trials, with online STDP test-time adaptation
bash scripts/run_brain_inspired_scripts/run_eval_libero.sh \
    --pretrained results/training/my_stdp_ft/checkpoints/steps_10000 \
    --suite all --trials 50 --online-stdp
```

Results: `results/evaluation/brain_inspired_eval_<timestamp>/<suite>/`.

---

Full CLI reference and the team's open-research notes: [`scripts/run_brain_inspired_scripts/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_brain_inspired_scripts/README.md).
