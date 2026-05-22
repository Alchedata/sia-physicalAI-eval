# Training › General Training

Source path: `AlphaBrain/training/`

Generic training entrypoints and shared utilities for VLA models. Continual Learning and Reinforcement Learning have their own pages:

- [Continual Learning](./training_continual_learning.md)
- [Reinforcement Learning](./training_reinforcement_learning.md)

---

## Training entrypoints

### `train_alphabrain.py` — main training entrypoint

::: AlphaBrain.training.train_alphabrain
    options:
      heading_level: 4

### `train_alphabrain_cotrain.py` — co-training

::: AlphaBrain.training.train_alphabrain_cotrain
    options:
      heading_level: 4

### `train_alphabrain_vlm.py` — VLM-only training

::: AlphaBrain.training.train_alphabrain_vlm
    options:
      heading_level: 4

### `train_stdp.py` — STDP spiking-model training

::: AlphaBrain.training.train_stdp
    options:
      heading_level: 4

---

## Trainer utilities

Shared training utilities: structured logging (overwatch), PEFT, finetune configuration, checkpoint tracking, and more.

### Overwatch (unified logging)

::: AlphaBrain.training.trainer_utils.overwatch
    options:
      heading_level: 4

### Finetune configuration

::: AlphaBrain.training.trainer_utils.finetune_config
    options:
      heading_level: 4

### Configuration tracker

::: AlphaBrain.training.trainer_utils.config_tracker
    options:
      heading_level: 4

### Trainer helper functions

::: AlphaBrain.training.trainer_utils.trainer_tools
    options:
      heading_level: 4

### PEFT integration

::: AlphaBrain.training.trainer_utils.peft
    options:
      heading_level: 4
      show_submodules: true
