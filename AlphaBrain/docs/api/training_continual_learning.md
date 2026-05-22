# Training › Continual Learning

Source path: `AlphaBrain/training/continual_learning/`

The Continual Learning (CL) module: algorithm base class, replay buffer, task sequences, and the training-loop entrypoint.

The top-level `__init__.py` re-exports `CLAlgorithm` and `ReplayBuffer` for backwards compatibility.

---

## Top-level re-exports

::: AlphaBrain.training.continual_learning
    options:
      heading_level: 3
      show_submodules: false

---

## Training entrypoint

::: AlphaBrain.training.continual_learning.train
    options:
      heading_level: 3

---

## Algorithms

### Base class

::: AlphaBrain.training.continual_learning.algorithms.base
    options:
      heading_level: 4

### Replay buffer

::: AlphaBrain.training.continual_learning.algorithms.replay_buffer
    options:
      heading_level: 4

### Subpackage exports

::: AlphaBrain.training.continual_learning.algorithms
    options:
      heading_level: 4
      show_submodules: false

---

## Datasets / task sequences

::: AlphaBrain.training.continual_learning.datasets.task_sequences
    options:
      heading_level: 4

::: AlphaBrain.training.continual_learning.datasets
    options:
      heading_level: 4
      show_submodules: false
