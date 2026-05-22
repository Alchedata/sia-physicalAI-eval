# Model › Modules

Source path: `AlphaBrain/model/modules/`

A framework-level building-block library that the individual frameworks under `model/framework/` compose on demand:

- **action_model/** — action heads (MLP, flow-matching, DiT, STDP spiking action model, …)
- **vlm/** — vision-language backbones (PaliGemma, Qwen2.5-VL, Qwen3-VL, Llama 3.2, Florence2, CosmosReason2)
- **world_model/** — world-model visual encoders (Cosmos, V-JEPA, WAN)
- **dino_model/** — DINO visual features and image transforms
- **projector/** — cross-modal projectors such as Q-Former

---

## Action Model

### Subpackage entry (diffusion factory)

::: AlphaBrain.model.modules.action_model
    options:
      heading_level: 4
      show_submodules: false

### MLP action head

::: AlphaBrain.model.modules.action_model.mlp_action_header
    options:
      heading_level: 4

### GR00T action head

::: AlphaBrain.model.modules.action_model.groot_action_header
    options:
      heading_level: 4

### Layerwise Flow-Matching action head

::: AlphaBrain.model.modules.action_model.LayerwiseFM_ActionHeader
    options:
      heading_level: 4

### Flow-Matching head

::: AlphaBrain.model.modules.action_model.flow_matching_head
    options:
      heading_level: 4
      show_submodules: true

### π0 Flow-Matching head

::: AlphaBrain.model.modules.action_model.pi0_flow_matching_head
    options:
      heading_level: 4
      show_submodules: true

### DiT modules

::: AlphaBrain.model.modules.action_model.DiT_modules
    options:
      heading_level: 4
      show_submodules: true

### STDP / spiking action model

::: AlphaBrain.model.modules.action_model.stdp
    options:
      heading_level: 4
      show_submodules: true

::: AlphaBrain.model.modules.action_model.spike_action_model_multitimestep
    options:
      heading_level: 4

---

## VLM backbones

### Factory

::: AlphaBrain.model.modules.vlm
    options:
      heading_level: 4
      show_submodules: false

### PaliGemma

::: AlphaBrain.model.modules.vlm.paligemma
    options:
      heading_level: 4

::: AlphaBrain.model.modules.vlm.paligemma_oft
    options:
      heading_level: 4

### Qwen 2.5 / 3 / 3.5 VL

::: AlphaBrain.model.modules.vlm.qwen2_5
    options:
      heading_level: 4

::: AlphaBrain.model.modules.vlm.qwen3
    options:
      heading_level: 4

::: AlphaBrain.model.modules.vlm.qwen3_5
    options:
      heading_level: 4

### Llama 3.2

::: AlphaBrain.model.modules.vlm.llama3_2
    options:
      heading_level: 4

### Florence2

::: AlphaBrain.model.modules.vlm.Florence2
    options:
      heading_level: 4

### CosmosReason2

::: AlphaBrain.model.modules.vlm.CosmosReason2
    options:
      heading_level: 4

---

## World Model

### Factory and base class

::: AlphaBrain.model.modules.world_model
    options:
      heading_level: 4
      show_submodules: false

::: AlphaBrain.model.modules.world_model.base
    options:
      heading_level: 4
      show_submodules: true

### Cosmos encoder

::: AlphaBrain.model.modules.world_model.cosmos
    options:
      heading_level: 4
      show_submodules: true

### V-JEPA encoder

::: AlphaBrain.model.modules.world_model.vjepa
    options:
      heading_level: 4
      show_submodules: true

### WAN encoder

::: AlphaBrain.model.modules.world_model.wan
    options:
      heading_level: 4
      show_submodules: true

---

## DINO

::: AlphaBrain.model.modules.dino_model.dino
    options:
      heading_level: 3

::: AlphaBrain.model.modules.dino_model.dino_transforms
    options:
      heading_level: 3

---

## Projector

::: AlphaBrain.model.modules.projector.qformer
    options:
      heading_level: 3
