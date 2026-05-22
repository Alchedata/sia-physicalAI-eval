"""
Spike Monitor for recording spike timing from LIF neuron layers.

Attaches forward hooks to snntorch.Leaky modules to capture spike outputs
and membrane potentials at each timestep, enabling STDP weight updates.
Also hooks preceding Linear layers to capture pre-synaptic inputs for
correct STDP weight update dimensions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import snntorch as snn


class SpikeRecord:
    """Container for spike data recorded from a single LIF layer."""

    def __init__(self, layer_name: str, device: torch.device):
        self.layer_name = layer_name
        self.device = device
        # Recorded per-timestep: list of [B, D] tensors
        self.spikes: List[torch.Tensor] = []
        self.membrane: List[torch.Tensor] = []

    def record(self, spike: torch.Tensor, mem: torch.Tensor):
        """Record one timestep of spike data."""
        self.spikes.append(spike.detach())
        self.membrane.append(mem.detach())

    def get_spike_tensor(self) -> torch.Tensor:
        """Return stacked spikes as [T, B, D]."""
        if not self.spikes:
            raise RuntimeError(f"No spikes recorded for layer {self.layer_name}")
        return torch.stack(self.spikes, dim=0)

    def get_membrane_tensor(self) -> torch.Tensor:
        """Return stacked membrane potentials as [T, B, D]."""
        if not self.membrane:
            raise RuntimeError(f"No membrane data recorded for layer {self.layer_name}")
        return torch.stack(self.membrane, dim=0)

    def reset(self):
        """Clear all recorded data for next forward pass."""
        self.spikes.clear()
        self.membrane.clear()

    @property
    def num_timesteps(self) -> int:
        return len(self.spikes)


class LinearInputRecord:
    """Container for inputs to a Linear layer (pre-synaptic activity for STDP)."""

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.inputs: List[torch.Tensor] = []

    def record(self, input_tensor: torch.Tensor):
        self.inputs.append(input_tensor.detach())

    def get_input_tensor(self) -> Optional[torch.Tensor]:
        """Return stacked inputs as [T, B, D_in], or None."""
        if not self.inputs:
            return None
        return torch.stack(self.inputs, dim=0)

    def reset(self):
        self.inputs.clear()

    @property
    def num_timesteps(self) -> int:
        return len(self.inputs)


class SpikeMonitor:
    """
    Monitor that hooks into LIF neurons and their preceding Linear layers
    to record spike timing and pre-synaptic inputs for STDP computation.

    For each (Linear → LIF) pair:
      - pre-synaptic: input to the Linear layer [B, D_in]
      - post-synaptic: spike output of the LIF layer [B, D_out]
      - Weight shape of Linear: [D_out, D_in]
      - STDP update shape: [D_out, D_in] (matches weight)

    Usage:
        monitor = SpikeMonitor(snn_model)
        monitor.enable()
        output = snn_model(input)
        records = monitor.get_records()
        monitor.reset()
        monitor.disable()
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._records: Dict[str, SpikeRecord] = {}
        self._linear_records: Dict[str, LinearInputRecord] = {}
        self._enabled = False
        # Map from LIF module id to (layer_name, preceding_linear, linear_name)
        self._lif_map: Dict[int, Tuple[str, Optional[nn.Linear], str]] = {}
        # Map from Linear module id to linear_name (for hooking)
        self._linear_map: Dict[int, str] = {}
        self._discover_lif_layers()

    def _discover_lif_layers(self):
        """Find all LIF neuron layers and their preceding linear layers."""
        prev_linear: Optional[nn.Linear] = None
        prev_name: str = ""

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prev_linear = module
                prev_name = name
            elif isinstance(module, snn.Leaky):
                self._lif_map[id(module)] = (name, prev_linear, prev_name)
                self._records[name] = SpikeRecord(name, self.device)
                if prev_linear is not None:
                    self._linear_map[id(prev_linear)] = name  # keyed by LIF name
                    self._linear_records[name] = LinearInputRecord(prev_name)
                prev_linear = None

    @property
    def layer_names(self) -> List[str]:
        """Return names of all monitored LIF layers."""
        return list(self._records.keys())

    @property
    def lif_linear_pairs(self) -> List[Tuple[str, Optional[nn.Linear], snn.Leaky]]:
        """Return (layer_name, preceding_linear, lif_module) tuples."""
        pairs = []
        for name, module in self.model.named_modules():
            if isinstance(module, snn.Leaky) and id(module) in self._lif_map:
                layer_name, linear, _ = self._lif_map[id(module)]
                pairs.append((layer_name, linear, module))
        return pairs

    def enable(self):
        """Attach forward hooks to all LIF layers and their preceding Linear layers."""
        if self._enabled:
            return
        self._enabled = True

        for name, module in self.model.named_modules():
            # Hook LIF layers for post-synaptic spikes
            if isinstance(module, snn.Leaky) and id(module) in self._lif_map:
                layer_name = self._lif_map[id(module)][0]
                hook = module.register_forward_hook(self._make_lif_hook(layer_name))
                self._hooks.append(hook)

            # Hook Linear layers for pre-synaptic inputs
            if isinstance(module, nn.Linear) and id(module) in self._linear_map:
                lif_name = self._linear_map[id(module)]
                hook = module.register_forward_hook(self._make_linear_hook(lif_name))
                self._hooks.append(hook)

    def disable(self):
        """Remove all forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._enabled = False

    def _make_lif_hook(self, layer_name: str):
        """Create a forward hook for a LIF layer (records post-synaptic spikes)."""
        record = self._records[layer_name]

        def hook_fn(module: snn.Leaky, input_tuple, output):
            if isinstance(output, tuple):
                spike = output[0]
            else:
                spike = output
            mem = module.mem if hasattr(module, "mem") and module.mem is not None else torch.zeros_like(spike)
            record.record(spike, mem)

        return hook_fn

    def _make_linear_hook(self, lif_name: str):
        """Create a forward hook for a Linear layer (records pre-synaptic inputs)."""
        linear_record = self._linear_records[lif_name]

        def hook_fn(module: nn.Linear, input_tuple, output):
            # Record the input to the Linear layer = pre-synaptic activity
            if input_tuple:
                linear_record.record(input_tuple[0])

        return hook_fn

    def get_records(self) -> Dict[str, SpikeRecord]:
        """Return all spike records keyed by layer name."""
        return self._records

    def get_linear_records(self) -> Dict[str, LinearInputRecord]:
        """Return all linear input records keyed by LIF layer name."""
        return self._linear_records

    def get_record(self, layer_name: str) -> SpikeRecord:
        if layer_name not in self._records:
            raise KeyError(f"No record for layer '{layer_name}'. Available: {list(self._records.keys())}")
        return self._records[layer_name]

    def get_linear_record(self, layer_name: str) -> Optional[LinearInputRecord]:
        return self._linear_records.get(layer_name)

    def reset(self):
        """Clear all recorded spike and linear input data."""
        for record in self._records.values():
            record.reset()
        for record in self._linear_records.values():
            record.reset()

    def get_spike_rates(self) -> Dict[str, float]:
        """Compute average spike rate for each layer."""
        rates = {}
        for name, record in self._records.items():
            if record.num_timesteps > 0:
                spikes = record.get_spike_tensor()
                rates[name] = spikes.float().mean().item()
            else:
                rates[name] = 0.0
        return rates

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get detailed statistics for each layer."""
        stats = {}
        for name, record in self._records.items():
            if record.num_timesteps == 0:
                stats[name] = {"spike_rate": 0.0, "mean_mem": 0.0, "timesteps": 0}
                continue
            spikes = record.get_spike_tensor()
            membrane = record.get_membrane_tensor()
            stats[name] = {
                "spike_rate": spikes.float().mean().item(),
                "mean_mem": membrane.float().mean().item(),
                "std_mem": membrane.float().std().item(),
                "timesteps": record.num_timesteps,
            }
        return stats

    def __del__(self):
        self.disable()
