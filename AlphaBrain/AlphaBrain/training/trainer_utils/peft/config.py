"""LoRA spec parsed from yaml `lora:` section.

Recognized fields (current schema, kept stable for backward-compat):
    enabled              bool
    rank                 int    (default 32)
    alpha                int    (default 16)
    dropout              float  (default 0.05)
    target_modules       str | list[str]  (default "all-linear")
    init_lora_weights    str    (default "gaussian")
    vlm_module           str | None       (default None → auto-detect)
    freeze_extra_modules str | list[str]  (default [])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


def is_lora_enabled(cfg: Any) -> bool:
    """Return True iff `cfg.lora.enabled` is set."""
    if cfg is None:
        return False
    lora_cfg = cfg.get("lora", {}) if hasattr(cfg, "get") else getattr(cfg, "lora", {})
    if isinstance(lora_cfg, DictConfig):
        return bool(lora_cfg.get("enabled", False))
    return bool(getattr(lora_cfg, "enabled", False) if lora_cfg else False)


@dataclass
class LoRASpec:
    """Backbone-agnostic LoRA application spec.

    Resolved from yaml `lora:` block via :meth:`from_omega`.
    """

    rank: int = 32
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Any = "all-linear"          # str | list[str]
    init_lora_weights: str = "gaussian"

    vlm_module: str | None = None               # None = auto-detect
    freeze_extra_modules: list[str] = field(default_factory=list)

    @classmethod
    def from_omega(cls, cfg: Any) -> "LoRASpec":
        """Parse from yaml/OmegaConf `lora:` block.

        Tolerant of:
        - Missing `lora` key (returns defaults; caller should check `is_lora_enabled`)
        - `freeze_extra_modules` as comma-separated string OR list
        - `target_modules` as string ("all-linear") OR list
        """
        lora_cfg = cfg.get("lora", {}) if hasattr(cfg, "get") else getattr(cfg, "lora", {})
        if lora_cfg is None:
            lora_cfg = {}
        get = lora_cfg.get if hasattr(lora_cfg, "get") else (lambda k, d=None: getattr(lora_cfg, k, d))

        freeze_extra = get("freeze_extra_modules", []) or []
        if isinstance(freeze_extra, str):
            freeze_extra = [m.strip() for m in freeze_extra.split(",") if m.strip()]
        elif isinstance(freeze_extra, (list, tuple)):
            freeze_extra = list(freeze_extra)
        else:
            # OmegaConf ListConfig
            try:
                freeze_extra = list(freeze_extra)
            except TypeError:
                freeze_extra = []

        target = get("target_modules", "all-linear")
        # OmegaConf list -> python list
        if not isinstance(target, str):
            try:
                target = list(target)
            except TypeError:
                pass

        return cls(
            rank=int(get("rank", 32)),
            alpha=int(get("alpha", 16)),
            dropout=float(get("dropout", 0.05)),
            target_modules=target,
            init_lora_weights=str(get("init_lora_weights", "gaussian")),
            vlm_module=get("vlm_module", None),
            freeze_extra_modules=freeze_extra,
        )

    def peft_config(self):
        """Build a `peft.LoraConfig` from this spec."""
        from peft import LoraConfig
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            init_lora_weights=self.init_lora_weights,
        )
