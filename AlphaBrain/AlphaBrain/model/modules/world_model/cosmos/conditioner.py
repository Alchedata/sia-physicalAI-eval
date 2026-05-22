# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from cosmos_policy.conditioner
# Simplified: no megatron/imaginaire/hydra dependencies.
# Uses standard Python dataclasses (frozen=False for mutability).

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Optional

import torch


class DataType(Enum):
    VIDEO = "video"
    IMAGE = "image"


@dataclass(frozen=False)
class BaseCondition:
    """
    Mutable base condition dataclass.

    _is_broadcasted: internal flag for distributed broadcast tracking.
    """

    _is_broadcasted: bool = False

    def to_dict(self, skip_underscore: bool = True) -> Dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not (f.name.startswith("_") and skip_underscore)
        }

    @property
    def is_broadcasted(self) -> bool:
        return self._is_broadcasted


@dataclass(frozen=False)
class Text2WorldCondition(BaseCondition):
    """
    Condition for text-to-world generation.

    crossattn_emb: Text embedding for cross-attention, shape (B, seq_len, dim).
    padding_mask: Attention mask for text tokens, shape (B, seq_len).
    fps: Frames-per-second conditioning, shape (B,).
    data_type: VIDEO or IMAGE.
    """

    crossattn_emb: Optional[torch.Tensor] = None
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None

    def edit_data_type(self, data_type: DataType) -> Text2WorldCondition:
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["data_type"] = data_type
        return type(self)(**kwargs)

    @property
    def is_video(self) -> bool:
        return self.data_type == DataType.VIDEO


@dataclass(frozen=False)
class Video2WorldCondition(Text2WorldCondition):
    """
    Condition for video-to-world generation.
    Extends Text2WorldCondition with video-specific fields if needed.
    """
    pass


@dataclass(frozen=False)
class MutableCondition(BaseCondition):
    """
    Generic mutable condition for cases where fields are set dynamically.
    """

    data: Optional[Dict[str, Any]] = None

    def set(self, key: str, value: Any) -> None:
        if self.data is None:
            self.data = {}
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if self.data is None:
            return default
        return self.data.get(key, default)


class VideoConditioner:
    """
    Lightweight video conditioner that wraps embedder modules and produces
    a Text2WorldCondition.

    Usage:
        conditioner = VideoConditioner(embedders={"t5": t5_embedder})
        condition = conditioner(batch)
    """

    def __init__(self, embedders: Dict[str, Any]):
        self.embedders = embedders

    def __call__(
        self,
        batch: Dict[str, Any],
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Text2WorldCondition:
        output: Dict[str, Any] = {}
        for name, embedder in self.embedders.items():
            dropout_rate = None
            if override_dropout_rate is not None:
                dropout_rate = override_dropout_rate.get(name)
            result = embedder(batch, dropout_rate=dropout_rate)
            if isinstance(result, dict):
                output.update(result)
            else:
                output[name] = result
        return Text2WorldCondition(**{k: v for k, v in output.items() if k in Text2WorldCondition.__dataclass_fields__})

    def get_condition_uncondition(
        self,
        data_batch: Dict[str, Any],
    ):
        """
        Returns (condition, un_condition).

        If all embedder dropout rates are 0, un_condition is None
        (always-conditional generation, no CFG).
        """
        cond_dropout_rates = {name: 0.0 for name in self.embedders}
        dropout_rates = {
            name: 1.0 if getattr(emb, "dropout_rate", 0.0) > 1e-4 else 0.0
            for name, emb in self.embedders.items()
        }

        condition = self(data_batch, override_dropout_rate=cond_dropout_rates)

        if cond_dropout_rates == dropout_rates:
            un_condition = None
        else:
            un_condition = self(data_batch, override_dropout_rate=dropout_rates)

        return condition, un_condition
