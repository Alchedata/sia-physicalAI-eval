# Copyright 2025 VLA-Engine. All rights reserved.
# Llama-OFT Framework — Llama 3.2 Vision + action token regression

"""
Llama-OFT Framework

Uses Llama 3.2 Vision as backbone with action special token for continuous action prediction.
Mirrors QwenOFT but swaps Qwen for Llama 3.2 Vision.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import logging
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.vlm import get_vlm_model
from AlphaBrain.model.modules.action_model.mlp_action_header import get_action_model
from AlphaBrain.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("LlamaOFT")
class Llama_OFT(BaseFramework):
    """
    Llama 3.2 Vision + action token OFT framework.
    Predicts continuous actions via L1 regression on action token hidden states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.llama_vl_interface = get_vlm_model(config=self.config)

        # Align action hidden dim with LLM hidden size
        config.framework.action_model.action_hidden_dim = self.llama_vl_interface.model.config.text_config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # Enable gradient checkpointing for memory efficiency (11B model)
        if hasattr(self.llama_vl_interface.model, 'gradient_checkpointing_enable'):
            self.llama_vl_interface.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for Llama model")

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_token = "🔍"
        self.action_token_id = self.llama_vl_interface.processor.tokenizer(
            "🔍", add_special_tokens=False
        )["input_ids"][0]

        self.l1_loss = nn.L1Loss()

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        # Add action tokens to instruction
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Build Llama inputs
        llama_inputs = self.llama_vl_interface.build_llama_inputs(
            images=batch_images, instructions=instructions
        )

        

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.llama_vl_interface(
                **llama_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = llama_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List = None,
        instructions: List[str] = None,
        examples: List[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        if examples is not None:
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            instructions = [example["lang"] for example in examples]
        else:
            batch_images = [to_pil_preserve(imgs) for imgs in batch_images]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        llama_inputs = self.llama_vl_interface.build_llama_inputs(
            images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.llama_vl_interface(
                **llama_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = llama_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        """Extract action token embeddings — same logic as QwenOFT."""
        if action_token_id is None:
            raise ValueError("action_token_id cannot be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Action tokens insufficient for chunk_len {self.chunk_len}: samples {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        action_queries = last_hidden.gather(dim=1, index=expanded_index)
        return action_queries
