"""
PaliGemmaOFT Data Pipeline

Adapts VLAE's existing LeRobot data loading to PaliGemmaOFT format.
Reuses the existing lerobot_datasets.py infrastructure, adding Pi0-specific transforms.

Pi0 expects:
  - images: dict of {camera_name: [B, H, W, 3] uint8 tensors}
  - image_masks: dict of {camera_name: [B] bool tensors} 
  - state: [B, state_dim] float32
  - tokenized_prompt: [B, max_token_len] int32
  - tokenized_prompt_mask: [B, max_token_len] bool
  - actions: [B, action_horizon, action_dim] float32
"""

import logging
from typing import Optional, Dict, List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Pi0DataConfig:
    """Configuration for Pi0-specific data processing."""
    image_resolution: tuple = (224, 224)  # H, W for SigLIP
    max_token_len: int = 200  # max language token length (pi05 default)
    action_horizon: int = 50
    action_dim: int = 7
    camera_names: tuple = ("image_0",)  # maps to observation camera keys
    include_state: bool = True
    state_dim: int = 7


class Pi0DataTransform:
    """
    Transform VLAE LeRobot data samples into PaliGemmaOFT format.
    
    Input (from LeRobot dataloader):
        dict with keys: image (List[PIL.Image]), lang (str), action (np.ndarray), state (np.ndarray)
        
    Output (for PaliGemmaOFT.forward()):
        dict with same keys, but images resized and ready for Pi0 processing
    """
    
    def __init__(self, config: Pi0DataConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer  # PaliGemma/Gemma tokenizer
        
    def __call__(self, sample: dict) -> dict:
        """Transform a single sample."""
        result = {}
        
        # ── Images ──
        images = sample.get("image", [])
        if isinstance(images, (list, tuple)):
            processed_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = img.resize(
                        (self.config.image_resolution[1], self.config.image_resolution[0]),
                        Image.BILINEAR
                    )
                    img = np.array(img)
                elif isinstance(img, np.ndarray):
                    # Resize numpy image
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize(
                        (self.config.image_resolution[1], self.config.image_resolution[0]),
                        Image.BILINEAR
                    )
                    img = np.array(pil_img)
                elif isinstance(img, torch.Tensor):
                    img = img.numpy()
                processed_images.append(img)
            result["image"] = processed_images
        else:
            result["image"] = images
        
        # ── Language ──
        result["lang"] = sample.get("lang", "")
        
        # ── Actions ──
        action = sample.get("action", None)
        if action is not None:
            if isinstance(action, np.ndarray):
                action = action.astype(np.float32)
            elif isinstance(action, torch.Tensor):
                action = action.float().numpy()
            
            # Ensure shape is [action_horizon, action_dim]
            if action.ndim == 1:
                action = action.reshape(1, -1)
            
            # Pad/truncate to action_horizon
            if action.shape[0] < self.config.action_horizon:
                pad = np.zeros(
                    (self.config.action_horizon - action.shape[0], action.shape[1]),
                    dtype=np.float32
                )
                action = np.concatenate([action, pad], axis=0)
            elif action.shape[0] > self.config.action_horizon:
                action = action[:self.config.action_horizon]
                
            result["action"] = action
        
        # ── State ──
        if self.config.include_state and "state" in sample:
            state = sample["state"]
            if isinstance(state, np.ndarray):
                state = state.astype(np.float32)
            elif isinstance(state, torch.Tensor):
                state = state.float().numpy()
            result["state"] = state
        
        return result


def get_pi0_dataset(data_cfg, mode="train", **kwargs):
    """
    Get dataset for PaliGemmaOFT training.
    
    Reuses VLAE's existing LeRobot data loading, wrapping it with Pi0-specific transforms.
    
    Args:
        data_cfg: dataset config (same as used by other VLAE frameworks)
        mode: "train" or "eval"
        
    Returns:
        dataset wrapped with Pi0DataTransform
    """
    from AlphaBrain.dataloader.lerobot_datasets import get_vla_dataset
    
    # Override action_indices in LIBERO data config if action_horizon > default
    action_horizon = getattr(data_cfg, 'action_horizon', 50)
    from AlphaBrain.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
    libero_cfg = ROBOT_TYPE_CONFIG_MAP.get("libero_franka", None)
    if libero_cfg is not None:
        if action_horizon > 8:  # default LIBERO action_indices is range(8)
            libero_cfg.action_indices = list(range(action_horizon))
            logger.info(f"[pi0_data] Overriding action_indices to range({action_horizon})")
        
        # Skip data-level q99 normalization — Pi0 does MEAN_STD normalization in model
        skip_action_norm = getattr(data_cfg, 'skip_action_norm', True)
        if skip_action_norm:
            from AlphaBrain.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor
            from AlphaBrain.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
            # Override transform to only do tensor conversion, no q99 normalization
            original_transform = libero_cfg.transform
            def raw_transform(self=libero_cfg):
                transforms = [StateActionToTensor(apply_to=self.action_keys)]
                return ComposedModalityTransform(transforms=transforms)
            libero_cfg.transform = raw_transform
            logger.info("[pi0_data] Disabled data-level action normalization (model handles MEAN_STD)")
    
    # Get the base LeRobot dataset
    base_dataset = get_vla_dataset(data_cfg, mode=mode, **kwargs)
    
    # Create Pi0 transform config from data_cfg
    pi0_config = Pi0DataConfig(
        action_horizon=getattr(data_cfg, 'action_horizon', 50),
        action_dim=getattr(data_cfg, 'action_dim', 7),
        include_state=getattr(data_cfg, 'include_state', True),
        state_dim=getattr(data_cfg, 'state_dim', 7),
    )
    
    transform = Pi0DataTransform(config=pi0_config)
    
    # Wrap the dataset with Pi0 transforms
    return Pi0DatasetWrapper(base_dataset, transform)


class Pi0DatasetWrapper:
    """Wraps a LeRobot dataset with Pi0-specific transforms."""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        return self.transform(sample)
    
    def __iter__(self):
        for sample in self.base_dataset:
            yield self.transform(sample)


# ── LIBERO-specific config ──

LIBERO_PI0_CONFIG = Pi0DataConfig(
    image_resolution=(224, 224),
    max_token_len=200,
    action_horizon=10,     # LIBERO uses shorter horizon
    action_dim=7,          # 6 DOF + gripper
    camera_names=("image_0",),
    include_state=True,
    state_dim=7,
)
