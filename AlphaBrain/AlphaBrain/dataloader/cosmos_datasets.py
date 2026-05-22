"""
LIBERO-Cosmos-Policy dataset loader for AlphaBrain training.

Dataset format:
  success_only/<suite>/  — demo HDF5 files (data/demo_X/...)
  all_episodes/          — rollout HDF5 files (flat structure)
  t5_embeddings.pkl      — T5 text embeddings dict
  dataset_statistics.json — min/max normalization stats
"""

import io
import json
import os
import pickle
import random
from collections import defaultdict

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers (inlined from cosmos_policy to avoid cross-repo imports)
# ---------------------------------------------------------------------------

def _compute_monte_carlo_returns(num_steps: int, terminal_reward: float, gamma: float) -> np.ndarray:
    T = num_steps
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = terminal_reward
    returns = np.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    if terminal_reward > 0:
        returns = 2 * returns / terminal_reward - 1
    else:
        returns = 2 * returns - 1
    return returns


def _get_action_chunk_with_padding(actions, relative_step_idx, chunk_size, num_steps):
    remaining = num_steps - relative_step_idx
    if remaining >= chunk_size:
        return actions[relative_step_idx: relative_step_idx + chunk_size]
    available = actions[relative_step_idx:]
    padding = np.tile(actions[-1], (chunk_size - remaining, 1))
    return np.concatenate([available, padding], axis=0)


def _rescale(arr, stat_min, stat_max):
    """Min-max normalize to [-1, 1]."""
    return 2.0 * ((arr - stat_min) / (stat_max - stat_min + 1e-8)) - 1.0


def _decode_jpeg(jpeg_bytes) -> np.ndarray:
    """Decode a single JPEG bytes object to (H, W, 3) uint8."""
    return np.array(Image.open(io.BytesIO(bytes(jpeg_bytes)))).astype(np.uint8)


def _decode_jpeg_dataset(jpeg_ds) -> np.ndarray:
    """Decode (T,) object array of JPEG bytes → (T, H, W, 3) uint8."""
    frames = [np.array(Image.open(io.BytesIO(bytes(b)))) for b in jpeg_ds]
    return np.stack(frames, axis=0).astype(np.uint8)


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, 3) uint8 to (size, size, 3)."""
    return np.array(Image.fromarray(img).resize((size, size))).astype(np.uint8)


def _duplicate(arr: np.ndarray, n: int) -> np.ndarray:
    """Stack arr n times along new first axis → (n, *arr.shape)."""
    return np.stack([arr] * n)


def _apply_image_aug(images: torch.Tensor, stronger: bool = False) -> torch.Tensor:
    """
    Apply random resized crop + color jitter to (C, T, H, W) uint8 tensor.
    Same augmentation applied to all frames (consistent spatial transform).

    Args:
        images: (C, T, H, W) uint8 tensor
        stronger: If True, apply stronger augmentations (wider color jitter + random rotation)
    """
    from torchvision.transforms import functional as F
    from torchvision import transforms as T

    _, _, H, W = images.shape
    images = images.permute(1, 0, 2, 3)  # (T, C, H, W)

    # Detect consecutive duplicate groups for efficiency
    unique_groups = []
    num_images = len(images)
    i = 0
    while i < num_images:
        group_start = i
        while i + 1 < num_images and torch.equal(images[i], images[i + 1]):
            i += 1
        unique_groups.append((group_start, i + 1))
        i += 1

    # Sample augmentation params once
    crop_i, crop_j, crop_h, crop_w = T.RandomResizedCrop.get_params(
        torch.zeros(H, W), scale=(0.9, 0.9), ratio=(1.0, 1.0)
    )

    # Random rotation (only for stronger augmentations)
    if stronger:
        angle = torch.FloatTensor(1).uniform_(-5, 5).item()
    else:
        angle = 0.0

    # Color jitter — wider ranges when stronger is True
    if stronger:
        brightness = torch.FloatTensor(1).uniform_(0.7, 1.3).item()
        contrast = torch.FloatTensor(1).uniform_(0.6, 1.4).item()
        saturation = torch.FloatTensor(1).uniform_(0.5, 1.5).item()
    else:
        brightness = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        contrast = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        saturation = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
    hue = torch.FloatTensor(1).uniform_(-0.05, 0.05).item()

    results = []
    for group_start, group_end in unique_groups:
        img = images[group_start]
        img = F.resized_crop(img, crop_i, crop_j, crop_h, crop_w, size=[H, W], antialias=True)
        if angle != 0.0:
            img = F.rotate(img, angle)
        img = F.adjust_brightness(img, brightness)
        img = F.adjust_contrast(img, contrast)
        img = F.adjust_saturation(img, saturation)
        img = F.adjust_hue(img, hue)
        for _ in range(group_end - group_start):
            results.append(img)

    augmented = torch.stack(results).permute(1, 0, 2, 3)  # (C, T, H, W)
    return augmented


def _preprocess_images(images: np.ndarray, final_size: int, use_aug: bool, stronger_aug: bool = False) -> torch.Tensor:
    """
    images: (T, H, W, 3) uint8
    Returns: (C, T, H, W) uint8 tensor
    """
    # Resize each frame
    T_len = images.shape[0]
    resized = np.stack([_resize_image(images[t], final_size) for t in range(T_len)], axis=0)
    # (T, H, W, C) -> (C, T, H, W)
    tensor = torch.from_numpy(np.transpose(resized, (3, 0, 1, 2))).to(torch.uint8)
    if use_aug:
        tensor = _apply_image_aug(tensor, stronger=stronger_aug)
    return tensor


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class CosmosLIBERODataset(Dataset):
    """
    Dataset for LIBERO-Cosmos-Policy format.

    Loads demo data (success_only/) eagerly and rollout data (all_episodes/) lazily.
    Returns samples compatible with the cosmos-policy training pipeline.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 16,
        final_image_size: int = 224,
        num_duplicates_per_image: int = 4,
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        gamma: float = 0.99,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.num_duplicates_per_image = num_duplicates_per_image
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.gamma = gamma

        # Paths
        self.demo_dir = os.path.join(data_dir, "success_only")
        self.rollout_dir = os.path.join(data_dir, "all_episodes")
        # t5_embeddings.pkl and dataset_statistics.json live under success_only/
        t5_path = os.path.join(data_dir, "success_only", "t5_embeddings.pkl")
        stats_path = os.path.join(data_dir, "success_only", "dataset_statistics.json")

        # Load T5 embeddings
        with open(t5_path, "rb") as f:
            self.t5_text_embeddings = pickle.load(f)

        # Load normalization statistics
        with open(stats_path, "r") as f:
            json_stats = json.load(f)
        self.dataset_stats = {k: np.array(v) for k, v in json_stats.items()}

        # Storage
        self.data = {}           # episode_idx -> demo episode dict
        self.num_episodes = 0
        self.num_steps = 0

        self.rollout_episode_metadata = {}   # episode_idx -> metadata dict
        self.rollout_num_episodes = 0

        # Load demos eagerly
        self._load_demos()

        # Build demo step index mapping
        self._build_demo_step_index_mapping()

        # Load rollout metadata lazily
        self._load_rollout_metadata()

        # Build rollout step index mapping
        self._build_rollout_step_index_mapping()

        # Calculate epoch structure
        self._calculate_epoch_structure()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_demos(self):
        """Load all demo episodes from success_only/ into memory."""
        if not os.path.exists(self.demo_dir):
            print(f"[CosmosLIBERODataset] No demo dir found at {self.demo_dir}, skipping demos.")
            return

        hdf5_files = []
        for root, _, files in os.walk(self.demo_dir, followlinks=True):
            for fname in files:
                if fname.lower().endswith((".h5", ".hdf5")):
                    hdf5_files.append(os.path.join(root, fname))
        hdf5_files = sorted(hdf5_files)

        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        print(f"[CosmosLIBERODataset] Loading {len(hdf5_files)} demo files...")
        for fpath in tqdm(hdf5_files, desc="Loading demos"):
            with h5py.File(fpath, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
                for demo_key in demo_keys:
                    obs_group = f[f"data/{demo_key}/obs"]
                    # Decode images (store as JPEG bytes to save RAM, decode on-the-fly)
                    agentview_jpeg = obs_group["agentview_rgb_jpeg"][:]      # (T,) object
                    eye_in_hand_jpeg = obs_group["eye_in_hand_rgb_jpeg"][:]  # (T,) object

                    actions = f[f"data/{demo_key}/actions"][:].astype(np.float32)
                    proprio = f[f"data/{demo_key}/robot_states"][:].astype(np.float32)
                    num_steps = len(actions)

                    # Parse task command from filename
                    command = self._parse_command_from_path(fpath)

                    # Normalize
                    if self.normalize_actions:
                        actions = _rescale(actions, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"])
                    if self.normalize_proprio:
                        proprio = _rescale(proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"])

                    # Monte Carlo returns (success demos always get reward=1)
                    returns = _compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)

                    self.data[self.num_episodes] = dict(
                        agentview_jpeg=agentview_jpeg,
                        eye_in_hand_jpeg=eye_in_hand_jpeg,
                        actions=actions,
                        proprio=proprio,
                        command=command,
                        num_steps=num_steps,
                        returns=returns,
                        success=True,
                    )
                    self.num_episodes += 1
                    self.num_steps += num_steps

        print(f"[CosmosLIBERODataset] Loaded {self.num_episodes} demo episodes, {self.num_steps} steps.")

    def _load_rollout_metadata(self):
        """Lazily load rollout metadata from all_episodes/."""
        if not os.path.exists(self.rollout_dir):
            print(f"[CosmosLIBERODataset] No rollout dir at {self.rollout_dir}, skipping rollouts.")
            return

        hdf5_files = sorted([
            os.path.join(self.rollout_dir, f)
            for f in os.listdir(self.rollout_dir)
            if f.lower().endswith((".h5", ".hdf5"))
        ])

        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:10]

        print(f"[CosmosLIBERODataset] Loading metadata for {len(hdf5_files)} rollout files...")
        for fpath in tqdm(hdf5_files, desc="Loading rollout metadata"):
            with h5py.File(fpath, "r") as f:
                num_steps = len(f["actions"])
                command = str(f.attrs.get("task_description", ""))
                success = bool(f.attrs.get("success", False))
                terminal_reward = 1.0 if success else 0.0
                returns = _compute_monte_carlo_returns(num_steps, terminal_reward=terminal_reward, gamma=self.gamma)

                self.rollout_episode_metadata[self.rollout_num_episodes] = dict(
                    file_path=fpath,
                    command=command,
                    num_steps=num_steps,
                    success=success,
                    returns=returns,
                )
                self.rollout_num_episodes += 1

        print(f"[CosmosLIBERODataset] Loaded metadata for {self.rollout_num_episodes} rollout episodes.")

    def _load_rollout_episode_data(self, metadata: dict) -> dict:
        """Load a single rollout episode from disk."""
        with h5py.File(metadata["file_path"], "r") as f:
            actions = f["actions"][:].astype(np.float32)
            proprio = f["proprio"][:].astype(np.float32)
            agentview_jpeg = f["primary_images_jpeg"][:]
            eye_in_hand_jpeg = f["wrist_images_jpeg"][:]

        if self.normalize_actions:
            actions = _rescale(actions, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"])
        if self.normalize_proprio:
            proprio = _rescale(proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"])

        return dict(
            agentview_jpeg=agentview_jpeg,
            eye_in_hand_jpeg=eye_in_hand_jpeg,
            actions=actions,
            proprio=proprio,
            command=metadata["command"],
            num_steps=metadata["num_steps"],
            success=metadata["success"],
            returns=metadata["returns"],
        )

    # ------------------------------------------------------------------
    # Index mappings
    # ------------------------------------------------------------------

    def _build_demo_step_index_mapping(self):
        self._step_to_episode_map = {}
        self._total_steps = 0
        for ep_idx, ep_data in self.data.items():
            for i in range(ep_data["num_steps"]):
                self._step_to_episode_map[self._total_steps] = (ep_idx, i)
                self._total_steps += 1

    def _build_rollout_step_index_mapping(self):
        self._rollout_success_step_to_episode_map = {}
        self._rollout_failure_step_to_episode_map = {}
        self._rollout_success_total_steps = 0
        self._rollout_failure_total_steps = 0

        for ep_idx, meta in self.rollout_episode_metadata.items():
            num_steps = int(meta["num_steps"])
            is_success = bool(meta["success"])
            for i in range(num_steps):
                if is_success:
                    self._rollout_success_step_to_episode_map[self._rollout_success_total_steps] = (ep_idx, i)
                    self._rollout_success_total_steps += 1
                else:
                    self._rollout_failure_step_to_episode_map[self._rollout_failure_total_steps] = (ep_idx, i)
                    self._rollout_failure_total_steps += 1

        self._rollout_total_steps = self._rollout_success_total_steps + self._rollout_failure_total_steps

    def _calculate_epoch_structure(self):
        """Compute adjusted counts to achieve target sampling probabilities."""
        d = self.num_steps
        s = self._rollout_success_total_steps
        f = self._rollout_failure_total_steps
        r = s + f

        if r == 0:
            self.adjusted_demo_count = d
            self.adjusted_success_rollout_count = 0
            self.adjusted_failure_rollout_count = 0
        else:
            p_demo = self.demonstration_sampling_prob
            p_succ = self.success_rollout_sampling_prob

            # Scale success/failure rollouts
            if p_succ <= 0 or s == 0:
                adj_s, adj_f = 0, f
            elif p_succ >= 1 or f == 0:
                adj_s, adj_f = s, 0
            else:
                s_new = int(f * p_succ / (1 - p_succ))
                f_new = int(s * (1 - p_succ) / p_succ)
                if s < s_new:
                    adj_s, adj_f = s_new, f
                else:
                    adj_s, adj_f = s, f_new

            # Scale demos vs rollouts
            adj_r = adj_s + adj_f
            if p_demo <= 0 or d == 0:
                adj_d = 0
            elif p_demo >= 1:
                adj_d, adj_s, adj_f = d, 0, 0
                adj_r = 0
            else:
                d_new = int(adj_r * p_demo / (1 - p_demo))
                r_new = int(d * (1 - p_demo) / p_demo)
                if d < d_new:
                    adj_d = d_new
                else:
                    adj_d = d
                    adj_s = int(r_new * p_succ)
                    adj_f = int(r_new * (1 - p_succ))

            self.adjusted_demo_count = adj_d
            self.adjusted_success_rollout_count = adj_s
            self.adjusted_failure_rollout_count = adj_f

        self.epoch_length = self.adjusted_demo_count + self.adjusted_success_rollout_count + self.adjusted_failure_rollout_count
        print(
            f"[CosmosLIBERODataset] Epoch length: {self.epoch_length} "
            f"(demos={self.adjusted_demo_count}, succ_rollouts={self.adjusted_success_rollout_count}, "
            f"fail_rollouts={self.adjusted_failure_rollout_count})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_command_from_path(fpath: str) -> str:
        """Parse task command from LIBERO HDF5 filename."""
        basename = os.path.basename(fpath)
        words = basename[:-10].split("_")  # strip _demo.hdf5 suffix (10 chars)
        command = ""
        for w in words:
            if "SCENE" in w:
                command = ""
                continue
            command = command + w + " "
        return command.strip()

    def _determine_sample_type(self, idx: int) -> str:
        if idx < self.adjusted_demo_count:
            return "demo"
        elif idx < self.adjusted_demo_count + self.adjusted_success_rollout_count:
            return "success_rollout"
        else:
            return "failure_rollout"

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        sample_type = self._determine_sample_type(idx)

        rollout_data_mask = 0 if sample_type == "demo" else 1
        rollout_data_success_mask = 1 if sample_type == "success_rollout" else 0

        if sample_type == "demo":
            global_step_idx = idx % self.num_steps
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            episode_data = self.data[episode_idx]
            episode_metadata = None
            global_rollout_idx = -1
        elif sample_type == "success_rollout":
            success_idx = (idx - self.adjusted_demo_count) % self._rollout_success_total_steps
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[success_idx]
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)
            global_rollout_idx = success_idx
        else:
            failure_idx = (idx - self.adjusted_demo_count - self.adjusted_success_rollout_count) % self._rollout_failure_total_steps
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[failure_idx]
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)
            global_rollout_idx = failure_idx

        # World model vs value function split (rollouts only)
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if random.random() < 0.5:
                is_world_model_sample = True
            else:
                is_value_function_sample = True
        else:
            is_world_model_sample = True

        # Future frame index
        future_frame_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)

        # Decode current and future frames
        cur_primary = _decode_jpeg(episode_data["agentview_jpeg"][relative_step_idx])
        cur_wrist = _decode_jpeg(episode_data["eye_in_hand_jpeg"][relative_step_idx])
        fut_primary = _decode_jpeg(episode_data["agentview_jpeg"][future_frame_idx])
        fut_wrist = _decode_jpeg(episode_data["eye_in_hand_jpeg"][future_frame_idx])

        blank = np.zeros_like(cur_primary)  # (H, W, 3) uint8

        # Build frame sequence (9 latent slots):
        # Slot 0: blank (1 copy)
        # Slot 1: blank proprio placeholder (4 copies)
        # Slot 2: wrist image (4 copies)
        # Slot 3: primary image (4 copies)
        # Slot 4: blank action placeholder (4 copies)
        # Slot 5: blank future proprio placeholder (4 copies)
        # Slot 6: future wrist image (4 copies)
        # Slot 7: future primary image (4 copies)
        # Slot 8: blank value placeholder (4 copies)
        # Total: 1 + 8*4 = 33 frames
        n = self.num_duplicates_per_image
        image_list = [
            np.expand_dims(blank, 0),                    # slot 0: 1 frame
            _duplicate(blank, n),                         # slot 1: proprio placeholder
            _duplicate(cur_wrist, n),                     # slot 2: wrist
            _duplicate(cur_primary, n),                   # slot 3: primary
            _duplicate(blank, n),                         # slot 4: action placeholder
            _duplicate(blank, n),                         # slot 5: future proprio placeholder
            _duplicate(fut_wrist, n),                     # slot 6: future wrist
            _duplicate(fut_primary, n),                   # slot 7: future primary
            _duplicate(blank, n),                         # slot 8: value placeholder
        ]

        # Latent indices (0-indexed slot positions)
        current_proprio_latent_idx = 1
        current_wrist_image_latent_idx = 2
        current_image_latent_idx = 3
        action_latent_idx = 4
        future_proprio_latent_idx = 5
        future_wrist_image_latent_idx = 6
        future_image_latent_idx = 7
        value_latent_idx = 8

        # Stack: (33, H, W, 3) uint8
        images_np = np.concatenate(image_list, axis=0)
        # Preprocess: resize + optional aug → (C, T, H, W) uint8
        video = _preprocess_images(images_np, self.final_image_size, self.use_image_aug, self.use_stronger_image_aug)

        # Action chunk
        action_chunk = _get_action_chunk_with_padding(
            episode_data["actions"], relative_step_idx, self.chunk_size, episode_data["num_steps"]
        )

        # Proprio
        proprio = episode_data["proprio"][relative_step_idx]
        future_proprio = episode_data["proprio"][future_frame_idx]

        # Value function return
        returns_arr = episode_data["returns"]
        value_function_return = float(returns_arr[future_frame_idx])

        # T5 embedding
        command = episode_data["command"]
        t5_emb = torch.squeeze(self.t5_text_embeddings[command])  # (512, 1024) bfloat16

        return {
            "video": video,                                                          # (C, 33, 224, 224) uint8
            "actions": torch.from_numpy(action_chunk.astype(np.float32)),           # (16, 7)
            "proprio": torch.from_numpy(proprio.astype(np.float32)),                # (9,)
            "future_proprio": torch.from_numpy(future_proprio.astype(np.float32)),  # (9,)
            "t5_text_embeddings": t5_emb,                                           # (512, 1024) bfloat16
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "__key__": idx,
            "rollout_data_mask": rollout_data_mask,
            "rollout_data_success_mask": rollout_data_success_mask,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 1 if is_value_function_sample else 0,
            "global_rollout_idx": global_rollout_idx,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
            "value_function_return": torch.tensor(value_function_return, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def cosmos_collate_fn(batch):
    """Default collate; tensors are stacked, scalars become tensors."""
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)
