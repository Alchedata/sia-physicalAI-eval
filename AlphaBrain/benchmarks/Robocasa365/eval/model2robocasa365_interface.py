from typing import Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from AlphaBrain.model.framework.config_utils import read_mode_config


class PolicyWarper:
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5678,
        image_size: list[int] = [224, 224],
        n_action_steps: int = 16,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.image_size = image_size
        self.n_action_steps = n_action_steps
        self.task_description = None

        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        self.model_config = model_config
        self.action_norm_stats = norm_stats[unnorm_key]["action"]
        self.state_norm_stats = norm_stats[unnorm_key].get("state", None)

    def reset(self, task_description: str | tuple) -> None:
        self.task_description = task_description

    def step(self, observations, **kwargs):
        task_descriptions = self._extract_task_descriptions(observations)
        if task_descriptions and task_descriptions != self.task_description:
            self.reset(task_descriptions)

        images = self._prepare_images(observations)
        states = self._prepare_states(observations)

        examples = []
        for idx in range(len(images)):
            examples.append(
                {
                    "image": images[idx],
                    "lang": task_descriptions[idx],
                    "state": states[idx],
                }
            )

        response = self.client.predict_action(
            {
                "examples": examples,
                "do_sample": False,
            }
        )
        normalized_actions = response["data"]["normalized_actions"]
        # Model output is already in the final action space (training data was pre-normalized to [-1,1]).
        # Skipping unnormalize to avoid double-normalization.
        raw_actions = np.clip(normalized_actions, -1.0, 1.0)
        raw_actions[..., 6:7] = (raw_actions[..., 6:7] > 0.5).astype(np.float32)
        raw_actions[..., 11:12] = (raw_actions[..., 11:12] > 0.5).astype(np.float32)

        action_dict = {
            "action.end_effector_position": raw_actions[:, : self.n_action_steps, 0:3],
            "action.end_effector_rotation": raw_actions[:, : self.n_action_steps, 3:6],
            "action.gripper_close": raw_actions[:, : self.n_action_steps, 6:7],
            "action.base_motion": raw_actions[:, : self.n_action_steps, 7:11],
            "action.control_mode": raw_actions[:, : self.n_action_steps, 11:12],
        }
        return {"actions": action_dict}

    def _prepare_images(self, observations) -> list[list[np.ndarray]]:
        views = [self._latest_step(observations[key]) for key in self.video_keys]
        batch_size = views[0].shape[0]
        images = []
        for batch_idx in range(batch_size):
            images.append([self._resize_image(view[batch_idx]) for view in views])
        return images

    def _prepare_states(self, observations) -> list[np.ndarray]:
        state_parts = []
        for key in self.state_keys:
            value = self._latest_step(observations[key])
            state_parts.append(value.astype(np.float32))

        states = np.concatenate(state_parts, axis=-1)
        if self.state_norm_stats is not None and len(self.state_norm_stats.get("min", [])) == states.shape[-1]:
            state_min = np.asarray(self.state_norm_stats["min"], dtype=np.float32)
            state_max = np.asarray(self.state_norm_stats["max"], dtype=np.float32)
            denom = np.maximum(state_max - state_min, 1e-6)
            states = 2.0 * (states - state_min) / denom - 1.0
            states = np.clip(states, -1.0, 1.0)
        return [state[np.newaxis, :] for state in states]

    def _extract_task_descriptions(self, observations) -> list[str]:
        raw = observations["annotation.human.task_description"]
        if isinstance(raw, np.ndarray):
            values = raw.tolist()
        else:
            values = list(raw)
        descriptions = []
        for item in values:
            if isinstance(item, (list, tuple, np.ndarray)):
                descriptions.append(str(item[0]))
            else:
                descriptions.append(str(item))
        return descriptions

    @staticmethod
    def _latest_step(value: np.ndarray) -> np.ndarray:
        if value.ndim == 5:
            return value[:, -1]
        if value.ndim == 3:
            return value[:, -1]
        if value.ndim == 4 or value.ndim == 2:
            return value
        raise ValueError(f"Unexpected observation shape: {value.shape}")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: dict[str, list[float]]) -> np.ndarray:
        action_max = np.asarray(action_norm_stats["max"], dtype=np.float32)
        action_min = np.asarray(action_norm_stats["min"], dtype=np.float32)
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        return (normalized_actions + 1.0) / 2.0 * (action_max - action_min) + action_min

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                "Your model was trained on more than one dataset, "
                f"please choose one of: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not available, please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
