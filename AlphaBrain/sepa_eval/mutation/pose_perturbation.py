"""
Pose Perturbation mutation operator.

Generates scene variants by applying random positional and rotational
perturbations to object poses found in the scene config.
"""
from __future__ import annotations

import copy
import random
from typing import Any

from sepa_eval.mutation.base_operator import MutationError, MutationOperator


class PosePerturbation(MutationOperator):
    """Perturb object positions and rotations by configurable delta steps.

    For each combination of ``delta_pos`` × ``delta_rot_deg``, one candidate
    is produced with all pose-related keys perturbed by a random amount
    drawn from ``Uniform(-delta, +delta)``.

    Scene config keys
    -----------------
    - Keys ending in ``"_pos"``  → each coordinate offset by Uniform(-delta_pos, +delta_pos)
    - Keys ending in ``"_rot"`` or ``"_quat"`` → first component offset by
      Uniform(-delta_rot_rad, +delta_rot_rad) (simplified representation)
    """

    name = "PosePerturbation"

    def __init__(
        self,
        delta_pos: list[float] = (0.02, 0.05, 0.10),
        delta_rot_deg: list[float] = (5.0, 15.0, 30.0),
    ) -> None:
        self.delta_pos = list(delta_pos)
        self.delta_rot_deg = list(delta_rot_deg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        seed_scene_config: dict,
        seed_instruction: str,
        parent_task_id: str = "unknown",
        benchmark: str = "unknown",
        **kwargs: Any,
    ) -> list:
        """Generate one candidate per (delta_pos, delta_rot_deg) combination.

        Parameters
        ----------
        seed_scene_config:
            Source scene config dict.  Any key ending in ``"_pos"`` must map
            to a list/tuple of numbers; any key ending in ``"_rot"`` or
            ``"_quat"`` must also map to a list/tuple of numbers.
        seed_instruction:
            Task instruction string (unchanged across variants).
        parent_task_id:
            ID of the seed trace this mutation originates from.
        benchmark:
            Benchmark name (e.g. ``"libero_spatial"``).

        Returns
        -------
        list[CandidateTask]
        """
        candidates = []

        for dp in self.delta_pos:
            for dr in self.delta_rot_deg:
                try:
                    perturbed = self._perturb(seed_scene_config, dp, dr)
                except (TypeError, ValueError, KeyError) as exc:
                    raise MutationError(
                        f"PosePerturbation failed (delta_pos={dp}, delta_rot_deg={dr}): {exc}"
                    ) from exc

                params = {"delta_pos": dp, "delta_rot_deg": dr}
                candidates.append(
                    self._make_candidate(
                        parent_task_id=parent_task_id,
                        benchmark=benchmark,
                        instruction=seed_instruction,
                        scene_config=perturbed,
                        mutation_params=params,
                    )
                )

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _perturb(self, config: dict, delta_pos: float, delta_rot_deg: float) -> dict:
        """Return a deep-copied config with poses perturbed."""
        result = copy.deepcopy(config)
        import math
        delta_rot_rad = math.radians(delta_rot_deg)

        for key, value in result.items():
            if key.endswith("_pos") or key.endswith("_pose"):
                result[key] = self._perturb_vector(value, delta_pos)
            elif key.endswith("_rot") or key.endswith("_quat"):
                result[key] = self._perturb_rotation(value, delta_rot_rad)

        return result

    @staticmethod
    def _perturb_vector(value: list, delta: float) -> list:
        """Add uniform noise in [-delta, +delta] to each element."""
        if not isinstance(value, (list, tuple)):
            return value
        return [v + random.uniform(-delta, delta) for v in value]

    @staticmethod
    def _perturb_rotation(value: list, delta_rad: float) -> list:
        """Add uniform noise to the first component of a rotation vector/quaternion.

        This is intentionally simplified: a production implementation would
        compose rotations properly, but for evaluation-diversity purposes a
        small scalar perturbation to the first component is sufficient to
        produce meaningfully distinct scenes.
        """
        if not isinstance(value, (list, tuple)):
            return value
        result = list(value)
        result[0] = result[0] + random.uniform(-delta_rad, delta_rad)
        return result
