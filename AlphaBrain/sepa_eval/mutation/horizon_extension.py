"""
Horizon Extension mutation operator.

Generates longer-horizon task variants by chaining the seed task as the
first subtask in a multi-step episode, multiplying the allowed episode
steps by a configurable factor.
"""
from __future__ import annotations

import copy
from typing import Any

from sepa_eval.mutation.base_operator import MutationError, MutationOperator

_DEFAULT_MAX_STEPS = 500  # fallback when scene_config lacks "max_episode_steps"


class HorizonExtension(MutationOperator):
    """Extend episode horizon by multiplying ``max_episode_steps``.

    For each factor in ``self.extension_factors`` one candidate is
    produced.  The instruction is prefixed with ``"First, "`` to signal
    that the task is the opening subtask of a longer sequence, unless
    it already begins with a sequence-marker word.
    """

    name = "HorizonExtension"

    _SEQUENCE_PREFIXES = ("first,", "step 1", "1.", "begin by", "start by")

    def __init__(self, extension_factors: list[float] = (1.5, 2.0)) -> None:
        self.extension_factors = list(extension_factors)

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
        """Generate one candidate per extension factor.

        Parameters
        ----------
        seed_scene_config:
            Source scene config dict.  ``"max_episode_steps"`` is
            multiplied by the factor; ``"horizon_extension_factor"`` is
            set to the factor.  All other keys are preserved unchanged.
        seed_instruction:
            Task instruction string.  Prefixed with ``"First, "`` if it
            does not already look like part of a sequence.
        parent_task_id:
            ID of the seed trace this mutation originates from.
        benchmark:
            Benchmark name.

        Returns
        -------
        list[CandidateTask]

        Raises
        ------
        MutationError
            If any factor is <= 1.0 (would not actually extend the horizon).
        """
        for f in self.extension_factors:
            if f <= 1.0:
                raise MutationError(
                    f"HorizonExtension: all factors must be > 1.0, got {f}"
                )

        extended_instruction = self._prefix_instruction(seed_instruction)

        candidates = []
        for factor in self.extension_factors:
            new_config = copy.deepcopy(seed_scene_config)
            base_steps = new_config.get("max_episode_steps", _DEFAULT_MAX_STEPS)
            new_config["max_episode_steps"] = int(base_steps * factor)
            new_config["horizon_extension_factor"] = factor

            params = {
                "extension_factor": factor,
                "original_max_steps": base_steps,
                "extended_max_steps": new_config["max_episode_steps"],
            }
            candidates.append(
                self._make_candidate(
                    parent_task_id=parent_task_id,
                    benchmark=benchmark,
                    instruction=extended_instruction,
                    scene_config=new_config,
                    mutation_params=params,
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prefix_instruction(self, instruction: str) -> str:
        """Prepend ``"First, "`` unless the instruction already signals sequencing."""
        lowered = instruction.strip().lower()
        if any(lowered.startswith(prefix) for prefix in self._SEQUENCE_PREFIXES):
            return instruction
        return f"First, {instruction}"
