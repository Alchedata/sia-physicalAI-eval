"""
Distractor Add mutation operator.

Generates scene variants by injecting same-category distractor objects
into the scene config, testing the model's ability to ignore irrelevant
objects when executing an instruction.
"""
from __future__ import annotations

import copy
from typing import Any

from sepa_eval.mutation.base_operator import MutationError, MutationOperator


class DistractorAdd(MutationOperator):
    """Add N same-category distractor objects to a scene config.

    For each count in ``self.counts`` one candidate is produced with
    ``scene_config["distractors"]`` and ``scene_config["num_distractors"]``
    populated accordingly.

    The distractor entries are intentionally lightweight dicts
    (``{"category": "same", "count": N, "id": i}``) so that downstream
    simulator loaders can hydrate them with full object descriptors.
    """

    name = "DistractorAdd"

    def __init__(self, counts: list[int] = (1, 2, 3)) -> None:
        self.counts = list(counts)

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
        """Generate one candidate per distractor count.

        Parameters
        ----------
        seed_scene_config:
            Source scene config dict.  ``"distractors"`` and
            ``"num_distractors"`` keys are created or overwritten; all
            other keys are preserved unchanged.
        seed_instruction:
            Task instruction string (unchanged across variants).
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
            If ``self.counts`` contains non-positive values.
        """
        for c in self.counts:
            if c < 1:
                raise MutationError(
                    f"DistractorAdd: all counts must be >= 1, got {c}"
                )

        candidates = []
        for count in self.counts:
            new_config = copy.deepcopy(seed_scene_config)
            new_config["num_distractors"] = count
            new_config["distractors"] = [
                {"category": "same", "count": count, "id": i}
                for i in range(count)
            ]

            params = {"num_distractors": count}
            candidates.append(
                self._make_candidate(
                    parent_task_id=parent_task_id,
                    benchmark=benchmark,
                    instruction=seed_instruction,
                    scene_config=new_config,
                    mutation_params=params,
                )
            )

        return candidates
