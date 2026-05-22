"""
Material Swap mutation operator.

Generates scene variants by replacing material, texture, and friction
properties with sampled alternatives drawn from curated value pools.
"""
from __future__ import annotations

import copy
import random
from typing import Any

from sepa_eval.mutation.base_operator import MutationError, MutationOperator

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------

MATERIALS = ["wood", "metal", "plastic", "rubber", "ceramic"]
TEXTURES = ["rough", "smooth", "matte", "glossy"]
FRICTION_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5]


class MaterialSwap(MutationOperator):
    """Swap material, texture, and friction coefficient in the scene config.

    Each generated variant draws a distinct random combination so that
    the returned list contains diverse candidates.  If fewer unique
    combinations are available than ``n_variants``, the returned list
    will be shorter.
    """

    name = "MaterialSwap"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        seed_scene_config: dict,
        seed_instruction: str,
        parent_task_id: str = "unknown",
        benchmark: str = "unknown",
        n_variants: int = 3,
        **kwargs: Any,
    ) -> list:
        """Generate ``n_variants`` candidates with distinct material combos.

        Parameters
        ----------
        seed_scene_config:
            Source scene config dict.  ``material``, ``texture``, and
            ``friction_coeff`` keys are overwritten; all other keys are
            preserved unchanged.
        seed_instruction:
            Task instruction string (unchanged across variants).
        parent_task_id:
            ID of the seed trace this mutation originates from.
        benchmark:
            Benchmark name.
        n_variants:
            How many distinct material/texture/friction combinations to
            generate.  Capped at ``len(MATERIALS) * len(TEXTURES)``.

        Returns
        -------
        list[CandidateTask]
        """
        if n_variants < 1:
            raise MutationError("n_variants must be >= 1")

        # Build a shuffled pool of (material, texture, friction) triples and
        # pick up to n_variants without replacement.
        pool: list[tuple] = [
            (mat, tex, fric)
            for mat in MATERIALS
            for tex in TEXTURES
            for fric in FRICTION_VALUES
        ]
        random.shuffle(pool)
        selected = pool[:n_variants]

        candidates = []
        for mat, tex, fric in selected:
            new_config = copy.deepcopy(seed_scene_config)
            new_config["material"] = mat
            new_config["texture"] = tex
            new_config["friction_coeff"] = fric

            params = {"material": mat, "texture": tex, "friction_coeff": fric}
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
