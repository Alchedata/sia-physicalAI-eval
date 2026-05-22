"""
Base class and error for all SEPA-Eval mutation operators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sepa_eval.memory.schema import CandidateTask


class MutationError(Exception):
    """Raised by a MutationOperator when it cannot produce a valid candidate."""


class MutationOperator(ABC):
    """Abstract base for every scenario mutation operator.

    Subclasses must set a class-level ``name`` attribute and implement
    :meth:`generate`.
    """

    name: str  # e.g. "PosePerturbation"

    @abstractmethod
    def generate(
        self,
        seed_scene_config: dict,
        seed_instruction: str,
        **kwargs: Any,
    ) -> "list[CandidateTask]":
        """Generate CandidateTask variants from a seed scene config + instruction.

        Parameters
        ----------
        seed_scene_config:
            The simulator scene configuration dict from the seed episode.
        seed_instruction:
            Natural-language task instruction from the seed episode.
        **kwargs:
            Operator implementations must accept (at minimum)
            ``parent_task_id: str`` and ``benchmark: str``.

        Returns
        -------
        list[CandidateTask]
            Zero or more candidates.  May return fewer than requested if
            constraints prevent generating the full set.

        Raises
        ------
        MutationError
            On unrecoverable failure (e.g. malformed config, API error).
        """
        ...

    def _make_candidate(
        self,
        parent_task_id: str,
        benchmark: str,
        instruction: str,
        scene_config: dict,
        mutation_params: dict,
    ) -> "CandidateTask":
        """Convenience factory that delegates to :meth:`CandidateTask.new`."""
        from sepa_eval.memory.schema import CandidateTask  # lazy import

        return CandidateTask.new(
            parent_task_id=parent_task_id,
            benchmark=benchmark,
            instruction=instruction,
            scene_config=scene_config,
            mutation_type=self.name,
            mutation_params=mutation_params,
        )
