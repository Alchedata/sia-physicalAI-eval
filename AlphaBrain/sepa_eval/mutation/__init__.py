"""
sepa_eval.mutation — Scenario mutation engine for SEPA-Eval.

Exports all mutation operators and the shared base/error types so that
callers can do::

    from sepa_eval.mutation import PosePerturbation, MaterialSwap, ...
"""
from sepa_eval.mutation.base_operator import MutationError, MutationOperator
from sepa_eval.mutation.distractor_add import DistractorAdd
from sepa_eval.mutation.horizon_extension import HorizonExtension
from sepa_eval.mutation.instruction_paraphrase import InstructionParaphrase
from sepa_eval.mutation.material_swap import MaterialSwap
from sepa_eval.mutation.pose_perturbation import PosePerturbation

__all__ = [
    "DistractorAdd",
    "HorizonExtension",
    "InstructionParaphrase",
    "MaterialSwap",
    "MutationError",
    "MutationOperator",
    "PosePerturbation",
]
