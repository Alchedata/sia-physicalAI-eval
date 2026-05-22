"""
ModelsRegistry: YAML-backed model registry with SQLite DB sync.

models.yaml is the source of truth; the DB models table is a cache.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sepa_eval.memory.eval_memory import EvalMemory

logger = logging.getLogger(__name__)

_DEFAULT_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "models.yaml",
)


class ModelsRegistry:
    """Load models from YAML and optionally sync to EvalMemory DB."""

    def __init__(
        self,
        yaml_path: str = _DEFAULT_YAML,
        memory: "EvalMemory | None" = None,
    ) -> None:
        self._yaml_path = yaml_path
        self._memory = memory

    def load(self) -> list[dict[str, Any]]:
        """Parse models.yaml and return list of model dicts."""
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for ModelsRegistry. "
                "Install it with: pip install pyyaml"
            ) from exc

        if not os.path.isfile(self._yaml_path):
            logger.warning("models.yaml not found at %s; returning []", self._yaml_path)
            return []

        with open(self._yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return data.get("models", [])

    def sync_to_db(self) -> int:
        """Load all models from YAML and register them in EvalMemory.

        Returns count of models synced.
        """
        if self._memory is None:
            raise ValueError("ModelsRegistry.sync_to_db() requires memory to be set")

        models = self.load()
        for m in models:
            model_id = m.get("model_id", "")
            if not model_id:
                logger.warning("Skipping model entry with missing model_id: %s", m)
                continue
            self._memory.register_model(
                model_id=model_id,
                framework=m.get("framework", ""),
                checkpoint=m.get("checkpoint", ""),
                benchmarks=m.get("benchmarks", []),
            )

        logger.info("Synced %d model(s) from %s to DB", len(models), self._yaml_path)
        return len(models)
