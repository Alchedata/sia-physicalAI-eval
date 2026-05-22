"""
HumanReviewQueue — persistent JSONL queue for async human oversight.

Each call to ``enqueue`` appends a single JSON line to the queue file.
Reviewers call ``approve`` or ``reject`` to append decision records.
``list_pending`` returns all entries from the file (both queued and decided).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HumanReviewQueue:
    """
    Append-only JSONL queue for human review of candidate tasks.

    The file grows monotonically; decisions are new lines, not in-place edits.
    This gives a full audit trail.
    """

    def __init__(
        self,
        queue_path: str = "./eval_memory/human_review_queue.jsonl",
    ) -> None:
        self.queue_path = Path(queue_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def enqueue(self, candidate: Any) -> None:
        """
        Append a review request entry for *candidate*.

        Writes JSON with keys:
          - task_id
          - instruction
          - mutation_type
          - created_at  (ISO-8601 UTC)
          - event       "queued"
        """
        entry = {
            "event": "queued",
            "task_id": getattr(candidate, "task_id", None),
            "instruction": getattr(candidate, "instruction", None),
            "mutation_type": getattr(candidate, "mutation_type", None),
            "created_at": _utc_now_iso(),
        }
        self._append(entry)

    def approve(self, task_id: str) -> None:
        """
        Record an approval decision.

        Appends JSON with keys:
          - event       "approved"
          - task_id
          - approved    True
          - reviewed_at (ISO-8601 UTC)
        """
        entry = {
            "event": "approved",
            "task_id": task_id,
            "approved": True,
            "reviewed_at": _utc_now_iso(),
        }
        self._append(entry)

    def reject(self, task_id: str, reason: str = "") -> None:
        """
        Record a rejection decision.

        Appends JSON with keys:
          - event       "rejected"
          - task_id
          - approved    False
          - reason
          - reviewed_at (ISO-8601 UTC)
        """
        entry = {
            "event": "rejected",
            "task_id": task_id,
            "approved": False,
            "reason": reason,
            "reviewed_at": _utc_now_iso(),
        }
        self._append(entry)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_pending(self) -> list[dict]:
        """
        Return all entries from the queue file.

        Each element is a parsed dict.  Both "queued" and decision records
        are included so callers can reconstruct the full audit trail.
        """
        if not self.queue_path.exists():
            return []
        entries: list[dict] = []
        with self.queue_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines rather than crashing.
                    continue
        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, entry: dict) -> None:
        """Create parent directories if needed and append a JSON line."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.queue_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()
