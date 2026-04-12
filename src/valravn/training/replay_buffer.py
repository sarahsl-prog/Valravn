from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class ReplayBuffer:
    """Manages replayable cases with automatic archival of hopeless cases.
    
    Cases that fail n_reject consecutive times are rejected from the buffer
    and archived for later analysis rather than permanently deleted.
    """
    
    def __init__(
        self,
        n_pass: int = 3,
        n_reject: int = 2,
        archive_path: Path | None = None,
    ) -> None:
        """Initialize the replay buffer.
        
        Args:
            n_pass: Number of consecutive passes to graduate a case
            n_reject: Number of consecutive fails to reject a case
            archive_path: Path to append rejected cases (defaults to
                state_dir/abandoned_cases.jsonl)
        """
        self.buffer: dict[str, dict] = {}  # {case_id: {case, passes, fails}}
        self.n_pass = n_pass
        self.n_reject = n_reject
        self.archive_path = archive_path
        self.archived_count: int = 0  # Track number of archived entries

    def add_failure(self, case_id: str, case: dict) -> None:
        """Add a case to the buffer with passes=0, fails=1."""
        self.buffer[case_id] = {"case": case, "passes": 0, "fails": 1}

    def record_outcome(self, case_id: str, success: bool) -> None:
        """Record the outcome of a replay attempt.

        If the case is not in the buffer, this is a no-op.
        On success: increment passes, reset fails. Graduate (delete) if passes >= n_pass.
        On failure: increment fails, reset passes. Archive and delete if fails >= n_reject.
        """
        if case_id not in self.buffer:
            return
        entry = self.buffer[case_id]
        if success:
            entry["passes"] += 1
            entry["fails"] = 0
            if entry["passes"] >= self.n_pass:
                del self.buffer[case_id]
        else:
            entry["fails"] += 1
            entry["passes"] = 0
            if entry["fails"] >= self.n_reject:
                # Q1: Archive rejected case before removing from buffer
                self._archive_entry(case_id, entry)
                del self.buffer[case_id]

    def _archive_entry(self, case_id: str, entry: dict[str, Any]) -> None:
        """Archive a rejected case to the archive file.
        
        Creates archive with case data, failure count, and timestamp for audit trail.
        Writes as JSONL (one JSON object per line) for append-friendly format.
        """
        if self.archive_path is None:
            return  # No archive configured, skip (legacy behavior)
        
        archive_record = {
            "case_id": case_id,
            "case_data": entry.get("case", {}),
            "final_fails": entry.get("fails", 0),
            "final_passes": entry.get("passes", 0),
            "n_reject_threshold": self.n_reject,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }
        
        try:
            # Append to JSONL file
            with open(self.archive_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(archive_record) + "\n")
            self.archived_count += 1
        except (IOError, OSError) as e:
            logger.error(
                "Failed to archive rejected case %r to %r: %s",
                case_id,
                self.archive_path,
                e,
            )

    def sample(self, n: int) -> list[dict]:
        """Return a random sample of min(n, len(buffer)) case dicts."""
        items = list(self.buffer.values())
        k = min(n, len(items))
        if k == 0:
            return []
        return [entry["case"] for entry in random.sample(items, k)]

    def save(self, path: Path) -> None:
        """Persist n_pass, n_reject, buffer, and archive metadata to a JSON file."""
        path = Path(path)
        payload = {
            "n_pass": self.n_pass,
            "n_reject": self.n_reject,
            "buffer": self.buffer,
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "archived_count": self.archived_count,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ReplayBuffer:
        """Reconstruct a ReplayBuffer from a JSON file produced by save()."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        archive_path = payload.get("archive_path")
        instance = cls(
            n_pass=payload["n_pass"],
            n_reject=payload["n_reject"],
            archive_path=Path(archive_path) if archive_path else None,
        )
        instance.buffer = payload["buffer"]
        instance.archived_count = payload.get("archived_count", 0)
        return instance
