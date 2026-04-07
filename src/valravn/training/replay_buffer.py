from __future__ import annotations

import json
import random
from pathlib import Path


class ReplayBuffer:
    def __init__(self, n_pass: int = 3, n_reject: int = 2) -> None:
        self.buffer: dict[str, dict] = {}  # {case_id: {case, passes, fails}}
        self.n_pass = n_pass
        self.n_reject = n_reject

    def add_failure(self, case_id: str, case: dict) -> None:
        """Add a case to the buffer with passes=0, fails=1."""
        self.buffer[case_id] = {"case": case, "passes": 0, "fails": 1}

    def record_outcome(self, case_id: str, success: bool) -> None:
        """Record the outcome of a replay attempt.

        If the case is not in the buffer, this is a no-op.
        On success: increment passes, reset fails. Graduate (delete) if passes >= n_pass.
        On failure: increment fails, reset passes.
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
                del self.buffer[case_id]

    def sample(self, n: int) -> list[dict]:
        """Return a random sample of min(n, len(buffer)) case dicts."""
        items = list(self.buffer.values())
        k = min(n, len(items))
        if k == 0:
            return []
        return [entry["case"] for entry in random.sample(items, k)]

    def save(self, path: Path) -> None:
        """Persist n_pass, n_reject, and buffer contents to a JSON file."""
        path = Path(path)
        payload = {
            "n_pass": self.n_pass,
            "n_reject": self.n_reject,
            "buffer": self.buffer,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ReplayBuffer:
        """Reconstruct a ReplayBuffer from a JSON file produced by save()."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        instance = cls(n_pass=payload["n_pass"], n_reject=payload["n_reject"])
        instance.buffer = payload["buffer"]
        return instance
