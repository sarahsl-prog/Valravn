from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class OptimizerState(BaseModel):
    change_ledger: list[str] = []
    open_hypotheses: list[str] = []
    phase: str = "exploratory"

    def record_change(self, iteration: int, action: str) -> None:
        self.change_ledger.append(f"[iter={iteration}] {action}")

    def add_hypothesis(self, hypothesis: str) -> None:
        if hypothesis not in self.open_hypotheses:
            self.open_hypotheses.append(hypothesis)

    def resolve_hypothesis(self, hypothesis: str) -> None:
        if hypothesis in self.open_hypotheses:
            self.open_hypotheses.remove(hypothesis)

    def transition_to_convergent(self) -> None:
        self.phase = "convergent"

    def transition_to_exploratory(self) -> None:
        self.phase = "exploratory"

    def to_context(self) -> str:
        return json.dumps(
            {
                "phase": self.phase,
                "changes": self.change_ledger[-10:],
                "hypotheses": self.open_hypotheses,
            }
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.model_dump_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "OptimizerState":
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
