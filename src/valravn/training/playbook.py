from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class SecurityPlaybook(BaseModel):
    entries: dict[str, dict] = {}
    version: int = 0

    def add_entry(self, entry_id: str, rule: str, rationale: str, iteration: int = 0) -> None:
        self.entries[entry_id] = {
            "rule": rule,
            "rationale": rationale,
            "added_iteration": iteration,
        }

    def update_entry(self, entry_id: str, rule: str, rationale: str) -> None:
        if entry_id not in self.entries:
            return
        self.entries[entry_id]["rule"] = rule
        self.entries[entry_id]["rationale"] = rationale

    def delete_entry(self, entry_id: str) -> None:
        self.entries.pop(entry_id, None)

    def to_prompt_section(self) -> str:
        if not self.entries:
            return "## Security Playbook\n\n(empty)"
        lines = ["## Security Playbook\n"]
        for entry_id, entry in self.entries.items():
            lines.append(f"- **{entry_id}**: {entry['rule']}")
            lines.append(f"  - Rationale: {entry['rationale']}")
        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.model_dump_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SecurityPlaybook":
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
