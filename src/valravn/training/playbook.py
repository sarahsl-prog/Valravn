from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ProtectedEntryError(Exception):
    """Raised when attempting to delete or modify a protected playbook entry."""
    pass


class SecurityPlaybook(BaseModel):
    entries: dict[str, dict] = {}
    version: int = 0
    protected_ids: set[str] = Field(default_factory=set)

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
        """Delete an entry from the playbook.
        
        Raises:
            ProtectedEntryError: If the entry_id is in the protected list.
        """
        if entry_id in self.protected_ids:
            raise ProtectedEntryError(
                f"Cannot delete protected entry '{entry_id}'. "
                f"Remove from protected_ids first."
            )
        self.entries.pop(entry_id, None)

    def protect_entry(self, entry_id: str) -> None:
        """Add an entry to the protected list.
        
        Protected entries cannot be deleted by the LLM mutator.
        This provides human-in-the-loop safety for critical rules.
        """
        if entry_id in self.entries:
            self.protected_ids.add(entry_id)

    def unprotect_entry(self, entry_id: str) -> None:
        """Remove an entry from the protected list."""
        self.protected_ids.discard(entry_id)

    def is_protected(self, entry_id: str) -> bool:
        """Check if an entry is protected."""
        return entry_id in self.protected_ids

    def to_prompt_section(self) -> str:
        if not self.entries:
            return "## Security Playbook\n\n(empty)"
        lines = ["## Security Playbook\n"]
        for entry_id, entry in self.entries.items():
            protected_mark = " [PROTECTED]" if entry_id in self.protected_ids else ""
            lines.append(f"- **{entry_id}**{protected_mark}: {entry['rule']}")
            lines.append(f"  - Rationale: {entry['rationale']}")
        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.model_dump_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SecurityPlaybook":
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
