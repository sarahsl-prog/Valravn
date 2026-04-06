from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ProgressAnchor(BaseModel):
    stage: int
    description: str
    typical_tools: list[str]
    completion_signal: str


class InvestigationBlueprint(BaseModel):
    incident_type: str
    anchors: list[ProgressAnchor]
    success_rate: float = 0.0


class ProgressMemory:
    def __init__(self) -> None:
        self.blueprints: list[InvestigationBlueprint] = []
        self._init_defaults()

    def _init_defaults(self) -> None:
        self.blueprints = [
            InvestigationBlueprint(
                incident_type="memory_analysis",
                anchors=[
                    ProgressAnchor(
                        stage=1,
                        description="Enumerate running processes and network connections",
                        typical_tools=["volatility", "pslist", "netscan"],
                        completion_signal="Process and connection list acquired",
                    ),
                    ProgressAnchor(
                        stage=2,
                        description="Detect code injection and rootkit artifacts",
                        typical_tools=["malfind", "ldrmodules", "ssdt"],
                        completion_signal="Suspicious injected regions identified",
                    ),
                    ProgressAnchor(
                        stage=3,
                        description="Extract artifacts and dump suspicious regions",
                        typical_tools=["memdump", "dumpfiles", "procdump"],
                        completion_signal="Artifact files extracted for further analysis",
                    ),
                ],
                success_rate=0.0,
            ),
            InvestigationBlueprint(
                incident_type="disk_forensics",
                anchors=[
                    ProgressAnchor(
                        stage=1,
                        description="Build filesystem timeline of activity",
                        typical_tools=["fls", "mactime", "log2timeline"],
                        completion_signal="Timeline CSV generated",
                    ),
                    ProgressAnchor(
                        stage=2,
                        description="Analyse file metadata and deleted file recovery",
                        typical_tools=["istat", "tsk_recover", "foremost"],
                        completion_signal="Recovered files catalogued",
                    ),
                    ProgressAnchor(
                        stage=3,
                        description="Run YARA signatures across disk image",
                        typical_tools=["yara", "yara-python", "clamav"],
                        completion_signal="YARA scan report complete",
                    ),
                ],
                success_rate=0.0,
            ),
            InvestigationBlueprint(
                incident_type="windows_artifacts",
                anchors=[
                    ProgressAnchor(
                        stage=1,
                        description="Parse Windows event logs for suspicious activity",
                        typical_tools=["evtx_dump", "chainsaw", "wevtutil"],
                        completion_signal="Event log anomalies identified",
                    ),
                    ProgressAnchor(
                        stage=2,
                        description="Examine registry hives for persistence and configuration",
                        typical_tools=["regripper", "regedit", "autoruns"],
                        completion_signal="Registry persistence keys reviewed",
                    ),
                    ProgressAnchor(
                        stage=3,
                        description="Analyse prefetch files and shimcache for execution history",
                        typical_tools=["pecmd", "appcompatprocessor", "winprefetchview"],
                        completion_signal="Execution history reconstructed",
                    ),
                ],
                success_rate=0.0,
            ),
        ]

    def retrieve_blueprint(self, description: str) -> InvestigationBlueprint:
        tokens = set(description.lower().split())
        best: InvestigationBlueprint | None = None
        best_score = -1

        for bp in self.blueprints:
            score = 0
            # Match against incident_type tokens
            for part in bp.incident_type.replace("_", " ").split():
                if part in tokens:
                    score += 2

            for anchor in bp.anchors:
                # Match anchor description words
                for word in anchor.description.lower().split():
                    if word in tokens:
                        score += 1
                # Match tool names
                for tool in anchor.typical_tools:
                    for tool_part in tool.replace("-", " ").replace("_", " ").split():
                        if tool_part in tokens:
                            score += 1

            if score > best_score:
                best_score = score
                best = bp

        # Fallback to first blueprint when no meaningful match found
        if best is None or best_score == 0:
            return self.blueprints[0]

        return best

    def add_blueprint(self, blueprint: InvestigationBlueprint) -> None:
        self.blueprints.append(blueprint)

    def save(self, path: Path) -> None:
        path = Path(path)
        data = [bp.model_dump() for bp in self.blueprints]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ProgressMemory:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        pm = cls.__new__(cls)
        pm.blueprints = [InvestigationBlueprint.model_validate(item) for item in data]
        return pm
