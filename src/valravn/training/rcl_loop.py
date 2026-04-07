from __future__ import annotations

import json
from pathlib import Path

from valravn.training.mutator import apply_mutation
from valravn.training.optimizer_state import OptimizerState
from valravn.training.playbook import SecurityPlaybook
from valravn.training.reflector import ReflectionDiagnostic, reflect_on_trajectory
from valravn.training.replay_buffer import ReplayBuffer

_ITERATION_FILE = "iteration.json"


class RCLTrainer:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        playbook_path = self.state_dir / "playbook.json"
        optimizer_path = self.state_dir / "optimizer_state.json"
        buffer_path = self.state_dir / "replay_buffer.json"
        iteration_path = self.state_dir / _ITERATION_FILE

        self.playbook = (
            SecurityPlaybook.load(playbook_path) if playbook_path.exists() else SecurityPlaybook()
        )
        self.optimizer_state = (
            OptimizerState.load(optimizer_path) if optimizer_path.exists() else OptimizerState()
        )
        self.replay_buffer = (
            ReplayBuffer.load(buffer_path) if buffer_path.exists() else ReplayBuffer()
        )
        self._iteration: int = (
            json.loads(iteration_path.read_text(encoding="utf-8"))["iteration"]
            if iteration_path.exists()
            else 0
        )

    def process_investigation_result(
        self,
        case_id: str,
        success_trace: str,
        failure_trace: str,
        success: bool,
    ) -> ReflectionDiagnostic | None:
        """Process the result of a single investigation.

        - If both traces are non-empty, run the reflector.
        - If the diagnostic attribution is "actionable_gap", apply a mutation and bump
          playbook.version.
        - If the investigation was not successful, add the case to the replay buffer.
        - Record the outcome in the replay buffer.
        - Increment the iteration counter and persist state.
        - Return the diagnostic (or None if no reflection was performed).
        """
        diagnostic: ReflectionDiagnostic | None = None

        if success_trace and failure_trace:
            diagnostic = reflect_on_trajectory(
                success_trace=success_trace,
                failure_trace=failure_trace,
                playbook_context=self.playbook.to_prompt_section(),
            )

            if diagnostic.attribution == "actionable_gap":
                apply_mutation(
                    playbook=self.playbook,
                    optimizer_state=self.optimizer_state,
                    iteration=self._iteration,
                    diagnostic_text=(
                        f"attribution={diagnostic.attribution}\n"
                        f"root_cause={diagnostic.root_cause}\n"
                        f"coverage_gap={diagnostic.coverage_gap}"
                    ),
                )
                self.playbook.version += 1

        if not success:
            if case_id not in self.replay_buffer.buffer:
                self.replay_buffer.add_failure(case_id, {"case_id": case_id})
            else:
                self.replay_buffer.record_outcome(case_id, success=False)
        else:
            self.replay_buffer.record_outcome(case_id, success=True)

        self._iteration += 1
        self.save_state()

        return diagnostic

    def save_state(self) -> None:
        """Persist playbook, optimizer state, replay buffer, and iteration to state_dir."""
        self.playbook.save(self.state_dir / "playbook.json")
        self.optimizer_state.save(self.state_dir / "optimizer_state.json")
        self.replay_buffer.save(self.state_dir / "replay_buffer.json")
        (self.state_dir / _ITERATION_FILE).write_text(
            json.dumps({"iteration": self._iteration}), encoding="utf-8"
        )
