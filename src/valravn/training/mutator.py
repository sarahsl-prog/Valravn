from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from valravn.training.optimizer_state import OptimizerState
from valravn.training.playbook import SecurityPlaybook

_SYSTEM_PROMPT = """\
You are a DFIR playbook editor. Given a diagnostic report and the current state of a
security playbook, decide on a single, minimal mutation to improve the playbook.

Choose exactly one of the following operations:

  ADD    — insert a new entry (provide entry_id, rule, and rationale)
  UPDATE — modify an existing entry (provide entry_id, rule, and rationale)
  DELETE — remove an existing entry that is harmful or redundant (provide entry_id and rationale)
  NOOP   — no change is needed or possible (e.g. intractable failures)

Rules:
  - Prefer NOOP when the attribution is "intractable" or "execution_variance".
  - Only use ADD or UPDATE when the attribution is "actionable_gap".
  - entry_id values should be short kebab-case identifiers (e.g. "rule-hash-verify").
  - Keep rules concise and actionable (one sentence imperative).
  - Keep rationale brief (one sentence).
"""


class MutationSpec(BaseModel):
    operation: str  # "ADD" | "UPDATE" | "DELETE" | "NOOP"
    entry_id: str = ""
    rule: str = ""
    rationale: str = ""


def _get_mutator_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(MutationSpec)


def apply_mutation(
    playbook: SecurityPlaybook,
    optimizer_state: OptimizerState,
    iteration: int,
    diagnostic_text: str,
) -> None:
    """Call the LLM to produce a MutationSpec and apply it to the playbook in-place."""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Diagnostic\n{diagnostic_text}\n\n"
                f"## Optimizer state\n{optimizer_state.to_context()}\n\n"
                f"{playbook.to_prompt_section()}"
            )
        ),
    ]
    spec: MutationSpec = _get_mutator_llm().invoke(messages)

    if spec.operation == "ADD":
        playbook.add_entry(
            entry_id=spec.entry_id,
            rule=spec.rule,
            rationale=spec.rationale,
            iteration=iteration,
        )
        optimizer_state.record_change(iteration, f"ADD {spec.entry_id}: {spec.rule}")

    elif spec.operation == "UPDATE":
        playbook.update_entry(
            entry_id=spec.entry_id,
            rule=spec.rule,
            rationale=spec.rationale,
        )
        optimizer_state.record_change(iteration, f"UPDATE {spec.entry_id}: {spec.rule}")

    elif spec.operation == "DELETE":
        playbook.delete_entry(entry_id=spec.entry_id)
        optimizer_state.record_change(iteration, f"DELETE {spec.entry_id}: {spec.rationale}")

    # NOOP: no changes to playbook or ledger
