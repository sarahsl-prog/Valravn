from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

_SYSTEM_PROMPT = """\
You are an expert DFIR training analyst comparing a successful and a failed agent trajectory.
Your task is to diagnose *why* the agent failed and classify the failure into one of three
attribution types:

  - actionable_gap       — The failure is traceable to a missing or incorrect playbook rule.
                           A targeted edit to the playbook could prevent the failure in future.
  - execution_variance   — The playbook already covers the situation, but the agent did not
                           follow it correctly (e.g. skipped a step, misread output).
                           No playbook change is warranted.
  - intractable          — The failure cannot be addressed by changing the playbook (e.g.
                           encrypted evidence, missing tools, inherently ambiguous task).
                           No playbook change is warranted.

For each analysis provide:
  - attribution  : one of the three types above (exact string)
  - root_cause   : a concise single-sentence explanation of the failure mechanism
  - coverage_gap : (only for actionable_gap) a brief description of the missing or incorrect
                   playbook rule; leave empty for the other attribution types
"""


class ReflectionDiagnostic(BaseModel):
    attribution: str  # "actionable_gap" | "execution_variance" | "intractable"
    root_cause: str
    coverage_gap: str = ""


def _get_reflector_llm() -> object:
    llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
    return llm.with_structured_output(ReflectionDiagnostic)


def reflect_on_trajectory(
    success_trace: str,
    failure_trace: str,
    playbook_context: str,
) -> ReflectionDiagnostic:
    """Analyse a success/failure trace pair and return a structured diagnostic."""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Success trace\n{success_trace}\n\n"
                f"## Failure trace\n{failure_trace}\n\n"
                f"{playbook_context}"
            )
        ),
    ]
    result: ReflectionDiagnostic = _get_reflector_llm().invoke(messages)
    return result
