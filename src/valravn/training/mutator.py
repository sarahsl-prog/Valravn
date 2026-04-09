from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from loguru import logger

from valravn.core.llm_factory import get_llm
from valravn.training.optimizer_state import OptimizerState
from valravn.training.playbook import SecurityPlaybook, ProtectedEntryError

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
  - NEVER output control characters, newlines, or shell metacharacters in entry_id or rule.
"""

# Safety limits
_MAX_ENTRY_ID_LEN = 64
_MAX_RULE_LEN = 500
_MAX_RATIONALE_LEN = 500
_MAX_PLAYBOOK_ENTRIES = 1000

_ALLOWED_OPERATIONS = {"ADD", "UPDATE", "DELETE", "NOOP"}

# Kebab-case pattern: only lowercase letters, numbers, and hyphens
_REGEX_ENTRY_ID = re.compile(r'^[a-z][a-z0-9-]*$')
# Rule cannot contain newlines or control characters
_REGEX_SAFE_TEXT = re.compile(r'^[\x20-\x7E\s]*$')  # Printable ASCII + whitespace


class InvalidMutationError(ValueError):
    """Raised when mutation spec fails validation."""
    pass


class MutationSpec(BaseModel):
    operation: str  # "ADD" | "UPDATE" | "DELETE" | "NOOP"
    entry_id: str = ""
    rule: str = ""
    rationale: str = ""

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Only exact operation names allowed, case-sensitive."""
        v_stripped = v.strip() if v else ""
        if v_stripped not in _ALLOWED_OPERATIONS:
            raise InvalidMutationError(
                f"operation must be one of {_ALLOWED_OPERATIONS}, got {v!r}"
            )
        return v_stripped

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, v: str, info) -> str:
        """Entry ID must be valid kebab-case when operation requires it."""
        operation = info.data.get("operation", "")
        v_stripped = v.strip() if v else ""
        
        # NOOP doesn't need entry_id
        if operation == "NOOP":
            return v_stripped
        
        # ADD, UPDATE, DELETE require entry_id
        if operation in ("ADD", "UPDATE", "DELETE"):
            if not v_stripped:
                raise InvalidMutationError(f"entry_id is required for {operation}")
            
            # Length check
            if len(v_stripped) > _MAX_ENTRY_ID_LEN:
                raise InvalidMutationError(
                    f"entry_id too long: {len(v_stripped)} chars (max {_MAX_ENTRY_ID_LEN})"
                )
            
            # Kebab-case pattern: starts with letter, then letters/digits/hyphens only
            if not _REGEX_ENTRY_ID.match(v_stripped):
                raise InvalidMutationError(
                    f"entry_id must be kebab-case (e.g. 'rule-hash-verify'), got {v_stripped!r}"
                )
        
        return v_stripped

    @field_validator("rule")
    @classmethod
    def validate_rule(cls, v: str, info) -> str:
        """Rule text must be safe and reasonable for ADD/UPDATE operations."""
        operation = info.data.get("operation", "")
        v_stripped = v.strip() if v else ""
        
        # Rule only required for ADD/UPDATE
        if operation in ("ADD", "UPDATE"):
            if not v_stripped:
                raise InvalidMutationError(f"rule is required for {operation}")
            
            # Length check
            if len(v_stripped) > _MAX_RULE_LEN:
                raise InvalidMutationError(
                    f"rule too long: {len(v_stripped)} chars (max {_MAX_RULE_LEN})"
                )
            
            # No shell injection patterns: no backticks, $(), |, ;, &&, || in rules
            if "\n" in v_stripped or "\r" in v_stripped:
                raise InvalidMutationError("rule cannot contain newlines")
            if "`" in v_stripped:
                raise InvalidMutationError("rule cannot contain backticks")
            if "$" in v_stripped:
                raise InvalidMutationError("rule cannot contain dollar signs")
            if "|" in v_stripped or ";" in v_stripped:
                raise InvalidMutationError("rule cannot contain shell metacharacters")
        
        return v_stripped

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str, info) -> str:
        """Rationale is optional but if provided must be safe."""
        v_stripped = v.strip() if v else ""
        
        if v_stripped:
            if len(v_stripped) > _MAX_RATIONALE_LEN:
                raise InvalidMutationError(
                    f"rationale too long: {len(v_stripped)} chars (max {_MAX_RATIONALE_LEN})"
                )
            # Basic safety check
            if "\n" in v_stripped or "\r" in v_stripped:
                raise InvalidMutationError("rationale cannot contain newlines")
        
        return v_stripped


def _get_mutator_llm():
    """Get LLM for playbook mutation with structured output."""
    # Note: Pydantic validation happens on MutationSpec instantiation
    # which is handled separately from LLM structured output
    return get_llm(module="mutator")


def apply_mutation(
    playbook: SecurityPlaybook,
    optimizer_state: OptimizerState,
    iteration: int,
    diagnostic_text: str,
) -> None:
    """Call the LLM to produce a MutationSpec and apply it to the playbook in-place.
    
    BUG-003 Safety checks:
    - Validate MutationSpec against injection attacks
    - Check playbook size limits (max 1000 entries)
    - Log all mutations with diagnostic context
    - Catch InvalidMutationError and log/raise as ValidationError
    """
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
    
    try:
        spec: MutationSpec = _get_mutator_llm().invoke(messages)
    except Exception as e:
        logger.error("LLM invocation failed during mutation: %s", e)
        raise InvalidMutationError(f"LLM failed to produce valid mutation spec: {e}") from e
    
    # BUG-003: Apply safety checks before applying mutation
    _check_mutation_safety(playbook, spec)

    if spec.operation == "ADD":
        logger.info(
            "ADD operation: entry_id=%r to playbook at iteration %d, diagnostic: %r",
            spec.entry_id,
            iteration,
            diagnostic_text[:200],
        )
        playbook.add_entry(
            entry_id=spec.entry_id,
            rule=spec.rule,
            rationale=spec.rationale,
            iteration=iteration,
        )
        optimizer_state.record_change(iteration, f"ADD {spec.entry_id}: {spec.rule}")

    elif spec.operation == "UPDATE":
        logger.info(
            "UPDATE operation: entry_id=%r at iteration %d, diagnostic: %r",
            spec.entry_id,
            iteration,
            diagnostic_text[:200],
        )
        playbook.update_entry(
            entry_id=spec.entry_id,
            rule=spec.rule,
            rationale=spec.rationale,
        )
        optimizer_state.record_change(iteration, f"UPDATE {spec.entry_id}: {spec.rule}")

    elif spec.operation == "DELETE":
        logger.info(
            "DELETE operation: entry_id=%r at iteration %d, rationale: %r",
            spec.entry_id,
            iteration,
            spec.rationale[:200] if spec.rationale else "(no rationale)",
        )
        # Q5: DELETE will raise ProtectedEntryError if entry is protected
        try:
            playbook.delete_entry(entry_id=spec.entry_id)
        except ProtectedEntryError as e:
            raise InvalidMutationError(f"DELETE blocked: {e}") from e
        optimizer_state.record_change(iteration, f"DELETE {spec.entry_id}: {spec.rationale}")

    elif spec.operation == "NOOP":
        logger.debug("NOOP operation: no mutation applied at iteration %d", iteration)

    # NOOP: no changes to playbook or ledger


def _check_mutation_safety(playbook: SecurityPlaybook, spec: MutationSpec) -> None:
    """Validate mutation spec against safety constraints.
    
    Raises InvalidMutationError if safety check fails.
    """
    # Check playbook size limit (prevent unbounded growth)
    if spec.operation == "ADD":
        if len(playbook.entries) >= _MAX_PLAYBOOK_ENTRIES:
            raise InvalidMutationError(
                f"Playbook at max capacity ({_MAX_PLAYBOOK_ENTRIES} entries). "
                "Cannot ADD more entries."
            )
    
    # Check DELETE targets existing entry (prevent deleting non-existent)
    if spec.operation == "DELETE":
        if spec.entry_id not in playbook.entries:
            raise InvalidMutationError(
                f"DELETE targets non-existent entry_id: {spec.entry_id!r}"
            )
    
    # Check UPDATE targets existing entry
    if spec.operation == "UPDATE":
        if spec.entry_id not in playbook.entries:
            raise InvalidMutationError(
                f"UPDATE targets non-existent entry_id: {spec.entry_id!r}"
            )
