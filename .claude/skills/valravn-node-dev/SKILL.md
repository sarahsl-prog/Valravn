---
name: valravn-node-dev
description: >
  Implement, modify, debug, or extend LangGraph nodes for the Valravn DFIR agent.
  Use this skill whenever you are adding a new node, changing existing node logic,
  modifying AgentState, updating graph wiring in graph.py, creating Pydantic models,
  or working in src/valravn/nodes/ or src/valravn/models/.
  Also triggers for: "add a node", "implement X in the graph", "wire up", "state field",
  "update plan node", "fix tool_runner", "new forensic step".
---

# Node Development Skill

## Workflow

1. **Read the existing node** (or the closest analogue) before writing new code
2. **Identify state contract** — which keys does the node read? which does it write? private (`_`) vs public
3. **Implement** following the patterns below
4. **Wire the graph** in `graph.py` if adding a new node
5. **Run tests** to verify: `pytest tests/unit/ -v`

## Node Anatomy

```python
from __future__ import annotations
from pathlib import Path
from loguru import logger
from valravn.core.llm_factory import get_llm
from valravn.core.parsing import parse_llm_json

def my_node(state: dict) -> dict:
    """LangGraph node: <one-line description>."""
    # 1. Extract what you need from state
    task = state["task"]
    output_dir = Path(state.get("_output_dir", "."))
    logger.info("Node: my_node | key={}", task.id[:8])

    # 2. Do work (LLM call, subprocess, pure computation)
    llm = get_llm(module="my_node")
    response = llm.invoke(messages)
    result = parse_llm_json(response.content, MyOutputModel)

    # 3. Return ONLY the keys you're changing
    return {
        "my_field": result.value,
        "_my_signal": True,
    }
```

## State Contract Rules

- Return only keys the node owns — do not echo unchanged state
- Private (`_`) keys: ephemeral signals between nodes; reset them after use (e.g., `_pending_anomalies: False` in the consuming node's return)
- New public keys require an addition to `AgentState` in `state.py`
- New private keys also need to be in `AgentState` (LangGraph strips unknown keys)
- Initialize new state keys in the `initial_state` dict in `graph.py`

## LLM Calls

```python
# Always use the factory — it handles fallback chains and retry
llm = get_llm(module="my_node")  # module name used for config/logging

# Invoke with messages
from langchain_core.messages import SystemMessage, HumanMessage
response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user_msg)])

# Parse structured output
result = parse_llm_json(response.content, MyPydanticModel)
```

## Adding a Node to the Graph

In `graph.py`:
```python
from valravn.nodes.my_node import my_node
builder.add_node("my_node", my_node)
builder.add_edge("previous_node", "my_node")
builder.add_edge("my_node", "next_node")
# or conditional:
builder.add_conditional_edges("my_node", route_function)
```

Initialize new state keys in `initial_state` in `graph.py`'s `run()` function.

## Routing Functions

```python
def route_after_my_node(state: AgentState) -> str:
    if state.get("_my_signal"):
        return "branch_a"
    return "branch_b"
```

## Pydantic Models

```python
from pydantic import BaseModel, field_validator

class MyOutput(BaseModel):
    steps: list[str]
    rationale: str

    @field_validator("steps")
    @classmethod
    def steps_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("steps must not be empty")
        return v
```

Domain models go in `src/valravn/models/`. Private/internal specs (only used within a node) can be defined in the node file itself (prefix with `_`).

## Common Mistakes

- Returning the full state dict instead of a partial — LangGraph merges, so return only changed keys
- Calling an LLM model directly instead of `get_llm()`
- Writing output to `/tmp/` instead of `output_dir / "analysis/"`
- Adding a state field without initializing it in `graph.py`'s `initial_state`

## References

For deep patterns: `references/node-patterns.md`
