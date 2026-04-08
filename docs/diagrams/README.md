# Valravn Documentation Diagrams

This directory contains Mermaid diagrams for visualizing the Valravn agent architecture.

## Diagrams

### `agent_workflow.mmd`
Visualizes the LangGraph-based investigation agent workflow:
- **Planning Phase**: Initial plan generation via LLM
- **Execution Loop**: Tool execution with self-assessment and anomaly detection
- **Conclusion Phase**: Report synthesis and generation

### `learning_modalities.mmd`
Illustrates the four learning mechanisms integrated into Valravn:
1. **Reflective Comparison Learning**: Success/failure trace analysis via Reflector
2. **Self-Guidance**: Real-time progress assessment with trust scheduling
3. **Replay-Based Reinforcement**: Buffer for difficult cases with pass/reject thresholds
4. **Playbook Mutation**: Automated rule updates via LLM-based mutation

## Rendering

These diagrams can be rendered using:
- [Mermaid Live Editor](https://mermaid.live)
- GitHub/GitLab markdown (via mermaid code blocks)
- VS Code extensions (Mermaid Preview)
- CLI: `mmdc -i diagram.mmd -o diagram.svg`
