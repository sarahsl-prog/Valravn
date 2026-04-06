# Feature Specification: Autonomous DFIR Agents

**Feature Branch**: `001-autonomous-dfir-agents`
**Created**: 2026-04-05
**Status**: Draft
**Input**: "Build autonomous AI agents on the SANS SIFT Workstation and teach the agents
how to sequence its approach, recognize when something doesn't add up, and self-correct
when it gets it wrong."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Guided Forensic Investigation (Priority: P1)

A DFIR analyst assigns an investigation task to an autonomous agent (e.g., "Investigate
this memory image for signs of lateral movement"). The agent independently determines
the correct sequence of forensic tools to apply, executes them against the evidence,
and delivers a findings report — without requiring step-by-step operator guidance.

**Why this priority**: This is the core value proposition. An agent that can plan and
execute a forensic workflow end-to-end is the minimum viable capability.

**Independent Test**: Give the agent a single investigation prompt and a valid evidence
artifact. Verify it produces a complete findings report citing tool output, with no
operator intervention during execution.

**Acceptance Scenarios**:

1. **Given** a mounted memory image and the prompt "identify active network connections
   at time of acquisition", **When** the agent is invoked, **Then** it selects the
   appropriate forensic tools, executes them in the correct order, and returns a report
   listing connections with supporting evidence within 10 minutes.

2. **Given** an investigation prompt outside the agent's known capability, **When** the
   agent is invoked, **Then** it reports the gap explicitly rather than fabricating
   findings.

---

### User Story 2 - Anomaly Recognition and Escalation (Priority: P2)

During an investigation the agent encounters output that contradicts expected behavior
or internal consistency (e.g., a timestamp that predates the OS install, a process
whose parent doesn't exist, tool output that conflicts across two runs). The agent
flags the anomaly, records it in the findings, and adjusts its investigation path
rather than proceeding as if nothing is wrong.

**Why this priority**: Undetected inconsistencies are the primary source of incorrect
forensic conclusions. An agent that blindly continues past contradictions produces
unreliable output.

**Independent Test**: Inject a known contradiction into test evidence. Verify that the
agent's report explicitly calls out the anomaly and that the investigation path changed
in response to it.

**Acceptance Scenarios**:

1. **Given** evidence containing a detectable internal inconsistency, **When** the
   agent processes it, **Then** the findings report contains an explicit anomaly entry
   describing the contradiction and its forensic significance.

2. **Given** two tool outputs that produce conflicting results for the same artifact,
   **When** the agent processes them, **Then** it records both results, flags the
   conflict, and does not suppress either result in the final report.

---

### User Story 3 - Self-Correction on Tool Failure (Priority: P3)

A tool invoked by the agent fails (non-zero exit, empty output, stderr error) or
returns output the agent determines is incomplete. The agent diagnoses the failure,
attempts a corrective action (adjusted parameters, alternative approach, re-mount),
and retries — up to a defined limit — before escalating the failure in the report.

**Why this priority**: Evidence processing environments are imperfect. An agent that
halts or silently drops results on first failure provides less value than one that
exhausts reasonable recovery paths.

**Independent Test**: Introduce a deliberate tool failure (e.g., point the agent at a
missing evidence path). Verify the agent attempts at least one corrective action,
documents the failure and recovery attempt, and either succeeds or escalates with a
clear failure record.

**Acceptance Scenarios**:

1. **Given** a tool invocation that exits with an error, **When** the agent encounters
   it, **Then** it reads the error output, formulates a corrective hypothesis, and
   retries at least once before recording failure.

2. **Given** a tool that returns empty output where output is expected, **When** the
   agent processes it, **Then** it treats the absence as an anomaly, investigates why,
   and records findings rather than silently proceeding.

3. **Given** a corrective retry that also fails, **When** the retry limit is reached,
   **Then** the agent escalates the failure in the report with all diagnostic context
   captured — it does not halt silently.

---

### Edge Cases

- What happens when the agent encounters evidence it has no skill coverage for (unknown
  file type, unsupported image format)?
- How does the agent behave when all retry attempts for a critical tool are exhausted
  and the investigation cannot proceed without that tool's output?
- What happens when anomaly detection produces a false positive on legitimate but
  unusual evidence?
- How does the agent handle evidence that spans multiple acquisition formats in a
  single investigation?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The agent MUST accept a natural-language investigation prompt and an
  evidence reference as its two primary inputs.
- **FR-002**: The agent MUST select and sequence forensic tools based on the
  investigation prompt and evidence type, without operator-supplied step-by-step
  instructions.
- **FR-003**: The agent MUST execute each selected tool using the authoritative paths
  and invocation conventions defined in the project constitution.
- **FR-004**: The agent MUST capture all tool stdout and stderr output and persist it
  to `./analysis/` before deriving any conclusions from it.
- **FR-005**: The agent MUST detect internal inconsistencies in tool output (conflicting
  values, impossible relationships, unexpected absences) and record them as anomalies.
- **FR-006**: The agent MUST adjust its investigation sequence when an anomaly is
  detected — either by adding targeted follow-up steps or by noting why no follow-up
  is warranted.
- **FR-007**: The agent MUST NOT modify, write to, or delete any file in `/cases/`,
  `/mnt/`, `/media/`, or any `evidence/` directory.
- **FR-008**: On tool failure, the agent MUST read error output, formulate a corrective
  hypothesis, and retry at least once with a modified approach before recording failure.
- **FR-009**: The retry limit for any single tool invocation MUST be configurable and
  MUST default to 3 attempts.
- **FR-010**: The agent MUST produce a final findings report in `./reports/` that cites
  the specific tool invocation and raw output behind every conclusion.
- **FR-011**: Every timestamp in agent output MUST be expressed in UTC.
- **FR-012**: The agent MUST complete its investigation and deliver a findings artifact
  without requesting operator confirmation or input mid-task.
- **FR-013**: The agent MUST report its own failures and gaps explicitly; silent failure
  is prohibited.

### Key Entities

- **Investigation Task**: An operator-supplied prompt paired with one or more evidence
  references; defines the scope of a single agent run.
- **Investigation Plan**: The ordered sequence of tool invocations the agent derives
  from the task; updated dynamically as anomalies are detected.
- **Tool Invocation Record**: A persisted record of a single tool execution: command,
  arguments, exit code, stdout, stderr, timestamp.
- **Anomaly**: A detected inconsistency or unexpected result; includes a description,
  the source tool outputs that produced it, and the agent's response action.
- **Findings Report**: The final artifact delivered to the operator; includes all
  conclusions, supporting evidence citations, anomalies, tool failures, and
  self-correction events.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a well-scoped investigation prompt, the agent delivers a complete
  findings report with zero operator interventions in 100% of runs.
- **SC-002**: The agent correctly identifies and records known injected anomalies in
  at least 90% of test cases where a detectable contradiction is present.
- **SC-003**: The agent successfully self-corrects and completes the investigation
  in at least 80% of cases where a recoverable tool failure is introduced.
- **SC-004**: Every conclusion in the findings report is traceable to a cited tool
  invocation and raw output — 0% of conclusions are unsupported assertions.
- **SC-005**: No evidence directory is modified in any agent run across all test cases.
- **SC-006**: All findings report timestamps are in UTC with no exceptions.

## Assumptions

- The SANS SIFT Workstation environment, tool paths, and skill routing defined in the
  project constitution are stable and available during agent execution.
- Evidence has already been acquired and is available in read-only mount points before
  the agent is invoked; acquisition is out of scope.
- The operator supplies a sufficiently scoped investigation prompt; interactive prompt
  refinement is out of scope for v1.
- A single agent handles one investigation task per run; parallel multi-task
  orchestration is out of scope for v1.
- The agent operates on the local SIFT workstation only; remote evidence or cloud
  storage integration is out of scope.
- Findings reports are text-based artifacts (Markdown or structured text); rich
  visualizations or dashboards are out of scope for v1.
