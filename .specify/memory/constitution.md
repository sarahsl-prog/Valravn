<!--
SYNC IMPACT REPORT
==================
Version change: [TEMPLATE] → 1.0.0 (initial ratification — all placeholders replaced)

Modified principles:
  None (first-time fill; template had no prior concrete principles)

Added sections:
  - Core Principles (I–V)
  - Tool Configuration
  - Forensic Workflow
  - Governance

Removed sections:
  - [SECTION_2_NAME] / [SECTION_3_NAME] placeholders replaced with
    "Tool Configuration" and "Forensic Workflow" respectively

Templates checked:
  ✅ .specify/templates/plan-template.md — Constitution Check gate present;
     no outdated principle names referenced
  ✅ .specify/templates/spec-template.md — no constitution references;
     no updates required
  ✅ .specify/templates/tasks-template.md — task phases generic; no
     principle-specific task types that conflict
  ⚠ .specify/templates/commands/ — directory does not exist; no command
     files to update

Follow-up TODOs:
  - None; all fields resolved from repo context and CLAUDE.md
-->

# Valravn Constitution

## Core Principles

### I. Evidence Integrity (NON-NEGOTIABLE)

Files residing in `/cases/`, `/mnt/`, `/media/`, or any `evidence/`
directory MUST NEVER be modified, written to, or deleted under any
circumstance. Every operation against acquired evidence MUST be
read-only. Chain of custody MUST be preserved at all times. Any tool
invocation that could alter evidence state is prohibited regardless of
operational convenience.

**Rationale**: Court admissibility depends entirely on demonstrable
evidence integrity. A single write to an evidence mount invalidates
the chain of custody and may render findings inadmissible.

### II. Deterministic Execution

All forensic conclusions MUST be grounded in raw, unmodified tool
output. Hallucination, assumption, or fabrication of artifacts, file
contents, timestamps, or system states is strictly prohibited.
Court-vetted CLI tools are the sole authoritative source of fact.
Every claim MUST cite the specific tool invocation and output that
produced it. On tool failure: read stderr → hypothesize root cause →
apply correction → retry. Failure MUST be reported; silent failure is
prohibited.

**Rationale**: Forensic testimony requires reproducibility. Fabricated
or inferred facts cannot survive cross-examination.

### III. Skill-First Routing

Before executing any forensic utility, the designated skill file for
that domain MUST be consulted. Skill files govern tool invocation
syntax, parameter selection, output interpretation, and known caveats.
Ad-hoc tool invocations that bypass skill guidance are prohibited.

| Domain | Skill |
|--------|-------|
| Timeline generation (Plaso) | `plaso-timeline` |
| File system & carving (Sleuth Kit) | `sleuthkit` |
| Memory forensics (Volatility 3 / Memory Baseliner) | `memory-analysis` |
| Windows artifacts (EZ Tools / Event Logs / Registry) | `windows-artifacts` |
| Threat hunting & IOC sweeps (YARA / Velociraptor) | `yara-hunting` |

**Rationale**: Skill files encode accumulated tool-specific knowledge
and prevent systematic errors (e.g., invoking Volatility 2 when
Volatility 3 is required, or using incorrect EZ Tools paths).

### IV. Output Discipline

All scripts, CSVs, JSON, timelines, and reports MUST be written
exclusively to `./analysis/`, `./exports/`, or `./reports/` relative
to the working case directory. Writing to `/`, evidence directories,
or any path outside the designated output tree is prohibited.
All timestamps in all output artifacts MUST be expressed in UTC.

**Rationale**: Controlled output routing ensures that working files
are never commingled with evidence, and that analysts can locate all
derived artifacts in predictable locations.

### V. Autonomous Execution

Every forensic workflow MUST execute fully autonomously from start to
finish. Check-ins, confirmations, mid-task questions, and "shall I
proceed?" pauses are prohibited. When a decision point is encountered
and no explicit instruction covers it, the most forensically sound
path MUST be selected and the choice MUST be documented in the output.
Final findings are delivered as a completed artifact, not as an
interactive conversation.

**Rationale**: Interruption-driven workflows introduce human error and
lengthen time-to-finding. Documented autonomous decisions remain
auditable without requiring interactive approval.

## Tool Configuration

Authoritative tool paths and invocation conventions for this
workstation. These MUST be used exactly as specified; aliases or
alternative paths are prohibited unless listed here.

| Tool | Invocation | Notes |
|------|-----------|-------|
| Volatility 3 | `python3 /opt/volatility3-2.20.0/vol.py` | Do NOT use `/usr/local/bin/vol.py` (that is Vol2) |
| Memory Baseliner | `python3 /opt/memory-baseliner/baseline.py` | |
| EZ Tools (root) | `dotnet /opt/zimmermantools/<Tool>.dll` | .NET runtime only; no SDK present |
| EZ Tools (subdir) | `dotnet /opt/zimmermantools/<Subdir>/<Tool>.dll` | e.g., `EvtxeCmd/EvtxECmd.dll` |
| YARA | `/usr/local/bin/yara` (v4.1.0) | |
| Sleuth Kit | `fls`, `icat`, `ils`, `blkls`, `mactime`, `tsk_recover` | System PATH |
| EWF tools | `ewfmount`, `ewfinfo`, `ewfverify` | System PATH |
| Plaso | `log2timeline.py`, `psort.py`, `pinfo.py` | GIFT PPA v20240308 |
| bulk_extractor | `bulk_extractor` (v2.0.3) | Defaults to 4 threads |
| photorec | `sudo photorec` | File carving by signature |
| dotnet runtime | `/usr/bin/dotnet` (v6.0.36) | Runtime only — `dotnet --version` will error |

**Not available on this instance**: MemProcFS, VSCMount (Windows-only).

EZ Tools MUST use native .NET over WINE. GUI tools (TimelineExplorer,
RegistryExplorer) require WINE or the Windows analysis VM.

Shell aliases available via `.bash_aliases`:

```bash
vss_carver              # sudo python /opt/vss_carver/vss_carver.py
vss_catalog_manipulator
lr                      # getfattr -Rn ntfs.streams.list (list NTFS ADS)
workbook-update         # update FOR508 workbook
```

## Forensic Workflow

The canonical execution sequence for any forensic engagement:

1. **Evidence intake** — verify hash integrity of all acquired images
   before any analysis begins.
2. **Skill selection** — identify the forensic domain(s) required and
   load the appropriate skill file(s) per Principle III.
3. **Tool execution** — invoke tools per skill guidance; capture all
   stdout/stderr to `./analysis/`.
4. **Output validation** — verify each tool completed successfully;
   on failure apply Principle II error-handling loop.
5. **Report generation** — produce findings artifact in
   `./reports/`; cite every conclusion with its source tool output.

All intermediate artifacts MUST remain in `./analysis/` until the
final report is produced. Temporary files MUST NOT be written to
evidence directories.

## Governance

This constitution supersedes all ad-hoc practices, verbal agreements,
and tool-specific documentation where they conflict. Compliance is
mandatory, not advisory.

**Amendment procedure**:
- Amendments MUST be documented with version bump, change rationale,
  and list of affected principles or sections.
- Version numbering follows semantic versioning:
  - MAJOR: Backward-incompatible governance change, principle removal,
    or redefinition that changes enforcement.
  - MINOR: New principle, new section, or material expansion of
    existing guidance.
  - PATCH: Clarification, wording improvement, or typo fix with no
    semantic change.
- `LAST_AMENDED_DATE` MUST be updated to the amendment date.
- `RATIFICATION_DATE` is immutable after initial ratification.

**Compliance review**:
- Every plan MUST include a Constitution Check gate (see
  `plan-template.md`) that verifies no principle is violated before
  Phase 0 research begins and again after Phase 1 design.
- Any complexity that appears to violate a principle MUST be justified
  in the Complexity Tracking table of the relevant plan.

**Version**: 1.0.0 | **Ratified**: 2026-04-04 | **Last Amended**: 2026-04-04
