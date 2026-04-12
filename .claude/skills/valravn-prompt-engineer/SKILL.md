---
name: valravn-prompt-engineer
description: >
  Design, refine, or debug system prompts and LLM instructions in Valravn nodes.
  Use this skill whenever you are editing _SYSTEM_PROMPT constants, correction context strings,
  changing JSON output schemas in prompts, improving DFIR tool syntax instructions,
  or working on hallucination problems in the planning or correction nodes.
  Also triggers for: "improve the prompt", "LLM is hallucinating", "wrong tool syntax",
  "refine the planner", "fix log2timeline flags", "better forensic instructions".
---

# Prompt Engineering Skill

## Prompt Locations

| Node | File | Prompt constant |
|------|------|----------------|
| Planner | `nodes/plan.py` | `_SYSTEM_PROMPT` |
| Tool correction | `nodes/tool_runner.py` | `_CORRECTION_CONTEXT` |
| Self-assessment | `nodes/self_assess.py` | inline in function |
| Anomaly detection | `nodes/anomaly.py` | inline in function |
| Conclusions | `nodes/conclusions.py` | inline in function |

## The Central Problem

The LLM must produce exact, executable SIFT tool commands. Hallucinated flags cause subprocess failures. The prompts are the primary defence.

## SIFT Tool Syntax (authoritative)

```
log2timeline.py (Plaso 20240308, GIFT PPA)
  Flags MUST come BEFORE the source path. Source path is THE LAST argument.
  Valid flags (from --help): --storage-file, --parsers, --timezone, --workers, --status-view
  Hallucinated flags that WILL error: --log2timeline-mode, --mode, --output, --format, --output-format
  CORRECT: log2timeline.py --storage-file ./analysis/case.plaso --parsers win10 --timezone UTC /mnt/ewf/ewf1
  WRONG:   log2timeline.py /mnt/ewf/ewf1 --storage-file ./analysis/case.plaso

fls (Sleuth Kit)
  CORRECT: fls [-r] [-m /] <image_or_device>

icat (Sleuth Kit)
  CORRECT: icat <image_or_device> <inode_number>

vol.py (Volatility 3, v2.20.0)
  CORRECT: python3 /opt/volatility3-2.20.0/vol.py -f <memory_image> <plugin.PluginName>

yara
  CORRECT: /usr/local/bin/yara [-r] <rules_file.yar> <target_path>
```

## Prompt Quality Checklist

When editing a prompt, verify:
- [ ] Output is JSON-only — "no markdown fences, no prose outside the JSON object" is stated explicitly
- [ ] Output schema is written out in full with field names and types
- [ ] Tool hallucination blocklist is present (especially for log2timeline.py)
- [ ] Path discipline is stated: "use `{analysis_dir}` for ALL output files, never /tmp/"
- [ ] Skill domain enum is explicit: `memory-analysis | sleuthkit | windows-artifacts | plaso-timeline | yara-hunting`
- [ ] Correction prompts include "do NOT change the source path" rule

## Making Changes

1. Read the current prompt constant
2. Identify the specific failure mode (wrong flag, wrong path, wrong JSON field name)
3. Add a concrete CORRECT/WRONG example pair (not just a rule statement — examples are more effective)
4. Run the evaluation suite to confirm improvement:
   ```bash
   python -m valravn.evaluation.evaluators --suite all
   ```
   Run before and after the change. Compare MLflow metrics (logged to `http://127.0.0.1:5000`).

## JSON Schema Discipline

The planner output schema:
```json
{
  "steps": [
    {
      "skill_domain": "<one of: memory-analysis | sleuthkit | windows-artifacts | plaso-timeline | yara-hunting>",
      "tool_cmd": ["<executable>", "<arg1>", "<arg2>"],
      "rationale": "<why this step addresses the investigation prompt>"
    }
  ]
}
```

The correction output schema:
```json
{"corrected_cmd": ["<executable>", "<arg1>", ...], "rationale": "<why this fixes the error>"}
```

Both schemas must be stated verbatim in their respective prompts.

## Prompt Iteration Pattern

```
Before:
  "Use tools available on SIFT"

After:
  "Use ONLY tools with these exact invocation forms: [list]
   Do NOT use these flags (they do not exist): [list]
   CORRECT example: [example]
   WRONG example: [example]"
```

Specificity beats generality. The LLM needs to see what "wrong" looks like to avoid it.
