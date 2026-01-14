# SFT Data Extraction Command - Implementation Plan

## Goal
Create a CLI command `repo-agent sft-extract` that parses trace logs from rollout runs and produces a step-level SFT dataset in Together/OpenAI chat format.

---

## Overview

Parse JSONL trace files to extract step-level training samples where:
- **Input**: system prompt + goal + conversation history up to that point
- **Target**: the next tool call JSON the assistant produced

One successful trajectory with 6 tool calls → 6 SFT training samples.

---

## File Structure

```
src/llm_repo_agent/
├── sft/
│   ├── __init__.py
│   ├── extract.py      # Core extraction logic
│   └── config.py       # SFTExtractConfig dataclass
└── main.py             # Add sft-extract subcommand
```

---

## Step 1: Create SFT Config Dataclass

**File**: `src/llm_repo_agent/sft/config.py`

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class SFTExtractConfig:
    """Configuration for SFT data extraction."""
    trace_dir: Path                    # Directory containing trace JSONL files
    output_path: Path                  # Output JSONL file path
    require_success: bool = True       # Only include successful runs (tests passed)
    require_valid_tool_ok: bool = True # Only include steps where tool_result.ok=True
    max_context_chars: int = 8000      # Truncate long tool outputs in context
    include_recovery: bool = False     # Include failed→recovery step pairs (future)
    progress: bool = True              # Show progress output
```

---

## Step 2: Create Extraction Module

**File**: `src/llm_repo_agent/sft/extract.py`

### Key Functions:

```python
def extract_sft_samples(cfg: SFTExtractConfig) -> List[Dict]:
    """Main entry point - extract samples from all traces in directory."""

def parse_single_trace(trace_path: Path, cfg: SFTExtractConfig) -> List[Dict]:
    """Parse one JSONL trace file into step-level samples."""

def is_run_successful(events: List[Dict]) -> bool:
    """Check if run ended with tests passing."""
    # Look for run_end.payload.state.last_test.ok == True
    # Or final.payload.final.test_result.ok == True

def extract_steps(events: List[Dict], cfg: SFTExtractConfig) -> List[Dict]:
    """Extract step-level samples from a successful run."""

def format_sft_sample(
    system_prompt: str,
    user_goal: str,
    context_messages: List[Dict],
    tool_call_json: str
) -> Dict:
    """Format a single SFT sample in Together chat format."""
```

### Extraction Logic:

1. **Load all events** from trace JSONL
2. **Check success** via `run_end` or `final` event
3. **If successful**, iterate through events:
   - Track `llm_request` events to get current message context
   - For each `llm_action` with type="tool_call":
     - Check if subsequent `tool_result.ok == True` (if require_valid_tool_ok)
     - Emit sample: `{messages: [system, goal, ...context, assistant_tool_call]}`
4. **Truncate** long tool outputs in context (grep results, file contents)

### Trace Event Flow:

```
run_start → llm_request → llm_action(tool_call) → tool_result →
          → llm_request → llm_action(tool_call) → tool_result →
          → ... →
          → llm_action(final) → tests → run_end
```

### Output Format (Together Chat JSONL):

```json
{"messages": [
  {"role": "system", "content": "You are a code-fixing agent..."},
  {"role": "user", "content": "GOAL:\nFix the bug in quicksort.py..."},
  {"role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"list_files\",\"args\":{\"rel_dir\":\".\",\"max_files\":50}}"}
]}
```

Each subsequent step includes prior assistant/user turns as context.

---

## Step 3: Add CLI Command

**File**: `src/llm_repo_agent/main.py`

```python
# Add subparser
sft_parser = subparsers.add_parser(
    "sft-extract",
    help="Extract SFT training data from trace logs"
)
sft_parser.add_argument(
    "--trace-dir", type=Path, required=True,
    help="Directory containing trace JSONL files (e.g., runs/prefs_cost_estimate_pilot)"
)
sft_parser.add_argument(
    "--output", type=Path, required=True,
    help="Output JSONL file path for SFT dataset"
)
sft_parser.add_argument(
    "--require-success", action="store_true", default=True,
    help="Only include runs where tests passed (default: True)"
)
sft_parser.add_argument(
    "--no-require-success", dest="require_success", action="store_false",
    help="Include all runs regardless of test outcome"
)
sft_parser.add_argument(
    "--max-context-chars", type=int, default=8000,
    help="Max chars for tool output in context (default: 8000)"
)
sft_parser.add_argument(
    "--progress", action="store_true", default=True,
    help="Show progress output"
)
sft_parser.set_defaults(func=cmd_sft_extract)

def cmd_sft_extract(args):
    from llm_repo_agent.sft.extract import extract_sft_samples
    from llm_repo_agent.sft.config import SFTExtractConfig

    cfg = SFTExtractConfig(
        trace_dir=args.trace_dir,
        output_path=args.output,
        require_success=args.require_success,
        max_context_chars=args.max_context_chars,
        progress=args.progress,
    )

    samples = extract_sft_samples(cfg)

    # Write output
    with open(cfg.output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"[sft-extract] Wrote {len(samples)} samples to {cfg.output_path}")
```

---

## Step 4: Filtering Rules

Per `sft_it_plan.md`, apply strict filtering:

1. **Successful runs only** (by default): `run_end.payload.state.last_test.ok == True`
2. **Valid tool calls only**: Skip steps where:
   - Tool call JSON is malformed
   - Required args missing (e.g., `rel_dir` for `list_files`)
   - `tool_result.ok == False` due to invalid args
3. **Known tools only**: `list_files`, `read_file`, `grep`, `write_file`
4. **Truncate context**: Cap tool output at `max_context_chars` to control token costs

---

## Step 5: Progress and Stats Output

```
[sft-extract] Scanning runs/prefs_cost_estimate_pilot...
[sft-extract] Found 45 trace files
[sft-extract] Processing fix_quicksort_abc123.jsonl... 6 steps
[sft-extract] Processing fix_mergesort_def456.jsonl... SKIP (tests failed)
[sft-extract] Processing fix_gcd_ghi789.jsonl... 4 steps
...
[sft-extract] ============================================================
[sft-extract] Summary:
[sft-extract]   Traces scanned: 45
[sft-extract]   Successful runs: 32
[sft-extract]   Skipped (tests failed): 13
[sft-extract]   Total SFT samples: 187
[sft-extract]   Avg steps per run: 5.8
[sft-extract] ============================================================
[sft-extract] Wrote 187 samples to sft_data.jsonl
```

---

## Example Usage

```bash
# Extract from existing trace logs
poetry run repo-agent sft-extract \
    --trace-dir runs/prefs_cost_estimate_pilot \
    --output data/sft_dataset.jsonl

# Include failed runs too (for debugging)
poetry run repo-agent sft-extract \
    --trace-dir runs/prefs_cost_estimate_pilot \
    --output data/sft_dataset_all.jsonl \
    --no-require-success
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/llm_repo_agent/sft/__init__.py` | Create (empty or exports) |
| `src/llm_repo_agent/sft/config.py` | Create (SFTExtractConfig) |
| `src/llm_repo_agent/sft/extract.py` | Create (extraction logic) |
| `src/llm_repo_agent/main.py` | Modify (add sft-extract subcommand) |

---

## Future Enhancements (Not in Initial Scope)

1. **Recovery examples**: Include steps where tool fails → model retries successfully
2. **Deduplication**: Skip near-identical samples across runs
3. **Train/eval split**: Auto-split output by task_id for held-out evaluation
4. **Validation**: Verify output parses correctly before writing
