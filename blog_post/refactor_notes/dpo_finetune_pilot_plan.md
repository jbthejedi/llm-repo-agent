# DPO Finetuning Pilot Plan

This document describes the planned changes for a minimal DPO finetuning pilot
using Together + Qwen. The goal is to validate the end-to-end training workflow
before building a full preference dataset pipeline or adding RAG.

## Goals

- Validate Together DPO finetuning with a Qwen model.
- Produce a small, reliable preference dataset from existing eval runs.
- Compare baseline vs tuned performance using the existing eval harness.

## Non-goals (for the pilot)

- No repo-level RAG.
- No large-scale dataset generation.
- No local model serving or vLLM integration.

## Proposed CLI Additions

New subcommand to generate preference data only:

```
repo-agent prefs \
  --suite eval/suites/my_suite.json \
  --rollouts 4 \
  --out runs/prefs/dpo_dataset.jsonl \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.7 \
  --seed 42 \
  --max-iters 20 \
  --test-policy on_write
```

### Behavior

- For each task in the suite, run N rollouts (independent attempts).
- Score each rollout using tests as the primary signal.
- Select the best and worst rollout to form a preference pair.
- Write one JSONL line per task into `--out`.
- Store trace IDs in metadata for debugging.

## Pilot Dataset Format (JSONL)

Each line is a preference pair with chat-completions message content.

```
{
  "prompt": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "GOAL:\nFix quicksort so tests pass."}
  ],
  "chosen": {
    "role": "assistant",
    "content": "{\"type\":\"final\",\"summary\":\"...\",\"changes\":[...],\"test_result\":{...}}"
  },
  "rejected": {
    "role": "assistant",
    "content": "{\"type\":\"final\",\"summary\":\"...\",\"changes\":[...],\"test_result\":{...}}"
  },
  "meta": {
    "task_id": "fix_quicksort",
    "suite": "my_suite",
    "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "temperature": 0.7,
    "seed": 42,
    "scores": {"chosen": 1.0, "rejected": 0.0},
    "tests_ok": {"chosen": true, "rejected": false},
    "trace_ids": {"chosen": "run_abc", "rejected": "run_xyz"},
    "rollout_counts": {"total": 4}
  }
}
```

Notes:
- For the pilot, we only store the final assistant response per rollout.
- Tool-call transcripts can be added later for a richer dataset.

## Scoring and Pair Selection

Primary signal:
- Tests pass => score 1.0
- Tests fail => score 0.0

Tie-breakers (applied in order if needed):
- Fewer iterations
- Fewer tool calls
- Fewer files touched
- Smaller diff (if available)

Pairing:
- Best score becomes `chosen`
- Worst score becomes `rejected`

## Implementation Plan (Code Changes)

### New modules

`src/llm_repo_agent/prefs/`
- `rollouts.py`: run N attempts per task with configurable temperature/seed.
- `score.py`: compute scores from test outcomes and run metadata.
- `pairs.py`: select best/worst and emit preference pairs.
- `schema.py` (optional): small helpers for JSONL formatting.

### CLI wiring

`src/llm_repo_agent/main.py`
- Add a `prefs` subcommand with the flags shown above.
- Reuse existing eval suite loading and runner utilities.

### Eval reuse

`src/llm_repo_agent/eval/runner.py`
- Expose a lightweight API to run a single task with a given config and return:
  - final action content
  - test result summary
  - tool/iteration counts
  - run_id (for trace linkage)

### Traces and metadata

- Store trace IDs in the JSONL metadata.
- No new trace event kinds required for the pilot.

## Training Step (Outside the Codebase)

For the pilot, DPO training can be run directly via Together using their
dataset format and training CLI/API. The only required output from the code is
`dpo_dataset.jsonl`. The training job returns a model ID that we can use for eval.

## Evaluation (Before/After)

Run eval twice with the existing CLI:

- Baseline: `--model Qwen/Qwen2.5-72B-Instruct-Turbo`
- Tuned: `--model <together_dpo_model_id>`

Compare pass rate, average steps, and tool calls to demonstrate improvement.

## Risks and Open Questions

- Confirm Together DPO format and required fields for preference data.
- Confirm whether the pilot should include tool-call transcripts or final-only.
- Ensure deterministic behavior with `--seed` is supported by the model/provider.

## Additional Considerations

### Edge case: No contrast in rollouts

What happens when all N rollouts either pass or all fail? There's no meaningful
preference signal in that case. Options:
- Skip tasks where `score_chosen == score_rejected` (no pair emitted).
- Log these as "no contrast" in a separate file for analysis.
- Require a `--min-gap` threshold to filter pairs where best/worst are too similar.

### Seed variation across rollouts

If `--seed 42` applies identically to all rollouts, outputs may be identical
(depending on provider behavior). Consider:
- `seed + rollout_index` for reproducible variation.
- Random seed per rollout (less reproducible but guaranteed diversity).

### Together format verification

The `chosen`/`rejected` structure looks standard, but Together may expect
different field names (e.g., `chosen_messages` vs `chosen`). Verify against
their docs before implementing.

### Assistant content encoding

The plan shows JSON-stringified content in `chosen.content`. Confirm this is
what Together expects—most DPO setups use natural language strings rather than
serialized JSON.

### Target sample size

Even a rough target (50 pairs? 200?) would help scope the pilot:
- Informs how many suite tasks to include.
- Helps estimate cost (N rollouts × M tasks × tokens-per-rollout).

### Cost estimate

Back-of-envelope estimate before running:
- ~X tokens per rollout (depends on task complexity and max-iters).
- N rollouts × M tasks = total API calls.
- Together pricing for Qwen model.
