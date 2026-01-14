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
- If tests tie, use tie-breakers to pick a preferred vs non-preferred rollout, or mark as no-contrast and increase rollouts/temperature.
- Write one JSONL line per task into `--out`.
- Store trace IDs in metadata for debugging.

## Pilot Dataset Format (JSONL)

Each line is a preference pair using Together's required format.

```
{
  "input": {
    "messages": [
      {"role": "system", "content": "<system prompt>"},
      {"role": "user", "content": "GOAL:\nFix quicksort so tests pass."}
    ]
  },
  "preferred_output": [
    {"role": "assistant", "content": "{\"type\":\"final\",\"summary\":\"...\",\"changes\":[...],\"test_result\":{...}}"}
  ],
  "non_preferred_output": [
    {"role": "assistant", "content": "{\"type\":\"final\",\"summary\":\"...\",\"changes\":[...],\"test_result\":{...}}"}
  ]
}
```

We'll also write a separate metadata file (`dpo_dataset_meta.jsonl`) for debugging:

```
{
  "task_id": "fix_quicksort",
  "suite": "my_suite",
  "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
  "temperature": 0.7,
  "seed": 42,
  "scores": {"preferred": 1.0, "non_preferred": 0.0},
  "tests_ok": {"preferred": true, "non_preferred": false},
  "trace_ids": {"preferred": "run_abc", "non_preferred": "run_xyz"},
  "rollout_counts": {"total": 4}
}
```

Notes:
- For the pilot, we only store the final assistant response per rollout.
- Tool-call transcripts would require multi-turn SFT or custom reward model.
- Together format requires exactly one assistant message per output array.
- If we later do SFT to improve tool calls, serialize tool calls/results inside user/assistant text to fit Together's format.

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

Note: We do not change the DPO loss; we only decide which rollout is preferred when building pairs.

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

## Confirmed Items

### ✅ Together DPO format

Together uses different field names than standard DPO datasets:
- `input.messages` (not `prompt`)
- `preferred_output` (not `chosen`)
- `non_preferred_output` (not `rejected`)
- Both outputs must be arrays with exactly one assistant message

Source: https://docs.together.ai/docs/preference-fine-tuning

### ✅ Seed parameter supported

Together's Chat Completions API supports `seed` parameter for reproducibility.
Use `seed + rollout_index` to get deterministic but varied rollouts.

Source: https://docs.together.ai/reference/chat-completions-1

### ✅ Final-only for pilot

Together's DPO format expects a single assistant message per output, which aligns
with our "final response only" approach. Tool-call transcripts would require
multi-turn SFT or a custom reward model—out of scope for the pilot.

## Pilot sizing (recommended)

For an end-to-end validation run, aim for ~10-30 preference pairs total (for example, 5-10 tasks with 2-4 rollouts each). This is enough to verify data formatting, upload, and training without worrying about cost.

## Open Questions

### Edge case: No contrast in rollouts

What happens when all N rollouts either pass or all fail? Options:
- Skip tasks where `score_preferred == score_non_preferred` (no pair emitted).
- Log these as "no contrast" in a separate file for analysis.
- Require a `--min-gap` threshold to filter pairs where best/worst are too similar.

### Cost estimate

Back-of-envelope estimate before running:
- ~X tokens per rollout (depends on task complexity and max-iters).
- N rollouts × M tasks = total API calls.
- Together pricing for Qwen model (final price reported after tokenization).
