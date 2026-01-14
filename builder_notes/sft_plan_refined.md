# SFT Plan (Refined): Train a Small Model to Be a JSON Tool‑Using Repo Agent

## Scope

This document is **SFT-only**. The objective is to instruction-tune a small model (target: **Qwen2.5‑7B**, optionally 14B) so it can reliably operate inside this repo-agent under `--tool-protocol=json`.

Primary focus:
- **Tool protocol compliance** (valid JSON, correct tool names/args).
- **Basic next-action policy** (productive tool sequencing; reach `write_file` when appropriate).

Non-goal (for this phase):
- Maximizing patch correctness on harder corpora (that’s a later phase once the model is reliably tool-using).

---

## Problem Statement

Smaller models often fail before they even start:
- Invalid JSON tool calls → `parse_errors` → the driver can’t execute tools.
- Hallucinated tool names/args or prose+JSON mixtures → tool execution fails.
- Shallow loops (e.g., repeated `read_file` with tiny `max_chars`) → never reaches `write_file`.

If the model can’t reliably participate in the tool loop, “patch quality” is hard to measure because it never reaches patching.

---

## Proposed Solution: Step‑Level SFT From Teacher Traces

Generate **teacher** tool-using traces on QuixBugs, then extract **step-level** supervised examples where the target is the next JSON tool call.

Key design choice:
- We do **not** need 1K–3K unique tasks.
- We treat **each agent step** as one training example: `(conversation_so_far) → (next JSON tool_call)`.

---

## Hypotheses / Success Criteria

### Hypotheses
- **H1 (Protocol):** SFT reduces `parse_errors` by teaching strict JSON tool-call format.
- **H2 (Trajectory):** SFT improves the rate of “productive” sequences (more useful `grep/read_file`, more runs that reach `write_file`).
- **H3 (Downstream):** On held-out QuixBugs tasks, reliability metrics improve (fewer loops, fewer wasted steps). Pass rate may improve, but is not guaranteed.

### Primary metrics (cheap + reliable)
- `parse_errors` ↓
- `loop_detections` ↓
- `avg_steps` ↓
- `% runs that call write_file at least once` ↑
- `tool_breakdown` becomes more “agent-like” (less thrash, more purposeful reads/greps)

### Secondary metric (optional)
- `success` / pass rate on held-out tasks (requires running tests during eval)

---

## Dataset: Exact Format + Examples

### Dataset format (Together SFT chat JSONL)

One JSON object per line:
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Under `--tool-protocol=json`:
- Tool calls are represented as **assistant messages** whose `content` is exactly one JSON object:
  ```json
  {"type":"tool_call","name":"read_file","args":{"rel_path":"python_programs/quicksort.py","max_chars":2000}}
  ```
- Tool results are represented as **user messages** prefixed with `[tool_result]`:
  ```
  [tool_result]
  ...tool output...
  ```

Important constraint (current implementation):
- `repo-agent sft-extract` currently emits **tool_call steps only** (it does not emit `final` examples).

<question>
<h4>Questions:</h4>

Do we want it to emit `final` examples?

> Answer (Codex):
> - **Not for v1 SFT**, as long as our main goal is tool-protocol compliance + next-action behavior. The highest-leverage improvement is reducing `parse_errors` and getting consistent `tool_call` JSON.
> - But we should **watch for a specific failure mode** after SFT: tests pass (or the model should stop), yet the model never emits `{"type":"final",...}` and runs hit `max_iters`.
> - If that shows up, the fix is straightforward:
> > - add a small number of `final` examples (e.g., 1 per successful rollout) **or**
> > - extend `repo-agent sft-extract` to optionally emit `final` steps from trace `final` events **or**
> > - add a driver policy like “stop when tests pass” (so termination doesn’t depend on the model).
</question>

### Example: first action
```json
{
  "messages": [
    {"role": "system", "content": "You are a repo-fixing agent...\nTOOLS:\n[...]"},
    {"role": "user", "content": "GOAL:\nFix the quicksort implementation in python_programs/quicksort.py so that the tests pass"},
    {"role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"list_files\",\"args\":{\"rel_dir\":\".\",\"max_files\":200}}"}
  ]
}
```

### Example: later action (with tool result in context)
```json
{
  "messages": [
    {"role": "system", "content": "You are a repo-fixing agent...\nTOOLS:\n[...]"},
    {"role": "user", "content": "GOAL:\nFix the quicksort implementation in python_programs/quicksort.py so that the tests pass"},
    {"role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"list_files\",\"args\":{\"rel_dir\":\".\",\"max_files\":200}}"},
    {"role": "user", "content": "[tool_result]\npython_programs/quicksort.py\npython_testcases/test_quicksort.py\n..."},
    {"role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"read_file\",\"args\":{\"rel_path\":\"python_programs/quicksort.py\",\"max_chars\":2000}}"}
  ]
}
```

### How we construct examples (from traces)

`repo-agent sft-extract` builds one step sample by pairing:
1. the most recent `llm_request.payload.messages` (the prompt the model saw),
2. the subsequent `llm_action` tool call (must be a valid `{"type":"tool_call",...}`),
3. and requiring a corresponding `tool_result` event (so the action actually executed).

<question>
<h4>Questions:</h4>
So we're filtering on only succesfull tool calls and removing rollouts with loop events, and we're only including PASS? I'm actually not clear again on how we prevent the situation where we contaminate the training data with had behaviours we don't want to model to learn. Even if we do the aforementioned filtering, the training data could still include tool calls that were valid but not suboptimal. Can we somehow remove those are is that something we just have to deal with and nbd?

> Answer (Codex):
> - Two clarifications:
> > 1) “Only successful tool calls” = `tool_result.ok == true` (step-level) via `--require-valid-tool-ok`.
> > 2) “Only PASS rollouts” = `last_test.ok == true` (run-level) via `--require-success` (this is our **default starting point** for cleaner targets).
>
> - We don’t need perfect “optimality labels” for v1 because the core objective is **tool-call validity + basic agent behavior**. Some “valid but slightly suboptimal” steps are acceptable noise in SFT.
>
> - What we *do* want to avoid teaching are systematic bad habits (thrash/loops). The practical strategy is:
> > - keep strict schema + `tool_result.ok` filters,
> > - drop steps after loop detection (or after obvious repeated calls),
> > - and, if needed, keep only the **best successful** rollouts per task (e.g., shortest PASS) instead of all PASS rollouts.
>
> - If you want to be more aggressive about removing “valid but suboptimal” tool calls without adding a judge model, the best heuristic is:
> > - per task, select top‑K successful rollouts by `steps` (or by “no loop detections”), and only extract SFT samples from those.
>
> Bottom line: for this phase it’s “nbd” as long as we filter out thrash and measure post-extract dataset quality; DPO (or more sophisticated filtering) is where you really optimize policy preferences.
</question><br>

Implementation reference: `src/llm_repo_agent/sft/extract.py`.

---

## Data Quality Decisions (From QA)

### Step-level invariants (what we keep)

**Hard requirements (default):**
- Target action is a valid JSON tool call:
  - `type == "tool_call"`
  - `name ∈ {list_files, read_file, grep, write_file}`
  - required arg keys present (per tool schema)
- Tool execution succeeded: `tool_result.ok == true`
  - This is enforced by `repo-agent sft-extract --require-valid-tool-ok` (default True).
- Tool outputs are truncated to a bounded size (`--max-context-chars`) to control sequence length.

**Anti-thrash filter (decision):**
- Do not train on obviously looping steps.
- Practical rule for v1: **drop steps after the first “Loop detected…” driver note** (and optionally drop exact repeats).
  - This is not implemented in the current extractor yet, but it is a clear next improvement if needed.

### Rollout-level invariants (what we keep)

We distinguish two training intents:

1) **Protocol compliance SFT (cheapest, highest leverage early)**
- We do **not** require the whole rollout to pass tests.
- We keep “good prefix” steps as long as they satisfy step-level invariants above.

2) **Policy imitation SFT (closer to “solve QuixBugs”)**
- Prefer to train on **successful rollouts** (tests pass) to avoid learning teacher mistakes.
- This is enforced by `repo-agent sft-extract --require-success` (default True).

**Decision for the first iteration:**
- Start with `--require-success` **enabled** (cleaner targets) and `--test-policy on_write` (see below).
- If dataset volume is too low, relax to `--no-require-success` but keep the step-level invariants and anti-thrash filtering.

---

## QuixBugs Task Split (Train/Test)

Why we split by task:
- Splitting by step leaks file paths and tool outputs across train/test and makes evaluation meaningless.

QuixBugs in your checkout:
- `python_programs/*.py`: 50 buggy programs
- `python_testcases/test_*.py`: 42 test files (best subset for pass/fail eval)

**Decision: split on the ~42 tasks that have testcases** so the eval metric “pass/fail” is available on held-out tasks.

Recommended split (pick one and stick to it):
- 70/30 for more stable eval, or
- 80/20 for more train data.

Make it deterministic (so comparisons are fair):
- hash-based split on `task_id`, or
- seeded shuffle with a fixed seed.

Artifacts:
- `eval/suites/quixbugs_train_suite.json`
- `eval/suites/quixbugs_test_suite.json`

---

## Teacher Trace Generation (How + Why)

### Why `--test-policy on_write` helps (even though it doesn’t guarantee pass)

When `--test-policy on_write` is enabled:
- After `write_file`, the driver runs tests and injects `[TESTS PASS/FAIL]` output into the next prompt context.
- A strong teacher is then more likely to produce “test-driven” corrective trajectories:
  - smaller, targeted edits,
  - grounded follow-up tool calls based on failure output,
  - iterative fixes instead of stopping after a single guess.

This tends to produce better traces to imitate (especially if you later filter to successful rollouts).

### Variance / temperature decisions

Concern: many rollouts may look similar on easy tasks (e.g., `fix_gcd`).

Decisions:
- We do not rely on high variance within a single easy task; diversity comes primarily from **many tasks** and different tool outputs.
- Use a small non-zero teacher temperature to allow alternate but still-correct routes:
  - **Teacher temperature: ~0.2** (safe range: 0.1–0.3).
  - At `temperature=0.0`, changing seeds may have little effect; with `temperature>0`, seeds matter and variance increases.
- If you still need diversity, prefer:
  - adding more tasks (expand from 30 → 42 tasks-with-tests),
  - allocating more rollouts only to harder tasks,
  - optionally using a second teacher model,
  - and/or deduplicating the final dataset.

---

## Pipeline: End-to-End Commands

### 1) Generate teacher rollouts + traces (train suite)

We can use `repo-agent prefs` as a rollout runner (it already runs multiple seeds and writes traces).

```bash
poetry run repo-agent prefs \
  --suite eval/suites/quixbugs_train_suite.json \
  --rollouts 8 \
  --trace-dir runs/quixbugs_teacher_traces \
  --out runs/quixbugs_teacher_traces/_prefs.jsonl \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.2 \
  --seed 42 \
  --max-workers 5 \
  --tool-protocol json \
  --test-policy on_write
```

Cheaper alternative (protocol-only traces):
- Use `--test-policy never`, then extract with `--no-require-success`.

### 2) Extract step-level SFT dataset from traces

Default (cleaner targets, success-only):
```bash
poetry run repo-agent sft-extract \
  --trace-dir runs/quixbugs_teacher_traces \
  --output runs/instruction_tuning/quixbugs_tool_sft_train.jsonl \
  --format json
```

If you need more volume (allow failed rollouts, still requires `tool_result.ok` by default):
```bash
poetry run repo-agent sft-extract \
  --trace-dir runs/quixbugs_teacher_traces \
  --output runs/instruction_tuning/quixbugs_tool_sft_train.jsonl \
  --format json \
  --no-require-success
```

### 3) Dataset volume checkpoint (decision)

After extraction:
- Count lines in `runs/instruction_tuning/quixbugs_tool_sft_train.jsonl`.
- Optionally deduplicate by hashing `messages` and re-count.

If under target (2k–3k examples):
- add tasks (expand toward all 42 tasks-with-tests),
- increase rollouts only for harder tasks,
- add a second teacher model,
- or slightly increase teacher temperature within 0.1–0.3.

### 4) Fine-tune (Together SFT + LoRA)

```bash
poetry run python sft_finetune_quick.py \
  --dataset runs/instruction_tuning/quixbugs_tool_sft_train.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct-Turbo \
  --suffix qwen25-7b-json-tools-sft \
  --epochs 1 \
  --batch-size max \
  --learning-rate 1e-5 \
  --lora \
  --watch
```

### 5) Evaluate on held-out QuixBugs tasks

```bash
poetry run repo-agent eval \
  --suite eval/suites/quixbugs_test_suite.json \
  --trace-dir runs/quixbugs_eval_ft \
  --report runs/quixbugs_eval_ft/report.json \
  --llm-provider together \
  --model <YOUR_FINETUNED_MODEL_NAME> \
  --tool-protocol json \
  --num-workers 4 \
  --test-policy on_write
```

---

## Expected Dataset Size (Step-Level)

Approximation:
```
num_samples ≈ (#train_tasks) × (rollouts_per_task) × (avg_valid_tool_calls_per_rollout)
```

Example:
- 29 train tasks (70% of 42)
- 8 rollouts each
- ~10 valid tool calls per rollout
→ ~2320 samples

---

## Known Limitations / Risks (SFT-Only)

- The extractor emits tool-call steps only; if `final` behavior matters, we may need to add final-step SFT examples.
- `write_file` requires outputting full file content; it works for QuixBugs-sized files but doesn’t scale as well as patch-based tools.
- SFT may make the model more reliable as an agent without making it much “smarter” at patching; that’s expected and acceptable for phase 1.
