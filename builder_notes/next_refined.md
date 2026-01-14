# Repo-Agent Fine-Tuning Plan (Refined)

## TL;DR

Goal: make a smaller model (Qwen2.5-7B or 14B) behave like a reliable **tool-using repo agent** under `--tool-protocol=json`.

Plan:
1. **Step-level SFT (chosen first)** to teach strict JSON tool-calling + basic “next action” competence.
2. Optional later: **step-level DPO** to reduce loops / wasted calls and improve “patch decision timing”.

Key design choice: **split QuixBugs by task** (not by individual steps) so you can measure generalization.

---

## Current State / Why This Got Tangled

What you’ve already built works: with a strong LLM, the repo agent can solve QuixBugs via tool calls and file patches.

What became difficult (and why it’s not “straightforward”):
- **DPO needs clean preference signal**. Rollout summaries are too weak/noisy as a proxy for patch correctness.
- **Patch-diff DPO is expensive** because “better patch” usually requires executing tests, and rollouts are long.
- You hit an annoying bimodal regime:
  - small models: **can’t follow the tool protocol**, so you don’t even reach patching
  - big models: **patch too well** on QuixBugs, leaving little measurable “delta”
- BugsInPy is a good “harder” benchmark but is operationally heavy (per-bug venvs, Python version constraints, etc.).

So the “clean” reframing is:
- First make the small model **reliably callable as an agent** (tool protocol + basic policy).
- Then (optionally) preference-tune **behavior** once the model can produce valid actions.

---

## Definitions (Important for Data Formatting)

### What `--tool-protocol=json` means in this repo-agent
- The assistant **outputs a single JSON object** in `content`, e.g.:
  ```json
  {"type":"tool_call","name":"read_file","args":{"rel_path":"pysnooper/tracer.py","max_chars":4000}}
  ```
- Tool results show up as **user messages** with a prefix:
  ```
  [tool_result]
  ...tool output...
  ```
- The model is only allowed these tools: `list_files`, `read_file`, `grep`, `write_file`.
- Tests are **driver-run**, not callable by the model. (The driver may append a `[TESTS PASS/FAIL]` snippet after a `write_file` depending on `--test-policy`.)

This matters because:
- Your SFT/DPO data must train the model to emit **the JSON tool-call objects**, not textual “TOOL: pytest” style outputs.

### Together dataset formats (what we actually upload)

**SFT (chat) JSONL:**
- Each line: `{"messages":[{"role":"system|user|assistant","content":"..."}, ...]}`
- Under `--tool-protocol=json`, tool calls are stored as **assistant messages** whose `content` is the JSON tool-call object.
- Tool outputs are stored as **user messages** prefixed with `[tool_result]`.

Minimal SFT example (one step):
```json
{
  "messages": [
    {"role":"system","content":"You are a repo-fixing agent..."},
    {"role":"user","content":"GOAL:\nFix quicksort so tests pass."},
    {"role":"assistant","content":"{\"type\":\"tool_call\",\"name\":\"list_files\",\"args\":{\"rel_dir\":\".\",\"max_files\":200}}"}
  ]
}
```

**DPO JSONL (Together-style):**
```json
{
  "input": {"messages": [...]},
  "preferred_output": [{"role":"assistant","content":"..."}],
  "non_preferred_output": [{"role":"assistant","content":"..."}]
}
```

---

## Paths We Discussed (With Limitations)

### At-a-glance comparison

| Path | What it improves | Data you need | Cost profile | Biggest risk |
|---|---|---:|---|---|
| Step-level **SFT** | tool-call JSON compliance; basic next-action policy | teacher traces → step dataset | $$ | doesn’t move patch correctness much |
| Step-level **DPO** | loop avoidance; efficiency; “patch timing” | step-level preference pairs | $$–$$$ | needs a model that already outputs valid actions |
| **SFT → DPO** | best combined behavior + reliability | both | $$$ | requires careful dataset + more iterations |

### Path 1 — Step-Level SFT (Chosen First)

**Goal:** reduce tool-call parse failures and make a 7B/14B model reliably run inside the agent loop.

**What you train on:** *the next action*, not “one example per task”.
- Each agent iteration becomes a training example: `(conversation_so_far) -> (next tool_call JSON)`.
- This explodes data volume even with only ~50 QuixBugs tasks.

**Why SFT works well here:**
- Tool-protocol compliance is mostly a **format + schema** problem.
- SFT directly maximizes likelihood of the correct JSON shape and correct tool name/args.

**Limitations of SFT:**
- It may not improve deep patch reasoning if the model is genuinely missing capability.
- It may learn to “look correct” without being efficient unless you also include efficiency signals (or follow with DPO).

---

### Path 2 — Step-Level DPO for Agent Behavior (Optional)

**Goal:** improve *policy preferences*:
- fewer loops / fewer wasted tool calls
- better “read enough context” behavior
- better “write_file now vs keep searching” timing

**Important constraint:** Together DPO is effectively “prefer this next assistant message over that one”.
So you almost always implement this as **step-level** pairs (one decision at a time), not whole-rollout “chosen vs rejected”.

**Why DPO is tricky/expensive:**
- If the base model can’t output valid JSON tool calls, your pairs are dominated by garbage → DPO doesn’t learn the intended behavior.
- If you want to preference-train patch quality tied to correctness, you need test-based labels → costly rollouts.

---

### Path 3 — SFT → DPO (Best of Both, If Budget Allows)

This is the “standard agent recipe”:
1. **SFT** to teach the tool protocol + basic trajectories.
2. **DPO** to shape preferences (efficiency, loop avoidance, earlier patching).

This is the path that gives you the cleanest resume story:
- “I built an agent harness”
- “I produced step-level supervised traces”
- “I then applied preference optimization to improve behavioral metrics”

---

## QuixBugs: How Many Tasks Do We Have?

In your local checkout:
- `python_programs/*.py`: **50** tasks
- `python_testcases/test_*.py`: **42** tests (not every program has a testcase)

This is enough for 1K–3K step-level examples with rollouts (even before you add other repos).

---

## Train/Test Split (QuixBugs) — What We Need to Decide

### What are we splitting?
Split **by task**, where a “task” corresponds to a specific buggy program (e.g., `python_programs/quicksort.py`).

Do **not** split by individual steps across the same task, because that leaks:
- file names and paths
- repeated tool outputs
- common patch patterns for that same file

### What should be in the test set?
Test set should be tasks that were **never used** to generate training traces.

Recommended:
- Build the split over the subset that has testcases (≈42 tasks) if you want pass/fail as an evaluation metric.
- You can still include “no-test” tasks in the *training* set for protocol compliance, but they’re weaker for correctness evaluation.

### Recommended split strategy (simple + reproducible)

**Option A (default):** 80/20 split on tasks-with-tests
- Train: 34 tasks
- Test: 8 tasks

**Option B (more stable eval):** 70/30 split on tasks-with-tests
- Train: 29 tasks
- Test: 13 tasks

> Recommendation: start with Option B if you can afford it. A test set of ~8 tasks is noisy; ~13–15 gives you a clearer read on improvement.

### How to make the split deterministic
Pick one deterministic method and commit it to the repo (so runs are comparable):
- Hash-based split on `task_id` (stable across machines), or
- Fixed RNG seed with sorted task list (stable if task list is stable)

Example hash rule:
- `md5(task_id) % 100 < 80` → train, else test.

### What we evaluate (beyond pass/fail)
Even if pass rate doesn’t move much, the SFT project can still be a win if these improve:
- `parse_errors` ↓ (invalid tool-call JSON)
- `% runs that ever call write_file` ↑
- `loop_detections` ↓
- `avg_steps` ↓ (fewer wasted calls)

---

## Concrete Execution Plan (SFT-First)

### 1) Build suites
- `quixbugs_train_suite.json` (train tasks only)
- `quixbugs_test_suite.json` (held-out tasks only)

> Tip: keep a single “master suite” and generate derived train/test suites deterministically from it so you don’t hand-edit lists.

### 2) Generate teacher traces on train suite
Use a strong teacher model with `--tool-protocol=json`.
Budget control knobs:
- `--rollouts N`
- `--max-iters`
- `--test-policy never` (cheaper; focuses on tool protocol)

Example (using `repo-agent prefs` as a rollout+trace generator):
```bash
poetry run repo-agent prefs \
  --suite eval/suites/quixbugs_train_suite.json \
  --rollouts 8 \
  --trace-dir runs/quixbugs_teacher_traces \
  --out runs/quixbugs_teacher_traces/_prefs.jsonl \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.2 \
  --tool-protocol json \
  --test-policy never
```

### 3) Extract step-level SFT dataset
Use `repo-agent sft-extract` to convert traces → `*.jsonl` with `{"messages":[...]}` format.

Filtering choice:
- For protocol compliance SFT, you can keep steps that are valid tool calls even if the rollout eventually fails.

Example (looser filtering for protocol-focused SFT):
```bash
poetry run repo-agent sft-extract \
  --trace-dir runs/quixbugs_teacher_traces \
  --output runs/instruction_tuning/quixbugs_tool_sft.jsonl \
  --no-require-success \
  --no-require-valid-tool-ok \
  --format json
```

### 4) Fine-tune (Together SFT + LoRA)
Train the smaller model on the extracted dataset.

Example:
```bash
poetry run python sft_finetune_quick.py \
  --dataset runs/instruction_tuning/quixbugs_tool_sft.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct-Turbo \
  --suffix qwen25-7b-json-tools-sft \
  --epochs 1 \
  --batch-size max \
  --learning-rate 1e-5 \
  --lora \
  --watch
```

### 5) Evaluate on the held-out test suite
Run `repo-agent eval` on `quixbugs_test_suite.json` and compare baseline vs finetuned:
- tool protocol metrics (parse errors, tool usage)
- optional correctness metrics (pass/fail) if you enable tests

Example:
```bash
poetry run repo-agent eval \
  --suite eval/suites/quixbugs_test_suite.json \
  --trace-dir runs/quixbugs_eval_ft \
  --report runs/quixbugs_eval_ft/report.json \
  --llm-provider together \
  --model <YOUR_FINETUNED_MODEL_NAME> \
  --tool-protocol json \
  --num-workers 4
```

---

## What This Project “Is” (Resume-Friendly Definition)

**Project statement:**
> Train a small open model to act as a reliable tool-using repo-fixing agent by instruction-tuning it on step-level tool-call trajectories, then measure improvements in agent reliability and efficiency on held-out bug-fix tasks.

**Deliverables:**
1. Deterministic QuixBugs train/test split suites
2. Trace → dataset extractor (already present) + documented data format
3. Together SFT fine-tune run (LoRA) + evaluation report
4. (Optional) DPO follow-on to improve loop avoidance / patch timing
