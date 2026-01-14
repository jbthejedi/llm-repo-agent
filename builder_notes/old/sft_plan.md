# SFT Plan: Train a Small Model to Be a JSON Tool-Using Repo Agent

## Problem Statement

We want a smaller, affordable model (target: **Qwen2.5-7B**, optionally 14B) to behave reliably as a **tool-using repo agent** under `--tool-protocol=json`.

Today, smaller models often fail before they even start:
- They don’t emit valid JSON tool-call objects consistently → high `parse_errors`.
- They hallucinate tool names/args or mix prose + JSON → driver can’t execute tools.
- They loop on shallow evidence (e.g., repeated `read_file` with tiny `max_chars`) and never reach `write_file`.

This makes “patch quality” hard to even measure, because the model can’t reliably enter the tool-using loop.

---

## Proposed Solution (Path 1 Only): Step-Level SFT

**Approach:** generate **teacher** tool-using traces on QuixBugs, then extract **step-level** supervised examples where the target is the next JSON tool-call.

Key idea: we do **not** need 1K–3K unique bugs. We treat **each agent step** as a training example.

---

## Hypotheses / Success Criteria

### Hypotheses
- **H1 (Protocol):** After SFT, the model produces valid `{"type":"tool_call",...}` JSON far more often → `parse_errors` decreases.
- **H2 (Trajectory):** After SFT, the model makes more “legal + productive” tool calls and is more likely to reach `write_file`.
- **H3 (Downstream):** On held-out QuixBugs tasks, agent reliability metrics improve (fewer loops, fewer wasted steps). Pass rate may improve, but is not guaranteed.

### Primary metrics (cheap + reliable)
- `parse_errors` (from traces) ↓
- `loop_detections` ↓
- `avg_steps` ↓
- `% runs that call write_file at least once` ↑
- `tool_breakdown` shifts toward productive sequences (more `read_file`/`grep` early, `write_file` when localized)

### Secondary metric (optional)
- `success` / pass rate on held-out tasks (requires running tests during eval)

---

## Data: What the SFT Dataset Looks Like (Exactly)

### Format: Together SFT chat JSONL

One JSON object per line:
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Roles used in **`--tool-protocol=json`** SFT:
- `system`: the repo-agent system prompt (includes the tool schema + strict output rules).
- `user`: the goal and tool results, where tool results are prefixed with `[tool_result]`.
- `assistant`: **exactly one JSON object** in `content`, typically:
  - tool call: `{"type":"tool_call","name":"read_file","args":{...}}`
  - (note) our current extractor only outputs **tool_call** steps, not `final`.

### Canonical tool-call JSON shape (the training target)
```json
{"type":"tool_call","name":"read_file","args":{"rel_path":"python_programs/quicksort.py","max_chars":2000}}
```

### Example 1 — First action (no prior tool results yet)
```json
{
  "messages": [
    {"role": "system", "content": "You are a repo-fixing agent...\nTOOLS:\n[...]"},
    {"role": "user", "content": "GOAL:\nFix the quicksort implementation in python_programs/quicksort.py so that the tests pass"},
    {"role": "assistant", "content": "{\"type\":\"tool_call\",\"name\":\"list_files\",\"args\":{\"rel_dir\":\".\",\"max_files\":200}}"}
  ]
}
```

### Example 2 — Later action (includes a tool result in-context)
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

### Where this format comes from in our codebase

We construct the dataset by extracting from trace logs via `repo-agent sft-extract`:
- It pairs each `llm_request` (the prompt the model saw) with the subsequent valid `tool_call` action.
- It only emits a sample once the tool call actually produced a `tool_result` event.
<question>
<h4>Questions:</h4>
Are there any invariants on the tool_result that we must enforce for the training data?
What are the invariants on the rollout level? For example:
What if we have a rollout from the teacher where iterations 1-4 are all correctly chosen tool calls, as in they were the right call to make. Then iteration 5-8, maybe a suboptimal tool call was made, or a loop event occurred, or reflection triggered. I guess what I'm saying is, we're training the model to return valid tool calls. A valid tool call could be attached to a rollout that includes suboptimal tool call or a tool calls that trigger loop events (or something else I'm not thinking about). So if we include these in the training data, is it possible that we encourage valid json but also encourage other unwanted behaviors? Or am I overthinking this?

> Answer (Codex):
> - You’re not overthinking it — **step-level SFT will teach whatever the target action is**. If you include steps where the teacher (or sampler) makes a “valid but dumb” call, the student will learn that behavior in those contexts.
>
> ### Step-level invariants (recommended)
> > **Hard requirements (keep):**
> > - The target action is valid JSON and matches the schema: `{"type":"tool_call","name":..., "args":...}` with `name ∈ {list_files, read_file, grep, write_file}` and required args present.
> > - The paired tool execution succeeded: `tool_result.ok == true` (this is exactly what `--require-valid-tool-ok` enforces today).
> > - Context is bounded: truncate tool outputs (`--max-context-chars`) so you don’t train on huge blobs and blow up sequence length.
> >
> > **Strongly recommended heuristics (drop):**
> > - Drop exact repeats / obvious thrash: same tool+args repeated back-to-back, repeated `grep` with same pattern+dir, repeated `read_file` on the same file with tiny `max_chars`.
> > - Drop steps *after* the run has clearly entered a loop (e.g., after a `driver_note` “Loop detected…”), if your goal includes loop avoidance.
>
> ### Rollout-level invariants (recommended)
> - For **protocol compliance SFT**, you do *not* need the whole rollout to be good or to pass tests. You can safely use *good prefix steps* from a rollout that later fails.
> - For **policy imitation** (closer to “solve QuixBugs”), restrict to:
> > - only successful rollouts (`--require-success`) *and/or*
> > - only the first K steps, or only steps up to the first loop/reflection signal.
>
> ### Will “bad later steps” contaminate the early good steps?
> - Not directly. Each SFT example is “prompt so far → next action”. If iterations 1–4 are genuinely good, training on those steps is fine even if iterations 5–8 go off the rails.
> - The contamination risk is only if you also include the *bad steps as targets*. That’s why a simple filter like “keep steps until the first loop tripwire” is a good default.
>
> If we want the simplest version that still works: keep the current extractor invariants (`--require-valid-tool-ok`, optionally `--require-success`) and add one extra filter later: **drop steps after loop detection**.
</question>

Implementation reference: `src/llm_repo_agent/sft/extract.py`.

---

## How We Construct the Dataset (Pipeline)

### 0) Choose the QuixBugs task universe

In your QuixBugs checkout:
- `python_programs/*.py`: 50 buggy programs
- `python_testcases/test_*.py`: 42 test files

For clean evaluation, we split **by task** (by buggy program / test file), not by steps.

### 1) Train/Test split (by task)

**Rule:** no traces from test tasks can appear in the training dataset.

Recommended split:
- Build a “master suite” of the ~42 Python tasks that have tests (e.g., `python_testcases/test_quicksort.py` → `python_programs/quicksort.py`).
- Deterministically split tasks into train/test:
  - 70/30 (more stable evaluation) or 80/20 (more train data).

Deterministic split options:
- **Hash split:** `md5(task_id) % 100 < train_pct` → train else test.
- **Seeded shuffle:** sort task_ids, shuffle with fixed seed, slice.

Outputs:
- `eval/suites/quixbugs_train_suite.json`
- `eval/suites/quixbugs_test_suite.json`

### 2) Generate teacher traces on train tasks

We need a teacher model that can reliably follow `--tool-protocol=json` (e.g., `gpt-4.1-mini` or a strong Together model).

We run multiple rollouts per task to increase data volume and diversify trajectories.

Example (reuse `repo-agent prefs` as a rollout+trace generator):
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
  --test-policy on_write
```

Notes:
- `--test-policy on_write` makes the driver run tests after `write_file`, which often yields higher-quality patch trajectories.
<question>
<h4>Questions:</h4>
Why would this yeild higher quality patches? just because the driver only tests on write, doesn't mean the test will pass.

> Answer (Codex):
> - You’re right that `on_write` doesn’t *guarantee* a pass. The reason it often yields higher-quality *training traces* is that it creates a **tight feedback loop**: after a `write_file`, the next prompt includes `[TESTS PASS/FAIL]` output (because the driver appends it to the tool result).
> - That feedback tends to push a good teacher model to:
> > - make **smaller, more targeted edits** (because it can quickly see regressions),
> > - iterate toward a fix rather than stopping after one patch,
> > - ground subsequent tool calls in concrete failure output instead of “guessing”.
>
> In terms of dataset quality:
> - If you extract with `--require-success` (default True), you’re implicitly filtering to rollouts where the patch *eventually* passes tests. With `on_write`, those successful rollouts tend to contain more “test-driven” corrective steps.
> - If you extract with `--no-require-success`, then `on_write` still helps because the model sees failures and may correct course; but you’ll want to drop steps after obvious thrash/loops (see earlier invariants).
>
> So the real value of `on_write` is not “tests always pass”, it’s “the model gets immediate, actionable signal after edits”, which usually produces better trajectories to imitate.
</question>
- If you want cheaper traces (protocol-only), set `--test-policy never` and later extract with `--no-require-success`.

### 3) Extract step-level SFT JSONL from traces

Run:
```bash
poetry run repo-agent sft-extract \
  --trace-dir runs/quixbugs_teacher_traces \
  --output runs/instruction_tuning/quixbugs_tool_sft_train.jsonl \
  --format json
```

Extraction filters you can control:
- `--require-success` (default True): only include runs where `last_test.ok == True`.
  - If you generate traces with `--test-policy never`, you must use `--no-require-success`.
- `--require-valid-tool-ok` (default True): only include steps where the tool call succeeded (`tool_result.ok == True`).
- `--max-context-chars` (default 8000): truncates tool outputs in the prompt to control sequence length.

<question>
<h4>Questions:</h4>

```bash
(llm-repo-agent-py3.13) (base) ➜  llm-repo-agent git:(dpo-finetuning) ✗  poetry run repo-agent prefs \
  --suite eval/suites/gcd.json \
  --rollouts 10 \
  --out runs/quixbugs_traces_teacher_qwen25_72b/instruction_tuning.jsonl \
  --trace-dir runs/quixbugs_traces_teacher_qwen25_72b \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.0 \
  --seed 42 \
  --max-workers 5 \
  --tool-protocol json \
  --test-policy on_write
[prefs] Loaded suite: example_suite (1 tasks)
[prefs] Rollouts per task: 10
[prefs] Temperature: 0.0
[prefs] Base seed: 42
[prefs] Starting 10 rollouts (1 tasks × 10 rollouts) with 5 workers
[prefs] [1/10] fix_gcd seed=46: PASS (steps=7)
[prefs] [2/10] fix_gcd seed=45: PASS (steps=12)
[prefs] [3/10] fix_gcd seed=44: PASS (steps=12)
[prefs] [4/10] fix_gcd seed=43: PASS (steps=12)
[prefs] [5/10] fix_gcd seed=42: PASS (steps=16)
[prefs] [6/10] fix_gcd seed=47: PASS (steps=11)
[prefs] [7/10] fix_gcd seed=48: PASS (steps=12)
[prefs] [8/10] fix_gcd seed=49: PASS (steps=12)
[prefs] [9/10] fix_gcd seed=50: PASS (steps=12)
[prefs] [10/10] fix_gcd seed=51: PASS (steps=12)
```

If you look at the example above, you'll see many of the PASS had the same num of steps. You say that we can use num_rollouts=N, N >= 8, for the `prefs` arg, but one thing I'm worried about is that the rollouts won't have enough variance to actually create meaningful training data. If the trajectory is that same, then we might as well have the model train on the same sets of data points over and over—which would do nothing. My first thought is to increase the temperature. But would this just introduce more noise, or would it actually change the rollout in a meaninfuly way? I guess, maybe the question is, are there actually multiple routes to the optimal PASS answer that are meaningful for training, or does a introduction of any suboptimal rollout trajectory hurt the model in any way? Or am I overthinking this?

> Answer (Codex):
> - For **protocol/compliance SFT**, you don’t actually *need* a ton of variance within a single task. The main diversity comes from **different tasks + different tool outputs**, not from the teacher “exploring” wildly.
> - If a task (like `fix_gcd`) is easy and the teacher is strong, it’s normal to see many rollouts converge to near-identical trajectories (same tools in the same order). That doesn’t break SFT — it just means extra rollouts have diminishing returns for that task.
>
> **Will training on near-duplicate steps “do nothing”?**
> - It won’t add much new information, but it *can* still help by reinforcing the exact JSON schema and tool usage. The downside is mostly **cost/compute waste**, not correctness.
> - If you care, you can deduplicate samples at the dataset stage (hash `messages`).
<question>
>><h4>Questions:</h4>
>>Well, I guess my concern is that, we say we want 2k-3k training samples, but if 6/10 rollouts are pretty much the same, then that's more like ~600 training samples. I mean am I wrong? Should we still move ahead as planned?
>>
>> > Answer (Codex):
>> > - The “~600 samples” conclusion is usually **too pessimistic**, because duplicates don’t collapse the dataset down to “number of distinct rollouts”.
>> > - Each rollout contains many steps, and even “similar” rollouts differ in small but useful ways (tool output truncation, file ordering, which file got opened first, etc.). But yes: **some steps will be near-duplicates**, especially on easy tasks.
>> >
>> > Here’s the grounded way to think about it:
>> > - Target is **2k–3k *step examples*** after filtering/dedup — not 2k–3k unique tasks.
>> > - You’ll get most diversity from **many different tasks**, not from high variance within one task like `fix_gcd`.
>> >
>> > What we should do (practical, low-risk):
>> > 1) Move ahead, but run a **quick measurement pass** after generating traces:
>> > > - `repo-agent sft-extract ...` → count lines in the output JSONL
>> > > - optionally dedup by hashing `messages` and re-count
>> > 2) If you’re under target:
>> > > - add more tasks (use the 30-task `eval/suites/sft_finetune_task_suite.json` as a baseline, or expand toward the full ~42 tasks-with-tests)
>> > > - increase rollouts only where it adds diversity (harder tasks), not on trivial ones
>> > > - bump teacher `temperature` to ~0.2 (small), or add a second teacher model for diversity
>> >
>> > So: yes, we should still proceed — but treat “do we have enough samples?” as a **measured checkpoint**, not an assumption.
</question>

>
> **Should we increase teacher temperature?**
> - Yes, but only **slightly**. For a teacher, `temperature ~ 0.1–0.3` is often the sweet spot: enough to produce alternative but still-correct routes, without turning into noisy/hallucinated tool calls.
> - At `temperature=0.0`, seeds may have little effect (depends on provider); at `temperature>0`, seeds actually matter and will increase trajectory variance.
>
> **Does including “suboptimal” rollouts hurt?**
> - It only hurts if you train on the *bad actions* as targets.
> - For SFT, you can keep variance without harming quality by filtering:
> > - keep only steps from **successful rollouts** (`--require-success`) if you’re optimizing “solve the task”
> > - keep steps where `tool_result.ok == true` (`--require-valid-tool-ok`)
> > - drop steps after loop/reflection signals (so you don’t teach thrash)
>
> **Are there multiple meaningful routes to PASS?**
> - Sometimes yes (different files searched first, different evidence-gathering order). Sometimes no (a short, canonical path). Either way, you’ll still get plenty of dataset diversity across the ~30–40 QuixBugs tasks; don’t over-index on one task’s step-count histogram.
</question><br>

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

---

## Expected Dataset Size

Rough estimate (step-level):
```
num_samples ≈ (#train_tasks) × (rollouts_per_task) × (avg_valid_tool_calls_per_rollout)
```

Example:
- 29 train tasks (70% of 42 tested tasks)
- 8 rollouts each
- ~10 valid tool calls per rollout
→ `29 × 8 × 10 = 2320` SFT samples

This hits the 1K–3K target without needing another bug repo.

---

## Evaluation Plan (Held-Out Tasks)

### What we evaluate
On `eval/suites/quixbugs_test_suite.json`:
- **Baseline model** (same provider/protocol)
- **SFT finetuned model**

We compare:
- `parse_errors`, `loop_detections`, `avg_steps`, `tool_breakdown`
- optional: `success` / pass rate (if tests run)

### How we evaluate
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

## Known Limitations / Risks (SFT-Only)

- **Extractor outputs tool_call steps only** (`repo-agent sft-extract` currently skips `final`). If the base model’s `final` behavior regresses, we may need to add “final examples” to SFT.
- SFT teaches the model to imitate the teacher’s behavior on QuixBugs; generalization to larger repos may still be limited.
- `write_file` requires outputting full file content; this is feasible for QuixBugs-sized files but is not ideal long-term (patch-based tools would scale better).
