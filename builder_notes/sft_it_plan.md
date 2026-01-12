# SFT Instruction-Tuning Plan (Tool-Call Schema Compliance)

## Goal

Train a finetunable **Qwen2.5-7B Instruct** model with **LoRA + SFT** so it reliably emits **valid tool calls with required args** in our repo-agent loop. This reduces failures like missing `rel_dir`/`max_files`/`rel_path` and makes the 7B model usable as a cheaper generator for downstream preference-data generation (DPO later).

This phase optimizes **format compliance + basic tool-use competence**, not maximum bug-fixing quality.

## Tools (current agent)

The model must produce valid tool-call JSON for these tools:

- `list_files(rel_dir, max_files)`
- `read_file(rel_path, max_chars)`
- `grep(pattern, rel_dir, max_hits)`
- `write_file(rel_path, content)`

`run_tests` remains driver-only (not a model tool).

## Data Strategy

### Data sources

1. **Reuse existing trace logs** (`runs/**/*.jsonl`): they already contain:
   - `llm_request` (prompt/messages sent)
   - `llm_action` (tool call the model produced)
   - `tool_result` (tool outcome; used as next-step context)
   - `final` / `run_end` (for filtering to successful runs)

2. **Collect additional teacher traces if needed** using a strong model (e.g., Qwen-72B / GPT) on a larger QuixBugs suite.
   - Prefer `repo-agent eval` for this collection (we don’t need preference-pair selection for SFT).
   - Use **2–3 rollouts per task** for diversity (seeds/temperature), across **~20–30 tasks**.

### Target size (rough)

- Aim for **~100–150 successful trajectories**.
- Each trajectory typically yields **~3–10 tool-call steps**.
- Expect **~500–1000 step-level SFT samples** after filtering (enough to move the needle for schema-following on 7B).

## Dataset Construction (Step-Level SFT)

### What a training sample is

We build the SFT dataset at the **step level**:

- **Input**: system + goal + recent context (previous tool results).
- **Target**: the **next tool call** the assistant should output (name + fully specified args).

One rollout with ~6 tool calls can yield ~6 SFT samples.

**What an SFT sample looks like (Together/OpenAI chat format):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a code-fixing agent..."},
    {"role": "user", "content": "GOAL:\nFix the bug in quicksort.py..."},
    {"role": "assistant", "content": "{\"type\": \"tool_call\", \"name\": \"list_files\", \"args\": {\"rel_path\": \".\"}}"},
    {"role": "user", "content": "[tool_result] quicksort.py\ntest_quicksort.py\n..."},
    {"role": "assistant", "content": "{\"type\": \"tool_call\", \"name\": \"read_file\", \"args\": {\"rel_path\": \"quicksort.py\"}}"},
    {"role": "user", "content": "[tool_result] def quicksort(arr):\n    ..."},
    {"role": "assistant", "content": "{\"type\": \"tool_call\", \"name\": \"write_file\", \"args\": {\"rel_path\": \"quicksort.py\", \"content\": \"def quicksort(arr):\\n    ...fixed...\"}}"}
  ]
}
```

Each sample is one **turn** - you'd split the above into 3 separate training examples:
- Sample 1: system + goal → `list_files` call
- Sample 2: system + goal + list result → `read_file` call
- Sample 3: system + goal + list result + read result → `write_file` call

### Extraction procedure

For each trace run:

1. Reconstruct the message context from `llm_request` and `tool_result` events.
2. For each `llm_action` that is a `tool_call`, emit one training example:
   - `messages` = context up to that point
   - `assistant_target` = the tool-call JSON (`{"type":"tool_call","name":"...","args":{...}}`)

### Filtering and cleaning

Start strict (quality > quantity):

- Drop malformed tool calls (missing required args, invalid JSON, unknown tool).
- Drop steps where the immediate tool execution returned `ok=False` due to missing/invalid args.
- Prefer trajectories that **eventually pass tests** (less label noise).
- Truncate long tool outputs included in context (e.g., cap read/grep snippets) to control token costs.

Optional later: keep a small set of “recovery” examples (tool call fails → model corrects next step).

## Training (Together LoRA SFT)

### Base model

- Use a finetunable base like `Qwen/Qwen2.5-7B-Instruct` (not `*-Turbo`).

### Objective

- Teach the model to emit **valid, fully specified tool calls** under diverse contexts.

### Hyperparameters (starting point)

- LoRA enabled (default Together LoRA path)
- 1–3 epochs
- small LR (e.g., `1e-5`)
- batch size: `"max"`
- LoRA rank/alpha: `r=16`, `alpha=32`, modules: `all-linear`

## Evaluation

### Held-out split

- Evaluate on **QuixBugs tasks not used during SFT data collection** (e.g., train: 20–25 tasks, eval: 10–15 tasks).

### Primary metric (what we’re optimizing)

**Tool-call validity rate** on held-out tasks:

- required args present
- tool call parses
- tool executes successfully (`tool_result.ok=True`)

### Secondary metrics

- pass rate on held-out tasks
- average steps / tool calls

Compare **baseline 7B** vs **SFT LoRA 7B**.

## Iteration Path

1. If validity improves: proceed to use the SFT 7B as a cheaper model for generating DPO preference data.
2. If validity improves but pass rate doesn’t: expand context diversity and/or add limited “stop/final” examples.
3. Once the model is schema-competent, consider DPO for “which valid trajectory/patch is better.”

## Immediate Next Steps

1. Inventory existing trace logs and estimate how many valid tool-call steps can be extracted.
2. If needed, run a larger QuixBugs collection suite with a strong teacher model (20–30 tasks, 2–3 rollouts each).
3. Implement an extractor script to build a step-level SFT JSONL dataset from trace logs (strict filtering + truncation).
4. Kick off Together LoRA SFT on `Qwen/Qwen2.5-7B-Instruct`.
5. Evaluate on a held-out QuixBugs suite and decide whether to scale data collection and/or move on to DPO.
