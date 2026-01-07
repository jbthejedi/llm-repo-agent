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
<question>
Q: what does "score" mean? How? Oh, I kept reading and I think I see the "scoring & pairs" section below
</question>
<answer>
A: Score is a numeric ranking per rollout: tests pass=1.0, fail=0.0, with tie-breakers like fewer iterations/tools/files touched and smaller diffs. It lets us rank rollouts and pick preferred vs non-preferred.
</answer><br>

- Select the best and worst rollout to form a preference pair.
<question>
Q: Why do we select the best and the worst to form a preference pair. Is it something about the DPO loss function? Is it contrastive loss or something. Does the greater contrast help it the algo more?
</question>
<answer>
A: DPO is pairwise preference learning. Best vs worst maximizes the preference gap, making the signal less ambiguous. It is not exactly contrastive loss, but stronger contrast generally yields a clearer preference gradient. You can also do best vs median if you want a softer contrast.
</answer><br>

- Write one JSONL line per task into `--out`.
<question>
Q: So one task will generate one preference pair data point?
</question>
<answer>
A: For the pilot, yes: one pair per task per batch of rollouts. Later we can emit multiple pairs per task (best vs each non-best) to increase data size.
</answer><br>

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
<question>
Q: So basically, the preference pair will be signaling "this was the tool that resulted in the bug fix"? Is that what it is? Actually, yea. What's the signal we're providing? The goal is to have the tuned model do what? Converge on a bug fix more quickly?
</question>
<answer>
A: The signal is which final response led to better test outcomes. It does not directly supervise tool choice; it rewards end results. The primary goal is higher pass rate, with secondary improvement in efficiency if we include step/tool counts in scoring.
</answer>
<question>
Q: So in this case, where we finetune using the suggested preference pairs, the ideal behavior is that tool suggestions to remain the same, but the suggested fix—once the agent has enough information to know what's going on (like the file has been read)—is more optimal?
</question>
<answer>
A: Yes, that is the expected effect for a final-only DPO pilot. The preference signal is on the final response, so it primarily nudges the model toward better final fixes once enough evidence is gathered. Tool usage may shift indirectly, but it is not explicitly trained unless we include tool-call transcripts.
</answer><br>

- Tool-call transcripts would require multi-turn SFT or custom reward model.
<question>
Q: What is a tool-call transcript? And what are you saying that this is something DPO can't do? Help me understand so I can know if I care about this or not
</question>
<answer>
A: A tool-call transcript is the full multi-turn sequence: assistant tool_calls, tool results, and subsequent assistant turns. Together's DPO format expects a single assistant message per output, so it cannot directly train on multi-turn tool usage. Tool-use learning would need multi-turn SFT or a different training setup.
</answer>
<question>
Q: But my question is basically: does the DPO algo in general not support multi-turn tool usage, or is that a Together limitation? Is it that multi-turn usage training is conventionally for SFT? I'm trying to understand where the exact limitation is and what is the convention.
</question>
<answer>
A: DPO as an algorithm can be applied to multi-turn trajectories if you can provide full transcripts and preferred vs non-preferred continuations. The limitation here is Together’s preference dataset schema (single assistant message per output). Conventionally, tool use is taught with multi-turn SFT on full transcripts, and DPO is layered later to refine preferences, but DPO itself is not inherently single-turn.
</answer>
<question>
Q: Right, so tuning for better tool calls is essentiall "instruction tuning", right? That's conventionally in the SFT land.
Help me understand why optimizing tool calls is a better fit for instruction tuning and not RLHF (DPO)
</question>
<answer>
A: Yes, tool-call quality is typically improved with SFT because you can supervise the exact sequence and schema compliance (roles, arguments, ordering) from labeled transcripts. DPO/RLHF can help, but only if you have preference pairs over full tool-use trajectories or a reward signal that reflects tool success; single-message preferences don’t directly teach tool-call formatting. So SFT is the most direct, reliable way to teach structured tool usage, with DPO layered later to refine preferences.
</answer>
<question>
Q: When you say "supervised" this is essentially just the autoregressive loss using instruction tuning data (optimal tool calls) with like a LORA or something, right?
</question><br>
<answer>
A: Exactly. SFT is standard next-token loss on curated transcripts (optimal tool calls + responses). You can apply it via full fine-tuning or LoRA adapters; the training objective is the same, just different parameter efficiency.
</answer>
<question>
Q: Does Together allow SFT (lora) with an instruction tuning dataset? Could we apply it to our problem here to improve tool calls with Qwen? Is that data we can generate?
</question>
<answer>
A: Yes, Together supports SFT/LoRA with instruction-style conversational datasets. In principle you can use it to improve tool-call behavior, but their data format enforces alternating user/assistant messages, so you can’t include `tool` role messages directly. You’d need to encode tool calls/results inside user/assistant content (e.g., assistant emits a JSON tool_call, user provides the tool result). That data is generatable from your traces, but it requires a deliberate serialization format.
</answer>
<question>
Q: I'm not sure if I understand the limitation you're pointing to.

```
{role: system, content: "youre an agent in a loop"}
{role: user, content: "GOAL: fix quicksort"}
{role: assistant, tool_calls: [...]}   <-- appended after API call
{role: tool, tool_call_id: "...", content: "..."}  <-- appended before next API call
{role: assistant, tool_calls: [...]}
{role: tool, tool_call_id: "...", content: "..."}
...
```

Above is a multi-turn chat example. ohhh I see. Ooff that seems like it would cause major problems. I'm not sure I understand your solution either. It sounds hacky :/ —that's upsetting that they would force such a limiting constraint
</question><br>
<answer>
A: The limitation is about *training data format*, not inference. Together’s SFT and DPO schemas only accept `system/user/assistant` messages in `input.messages`, and for DPO the `preferred_output` / `non_preferred_output` must each be a single assistant message. That means you can’t include `tool` role messages or a multi-turn tool transcript in the training data. The workaround is to serialize tool calls and tool results into plain text inside the assistant/user messages (a common pattern), or use a custom training pipeline that supports tool-role transcripts. It’s not hacky so much as a schema constraint, but it is less faithful than true tool-role supervision.
</answer>
<question>
Q: wait, a custom training pipeline? What do you mean? Like you mean spinning up a VM with A100s or something and implementing the pipeline via huggingface or something? Then training that way? If serializing tool calls and tool results into plain text is a common patter and won't change the inference format, we should probably stick with that.
</question><br>
<answer>
A: Yes—“custom pipeline” means training outside Together’s hosted schema (e.g., Hugging Face/TRL on your own GPU or a hosted GPU), where you can feed full tool-role transcripts and customize the loss. For this pilot, I agree: stick with serialization into plain text inside user/assistant messages, because it fits Together’s format and keeps inference unchanged.
</answer>

- Together format requires exactly one assistant message per output array.
<question>
Q: Is this related to the above bullet? Because together requires exactly/only one assistant message per output array, we can't model multi-turn responses? Will this hinder us?
</question>
<answer>
A: Yes, related. Because each output must be one assistant message, we cannot include the full multi-turn tool interaction in the DPO dataset. For the pilot this is acceptable, but if you want to improve tool behavior specifically, we will need a richer dataset or different training method later.
</answer>

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

## Open Questions

### Edge case: No contrast in rollouts

What happens when all N rollouts either pass or all fail? Options:
- Skip tasks where `score_preferred == score_non_preferred` (no pair emitted).
- Log these as "no contrast" in a separate file for analysis.
- Require a `--min-gap` threshold to filter pairs where best/worst are too similar.

### Target sample size

Even a rough target (50 pairs? 200?) would help scope the pilot:
- Informs how many suite tasks to include.
- Helps estimate cost (N rollouts × M tasks × tokens-per-rollout).

### Cost estimate

Back-of-envelope estimate before running:
- ~X tokens per rollout (depends on task complexity and max-iters).
- N rollouts × M tasks = total API calls.
- Together pricing for Qwen model.
