# Native Tool Calling SFT: Current State + Open Questions (Together)

## Goal
We want the finetuned model (e.g. `Qwen2.5-7B` on Together) to behave well under **native tool calling** using the OpenAI‑compatible Chat Completions interface:

- The model returns **structured** tool calls via `assistant.tool_calls` (not JSON embedded in `assistant.content`).
- The driver executes the tool and replies with a `tool` role message that includes `tool_call_id`.
- The conversation continues: `assistant(tool_calls) -> tool(result) -> assistant(...) -> ...`.

This is the same mechanism our repo-agent uses at runtime today.

## How the agent works right now (runtime)
At inference time we run:

- Chat Completions API with `tools=[...]` and `tool_choice="auto"`.
- The model returns either:
  - `assistant.tool_calls=[{id, function: {name, arguments}}]` (native tool call), or
  - a normal `assistant.content` final JSON (our `{"type":"final", ...}` contract).
- After every tool call, we append the tool result as a `tool` role message.
- We also inject reflection “driver notes”. Important constraint: for strict providers (OpenAI), the message order must keep **adjacent** `assistant(tool_calls)` → `tool(tool_call_id=...)`.

## What our SFT extractor currently outputs
The current extracted SFT samples (example from `runs/instruction_tuning_test_3/sft_dataset.jsonl`) look like:

```json
{
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"GOAL: ..."},
    {"role":"assistant","content":"{\"type\":\"tool_call\",\"name\":\"read_file\",\"args\":{...}}"}
  ]
}
```

And when tool results are included, they are represented as:

- `{"role":"user","content":"[tool_result]\\n..."}`

So we are training a **text protocol**:

- assistant emits JSON tool calls in **content**
- tool results are fed back as **user text**

This is *not* the same as native tool calling.

## The mismatch / problem
We have two possible protocols:

### A) Native tool calling (what runtime uses)
The model produces a structured `tool_calls` field, and tool results are `tool` role messages.

### B) JSON-in-content tool calling (what current SFT dataset contains)
The model produces a JSON tool call inside `assistant.content`, and tool results are encoded as text inside `user.content`.

Right now:

- Runtime uses (A).
- SFT dataset is training (B).

If we finetune on (B) but run with (A), we risk teaching the model the *opposite* behavior:

- The model may start emitting JSON tool calls in `assistant.content` (which our runtime explicitly discourages and doesn’t parse in the native-tool path).
- Or it may produce inconsistent tool arguments because the channel format differs.

## What we want the SFT dataset to look like (native tool calling)
If Together fine-tuning supports it, we want each training example to include actual tool call structure:

```json
{
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"GOAL: Fix gcd ..."},
    {
      "role":"assistant",
      "tool_calls":[
        {
          "id":"call_1",
          "type":"function",
          "function":{
            "name":"read_file",
            "arguments":"{\"rel_path\":\"python_programs/gcd.py\",\"max_chars\":1000}"
          }
        }
      ]
    },
    {"role":"tool","tool_call_id":"call_1","content":"<file contents>"},
    {
      "role":"assistant",
      "tool_calls":[ ... next tool call ... ]
    }
  ]
}
```

This trains the model to:

- decide to call a tool (vs “final”)
- choose the right tool name
- output valid JSON args for the tool schema
- chain tool calls correctly across turns based on tool results

## The critical open question (Together)
We need to confirm Together’s fine-tuning training-data schema supports **native tool calling** fields:

1) Does Together accept `tool_calls` on assistant messages in training JSONL?
2) Does Together accept `role: "tool"` messages with `tool_call_id` in training JSONL?
3) If “tool calling” fine-tuning isn’t supported, what is the expected alternative?
   - Often the fallback is training a **text protocol** (JSON tool calls in content), then at inference time *not* using `tools=...` and instead parsing the JSON yourself.

Until we know Together’s accepted schema, we can’t be sure whether:

- we should change our extractor to output native tool-call messages, or
- we should keep the JSON-in-content protocol and adjust runtime to match (not desired right now).

## What to check in Together docs (actionable checklist)
When reviewing Together docs / SDK:

- **Fine-tune job type**: confirm SFT job supports chat-format training (messages array).
- **Allowed message fields**:
  - `role`, `content`
  - whether `tool_calls` is allowed on assistant
  - whether `tool_call_id` is allowed on tool messages
- **If tool calling is supported**:
  - do they require a `tools` schema at training time?
  - do they require canonical tool names / JSON args?
- **Model constraints**:
  - which base models support chat SFT
  - whether tool-calling is supported for those models

## Proposed next step (once schema is confirmed)
If Together supports native tool-call fields:

- Update `repo-agent sft-extract` to optionally output a “native-tools” dataset format:
  - preserve `assistant.tool_calls` + `tool` messages
  - keep correct adjacency `assistant(tool_calls)` → `tool(result)`
  - ensure reflection notes never break adjacency

If Together does **not** support native tool-call fields:

- Decide whether we’re willing to:
  - keep finetuning on JSON-in-content protocol, **and**
  - switch runtime inference for finetuned models to parse JSON tool calls from content (no `tools=...`), OR
  - skip tool-call SFT and only finetune for high-level reasoning / planning (less ideal for our agent loop).

## Why this matters for the repo-agent project
Tool calling is the core capability of this agent. If we want finetuning gains that transfer to runtime behavior, training and inference must match the same protocol.

