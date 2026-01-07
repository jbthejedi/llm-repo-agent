# Notes v1 — Chat Completions Migration

This narrative captures the reasoning behind the switch to multi-turn Chat
Completions, how tool calls are threaded across iterations, and why reflection
notes are injected the way they are.

## The Multi-turn Loop (What Actually Happens Each Iteration)

The agent no longer compiles a fresh prompt every turn. Instead, the LLM adapter
maintains a single conversation transcript (`_messages`) and appends to it as
the run progresses.

High-level flow:

1) Start the conversation with a system message and the user goal.
2) Call `chat.completions.create(...)` with tool schemas enabled.
3) If the model returns a tool call:
   - Append the assistant message that contains the tool call.
   - Execute the tool.
4) On the *next* iteration, append the tool result as a `tool` message.
5) Ask the model for the next action.

That yields a transcript that grows like this:

```
{role: system, content: "youre an agent in a loop"}
{role: user, content: "GOAL: fix quicksort"}
{role: assistant, tool_calls: [...]}   <-- appended after API call
{role: tool, tool_call_id: "...", content: "..."}  <-- appended before next API call
{role: assistant, tool_calls: [...]}
{role: tool, tool_call_id: "...", content: "..."}
...
```

Important detail: the assistant tool call is appended *after* the API response,
and the tool result is appended *before* the next API call. This keeps the
conversation consistent and lets the model see exactly what it requested and
what it got back.

## How the Model Knows the Order of Tool Calls

Chat Completions expects the `messages` array in chronological order (oldest to
newest). The model reads the list top-to-bottom like a transcript. The ordering
is implicit in the list itself.

Two rules matter:

- Tool results appear after the assistant message that requested them.
- `tool_call_id` ties the tool result to its originating tool call.

If you reorder messages or insert a tool result before its tool call, the model
will get confused.

## Reflection in a Multi-turn World

Reflection is not part of the main chat loop. It is a separate LLM call that
produces a concise driver note (what went wrong, next focus, risks).

We inject that reflection into the main conversation as a **system message**:

```
{role: system, content: "DRIVER NOTE:\n- ..."}
```

Why system?

- It has the highest steering weight.
- It acts like an instruction from the driver scaffold rather than a normal
  observation.
- It reduces the chance the model ignores the feedback.

Multiple system messages are valid. Chat Completions reads them in order, and
later system messages typically override or refine earlier ones. That’s why the
reflection note is appended at the end of the transcript.

Reflection should also be **optional** for ablations and experimentation so we
can measure its impact.

## OpenAI-Compatible Client (Mental Model)

We use the OpenAI Python SDK (`from openai import OpenAI`), but point it at
Together by setting `base_url="https://api.together.xyz/v1"`.

That means:

- The **client library** is OpenAI’s.
- The **API** we’re calling is Together’s.
- The **interface** is OpenAI’s chat-completions schema (messages + tools +
  tool_calls), which Together implements.

This lets a single `ChatCompletionsLLM` adapter work across providers.

## Model Coverage and Caveats

- Qwen, Llama, Mixtral: usable through Together’s OpenAI-compatible endpoint.
- Tool calling quality varies by model (Qwen is stronger than many others).
- Grok is not on Together; it requires xAI’s API. If xAI supports an
  OpenAI-compatible endpoint, the same adapter could work with a different
  `base_url`.

OpenAI-compatible only guarantees request/response *shape*, not feature quality
or tool-calling reliability.

## Open Questions / Next Checks

- Confirm Together’s required DPO dataset fields and naming.
- Decide whether DPO pilot data should be final-only or include tool-call
  transcripts.
- Verify whether Together’s `seed` field provides deterministic outputs in
  practice.
