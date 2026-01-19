# LLM Repo Agent V2

## What changed since v1?
- New LLM Adapter for ChatCompletions and JsonTools
- Tool-protocol: native -> json
- Response API -> Chat Completions Multiturn
    - removed history
- Sandbox added
- Multithreading added
- prompt + driver hardening
- Multiple rollouts per task during evaluation
- SFT (@sft_plan_refined.md)
    - generation of data
    - finetune calling
    - `sft-extract` command
    - ended up using `prefs` command
- DPO (@sft_dpo_plan probably honorable mention)
    - generation of data via `prefs`
    - dpo finetune calling
    - cost estimation


## New LLM Adapters: ChatCompletions +  + Together
- Moved from ResponseAPI to Chat Completions
    - Removed history table: history now kept in user/assistant baked in in-chat history
- Json Tool Calling
    - tool-protocol=json now returns tool calls as json in the response instead of using native function calling. This is so we could do sft finetuning as Together's finetune jobs require the json to be explicit in the samples


## Prompt + Driver Hardening

**Files:** `src/llm_repo_agent/prompts.py`, `src/llm_repo_agent/agent.py`, `tests/test_driver_note_ordering.py`

- Made the system prompt `tool_protocol` aware (native vs json instructions).
- Added a **FIRST ACTION** rule requiring `list_files(rel_dir='.', ...)` before other actions.
- Added a **WRITE RULE** clarifying that only `write_file` edits the repo (final `changes` is descriptive, not an edit).
- Ensured reflection/driver notes are appended as `system` messages without breaking `assistant(tool_call) -> tool(result)` adjacency.