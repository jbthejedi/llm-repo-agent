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


## New Together LLM Adapter


## Prompt + Driver Hardening

**Files:** `src/llm_repo_agent/prompts.py`, `src/llm_repo_agent/agent.py`, `tests/test_driver_note_ordering.py`

- Made the system prompt `tool_protocol` aware (native vs json instructions).
- Added a **FIRST ACTION** rule requiring `list_files(rel_dir='.', ...)` before other actions.
- Added a **WRITE RULE** clarifying that only `write_file` edits the repo (final `changes` is descriptive, not an edit).
- Ensured reflection/driver notes are appended as `system` messages without breaking `assistant(tool_call) -> tool(result)` adjacency.