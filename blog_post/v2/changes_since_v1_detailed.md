# Changes Since v0

Base commit ("v0"): `248d275bdc0960d3815cb441364e226493b56802`

This document summarizes the major code/workflow changes made since v0 while adding Together support and building the DPO + SFT data pipelines.

## LLM Layer Refactor (Responses API -> Chat Completions)

**Files:** `src/llm_repo_agent/llm.py`, `src/llm_repo_agent/tool_schema.py`, `tests/test_llm_responses.py` (removed)

- Standardized on OpenAI-compatible **Chat Completions** (`client.chat.completions.create`) for both OpenAI and Together.
- Implemented `ChatCompletionsLLM` with proper **multi-turn** state (accumulated `messages`) and native function calling via `assistant.tool_calls` -> `tool` messages.
- Added `LLMConfig` + a lightweight, class-based `LLMFactory` so provider/model selection lives in one place (no scattered provider conditionals).

## Tool Calling Protocols: `native` vs `json`

**Files:** `src/llm_repo_agent/llm.py`, `src/llm_repo_agent/prompts.py`, `src/llm_repo_agent/agent.py`, `src/llm_repo_agent/main.py`

- Added `--tool-protocol {native,json}` to `repo-agent run|eval|prefs`.
- Added `JsonToolLLM` (JSON-in-content tool calling) for models that are unreliable with native tool calls:
  - Assistant emits a single JSON object like `{ "type": "tool_call", ... }` in `content`.
  - Driver appends tool output as a user message prefixed with `[tool_result]`.
- Improved debuggability of JSON-tool failures by logging `llm_parse_error` events that include the raw offending output.

## Prompt + Driver Hardening

**Files:** `src/llm_repo_agent/prompts.py`, `src/llm_repo_agent/agent.py`, `tests/test_driver_note_ordering.py`

- Made the system prompt `tool_protocol` aware (native vs json instructions).
- Added a **FIRST ACTION** rule requiring `list_files(rel_dir='.', ...)` before other actions.
- Added a **WRITE RULE** clarifying that only `write_file` edits the repo (final `changes` is descriptive, not an edit).
- Ensured reflection/driver notes are appended as `system` messages without breaking `assistant(tool_call) -> tool(result)` adjacency.

## Preference Data Generation for DPO (`repo-agent prefs`)

**Files:** `src/llm_repo_agent/prefs/*`, `src/llm_repo_agent/eval/runner.py`, `src/llm_repo_agent/main.py`, `tests/test_prefs_*`

- Added a first-class `prefs` pipeline:
  - Runs N rollouts per task.
  - Scores rollouts and selects a preferred vs non-preferred rollout when contrast exists.
  - Writes Together-compatible preference JSONL plus a `_meta.jsonl` sidecar with scores/tests/trace ids.
- Added rollouts multithreading via `ThreadPoolExecutor` (`--max-workers`, default 4).
- Added dataset write behavior control (`--data-write-mode {overwrite,append}`) so multiple runs can accumulate pairs.

## Cost Estimation (`repo-agent estimate-cost`)

**Files:** `src/llm_repo_agent/estimate_cost.py`, `src/llm_repo_agent/main.py`, `tests/test_estimate_cost.py`

- Added a command to scan trace logs for `llm_usage` events and compute:
  - average prompt/completion tokens per LLM call
  - cost per call (given `--price-in/--price-out`)
  - implied calls per preference pair and scaled total cost for `--target-pairs`

## SFT Dataset Extraction (`repo-agent sft-extract`)

**Files:** `src/llm_repo_agent/sft/*`, `src/llm_repo_agent/main.py`, `tests/test_sft_extract.py`

- Implemented step-level SFT sample extraction from trace logs.
- By default filters to successful trajectories via `run_end.payload.state.last_test.ok == true`.
- Added safeguards for dataset quality:
  - optionally require `tool_result.obs.ok == true`
  - optional early cutoff on loop detection
  - optional `write_file` target filtering
  - optional requirement that the first tool call is root `list_files('.')`
- Added `--format {json,native}` to match the intended tool-calling training style:
  - `json`: assistant content contains JSON tool calls + user `[tool_result]`
  - `native`: assistant has `tool_calls` + `tool` role messages

## Eval Harness Improvements

**Files:** `src/llm_repo_agent/eval/runner.py`, `src/llm_repo_agent/eval/metrics.py`, `src/llm_repo_agent/main.py`

- Added `--rollouts` support for eval suites.
- Added parallel eval execution via `--num-workers`.
- Improved metrics around parse/tool errors (including heuristics for parse-like failures in exception strings).

## Tracing + Observability

**Files:** `src/llm_repo_agent/trace.py`, `src/llm_repo_agent/agent.py`

- Standardized trace events used across run/eval/prefs:
  - `llm_request`, `llm_usage`, `llm_action`, `tool_result`, `tests`, `reflection`, `final`, `run_end`
  - plus `driver_note`, `llm_parse_error` for diagnostics
- Trace metadata now records provider/model/task/test_cmd and is consumed by cost estimation and SFT extraction.

## Fine-tuning Helper Scripts (Together)

**Files:** `dpo_finetune_quick.py`, `sft_finetune_quick.py`, `src/llm_repo_agent/deploy_endpoint.py`

- Added standalone scripts to validate datasets, upload to Together, and start fine-tuning jobs:
  - DPO + LoRA (`training_method="dpo"`)
  - SFT + LoRA (`training_method="sft"`) with optional W&B args
- Expanded terminal status handling for `--watch` polling so scripts don't hang when Together reports completion.

## Project Config, Suites, and Docs

**Files:** `pyproject.toml`, `README.md`, `eval/suites/*.json`, `run_commands.sh`

- Added `together` as a dependency (alongside `openai`) and a pytest warning filter for Together SDK's pydantic deprecation warning.
- Updated README usage examples to include `--tool-protocol` and `sft-extract --format`.
- Added multiple eval suites for pilots/cost-estimation/SFT collection (e.g. `pref_data_gen_pilot_1.json`, `pref_cost_estimate_suite.json`, `sft_finetune_task_suite.json`, and per-task suites like `gcd.json`).

## Updates From This Chat

**Files:** `src/llm_repo_agent/sft/extract.py`, `src/llm_repo_agent/sft/config.py`, `src/llm_repo_agent/main.py`, `tests/test_sft_extract.py`

- Added `sft-extract` filters/flags:
  - `--drop-postfix-on-loop` (plus backward-compatible alias `--drop-post-fix-on-loop`)
  - `--filter-write-file-targets` (drop non-goal writes/tests)
  - `--require-root-list-files-first` (require initial `list_files` at repo root)
- Fixed goal-path regex used by SFT extraction when filtering write targets.
- Added/updated unit tests covering the new filters and behaviors.

**Files:** `src/llm_repo_agent/prompts.py`, `tests/test_prompt_tool_protocol.py`

- Tightened JSON tool protocol prompt rules:
  - Mandatory first action `list_files(rel_dir='.', max_files=...)`
  - Strict “FINAL OUTPUT” JSON-only requirement
  - Clearer write rule: only `write_file` edits the repo

**Files:** `src/llm_repo_agent/main.py`, `tests/test_tool_protocol_cli.py`

- Hardened `cmd_eval` to tolerate missing `num_workers` in test harnesses (`getattr(..., 0)`).

**Files:** `eval/suites/topological_ordering.json`, `eval/suites/failed_training_tasks.json`

- Fixed `topological_ordering.json` to point at the correct task.
- Added `failed_training_tasks.json` suite containing the seven “stuck” tasks for focused evaluation.

**Files:** `sft_finetune_quick.py`

- Added Together training knobs: `--warmup-ratio`, `--max-grad-norm`, `--train-on-inputs`, and `--learning-rate-scheduler-type`.
