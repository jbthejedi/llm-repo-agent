# llm-repo-agent

A disciplined, loop-based repo fixer. The model chooses one action at a time (list/read/grep/write), the driver executes safely, runs your test command after writes, and records a full JSONL trace (including reflections) for replay and debugging.

---

## Quickstart

1) Install deps and the CLI:
```bash
poetry install
poetry run repo-agent --help
```

2) Run on QuixBugs (small, real bug):  
```bash
# Quick fix (usually one write): quicksort
poetry run repo-agent \
  --repo ~/projects/QuixBugs \
  --goal "Fix quicksort so python_testcases/test_quicksort.py passes. Make the smallest correct change." \
  --trace runs/trace.jsonl \
  --test "python -m pytest -q python_testcases/test_quicksort.py"
```

3) Run a harder case to see reflection kick in:  
```bash
poetry run repo-agent \
  --repo ~/projects/QuixBugs \
  --goal "Fix breadth_first_search so python_testcases/test_breadth_first_search.py passes. Make the smallest correct change." \
  --trace runs/trace.jsonl \
  --test "python -m pytest -q python_testcases/test_breadth_first_search.py"
```

4) Inspect the trace:  
```bash
# All events
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run <run_id> --full

# Only reflections
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run <run_id> --kind reflection --full

# Only LLM prompt for a given request
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run <run_id> --kind llm_request --index 0 --pretty-only-prompt
```

---

## What it does

- **Constrained tools:** `list_files`, `read_file`, `grep`, `write_file`. No arbitrary shell, no network. Driver-only tests via `--test ...`.
- **ReACT + CoT:** LLM produces typed actions with optional `thought`; the driver enforces one action per turn and a “no final before evidence” rule.
- **Reflection:** On test failures, tool errors, or loop tripwire, a second LLM call distills lessons (`notes/next_focus/risks`) and logs them to history and trace.
- **Observability:** Everything goes to JSONL (`runs/trace.jsonl`): prompts, tool calls/results, reflections, trailing text warnings, tests, finals.

---

## LLM adapter contract (for custom models)

Return typed actions from `llm_repo_agent.actions`:
- `ToolCallAction(name: str, args: dict, thought: Optional[str])`
- `FinalAction(summary: str, changes: list, thought: Optional[str])`

Set `_last_raw` (parsed JSON) and `_last_trailing` (extra text) on the adapter if available; the driver logs them. Legacy dict-shaped actions are rejected.

Reflection uses `llm.reflect(messages)` (JSON mode) returning `Reflection(notes, next_focus, risks)`.

---

## How it works (short)

1) Prompt = system rules + goal + compact history + run summary.  
2) LLM returns one typed action. Driver executes via `ActionController`.  
3) After `write_file`, driver runs your test command (if provided).  
4) Reflection controller may trigger (failures/errors/loop) to add durable notes.  
5) Loop until final or max iterations. Final output includes test_result and any touched-file fallback in `changes`.

---

## Dev / tests

```bash
poetry run pytest
```

Trace parsing utility: `poetry run python -m llm_repo_agent.inspect_trace --help`

---

## Why this repo

Small, auditable example of an “agentic” loop with safety rails, typed actions, strict JSON prompting, reflection, and full traceability—designed to be easy to read, debug, and extend.***
