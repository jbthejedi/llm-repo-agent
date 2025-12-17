# llm-repo-agent

A simple loop-based repo agent that uses an LLM to choose single actions (tool calls) and runs them against a repository.


---

## Inspecting run traces 🔍

The agent logs a JSONL trace (default `runs/trace.jsonl`) containing timestamped events for each run. Each event includes a `run_id` so you can filter runs.

You can inspect a run with the small helper CLI provided by the package:

```bash
# Show events for a run (pass the run id used when running the agent)
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run <run_id>

# Limit the number of events shown
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run <run_id> --max 50

# Show only the raw LLM prompt for a specific llm_request event (no metadata)
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run <run_id> --kind llm_request --index 0 --pretty-only-prompt

# Write the prompt for a specific llm_request event to a file for sharing or repro
poetry run python -m llm_repo_agent.inspect_trace --trace runs/trace.jsonl --run <run_id> --kind llm_request --index 0 --dump-prompt /tmp/prompt.txt
```
Or use the Python API to reconstruct a run history and pretty-print it:

```python
from pathlib import Path
from llm_repo_agent.trace import Trace

trace = Trace(Path("runs/trace.jsonl"), run_id="<run_id>")
history = trace.get_run_history("<run_id>")
for ev in history:
    print(ev)
```

This will output a sequence of `tool_call` and `observation` entries similar to the agent's in-memory history, which can be used to replay or analyze what happened during a run.

---

## LLM Adapter contract 🔧

The agent expects LLM adapters to return *typed* Action objects. Adapters SHOULD parse raw model responses and return one of the action types defined in `llm_repo_agent.actions`:

- `ToolCallAction(name: str, args: dict)` — instructs the agent to run a single tool with the provided name and args.
- `FinalAction(summary: str, changes: list)` — signals task completion and optional repository changes.

Adapters are also encouraged to capture raw model output for traceability by setting these attributes on themselves before returning an action:

- `_last_raw` — the raw JSON-parsed object produced by the model
- `_last_trailing` — any trailing text or extra JSON after the first parsed object (if present)

Backward compatibility: the agent no longer accepts legacy dict-shaped actions; adapters MUST return typed actions to make behavior explicit and testable.
