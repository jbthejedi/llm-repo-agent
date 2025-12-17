import json
from pathlib import Path
from llm_repo_agent.inspect_trace import main


def test_prompt_with_history_single_line(tmp_path, capsys):
    trace_file = tmp_path / "trace.jsonl"
    # Build a small sequence of events: read, tool_call, llm_request with newlines, tool_result
    ev1 = {"ts": 1.0, "kind": "tool_call", "payload": {"name": "list_files", "args": {"rel_dir": "."}}, "run_id": "r1"}
    ev2 = {"ts": 2.0, "kind": "tool_call", "payload": {"name": "read_file", "args": {"path": "foo.py"}}, "run_id": "r1"}
    ev3 = {
        "ts": 3.0,
        "kind": "llm_request",
        "payload": {"t": 0, "messages": [
            {"role": "system", "content": "System line1\nline2"},
            {"role": "user", "content": "User first\nsecond"},
        ]},
        "run_id": "r1",
    }
    ev4 = {"ts": 4.0, "kind": "tool_result", "payload": {"summary": "ok"}, "run_id": "r1"}

    trace_file.write_text("\n".join(json.dumps(e) for e in [ev1, ev2, ev3, ev4]) + "\n")

    rc = main([
        "--trace",
        str(trace_file),
        "--run",
        "r1",
        "--kind",
        "llm_request",
        "--index",
        "0",
        "--prompt-with-history",
        "--history-window",
        "2",
    ])
    assert rc == 0

    captured = capsys.readouterr()
    out = captured.out
    # PROMPT line should exist and be single-line (no embedded newlines inside the prompt)
    lines = out.splitlines()
    prompt_lines = [l for l in lines if l.startswith("PROMPT:")]
    assert len(prompt_lines) == 1
    prompt_line = prompt_lines[0]
    # Should contain flattened pieces from system and user messages
    assert "System line1 line2" in prompt_line
    assert "User first second" in prompt_line

    # HISTORY header and at least one history line should be present
    assert "HISTORY (events" in out
    assert "tool_call" in out or "tool_result" in out
