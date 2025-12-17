import json
from pathlib import Path
from llm_repo_agent.inspect_trace import main


def test_prompt_with_history_preserve_newlines(tmp_path, capsys):
    trace_file = tmp_path / "trace.jsonl"
    # Build a small sequence of events: tool_call, llm_request with newlines, tool_result
    ev1 = {"ts": 1.0, "kind": "tool_call", "payload": {"name": "list_files", "args": {"rel_dir": "."}}, "run_id": "r1"}
    ev2 = {
        "ts": 2.0,
        "kind": "llm_request",
        "payload": {"t": 0, "messages": [
            {"role": "system", "content": "SysA\nSysB"},
            {"role": "user", "content": "UserA\nUserB"},
        ]},
        "run_id": "r1",
    }
    ev3 = {"ts": 3.0, "kind": "tool_result", "payload": {"summary": "ok"}, "run_id": "r1"}

    trace_file.write_text("\n".join(json.dumps(e) for e in [ev1, ev2, ev3]) + "\n")

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
        "--preserve-newlines",
        "--history-window",
        "1",
    ])
    assert rc == 0

    captured = capsys.readouterr()
    out = captured.out
    # Should print PROMPT: header and then indented lines preserving newlines
    assert "PROMPT:" in out
    # The printed block includes role on the first line and subsequent lines for wrapped content
    assert "  system: SysA" in out
    assert "  SysB" in out
    assert "  user: UserA" in out
    assert "  UserB" in out
    # HISTORY header should still be present
    assert "HISTORY (events" in out
