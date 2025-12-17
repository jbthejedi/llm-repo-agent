import json
from pathlib import Path
from llm_repo_agent.trace import Trace
from llm_repo_agent.inspect_trace import main


def test_pretty_only_prompt(tmp_path, capsys):
    trace_file = tmp_path / "trace.jsonl"
    ev = {
        "ts": 1.0,
        "kind": "llm_request",
        "payload": {
            "t": 0,
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
            ],
        },
        "run_id": "r1",
    }
    trace_file.write_text(json.dumps(ev) + "\n")

    # Invoke main with pretty-only-prompt
    rc = main([
        "--trace",
        str(trace_file),
        "--run",
        "r1",
        "--kind",
        "llm_request",
        "--index",
        "0",
        "--pretty-only-prompt",
    ])
    assert rc == 0

    captured = capsys.readouterr()
    out = captured.out
    # Should include message contents and role labels
    assert "system: System message" in out
    assert "user: User message" in out
    # Should NOT include metadata lines like [llm_request] or (run=
    assert "[llm_request]" not in out
    assert "(run=" not in out
