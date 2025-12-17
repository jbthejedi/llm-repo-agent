import json
from pathlib import Path
from llm_repo_agent.inspect_trace import main


def test_dump_prompt_writes_file(tmp_path, capsys):
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

    dump_path = tmp_path / "prompt.txt"

    rc = main([
        "--trace",
        str(trace_file),
        "--run",
        "r1",
        "--kind",
        "llm_request",
        "--index",
        "0",
        "--dump-prompt",
        str(dump_path),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Wrote prompt to" in captured.out

    assert dump_path.exists()
    content = dump_path.read_text()
    assert "system: System message" in content
    assert "user: User message" in content
