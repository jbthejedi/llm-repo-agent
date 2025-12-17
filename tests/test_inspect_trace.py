import json
from pathlib import Path
from llm_repo_agent.trace import Trace
from llm_repo_agent.inspect_trace import pretty_print_events


def test_pretty_print_full(tmp_path, capsys):
    trace_file = tmp_path / "trace.jsonl"
    # create a fake llm_request event with long content
    ev = {
        "ts": 1.0,
        "kind": "llm_request",
        "payload": {
            "t": 0,
            "messages": [
                {"role": "system", "content": "A" * 5000},
                {"role": "user", "content": "B" * 5000},
            ],
        },
        "run_id": "r1",
    }
    trace_file.write_text(json.dumps(ev) + "\n")

    # read via Trace.iter_run_events and pretty_print full
    trace = Trace(trace_file, run_id="r1")
    events = list(trace.iter_run_events("r1"))
    # should not raise and should include full content when full=True
    pretty_print_events(events, full=True)
    captured = capsys.readouterr()
    assert "role: system" in captured.out
    assert "role: user" in captured.out
    assert 'A' * 100 in captured.out
