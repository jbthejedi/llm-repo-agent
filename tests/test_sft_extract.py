import json
from argparse import Namespace
from pathlib import Path

import llm_repo_agent.main as main
from llm_repo_agent.sft.config import SFTExtractConfig
from llm_repo_agent.sft.extract import extract_sft_samples


def _write_trace(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_sft_extract_basic(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "run.jsonl"

    events = [
        {
            "kind": "llm_request",
            "payload": {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "GOAL:\nFix it"},
                ]
            },
        },
        {
            "kind": "llm_action",
            "payload": {
                "action": {
                    "type": "tool_call",
                    "name": "list_files",
                    "args": {"rel_dir": ".", "max_files": 10},
                }
            },
        },
        {
            "kind": "tool_result",
            "payload": {"tool": "list_files", "obs": {"ok": True, "output": "a.py", "meta": {}}},
        },
        {
            "kind": "run_end",
            "payload": {"state": {"last_test": {"ok": True, "output": "ok"}}},
        },
    ]
    _write_trace(trace_file, events)

    cfg = SFTExtractConfig(
        trace_dir=trace_dir,
        output_path=tmp_path / "out.jsonl",
        require_success=True,
        require_valid_tool_ok=True,
        progress=False,
    )

    samples = extract_sft_samples(cfg)
    assert len(samples) == 1
    messages = samples[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[-1]["role"] == "assistant"
    action = json.loads(messages[-1]["content"])
    assert action["name"] == "list_files"
    assert action["args"]["rel_dir"] == "."


def test_sft_extract_skips_invalid_tool_args(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "run.jsonl"

    events = [
        {
            "kind": "llm_request",
            "payload": {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "GOAL:\nFix"}]},
        },
        {
            "kind": "llm_action",
            "payload": {
                "action": {"type": "tool_call", "name": "list_files", "args": {"max_files": 10}},
            },
        },
        {
            "kind": "tool_result",
            "payload": {"tool": "list_files", "obs": {"ok": True, "output": "a.py", "meta": {}}},
        },
        {"kind": "run_end", "payload": {"state": {"last_test": {"ok": True, "output": "ok"}}}},
    ]
    _write_trace(trace_file, events)

    cfg = SFTExtractConfig(
        trace_dir=trace_dir,
        output_path=tmp_path / "out.jsonl",
        require_success=True,
        require_valid_tool_ok=True,
        progress=False,
    )

    samples = extract_sft_samples(cfg)
    assert samples == []


def test_sft_extract_requires_success(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "run.jsonl"

    events = [
        {
            "kind": "llm_request",
            "payload": {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "GOAL:\nFix"}]},
        },
        {
            "kind": "llm_action",
            "payload": {
                "action": {
                    "type": "tool_call",
                    "name": "list_files",
                    "args": {"rel_dir": ".", "max_files": 10},
                }
            },
        },
        {
            "kind": "tool_result",
            "payload": {"tool": "list_files", "obs": {"ok": True, "output": "a.py", "meta": {}}},
        },
        {"kind": "run_end", "payload": {"state": {"last_test": {"ok": False, "output": "fail"}}}},
    ]
    _write_trace(trace_file, events)

    cfg = SFTExtractConfig(
        trace_dir=trace_dir,
        output_path=tmp_path / "out.jsonl",
        require_success=True,
        require_valid_tool_ok=True,
        progress=False,
    )

    samples = extract_sft_samples(cfg)
    assert samples == []


def test_sft_extract_allows_failed_tool_when_configured(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "run.jsonl"

    events = [
        {
            "kind": "llm_request",
            "payload": {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "GOAL:\nFix"}]},
        },
        {
            "kind": "llm_action",
            "payload": {
                "action": {
                    "type": "tool_call",
                    "name": "list_files",
                    "args": {"rel_dir": ".", "max_files": 10},
                }
            },
        },
        {
            "kind": "tool_result",
            "payload": {"tool": "list_files", "obs": {"ok": False, "output": "bad", "meta": {}}},
        },
        {"kind": "run_end", "payload": {"state": {"last_test": {"ok": True, "output": "ok"}}}},
    ]
    _write_trace(trace_file, events)

    cfg = SFTExtractConfig(
        trace_dir=trace_dir,
        output_path=tmp_path / "out.jsonl",
        require_success=True,
        require_valid_tool_ok=False,
        progress=False,
    )

    samples = extract_sft_samples(cfg)
    assert len(samples) == 1


def test_sft_extract_converts_tool_messages_and_truncates(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "run.jsonl"

    events = [
        {
            "kind": "llm_request",
            "payload": {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "GOAL:\nFix it"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "X" * 50},
                ]
            },
        },
        {
            "kind": "llm_action",
            "payload": {
                "action": {
                    "type": "tool_call",
                    "name": "read_file",
                    "args": {"rel_path": "a.py", "max_chars": 10},
                }
            },
        },
        {
            "kind": "tool_result",
            "payload": {"tool": "read_file", "obs": {"ok": True, "output": "ok", "meta": {}}},
        },
        {"kind": "run_end", "payload": {"state": {"last_test": {"ok": True, "output": "ok"}}}},
    ]
    _write_trace(trace_file, events)

    cfg = SFTExtractConfig(
        trace_dir=trace_dir,
        output_path=tmp_path / "out.jsonl",
        require_success=True,
        require_valid_tool_ok=True,
        max_context_chars=10,
        progress=False,
    )

    samples = extract_sft_samples(cfg)
    assert len(samples) == 1
    messages = samples[0]["messages"]
    tool_result_msg = [m for m in messages if m["role"] == "user" and "[tool_result]" in m["content"]][0]
    assert len(tool_result_msg["content"]) <= len("[tool_result]\n") + 10


def test_sft_extract_handles_tool_calls_in_messages(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "run.jsonl"

    events = [
        {
            "kind": "llm_request",
            "payload": {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "GOAL:\nFix it"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "list_files",
                                    "arguments": "{\"rel_dir\": \".\", \"max_files\": 5}",
                                },
                            }
                        ],
                    },
                ]
            },
        },
        {
            "kind": "llm_action",
            "payload": {
                "action": {
                    "type": "tool_call",
                    "name": "read_file",
                    "args": {"rel_path": "a.py", "max_chars": 10},
                }
            },
        },
        {
            "kind": "tool_result",
            "payload": {"tool": "read_file", "obs": {"ok": True, "output": "ok", "meta": {}}},
        },
        {"kind": "run_end", "payload": {"state": {"last_test": {"ok": True, "output": "ok"}}}},
    ]
    _write_trace(trace_file, events)

    cfg = SFTExtractConfig(
        trace_dir=trace_dir,
        output_path=tmp_path / "out.jsonl",
        require_success=True,
        require_valid_tool_ok=True,
        progress=False,
    )

    samples = extract_sft_samples(cfg)
    assert len(samples) == 1
    messages = samples[0]["messages"]
    assert any(
        m["role"] == "assistant" and "\"name\": \"list_files\"" in m["content"]
        for m in messages
    )


def test_cmd_sft_extract_writes_output(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "run.jsonl"
    output_path = tmp_path / "out.jsonl"

    events = [
        {
            "kind": "llm_request",
            "payload": {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "GOAL:\nFix"}]},
        },
        {
            "kind": "llm_action",
            "payload": {
                "action": {
                    "type": "tool_call",
                    "name": "list_files",
                    "args": {"rel_dir": ".", "max_files": 10},
                }
            },
        },
        {
            "kind": "tool_result",
            "payload": {"tool": "list_files", "obs": {"ok": True, "output": "a.py", "meta": {}}},
        },
        {"kind": "run_end", "payload": {"state": {"last_test": {"ok": True, "output": "ok"}}}},
    ]
    _write_trace(trace_file, events)

    args = Namespace(
        trace_dir=str(trace_dir),
        output=str(output_path),
        require_success=True,
        require_valid_tool_ok=True,
        max_context_chars=8000,
        quiet=True,
    )

    main.cmd_sft_extract(args)
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
