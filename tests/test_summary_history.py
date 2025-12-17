from llm_repo_agent.summary import RunSummary, TestResult, summarize_history
from llm_repo_agent.history import History


def test_history_append_and_prompt():
    h = History()
    h.append_tool_call("list_files", {"rel_dir": ".", "max_files": 10})
    h.append_observation("list_files", {"ok": True, "output": "a\nb\n"})
    h.append_tool_call("read_file", {"rel_path": "foo.py", "max_chars": 100})
    p = h.to_prompt_list(2)
    assert len(p) == 2
    assert p[0]["kind"] == "observation" or p[0]["kind"] == "tool_call"
    assert h.has_any_observation()
    assert not h.detect_loop(2)
    h.append_tool_call("list_files", {"rel_dir": ".", "max_files": 10})
    h.append_tool_call("list_files", {"rel_dir": ".", "max_files": 10})
    assert h.detect_loop(2)


def test_history_touched_files():
    h = History()
    h.append_observation("write_file", {"ok": True, "output": "Wrote a.py", "meta": {"rel_path": "a.py"}})
    h.append_observation("write_file", {"ok": True, "output": "Wrote b.py", "meta": {"rel_path": "b.py"}})
    # duplicate should not repeat
    h.append_observation("write_file", {"ok": True, "output": "Wrote a.py", "meta": {"rel_path": "a.py"}})
    assert h.touched_files() == ["a.py", "b.py"]


def test_summarize_history():
    h = History()
    h.append_driver_note("loop detected")
    h.append_observation("write_file", {"ok": True, "output": "Wrote a.py", "meta": {"rel_path": "a.py"}})
    h.append_observation("driver.run_tests", {"ok": True, "output": "ok"})

    summary = summarize_history(h, run_id="run123")
    assert summary.notes == ["loop detected"]
    assert summary.files_touched == ["a.py"]
    assert isinstance(summary.last_test, TestResult)
    assert summary.last_test.ok is True
    assert summary.run_id == "run123"
