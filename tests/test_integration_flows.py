"""Integration-style tests for eval and prefs flows."""

import json
from pathlib import Path

from llm_repo_agent.actions import FinalAction, ToolCallAction
from llm_repo_agent.eval.runner import EvalConfig, EvalRunner
from llm_repo_agent.eval.tasks import EvalSuite, TaskSpec, save_suite
from llm_repo_agent.llm import LLMFactory
from llm_repo_agent.prefs.rollouts import PrefsConfig, run_rollouts


class SeededDummyLLM:
    """Dummy LLM that varies behavior by seed for contrast."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.calls = 0
        self._messages = []
        self._last_tool_call_id = None
        self._last_raw = None

    def start_conversation(self, system_prompt: str, user_goal: str) -> None:
        self.calls = 0
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_goal},
        ]

    def next_action(self, tool_result: str | None = None):
        self.calls += 1
        if self.seed % 2 == 0:
            action = FinalAction(summary="Done.", changes=[])
            self._last_raw = action.to_dict()
            return action
        if self.calls == 1:
            action = ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 5})
            self._last_tool_call_id = "call_1"
            self._last_raw = action.to_dict()
            return action
        action = FinalAction(summary="Done.", changes=[])
        self._last_raw = action.to_dict()
        return action


def _write_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "hello.txt").write_text("hello")
    return repo_root


def test_eval_runner_integration(tmp_path):
    repo_root = _write_repo(tmp_path)
    task = TaskSpec(task_id="list_files", repo=str(repo_root), goal="List files")

    cfg = EvalConfig(
        trace_dir=tmp_path / "traces",
        sandbox=False,
        test_policy="never",
        progress=False,
    )
    runner = EvalRunner(cfg=cfg, llm_factory=lambda: SeededDummyLLM(seed=1))
    result = runner.run_task(task)

    assert result.error is None
    assert result.tool_calls == 1
    assert result.steps >= 2

    trace_files = list(cfg.trace_dir.glob("list_files_*.jsonl"))
    assert trace_files, "Expected a trace file to be written."

    events = [json.loads(line)["kind"] for line in trace_files[0].read_text().splitlines()]
    assert "run_start" in events
    assert "run_end" in events


def test_prefs_run_rollouts_integration(tmp_path, monkeypatch):
    repo_root = _write_repo(tmp_path)
    suite = EvalSuite(name="tiny_suite", tasks=[
        TaskSpec(task_id="list_files", repo=str(repo_root), goal="List files"),
    ])
    suite_path = tmp_path / "suite.json"
    save_suite(suite, suite_path)

    def dummy_builder(cfg):
        return SeededDummyLLM(seed=cfg.seed or 0)

    monkeypatch.setitem(LLMFactory._registry, "openai", dummy_builder)

    cfg = PrefsConfig(
        trace_dir=tmp_path / "traces",
        out_path=tmp_path / "prefs.jsonl",
        meta_path=tmp_path / "prefs_meta.jsonl",
        write_mode="overwrite",
        rollouts=2,
        sandbox=False,
        test_policy="never",
        llm_provider="openai",
        progress=False,
    )

    runner = run_rollouts(suite_path, cfg)

    out_lines = cfg.out_path.read_text().splitlines()
    assert len(out_lines) == 1
    out_obj = json.loads(out_lines[0])
    assert set(out_obj.keys()) == {"input", "preferred_output", "non_preferred_output"}

    meta_lines = cfg.meta_path.read_text().splitlines()
    assert len(meta_lines) == 1
    meta_obj = json.loads(meta_lines[0])
    assert meta_obj["task_id"] == "list_files"

    assert len(runner.pairs) == 1
    assert runner.no_contrast_tasks == []
