from argparse import Namespace
from types import SimpleNamespace
from pathlib import Path

import llm_repo_agent.main as main


def test_cmd_run_passes_tool_protocol(monkeypatch, tmp_path):
    captured = {}

    def fake_build(cfg):
        captured["cfg"] = cfg
        return SimpleNamespace(model="gpt-4.1-mini", tool_protocol=cfg.tool_protocol)

    def fake_run(self, goal, test_cmd):
        return {"type": "final", "summary": "ok", "changes": []}

    monkeypatch.setattr(main.llm_module.LLMFactory, "build", fake_build)
    monkeypatch.setattr(main.agent_module.RepoAgent, "run", fake_run)

    args = Namespace(
        repo=str(tmp_path),
        goal="Fix it",
        trace=str(tmp_path / "trace.jsonl"),
        test="",
        llm_provider="openai",
        model="gpt-4.1-mini",
        together_api_key=None,
        tool_protocol="json",
        sandbox=False,
        sandbox_dir=None,
        keep_sandbox=False,
        test_policy="on_write",
    )

    main.cmd_run(args)
    assert captured["cfg"].tool_protocol == "json"


def test_cmd_eval_passes_tool_protocol(monkeypatch, tmp_path):
    captured = {}

    def fake_load_suite(path: Path):
        return SimpleNamespace(name="suite", description=None, tasks=[])

    class DummyRunner:
        def __init__(self, **kwargs):
            captured["cfg"] = kwargs.get("cfg")
        def run_suite(self, suite):
            return []

    monkeypatch.setattr(main.eval_tasks, "load_suite", fake_load_suite)
    monkeypatch.setattr(main.eval_runner, "EvalRunner", DummyRunner)

    args = Namespace(
        suite=str(tmp_path / "suite.json"),
        trace_dir=str(tmp_path / "trace"),
        report=None,
        sandbox=True,
        keep_sandbox=False,
        test_policy="on_write",
        max_iters=5,
        model="gpt-4.1-mini",
        llm_provider="openai",
        together_api_key=None,
        tool_protocol="json",
        quiet=True,
    )

    main.cmd_eval(args)
    assert captured["cfg"].tool_protocol == "json"
