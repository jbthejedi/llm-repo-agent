import json
from pathlib import Path

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.actions import ToolCallAction, FinalAction
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def start_conversation(self, system_prompt, user_goal):
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_goal},
        ]

    def next_action(self, tool_result=None):
        self.calls += 1
        if self.calls == 1:
            return ToolCallAction(name="write_file", args={"rel_path": "foo.txt", "content": "x"})
        return FinalAction(summary="Done", changes=[])


def _count_test_events(trace_file: Path) -> int:
    lines = trace_file.read_text(encoding="utf-8").splitlines()
    return sum(1 for l in lines if json.loads(l).get("kind") == "tests")


def test_test_policy_on_final_runs_once(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    trace_file = tmp_path / "trace.jsonl"

    agent = RepoAgent(
        llm=DummyLLM(),
        tools=RepoTools(repo_root=repo_root),
        trace=Trace(trace_file, run_id="r1"),
        cfg=AgentConfig(test_policy="on_final"),
    )

    final = agent.run(goal="goal", test_cmd=["python", "-c", "print('ok')"])
    assert final.get("type") == "final"
    assert final.get("test_result") is not None

    assert _count_test_events(trace_file) == 1


def test_test_policy_never_runs_zero_tests(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    trace_file = tmp_path / "trace.jsonl"

    agent = RepoAgent(
        llm=DummyLLM(),
        tools=RepoTools(repo_root=repo_root),
        trace=Trace(trace_file, run_id="r2"),
        cfg=AgentConfig(test_policy="never"),
    )

    final = agent.run(goal="goal", test_cmd=["python", "-c", "print('ok')"])
    assert final.get("type") == "final"
    assert "test_result" not in final

    assert _count_test_events(trace_file) == 0
