from pathlib import Path

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace
from llm_repo_agent.actions import ToolCallAction, FinalAction


def test_final_changes_fallback(tmp_path):
    """If the model omits changes but files were touched, we auto-populate."""
    class DummyLLM:
        def __init__(self):
            self.calls = 0
        def next_action(self, messages):
            self.calls += 1
            if self.calls == 1:
                return ToolCallAction(name="write_file", args={"rel_path": "foo.txt", "content": "x"})
            return FinalAction(summary="Done", changes=[])
    repo_root = tmp_path
    tools = RepoTools(repo_root=repo_root)
    trace = Trace(tmp_path / "trace.jsonl", run_id="r1")
    agent = RepoAgent(llm=DummyLLM(), tools=tools, trace=trace, cfg=AgentConfig())

    res = agent.run(goal="fallback", test_cmd=[])
    assert res["type"] == "final"
    assert res["changes"], "Expected changes to be filled from touched files"
    assert res["changes"][0]["path"] == "foo.txt"
