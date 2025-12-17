import json
from pathlib import Path

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace


def test_run_start_and_end_and_history(tmp_path):
    from llm_repo_agent.actions import ToolCallAction, FinalAction
    class DummyLLM:
        def __init__(self):
            self.calls = 0
        def next_action(self, messages):
            self.calls += 1
            if self.calls == 1:
                return ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 10})
            else:
                return FinalAction(summary="Done", changes=[])

    repo_root = Path(".")
    tools = RepoTools(repo_root=repo_root)
    trace_file = tmp_path / "trace.jsonl"
    trace = Trace(trace_file, run_id="run123")

    llm = DummyLLM()
    agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig())

    res = agent.run(goal="Lifecycle test", test_cmd=[])
    assert isinstance(res, dict)

    lines = trace_file.read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(l)['kind'] for l in lines]
    assert 'run_start' in kinds
    assert 'run_end' in kinds

    history = trace.get_run_history('run123')
    # Expect at least one tool_call and one observation
    kinds = [h['kind'] for h in history]
    assert 'tool_call' in kinds
    assert 'observation' in kinds
