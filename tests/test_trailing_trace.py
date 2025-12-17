import json
from pathlib import Path
from types import SimpleNamespace

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.llm import LLM
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace


def test_agent_logs_trailing_json(tmp_path):
    # Dummy LLM that sets _last_trailing and returns a final object
    from llm_repo_agent.actions import FinalAction
    class DummyLLM:
        def next_action(self, messages):
            self._last_trailing = '{"type":"final","summary":"second"}'
            return FinalAction(summary="first", changes=[])

    repo_root = Path(".")
    tools = RepoTools(repo_root=repo_root)
    trace_file = tmp_path / "trace.jsonl"
    trace = Trace(trace_file, run_id="test")

    llm = DummyLLM()
    agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig())

    res = agent.run(goal="Just test logging", test_cmd=[])
    assert isinstance(res, dict)

    # Read trace file and find the llm_trailing_text event
    lines = trace_file.read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(l)['kind'] for l in lines]
    assert 'llm_trailing_text' in kinds
    # ensure payload contains our trailing snippet
    ev = None
    for l in lines:
        data = json.loads(l)
        if data['kind'] == 'llm_trailing_text':
            ev = data
            break
    assert ev is not None
    assert 'trailing' in ev['payload']
    assert 'second' in ev['payload']['trailing']
