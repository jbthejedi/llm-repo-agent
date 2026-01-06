import json
from pathlib import Path
from types import SimpleNamespace

import llm_repo_agent.agent as agent_module
import llm_repo_agent.tools as tools_module
import llm_repo_agent.trace as trace_module
import llm_repo_agent.actions as actions_module


def test_agent_logs_trailing_json(tmp_path):
    # Dummy LLM that sets _last_trailing and returns a final object
    class DummyLLM:
        def next_action(self, messages):
            self._last_trailing = '{"type":"final","summary":"second"}'
            return actions_module.FinalAction(summary="first", changes=[])

    repo_root = Path(".")
    tools = tools_module.RepoTools(repo_root=repo_root)
    trace_file = tmp_path / "trace.jsonl"
    trace = trace_module.Trace(trace_file, run_id="test")

    llm = DummyLLM()
    agent = agent_module.RepoAgent(llm=llm, tools=tools, trace=trace, cfg=agent_module.AgentConfig())

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
