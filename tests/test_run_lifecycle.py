import json
from pathlib import Path

import llm_repo_agent.agent as agent_module
import llm_repo_agent.tools as tools_module
import llm_repo_agent.trace as trace_module
import llm_repo_agent.actions as actions_module


def test_run_start_and_end_and_history(tmp_path):
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
                return actions_module.ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 10})
            else:
                return actions_module.FinalAction(summary="Done", changes=[])

    repo_root = Path(".")
    tools = tools_module.RepoTools(repo_root=repo_root)
    trace_file = tmp_path / "trace.jsonl"
    trace = trace_module.Trace(trace_file, run_id="run123")

    llm = DummyLLM()
    agent = agent_module.RepoAgent(llm=llm, tools=tools, trace=trace, cfg=agent_module.AgentConfig())

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
