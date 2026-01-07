import json
from pathlib import Path

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace


def test_agent_rejects_raw_dict_action(tmp_path):
    # Dummy LLM that returns a raw dict (legacy behavior)
    class DummyLLM:
        def start_conversation(self, system_prompt, user_goal):
            self._messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_goal},
            ]

        def next_action(self, tool_result=None):
            return {"type": "tool_call", "name": "list_files", "args": {"rel_dir": ".", "max_files": 5}}

    repo_root = Path(".")
    tools = RepoTools(repo_root=repo_root)
    trace_file = tmp_path / "trace.jsonl"
    trace = Trace(trace_file, run_id="test")

    llm = DummyLLM()
    agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig())

    try:
        agent.run(goal="Should fail", test_cmd=[])
        assert False, "Expected RuntimeError due to non-typed action"
    except RuntimeError as e:
        assert "must return a typed Action" in str(e)

    # Ensure we logged the parse error in the trace
    lines = trace_file.read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(l)['kind'] for l in lines]
    assert 'llm_parse_error' in kinds
