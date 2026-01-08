import json
from pathlib import Path

from llm_repo_agent.agent import AgentConfig, RepoAgent
from llm_repo_agent.actions import FinalAction
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace


def test_llm_usage_logged(tmp_path):
    class DummyLLM:
        def start_conversation(self, system_prompt, user_goal):
            self._messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_goal},
            ]

        def next_action(self, tool_result=None):
            self._last_usage = {
                "prompt_tokens": 12,
                "completion_tokens": 34,
                "total_tokens": 46,
            }
            return FinalAction(summary="Done", changes=[])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    tools = RepoTools(repo_root=repo_root)
    trace_path = tmp_path / "trace.jsonl"
    trace = Trace(trace_path, run_id="run123")
    agent = RepoAgent(llm=DummyLLM(), tools=tools, trace=trace, cfg=AgentConfig(progress=False))

    agent.run(goal="Test usage logging", test_cmd=[])

    kinds = []
    payloads = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        evt = json.loads(line)
        kinds.append(evt["kind"])
        if evt["kind"] == "llm_usage":
            payloads.append(evt["payload"])

    assert "llm_usage" in kinds
    assert payloads
    payload = payloads[0]
    assert payload["prompt_tokens"] == 12
    assert payload["completion_tokens"] == 34
    assert payload["total_tokens"] == 46
