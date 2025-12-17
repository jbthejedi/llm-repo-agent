from pathlib import Path

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.llm import LLM
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace

from llm_repo_agent.actions import ToolCallAction, FinalAction


class DummyLLM:
    """LLM that writes a file then returns final."""
    def next_action(self, messages):
        # First call: ask to write file
        if not hasattr(self, 'calls'):
            self.calls = 1
            return ToolCallAction(name='write_file', args={'rel_path': 'foo.txt', 'content': 'x'})
        # second: final
        self.calls += 1
        return FinalAction(summary='Done', changes=[])


def test_final_contains_test_result(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    tools = RepoTools(repo_root=repo_root)
    trace = Trace(tmp_path / "trace.jsonl", run_id='r1')

    llm = DummyLLM()
    agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig())

    # Use a quick passing command
    test_cmd = ["python", "-c", "print('ok')"]
    final = agent.run(goal="do thing", test_cmd=test_cmd)

    assert isinstance(final, dict)
    assert final.get('type') == 'final'
    assert 'test_result' in final
    assert final['test_result']['ok'] is True
    assert 'ok' in final['test_result']['output_snippet']
