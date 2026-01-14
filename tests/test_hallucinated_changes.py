"""Tests for rejection of finals with hallucinated changes.

When a model emits type='final' with a non-empty changes array but never
actually called write_file, the agent should reject the final and continue.
"""

from pathlib import Path

from llm_repo_agent.agent import RepoAgent, AgentConfig
from llm_repo_agent.tools import RepoTools
from llm_repo_agent.trace import Trace
from llm_repo_agent.actions import ToolCallAction, FinalAction


class HallucinatingLLM:
    """LLM that claims to make changes without calling write_file."""

    def __init__(self):
        self.calls = 0
        self._messages = []
        self._pending_driver_notes = []

    def start_conversation(self, system_prompt, user_goal):
        self.calls = 0
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_goal},
        ]

    def add_driver_note(self, note):
        self._pending_driver_notes.append(note)

    def next_action(self, tool_result=None):
        self.calls += 1
        # First call: list files (mandatory first action)
        if self.calls == 1:
            return ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 10})
        # Second call: hallucinate a change without writing
        if self.calls == 2:
            return FinalAction(
                summary="Fixed the bug",
                changes=[{"path": "foo.py", "description": "Fixed the bug"}],
            )
        # After rejection, actually write the file
        if self.calls == 3:
            return ToolCallAction(name="write_file", args={"rel_path": "foo.py", "content": "fixed"})
        # Final after actual write
        return FinalAction(
            summary="Fixed the bug",
            changes=[{"path": "foo.py", "description": "Fixed the bug"}],
        )


def test_agent_rejects_hallucinated_changes(tmp_path):
    """Agent should reject final with changes if no write_file was called."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "foo.py").write_text("buggy")

    tools = RepoTools(repo_root=repo_root)
    trace = Trace(tmp_path / "trace.jsonl", run_id="r1")
    llm = HallucinatingLLM()
    agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig(progress=False))

    final = agent.run(goal="Fix the bug", test_cmd=[])

    # Agent should have called next_action 4 times:
    # 1. list_files
    # 2. hallucinated final (rejected)
    # 3. write_file
    # 4. final (accepted)
    assert llm.calls == 4, f"Expected 4 calls but got {llm.calls}"

    # Final should be accepted now with actual changes
    assert final["type"] == "final"
    assert final["changes"], "Expected changes to be present"
    assert (repo_root / "foo.py").read_text() == "fixed"


def test_agent_accepts_final_with_no_changes():
    """Agent should accept final with empty changes (no hallucination)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        class NoChangeLLM:
            def __init__(self):
                self.calls = 0
                self._messages = []
                self._pending_driver_notes = []

            def start_conversation(self, system_prompt, user_goal):
                self._messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_goal},
                ]

            def add_driver_note(self, note):
                pass

            def next_action(self, tool_result=None):
                self.calls += 1
                if self.calls == 1:
                    return ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 10})
                # Final with no changes - should be accepted
                return FinalAction(summary="No changes needed", changes=[])

        tools = RepoTools(repo_root=repo_root)
        trace = Trace(tmp_path / "trace.jsonl", run_id="r1")
        llm = NoChangeLLM()
        agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig(progress=False))

        final = agent.run(goal="Check files", test_cmd=[])

        # Should accept on second call (no rejection)
        assert llm.calls == 2
        assert final["type"] == "final"
        assert final["changes"] == []


def test_agent_accepts_final_after_actual_write(tmp_path):
    """Agent should accept final with changes if write_file was called."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "bar.py").write_text("original")

    class WriteFirstLLM:
        def __init__(self):
            self.calls = 0
            self._messages = []
            self._pending_driver_notes = []

        def start_conversation(self, system_prompt, user_goal):
            self._messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_goal},
            ]

        def add_driver_note(self, note):
            pass

        def next_action(self, tool_result=None):
            self.calls += 1
            if self.calls == 1:
                return ToolCallAction(name="list_files", args={"rel_dir": ".", "max_files": 10})
            if self.calls == 2:
                return ToolCallAction(name="write_file", args={"rel_path": "bar.py", "content": "updated"})
            return FinalAction(
                summary="Updated file",
                changes=[{"path": "bar.py", "description": "Updated content"}],
            )

    tools = RepoTools(repo_root=repo_root)
    trace = Trace(tmp_path / "trace.jsonl", run_id="r1")
    llm = WriteFirstLLM()
    agent = RepoAgent(llm=llm, tools=tools, trace=trace, cfg=AgentConfig(progress=False))

    final = agent.run(goal="Update file", test_cmd=[])

    # Should accept on third call (no rejection because write happened)
    assert llm.calls == 3
    assert final["type"] == "final"
    assert final["changes"]
    assert (repo_root / "bar.py").read_text() == "updated"
