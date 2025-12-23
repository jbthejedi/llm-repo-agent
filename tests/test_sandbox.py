from pathlib import Path

from llm_repo_agent.sandbox import materialize_repo_sandbox, cleanup_sandbox


def test_materialize_and_cleanup_sandbox(tmp_path):
    src = tmp_path / "src_repo"
    src.mkdir()
    (src / "a.txt").write_text("hello")

    sandbox = materialize_repo_sandbox(src)
    assert sandbox.root.exists()
    assert (sandbox.root / "a.txt").read_text() == "hello"

    # mutate sandbox; source should stay intact
    (sandbox.root / "a.txt").write_text("changed")
    assert (src / "a.txt").read_text() == "hello"

    cleanup_sandbox(sandbox)
    assert not sandbox.root.exists()


def test_materialize_with_destination(tmp_path):
    src = tmp_path / "src_repo"
    src.mkdir()
    (src / "b.txt").write_text("hi")

    dest = tmp_path / "custom_sandbox"
    sandbox = materialize_repo_sandbox(src, dest)
    assert sandbox.root == dest.resolve()
    assert (sandbox.root / "b.txt").read_text() == "hi"
