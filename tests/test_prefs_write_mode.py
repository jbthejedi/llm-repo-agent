"""Tests for prefs/rollouts.py write mode behavior."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

import llm_repo_agent.main as main
from llm_repo_agent.prefs.rollouts import PrefsConfig, PrefsRunner
from llm_repo_agent.prefs.schema import PreferenceMeta, PreferencePair


def _make_pair() -> PreferencePair:
    return PreferencePair(
        input_messages=[
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "GOAL:\nDo the thing."},
        ],
        preferred_content='{"type":"final","summary":"ok"}',
        non_preferred_content='{"type":"final","summary":"bad"}',
    )


def _make_meta() -> PreferenceMeta:
    return PreferenceMeta(
        task_id="task1",
        suite="suite1",
        model="model1",
        temperature=0.1,
        seed=1,
        scores={"preferred": 1.0, "non_preferred": 0.0},
        tests_ok={"preferred": True, "non_preferred": False},
        trace_ids={"preferred": "run_a", "non_preferred": "run_b"},
        rollout_counts={"total": 2},
    )


def test_prefs_config_invalid_write_mode():
    with pytest.raises(ValueError):
        PrefsConfig(write_mode="bad-mode")


def test_write_output_overwrite_replaces(tmp_path):
    out_path = tmp_path / "out.jsonl"
    meta_path = tmp_path / "out_meta.jsonl"
    out_path.write_text("old\n")
    meta_path.write_text("old-meta\n")

    cfg = PrefsConfig(
        out_path=out_path,
        meta_path=meta_path,
        write_mode="overwrite",
        progress=False,
    )
    runner = PrefsRunner(cfg)
    pair = _make_pair()
    meta = _make_meta()
    runner.pairs = [pair]
    runner.metas = [meta]

    runner.write_output()

    out_lines = out_path.read_text().splitlines()
    assert len(out_lines) == 1
    assert json.loads(out_lines[0]) == pair.to_together_format()

    meta_lines = meta_path.read_text().splitlines()
    assert len(meta_lines) == 1
    assert json.loads(meta_lines[0]) == meta.to_dict()


def test_write_output_append_adds_newline(tmp_path):
    out_path = tmp_path / "out.jsonl"
    meta_path = tmp_path / "out_meta.jsonl"
    out_path.write_text("old")
    meta_path.write_text("old-meta")

    cfg = PrefsConfig(
        out_path=out_path,
        meta_path=meta_path,
        write_mode="append",
        progress=False,
    )
    runner = PrefsRunner(cfg)
    pair = _make_pair()
    meta = _make_meta()
    runner.pairs = [pair]
    runner.metas = [meta]

    runner.write_output()

    out_text = out_path.read_text()
    assert out_text.startswith("old\n")
    out_lines = out_text.splitlines()
    assert out_lines[0] == "old"
    assert json.loads(out_lines[1]) == pair.to_together_format()

    meta_text = meta_path.read_text()
    assert meta_text.startswith("old-meta\n")
    meta_lines = meta_text.splitlines()
    assert meta_lines[0] == "old-meta"
    assert json.loads(meta_lines[1]) == meta.to_dict()


def test_cmd_prefs_passes_write_mode(monkeypatch, tmp_path):
    captured = {}

    def fake_run_rollouts(suite_path: Path, cfg: PrefsConfig) -> None:
        captured["suite_path"] = suite_path
        captured["cfg"] = cfg

    monkeypatch.setattr(main.prefs_rollouts, "run_rollouts", fake_run_rollouts)

    args = Namespace(
        suite=str(tmp_path / "suite.json"),
        trace_dir=str(tmp_path / "trace"),
        out=str(tmp_path / "out.jsonl"),
        data_write_mode="append",
        rollouts=2,
        max_workers=4,
        sandbox=True,
        keep_sandbox=False,
        test_policy="on_write",
        max_iters=5,
        model="model",
        llm_provider="together",
        together_api_key=None,
        tool_protocol="json",
        temperature=0.1,
        seed=7,
        quiet=True,
    )

    main.cmd_prefs(args)

    assert captured["suite_path"] == Path(args.suite)
    assert captured["cfg"].write_mode == "append"
    assert captured["cfg"].tool_protocol == "json"
