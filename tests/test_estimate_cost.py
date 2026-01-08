import json
from pathlib import Path

from llm_repo_agent.estimate_cost import collect_usage_stats, count_pairs, estimate_cost


def test_estimate_cost_from_trace_and_dataset(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "task_run.jsonl"
    events = [
        {"kind": "llm_usage", "payload": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
        {"kind": "llm_usage", "payload": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}},
        {"kind": "other", "payload": {}},
    ]
    trace_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(["{}", "{}", "{}"]) + "\n", encoding="utf-8")

    usage = collect_usage_stats(trace_dir)
    assert usage.calls == 2
    assert usage.prompt_tokens == 180
    assert usage.completion_tokens == 70
    assert usage.total_tokens == 250

    pair_count = count_pairs(dataset)
    assert pair_count == 3

    result = estimate_cost(
        usage=usage,
        pair_count=pair_count,
        price_in=0.30,
        price_out=0.30,
        target_pairs=3000,
    )

    assert result["calls"] == 2
    assert result["pairs"] == 3
    assert result["avg_prompt_tokens"] == 90.0
    assert result["avg_completion_tokens"] == 35.0
    assert result["calls_per_pair"] == 2 / 3
