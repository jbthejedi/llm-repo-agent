from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class UsageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    files_scanned: int = 0

    def avg_prompt(self) -> float:
        return self.prompt_tokens / self.calls if self.calls else 0.0

    def avg_completion(self) -> float:
        return self.completion_tokens / self.calls if self.calls else 0.0

    def avg_total(self) -> float:
        return self.total_tokens / self.calls if self.calls else 0.0


def collect_usage_stats(trace_dir: Path) -> UsageStats:
    stats = UsageStats()
    for path in sorted(trace_dir.rglob("*.jsonl")):
        stats.files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("kind") != "llm_usage":
                continue
            payload = evt.get("payload", {})
            prompt_tokens = payload.get("prompt_tokens")
            completion_tokens = payload.get("completion_tokens")
            total_tokens = payload.get("total_tokens")
            if prompt_tokens is None or completion_tokens is None:
                continue
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
            stats.calls += 1
            stats.prompt_tokens += int(prompt_tokens)
            stats.completion_tokens += int(completion_tokens)
            stats.total_tokens += int(total_tokens)
    return stats


def count_pairs(dataset_path: Path) -> int:
    count = 0
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def estimate_cost(
    usage: UsageStats,
    pair_count: int,
    price_in: float,
    price_out: float,
    target_pairs: int,
) -> Dict[str, Any]:
    avg_prompt = usage.avg_prompt()
    avg_completion = usage.avg_completion()
    avg_total = usage.avg_total()
    cost_per_call = (avg_prompt / 1_000_000) * price_in + (avg_completion / 1_000_000) * price_out
    calls_per_pair = (usage.calls / pair_count) if pair_count else 0.0
    cost_per_pair = cost_per_call * calls_per_pair
    scaled_cost = cost_per_pair * target_pairs

    return {
        "calls": usage.calls,
        "pairs": pair_count,
        "files_scanned": usage.files_scanned,
        "avg_prompt_tokens": avg_prompt,
        "avg_completion_tokens": avg_completion,
        "avg_total_tokens": avg_total,
        "cost_per_call": cost_per_call,
        "calls_per_pair": calls_per_pair,
        "cost_per_pair": cost_per_pair,
        "target_pairs": target_pairs,
        "scaled_cost": scaled_cost,
        "price_in_per_1m": price_in,
        "price_out_per_1m": price_out,
    }
