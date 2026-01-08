"""Run N rollouts per task for preference data generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm_repo_agent.eval.runner import EvalConfig, EvalRunner, TaskResult
from llm_repo_agent.eval.tasks import TaskSpec, EvalSuite, load_suite
from llm_repo_agent.llm import LLM, LLMFactory, LLMConfig
from llm_repo_agent.prompts import system_prompt

from .schema import PreferencePair, PreferenceMeta, format_together_jsonl
from .score import score_rollout, RolloutScore
from .pairs import select_pair, PairResult


@dataclass
class RolloutResult:
    """Result from running a single rollout."""
    task_result: TaskResult
    score: RolloutScore
    final_content: str  # JSON-serialized final output
    seed: int


@dataclass
class PrefsConfig:
    """Configuration for preference data generation.

    Attributes:
        trace_dir: Directory to store trace files.
        out_path: Path for output JSONL file.
        meta_path: Path for metadata JSONL file.
        write_mode: Output write mode ("overwrite" or "append").
        rollouts: Number of rollouts per task.
        sandbox: Whether to run in sandbox mode.
        keep_sandbox: Whether to keep sandbox after run.
        test_policy: When to run tests.
        max_iters: Maximum agent iterations per task.
        model: Model identifier to use.
        llm_provider: Provider backend.
        together_api_key: Optional Together API key override.
        temperature: Sampling temperature.
        base_seed: Base seed for reproducibility.
        progress: Whether to show progress output.
    """
    trace_dir: Path = field(default_factory=lambda: Path("runs/prefs"))
    out_path: Path = field(default_factory=lambda: Path("runs/prefs/dpo_dataset.jsonl"))
    meta_path: Optional[Path] = None  # Defaults to out_path with _meta suffix
    write_mode: str = "overwrite"  # "overwrite" | "append"
    rollouts: int = 4
    sandbox: bool = True
    keep_sandbox: bool = False
    test_policy: str = "on_write"
    max_iters: int = 20
    model: Optional[str] = None
    llm_provider: str = "together"
    together_api_key: Optional[str] = None
    temperature: float = 0.7
    base_seed: int = 42
    progress: bool = True

    def __post_init__(self):
        if self.write_mode not in {"overwrite", "append"}:
            raise ValueError(f"Invalid write_mode: {self.write_mode!r}. Use 'overwrite' or 'append'.")
        if self.meta_path is None:
            stem = self.out_path.stem
            self.meta_path = self.out_path.parent / f"{stem}_meta.jsonl"


class PrefsRunner:
    """Runs rollouts and generates preference pairs."""

    def __init__(self, cfg: PrefsConfig):
        self.cfg = cfg
        self.pairs: List[PreferencePair] = []
        self.metas: List[PreferenceMeta] = []
        self.no_contrast_tasks: List[str] = []

    def _make_llm_factory(self, seed: int) -> Callable[[], LLM]:
        """Create an LLM factory with a specific seed."""
        def factory() -> LLM:
            return LLMFactory.build(LLMConfig(
                provider=self.cfg.llm_provider,
                model=self.cfg.model,
                together_api_key=self.cfg.together_api_key,
                temperature=self.cfg.temperature,
                seed=seed,
            ))
        return factory

    def run_task_rollouts(self, task: TaskSpec, suite_name: str) -> Optional[tuple[PreferencePair, PreferenceMeta]]:
        """Run N rollouts for a single task and return preference pair if contrast exists.

        Args:
            task: The task specification.
            suite_name: Name of the suite (for metadata).

        Returns:
            Tuple of (PreferencePair, PreferenceMeta) if contrast exists, None otherwise.
        """
        rollout_results: List[RolloutResult] = []

        for i in range(self.cfg.rollouts):
            seed = self.cfg.base_seed + i

            if self.cfg.progress:
                print(f"\n[prefs] Task {task.task_id}, rollout {i+1}/{self.cfg.rollouts} (seed={seed})")

            # Create eval config for this rollout
            eval_cfg = EvalConfig(
                trace_dir=self.cfg.trace_dir,
                sandbox=self.cfg.sandbox,
                keep_sandbox=self.cfg.keep_sandbox,
                test_policy=self.cfg.test_policy,
                max_iters=self.cfg.max_iters,
                model=self.cfg.model,
                llm_provider=self.cfg.llm_provider,
                together_api_key=self.cfg.together_api_key,
                progress=self.cfg.progress,
            )

            # Create runner with seed-specific LLM factory
            runner = EvalRunner(cfg=eval_cfg, llm_factory=self._make_llm_factory(seed))
            task_result = runner.run_task(task)

            # Score the rollout
            score = score_rollout(task_result)

            # Extract final content as JSON
            final_obj = {
                "type": "final",
                "summary": task_result.final_summary,
                "changes": [{"path": p, "description": "Edited file"} for p in task_result.files_touched],
            }
            if task_result.success is not None:
                final_obj["test_result"] = {
                    "ok": task_result.success,
                    "summary": task_result.test_output[:200] if task_result.test_output else "",
                }
            final_content = json.dumps(final_obj)

            rollout_results.append(RolloutResult(
                task_result=task_result,
                score=score,
                final_content=final_content,
                seed=seed,
            ))

            if self.cfg.progress:
                status = "PASS" if task_result.success else ("FAIL" if task_result.success is False else "N/A")
                print(f"[prefs] Rollout {i+1} result: {status}, steps={task_result.steps}, tool_calls={task_result.tool_calls}")

        # Select best/worst pair
        scores = [r.score for r in rollout_results]
        pair_result = select_pair(scores)

        if pair_result is None or not pair_result.has_contrast:
            if self.cfg.progress:
                print(f"[prefs] No contrast for task {task.task_id} - skipping")
            self.no_contrast_tasks.append(task.task_id)
            return None

        preferred = rollout_results[pair_result.preferred_idx]
        non_preferred = rollout_results[pair_result.non_preferred_idx]

        # Build preference pair
        sys_prompt = system_prompt()
        user_goal = f"GOAL:\n{task.goal}"

        pref_pair = format_together_jsonl(
            system_prompt=sys_prompt,
            user_goal=user_goal,
            preferred_content=preferred.final_content,
            non_preferred_content=non_preferred.final_content,
        )

        # Build metadata
        meta = PreferenceMeta(
            task_id=task.task_id,
            suite=suite_name,
            model=self.cfg.model or "unknown",
            temperature=self.cfg.temperature,
            seed=self.cfg.base_seed,
            scores={
                "preferred": pair_result.preferred_score.primary,
                "non_preferred": pair_result.non_preferred_score.primary,
            },
            tests_ok={
                "preferred": preferred.task_result.success or False,
                "non_preferred": non_preferred.task_result.success or False,
            },
            trace_ids={
                "preferred": preferred.task_result.run_id,
                "non_preferred": non_preferred.task_result.run_id,
            },
            rollout_counts={"total": self.cfg.rollouts},
        )

        if self.cfg.progress:
            print(f"[prefs] Selected pair: preferred=rollout[{pair_result.preferred_idx}] (score={pair_result.preferred_score.primary}), "
                  f"non_preferred=rollout[{pair_result.non_preferred_idx}] (score={pair_result.non_preferred_score.primary})")

        return pref_pair, meta

    def run_suite(self, suite: EvalSuite) -> None:
        """Run all tasks in a suite and generate preference pairs.

        Args:
            suite: The evaluation suite to process.
        """
        self.pairs = []
        self.metas = []
        self.no_contrast_tasks = []

        for task in suite.tasks:
            if self.cfg.progress:
                print(f"\n{'='*60}")
                print(f"[prefs] Processing task: {task.task_id}")
                print(f"[prefs] Goal: {task.goal}")
                print(f"{'='*60}")

            result = self.run_task_rollouts(task, suite.name)
            if result is not None:
                pref_pair, meta = result
                self.pairs.append(pref_pair)
                self.metas.append(meta)

        if self.cfg.progress:
            print(f"\n{'='*60}")
            print(f"[prefs] Summary:")
            print(f"[prefs] Total tasks: {len(suite.tasks)}")
            print(f"[prefs] Preference pairs generated: {len(self.pairs)}")
            print(f"[prefs] No-contrast tasks: {len(self.no_contrast_tasks)}")
            if self.no_contrast_tasks:
                print(f"[prefs] Skipped: {', '.join(self.no_contrast_tasks)}")
            print(f"{'='*60}")

    def write_output(self) -> None:
        """Write preference pairs and metadata to JSONL files."""
        # Ensure output directories exist
        self.cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self.cfg.write_mode == "append" else "w"

        def _ensure_newline(path: Path) -> None:
            if self.cfg.write_mode != "append":
                return
            if not path.exists():
                return
            if path.stat().st_size == 0:
                return
            with path.open("rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    with path.open("a", encoding="utf-8") as f_text:
                        f_text.write("\n")

        # Write preference pairs
        _ensure_newline(self.cfg.out_path)
        with self.cfg.out_path.open(mode, encoding="utf-8") as f:
            for pair in self.pairs:
                f.write(pair.to_jsonl() + "\n")

        if self.cfg.progress:
            verb = "Appended" if self.cfg.write_mode == "append" else "Wrote"
            print(f"[prefs] {verb} {len(self.pairs)} preference pairs to: {self.cfg.out_path}")

        # Write metadata
        if self.cfg.meta_path:
            _ensure_newline(self.cfg.meta_path)
            with self.cfg.meta_path.open(mode, encoding="utf-8") as f:
                for meta in self.metas:
                    f.write(meta.to_jsonl() + "\n")

            if self.cfg.progress:
                verb = "Appended" if self.cfg.write_mode == "append" else "Wrote"
                print(f"[prefs] {verb} metadata to: {self.cfg.meta_path}")


def run_rollouts(
    suite_path: Path,
    cfg: PrefsConfig,
) -> PrefsRunner:
    """Main entry point: load suite, run rollouts, write output.

    Args:
        suite_path: Path to the suite JSON file.
        cfg: Configuration for preference generation.

    Returns:
        PrefsRunner instance with results.
    """
    suite = load_suite(suite_path)

    if cfg.progress:
        print(f"[prefs] Loaded suite: {suite.name} ({len(suite.tasks)} tasks)")
        print(f"[prefs] Rollouts per task: {cfg.rollouts}")
        print(f"[prefs] Temperature: {cfg.temperature}")
        print(f"[prefs] Base seed: {cfg.base_seed}")

    runner = PrefsRunner(cfg)
    runner.run_suite(suite)
    runner.write_output()

    return runner
