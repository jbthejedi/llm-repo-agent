# DPO Finetuning General Plan

This document describes the planned changes for General DPO

## Goals

- Plan further than just the pilot


## Cost
How much will it cost to run a dpo finetune using together that actually moves that needle with Qwen?
Thoughts:
- Use Qwen-72B-instruct-turbo to generate dpo preference pairs and tune Qwen-7B-instruct turbo

<question>
Q: How many preference pairs per task do we need? ~1k-5k total pairs for a small LoRA improvement?
How many rollouts do you think we'll need for the preference data generation? ~4? If so, if we want 3k prefence pairs, that's 4*3k = 12K total rollouts. How many calls to Qwen are there per rollout? If `max_iters = 20` (default) and maybe it's ~10 iters per rollout and 2 ~1.3 Qwen calls per iteration -> 12k total rollouts * 10 iters * 1.3 = 156K calls to Qwen. What's the cost per Qwen 72B call? Lmk if my math is right so we can get an estimate. What other costs do I need to think of?
</question>
<answer>
A: There is no fixed “pairs per task.” Think in total pairs: ~1k-5k can move a small LoRA, 10k+ is more reliable. Rollouts = tasks * rollouts_per_task, and pairs = tasks * (pairs_per_task). Calls per rollout are roughly `avg_iters + avg_reflections` (each iteration is one chat call; reflections add extra calls), so your math should be framed as `rollouts * (avg_iters + avg_reflections)` rather than a constant 1.3 multiplier. Costs are token-based: total cost = (prompt_tokens + completion_tokens) summed across all calls, multiplied by Together’s per‑1M‑token rates. Other costs: DPO training tokens (training job uses both preferred+non-preferred outputs), dataset upload/storage, and optional endpoint hosting if you deploy a tuned model. Best way to estimate: run a small sample (e.g., 10 rollouts), measure average tokens per call from traces, then extrapolate.
</answer>
<question>
Q: How many preference pairs per task? If it's 1:1, then we need a lot of tasks to get 3k total pairs, right? QuixBugs doesn't have that many tasks. What do we do?
</question>
<answer>
A: It does not have to be 1 pair per task. You can create multiple pairs per task (best vs each non-best rollout), which scales pairs without adding new tasks. You can also increase rollouts per task for more pair diversity. If QuixBugs is still too small, mix in other code-repair tasks or add new tasks from similar benchmarks to increase diversity and avoid overfitting to a tiny task set.
</answer>
<question>
Q: How many tasks are there in QuixBugs? Let's decide on a size for our initial DPO preference pair dataset: how big should it be to move the QuixBugs needle? Actually, can we use QuixBugs to test a DPO finetuned Qwen7B-instruct-turbo (or maybe we just finetune 72B idk), or do we need a better repo to test dpo against? I'll feel like there's still room to improve Qwen's performance on QuixBugs. How many pairs per task do we need to reach our target preference training data set size? Actually, we can't use all QuixBugz tasks right? We need train tasks and test tasks to show that the model is generalizing across tasks.
</question>
<answer>
A: QuixBugs has on the order of ~40 Python tasks (exact count depends on the variant; you can script a count from `python_testcases`). For a first “needle‑moving” DPO run, target 1k–3k total pairs; that is often enough for a small LoRA improvement on a narrow domain. You can absolutely use QuixBugs as the eval set, but you should split tasks into train vs test (e.g., 70/30) to show generalization. If you need more pairs than the task count allows, generate multiple pairs per task (best vs each non‑best rollout). For model choice: fine‑tuning Qwen‑7B is cheaper and faster; use Qwen‑72B for data generation if you want higher‑quality rollouts. QuixBugs is a reasonable benchmark for demonstrating improvement, but a second held‑out suite (or a small extra benchmark) makes the story stronger.
</answer>
<question>
Q: what do you mean a second held-out suite, just a "test" suite of tasks from quixbugs? What would an example of a small extra benchmark be?
</question>
<answer>
A: A second held‑out suite means tasks the model never sees during training. That can be a QuixBugs test split, but it’s stronger if it’s a different dataset entirely. Examples: a small subset of Defects4J (Java), HumanEval‑Fix (Python), SWE‑bench Lite (repo‑level), or another small bug‑fix benchmark. Even 10–20 tasks from a different source makes the generalization claim more credible.
</answer>
<question>
Q: What else am I missing about how to plan this DPO finetune that wasn't covered in "dpo_finetune_pilot_plan.md"?
</question>
<answer>
A: Key planning items to add: (1) train/test split strategy and leakage checks, (2) data quality filters (dedupe identical rollouts, remove low‑signal pairs), (3) token budget and max context constraints (preferred + non‑preferred share context length), (4) hyperparameter plan (dpo_beta, epochs, batch size), (5) evaluation protocol (same test policy, fixed seeds, multiple runs), and (6) rollback criteria if performance regresses. Also decide whether to keep the base model fixed and only swap the tuned model during eval to isolate the training effect.
</answer>