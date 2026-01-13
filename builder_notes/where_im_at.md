
## What I've done. where I'm at
I think I'm gonna have to go back the drawing board on this project and re-evaluate my approach and what I want to do.
So far, I create the repo agent with eval harness that can fix all QuixBugs bugs just fine provided the LLM is powerful enough. I wanted to add DPO so I could showcase it on my resume. Ran into a ton of issues. This is not so easy—at least how we've been approaching/thinking about it.
Main constraint for me is pricing to create DPO dataset and host endpoints for larger modesl (H100s which is all that seems to be available on Together AI)

Here are the issues I ran into
- wanted to do dpo. thought I could use rollout summaries as a proxy signal to improve patches and tool call reasoning: Theoretically too weak
- Thought about using patch diffs.
  - with the quixbugs repo, the models that are too weak (8B, 14B, 30B) can't follow instructions well. The stronger models (70B+) follow instructions well, but also patch too well. I need there to be some delta—maybe 60% pass rate so I can see an improvement after DPO
  - I thought about switching to BugsInPy, I tried the Qwen2.5-72B (performs very well on QuixBugs)—follows instructions just fine, but not smart enough to pass one test it seems. Gpt-4.1-mini—does really well on QuixBugs—keeps getting stuck in loops on just one BugsInPy task
  - Also, another issue with generating DPO data is the cost-per-1M tokens on TogetherAI. I need to be conscious of this when generating preference pair data

  - What about doing SFT to improve a smaller models (7B) ability to follow instructions? I did start some work on this. The issue I ran into was QuixBugs repo doesn't have enough tasks to generate an sft dataset of enough samples 1K-3K.

Idk, at this point I'm getting frustrated. I need to take a step back because I'm starting to get a bit tangled
<answer>
You’re not crazy. What you’re trying to do is basically “make RLHF/DPO work for an agentic coding system” and that’s not straightforward, especially under tight budget constraints.

The hard parts you ran into are real:

DPO needs a strong, well-aligned preference signal. Rollout summaries are almost always too weak/noisy (models can write convincing summaries while doing the wrong thing).
Patch-diff DPO is tricky because “better patch” is hard to label without executing tests, and many diffs are incomparable (different files/strategies).
You need a base model in the “learnable regime”. If it can’t follow the tool protocol, it never reaches the patching stage → you can’t preference-train patch decisions. If it’s already too good, you won’t see measurable lift.
Domain shift is brutal (QuixBugs → BugsInPy): environment setup, larger codebases, flaky deps, and longer-horizon debugging create failure modes that look like “loops” rather than clean wrong answers.
Cost is dominated by rollouts, not the DPO trainer—getting enough high-quality preference pairs for an agent is expensive because each data point is multi-turn and tool-heavy.
If you want a path that’s resume-worthy but actually feasible:

Do SFT first on tool-protocol correctness + “when to stop reading and write a patch” (you can generate this cheaply from your own traces).
Keep DPO narrow: single-step preferences at a decision point (“write_file now” vs “keep searching”) using automatically-derived labels (e.g., does the next action lead to a passing test within N steps). That’s much cheaper than full-rollout preference labels.
Pick a task distribution where a 7–14B model gets like 20–60% pass rate with correct tool usage (so you can move it), and evaluate on a held-out subset.
If you want, tell me your target budget (roughly $) + which model sizes you can realistically fine-tune/host on Together, and I’ll propose a concrete “minimal viable SFT → minimal viable DPO” plan that fits it.
</answer>