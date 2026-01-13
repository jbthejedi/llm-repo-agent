You’re not tangled because you’re sloppy—you’re tangled because the problem you picked (DPO for patching) has a nasty property: **the reward signal you actually care about is sparse, expensive, and dataset-dependent**. If you don’t design around that, you end up paying for tokens to discover you can’t get a clean delta.

Here’s the clean reframing:

### The real bottleneck isn’t DPO

It’s **finding a regime where you can reliably produce (good, bad) pairs from the same policy**—cheaply—and where “good” correlates with generalization, not hacks.

Right now you’ve got two failure modes:

1. **Weak models** can’t obey the agent protocol → garbage rollouts → your pairs are mostly “instruction-following failure,” not patch quality.
2. **Strong models** patch too well on QuixBugs → you can’t show an improvement curve.

So the move is: **separate “agent behavior correctness” from “patch correctness,” then decide what you actually want to showcase.**

---

## Option A: Make DPO about *agent behavior* (cheap, controllable, resume-clean)

If your goal is “I can do DPO in an agentic setting,” you don’t need to bet everything on functional correctness of patches.

**Target:** preference-tune *tool-call discipline + reasoning format + loop avoidance + evidence-based edits*.

**Pairs:** cheap to generate because you can create them without a super-strong model:

* “Good” = follows protocol, uses tools in the right order, stops when stuck, produces minimal diff, cites failing test output, doesn’t thrash.
* “Bad” = violates invariants (edits unrelated files, ignores test failures, repeats loops, changes too much, doesn’t run tests, hallucinated file paths).

You can score this with:

* deterministic checks (did it run tests? did it touch only allowed files? diff size? repeated actions?)
* light heuristics (did it quote failing assertion lines? did it explain why change fixes that line?)

**Result:** you can DPO a 7B/14B to become a *disciplined repo agent*, even if it’s not magically smarter. That is still an impressive, honest story: “Preference optimization improves agent reliability and reduces failure modes.”

This also dodges the QuixBugs “too few tasks” issue because you’re not looking for 1K unique bugs—you’re looking for **many rollouts per task** where behavior varies.
