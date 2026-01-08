## What happens today (current behavior)

Right now your driver does this:

1. Model chooses an action
2. If the action is `write_file`, the driver **immediately runs tests**
3. The test output is:

   * logged to `Trace` (`trace.log("tests", ...)`)
   * appended to `History` as an observation (`history.append_observation("driver.run_tests", ...)`)
4. Next turn, the model effectively “sees” the fact that tests passed/failed via your derived summary/history.

This block is the culprit:

```py
test_res = None
if parsed_action.name in ("write_file", ) and test_cmd:
  test_res = self.tools.run_tests(test_cmd)
  ...
  history.append_observation("driver.run_tests", {...})
```

### Why that becomes a problem for eval + preference rollouts

If the agent writes a file 5 times in one run, you run tests 5 times.

Now imagine preference data generation where you do N rollouts per task:

* tasks = 30
* rollouts per task = 8
* avg writes per rollout = 4
* tests per rollout = 4 (today)

Total test executions = 30 * 8 * 4 = **960 test runs**

That’s the “extra test runs” explosion. It kills your 2-week timeline and makes eval slow/noisy.

Also: it changes the “agent capability” you’re measuring, because the model is getting iterative feedback every write, which may or may not be what you want for a given benchmark.

---

## What “test_policy” means

“test_policy” is simply: **when does the driver run tests?**

You’re not changing *what tests are*, just *when they run*.

Think of it as a trigger switch for `self.tools.run_tests(test_cmd)`.

---

## Policy: `on_write`

### Trigger condition

Run tests whenever the agent performs a `write_file`.

### Behavior

* Tests can run many times per run.
* The model gets rapid feedback and can iterate: write → test → fix → test → fix.

### What gets logged

Same as today:

* `Trace` has many `"tests"` events.
* `History` has many `"driver.run_tests"` observations.
* `final_obj["test_result"]` is easy because `summary.last_test` exists (your code already relies on that).

### Best for

Interactive use / “fix my repo” mode, where you want fast feedback loops.

---

## Policy: `on_final`

### Trigger condition

Run tests exactly once **at the end of the run**.

There are two ways to define “end”:

* When the model returns `FinalAction`
* Or when you stop due to max iters / abort

### Behavior

* During the run: **no auto tests** after `write_file`.
* At the end: run tests once and attach the result.

### What gets logged

* Exactly one `"tests"` event in the trace
* Exactly one `"driver.run_tests"` observation in history
* `final_obj["test_result"]` still works because you append the final test result to history before producing final output.

### Why this is valuable for eval + DPO

* Bounded compute: 1 test run per rollout instead of “per write”.
* Cleaner preference signal: your “chosen vs rejected” can just be based on that final outcome.
* Makes rollouts comparable (each rollout gets one final scoring pass).

### Best for

Eval harness runs and preference rollouts where you care about throughput and consistent scoring.

---

## Policy: `never`

### Trigger condition

The driver never runs tests.

### Behavior

* The agent never sees test feedback.
* You must score outside the agent (in the eval harness / preference scorer).

### What gets logged

* No `"tests"` events in trace
* No `"driver.run_tests"` in history
* `final_obj["test_result"]` would be absent unless you special-case it.

### Best for

Benchmarks where you explicitly want “no execution feedback,” or when the evaluator is the only place allowed to run commands.

(For your project, you *might* use this later, but you don’t need it to ship the two-week MVP.)

---

## The minimal code change (exactly what you do)

### 1) Add the flag to `AgentConfig`

In `agent.py`:

```py
@dataclass
class AgentConfig:
  ...
  test_policy: str = "on_write"  # "on_write" | "on_final" | "never"
```

### 2) Gate the existing “run tests after write_file” block

Replace:

```py
if parsed_action.name in ("write_file", ) and test_cmd:
```

with:

```py
if self.cfg.test_policy == "on_write" and parsed_action.name == "write_file" and test_cmd:
```

### 3) Add “run once at the end” logic for `on_final`

Right before you log the final action (or before `run_end`), add:

```py
if self.cfg.test_policy == "on_final" and test_cmd:
    test_res = self.tools.run_tests(test_cmd)
    output_snippet = test_res.output[:12000] if test_res.output else ""
    self.trace.log("tests", {"ok": test_res.ok, "output": output_snippet})
    history.append_observation("driver.run_tests", {"ok": test_res.ok, "output": output_snippet})
```

This ensures `summary.last_test` exists so your existing code that injects `test_result` into the final object continues to work (your tests currently assert that).

### 4) For `never`

You can initially implement it as simply “skip tests entirely.” That means final output won’t have `test_result`. If you want to preserve your current invariant (“final includes test_result whenever test_cmd exists”), then `never` shouldn’t exist yet—or it should be renamed to something like `external` and handled in the eval harness instead.

---

## Practical recommendation for your RLHF/DPO build

* **Interactive runs:** `on_write`
* **Eval runs / preference rollouts:** `on_final`

That gives you the compute control you need without changing your agent’s conceptual design.

If you later want the model to *choose* when to run tests, you’d add a `run_tests` tool callable by the model—but that’s a separate step, and not required to ship the MVP.
