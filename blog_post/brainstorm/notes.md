# To Explain

## Potential narrative
What's the narrative for the blog post? I think I can simply walk readers through exactly how I built this, removing any details that don't add anything.

* context: prompted GPT with agent project. Can show users snippets of what the original code looked like
* Problem: dict soup + ad-hoc state made the agent hard to reason about. Couldn't understand code. Also, the code was failing in some places. Needed to understand code in order to know how to best fix.
* Observation: in agents, debugging is everything; you need an audit trail.
* Design move: make History the ledger; derive a compact RunSummary. There were many more design choices I made actually. Would need to distill and cut for reader.
* Result: main result - I could actually understand what was going on to write about it.simpler loop, less hidden mutation, clearer prompts, better testability.

*purpose for this narrative? What value does it add?* Really it's just to show hiring managers that I'm a great engineer, not just an ML guy. I can also talk about my first experience with the boundaries of LLMs as code generators and if they produce good/readable code (I don't think the first version of the program was good for readability at all. It also works better after my refactoring.)
*drawbacks* tmi for reader, doesn't add any value. Risks losing reader interest
Does this narrative add anything other than just explaining exactly what the finished code is?
*possible alternatives* Split into two articles. Drawbacks? Significantly more work
*alternateive* It might not be too big of a deal to split the post into two parts. Part I being the actual tool; straight to the point. Part II) briefly talk about refatcoring and LLMs as code generators and coding assistants.
*caveat* this is my first forray into using agents as code generators for a complex system, not just an ML model. To be fair ML models are very complex, but this was a slightly different task. I'm not sure how to explain what the difference was. But GPT 5.1 did a pretty good job at generating cohesive, clean ML model code. Not as great of a job generating this code

## In general
We give
- context 
  - history of actions and observations
  - summary of the current state
- list of deterministic actions
- LLM chooses action based on context
- Driver executes action and adds the observed result to the working memory
- Repeat

## Why do we let only the driver run test_cmd?
```python
if parsed_action.name in ("write_file", ) and test_cmd:
```
We keep test_cmd driver-only to bound the model’s power and cost:

Safety/control: tests can run arbitrary commands; letting the model call them directly is a bigger footgun. The driver decides when/how to invoke.
Cost/perf: tests can be slow/expensive; the driver can gate them (e.g., only after writes) instead of the model spamming runs.
Simplicity: the tool schema stays minimal (list/read/write/grep), and the driver applies a consistent rule (“after write_file, run tests if provided”).
So yes, it’s scope control: reduce risk, keep behavior predictable, and avoid the model hammering tests.

gpt-codex-prompt: When prompting gpt to construct the agent code, I did say "this is for a blog post", so I think it optimized for like a minimal working example + showable functionality so that I could write about it on my blog.

gpt-codex-response: Makes sense—that brief kept the scope to “showable, minimal, safe.” It chose the simplest control flow, a small tool surface, and driver-owned tests to avoid surprises. If you want to evolve it beyond the demo, the next levers are: preflight test check/early exit, diff-aware writes, and richer summary/history prompts.

Letting the LLM run tests directly widens the blast radius:

Safety/footguns: tests are arbitrary commands; the model could spam slow/expensive invocations or run surprising scripts.
Cost/perf: it might run tests too often (or the full suite) and burn time/money.
Control flow: you lose the driver’s simple rule “only after write_file”; now the model decides when/which tests, making behavior harder to predict.
Surface area: you’d need to expose a run_tests tool with args, permissions, and maybe timeouts—more schema, more failure modes.
Loop risk: the model could get stuck rerunning tests instead of progressing.
So scope increases in safety, cost, complexity, and predictability. If you ever add it, wrap it in tight constraints (specific command, rate limits, timeouts).

## Agentic programming framework
### ReACT
Thought: The agent uses its internal reasoning (an "inner monologue") to analyze the current situation, break down the task into smaller steps, and determine the next logical step.

*Example* When the LLM uses the history. Should probably explicitly prompt CoT

Action: Based on its thought process, the agent selects and executes a specific action, such as using a search tool, accessing a database, or making an API call.

*Example* the suggested ToolAction or FinalAction

Observation: The agent receives and processes the result or output from the action, which then informs its next "Thought". 

*Example* When we log the observation to the History (working memory)

## History (Working memory)
* Added to the prompt each time
* Append
    * ToolCallEvent: information on what tool was called
    * ObservationEvent: information on the result of a tool call
    * LLMActionEvent: information on what action the LLM decides to take
    * DriverNoteEvent: information on misc events that can happen during run like loop_event and json parsing issues

## The Driver (agent::RepoAgent::run)
This is the workflow orchestrator and is shown in the [workflow diagram](#fig-agent-workflow)
I won't add too much detail here, because I think it's easier to understand the flow visually via the diagram as opposed to me explaining it in words. But this is the flow logic of the agent. If the LLM is an unbounded, stochastic "brain", the driver is what constrains that brain with determinism so we can acheive outcomes in an expected way.

## The Action Controller (controller::ActionController)
A controller is a coordinator between intent and execution. It's the piece that takes a decision and routes it to the right handler. In our case, `ActionController` takes in a `ToolCallAction` and knows how to execute the corresponding bash command via the Python `os` module.


## Actions
When an LLM api request is made, we parse the response and return an Action. Actions come in two types:
1. ToolCallAction
2. FinalAction.

ToolCallAction is a class the represents a set of actions, *tools*, to be taken when the agent decides it needs more information before being able to acheive the *goal*. The *tools* are bash commands (called via Python `os` module).

## Tools
As stated above, *tools* are bash commands (called via Python `os` module). But in this section, we describe tools as they relate to the OpenAI API interface. In this case, the avaible tools need to be passed to the API, providing the LLM with context. Previously, all of this context had to be provided via text prompt. This is incredibly useful.


## Run summary
- A representation of the current state of the workflow
- Tracks notes, files touched, last test run
- State class represents the current state
- "In the current iteration, what has happened thus far" - is this true?


## Audit log (Trace)
While History is the model’s working memory; Trace is the audit log. Trace records every event to JSONL with run_id/meta so you can inspect/replay/debug runs (inspect_trace.py, Trace.get_run_history). It’s not fed back to the model—only History (or a window of it) plus RunSummary go into the prompt. Trace exists for observability, repro, and debugging, not for steering the loop.

<figure id="fig-agent-workflow">
  <img src="./diagrams/agent_workflow.png" alt="Diagram of the workflow executed by the driver."/>
  <figcaption>
    Fig 1: A diagram of the workflow executed by the driver agent::RepoAgent::run
  </figcaption>
</figure>


## Engineering choices

### LLM interface
llm:LLM is an interface so we can write adapters for other LLMs like LLaMA or Mixtral.

### OpenAIResponsesLLM Adapter
Adapts the OpenAI responses API into something that `RepoAgent::init(llm: LLM)` expects

## Prompt Compiler
