**Together charges by token count** (priced “per 1M tokens”), and for most models the **input tokens and output tokens can have different rates**. You estimate preference-data cost by measuring average `prompt_tokens` + `completion_tokens` for the exact calls your generator/judge script makes, then scaling up. ([Together AI][1])

### What numbers you actually need

For each API call type you use (generator, judge, critique, etc.), you want:

* **avg input tokens** (`prompt_tokens`)
* **avg output tokens** (`completion_tokens`)
* **calls per datapoint** (e.g., 2 generations + 1 judge = 3 calls, unless you use `n=2` in one call)

Together’s Chat Completions response includes a `usage` object with `prompt_tokens`, `completion_tokens`, `total_tokens`, so you can log this directly. ([Together.ai Docs][2])

### Cost formula

For one call:

[
\text{cost} = \frac{\text{prompt_tokens}}{10^6}\cdot P_{in} ;+; \frac{\text{completion_tokens}}{10^6}\cdot P_{out}
]

Where (P_{in}, P_{out}) come from the pricing page for the model you’re using. ([Together AI][1])

Example: **Qwen2.5 7B Instruct Turbo** is listed at **$0.30 / $0.30 per 1M input/output tokens**. ([Together AI][1])

### Practical way to estimate (run 100–500 examples, then scale)

Here’s a tiny snippet you can drop into your data-gen script:

```python
from together import Together
import os

client = Together(api_key=os.environ["TOGETHER_API_KEY"])

PRICE_IN_PER_1M = 0.30   # look up for your model
PRICE_OUT_PER_1M = 0.30  # look up for your model

def call_cost(usage):
    return (usage.prompt_tokens / 1_000_000) * PRICE_IN_PER_1M + \
           (usage.completion_tokens / 1_000_000) * PRICE_OUT_PER_1M

total_cost = 0.0
total_prompt = 0
total_completion = 0

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Generate two candidate fixes for this bug..."}],
    max_tokens=400,
)

usage = resp.usage  # has prompt_tokens, completion_tokens, total_tokens
total_cost += call_cost(usage)
total_prompt += usage.prompt_tokens
total_completion += usage.completion_tokens

print({
    "prompt_tokens": usage.prompt_tokens,
    "completion_tokens": usage.completion_tokens,
    "estimated_cost_usd": total_cost,
})
```

### The part people miss with preference data

A “datapoint” usually costs more than one call:

* **Generator:** 2 candidates (either `n=2` in one request, or two separate requests)
* **Judge:** 1 request that reads prompt + both candidates (big input, small output)

So your cost per datapoint is roughly:

* If you do `n=2` in **one** generator call: **1× input + 2× output + judge call**
* If you do **two separate** generator calls: **2× input + 2× output + judge call**

Using token logs will tell you which you’re actually paying.

If you tell me which model you’re using for (1) generation and (2) judging, and roughly how long your “repo-fixing agent” prompt is, I can give you a tight back-of-the-envelope $ estimate for 10k / 50k / 100k preference pairs.

[1]: https://www.together.ai/pricing "Together AI -  Pricing"
[2]: https://docs.together.ai/reference/chat-completions-1?utm_source=chatgpt.com "Create Chat Completion - Together.ai Docs"

## Cost Estimate for Qwen-72B-Instruct-Turbo

I ran

```bash
# DPO finetune
 poetry run repo-agent prefs \
  --suite eval/suites/pref_data_gen_pilot_1.json \
  --rollouts 4 \
  --out runs/prefs_cost_estimate_pilot/dpo_dataset_cost_est.jsonl \
  --trace-dir runs/prefs_cost_estimate_pilot \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.7 \
  --seed 42
```

```bash
# Estimate costs
poetry run repo-agent estimate-cost \
  --trace-dir runs/prefs_cost_estimate_pilot \
  --dataset runs/prefs_cost_estimate_pilot/dpo_dataset_cost_est.jsonl \
  --price-in 1.20 \
  --price-out 1.20 \
  --target-pairs 3000
```

## TODO tomorrow (01/08)
- Generate more data for the preference data used in the cost estimate
- Try llama 4 scout