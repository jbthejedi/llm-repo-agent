# Run eval suite using Together Base Qwen model
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/my_eval_6 \
    --report runs/my_eval_6/report.json \
    --llm-provider together \
    --model Qwen/Qwen2.5-72B-Instruct-Turbo 

# Use Qwen2.5 7B
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/test_qwen_7b \
    --report runs/test_qwen_7b/report.json \
    --llm-provider together \
    --model Qwen/Qwen2.5-7B-Instruct-Turbo 

# meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/test_llama_31_8b_it \
    --report runs/test_llama_31_8b_it/report.json \
    --llm-provider together \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

# LLama 4 scout
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/llama_4_scout \
    --report runs/llama_4_scout/report.json \
    --llm-provider together \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct

# Deepseek distilled
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/deepseek_r1 \
    --report runs/deepseek_r1/report.json \
    --llm-provider together \
    --model deepseek-ai/DeepSeek-R1-0528-tput

# Run eval suite using DPO finetuned model
# endpoint_id: justinbarrye_c241/Qwen2.5-7B-Instruct-dpo-lora-4bf2fea2-4d8be882
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/my_eval_6 \
    --report runs/my_eval_6/report.json \
    --llm-provider together \
    --model justinbarrye_c241/Qwen2.5-7B-Instruct-dpo-lora-4bf2fea2-4d8be882

# DPO finetune
 poetry run repo-agent prefs \
  --suite eval/suites/pref_data_gen_pilot_1.json \
  --rollouts 4 \
  --out runs/test_multithreading/dpo_dataset_cost_est.jsonl \
  --trace-dir runs/test_multithreading \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.7 \
  --seed 42 \
  --max-workers 4

# Estimate costs
# Qwen-72
poetry run repo-agent estimate-cost \
  --trace-dir runs/prefs_cost_estimate_pilot \
  --dataset runs/prefs_cost_estimate_pilot/dpo_dataset_cost_est.jsonl \
  --price-in 1.20 \
  --price-out 1.20 \
  --target-pairs 3000

# Run next after verifiying estimate costs works
 poetry run repo-agent prefs \
  --suite eval/suites/pref_cost_estimate_suite.json \
  --rollouts 7 \
  --out runs/prefs_cost_estimate_run_1/dpo_dataset_cost_est.jsonl \
  --trace-dir runs/prefs_cost_estimate_run_1\
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.7 \
  --seed 42