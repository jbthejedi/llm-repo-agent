#########################
### RUN EVAL SUITE SFT ##
#########################
# Run eval suite using Together Base Qwen model
poetry run repo-agent eval \
    --suite eval/suites/bugsinpy_pysnooper_suite.json \
    --trace-dir runs/pysnooper_eval_qwen72b \
    --report runs/pysnooper_eval_qwen72b/report.json \
    --llm-provider together \
    --model Qwen/Qwen2.5-72B-Instruct-Turbo \
    --tool-protocol json \
    --max-iters 100 \
    --no-sandbox
poetry run repo-agent eval \
    --suite eval/suites/bugsinpy_pysnooper_suite.json \
    --trace-dir runs/pysnooper_eval_gpt \
    --report runs/pysnooper_eval_gpt/report.json \
    --llm-provider openai \
    --model gpt-4.1-mini \
    --tool-protocol json \
    --max-iters 100 \
    --no-sandbox

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

#########################
######GEN DPO DATA ######
#########################

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

########################################
########### GENERATE SFT DATA #########
########################################

# Run DPO finetune command to create SFT dataset using qwen-72B
# test
 poetry run repo-agent prefs \
  --suite eval/suites/gcd.json \
  --rollouts 10 \
  --out runs/instruction_tuning_test_2/instruction_tuning_test.jsonl \
  --trace-dir runs/instruction_tuning_test_2 \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8

# another test
 poetry run repo-agent prefs \
  --suite eval/suites/gcd.json \
  --rollouts 10 \
  --out runs/instruction_tuning_test_3/instruction_tuning_test.jsonl \
  --trace-dir runs/instruction_tuning_test_3 \
  --llm-provider openai \
  --model gpt-4.1-mini \
  --temperature 0.0 \
  --seed 42 \
  --max-workers 8

# Run DPO finetune command to create SFT dataset using gpt41mini
 poetry run repo-agent prefs \
  --suite eval/suites/sft_finetune_task_suite.json \
  --rollouts 3 \
  --out runs/instruction_tuning/dpo_dataset_cost_est.jsonl \
  --trace-dir runs/instruction_tuning \
  --llm-provider openai \
  --model "gpt-4.1-mini" \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8

# DeepSeek-R1-Distill-Qwen-14B
 poetry run repo-agent prefs \
  --suite eval/suites/sft_finetune_task_suite.json \
  --rollouts 1 \
  --out runs/test_deepseek_r1_qwen/dpo_dataset_cost_est.jsonl \
  --trace-dir runs/test_deepseek_r1_qwen \
  --llm-provider together \
  --model deepseek-ai/deepseek-r1-distill-qwen-14b \
  --temperature 0.0 \
  --seed 42 \
  --max-workers 4

#########################
## ESTIMATE COSTS #######
#########################

# Estimate costs
# Qwen-72
poetry run repo-agent estimate-cost \
  --trace-dir runs/prefs_cost_estimate_pilot \
  --dataset runs/prefs_cost_estimate_pilot/dpo_dataset_cost_est.jsonl \
  --price-in 1.20 \
  --price-out 1.20 \
  --target-pairs 3000


#############################
#### SFT EXTRACT DATA #######
#############################
poetry run repo-agent sft-extract \
  --trace-dir runs/instruction_tuning_test_3 \
  --output runs/instruction_tuning_test_3/sft_dataset.jsonl

##########################################
#### TEST JSON TOOL CALLING (TEXT) #######
##########################################

# Llama 4 Maverick
 poetry run repo-agent prefs \
  --suite eval/suites/pref_cost_estimate_suite.json \
  --rollouts 1 \
  --out runs/test_llama_3it_maverick/json_tool_calling.jsonl \
  --trace-dir runs/test_llama_3it_maverick \
  --llm-provider together \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --temperature 0.0 \
  --seed 42 \
  --max-workers 4 \
  --tool-protocol json

 poetry run repo-agent prefs \
  --suite eval/suites/gcd.json \
  --rollouts 10 \
  --out runs/json_tool_calling_test/json_tool_calling.jsonl \
  --trace-dir runs/json_tool_calling_test \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8 \
  --tool-protocol json

  poetry run repo-agent prefs \
  --suite eval/suites/gcd.json \
  --rollouts 4 \
  --out runs/json_tool_calling_test/json_tool_calling_test.jsonl \
  --trace-dir runs/json_tool_calling_test \
  --llm-provider openai \
  --model "gpt-4.1-mini" \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8

 poetry run repo-agent prefs \
  --suite eval/suites/pref_cost_estimate_suite.json \
  --rollouts 2 \
  --out runs/test_qwen25_72B_it_json/json_tool_calling.jsonl \
  --trace-dir runs/test_qwen25_72B_it_json \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8 \
  --tool-protocol json

  poetry run repo-agent prefs \
  --suite eval/suites/pref_cost_estimate_suite.json \
  --rollouts 2 \
  --out runs/test_qwen25_7_it_json/json_tool_calling.jsonl \
  --trace-dir runs/test_qwen25_7_it_json \
  --llm-provider together \
  --model Qwen/Qwen2.5-7B-Instruct-Turbo \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8 \
  --tool-protocol json


poetry run repo-agent prefs \
  --suite eval/suites/pref_cost_estimate_suite.json \
  --rollouts 2 \
  --out runs/test_qwen3_8b_it_json/json_tool_calling.jsonl \
  --trace-dir runs/test_qwen3_8b_it_json \
  --llm-provider together \
  --model Qwen/Qwen3-8B \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8 \
  --tool-protocol json

poetry run repo-agent prefs \
  --suite eval/suites/pref_cost_estimate_suite.json \
  --rollouts 2 \
  --out runs/test_deepseek_r1_qwen/json_tool_calling.jsonl \
  --trace-dir runs/test_deepseek_r1_qwen \
  --llm-provider together \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --temperature 0.1 \
  --seed 42 \
  --max-workers 8 \
  --tool-protocol json


##############################
#### CALL SFT TOGETHER #######
##############################
poetry run python sft_finetune_quick.py \
  --dataset runs/instruction_tuning/sft_dataset.jsonl \
  --model Qwen/Qwen3-8B \
  --suffix qwen3-8b-sft-pilot \
  --epochs 3 \
  --batch-size max \
  --learning-rate 1e-5 \
  --lora \
  --watch \
  --wandb-project-name repo-agent-finetunes \
  --wandb-name qwen3-8b-sft-json-tools \
  --wandb-api-key <key_here>

poetry run python sft_finetune_quick.py \
  --dataset runs/instruction_tuning/sft_dataset.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --suffix ds-r1-distilled-qwen-14b-sft-pilot \
  --epochs 3 \
  --batch-size max \
  --learning-rate 1e-5 \
  --lora \
  --watch \
  --wandb-project-name repo-agent-finetunes \
  --wandb-name qwen3-8b-sft-json-tools \
  --wandb-api-key 


##############################
#### TOGETHER DEPLOY ENDPOINT
##############################
poetry run together endpoints create \
  --model "justinbarrye_c24l/Qwen3-8B-qwen25-7b-it-json-tools-f30fe75b \
  --gpu h100 \
  --gpu-count 1 \
  --min-replicas 1 \
  --max-replicas 1 \
  --display-name "qwen3-8b-sft" \
  --no-prompt-cache \
  --no-speculative-decoding