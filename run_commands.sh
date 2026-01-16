#########################
### RUN EVAL SUITE SFT ##
#########################
# Run eval suite using Together Base Qwen model
poetry run repo-agent eval \
    --suite eval/suites/bugsinpy_pysnooper_suite.json \
    --trace-dir runs/pysnooper_eval_gpt \
    --report runs/pysnooper_eval_gpt/report.json \
    --llm-provider openai \
    --model gpt-4.1-mini \
    --tool-protocol json \
    --max-iters 100 \
    --no-sandbox

# Run eval suite using DPO finetuned model
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/qwen25_7b_it \
    --report runs/qwen25_7b_it/report.json \
    --llm-provider together \
    --model Qwen/Qwen2.5-7B-Instruct-Turbo \
    --tool-protocol json \
    --rollouts 5 \
    --num-workers 5 \
    --print-mode standard

# justinbarrye_c241/Qwen2.5-7B-Instruct-qwen25-7b-instruct-sft-pilot-0078c2e9-7ed87e84
poetry run repo-agent eval \
    --suite eval/suites/my_suite.json \
    --trace-dir runs/qwen25_7b_it_sft \
    --report runs/qwen25_7b_it_sft/report.json \
    --llm-provider together \
    --model justinbarrye_c241/Qwen2.5-7B-Instruct-qwen25-7b-instruct-sft-pilot-0078c2e9-7ed87e84 \
    --tool-protocol json \
    --rollouts 5

#########################
###### GEN DPO DATA ######
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
  # --suite eval/suites/sft_finetune_task_suite.json \
 poetry run repo-agent prefs \
  --suite eval/suites/sft_finetune_task_suite.json \
  --rollouts 10 \
  --out runs/test_sanity/instruction_tuning.jsonl \
  --trace-dir runs/test_sanity \
  --llm-provider together \
  --model Qwen/Qwen2.5-72B-Instruct-Turbo \
  --temperature 0.2 \
  --seed 42 \
  --max-workers 5 \
  --tool-protocol json \
  --test-policy on_write

 poetry run repo-agent prefs \
  --suite eval/suites/my_suite.json \
  --rollouts 5 \
  --out runs/test_qwen25_7b_it/instruction_tuning.jsonl \
  --trace-dir runs/test_qwen25_7b_it \
  --llm-provider together \
  --model Qwen/Qwen2.5-7B-Instruct-Turbo \
  --temperature 0.0 \
  --seed 42 \
  --max-workers 5 \
  --tool-protocol json \
  --test-policy on_write

# justinbarrye_c241/Qwen2.5-7B-Instruct-qwen25-7b-instruct-sft-pilot-0078c2e9-7ed87e84
 poetry run repo-agent prefs \
  --suite eval/suites/sft_eval_suite.json \
  --rollouts 5 \
  --out runs/test_gcd/instruction_tuning.jsonl \
  --trace-dir runs/test_gcd \
  --llm-provider together \
  --model justinbarrye_c241/Qwen2.5-7B-Instruct-qwen25-7b-instruct-sft-pilot-0078c2e9-7ed87e84 \
  --temperature 0.0 \
  --seed 42 \
  --max-workers 3 \
  --tool-protocol json \
  --test-policy on_write

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
  --trace-dir runs/quixbugs_traces_teacher_qwen25_72b \
  --output runs/quixbugs_traces_teacher_qwen25_72b/quixbugs_tool_sft_train.jsonl \
  --format json \
  --require-success \
  --drop-postfix-on-loop \
  --require-valid-tool-ok \
  --max-context-chars 8000 \
  --filter-write-file-targets \
  --require-root-list-files-first

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
  --dataset runs/quixbugs_traces_teacher_qwen25_72b/quixbugs_tool_sft_train.jsonl \
  --model Qwen/Qwen2.5-7B \
  --suffix qwen25-7b-sft-pilot-2 \
  --epochs 3 \
  --batch-size 8 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --train-on-inputs auto \
  --learning-rate 5e-6 \
  --lora \
  --watch \
  --wandb-project-name repo-agent-finetunes \
  --wandb-name qwen25-7b-sft-json-tools-2

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