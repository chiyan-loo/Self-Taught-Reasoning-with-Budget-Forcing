#!/bin/bash

# Example 3: Run BOTH Normal and Budget Forcing in one go
# Creates: comparison_normal_samples.jsonl, comparison_normal_stats.json,
#          comparison_budget_samples.jsonl, comparison_budget_stats.json
python3 eval/inference.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --dataset HuggingFaceH4/MATH-500 \
    --modes normal budget alternating \
    --num_budget_steps 2 \
    --num_samples 250 \
    --output_name MATH-500_250samples_2steps

# Example 4: Evaluation on GSM8K (custom keys)
python3 eval/inference.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --dataset gsm8k \
    --split test \
    --question_key question \
    --answer_key answer \
    --num_samples 20 \
    --output_name gsm8k_eval

# Example 5: Evaluation with Alternating mode
# Creates: math_alternating_alternating_samples.jsonl, math_alternating_alternating_stats.json
python3 eval/inference.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --dataset HuggingFaceH4/MATH-500 \
    --modes alternating \
    --num_budget_steps 4 \
    --num_samples 100 \
    --output_name MATH-500_100samples_4steps
