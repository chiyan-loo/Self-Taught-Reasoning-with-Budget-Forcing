#!/bin/bash

# Example 1: Generate reasoning traces with budget forcing until correct
# Stops when correct (with 80% chance to continue at step 0) or after 5 steps
# over_gen_exponent 2.0 makes the probability drop quadratically
# Using temperature 0.8 for more diverse/random reasoning traces
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode budget \
    --max_steps 5 \
    --over_gen_prob 0.80 \
    --over_gen_exponent 3.0 \
    --temperature 0.8 \
    --top_p 0.95 \
    --num_samples 1000 \
    --output_file MATH_traces_bf_1000.jsonl

# Example 2: Generate reasoning traces with alternating "Wait" and "Alternatively"
# More aggressive over-generation (prob 0.9) but with high decay (exp 3.0)
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --dataset nlile/hendrycks-MATH-benchmark \
    --mode alternating \
    --max_steps 5 \
    --over_gen_prob 0.9 \
    --over_gen_exponent 3.0 \
    --temperature 0.8 \
    --top_p 0.9 \
    --num_samples 50 \
    --output_file traces_alternating.jsonl
