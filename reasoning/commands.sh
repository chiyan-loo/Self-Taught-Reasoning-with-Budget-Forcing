# Examples

# Generate reasoning traces with a 5 alternating
# Using temperature 0.7 for more diverse/random reasoning traces
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode alternating \
    --num_waits 5 \
    --temperature 0.7 \
    --top_p 0.95 \
    --num_samples 2000 \
    --output_file MATH_traces_Qwen2.5-3B-Instruct_alt_5alt_2000.jsonl \
    --max_tokens 4096 \
    --max_model_len 4096

# Generate reasoning traces without budget forcing
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode none \
    --temperature 0.7 \
    --top_p 0.95 \
    --num_samples 2000 \
    --output_file MATH_traces_Qwen2.5-3B-Instruct_no-bf_2000.jsonl \
    --max_tokens 4096 \
    --max_model_len 4096


# Filter repeating traces from generated file
python3 reasoning/filter_repeating_traces.py \
    --input reasoning/MATH_traces_Qwen2.5-3B-Instruct_alt_5alt_1000.jsonl \
    --output reasoning/MATH_traces_Qwen2.5-3B-Instruct_alt_5alt_1000_filtered_2.jsonl \
    --threshold 3 \
    --min_len 20 \
    # --max_samples 250

python3 reasoning/filter_repeating_traces.py \
    --input reasoning/bf/MATH_traces_Qwen2.5-3B-Instruct_bf_3w_2000_correct.jsonl \
    --output reasoning/MATH_traces_Qwen2.5-3B-Instruct_bf_3w_2000_filtered_2.jsonl \
    --threshold 3 \
    --min_len 21 \
    --max_samples 250



# Compute token statistics for a dataset
python3 reasoning/compute_sample_stats.py \
    reasoning/MATH_traces_Qwen2.5-3B-Instruct_alt_5alt_1000_correct_filtered.jsonl \
    --prompt_column problem \
    --response_column model_response

python3 reasoning/compute_sample_stats.py \
    results/bf3wait/Qwen__Qwen2.5-3B-Instruct/samples_minerva_math500_2026-04-29T22-06-51.848056.jsonl \
    --prompt_column problem \
    --response_column model_response
