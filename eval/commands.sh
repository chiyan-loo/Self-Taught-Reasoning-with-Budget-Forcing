# Examples

# Baseline
lm-eval run \
    --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0

# Budget forcing
lm-eval run \
    --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --output_path ./results/bf3wait \
    --log_samples \
    --gen_kwargs max_gen_toks=4096,max_tokens_thinking=3600,thinking_n_ignore=3,thinking_n_ignore_str=Wait \
    --num_fewshot 0


# Fine-tuned
lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-Instruct-Reasoning-no-bf-10epoch-250samples-2 \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --output_path ./results/fine_tuned \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0



lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-Instruct-Reasoning-alt-5alt-10epoch-250samples-merged \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --output_path ./results/fine_tuned \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0


lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-Instruct-Reasoning-alt-3alt-10epoch-0 \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --output_path ./results/fine_tuned \
    --log_samples \
    --gen_kwargs max_gen_toks=4096 \
    --num_fewshot 0




