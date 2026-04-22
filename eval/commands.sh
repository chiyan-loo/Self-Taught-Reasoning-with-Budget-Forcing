
# for eval run
lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-Reasoning-Traces-merged \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --limit 200 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=8192 \
    --num_fewshot 0 \
    --system_instruction "Reason step by step, and put your final answer within \\boxed{}." 