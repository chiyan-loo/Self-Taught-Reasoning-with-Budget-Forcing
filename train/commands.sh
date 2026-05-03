# Examples


python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name_or_path "reasoning/MATH_traces_Qwen2.5-3B-Instruct_no-bf_2000_correct.jsonl" \
    --dataset_split "train" \
    --prompt_column "problem" \
    --response_column "model_response" \
    --max_train_samples 250 \
    --output_dir "./output/Qwen2.5-3B-Instruct-Reasoning-no-bf-10epoch-250samples" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --load_in_4bit \
    --max_seq_length 3500 \
    --attn_implementation "sdpa" \
    --learning_rate 5e-5 \
    --epochs 10


# on generated dataset with correct reasoning traces
python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name_or_path "reasoning/alt/MATH_traces_Qwen2.5-3B-Instruct_alt_3alt_2000_correct_filtered.jsonl" \
    --dataset_split "train" \
    --prompt_column "problem" \
    --response_column "model_response" \
    --max_train_samples 250 \
    --output_dir "./output/Qwen2.5-3B-Instruct-Reasoning-alt-3alt-10epoch-0" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 4096 \
    --attn_implementation "sdpa" \
    --learning_rate 5e-5 \
    --epochs 10


python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name_or_path "reasoning/MATH_traces_Qwen2.5-3B-Instruct_bf_3w_2000_filtered_2.jsonl" \
    --dataset_split "train" \
    --prompt_column "problem" \
    --response_column "model_response" \
    --max_train_samples 250 \
    --output_dir "./output/Qwen2.5-3B-Instruct-Reasoning-bf-3w-10epoch-250samples-2" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --load_in_4bit \
    --max_seq_length 4096 \
    --attn_implementation "sdpa" \
    --learning_rate 5e-5 \
    --epochs 10


python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name_or_path "reasoning/MATH_traces_Qwen2.5-3B-Instruct_no-bf_2000_correct.jsonl" \
    --dataset_split "train" \
    --prompt_column "problem" \
    --response_column "model_response" \
    --max_train_samples 250 \
    --output_dir "./output/Qwen2.5-3B-Instruct-Reasoning-no-bf-10epoch-250samples-2" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --load_in_4bit \
    --max_seq_length 4096 \
    --attn_implementation "sdpa" \
    --learning_rate 5e-5 \
    --epochs 10




# merge
python train/merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --adapter_path "./output/Qwen2.5-3B-Instruct-Reasoning-bf-3w-10epoch-LoRA" \
    --output_dir "./output/Qwen2.5-3B-Instruct-Reasoning-bf-3w-10epoch-merged"