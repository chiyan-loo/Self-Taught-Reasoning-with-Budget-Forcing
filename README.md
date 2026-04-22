<div align="center">
  <h1>Enhancing Reasoning Traces for Fine-Tuning</h1>
  <p>Minimal recipe for enhancing reasoning traces for fine-tuning to achieve strong reasoning performance
</div>
<br>

## Structure

- `data/`: contains the dataset
- `train/`: training scripts
- `data/`: contains the dataset
- `train/`: training scripts
- `reasoning/`: synthetic reasoning generation and enhancement scripts

## Initial Reasoning Trace Generation

Generate initial reasoning traces for a dataset using a local vLLM instance. This script takes a dataset (like MATH-500) and produces step-by-step reasoning for each problem.

```bash
python reasoning/reasoning_traces.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_name_or_path "HuggingFaceH4/MATH-500" \
    --dataset_split "test" \
    --prompt_column "problem" \
    --output_file data/reasoning_traces.jsonl \
    --tensor_parallel_size 1
```

## Enhance Reasoning Traces

Enhance existing reasoning traces by having an LLM rewrite them to incorporate three key reasoning habits:

1.  **Elaborated Reasoning** — Comprehensive exploration of logical steps without premature conclusions.
2.  **Self-Verification** — Regular validation of intermediate results and logical consistency.
3.  **Exploratory Approach** — Consideration of multiple possibilities before reaching conclusions.

The script supports two inference backends:
- **api**: Any OpenAI-compatible API endpoint (OpenAI, Together, Groq, etc.)
- **vllm**: Local vLLM server or in-process generation.



## Installation

Set up your environment using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage



### 2. Fine-Tune with LoRA

Train a smaller student model on the reasoning traces using Supervised Fine-Tuning (SFT).

```bash
python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B" \
    --dataset_name_or_path "nlile/hendrycks-MATH-benchmark" \
    --dataset_split "train" \
    --response_column "solution" \
    --output_dir "./output/Qwen2.5-3B-MATH-2" \
    --max_train_samples 800 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --system_prompt "Reason step by step, and put your final answer within \\boxed{}." \
    --load_in_4bit
```

### 3. Merge LoRA Weights

Merge the LoRA adapter back into the base model for faster inference and evaluation.

```bash
python train/merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen2.5-3B" \
    --adapter_path "./output/Qwen2.5-3B-MATH-2" \
    --output_dir "./output/Qwen2.5-3B-MATH-2-merged"
```

### 4. Evaluate

Use the LM Evaluation Harness to evaluate the fine-tuned model on the MATH500 benchmark.

```bash
lm-eval run \
    --model vllm \
    --model_args pretrained=./output/Qwen2.5-3B-MATH-2-merged \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --limit 200 \
    --output_path ./results \
    --log_samples \
    --gen_kwargs max_gen_toks=2048 \
    --num_fewshot 0 \
    --system_instruction "Reason step by step, and put your final answer within \\boxed{}." \
    --gen_kwargs do_sample=True,temperature=0.8,top_p=0.95
```