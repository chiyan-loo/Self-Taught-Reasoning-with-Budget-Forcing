<div align="center">
  <h1>Self-Taught Reasoning with Simple Budget Forcing (In Progress)</h1>
  </p>
</div>

## Overview

**Question**: Training strong reasoning models typically requires either expensive human annotation or distillation from larger models. This raises the question: can a model learn complex reasoning without human labels or a stronger teacher?

**Method**: To self-generate high-quality reasoning traces, we adopt a simple budget forcing method where we append “wait” and “alternatively” to the response to encourage longer reasoning and instill self-verification during synthetic data generation. We then take the correctly generated responses, filter out repetitive responses, and fine tune on the filtered dataset. We use 0.7 temperature for diversity during reasoning trace generation and greedy decoding during evaluation.

**Results**: Preliminary results show that fine tuning Qwen 2.5 3B Instruct with QLoRA on less than 250 self-generated MATH reasoning traces (3 budget forcing steps) yields a 3.8% accuracy gain on MATH-500 while typical self-taught reasoning only increases accuracy by 0.8% with the same amount of samples. (both evaluated with 1 training iteration)

| Method | MATH-500 Accuracy | Δ vs Baseline |
|---|---|---|
| Qwen 2.5 3B Instruct (baseline) | 64.4% | — |
| + Self-generated reasoning traces | 65.2% | +0.8% |
| + Budget forcing (3 waits) | 65.8% | +1.4% |
| + Self-generated reasoning traces with Budget forcing (3 "wait") | **65.2%** | **+0.8%** |
| + Self-generated reasoning traces with Budget forcing (3 alternating "wait", "alternatively") | **68.2%** | **+3.8%** |

| Method | MATH-500 Accuracy | Δ vs Baseline |
|---|---|---|
| Qwen 2.5 3B Instruct (baseline) | 64.4% | — |
| + Self-generated reasoning traces | 65.2% | +0.8% |
| + Budget forcing (3 waits) | 65.8% | +1.4% |
| + Self-generated reasoning traces with Budget forcing (3 "wait") | 65.2% | +0.8% |
| + Self-generated reasoning traces with Budget forcing (3 alternating "wait", "alternatively") | **65.6%** | **+1.2%** |
| + Self-generated reasoning traces with Budget forcing (5 alternating "wait", "alternatively") | **67.0%** | **+2.6%** |


## Current Issues
- Models tend to generate repetitive responses during reasoning trace generation and evaluation.
- I would like to scale to larger models, longer reasoning traces, and full fine-tuning, but I don't have access to the necessary hardware. 😭

## Structure

- `reasoning/`: Synthetic reasoning generation scripts
- `train/`: Training scripts
- `eval/`: Evaluation scripts (lm-evaluation-harness)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the evaluation harness:
   ```bash
   pip install -e eval/lm-evaluation-harness
   ```

## Reasoning Trace Generation

To generate synthetic reasoning traces using budget forcing:

```bash
# Example: budget forcing steps
python3 reasoning/generate_traces.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split train \
    --mode budget \
    --num_waits 3 \
    --temperature 0.7 \
    --top_p 0.95 \
    --num_samples 2000 \
    --output_file MATH_traces_bf_3w.jsonl
```

### Data Filtering

After generation, use the filtering script to remove traces with repetitive sentences:

```bash
python3 reasoning/filter_repeating_traces.py
```

## Training

Fine-tune the model using LoRA on the filtered dataset:

```bash
python train/lora.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_name_or_path "reasoning/MATH_traces_bf_3w_correct_filtered.jsonl" \
    --dataset_split "train" \
    --prompt_column "problem" \
    --response_column "model_response" \
    --output_dir "./output/qwen2.5-3b-reasoning" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --load_in_4bit \
    --epochs 10
```

## Evaluation

Evaluate the model using `lm-evaluation-harness` with `vLLM` backend:

```bash
lm-eval run \
    --model vllm \
    --model_args pretrained=./output/qwen2.5-3b-reasoning \
    --tasks minerva_math500 \
    --batch_size auto \
    --apply_chat_template \
    --output_path ./results \
    --log_samples
```

---

## Citation

Budget forcing comes from the s1 paper:

```bibtex
@misc{muennighoff2025s1simpletesttimescaling,
      title={s1: Simple test-time scaling},
      author={Niklas Muennighoff and Zitong Yang and Weijia Shi and Xiang Lisa Li and Li Fei-Fei and Hannaneh Hajishirzi and Luke Zettlemoyer and Percy Liang and Emmanuel Candès and Tatsunori Hashimoto},
      year={2025},
      eprint={2501.19393},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.19393},
}
```
