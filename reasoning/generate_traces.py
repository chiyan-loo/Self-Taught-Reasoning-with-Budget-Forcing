import sys
import os
import json
import re
import argparse
import numpy as np
from functools import partial
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from math_verify import parse, verify


def main():
    parser = argparse.ArgumentParser(description="Mass inference and evaluation on datasets")
    # Model and Dataset config
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--question_key", type=str, default="problem", help="Column name for the question")
    parser.add_argument("--answer_key", type=str, default="answer", help="Column name for the ground truth answer")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Maximum model length")
    
    # Generation config
    parser.add_argument("--mode", type=str, default="budget", choices=["budget", "alternating"], help="Generation mode")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of budget forcing steps")
    parser.add_argument("--over_gen_prob", type=float, default=0.2, help="Probability to continue generating after correct answer is found")
    parser.add_argument("--over_gen_exponent", type=float, default=1.0, help="Exponent for probability decay (1.0=linear, 2.0=quadratic, etc.)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per generation step")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output config
    parser.add_argument("--output_file", type=str, default="traces.jsonl", help="Output file name")
    args = parser.parse_args()

    # Ensure outputs are in the eval folder (relative to this script)
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Loading model: {args.model}")
    model = LLM(
        args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)

    # Base sampling params
    stop_token_ids = tok("<|im_end|>")["input_ids"]
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
    )

    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare outputs
    samples_file = os.path.join(eval_dir, args.output_file)
    correct_file = samples_file.replace(".jsonl", "_correct.jsonl")
    
    if os.path.exists(samples_file):
        os.remove(samples_file)
    if os.path.exists(correct_file):
        os.remove(correct_file)

    print(f"\nGenerating traces in {args.mode} mode")
    system_prompt = "Reason step by step, and put your final answer within \\boxed{}."
    
    # Prepare initial prompts
    prompts = []
    for example in ds:
        problem = example[args.question_key]
        base_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n"
        prompts.append(base_prompt)

    active_indices = list(range(len(ds)))
    final_responses = [None] * len(ds)
    is_correct_list = [False] * len(ds)
    steps_taken = [0] * len(ds)

    # Iterative Generation Loop
    for step in range(args.max_steps + 1):
        if not active_indices:
            break
            
        print(f"\n--- Step {step} ({len(active_indices)} active samples) ---")
        current_prompts = [prompts[i] for i in active_indices]
        outputs = model.generate(current_prompts, sampling_params=sampling_params, use_tqdm=True)
        
        new_active_indices = []
        for j, idx in enumerate(active_indices):
            gen_text = outputs[j].outputs[0].text
            prompts[idx] += gen_text
            steps_taken[idx] = step
            
            # Check correctness
            ground_truth = str(ds[idx][args.answer_key]).strip()
            is_correct = verify(parse(ground_truth), parse(prompts[idx]))
            is_correct_list[idx] = is_correct

            # Decide whether to stop
            should_stop = False
            
            # Check if we are approaching max_model_len
            current_tokens = len(outputs[j].prompt_token_ids) + len(outputs[j].outputs[0].token_ids)
            if current_tokens > args.max_model_len - 300:
                print(f"Sample {idx} reached length limit ({current_tokens} tokens). Stopping.")
                should_stop = True

            if not should_stop and is_correct:
                # Calculate decaying over-generation probability
                # More likely to continue earlier in steps
                decay = (1.0 - step / args.max_steps) ** args.over_gen_exponent
                current_over_gen_prob = args.over_gen_prob * decay
                if random.random() > current_over_gen_prob:
                    should_stop = True
            
            if step == args.max_steps:
                should_stop = True
                
            if should_stop:
                final_responses[idx] = prompts[idx]
            else:
                # Continue: append budget token
                token = "Wait"
                if args.mode == "alternating":
                    token = "Wait" if step % 2 == 0 else "Alternatively"
                
                prompts[idx] += f"\n{token}\n"
                new_active_indices.append(idx)
        
        active_indices = new_active_indices

    # Final result collection
    mode_correct = 0
    saved_count = 0
    for i, example in enumerate(ds):
        if final_responses[i] is None:
            final_responses[i] = prompts[i] # Should not happen usually

        problem = example[args.question_key]
        ground_truth = str(example[args.answer_key]).strip()
        is_correct = is_correct_list[i]
        
        if is_correct:
            mode_correct += 1
            
        sample = {
            "index": i,
            "problem": problem,
            "ground_truth": ground_truth,
            "model_response": final_responses[i],
            "is_correct": bool(is_correct),
            "steps": steps_taken[i]
        }
        
        # Always save to main output file
        with open(samples_file, "a") as f:
            f.write(json.dumps(sample) + "\n")
        
        # Also save to correct-only file
        if is_correct:
            with open(correct_file, "a") as f:
                f.write(json.dumps(sample) + "\n")
            saved_count += 1

    print("\n" + "="*60)
    print(f" TRACE GENERATION SUMMARY")
    print("="*60)
    print(f" Total Samples:    {len(ds)}")
    print(f" Total Correct:    {mode_correct} ({mode_correct/len(ds):.2%})")
    print(f" Output (All):     {samples_file}")
    print(f" Output (Correct): {correct_file}")
    print("="*60)

if __name__ == "__main__":
    main()