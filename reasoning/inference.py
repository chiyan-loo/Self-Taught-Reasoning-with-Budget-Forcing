import sys
import os
import json
import re
import argparse
import numpy as np
from functools import partial
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

def extract_answer(text):
    # Look for the last \boxed{...} in the text
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    
    # Fallback to the whole text if boxed is not found
    return text.strip()

def is_math_equal(pred, gold):
    if pred == gold:
        return True
    
    # Simple normalization for common formatting differences
    def normalize(s):
        if s is None:
            return ""
        s = str(s).strip()
        # Remove whitespace
        s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        # Remove \left and \right
        s = s.replace("\\left", "").replace("\\right", "")
        # Replace \dfrac with \frac
        s = s.replace("\\dfrac", "\\frac")
        return s
        
    return normalize(pred) == normalize(gold)

def main():
    parser = argparse.ArgumentParser(description="Mass inference and evaluation on datasets")
    # Model and Dataset config
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--question_key", type=str, default="problem", help="Column name for the question")
    parser.add_argument("--answer_key", type=str, default="answer", help="Column name for the ground truth answer")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (0 for all)")
    
    # Generation config
    parser.add_argument("--modes", type=str, nargs='+', default=["normal"], choices=["normal", "budget", "alternating"], help="Evaluation modes to run (list: normal, budget, alternating)")
    parser.add_argument("--num_budget_steps", type=int, default=5, help="Number of steps for budget forcing")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for final generation")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    
    # Output config
    parser.add_argument("--output_name", type=str, default="eval", help="Base name for output files (will append _samples_{mode}.jsonl and _stats_{mode}.json)")
    args = parser.parse_args()

    # Ensure outputs are in the eval folder (relative to this script)
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Loading model: {args.model}")
    model = LLM(
        args.model,
        max_model_len=8192,
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
        temperature=0.0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
    )

    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    for mode in args.modes:
        samples_file = os.path.join(eval_dir, f"{args.output_name}_{mode}_samples.jsonl")
        stats_file = os.path.join(eval_dir, f"{args.output_name}_{mode}_stats.json")
        
        # Clear previous output
        if os.path.exists(samples_file):
            os.remove(samples_file)
        if os.path.exists(stats_file):
            os.remove(stats_file)

        print(f"\nEvaluating mode: {mode}")
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        
        # Prepare prompts
        prompts = []
        prompt_token_counts = []
        for example in ds:
            problem = example[args.question_key]
            base_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
            prompt_token_counts.append(len(tok.encode(base_prompt)))
            
            if mode in ["budget", "alternating"]:
                base_prompt += "<|im_start|>think\n"
            prompts.append(base_prompt)

        # Batch Generation
        if mode == "normal":
            print("\nGenerating normal responses...")
            outputs = model.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
            final_responses = [prompts[i] + outputs[i].outputs[0].text for i in range(len(ds))]
        else:
            curr_prompts = list(prompts)
            
            # Initial generation
            print(f"\nGenerating initial responses for {mode} mode...")
            outputs = model.generate(
                curr_prompts, 
                sampling_params=sampling_params, 
                use_tqdm=True
            )
            for i in range(len(ds)):
                curr_prompts[i] += outputs[i].outputs[0].text
                
            # Budget forcing loop
            for step in range(args.num_budget_steps):
                for i in range(len(ds)):
                    if mode == "alternating":
                        ignore_str = "Wait" if step % 2 == 0 else "Alternatively"
                    else:
                        ignore_str = "Wait"
                        
                    curr_prompts[i] += "\n" + ignore_str + "\n"
                
                print(f"\nBudget Step {step+1}/{args.num_budget_steps} for {mode} mode...")
                outputs = model.generate(
                    curr_prompts, 
                    sampling_params=sampling_params, 
                    use_tqdm=True
                )
                for i in range(len(ds)):
                    curr_prompts[i] += outputs[i].outputs[0].text
                                            
            final_responses = list(curr_prompts)

        # Process results
        mode_samples = []
        mode_correct = 0
        mode_token_lengths = []

        for i, (example, final_response) in enumerate(zip(ds, final_responses)):
            problem = example[args.question_key]
            ground_truth = str(example[args.answer_key]).strip()
            prompt_token_count = prompt_token_counts[i]
            
            # Stats calculation
            total_token_count = len(tok.encode(final_response))
            gen_tokens = total_token_count - prompt_token_count
            mode_token_lengths.append(gen_tokens)

            extracted_answer = extract_answer(final_response)
            is_correct = is_math_equal(extracted_answer, ground_truth)
            if is_correct:
                mode_correct += 1
                
            sample = {
                "index": i,
                "problem": problem,
                "ground_truth": ground_truth,
                "model_response": final_response,
                "extracted_answer": extracted_answer,
                "is_correct": bool(is_correct),
                "generated_tokens": gen_tokens
            }
            mode_samples.append(sample)
            
            with open(samples_file, "a") as f:
                f.write(json.dumps(sample) + "\n")

        # Final stats for this mode
        mode_acc = mode_correct / len(ds)
        mode_avg_len = np.mean(mode_token_lengths)

        print("\n" + "="*60)
        print(f" MODE: {mode.upper()} - {args.dataset}")
        print("="*60)
        print(f" Accuracy:        {mode_acc:.2%} ({mode_correct}/{len(ds)})")
        print(f" Avg Gen Tokens:  {mode_avg_len:.1f}")
        print("="*60)
        
        summary_stats = {
            "dataset": args.dataset,
            "mode": mode,
            "num_samples": len(ds),
            "accuracy": mode_acc,
            "avg_gen_len": float(mode_avg_len)
        }
        
        # Output results notification
        print(f" Detailed samples: {samples_file}")
        with open(stats_file, "w") as f:
            json.dump(summary_stats, f, indent=4)
        print(f" Summary stats:    {stats_file}")

if __name__ == "__main__":
    main()