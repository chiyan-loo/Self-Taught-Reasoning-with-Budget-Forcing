import json
import argparse
import re
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

def get_value(data, path):
    """Retrieve a value from a nested dictionary/list using dot notation and indices."""
    if not path:
        return ""
    
    parts = path.split('.')
    curr = data
    
    try:
        for part in parts:
            if '[' in part and part.endswith(']'):
                # Handle list indexing like 'resps[0]'
                name, idx_str = part[:-1].split('[')
                if name:
                    curr = curr[name]
                curr = curr[int(idx_str)]
            else:
                curr = curr[part]
        
        # If the result is still a list/dict, try to get the first element or a reasonable string
        if isinstance(curr, list) and len(curr) > 0:
            curr = curr[0]
        
        return str(curr) if curr is not None else ""
    except (KeyError, IndexError, TypeError, ValueError):
        return ""

def is_repeating(text, threshold=2, min_len=100):
    """
    Checks if there are repeating sentences or lines in the text.
    """
    if not text:
        return False
    # Split by common sentence endings and newlines
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    
    # Clean and filter sentences
    cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) >= min_len]
    
    if not cleaned_sentences:
        return False
    
    counts = Counter(cleaned_sentences)
    
    for sentence, count in counts.items():
        if count >= threshold:
            return True
            
    return False

def main():
    parser = argparse.ArgumentParser(description="Compute token statistics for a JSONL dataset.")
    parser.add_argument("input_file", type=str, help="Path to the JSONL file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name for tokenizer.")
    parser.add_argument("--prompt_column", type=str, default="problem", help="Column for the prompt (supports dot.notation and [0] indexing).")
    parser.add_argument("--response_column", type=str, default="model_response", help="Column for the response (supports dot.notation and [0] indexing).")
    parser.add_argument("--system_prompt", type=str, default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", help="System prompt to use.")
    parser.add_argument("--repetition_threshold", type=int, default=3, help="Threshold for repeating sentences (default: 3).")
    parser.add_argument("--min_sentence_len", type=int, default=50, help="Minimum sentence length for repetition check (default: 50).")
    parser.add_argument("--correctness_column", type=str, default="math_verify", help="Column for correctness (checks math_verify, then exact_match if not found).")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    sample_data = []
    repeating_count = 0
    
    print(f"Processing {args.input_file}...")
    with open(args.input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                prompt = get_value(data, args.prompt_column)
                response = get_value(data, args.response_column)
                
                if not prompt and not response:
                    # Fallback for common lm-eval structure if default columns fail
                    if "doc" in data and "problem" in data["doc"]:
                        prompt = data["doc"]["problem"]
                    if "filtered_resps" in data and len(data["filtered_resps"]) > 0:
                        response = data["filtered_resps"][0]
                
                # Repetition check
                if is_repeating(response, threshold=args.repetition_threshold, min_len=args.min_sentence_len):
                    repeating_count += 1

                # Correctness check
                correct_val = data.get(args.correctness_column)
                if correct_val is None:
                    correct_val = data.get("exact_match")
                
                # Handle list if necessary
                if isinstance(correct_val, list) and len(correct_val) > 0:
                    correct_val = correct_val[0]
                
                is_correct = False
                if isinstance(correct_val, (int, float)):
                    is_correct = correct_val > 0
                elif isinstance(correct_val, bool):
                    is_correct = correct_val
                elif isinstance(correct_val, str):
                    is_correct = correct_val.lower() in ("true", "1", "yes", "correct")

                # Format exactly like lora.py
                user_msg = f"Problem:\n{prompt}\n\nSolution:"
                assistant_msg = response
                
                # If the assistant_msg looks like a full chat (contains assistant tag), extract only the assistant part
                if "<|im_start|>assistant" in assistant_msg:
                    parts = assistant_msg.split("<|im_start|>assistant\n")
                    if len(parts) > 1:
                        assistant_msg = parts[-1].split("<|im_end|>")[0].strip()

                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                
                messages.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ])
                
                # Apply chat template
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                
                # Tokenize
                tokens = tokenizer.encode(text)
                sample_data.append({
                    "len": len(tokens),
                    "correct": is_correct
                })
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
                
    if not sample_data:
        print("No data found.")
        return
    
    total_lens = [s["len"] for s in sample_data]
    total_samples = len(sample_data)
    total_correct = sum(1 for s in sample_data if s["correct"])
        
    print("\n" + "="*40)
    print("📊 DATASET TOKEN STATISTICS")
    print(f"  File:           {args.input_file}")
    print(f"  Total samples:  {total_samples}")
    print(f"  Overall Accuracy: {total_correct/total_samples*100:.2f}%")
    print("="*40)
    
    print(f"  Average tokens: {np.mean(total_lens):.2f}")
    print(f"  Min tokens:     {np.min(total_lens)}")
    print(f"  Max tokens:     {np.max(total_lens)}")
    print(f"  Median (P50):   {np.percentile(total_lens, 50):.2f}")
    print(f"  P90:            {np.percentile(total_lens, 90):.2f}")
    print(f"  P95:            {np.percentile(total_lens, 95):.2f}")
    print(f"  P99:            {np.percentile(total_lens, 99):.2f}")
    
    print(f"\n  🔄 REPETITION STATISTICS")
    print(f"  Samples with repetition: {repeating_count} ({repeating_count/total_samples*100:.2f}%)")
    print(f"  Threshold:               {args.repetition_threshold}")
    print(f"  Min sentence length:     {args.min_sentence_len}")

    # Check distribution and accuracy in ranges
    ranges = [
        (0, 512), 
        (512, 1024), 
        (1024, 2048), 
        (2048, 4096), 
        (4096, 8192), 
        (8192, 16384),
        (16384, 1000000)
    ]
    
    print("\nRange Distribution and Accuracy:")
    for low, high in ranges:
        samples_in_range = [s for s in sample_data if low <= s['len'] < high]
        if not samples_in_range:
            continue
        
        count = len(samples_in_range)
        correct_in_range = sum(1 for s in samples_in_range if s['correct'])
        accuracy = (correct_in_range / count * 100) if count > 0 else 0
        
        range_label = f"{low}-{high}" if high < 1000000 else f"{low}+"
        print(f"  {range_label:13} tokens: {count:5} ({count/total_samples*100:6.2f}%) | Accuracy: {accuracy:6.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
