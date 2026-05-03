import json
import os
import re
import argparse
from collections import Counter

def is_repeating(text, threshold=2, min_len=100):
    """
    Checks if there are repeating sentences or lines in the text.
    
    Args:
        text: The text to check.
        threshold: How many times a sentence can appear before being considered repeating.
        min_len: Minimum length of a sentence to be considered for repetition check.
                 Short sentences are ignored.
    """
    # Split by common sentence endings and newlines
    # This regex splits by . ! ? followed by space/newline, or just newlines
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

def filter_jsonl(input_path, output_path, threshold=2, min_len=100, max_samples=None):
    print(f"Filtering {input_path}...")
    print(f"Settings: threshold={threshold}, min_len={min_len}, max_samples={max_samples}")
    
    total_samples = 0
    filtered_samples = 0
    saved_samples = 0
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if max_samples is not None and saved_samples >= max_samples:
                break
                
            total_samples += 1
            try:
                sample = json.loads(line)
                # Check for repetition in 'model_response'
                trace = sample.get('model_response', '')
                
                if is_repeating(trace, threshold=threshold, min_len=min_len):
                    filtered_samples += 1
                    continue
                
                f_out.write(json.dumps(sample) + '\n')
                saved_samples += 1
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {total_samples}")
                continue

    print(f"Finished.")
    print(f"Processed: {total_samples} samples")
    print(f"Filtered out (repeating): {filtered_samples}")
    print(f"Saved: {saved_samples} samples")
    print(f"Filtered file saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter repeating traces from JSONL file.")
    parser.add_argument("--input", type=str, default="reasoning/MATH_traces_Qwen2.5-3B-Instruct_bf_3w_2000_correct.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file (default: input_filtered.jsonl)")
    parser.add_argument("--threshold", type=int, default=2, help="Repetition threshold (default: 2)")
    parser.add_argument("--min_len", type=int, default=100, help="Minimum sentence length for repetition check (default: 100)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to save in the output file")
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.replace(".jsonl", "_filtered.jsonl")
        if args.output == args.input:
            args.output = args.input + ".filtered"
            
    filter_jsonl(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        min_len=args.min_len,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
