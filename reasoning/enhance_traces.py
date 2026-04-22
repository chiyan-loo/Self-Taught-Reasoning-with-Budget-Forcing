"""
Script to enhance reasoning traces via a multi-step LLM workflow.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Optional

try:
    from reasoning.enhance_traces_prompts import (
        STEP1_ELABORATED_REASONING_SYSTEM,
        STEP2_SELF_VERIFICATION_SYSTEM,
        STEP3_EXPLORATORY_APPROACH_SYSTEM,
    )
except ImportError:
    from enhance_traces_prompts import (
        STEP1_ELABORATED_REASONING_SYSTEM,
        STEP2_SELF_VERIFICATION_SYSTEM,
        STEP3_EXPLORATORY_APPROACH_SYSTEM,
    )

def _extract_trace(text: str) -> str:
    """Extract the reasoning trace from the XML tags. Returns the longest match."""
    # First priority: explicit enhanced_trace tags
    matches = re.findall(r"<enhanced_trace>(.*?)</enhanced_trace>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        longest_match = max(matches, key=len).strip()
        if longest_match:
            return longest_match
    
    # Second priority: strip <think> blocks and return the rest
    # This is useful for reasoning models that might ignore our custom tags
    # but follow the model-native thinking format.
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    if not clean_text or clean_text == "...":
         # Fallback if the body is empty or just dots - maybe it put everything IN the think block?
         # (Though usually we want to avoid this, it is better than nothing)
         think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
         if think_match:
             return think_match.group(1).strip()

    return clean_text or text.strip()

def _make_user_prompt(problem: str, trace: str) -> str:
    """Build the user prompt containing the original problem + trace."""
    return (
        f"### Original Problem\n{problem}\n\n"
        f"### Reasoning Trace to Enhance\n{trace}"
    )


# The ordered pipeline of enhancement steps.
ENHANCEMENT_STEPS = [
    {
        "name": "Elaborated Reasoning",
        "system": STEP1_ELABORATED_REASONING_SYSTEM,
    },
    {
        "name": "Self-Verification",
        "system": STEP2_SELF_VERIFICATION_SYSTEM,
    },
    {
        "name": "Exploratory Approach",
        "system": STEP3_EXPLORATORY_APPROACH_SYSTEM,
    },
]


try:
    from reasoning.backends import APIBackend, VLLMBackend
except ImportError:
    from backends import APIBackend, VLLMBackend



# ---------------------------------------------------------------------------
# Data loading / saving helpers
# ---------------------------------------------------------------------------


def load_input_data(
    input_path: str,
    dataset_name: Optional[str],
    dataset_split: str,
    problem_column: str,
    trace_column: str,
    max_samples: Optional[int],
    shuffle_seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Load the reasoning dataset from a JSONL file or HuggingFace dataset.

    Returns a list of dicts, each containing at least 'problem' and
    'reasoning_trace' keys (plus any other original columns).
    """
    records: list[dict[str, Any]] = []

    if input_path and os.path.isfile(input_path):
        print(f"Loading data from JSONL file: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif dataset_name:
        print(f"Loading HuggingFace dataset: {dataset_name} (split={dataset_split})")
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=dataset_split)
        for item in ds:
            records.append(dict(item))
    else:
        print(
            "ERROR: You must provide either --input_file (path to a JSONL "
            "file) or --dataset_name (a HuggingFace dataset identifier).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Normalize column names
    for rec in records:
        if problem_column != "problem" and problem_column in rec:
            rec["problem"] = rec[problem_column]
        if trace_column != "reasoning_trace" and trace_column in rec:
            rec["reasoning_trace"] = rec[trace_column]

    # Validate required fields
    for i, rec in enumerate(records):
        if "problem" not in rec:
            print(
                f"ERROR: Record {i} is missing a '{problem_column}' field. "
                f"Available keys: {list(rec.keys())}. "
                f"Use --problem_column to specify the correct field.",
                file=sys.stderr,
            )
            sys.exit(1)
        if "reasoning_trace" not in rec:
            print(
                f"ERROR: Record {i} is missing a '{trace_column}' field. "
                f"Available keys: {list(rec.keys())}. "
                f"Use --trace_column to specify the correct field.",
                file=sys.stderr,
            )
            sys.exit(1)

    if shuffle_seed is not None:
        print(f"Shuffling dataset with seed {shuffle_seed}…")
        random.seed(shuffle_seed)
        random.shuffle(records)

    if max_samples is not None:
        records = records[:max_samples]
        print(f"Limiting to {len(records)} samples.")

    print(f"Loaded {len(records)} samples.")
    return records


def save_output_data(records: list[dict[str, Any]], output_path: str) -> None:
    """Save enhanced records to a JSONL file."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} enhanced records to {output_path}")


# ---------------------------------------------------------------------------
# Multi-step enhancement pipeline
# ---------------------------------------------------------------------------


def enhance_dataset(
    records: list[dict[str, Any]],
    backend: APIBackend | VLLMBackend,
    steps: list[dict[str, str]],
    batch_size: int = 8,
    save_intermediates: bool = False,
    output_path: str = "",
    debug_path: Optional[str] = None,
    responses_log_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Run the multi-step enhancement pipeline over all records.

    Each step rewrites the current reasoning trace to layer on a new
    reasoning habit.  The trace is updated in-place through the steps,
    and the final version is stored in the 'enhanced_reasoning_trace' key.
    The raw (unextracted) model output is stored in 'raw_enhanced_reasoning_trace'.
    """
    total = len(records)

    # Initialize responses log if requested
    log_file = None
    if responses_log_path:
        log_dir = os.path.dirname(responses_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(responses_log_path, "w", encoding="utf-8")
        print(f"Logging raw LLM responses to: {responses_log_path}")

    try:
        for step_idx, step in enumerate(steps):
            step_name = step["name"]
            system_prompt = step["system"]
            print(f"\n{'='*70}")
            print(
                f"Step {step_idx + 1}/{len(steps)}: {step_name}  "
                f"({total} samples)"
            )
            print(f"{'='*70}")

            # Determine which trace to use as input for this step
            trace_key = (
                "reasoning_trace"
                if step_idx == 0
                else "enhanced_reasoning_trace"
            )

            # Process in batches
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch = records[batch_start:batch_end]

                print(
                    f"  Processing samples {batch_start + 1}–{batch_end} "
                    f"of {total}…"
                )

                user_prompts = [
                    _make_user_prompt(rec["problem"], rec[trace_key])
                    for rec in batch
                ]

                try:
                    results = backend.generate_batch(system_prompt, user_prompts)
                except Exception as e:
                    print(
                        f"  ✗ Batch failed ({e}). Falling back to per-sample "
                        f"generation…"
                    )
                    results = []
                    for j, prompt in enumerate(user_prompts):
                        try:
                            result = backend.generate(system_prompt, prompt)
                            results.append(result)
                        except Exception as inner_e:
                            print(
                                f"    ✗ Sample {batch_start + j + 1} failed: "
                                f"{inner_e}. Keeping original trace."
                            )
                            results.append(batch[j][trace_key])

                # Store results
                for j, result in enumerate(results):
                    idx = batch_start + j
                    user_prompt = user_prompts[j]
                    
                    # Log raw response if requested
                    if log_file:
                        log_entry = {
                            "sample_idx": idx,
                            "step": step_name,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "raw_response": result
                        }
                        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        log_file.flush()

                    extracted_result = _extract_trace(result)
                    records[idx]["enhanced_reasoning_trace"] = extracted_result
                    records[idx]["raw_enhanced_reasoning_trace"] = result

                    # Optionally keep per-step snapshots for inspection
                    if save_intermediates:
                        base_key = step_name.lower().replace(' ', '_')
                        records[idx][f"trace_after_step{step_idx + 1}_{base_key}"] = extracted_result
                        records[idx][f"raw_trace_after_step{step_idx + 1}_{base_key}"] = result

            # Save debug info after each step
            if save_intermediates:
                actual_debug_path = debug_path or (output_path.replace(".jsonl", "_debug.jsonl") if output_path else "debug_traces.jsonl")
                save_output_data(records, actual_debug_path)
                print(f"  ✓ Debug file updated: {actual_debug_path}")

    finally:
        if log_file:
            log_file.close()

    return records



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhance reasoning traces via a multi-step LLM workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input / Output ---
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to a JSONL file with reasoning traces to enhance.",
    )
    io_group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "HuggingFace dataset identifier (alternative to --input_file). "
            "The dataset must contain problem and trace columns."
        ),
    )
    io_group.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Which split to load when using --dataset_name.",
    )
    io_group.add_argument(
        "--output_file",
        type=str,
        default="results/enhanced_traces.jsonl",
        help="Where to save the final enhanced traces.",
    )
    io_group.add_argument(
        "--debug_file",
        type=str,
        default="results/debug_traces.jsonl",
        help="Where to save intermediate debug traces (defaults to output_file + '_debug.jsonl').",
    )
    io_group.add_argument(
        "--responses_log",
        type=str,
        default=None,
        help="Path to a file where all raw LLM prompts and responses will be logged.",
    )
    io_group.add_argument(
        "--problem_column",
        type=str,
        default="problem",
        help="Column name containing the problem / question.",
    )
    io_group.add_argument(
        "--trace_column",
        type=str,
        default="solution",
        help="Column name containing the existing reasoning trace.",
    )
    io_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples to process.",
    )
    io_group.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Seed for shuffling the dataset before processing. Set to None to disable shuffling.",
    )

    # --- Backend ---
    backend_group = parser.add_argument_group("Backend")
    backend_group.add_argument(
        "--backend",
        type=str,
        choices=["api", "vllm"],
        default="vllm",
        help="Inference backend: 'api' (OpenAI-compatible) or 'vllm' (local).",
    )
    backend_group.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name / path for generation.",
    )

    # --- API-specific ---
    api_group = parser.add_argument_group("API Backend Options")
    api_group.add_argument(
        "--api_base",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI-compatible API.",
    )
    api_group.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "API key. Defaults to the OPENAI_API_KEY environment variable "
            "if not provided."
        ),
    )
    api_group.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of retries on API errors.",
    )

    # --- vLLM-specific ---
    vllm_group = parser.add_argument_group("vLLM Backend Options")
    vllm_group.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM's KV cache.",
    )
    vllm_group.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model length (context size). Useful for large models on smaller GPUs.",
    )

    # --- Generation ---
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens to generate per enhancement step.",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    gen_group.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter.",
    )
    gen_group.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of samples to process per batch.",
    )

    # --- Pipeline control ---
    pipeline_group = parser.add_argument_group("Pipeline Control")
    pipeline_group.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["elaborated", "verification", "exploratory"],
        choices=["elaborated", "verification", "exploratory"],
        help=(
            "Which enhancement steps to run, and in what order. "
            "Default: all three in sequence."
        ),
    )
    pipeline_group.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Save intermediate traces after each step (for debugging).",
    )


    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEP_MAP = {
    "elaborated": ENHANCEMENT_STEPS[0],
    "verification": ENHANCEMENT_STEPS[1],
    "exploratory": ENHANCEMENT_STEPS[2],
}


def main():
    args = parse_args()

    # ---- Resolve API key ----
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if args.backend == "api" and not api_key:
        print(
            "ERROR: An API key is required for the API backend. "
            "Set --api_key or the OPENAI_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Select enhancement steps ----
    selected_steps = [STEP_MAP[s] for s in args.steps]
    step_names = " → ".join(s["name"] for s in selected_steps)
    print(f"Enhancement pipeline: {step_names}")

    # ---- Initialize backend ----
    print(f"\nInitializing {args.backend.upper()} backend (model={args.model})…")
    if args.backend == "api":
        backend = APIBackend(
            model=args.model,
            api_base=args.api_base,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_retries=args.max_retries,
        )
    else:
        backend = VLLMBackend(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

    # ---- Load data ----
    records = load_input_data(
        input_path=args.input_file,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        problem_column=args.problem_column,
        trace_column=args.trace_column,
        max_samples=args.max_samples,
        shuffle_seed=args.shuffle_seed,
    )

    if not records:
        print("No records to process. Exiting.")
        sys.exit(0)

    # ---- Print a preview ----
    print("\n--- Preview of first sample ---")
    print(f"Problem:  {records[0]['problem'][:200]}…")
    trace_preview = records[0]["reasoning_trace"][:300]
    print(f"Trace:    {trace_preview}…")
    print("-------------------------------\n")

    # ---- Run the enhancement pipeline ----
    t_start = time.time()
    records = enhance_dataset(
        records=records,
        backend=backend,
        steps=selected_steps,
        batch_size=args.batch_size,
        save_intermediates=args.save_intermediates,
        output_path=args.output_file,
        debug_path=args.debug_file,
        responses_log_path=args.responses_log,
    )
    elapsed = time.time() - t_start
    print(f"\n✓ Enhancement complete in {elapsed:.1f}s")

    # ---- Save ----
    save_output_data(records, args.output_file)

    # ---- Summary stats ----
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Samples processed : {len(records)}")
    print(f"  Steps applied     : {step_names}")
    print(f"  Output file       : {args.output_file}")
    if args.responses_log:
        print(f"  Responses log     : {args.responses_log}")

    # Show a brief comparison for the first sample
    if records:
        orig = records[0]["reasoning_trace"]
        enhanced = records[0].get("enhanced_reasoning_trace", "")
        print(f"  Avg original len  : {sum(len(r['reasoning_trace']) for r in records) / len(records):.0f} chars")
        if enhanced:
            print(f"  Avg enhanced len  : {sum(len(r.get('enhanced_reasoning_trace', '')) for r in records) / len(records):.0f} chars")
            ratio = (
                sum(len(r.get("enhanced_reasoning_trace", "")) for r in records)
                / max(sum(len(r["reasoning_trace"]) for r in records), 1)
            )
            print(f"  Expansion ratio   : {ratio:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
