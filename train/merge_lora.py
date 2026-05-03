import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_lora():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen2.5-3b-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Dtype to save the merged model in")
    args = parser.parse_args()

    # Force deterministic ops
    torch.use_deterministic_algorithms(True)

    print(f"Loading base model from {args.base_model_name_or_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.float32,   # ← full precision for merge math
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, trust_remote_code=True
    )

    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    # Cast down only after merge is complete
    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    save_dtype = dtype_map[args.save_dtype]
    if save_dtype != torch.float32:
        print(f"Casting to {args.save_dtype}...")
        model = model.to(save_dtype)

    print(f"Saving to {args.output_dir}...")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    merge_lora()