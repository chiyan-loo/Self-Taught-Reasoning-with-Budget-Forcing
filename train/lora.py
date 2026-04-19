import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    
    # Model Arguments
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="Qwen/Qwen2.5-3B-Instruct", 
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    
    # Data Arguments
    parser.add_argument(
        "--dataset_name_or_path", 
        type=str, 
        required=True, 
        help="Dataset name or path"
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        default="text", 
        help="Column containing the text to train on"
    )
    parser.add_argument(
        "--dataset_split", 
        type=str, 
        default="train", 
        help="Which split of the dataset to load"
    )
    parser.add_argument(
        "--prompt_column", 
        type=str, 
        default="problem", 
        help="Column containing the question/instruction"
    )
    parser.add_argument(
        "--response_column", 
        type=str, 
        default="solution", 
        help="Column containing the answer/response"
    )
    parser.add_argument(
        "--max_train_samples", 
        type=int, 
        default=None, 
        help="Limit the number of training samples"
    )
    parser.add_argument(
        "--shuffle_seed", 
        type=int, 
        default=42, 
        help="Seed for shuffling the dataset"
    )
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default="Reason step by step, and put your final answer within \\boxed{}.",
        help="System prompt to use during training"
    )
    
    # LoRA Arguments
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16, 
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32, 
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05, 
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=2048, 
        help="Maximum sequence length"
    )

    
    # Training Arguments
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4, 
        help="Initial learning rate (AdamW optimizer)"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=2, 
        help="Total number of training epochs to perform"
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=4, 
        help="Batch size per GPU/CPU for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4, 
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass"
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=10, 
        help="Number of steps between logging"
    )
    
    # Memory Optimization Arguments
    parser.add_argument(
        "--load_in_4bit", 
        action="store_true", 
        default=True,
        help="Load the model in 4-bit quantization (QLoRA)"
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="Load the model in 8-bit quantization"
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true", 
        default=True,
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--optim", 
        type=str, 
        default="paged_adamw_32bit", 
        help="The optimizer to use"
    )
    
    # Parse known arguments first (our custom ones)
    args, unknown = parser.parse_known_args()
    
    # Use HfArgumentParser for SFTConfig to handle unknown args
    hf_parser = HfArgumentParser((SFTConfig,))
    training_args = hf_parser.parse_args_into_dataclasses(args=unknown)[0]
    
    # Override explicitly parsed training args
    training_args.learning_rate = args.learning_rate
    training_args.num_train_epochs = args.num_train_epochs
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_args.dataset_text_field = args.text_column
    training_args.max_length = args.max_seq_length
    training_args.gradient_checkpointing = args.gradient_checkpointing
    training_args.optim = args.optim
    
    # Auto-set bf16/fp16 based on hardware
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            training_args.bf16 = True
        else:
            training_args.fp16 = True
    
    return args, training_args

def main(args, training_args):
    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load base model with optional quantization
    print(f"Loading base model from {args.model_name_or_path}...")
    
    bnb_config = None
    if args.load_in_4bit:
        print("Using 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.load_in_8bit:
        print("Using 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 3. Apply LoRA Config
    print("Applying PEFT/LoRA configuration...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Default targeting for standard model architectures like Llama/Mistral/Qwen
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )
    
    # 4. Load Dataset
    print(f"Loading dataset from {args.dataset_name_or_path}...")
    dataset = load_dataset(args.dataset_name_or_path, split=args.dataset_split)

    print(f"Shuffling dataset with seed {args.shuffle_seed}...")
    dataset = dataset.shuffle(seed=args.shuffle_seed)
    
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))
        print(f"Subsetted dataset to {len(dataset)} samples.")
    
    # 5. Handle missing text column with automatic formatting
    if args.text_column not in dataset.column_names:
        print(f"Column '{args.text_column}' not found. Attempting to format from available columns...")
        
        def format_example(example):
            # Strict detection of specified columns
            if args.prompt_column not in example or args.response_column not in example:
                raise ValueError(
                    f"Dataset is missing required columns: '{args.prompt_column}' or '{args.response_column}'. "
                    f"Found columns: {list(example.keys())}. "
                    f"Use --prompt_column and --response_column to specify the correct fields."
                )
            
            user_msg = f"Problem:\n{example[args.prompt_column]}\n\nSolution:"
            assistant_msg = example[args.response_column]

            # Apply chat template if available, otherwise use a generic format
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            
            messages.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ])
            
            try:
                if tokenizer.chat_template is not None:
                    example[args.text_column] = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
            except Exception:
                # Fallback if chat template fails (e.g. missing system role requirement)
                raise ValueError(
                    f"Chat template failed for example: {example}. "
                )
                
            return example

        dataset = dataset.map(format_example, desc="Formatting dataset")
        
        print("\n--- Example Processed Training Sample ---")
        print(dataset[0][args.text_column])
        print("------------------------------------------\n")
    
    # 6. Initialize SFTTrainer from trl
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # 7. Train and Save
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final model adapter to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Done!")

if __name__ == "__main__":
    args, training_args = parse_args()
    main(args, training_args)
