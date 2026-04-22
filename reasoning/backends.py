import sys
import time
from typing import Optional

class APIBackend:
    """OpenAI-compatible chat completion backend."""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            print(
                "ERROR: The 'openai' package is required for the API backend.\n"
                "Install it with: pip install openai",
                file=sys.stderr,
            )
            sys.exit(1)

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request with retry logic."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    print(
                        f"  ⚠ API error (attempt {attempt}/{self.max_retries}): "
                        f"{e}. Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"  ✗ API error (attempt {attempt}/{self.max_retries}): "
                        f"{e}. Giving up on this sample."
                    )
                    raise

    def generate_batch(
        self, system_prompt: str, user_prompts: list[str]
    ) -> list[str]:
        """Generate completions for a batch of prompts (sequential for API)."""
        results = []
        for prompt in user_prompts:
            results.append(self.generate(system_prompt, prompt))
        return results


class VLLMBackend:
    """vLLM in-process generation backend."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
    ):
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            print(
                "ERROR: 'vllm' and 'transformers' packages are required for "
                "the vLLM backend.\n"
                "Install them with: pip install vllm transformers",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Loading vLLM model {model}…")
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self.model_name = model

    def _format_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Apply the model's chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback for tokenizers without a chat template
        return (
            f"System: {system_prompt}\n\n"
            f"User: {user_prompt}\n\n"
            f"Assistant: "
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a single completion."""
        formatted = self._format_chat(system_prompt, user_prompt)
        outputs = self.llm.generate([formatted], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self, system_prompt: str, user_prompts: list[str]
    ) -> list[str]:
        """Generate completions for a batch (vLLM handles batching natively)."""
        formatted = [
            self._format_chat(system_prompt, p) for p in user_prompts
        ]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
