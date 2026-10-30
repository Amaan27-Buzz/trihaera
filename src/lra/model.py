"""Local Hugging Face SmolLM inference; CUDA first, no remote API.

The small SmolLM2 checkpoint is a *baseline* for testing LRA retrieval.
It is not claimed to have unlimited native attention or strong reasoning.
"""
from dataclasses import dataclass
import time


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


@dataclass
class ModelResult:
    text: str
    latency_seconds: float
    prompt_tokens: int
    output_tokens: int
    max_cuda_allocated_bytes: int | None


class SmolLM:
    def __init__(self, *, model_id: str = DEFAULT_MODEL,
                 device: str = "cuda", max_new_tokens: int = 192,
                 offline: bool = False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; use --device cpu")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.torch = torch
        self.device = torch.device(device)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=False, local_files_only=offline
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=False, use_safetensors=True,
            local_files_only=offline, dtype=dtype
        ).to(self.device).eval()
        self.window = int(getattr(self.model.config, "max_position_embeddings", 8192))
        if self.tokenizer.chat_template is None:
            raise RuntimeError("Please select an instruct checkpoint with a chat template")
        if self.window <= self.max_new_tokens + 64:
            raise ValueError("Insufficient model context for requested generation budget")

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def count_chat_tokens(self, messages: list[dict]) -> int:
        prompt = self.render(messages)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(self, messages: list[dict]) -> ModelResult:
        torch = self.torch
        formatted = self.render(messages)
        encoded = self.tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        count = int(encoded["input_ids"].shape[-1])
        if count + self.max_new_tokens > self.window:
            raise ValueError(
                f"Prompt {count} + output budget {self.max_new_tokens} "
                f"exceeds model context window {self.window}"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output = generated[0, count:]
        text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        peak = (int(torch.cuda.max_memory_allocated())
                if self.device.type == "cuda" else None)
        return ModelResult(
            text=text, latency_seconds=elapsed, prompt_tokens=count,
            output_tokens=int(output.numel()), max_cuda_allocated_bytes=peak
        )
