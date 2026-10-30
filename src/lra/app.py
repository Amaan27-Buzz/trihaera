"""LRA: one chronological chat, bounded working context, durable history."""
from .context import build_context
from .memory import extract_explicit_facts


class LRA:
    def __init__(self, history, model, *, embedder=None,
                 max_input_tokens: int = 3072, recent_messages: int = 10):
        self.history = history
        self.model = model
        self.embedder = embedder
        self.max_input_tokens = max_input_tokens
        self.recent_messages = recent_messages

    def ask(self, user_text: str) -> dict:
        if not user_text.strip():
            raise ValueError("No user message")
        source_id = self.history.append(
            "user", user_text.strip(), embedder=self.embedder
        )
        for key, value in extract_explicit_facts(user_text):
            self.history.remember(key, value, source_message_id=source_id)
        cap = min(
            self.max_input_tokens,
            self.model.window - self.model.max_new_tokens - 16
        )
        plan = build_context(
            self.history, user_text, token_count=self.model.count_chat_tokens,
            budget=cap, recent_messages=self.recent_messages,
            embedder=self.embedder
        )
        generated = self.model.generate(plan.messages)
        assistant_id = None
        if generated.text:
            assistant_id = self.history.append(
                "assistant", generated.text, embedder=self.embedder
            )
        return {
            "answer": generated.text,
            "user_message_id": source_id,
            "assistant_message_id": assistant_id,
            "retrieved_message_ids": list(plan.retrieved_ids),
            "active_memory_keys": list(plan.memory_keys),
            "input_context_tokens": plan.input_tokens,
            "context_compacted": plan.truncated,
            "generation_seconds": generated.latency_seconds,
            "output_tokens": generated.output_tokens,
            "peak_cuda_allocated_bytes": generated.max_cuda_allocated_bytes,
        }
