"""Regression tests for baseline failures observed on Windows RTX 5060.

Run without torch, transformers, downloaded models, or a GPU.
"""
from types import SimpleNamespace
from pathlib import Path

from model_lab.cases import needle_case
from scripts.benchmark_models import build_needle, count_chat_tokens


class FakeChatTokenizer:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            return_dict, return_tensors):
        assert tokenize and add_generation_prompt and return_dict
        assert return_tensors == "pt"
        self.calls += 1
        # Shape is intentionally tensor-like, not a token list; this
        # reproduces why len(outer BatchEncoding) is not a safe token count.
        tokens = 19 + sum(len(message["content"].split()) for message in messages)
        return {"input_ids": SimpleNamespace(shape=(1, tokens))}


def test_context_probe_constructs_real_512_token_input():
    tokenizer = FakeChatTokenizer()
    result = build_needle(tokenizer, 512)
    count = count_chat_tokens(tokenizer, result.chat())
    assert count >= 512
    assert count < 530
    assert tokenizer.calls < 32


def test_context_probe_constructs_real_2048_token_input():
    tokenizer = FakeChatTokenizer()
    result = build_needle(tokenizer, 2048)
    count = count_chat_tokens(tokenizer, result.chat())
    assert count >= 2048
    assert count < 2070
    assert tokenizer.calls < 36


def test_context_probes_keep_original_and_final_turns():
    tokenizer = FakeChatTokenizer()
    case = build_needle(tokenizer, 512)
    assert "glacier lime 407" in case.chat()[0]["content"]
    assert case.chat()[-1]["role"] == "user"


def test_no_auto_processor_in_text_only_qwen_loader():
    import inspect
    from scripts.benchmark_models import load_reference
    source = inspect.getsource(load_reference)
    assert "AutoModelForMultimodalLM" in source
    assert "AutoProcessor.from_pretrained" not in source
    assert "tokenizer = AutoTokenizer.from_pretrained" in source
