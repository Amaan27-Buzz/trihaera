"""Optional live demo: learned search policy + self-hosted SearXNG.

LRA's normal single companion chat is unchanged. This is an isolated
search-native research prototype. It *requires* a separately running local
SearXNG with JSON enabled (search results are real, not prewritten fixtures).
No Tavily, hosted inference, or cloud LLM is used.
"""
import argparse
import json

from lra.model import DEFAULT_MODEL, SmolLM
from lra.search_runtime import LocalSearXNG, SearchRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question", nargs="+")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", help="Path to trained search-native LoRA adapter")
    p.add_argument("--endpoint", default="http://127.0.0.1:8080")
    p.add_argument("--offline", action="store_true",
                   help="Disable Hugging Face model downloads; web searches remain live")
    p.add_argument("--max-new-tokens", type=int, default=192)
    a = p.parse_args()
    model = SmolLM(
        model_id=a.model, device=a.device, offline=a.offline,
        max_new_tokens=a.max_new_tokens,
    )
    if a.adapter:
        from pathlib import Path
        from peft import PeftModel
        if not (Path(a.adapter) / "adapter_config.json").is_file():
            p.error("Missing trained adapter at --adapter")
        model.model = PeftModel.from_pretrained(
            model.model, a.adapter, is_trainable=False
        ).eval()
    runner = SearchRuntime(model, LocalSearXNG(a.endpoint))
    try:
        result = runner.ask(" ".join(a.question))
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        p.exit(2, f"Search policy/backend failed: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
