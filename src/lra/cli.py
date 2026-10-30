"""Entry point for one persistent LRA conversation."""
import argparse
from pathlib import Path

from .storage import History
from .app import LRA
from .model import DEFAULT_MODEL


def _print_rows(rows):
    for r in rows:
        print(f'#{r["id"]} [{r["created_at"]}] {r["role"]}: {r["content"]}')


def _remember(history: History, expression: str, *, correction: bool = False):
    if "=" not in expression:
        raise ValueError("Expected KEY=VALUE")
    key, value = expression.split("=", 1)
    if not key.strip() or not value.strip():
        raise ValueError("Expected nonempty KEY=VALUE")
    source = history.append(
        "user", ("/correct " if correction else "/remember ") + expression
    )
    old = history.memory(key)
    memory_id = history.remember(key, value, source_message_id=source)
    print(f"Saved memory #{memory_id}: {key.strip()} (source #{source})")
    if old:
        print(f"Superseded memory #{old['id']} (original source remains available)")


def _command(history, raw: str, embedder=None):
    stripped = raw.strip()
    if stripped in ("/quit", "/exit"):
        return False
    if stripped == "/help":
        print("Commands: /history [N], /search QUERY, /memories, "
              "/remember KEY=VALUE, /correct KEY=VALUE, /stats, /quit")
    elif stripped.startswith("/history"):
        parts = stripped.split()
        count = int(parts[1]) if len(parts) > 1 else 20
        _print_rows(history.messages(limit=max(1, min(count, 500))))
    elif stripped.startswith("/search "):
        _print_rows(history.search(stripped[8:], limit=15, embedder=embedder))
    elif stripped == "/memories":
        for note in history.memories(limit=100):
            print(f'{note["key"]}: {note["value"]} '
                  f'[source #{note["source_message_id"]}]')
    elif stripped.startswith("/remember "):
        _remember(history, stripped[len("/remember "):])
    elif stripped.startswith("/correct "):
        _remember(history, stripped[len("/correct "):], correction=True)
    elif stripped == "/stats":
        print(history.stats())
    else:
        print("Unknown command. Type /help")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(
        description="LRA — one persistent recoverable chat with local SmolLM")
    p.add_argument("--db", type=Path, default=Path("data/lra.sqlite3"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-input-tokens", type=int, default=3072)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--recent-messages", type=int, default=10)
    p.add_argument("--semantic", action="store_true",
                   help="Optional local MiniLM embeddings (requires extra install)")
    p.add_argument("--offline", action="store_true",
                   help="Use already cached models only")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("chat", help="Resume the single persistent chat")
    ask = commands.add_parser("ask", help="One question; save both turns to history")
    ask.add_argument("question", nargs="+")
    commands.add_parser("stats", help="Inspect local canonical history without model")
    commands.add_parser("notes", help="Show active source-linked notes")
    commands.add_parser("history", help="Print last 30 original messages")
    search = commands.add_parser("search", help="Search canonical original history")
    search.add_argument("query", nargs="+")
    remember = commands.add_parser("remember", help="Save KEY=VALUE")
    remember.add_argument("expression")
    correct = commands.add_parser("correct", help="Supersede KEY=VALUE")
    correct.add_argument("expression")
    commands.add_parser("index", help="Backfill optional semantic vectors")
    a = p.parse_args(argv)
    if a.max_input_tokens < 128 or a.max_new_tokens < 1 or a.recent_messages < 1:
        p.error("Invalid context/response limits")

    with History(a.db) as history:
        if a.command in ("stats", "notes", "history", "remember", "correct", "search"):
            if a.command == "stats":
                print(history.stats())
            elif a.command == "notes":
                _command(history, "/memories")
            elif a.command == "history":
                _print_rows(history.messages(limit=30))
            elif a.command == "remember":
                _remember(history, a.expression)
            elif a.command == "correct":
                _remember(history, a.expression, correction=True)
            else:
                _command(history, "/search " + " ".join(a.query))
            return

        embedder = None
        if a.semantic:
            from .semantic import MiniLMEmbedder
            embedder = MiniLMEmbedder(offline=a.offline)
            print(f"Semantic index enabled: {embedder.model_id}")
            print(f"Indexed {history.index_vectors(embedder)} missing messages")

        if a.command == "index":
            if embedder is None:
                p.error("index requires --semantic")
            print(history.stats())
            return

        from .model import SmolLM
        model = SmolLM(
            model_id=a.model, device=a.device,
            max_new_tokens=a.max_new_tokens, offline=a.offline
        )
        app = LRA(history, model, embedder=embedder,
                  max_input_tokens=a.max_input_tokens,
                  recent_messages=a.recent_messages)

        def respond(q):
            result = app.ask(q)
            print("\nSmolLM:", result["answer"])
            if result["retrieved_message_ids"]:
                print("Retrieved original message IDs:",
                      result["retrieved_message_ids"])
            print(f'Context: {result["input_context_tokens"]} tokens; '
                  f'output {result["output_tokens"]} tokens; '
                  f'{result["generation_seconds"]:.2f}s')
            return result

        if a.command == "ask":
            respond(" ".join(a.question))
            return
        print("LRA — single persistent local chat")
        print(f"{history.count()} canonical messages already archived; "
              f"model: {a.model} on {a.device}. /help for commands.")
        while True:
            try:
                raw = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession closed. History retained.")
                break
            if not raw:
                continue
            if raw.startswith("/"):
                try:
                    if not _command(history, raw, embedder=embedder):
                        break
                except ValueError as exc:
                    print(f"Command error: {exc}")
                continue
            try:
                respond(raw)
            except (RuntimeError, ValueError) as exc:
                # The user turn was already archived before the model call.
                print(f"Generation failed: {exc}. Original user turn retained.")


if __name__ == "__main__":
    main()
