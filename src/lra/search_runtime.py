"""Model-led multi-step search with auditable sources.

The retrieval backend is a self-hosted search metasearch service. The model,
after targeted LoRA training, generates structured search decisions itself.
A deterministic engine performs I/O, validates source IDs and budgets, and
never executes code or URLs provided by the model.
"""
from __future__ import annotations
import json
from urllib.parse import urlsplit, urlencode
from urllib.request import urlopen

from .search_examples import to_messages
from .search_native import parse_decision, render_state, POLICY_SYSTEM


class LocalSearXNG:
    def __init__(self, endpoint: str = "http://127.0.0.1:8080",
                 timeout: int = 10):
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or
            parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
            raise ValueError("Only a loopback HTTP SearXNG endpoint is accepted")
        if timeout < 1 or timeout > 30:
            raise ValueError("Invalid search timeout")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, *, limit: int = 4) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 180:
            raise ValueError("Invalid search query")
        if limit not in range(1, 9):
            raise ValueError("Invalid result limit")
        url = f"{self.endpoint}/search?" + urlencode({
            "q": query, "format": "json", "language": "en",
        })
        with urlopen(url, timeout=self.timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Search backend returned HTTP {response.status}")
            payload = response.read(512_000 + 1)
        if len(payload) > 512_000:
            raise RuntimeError("Search result payload too large")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
            raise RuntimeError("Invalid JSON response; is SearXNG JSON enabled?")
        out = []
        urls = set()
        for result in parsed["results"]:
            if not isinstance(result, dict):
                continue
            target = result.get("url", "")
            if not isinstance(target, str):
                continue
            parts = urlsplit(target)
            if parts.scheme not in ("https", "http") or not parts.hostname:
                continue
            if target in urls:
                continue
            urls.add(target)
            title = result.get("title", "")
            snippet = result.get("content", "")
            if not isinstance(title, str) or not isinstance(snippet, str):
                continue
            out.append({
                "title": title[:140], "text": snippet[:650],
                "url": target,
            })
            if len(out) >= limit:
                break
        return out


class SearchRuntime:
    def __init__(self, model, backend, *, max_rounds: int = 2,
                 results_per_query: int = 3, max_sources: int = 8):
        if not 1 <= max_rounds <= 4 or not 1 <= results_per_query <= 8:
            raise ValueError("Invalid research budget")
        if not 1 <= max_sources <= 8:
            raise ValueError("Source budget must be between 1 and 8")
        self.model = model
        self.backend = backend
        self.max_rounds = max_rounds
        self.results_per_query = results_per_query
        self.max_sources = max_sources

    def ask(self, question: str) -> dict:
        if not question.strip():
            raise ValueError("Question required")
        seen_urls = set()
        sources = []
        queries = []
        steps = []
        for step in range(self.max_rounds + 1):
            state = render_state(question, sources, queries)
            generated = self.model.generate([
                {"role": "system", "content": POLICY_SYSTEM},
                {"role": "user", "content": state},
            ])
            action = parse_decision(
                generated.text, valid_sources={s["id"] for s in sources}
            )
            steps.append({
                "decision": action.decision,
                "queries": list(action.queries),
                "citations": list(action.citations),
                "raw_model_output": generated.text,
            })
            if action.decision == "abstain":
                return {
                    "status": "abstain", "answer": action.answer,
                    "sources": [], "steps": steps, "search_rounds": step,
                }
            if action.decision == "answer":
                if sources and not action.citations:
                    return {
                        "status": "unverified", "answer": (
                            "I couldn't verify an answer with source citations."
                        ), "sources": sources, "steps": steps,
                        "search_rounds": step,
                    }
                cited = [s for s in sources if s["id"] in action.citations]
                return {
                    "status": "answer", "answer": action.answer,
                    "sources": cited, "steps": steps, "search_rounds": step,
                }
            if step >= self.max_rounds:
                return {
                    "status": "budget_exhausted",
                    "answer": "I couldn't establish a sourced answer within the search budget.",
                    "sources": sources, "steps": steps, "search_rounds": step,
                }
            fresh = 0
            for query in action.queries:
                if len(sources) >= self.max_sources:
                    break
                queries.append(query)
                for item in self.backend.search(query, limit=self.results_per_query):
                    url = item.get("url")
                    if not isinstance(url, str) or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    sources.append({
                        "id": f"s{len(sources)+1}",
                        "title": item.get("title", ""),
                        "text": item.get("text", ""),
                        "url": url,
                    })
                    fresh += 1
                    if len(sources) >= self.max_sources:
                        break
            if not fresh:
                return {
                    "status": "no_results",
                    "answer": "I couldn't find relevant sources for that question.",
                    "sources": [], "steps": steps, "search_rounds": step + 1,
                }
        raise AssertionError("Unreachable search state")
