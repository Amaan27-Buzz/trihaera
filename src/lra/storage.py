"""SQLite canonical event history, source-linked memories and hybrid retrieval.

No conversation segmentation: this database is ONE continuous chat. Every
message remains intact after prompt compaction. Old memory revisions remain
auditable; correcting a note does not rewrite past messages.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _words(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", value.casefold(), re.UNICODE)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    aa = math.sqrt(sum(x*x for x in a))
    bb = math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b)) / (aa*bb) if aa and bb else 0.0


class History:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        with self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                    content, content='messages', content_rowid='id',
                    tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS messages_fts_insert
                AFTER INSERT ON messages BEGIN
                    INSERT INTO message_fts(rowid,content) VALUES (new.id,new.content);
                END;
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL REFERENCES messages(id),
                    supersedes_id INTEGER REFERENCES memories(id),
                    is_active INTEGER NOT NULL CHECK (is_active IN (0,1)),
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS active_memory_key
                    ON memories(key) WHERE is_active=1;
                CREATE TABLE IF NOT EXISTS vectors (
                    message_id INTEGER NOT NULL REFERENCES messages(id),
                    model_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    PRIMARY KEY(message_id, model_id)
                );
            """)

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def append(self, role: str, content: str, *, embedder=None) -> int:
        if role not in ("user", "assistant", "system") or not content.strip():
            raise ValueError("Need a valid role and nonempty message")
        vector = None
        if embedder is not None:
            vector = [float(x) for x in embedder.encode(content)]
            if not vector or not all(math.isfinite(x) for x in vector):
                raise ValueError("Invalid embedding returned")
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO messages(role,content,created_at) VALUES(?,?,?)",
                (role, content, _timestamp()))
            message_id = int(cursor.lastrowid)
            if vector is not None:
                self.db.execute(
                    "INSERT INTO vectors VALUES(?,?,?)",
                    (message_id, embedder.model_id, json.dumps(vector)))
        return message_id

    def remember(self, key: str, value: str, *, source_message_id: int) -> int:
        key = key.strip().casefold()
        value = value.strip()
        if not key or not value or len(key) > 100 or len(value) > 4000:
            raise ValueError("Memory key/value must be nonempty and bounded")
        source = self.db.execute(
            "SELECT role FROM messages WHERE id=?", (source_message_id,)).fetchone()
        if source is None or source["role"] != "user":
            raise ValueError("Memory requires a canonical user message")
        with self.db:
            previous = self.db.execute(
                "SELECT id FROM memories WHERE key=? AND is_active=1", (key,)
            ).fetchone()
            old = int(previous["id"]) if previous else None
            if old:
                self.db.execute(
                    "UPDATE memories SET is_active=0 WHERE id=?", (old,))
            inserted = self.db.execute(
                """INSERT INTO memories
                   (key,value,source_message_id,supersedes_id,is_active,created_at)
                   VALUES(?,?,?,?,1,?)""",
                (key, value, source_message_id, old, _timestamp()))
        return int(inserted.lastrowid)

    def memory(self, key: str):
        row = self.db.execute(
            """SELECT * FROM memories WHERE key=? AND is_active=1""",
            (key.strip().casefold(),)).fetchone()
        return dict(row) if row else None

    def memories(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM memories WHERE is_active=1 ORDER BY id DESC LIMIT ?",
            (max(1, limit),)).fetchall()]

    def memory_history(self, key: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM memories WHERE key=? ORDER BY id",
            (key.strip().casefold(),)).fetchall()]

    def messages(self, *, limit: int = 20, before_id: int | None = None) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be >=1")
        if before_id is None:
            rows = self.db.execute(
                "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self.db.execute(
                "SELECT * FROM messages WHERE id < ? ORDER BY id DESC LIMIT ?",
                (before_id, limit))
        return [dict(r) for r in reversed(rows.fetchall())]

    def get(self, message_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM messages").fetchone()[0])

    def search(self, query: str, *, limit: int = 8,
               exclude_ids: set[int] | None = None, embedder=None) -> list[dict]:
        """Lexical FTS plus optional persisted semantic vectors, ranked via RRF.

        Every returned item has exact canonical message id and original text.
        The generated answer must not be treated as a source of truth.
        """
        if limit < 1:
            raise ValueError("limit must be >=1")
        exclude_ids = exclude_ids or set()
        terms = list(dict.fromkeys(_words(query)))[:12]
        lexical = []
        if terms:
            safe = " OR ".join('"' + t.replace('"','') + '"' for t in terms)
            try:
                lexical = self.db.execute(
                    """SELECT m.*, bm25(message_fts) AS bm FROM message_fts
                       JOIN messages m ON m.id=message_fts.rowid
                       WHERE message_fts MATCH ?
                       ORDER BY bm ASC LIMIT ?""",
                    (safe, max(40, limit*4))).fetchall()
            except sqlite3.OperationalError:
                lexical = []
        ranks = {}
        for i, r in enumerate(lexical):
            if r["id"] not in exclude_ids:
                ranks[int(r["id"])] = 1.0 / (60 + i + 1)

        if embedder is not None:
            query_embedding = [float(x) for x in embedder.encode(query)]
            if not all(math.isfinite(x) for x in query_embedding):
                raise ValueError("Nonfinite query embedding")
            scores = []
            for row in self.db.execute(
                "SELECT message_id,embedding_json FROM vectors WHERE model_id=?",
                (embedder.model_id,)):
                mid = int(row["message_id"])
                if mid in exclude_ids:
                    continue
                similarity = _cosine(query_embedding, json.loads(row["embedding_json"]))
                scores.append((similarity, mid))
            scores.sort(reverse=True)
            for rank, (_, mid) in enumerate(scores[:max(40, limit*4)]):
                ranks[mid] = ranks.get(mid, 0) + 1.0 / (60 + rank + 1)

        # If no term matches, preserve a small recency fallback; never invent
        # history. A profile/fact search remains separate from this evidence.
        selected = sorted(ranks, key=lambda mid: (-ranks[mid], -mid))[:limit]
        if not selected and terms:
            q = f"%{terms[0]}%"
            fallback = self.db.execute(
                "SELECT id FROM messages WHERE lower(content) LIKE ? ORDER BY id DESC LIMIT ?",
                (q, limit)).fetchall()
            selected = [int(row["id"]) for row in fallback if row["id"] not in exclude_ids]
        return [self.get(mid) | {"retrieval_score": ranks.get(mid, 0.0)}
                for mid in selected]

    def index_vectors(self, embedder, *, limit: int | None = None) -> int:
        if limit is not None and limit < 1:
            raise ValueError("limit must be >=1")
        sql = """SELECT m.id,m.content FROM messages m
                 LEFT JOIN vectors v ON v.message_id=m.id AND v.model_id=?
                 WHERE v.message_id IS NULL ORDER BY m.id"""
        params = [embedder.model_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.db.execute(sql, params).fetchall()
        count = 0
        for row in rows:
            embedding = [float(x) for x in embedder.encode(row["content"])]
            if not embedding or not all(math.isfinite(x) for x in embedding):
                raise ValueError("Invalid embedding")
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO vectors VALUES(?,?,?)",
                    (row["id"], embedder.model_id, json.dumps(embedding)))
            count += 1
        return count

    def stats(self) -> dict:
        return {
            "messages": self.count(),
            "active_memories": self.db.execute(
                "SELECT count(*) FROM memories WHERE is_active=1").fetchone()[0],
            "memory_revisions": self.db.execute(
                "SELECT count(*) FROM memories").fetchone()[0],
            "vectors": self.db.execute(
                "SELECT count(*) FROM vectors").fetchone()[0],
            "database": str(self.path.resolve()),
        }
