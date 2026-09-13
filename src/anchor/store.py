"""Persistence: a SQLite document/chunk store with FTS5 BM25, and a numpy vector store.

Two storage concerns, one module, because they are ingested and queried as a pair:
every chunk row in SQLite has exactly one float32 row in the matrix, keyed by the
same content-addressed ``chunk_id``.

Design notes that matter elsewhere in the system:

* SQLite's ``bm25()`` returns a NEGATIVE, smaller-is-better score. Every score in
  anchor sorts descending, so :meth:`SqliteStore.search_bm25` returns ``-bm25(...)``
  and is therefore positive-better like every other score in the pipeline.
* Vectors are float32 and L2-normalised by the embedder, so cosine similarity is a
  plain dot product and search is one matmul.
* Ordering is always ``(score DESC, chunk_id ASC)``. The chunk_id tie-break is what
  makes an ablation run byte-reproducible.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from anchor.errors import IndexMismatchError, StoreError
from anchor.settings import Settings
from anchor.types import Chunk, Document, DocumentStore, VectorStore

__all__ = ["SCHEMA_SQL", "SqliteStore", "NumpyVectorStore", "open_stores"]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    path        TEXT NOT NULL DEFAULT '',
    title       TEXT NOT NULL DEFAULT '',
    sha256      TEXT NOT NULL DEFAULT '',
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    ordinal      INTEGER NOT NULL DEFAULT 0,
    heading_path TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL DEFAULT '',
    char_start   INTEGER NOT NULL DEFAULT 0,
    char_end     INTEGER NOT NULL DEFAULT 0,
    n_tokens     INTEGER NOT NULL DEFAULT 0,
    title        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_ordinal ON chunks (doc_id, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    text,
    heading_path,
    title,
    content='chunks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text, heading_path, title)
    VALUES (new.rowid, new.text, new.heading_path, new.title);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text, heading_path, title)
    VALUES ('delete', old.rowid, old.text, old.heading_path, old.title);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text, heading_path, title)
    VALUES ('delete', old.rowid, old.text, old.heading_path, old.title);
    INSERT INTO chunks_fts (rowid, text, heading_path, title)
    VALUES (new.rowid, new.text, new.heading_path, new.title);
END;
"""


_FTS_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _sanitize_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    FTS5 has a query grammar (``AND``, ``OR``, ``NOT``, ``NEAR``, ``*``, ``:``,
    quotes, parentheses) and a user question is not written in it. Rather than
    escape, we discard: split on non-alphanumerics, drop 1-char noise, quote every
    surviving token so it cannot be read as an operator, and OR them together.
    Returns '' when nothing survives, which the caller turns into no hits.
    """
    tokens = [t for t in _FTS_SPLIT.split(query.lower()) if len(t) >= 2]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


class SqliteStore:
    """A :class:`anchor.types.DocumentStore` backed by SQLite + FTS5.

    ``check_same_thread=False`` because FastAPI serves requests on a thread pool;
    every method therefore takes a re-entrant lock, and writes are batched with
    ``executemany`` inside a single transaction.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        if db_path != ":memory:":
            Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - unopenable path
            raise StoreError(f"could not open sqlite database at {db_path!r}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            # REPLACE only fires DELETE triggers when recursive triggers are on.
            # Without this the FTS index silently keeps stale rows on re-ingest.
            self._conn.execute("PRAGMA recursive_triggers=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error as exc:  # pragma: no cover - exotic sqlite build
            raise StoreError(f"could not configure sqlite connection: {exc}") from exc

    # -- schema ---------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            try:
                self._conn.executescript(SCHEMA_SQL)
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(
                    "could not create the anchor schema (is this sqlite build missing FTS5?): "
                    f"{exc}"
                ) from exc

    # -- writes ---------------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(doc_id, path, title, sha256, n_chunks, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc.doc_id, doc.path, doc.title, doc.sha256, doc.n_chunks, doc.ingested_at),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"could not upsert document {doc.doc_id!r}: {exc}") from exc

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.ordinal,
                c.heading_path,
                c.text,
                c.char_start,
                c.char_end,
                c.n_tokens,
                c.title,
            )
            for c in chunks
        ]
        with self._lock:
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO chunks "
                    "(chunk_id, doc_id, ordinal, heading_path, text, char_start, char_end, "
                    " n_tokens, title) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"could not upsert {len(rows)} chunks: {exc}") from exc

    def delete_document(self, doc_id: str) -> int:
        """Delete a document and its chunks. Returns the number of chunks removed."""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                n = int(row["n"]) if row else 0
                self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                self._conn.commit()
                return n
            except sqlite3.Error as exc:
                raise StoreError(f"could not delete document {doc_id!r}: {exc}") from exc

    # -- reads ----------------------------------------------------------------

    @staticmethod
    def _to_document(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            path=row["path"],
            title=row["title"],
            sha256=row["sha256"],
            n_chunks=int(row["n_chunks"]),
            ingested_at=row["ingested_at"],
        )

    @staticmethod
    def _to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            ordinal=int(row["ordinal"]),
            heading_path=row["heading_path"],
            text=row["text"],
            char_start=int(row["char_start"]),
            char_end=int(row["char_end"]),
            n_tokens=int(row["n_tokens"]),
            title=row["title"],
        )

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
            except sqlite3.Error as exc:
                raise StoreError(f"could not read document {doc_id!r}: {exc}") from exc
        return self._to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
            except sqlite3.Error as exc:
                raise StoreError(f"could not list documents: {exc}") from exc
        return [self._to_document(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
                ).fetchone()
            except sqlite3.Error as exc:
                raise StoreError(f"could not read chunk {chunk_id!r}: {exc}") from exc
        return self._to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch many chunks, returned in the order they were ASKED FOR.

        Retrieval hands us a ranked id list; preserving that order here is what
        lets the callers keep rank alignment without a second sort.
        """
        ids = list(chunk_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
                ).fetchall()
            except sqlite3.Error as exc:
                raise StoreError(f"could not read {len(ids)} chunks: {exc}") from exc
        by_id = {r["chunk_id"]: self._to_chunk(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def list_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        with self._lock:
            try:
                if doc_id is None:
                    rows = self._conn.execute(
                        "SELECT * FROM chunks ORDER BY doc_id, ordinal"
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
                    ).fetchall()
            except sqlite3.Error as exc:
                raise StoreError(f"could not list chunks: {exc}") from exc
        return [self._to_chunk(r) for r in rows]

    def neighbours(self, doc_id: str, ordinal: int, window: int = 1) -> list[Chunk]:
        """Chunks in ``[ordinal - window, ordinal + window]``, EXCLUDING the centre."""
        if window <= 0:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM chunks WHERE doc_id = ? AND ordinal BETWEEN ? AND ? "
                    "AND ordinal != ? ORDER BY ordinal",
                    (doc_id, ordinal - window, ordinal + window, ordinal),
                ).fetchall()
            except sqlite3.Error as exc:
                raise StoreError(
                    f"could not read neighbours of {doc_id!r}#{ordinal}: {exc}"
                ) from exc
        return [self._to_chunk(r) for r in rows]

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        """FTS5 BM25 search. Scores are NEGATED so they are positive-better."""
        if k <= 0:
            return []
        match = _sanitize_fts_query(query)
        if not match:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT c.chunk_id AS chunk_id, -bm25(chunks_fts) AS score "
                    "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
                    "WHERE chunks_fts MATCH ? "
                    "ORDER BY score DESC, chunk_id LIMIT ?",
                    (match, k),
                ).fetchall()
            except sqlite3.Error as exc:
                raise StoreError(f"bm25 search failed for {query!r}: {exc}") from exc
        return [(r["chunk_id"], float(r["score"])) for r in rows]

    def count_documents(self) -> int:
        with self._lock:
            try:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
            except sqlite3.Error as exc:
                raise StoreError(f"could not count documents: {exc}") from exc
        return int(row["n"]) if row else 0

    def count_chunks(self) -> int:
        with self._lock:
            try:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            except sqlite3.Error as exc:
                raise StoreError(f"could not count chunks: {exc}") from exc
        return int(row["n"]) if row else 0

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - double close
                pass


class NumpyVectorStore:
    """An ``(n, dim)`` float32 matrix plus a parallel chunk_id list.

    Deliberately not a vector database. At corpus scale (hundreds of chunks) one
    ``matrix @ query`` is microseconds, has zero dependencies and zero operational
    surface, and — because vectors are L2-normalised — the dot product IS cosine.
    Persistence is ``<path>`` (the raw .npy payload) plus ``<path>.meta.json``
    holding ``{embedder, dim, count, chunk_ids}``; the embedder name and dim are
    what lets :func:`anchor.context.build_context` refuse a stale index instead of
    silently returning nonsense.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._row_of: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._embedder_name: str = ""
        self._dim: int = 0

    # -- paths ----------------------------------------------------------------

    @property
    def _meta_path(self) -> Path:
        return Path(f"{self.path}.meta.json")

    # -- lifecycle ------------------------------------------------------------

    def reset(self, embedder_name: str, dim: int) -> None:
        with self._lock:
            if dim <= 0:
                raise IndexMismatchError(f"vector dimension must be positive, got {dim}")
            self._ids = []
            self._row_of = {}
            self._matrix = np.zeros((0, dim), dtype=np.float32)
            self._embedder_name = embedder_name
            self._dim = dim

    def add(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != len(ids):
            raise IndexMismatchError(
                f"got {len(ids)} chunk ids but {arr.shape[0]} vectors; they must match"
            )
        with self._lock:
            if self._dim == 0:
                self._dim = int(arr.shape[1])
                self._matrix = np.zeros((0, self._dim), dtype=np.float32)
            if int(arr.shape[1]) != self._dim:
                raise IndexMismatchError(
                    f"vector width {arr.shape[1]} does not match the index dimension {self._dim}; "
                    "re-run `anchor ingest --force`"
                )
            fresh_ids: list[str] = []
            fresh_rows: list[np.ndarray] = []
            for chunk_id, vec in zip(ids, arr, strict=True):
                existing = self._row_of.get(chunk_id)
                if existing is not None:
                    # Replace in place: re-ingesting must never duplicate a chunk.
                    self._matrix[existing] = vec
                else:
                    fresh_ids.append(chunk_id)
                    fresh_rows.append(vec)
            if fresh_ids:
                block = np.asarray(fresh_rows, dtype=np.float32).reshape(len(fresh_rows), self._dim)
                self._matrix = np.vstack([self._matrix, block]).astype(np.float32, copy=False)
                for offset, chunk_id in enumerate(fresh_ids):
                    self._row_of[chunk_id] = len(self._ids) + offset
                self._ids.extend(fresh_ids)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self._lock:
            if k <= 0 or not self._ids:
                return []
            query = np.asarray(vector, dtype=np.float32).reshape(-1)
            if query.shape[0] != self._matrix.shape[1]:
                raise IndexMismatchError(
                    f"query vector has dim {query.shape[0]} but the index has dim "
                    f"{self._matrix.shape[1]}; re-run `anchor ingest --force`"
                )
            scores = self._matrix @ query
            n = scores.shape[0]
            take = min(k, n)
            if take < n:
                head = np.argpartition(-scores, take - 1)[:take]
            else:
                head = np.arange(n)
            ranked = sorted(
                (int(i) for i in head), key=lambda i: (-float(scores[i]), self._ids[i])
            )
            return [(self._ids[i], float(scores[i])) for i in ranked[:take]]

    def get(self, chunk_id: str) -> np.ndarray | None:
        with self._lock:
            row = self._row_of.get(chunk_id)
            if row is None:
                return None
            return np.array(self._matrix[row], dtype=np.float32)

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def meta(self) -> dict[str, Any]:
        with self._lock:
            return {"embedder": self._embedder_name, "dim": self._dim, "count": len(self._ids)}

    def persist(self) -> None:
        with self._lock:
            target = Path(self.path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Write through an explicit handle so numpy does not append a
                # second ".npy" when the configured path already carries one.
                with open(target, "wb") as handle:
                    np.save(handle, self._matrix, allow_pickle=False)
                self._meta_path.write_text(
                    json.dumps(
                        {
                            "embedder": self._embedder_name,
                            "dim": self._dim,
                            "count": len(self._ids),
                            "chunk_ids": self._ids,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                raise StoreError(f"could not persist vectors to {self.path!r}: {exc}") from exc

    def load(self) -> None:
        """Load a persisted index. A missing file is NOT an error: a fresh checkout
        has no index until `anchor ingest` runs."""
        with self._lock:
            target = Path(self.path).expanduser()
            if not target.is_file() or not self._meta_path.is_file():
                return
            try:
                with open(target, "rb") as handle:
                    matrix = np.load(handle, allow_pickle=False)
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise StoreError(f"could not load vectors from {self.path!r}: {exc}") from exc
            ids = list(meta.get("chunk_ids", []))
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(ids):
                raise IndexMismatchError(
                    f"vector file {self.path!r} holds {matrix.shape} rows but its metadata lists "
                    f"{len(ids)} chunk ids; re-run `anchor ingest --force`"
                )
            self._matrix = matrix
            self._ids = ids
            self._row_of = {cid: i for i, cid in enumerate(ids)}
            self._dim = int(meta.get("dim", matrix.shape[1] if matrix.ndim == 2 else 0))
            self._embedder_name = str(meta.get("embedder", ""))

    def close(self) -> None:
        """Drop the in-memory matrix. Persisting is an explicit, separate act."""
        with self._lock:
            self._ids = []
            self._row_of = {}
            self._matrix = np.zeros((0, 0), dtype=np.float32)


def open_stores(settings: Settings) -> tuple[DocumentStore, VectorStore]:
    """Open both stores, creating parent directories and the schema as needed."""
    db_path = settings.resolve(settings.paths.db_path)
    vec_path = settings.resolve(settings.paths.vectors_path)
    settings.resolve(settings.paths.data_dir).mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vec_path.parent.mkdir(parents=True, exist_ok=True)

    store = SqliteStore(str(db_path))
    store.initialize()
    vectors = NumpyVectorStore(str(vec_path))
    vectors.load()
    return store, vectors
