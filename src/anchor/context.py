"""The composition root.

Imports downward only; ONLY api.py, cli.py, eval/* and tests import this. The
deferred imports inside ``build_context`` keep module import time free of DB and
network work (wire-level convention 10) and keep the dependency arrow one-way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anchor.errors import IndexMismatchError
from anchor.settings import Settings, get_settings
from anchor.types import DocumentStore, Embedder, Judge, LLMProvider, VectorStore

if TYPE_CHECKING:  # pragma: no cover
    from anchor.retrieval import Retriever
    from anchor.service import AnswerService


@dataclass
class AppContext:
    settings: Settings
    store: DocumentStore
    vectors: VectorStore
    embedder: Embedder
    llm: LLMProvider
    judge: Judge
    retriever: "Retriever"
    answer_service: "AnswerService"

    def close(self) -> None:
        self.vectors.close()
        self.store.close()


def build_context(settings: Settings | None = None) -> AppContext:
    """Wire the whole system. The one place construction order is decided."""
    from anchor.embed import get_embedder
    from anchor.judge import get_judge
    from anchor.llm import get_llm
    from anchor.retrieval import Retriever
    from anchor.service import AnswerService
    from anchor.store import open_stores

    cfg = settings or get_settings()
    store, vectors = open_stores(cfg)
    embedder = get_embedder(cfg)
    if vectors.count() > 0:
        meta: dict[str, Any] = vectors.meta()
        if meta.get("embedder") != embedder.name or int(meta.get("dim", 0)) != embedder.dim:
            raise IndexMismatchError(
                f"vector index was built with embedder={meta.get('embedder')!r} "
                f"dim={meta.get('dim')} but the configured embedder is {embedder.name!r} "
                f"dim={embedder.dim}. Re-run `anchor ingest --force`."
            )
    llm = get_llm(cfg)
    judge = get_judge(cfg, llm)
    retriever = Retriever(settings=cfg, store=store, vectors=vectors, embedder=embedder)
    answer_service = AnswerService(settings=cfg, retriever=retriever, llm=llm, judge=judge)
    return AppContext(
        settings=cfg,
        store=store,
        vectors=vectors,
        embedder=embedder,
        llm=llm,
        judge=judge,
        retriever=retriever,
        answer_service=answer_service,
    )
