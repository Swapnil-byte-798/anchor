"""Retrieval: BM25, dense, RRF fusion, MMR, lexical rerank, the abstain gate, packing.

Four modes share one code path so that the ablation compares strategies, not
implementations:

* ``bm25``          — the declared BASELINE. FTS5 only.
* ``dense``         — hashing-embedder cosine only.
* ``hybrid``        — reciprocal rank fusion of both legs, then MMR for diversity.
* ``hybrid_rerank`` — hybrid, then a lexical rerank over the head of the list.

Every mode produces candidates whose ``Candidate.score`` is meaningful, and every
mode is gated by the SAME two thresholds. See :func:`evaluate_gate` for why that is
possible and why it is the property the whole ablation rests on.
"""

from __future__ import annotations

import numpy as np

from anchor.errors import RetrievalError
from anchor.obs import TraceRecorder
from anchor.settings import Settings
from anchor.textutil import count_tokens, overlap_score, truncate_to_tokens
from anchor.types import (
    Candidate,
    Chunk,
    ContextBlock,
    DocumentStore,
    Embedder,
    GateDecision,
    RetrievalResult,
    VectorStore,
)

__all__ = [
    "search_lexical",
    "search_dense",
    "reciprocal_rank_fusion",
    "mmr",
    "lexical_rerank",
    "evaluate_gate",
    "pack_context",
    "Retriever",
]

#: A trailing block is truncated rather than dropped only if at least this many
#: tokens of it survive; below that the fragment is noise, not evidence.
_MIN_TRUNCATED_BLOCK_TOKENS = 60

#: Two chunks whose lexical overlap exceeds this are treated as the same evidence.
_NEAR_DUPLICATE_OVERLAP = 0.9


def _sort_key(candidate: Candidate) -> tuple[float, str]:
    """Descending score, ascending chunk_id. The tie-break is what makes a run
    byte-reproducible across machines."""
    return (-candidate.score, candidate.chunk_id)


def search_lexical(store: DocumentStore, query: str, k: int) -> list[Candidate]:
    """BM25 over FTS5. Scores arrive positive-better from the store."""
    if not query.strip() or k <= 0:
        return []
    hits = store.search_bm25(query, k)
    if not hits:
        return []
    chunks = {c.chunk_id: c for c in store.get_chunks([cid for cid, _ in hits])}
    out: list[Candidate] = []
    rank = 0
    for chunk_id, score in hits:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        rank += 1
        out.append(Candidate(chunk=chunk, bm25_rank=rank, bm25_score=float(score)))
    return out


def search_dense(
    store: DocumentStore, vectors: VectorStore, embedder: Embedder, query: str, k: int
) -> list[Candidate]:
    """Cosine search over the vector matrix.

    An empty index is a legitimate state (fresh checkout, retrieval-only ablation
    on the bm25 leg), so this returns ``[]`` rather than raising.
    """
    if not query.strip() or k <= 0 or vectors.count() == 0:
        return []
    vector = embedder.embed_query(query)
    hits = vectors.search(vector, k)
    if not hits:
        return []
    chunks = {c.chunk_id: c for c in store.get_chunks([cid for cid, _ in hits])}
    out: list[Candidate] = []
    rank = 0
    for chunk_id, score in hits:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            # A vector whose chunk was re-ingested under a new id. Skip, don't fail.
            continue
        rank += 1
        out.append(Candidate(chunk=chunk, dense_rank=rank, dense_score=float(score)))
    return out


def reciprocal_rank_fusion(runs: list[list[Candidate]], rrf_k: int = 60) -> list[Candidate]:
    """Fuse ranked runs by ``sum(1 / (rrf_k + rank))``.

    RRF is used rather than score interpolation because BM25 scores and cosine
    scores live on incomparable scales; ranks do not. Each leg's own rank and score
    are carried through onto the merged candidate so the trace can still explain
    where a chunk came from.
    """
    merged: dict[str, Candidate] = {}
    for run in runs:
        for rank, candidate in enumerate(run, start=1):
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                existing = Candidate(chunk=candidate.chunk)
                merged[candidate.chunk_id] = existing
            if candidate.bm25_rank is not None:
                existing.bm25_rank = candidate.bm25_rank
                existing.bm25_score = candidate.bm25_score
            if candidate.dense_rank is not None:
                existing.dense_rank = candidate.dense_rank
                existing.dense_score = candidate.dense_score
            existing.rrf_score += 1.0 / (rrf_k + rank)
    fused = sorted(merged.values(), key=lambda c: (-c.rrf_score, c.chunk_id))
    for position, candidate in enumerate(fused, start=1):
        candidate.final_rank = position
    return fused


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def mmr(
    candidates: list[Candidate],
    query_vector: np.ndarray,
    chunk_vectors: dict[str, np.ndarray],
    lambda_: float,
    top_k: int,
) -> list[Candidate]:
    """Maximal Marginal Relevance over the fused list.

    Selects ``top_k`` picks that trade relevance against redundancy, marks them
    ``selected_by_mmr``, and returns the FULL list with the picks first followed by
    the unselected candidates in their incoming order. Keeping the tail matters:
    recall@10 is a headline ablation metric and must not be truncated to ``top_k``.
    """
    if not candidates or top_k <= 0:
        return candidates
    max_rrf = max((c.rrf_score for c in candidates), default=0.0)

    def relevance(candidate: Candidate) -> float:
        vector = chunk_vectors.get(candidate.chunk_id)
        if vector is not None and query_vector.size:
            return _cosine(query_vector, vector)
        # No vector for this chunk (bm25-only hit): fall back to the fused score,
        # rescaled into [0, 1] so it is comparable with a cosine.
        return (candidate.rrf_score / max_rrf) if max_rrf > 0 else 0.0

    remaining = list(candidates)
    picked: list[Candidate] = []
    while remaining and len(picked) < top_k:
        best: Candidate | None = None
        best_key: tuple[float, str] | None = None
        for candidate in remaining:
            vector = chunk_vectors.get(candidate.chunk_id)
            redundancy = 0.0
            if vector is not None:
                for chosen in picked:
                    other = chunk_vectors.get(chosen.chunk_id)
                    if other is not None:
                        redundancy = max(redundancy, _cosine(vector, other))
            value = lambda_ * relevance(candidate) - (1.0 - lambda_) * redundancy
            key = (-value, candidate.chunk_id)
            if best_key is None or key < best_key:
                best, best_key = candidate, key
        if best is None:  # pragma: no cover - unreachable while remaining is non-empty
            break
        best.selected_by_mmr = True
        picked.append(best)
        remaining.remove(best)

    ordered = picked + remaining
    for position, candidate in enumerate(ordered, start=1):
        candidate.final_rank = position
    return ordered


def lexical_rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """Cheap, deterministic cross-encoder stand-in: weighted lexical overlap.

    Only the head window is rescored — a reranker is expensive by nature and is
    always applied to a candidate window, so the shape here is the shape a real
    cross-encoder would have. The tail keeps its fused order.
    """
    if not candidates:
        return candidates
    window = min(len(candidates), max(top_k * 4, 20))
    head = candidates[:window]
    tail = candidates[window:]
    for candidate in head:
        chunk = candidate.chunk
        heading = f"{chunk.heading_path} {chunk.title}".strip()
        candidate.rerank_score = 0.7 * overlap_score(query, chunk.text) + 0.3 * overlap_score(
            query, heading
        )
    head.sort(key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
    ordered = head + tail
    for position, candidate in enumerate(ordered, start=1):
        candidate.final_rank = position
    return ordered


def evaluate_gate(query: str, candidates: list[Candidate], settings: Settings) -> GateDecision:
    """The abstain gate. ONE feature, and it is a RATIO — that is the whole point.

    ``lexical_overlap`` is the best fraction of the question's content tokens found
    in any of the top-5 chunks. It has no units, so it does not move when the
    underlying score scale changes: BM25 scores are unbounded positives, cosines
    live in [-1, 1] and RRF scores cluster near 0.016, and a gate keyed on raw
    magnitudes would need a different threshold per mode — an ablation comparing
    four separately-tuned gates measures the tuning, not the retrieval.

    ``top_share`` (the top-1 score over the sum of the top-5 scores) is still
    COMPUTED and recorded, because it is useful when reading a trace, but it is no
    longer a gate branch. It was one, keyed on ``gate.min_top_share``, and the
    branch was removed after measurement: under RRF the top-5 scores are
    1/(k+1)..1/(k+5) and therefore near-equal by construction, so top_share is
    pinned around 0.205 for dense and hybrid and carries no information; across
    all 43 golden cases in all four modes the branch fired exactly zero times.
    A feature that has never made a decision is not a guardrail.

    ``near_misses`` and ``thresholds`` are always populated so that a refusal is
    debuggable from the trace alone, and so that :mod:`anchor.eval.report` can sweep
    the threshold after the fact instead of asserting an operating point.
    """
    top = sorted(candidates, key=_sort_key)[:5]
    near_misses = [c.chunk_id for c in top[:3]]
    thresholds = {
        "min_lexical_overlap": float(settings.gate.min_lexical_overlap),
    }

    if not candidates:
        return GateDecision(
            passed=False,
            reason="no_candidates",
            lexical_overlap=0.0,
            top_share=0.0,
            thresholds=thresholds,
            near_misses=near_misses,
        )

    lexical_overlap = max(overlap_score(query, c.chunk.text) for c in top)
    scores = [max(0.0, c.score) for c in top]
    total = sum(scores)
    top_share = (scores[0] / total) if total > 0 else 0.0

    if not settings.gate.enabled:
        return GateDecision(
            passed=True,
            reason="gate_disabled",
            lexical_overlap=lexical_overlap,
            top_share=top_share,
            thresholds=thresholds,
            near_misses=near_misses,
        )

    if lexical_overlap < settings.gate.min_lexical_overlap:
        reason, passed = "low_lexical_overlap", False
    else:
        reason, passed = "passed", True

    return GateDecision(
        passed=passed,
        reason=reason,
        lexical_overlap=lexical_overlap,
        top_share=top_share,
        thresholds=thresholds,
        near_misses=near_misses,
    )


def _block_from_chunk(chunk: Chunk, marker: str, text: str | None = None) -> ContextBlock:
    body = chunk.text if text is None else text
    return ContextBlock(
        marker=marker,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        heading_path=chunk.heading_path,
        text=body,
        n_tokens=count_tokens(body),
    )


def pack_context(
    store: DocumentStore,
    candidates: list[Candidate],
    settings: Settings,
    max_blocks: int | None = None,
) -> list[ContextBlock]:
    """Turn ranked candidates into labelled ``[S1]..[Sn]`` blocks under a token budget.

    Near-duplicates are dropped because two copies of the same paragraph buy no
    evidence and cost budget. The last block that does not fit whole is TRUNCATED
    rather than dropped when a usable amount of it survives — dropping it would
    silently lose the citation target the generator was about to use.

    ``max_blocks`` is the effective block cap; ``None`` means ``retrieval.top_k``
    from config. :meth:`Retriever.retrieve` passes the PER-REQUEST value so that a
    caller's ``top_k`` actually controls how many blocks come back — reading the
    config value here instead is what made the API/CLI/tool ``top_k`` a no-op.
    """
    budget = int(settings.retrieval.context_token_budget)
    cap = settings.retrieval.top_k if max_blocks is None else max_blocks
    max_blocks = max(int(cap), 1)
    window = int(settings.retrieval.neighbour_window)
    blocks: list[ContextBlock] = []
    used = 0
    seen_ids: set[str] = set()

    def try_add(chunk: Chunk) -> bool:
        nonlocal used
        if len(blocks) >= max_blocks or chunk.chunk_id in seen_ids:
            return False
        for existing in blocks:
            if overlap_score(chunk.text, existing.text) > _NEAR_DUPLICATE_OVERLAP:
                seen_ids.add(chunk.chunk_id)
                return False
        remaining = budget - used
        if remaining <= 0:
            return False
        tokens = count_tokens(chunk.text)
        if tokens <= remaining:
            body = chunk.text
        elif remaining >= _MIN_TRUNCATED_BLOCK_TOKENS:
            body = truncate_to_tokens(chunk.text, remaining)
        else:
            return False
        if not body.strip():
            return False
        marker = f"S{len(blocks) + 1}"
        block = _block_from_chunk(chunk, marker, body)
        blocks.append(block)
        seen_ids.add(chunk.chunk_id)
        used += block.n_tokens
        return True

    for candidate in candidates:
        if len(blocks) >= max_blocks:
            break
        added = try_add(candidate.chunk)
        if added and window > 0:
            chunk = candidate.chunk
            for neighbour in store.neighbours(chunk.doc_id, chunk.ordinal, window):
                if len(blocks) >= max_blocks:
                    break
                try_add(neighbour)
    return blocks


class Retriever:
    """The one retrieval entry point. ``settings.retrieval.mode`` selects the legs."""

    def __init__(
        self, *, settings: Settings, store: DocumentStore, vectors: VectorStore, embedder: Embedder
    ) -> None:
        self.settings = settings
        self.store = store
        self.vectors = vectors
        self.embedder = embedder

    # -- legs -----------------------------------------------------------------

    def _lexical(self, query: str, recorder: TraceRecorder | None) -> list[Candidate]:
        if recorder is None:
            return search_lexical(self.store, query, self.settings.retrieval.k_lexical)
        with recorder.stage("retrieve.lexical"):
            return search_lexical(self.store, query, self.settings.retrieval.k_lexical)

    def _dense(self, query: str, recorder: TraceRecorder | None) -> list[Candidate]:
        if recorder is None:
            return search_dense(
                self.store, self.vectors, self.embedder, query, self.settings.retrieval.k_dense
            )
        with recorder.stage("retrieve.dense"):
            return search_dense(
                self.store, self.vectors, self.embedder, query, self.settings.retrieval.k_dense
            )

    def _rank_single_leg(self, candidates: list[Candidate]) -> list[Candidate]:
        ordered = sorted(candidates, key=_sort_key)
        for position, candidate in enumerate(ordered, start=1):
            candidate.final_rank = position
        return ordered

    def _apply_mmr(self, query: str, fused: list[Candidate], top_k: int) -> list[Candidate]:
        chunk_vectors: dict[str, np.ndarray] = {}
        for candidate in fused:
            vector = self.vectors.get(candidate.chunk_id)
            if vector is not None:
                chunk_vectors[candidate.chunk_id] = vector
        query_vector = (
            self.embedder.embed_query(query)
            if chunk_vectors
            else np.zeros(0, dtype=np.float32)
        )
        return mmr(fused, query_vector, chunk_vectors, self.settings.retrieval.mmr_lambda, top_k)

    # -- entry point ----------------------------------------------------------

    def retrieve(
        self, query: str, recorder: TraceRecorder | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        mode = self.settings.retrieval.mode
        # A per-request value below 1 is not a retrieval strategy, it is a bad
        # argument: 0 has always meant "use the configured default" at the CLI, and
        # a negative used to slip through to disable MMR and shrink the candidate
        # window. Both now mean "default" here; the rejection proper belongs on the
        # request model (QueryRequest.top_k) and the CLI flag.
        requested = int(top_k) if top_k is not None else 0
        k = requested if requested >= 1 else int(self.settings.retrieval.top_k)

        if mode == "bm25":
            ranked = self._rank_single_leg(self._lexical(query, recorder))
        elif mode == "dense":
            ranked = self._rank_single_leg(self._dense(query, recorder))
        elif mode in ("hybrid", "hybrid_rerank"):
            lexical = self._lexical(query, recorder)
            dense = self._dense(query, recorder)
            runs = [run for run in (lexical, dense) if run]
            if recorder is None:
                ranked = reciprocal_rank_fusion(runs, self.settings.retrieval.rrf_k)
            else:
                with recorder.stage("retrieve.fusion"):
                    ranked = reciprocal_rank_fusion(runs, self.settings.retrieval.rrf_k)
            if self.settings.retrieval.use_mmr and ranked:
                if recorder is None:
                    ranked = self._apply_mmr(query, ranked, k)
                else:
                    with recorder.stage("retrieve.mmr"):
                        ranked = self._apply_mmr(query, ranked, k)
            if mode == "hybrid_rerank" and ranked:
                if recorder is None:
                    ranked = lexical_rerank(query, ranked, k)
                else:
                    with recorder.stage("retrieve.rerank"):
                        ranked = lexical_rerank(query, ranked, k)
        else:  # pragma: no cover - Literal-typed, but fail loudly if widened
            raise RetrievalError(
                f"unknown retrieval mode {mode!r}; expected one of "
                "bm25, dense, hybrid, hybrid_rerank"
            )

        if recorder is not None:
            recorder.record_candidates(ranked)

        if recorder is None:
            gate = evaluate_gate(query, ranked, self.settings)
        else:
            with recorder.stage("retrieve.gate"):
                gate = evaluate_gate(query, ranked, self.settings)
            recorder.record_gate(gate)

        blocks: list[ContextBlock] = []
        if gate.passed:
            if recorder is None:
                blocks = pack_context(self.store, ranked[: k * 4], self.settings, max_blocks=k)
            else:
                with recorder.stage("retrieve.pack"):
                    blocks = pack_context(self.store, ranked[: k * 4], self.settings, max_blocks=k)
                recorder.record_context(blocks)

        return RetrievalResult(query=query, candidates=ranked, gate=gate, blocks=blocks)
