"""Fusion, diversity, the abstain gate, packing, and all four retrieval modes.

The most important test in this file is
``test_the_gate_is_scale_free_across_score_distributions``. One threshold pair has to
be valid for BM25 scores (unbounded positives), cosines (in [-1, 1]) and RRF scores
(clustered near 0.016), because the ablation compares those three modes directly. If
the gate needed per-mode tuning, the ablation would be measuring the tuning.
"""

from __future__ import annotations

import numpy as np
import pytest

from anchor.retrieval import (
    Retriever,
    evaluate_gate,
    lexical_rerank,
    mmr,
    pack_context,
    reciprocal_rank_fusion,
    search_dense,
    search_lexical,
)
from anchor.settings import Settings
from anchor.textutil import content_tokens
from anchor.types import Candidate, Chunk, Document

# ---------------------------------------------------------------------------
# a tiny in-memory DocumentStore / VectorStore pair
# ---------------------------------------------------------------------------


class FakeStore:
    """An in-memory :class:`anchor.types.DocumentStore`.

    ``search_bm25`` is a deliberately crude term-overlap scorer. It only has to be
    positive-better and deterministically ordered; the real BM25 is exercised against
    SQLite in ``test_ingest_store.py``. Using a fake here proves the retriever talks
    to the PROTOCOL and not to SQLite.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.docs: dict[str, Document] = {}

    def initialize(self) -> None:
        return None

    def upsert_document(self, doc: Document) -> None:
        self.docs[doc.doc_id] = doc

    def upsert_chunks(self, chunks) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c

    def get_document(self, doc_id: str):
        return self.docs.get(doc_id)

    def list_documents(self) -> list[Document]:
        return sorted(self.docs.values(), key=lambda d: d.doc_id)

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id)

    def get_chunks(self, chunk_ids) -> list[Chunk]:
        return [self.chunks[i] for i in chunk_ids if i in self.chunks]

    def list_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        items = [c for c in self.chunks.values() if doc_id is None or c.doc_id == doc_id]
        return sorted(items, key=lambda c: (c.doc_id, c.ordinal))

    def neighbours(self, doc_id: str, ordinal: int, window: int = 1) -> list[Chunk]:
        if window <= 0:
            return []
        lo, hi = ordinal - window, ordinal + window
        return sorted(
            (
                c
                for c in self.chunks.values()
                if c.doc_id == doc_id and lo <= c.ordinal <= hi and c.ordinal != ordinal
            ),
            key=lambda c: c.ordinal,
        )

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = set(content_tokens(query))
        if not terms:
            return []
        scored = []
        for chunk in self.chunks.values():
            tokens = content_tokens(chunk.text)
            hits = sum(1 for t in tokens if t in terms)
            if hits:
                scored.append((chunk.chunk_id, float(hits)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def delete_document(self, doc_id: str) -> int:
        victims = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        for cid in victims:
            del self.chunks[cid]
        self.docs.pop(doc_id, None)
        return len(victims)

    def count_documents(self) -> int:
        return len(self.docs)

    def count_chunks(self) -> int:
        return len(self.chunks)

    def close(self) -> None:
        return None


class FakeVectors:
    """An in-memory :class:`anchor.types.VectorStore` over a plain dict."""

    def __init__(self, vectors: dict[str, np.ndarray] | None = None, dim: int = 4) -> None:
        self.vectors = dict(vectors or {})
        self.dim = dim
        self.embedder_name = "fake"

    def reset(self, embedder_name: str, dim: int) -> None:
        self.vectors.clear()
        self.embedder_name = embedder_name
        self.dim = dim

    def add(self, chunk_ids, vectors) -> None:
        rows = np.asarray(vectors, dtype=np.float32)
        for chunk_id, vector in zip(chunk_ids, rows, strict=True):
            self.vectors[chunk_id] = vector

    def search(self, vector, k: int) -> list[tuple[str, float]]:
        query = np.asarray(vector, dtype=np.float32)
        scored = [(cid, float(vec @ query)) for cid, vec in self.vectors.items()]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def get(self, chunk_id: str):
        return self.vectors.get(chunk_id)

    def count(self) -> int:
        return len(self.vectors)

    def meta(self) -> dict:
        return {"embedder": self.embedder_name, "dim": self.dim, "count": len(self.vectors)}

    def persist(self) -> None:
        return None

    def load(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeEmbedder:
    """Returns a stored query vector so MMR geometry is exact and readable."""

    def __init__(self, query_vector: np.ndarray, dim: int = 4) -> None:
        self.name = "fake"
        self.dim = dim
        self.query_vector = query_vector

    def embed_documents(self, texts):
        return np.vstack([self.query_vector for _ in texts]).astype(np.float32)

    def embed_query(self, text: str):
        return self.query_vector


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in list(__import__("os").environ) if k.startswith("ANCHOR_")]:
        monkeypatch.delenv(key, raising=False)


def settings(**overrides) -> Settings:
    return Settings.load(config_file="", **overrides)


def make_chunk(chunk_id: str, doc_id: str, text: str, ordinal: int = 0, heading: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=ordinal,
        heading_path=heading,
        text=text,
        char_start=0,
        char_end=len(text),
        n_tokens=len(text.split()),
        title=doc_id.replace("-", " ").title(),
    )


def cand(chunk_id: str, doc_id: str = "d", text: str = "body text", **kwargs) -> Candidate:
    return Candidate(chunk=make_chunk(chunk_id, doc_id, text), **kwargs)


def unit(values: list[float]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# reciprocal rank fusion
# ---------------------------------------------------------------------------


def test_rrf_matches_a_hand_computed_example() -> None:
    a, b, c, d = (cand(x) for x in "abcd")
    runs = [
        [a, b, c],
        [Candidate(chunk=b.chunk), Candidate(chunk=a.chunk), Candidate(chunk=d.chunk)],
        [Candidate(chunk=c.chunk), Candidate(chunk=a.chunk), Candidate(chunk=b.chunk)],
    ]
    fused = reciprocal_rank_fusion(runs, rrf_k=60)
    scores = {x.chunk_id: x.rrf_score for x in fused}

    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 63)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [x.chunk_id for x in fused] == ["a", "b", "c", "d"]
    assert [x.final_rank for x in fused] == [1, 2, 3, 4]


def test_rrf_keeps_each_leg_rank_and_score() -> None:
    lexical = [cand("x", bm25_rank=1, bm25_score=9.5)]
    dense = [cand("x", dense_rank=3, dense_score=0.71)]
    fused = reciprocal_rank_fusion([lexical, dense], rrf_k=60)
    assert len(fused) == 1
    merged = fused[0]
    assert merged.bm25_rank == 1 and merged.bm25_score == pytest.approx(9.5)
    assert merged.dense_rank == 3 and merged.dense_score == pytest.approx(0.71)
    assert merged.rrf_score == pytest.approx(1 / 61 + 1 / 61)
    # Candidate.score falls back to the fused score when nothing reranked it.
    assert merged.score == pytest.approx(merged.rrf_score)


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------


def test_mmr_prefers_a_second_document_over_a_near_duplicate() -> None:
    query = unit([1, 1, 1, 1])
    vectors = {
        "a1": unit([1, 1, 0, 0.05]),  # doc a
        "a2": unit([1, 1, 0.1, 0]),  # doc a, near-duplicate of a1
        "b1": unit([0, 0, 1, 1]),  # doc b, orthogonal to a1/a2
    }
    # Plain relevance would rank a2, a1, b1 — i.e. two near-identical doc-a chunks.
    relevance = sorted(vectors, key=lambda cid: -float(vectors[cid] @ query))
    assert relevance[:2] == ["a2", "a1"]

    candidates = [
        cand("a2", "a", rrf_score=0.030),
        cand("a1", "a", rrf_score=0.029),
        cand("b1", "b", rrf_score=0.028),
    ]
    ordered = mmr(candidates, query, vectors, lambda_=0.5, top_k=2)

    picks = [c for c in ordered if c.selected_by_mmr]
    assert [c.chunk_id for c in picks] == ["a2", "b1"]
    assert [c.chunk.doc_id for c in picks] == ["a", "b"]
    # The unselected tail is kept so recall@10 is still measurable.
    assert [c.chunk_id for c in ordered] == ["a2", "b1", "a1"]
    assert [c.final_rank for c in ordered] == [1, 2, 3]


def test_mmr_falls_back_to_fused_score_when_a_chunk_has_no_vector() -> None:
    candidates = [cand("a", rrf_score=0.03), cand("b", rrf_score=0.01)]
    ordered = mmr(candidates, np.zeros(0, dtype=np.float32), {}, lambda_=0.7, top_k=2)
    assert [c.chunk_id for c in ordered] == ["a", "b"]
    assert all(c.selected_by_mmr for c in ordered)


# ---------------------------------------------------------------------------
# lexical rerank
# ---------------------------------------------------------------------------


def test_lexical_rerank_promotes_the_textually_closest_chunk() -> None:
    candidates = [
        cand("far", "d", "an unrelated paragraph about terminating tls", rrf_score=0.03),
        cand("near", "d", "index snapshots are retained for 90 days", rrf_score=0.01),
    ]
    ordered = lexical_rerank("how long are index snapshots retained", candidates, top_k=5)
    assert [c.chunk_id for c in ordered] == ["near", "far"]
    assert ordered[0].rerank_score is not None
    assert ordered[0].score == pytest.approx(ordered[0].rerank_score)
    assert [c.final_rank for c in ordered] == [1, 2]


# ---------------------------------------------------------------------------
# the abstain gate
# ---------------------------------------------------------------------------

GATE_QUERY = "how long are index snapshots retained before deletion"
GATE_TEXT = "Index snapshots are retained for 90 days before deletion in every region."


def test_gate_fails_with_no_candidates() -> None:
    decision = evaluate_gate(GATE_QUERY, [], settings())
    assert decision.passed is False
    assert decision.reason == "no_candidates"
    assert decision.thresholds["min_lexical_overlap"] == pytest.approx(0.34)
    assert decision.near_misses == []


def test_gate_fails_on_low_lexical_overlap() -> None:
    candidates = [
        cand("c1", "d", "The cafeteria serves lunch between eleven and two.", bm25_score=9.0),
        cand("c2", "d", "Parking permits are issued quarterly by facilities.", bm25_score=1.0),
    ]
    decision = evaluate_gate(GATE_QUERY, candidates, settings())
    assert decision.passed is False
    assert decision.reason == "low_lexical_overlap"
    assert decision.lexical_overlap < 0.34
    # A refusal must still be debuggable from the trace alone.
    assert decision.near_misses == ["c1", "c2"]
    assert set(decision.thresholds) == {"min_lexical_overlap"}


def test_top_share_is_recorded_but_is_no_longer_a_gate_branch() -> None:
    """`top_share` is a debugging feature now, not a decision.

    It used to be a second gate branch keyed on `gate.min_top_share`. The branch
    was removed after measurement: under RRF the top-5 scores are
    1/(k+1)..1/(k+5) and therefore near-equal by construction, so the feature sat
    at ~0.205 for dense and hybrid and fired on ZERO of the 43 golden cases in
    all four modes. Five equally-scored candidates put top_share at exactly 1/5
    — and the gate passes anyway, because lexical overlap is the only thing it
    decides on.
    """
    candidates = [cand(f"c{i}", "d", GATE_TEXT, bm25_score=4.0) for i in range(5)]
    decision = evaluate_gate(GATE_QUERY, candidates, settings())
    assert decision.top_share == pytest.approx(0.2)
    assert decision.lexical_overlap >= 0.34
    assert decision.passed is True
    assert decision.reason == "passed"


def test_gate_passes_on_a_good_candidate_set() -> None:
    candidates = [
        cand("c1", "d", GATE_TEXT, bm25_score=12.0),
        cand("c2", "d", "Unrelated prose about parking permits.", bm25_score=1.0),
    ]
    decision = evaluate_gate(GATE_QUERY, candidates, settings())
    assert decision.passed is True
    assert decision.reason == "passed"
    assert decision.lexical_overlap >= 0.34


def test_gate_can_be_disabled_but_still_reports_no_candidates() -> None:
    off = settings(gate={"enabled": False})
    weak = [cand("c1", "d", "Entirely unrelated prose.", bm25_score=1.0)]
    assert evaluate_gate(GATE_QUERY, weak, off).passed is True
    assert evaluate_gate(GATE_QUERY, weak, off).reason == "gate_disabled"
    assert evaluate_gate(GATE_QUERY, [], off).passed is False
    assert evaluate_gate(GATE_QUERY, [], off).reason == "no_candidates"


def test_the_gate_is_scale_free_across_score_distributions() -> None:
    """One threshold must behave identically on bm25-scale and rrf-scale scores.

    This is the property the whole ablation rests on: bm25, dense, hybrid and
    hybrid_rerank are compared against each other under the SAME gate, so the gate
    must key on a ratio, not on a magnitude.
    """
    texts = [
        GATE_TEXT,
        "Retention applies to every region without exception.",
        "The gateway terminates TLS for each tenant.",
        "Parking permits are issued quarterly.",
        "The cafeteria serves lunch daily.",
    ]
    shape = [12.4, 8.1, 6.2, 3.1, 1.2]
    factor = 0.0026  # bm25 magnitudes -> rrf magnitudes, proportions preserved

    bm25_scale = [
        cand(f"c{i}", "d", text, bm25_score=score)
        for i, (text, score) in enumerate(zip(texts, shape, strict=True))
    ]
    rrf_scale = [
        cand(f"c{i}", "d", text, rrf_score=score * factor)
        for i, (text, score) in enumerate(zip(texts, shape, strict=True))
    ]
    assert bm25_scale[0].score > 100 * rrf_scale[0].score, "the two scales must really differ"

    cfg = settings()
    lexical_decision = evaluate_gate(GATE_QUERY, bm25_scale, cfg)
    fused_decision = evaluate_gate(GATE_QUERY, rrf_scale, cfg)

    assert lexical_decision.passed == fused_decision.passed is True
    assert lexical_decision.reason == fused_decision.reason
    assert lexical_decision.top_share == pytest.approx(fused_decision.top_share, rel=1e-6)
    assert lexical_decision.lexical_overlap == pytest.approx(fused_decision.lexical_overlap)

    # And they must also agree when the SAME raised threshold rejects both.
    strict = settings(gate={"min_lexical_overlap": 0.99})
    assert (
        evaluate_gate(GATE_QUERY, bm25_scale, strict).reason
        == evaluate_gate(GATE_QUERY, rrf_scale, strict).reason
        == "low_lexical_overlap"
    )


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------


def test_pack_context_assigns_sequential_markers_and_respects_the_budget() -> None:
    long_text = " ".join(f"Fact number {i} is recorded here." for i in range(40))
    chunks = [make_chunk(f"c{i}", "d", long_text, ordinal=i) for i in range(5)]
    # Make each chunk textually distinct so none is dropped as a duplicate.
    for i, c in enumerate(chunks):
        c.text = f"Chunk {i} unique opener. " + " ".join(
            f"Detail {i}-{j} is recorded here." for j in range(40)
        )
    store = FakeStore(chunks)
    cfg = settings(retrieval={"context_token_budget": 200, "top_k": 5})

    blocks = pack_context(store, [Candidate(chunk=c) for c in chunks], cfg)
    assert blocks
    assert [b.marker for b in blocks] == [f"S{i + 1}" for i in range(len(blocks))]
    assert sum(b.n_tokens for b in blocks) <= 200
    assert len(blocks) < len(chunks), "the budget must actually bite"
    for block in blocks:
        assert block.n_tokens > 0
        assert block.chunk_id and block.doc_id


def test_pack_context_drops_near_duplicates() -> None:
    # What an overlapping chunk actually looks like: the same sentences, one short
    # addition. Packing both would spend budget to buy no extra evidence.
    body = (
        "Index snapshots are retained for 90 days before deletion in every region "
        "without exception."
    )
    chunks = [
        make_chunk("c1", "d1", body, ordinal=0),
        make_chunk("c2", "d2", body + " Confirmed.", ordinal=0),
        make_chunk("c3", "d3", "The gateway terminates TLS for every tenant request.", ordinal=0),
    ]
    store = FakeStore(chunks)
    blocks = pack_context(store, [Candidate(chunk=c) for c in chunks], settings())
    assert [b.chunk_id for b in blocks] == ["c1", "c3"]
    assert [b.marker for b in blocks] == ["S1", "S2"]


def test_pack_context_never_exceeds_top_k_blocks() -> None:
    chunks = [
        make_chunk(f"c{i}", "d", f"Distinct body number {i} about topic {i}.", i)
        for i in range(9)
    ]
    store = FakeStore(chunks)
    cfg = settings(retrieval={"top_k": 3, "context_token_budget": 5000})
    blocks = pack_context(store, [Candidate(chunk=c) for c in chunks], cfg)
    assert len(blocks) == 3


def test_pack_context_appends_neighbours_when_configured() -> None:
    chunks = [
        make_chunk(f"c{i}", "d", f"Paragraph {i} concerns subject {i} alone.", i)
        for i in range(3)
    ]
    store = FakeStore(chunks)
    cfg = settings(retrieval={"neighbour_window": 1, "top_k": 5, "context_token_budget": 5000})
    blocks = pack_context(store, [Candidate(chunk=chunks[1])], cfg)
    assert [b.chunk_id for b in blocks] == ["c1", "c0", "c2"]


# ---------------------------------------------------------------------------
# the legs and all four modes
# ---------------------------------------------------------------------------

CORPUS = [
    ("r1", "retention", "Index snapshots are retained for 90 days before deletion.", 0),
    ("r2", "retention", "The retention window applies to every region without exception.", 1),
    ("g1", "gateway", "The Meridian gateway terminates TLS and routes tenant requests.", 0),
    ("g2", "gateway", "A misrouted request is retried once against the home region.", 1),
    ("q1", "quotas", "Every tenant starts with 600 requests per minute on /v1/generate.", 0),
]
QUESTION = "how long are index snapshots retained before deletion"


def build_retriever(mode: str) -> Retriever:
    chunks = [make_chunk(cid, doc, text, ordinal) for cid, doc, text, ordinal in CORPUS]
    store = FakeStore(chunks)
    from anchor.embed import HashingEmbedder

    embedder = HashingEmbedder(dim=128, seed=1337)
    vectors = FakeVectors(dim=128)
    matrix = embedder.embed_documents([c.text for c in chunks])
    vectors.reset("hashing", 128)
    vectors.add([c.chunk_id for c in chunks], matrix)
    cfg = settings(retrieval={"mode": mode, "top_k": 3, "context_token_budget": 400})
    return Retriever(settings=cfg, store=store, vectors=vectors, embedder=embedder)


def test_search_lexical_sets_ranks_and_is_empty_on_an_empty_query() -> None:
    retriever = build_retriever("bm25")
    hits = search_lexical(retriever.store, QUESTION, 10)
    assert hits and hits[0].chunk_id == "r1"
    assert [h.bm25_rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.bm25_score is not None for h in hits)
    assert search_lexical(retriever.store, "   ", 10) == []
    assert search_lexical(retriever.store, QUESTION, 0) == []


def test_search_dense_returns_empty_on_an_empty_index() -> None:
    retriever = build_retriever("dense")
    empty = FakeVectors(dim=128)
    assert search_dense(retriever.store, empty, retriever.embedder, QUESTION, 5) == []


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_retrieval_mode_produces_a_usable_result(mode: str) -> None:
    retriever = build_retriever(mode)
    result = retriever.retrieve(QUESTION)

    assert result.query == QUESTION
    assert result.candidates, f"{mode} returned no candidates"
    assert result.candidates[0].chunk_id == "r1", f"{mode} put the wrong chunk first"
    assert all(c.final_rank is not None for c in result.candidates)
    assert [c.final_rank for c in result.candidates] == list(
        range(1, len(result.candidates) + 1)
    )
    # Every mode must yield a non-zero, comparable score off Candidate.score.
    assert result.candidates[0].score > 0.0
    assert result.gate.passed is True, f"{mode} gate refused: {result.gate.reason}"
    assert result.blocks, f"{mode} packed no context"
    assert result.blocks[0].chunk_id == "r1"
    assert [b.marker for b in result.blocks] == [f"S{i + 1}" for i in range(len(result.blocks))]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_is_byte_reproducible(mode: str) -> None:
    first = build_retriever(mode).retrieve(QUESTION)
    second = build_retriever(mode).retrieve(QUESTION)
    assert first.model_dump() == second.model_dump()


def test_an_off_topic_question_is_refused_before_packing() -> None:
    retriever = build_retriever("hybrid")
    result = retriever.retrieve("what is the Helios queue rotation in the Zurich region")
    assert result.gate.passed is False
    assert result.gate.reason == "low_lexical_overlap"
    assert result.blocks == [], "a failed gate must never produce context"
