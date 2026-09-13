"""Ingest, chunking, SQLite/FTS5, the vector store and the hashing embedder.

The load-bearing assertion in this file is the char-offset round-trip. Gold labels
are verbatim answer spans resolved against chunk text at eval time, so if
``doc.text[char_start:char_end]`` ever stops being the chunk text, every label in the
golden set becomes unresolvable and the chunking ablation becomes meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from anchor.embed import HashingEmbedder, get_embedder
from anchor.errors import IndexMismatchError, IngestError
from anchor.ingest import chunk_document, ingest_corpus, load_corpus, load_file
from anchor.settings import ChunkingSettings, Settings
from anchor.store import NumpyVectorStore, SqliteStore, open_stores
from anchor.types import Chunk, Document, LoadedDoc

# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ANCHOR_* env var from the developer's shell may reach these tests."""
    for key in [k for k in list(__import__("os").environ) if k.startswith("ANCHOR_")]:
        monkeypatch.delenv(key, raising=False)


def make_settings(tmp_path, **overrides) -> Settings:
    base: dict = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "corpus_dir": str(tmp_path / "corpus"),
            "db_path": str(tmp_path / "data" / "index.db"),
            "vectors_path": str(tmp_path / "data" / "vectors.npy"),
        }
    }
    base.update(overrides)
    return Settings.load(config_file="", **base)


def loaded(text: str, doc_id: str = "doc", title: str = "Doc") -> LoadedDoc:
    return LoadedDoc(doc_id=doc_id, path=f"{doc_id}.md", title=title, text=text, sha256="x" * 64)


def chunk(chunk_id: str, doc_id: str, ordinal: int, text: str, heading: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=ordinal,
        heading_path=heading,
        text=text,
        char_start=0,
        char_end=len(text),
        n_tokens=len(text.split()),
        title=doc_id.title(),
    )


def paragraphs(n: int, word: str = "alpha") -> str:
    """``n`` one-sentence paragraphs of ten words each, so token counts are predictable.

    Each starts with a capital: the shared sentence-boundary rule in ``textutil`` only
    breaks before an uppercase letter or a digit, and the overlap carry is defined in
    whole sentences.
    """
    return "\n\n".join(
        " ".join([f"{word.capitalize()}{i}"] + [f"{word}{i}"] * 9) + "." for i in range(n)
    )


SAMPLE = """\
# Gateway Overview

The Meridian gateway terminates TLS and routes every tenant request to a regional
backend. It is owned by the platform team.

## Routing

Requests to /v1/generate are routed by tenant identifier. A misrouted request is
retried once against the home region before failing.

## Ownership

The platform on-call engineer owns the gateway during an incident.
"""


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def test_chunk_offsets_round_trip_exactly() -> None:
    doc = loaded(SAMPLE)
    chunks = chunk_document(doc, ChunkingSettings())
    assert chunks
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text
        assert c.text == c.text.strip()
        assert c.text


def test_chunks_never_cross_a_heading_boundary() -> None:
    doc = loaded(SAMPLE)
    chunks = chunk_document(doc, ChunkingSettings(min_chunk_tokens=0))
    paths = [c.heading_path for c in chunks]
    assert "Gateway Overview" in paths
    assert "Gateway Overview > Routing" in paths
    assert "Gateway Overview > Ownership" in paths
    routing = next(c for c in chunks if c.heading_path.endswith("Routing"))
    assert "/v1/generate" in routing.text
    assert "on-call engineer" not in routing.text


def test_ordinals_are_zero_based_and_dense() -> None:
    chunks = chunk_document(loaded(SAMPLE), ChunkingSettings(min_chunk_tokens=0))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunking_is_deterministic_for_identical_input() -> None:
    doc = loaded(SAMPLE)
    first = chunk_document(doc, ChunkingSettings())
    second = chunk_document(doc, ChunkingSettings())
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_overlap_zero_versus_sixty_four_changes_the_chunks() -> None:
    doc = loaded("# Section\n\n" + paragraphs(12))
    settings_no_overlap = ChunkingSettings(chunk_tokens=60, chunk_overlap=0, min_chunk_tokens=0)
    settings_overlap = ChunkingSettings(chunk_tokens=60, chunk_overlap=30, min_chunk_tokens=0)

    plain = chunk_document(doc, settings_no_overlap)
    overlapping = chunk_document(doc, settings_overlap)
    assert len(plain) > 1, "the fixture must split, otherwise overlap is untestable"

    # overlap == 0 must be a true no-op: chunks are strictly disjoint.
    for earlier, later in zip(plain, plain[1:], strict=False):
        assert later.char_start >= earlier.char_end

    # overlap > 0 carries trailing sentences forward, so ranges intersect...
    assert any(
        later.char_start < earlier.char_end
        for earlier, later in zip(overlapping, overlapping[1:], strict=False)
    )
    # ...and the resulting chunk ids differ from the non-overlapping run.
    assert {c.chunk_id for c in plain} != {c.chunk_id for c in overlapping}
    for c in overlapping:
        assert doc.text[c.char_start : c.char_end] == c.text


def test_short_trailing_chunk_is_merged_into_its_neighbour() -> None:
    doc = loaded("# Section\n\n" + paragraphs(5))
    cfg = {"chunk_tokens": 30, "chunk_overlap": 0}
    unmerged = chunk_document(doc, ChunkingSettings(min_chunk_tokens=0, **cfg))
    merged = chunk_document(doc, ChunkingSettings(min_chunk_tokens=25, **cfg))
    assert len(unmerged) > len(merged) >= 1
    assert min(c.n_tokens for c in merged) >= 25
    for c in merged:
        assert doc.text[c.char_start : c.char_end] == c.text


def test_a_lone_short_section_is_emitted_rather_than_lost() -> None:
    doc = loaded("# Tiny\n\nThe retention window is 90 days.")
    chunks = chunk_document(doc, ChunkingSettings(min_chunk_tokens=500))
    assert len(chunks) == 1
    assert "90 days" in chunks[0].text
    assert chunks[0].heading_path == "Tiny"


def test_an_oversized_paragraph_is_split_on_sentences() -> None:
    body = " ".join(f"Sentence number {i} carries a distinct fact." for i in range(40))
    doc = loaded(f"# Big\n\n{body}")
    cfg = ChunkingSettings(chunk_tokens=50, chunk_overlap=0, min_chunk_tokens=0)
    chunks = chunk_document(doc, cfg)
    assert len(chunks) > 1
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_corpus_reads_title_doc_id_and_sorts(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "nested").mkdir(parents=True)
    (corpus / "zulu.md").write_text("# Zulu Doc\n\nZulu body.\n", encoding="utf-8")
    (corpus / "alpha.md").write_text("# Alpha Doc\n\nAlpha body.\n", encoding="utf-8")
    (corpus / "nested" / "notes.txt").write_text("plain text body\n", encoding="utf-8")
    (corpus / ".hidden.md").write_text("# Hidden\n\nnope\n", encoding="utf-8")
    (corpus / "metrics.csv").write_text("month,value\n2024-01,1\n", encoding="utf-8")

    docs = load_corpus(str(corpus))
    assert [d.doc_id for d in docs] == ["alpha", "nested-notes", "zulu"]
    assert docs[0].title == "Alpha Doc"
    assert docs[1].title == "Notes"  # no '# ' heading -> title-cased stem
    assert all(d.sha256 for d in docs)


def test_load_corpus_normalises_crlf_but_keeps_fenced_code(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    body = "# T\r\n\r\ntrailing spaces here   \r\n\r\n```\ncode  with  spacing   \n```\r\n"
    (corpus / "t.md").write_text(body, encoding="utf-8", newline="")
    doc = load_file(str(corpus / "t.md"), str(corpus))
    assert "\r" not in doc.text
    assert "trailing spaces here\n" in doc.text
    assert "code  with  spacing   \n" in doc.text


def test_load_corpus_rejects_a_missing_directory(tmp_path) -> None:
    with pytest.raises(IngestError):
        load_corpus(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# SqliteStore
# ---------------------------------------------------------------------------


def test_sqlite_round_trip_documents_and_chunks() -> None:
    store = SqliteStore(":memory:")
    store.initialize()
    store.upsert_document(
        Document(doc_id="d1", path="d1.md", title="D1", sha256="abc", n_chunks=2)
    )
    store.upsert_chunks([chunk("c1", "d1", 0, "first body"), chunk("c2", "d1", 1, "second body")])

    assert store.count_documents() == 1
    assert store.count_chunks() == 2
    assert store.get_document("d1").title == "D1"
    assert store.get_document("missing") is None
    assert store.get_chunk("c2").text == "second body"
    assert [c.chunk_id for c in store.list_chunks("d1")] == ["c1", "c2"]
    # get_chunks preserves the ORDER IT WAS ASKED FOR, which retrieval relies on.
    assert [c.chunk_id for c in store.get_chunks(["c2", "c1"])] == ["c2", "c1"]
    assert store.get_chunks(["nope"]) == []
    store.close()


def test_bm25_scores_are_positive_better_and_obviously_ranked() -> None:
    store = SqliteStore(":memory:")
    store.initialize()
    store.upsert_document(Document(doc_id="d", path="d.md", title="D", sha256="s"))
    store.upsert_chunks(
        [
            chunk("winner", "d", 0, "quota quota quota the tenant quota is enforced per quota"),
            chunk("middle", "d", 1, "the tenant quota is mentioned once here"),
            chunk("loser", "d", 2, "unrelated prose about terminating tls at the edge"),
        ]
    )
    hits = store.search_bm25("quota", 10)
    assert hits, "FTS5 returned nothing for a term that is present"
    assert hits[0][0] == "winner"
    assert [cid for cid, _ in hits] == ["winner", "middle"]
    assert all(score > 0 for _, score in hits), "scores must be positive-better"
    assert hits[0][1] > hits[1][1]
    store.close()


def test_bm25_query_sanitising_survives_operators_and_punctuation() -> None:
    store = SqliteStore(":memory:")
    store.initialize()
    store.upsert_chunks([chunk("c1", "d", 0, "requests to the /v1/generate endpoint are routed")])
    assert store.search_bm25('OR AND "NEAR(" /v1/generate*', 5)[0][0] == "c1"
    assert store.search_bm25("!!! ?? a", 5) == []
    assert store.search_bm25("", 5) == []
    store.close()


def test_neighbours_excludes_the_centre_and_respects_the_window() -> None:
    store = SqliteStore(":memory:")
    store.initialize()
    store.upsert_chunks([chunk(f"c{i}", "d", i, f"body {i}") for i in range(5)])
    assert [c.ordinal for c in store.neighbours("d", 2, 1)] == [1, 3]
    assert [c.ordinal for c in store.neighbours("d", 0, 2)] == [1, 2]
    assert store.neighbours("d", 2, 0) == []
    store.close()


def test_delete_document_returns_the_count_and_clears_the_fts_index() -> None:
    store = SqliteStore(":memory:")
    store.initialize()
    store.upsert_document(Document(doc_id="d", path="d.md", title="D", sha256="s"))
    store.upsert_chunks([chunk(f"c{i}", "d", i, "quota enforcement detail") for i in range(3)])
    assert store.search_bm25("quota", 10)

    assert store.delete_document("d") == 3
    assert store.count_chunks() == 0
    assert store.count_documents() == 0
    # The FTS triggers must have removed the rows too, or re-ingest returns ghosts.
    assert store.search_bm25("quota", 10) == []
    store.close()


def test_reupserting_a_chunk_does_not_duplicate_it_in_fts() -> None:
    store = SqliteStore(":memory:")
    store.initialize()
    store.upsert_chunks([chunk("c1", "d", 0, "original quota wording")])
    store.upsert_chunks([chunk("c1", "d", 0, "revised quota wording")])
    hits = store.search_bm25("quota", 10)
    assert [cid for cid, _ in hits] == ["c1"]
    assert store.get_chunk("c1").text == "revised quota wording"
    store.close()


# ---------------------------------------------------------------------------
# NumpyVectorStore
# ---------------------------------------------------------------------------


def unit(values: list[float]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_vector_store_add_search_count_and_meta(tmp_path) -> None:
    vectors = NumpyVectorStore(str(tmp_path / "v.npy"))
    vectors.reset("hashing", 3)
    vectors.add(["a", "b", "c"], np.vstack([unit([1, 0, 0]), unit([0, 1, 0]), unit([1, 1, 0])]))

    assert vectors.count() == 3
    assert vectors.meta() == {"embedder": "hashing", "dim": 3, "count": 3}
    hits = vectors.search(unit([1, 0, 0]), 2)
    assert [cid for cid, _ in hits] == ["a", "c"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert vectors.get("b") is not None
    assert vectors.get("missing") is None
    vectors.close()


def test_vector_store_replaces_rather_than_duplicates(tmp_path) -> None:
    vectors = NumpyVectorStore(str(tmp_path / "v.npy"))
    vectors.reset("hashing", 2)
    vectors.add(["a"], unit([1, 0]).reshape(1, 2))
    vectors.add(["a"], unit([0, 1]).reshape(1, 2))
    assert vectors.count() == 1
    assert vectors.search(unit([0, 1]), 1)[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vector_store_persist_and_reload(tmp_path) -> None:
    path = str(tmp_path / "v.npy")
    vectors = NumpyVectorStore(path)
    vectors.reset("hashing", 3)
    vectors.add(["a", "b"], np.vstack([unit([1, 0, 0]), unit([0, 1, 0])]))
    vectors.persist()

    reloaded = NumpyVectorStore(path)
    reloaded.load()
    assert reloaded.count() == 2
    assert reloaded.meta() == {"embedder": "hashing", "dim": 3, "count": 2}
    assert reloaded.search(unit([1, 0, 0]), 1)[0][0] == "a"


def test_vector_store_load_tolerates_a_missing_file(tmp_path) -> None:
    vectors = NumpyVectorStore(str(tmp_path / "absent.npy"))
    vectors.load()
    assert vectors.count() == 0
    assert vectors.search(unit([1, 0]), 3) == []


def test_vector_store_rejects_a_dimension_mismatch(tmp_path) -> None:
    vectors = NumpyVectorStore(str(tmp_path / "v.npy"))
    vectors.reset("hashing", 4)
    with pytest.raises(IndexMismatchError):
        vectors.add(["a"], np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(IndexMismatchError):
        vectors.add(["a", "b"], np.zeros((1, 4), dtype=np.float32))


def test_vector_store_reset_clears_everything(tmp_path) -> None:
    vectors = NumpyVectorStore(str(tmp_path / "v.npy"))
    vectors.reset("hashing", 2)
    vectors.add(["a"], unit([1, 0]).reshape(1, 2))
    vectors.reset("other", 5)
    assert vectors.count() == 0
    assert vectors.meta() == {"embedder": "other", "dim": 5, "count": 0}


# ---------------------------------------------------------------------------
# HashingEmbedder
# ---------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_across_instances() -> None:
    a = HashingEmbedder(dim=64, seed=1337).embed_query("the retention window is 90 days")
    b = HashingEmbedder(dim=64, seed=1337).embed_query("the retention window is 90 days")
    assert np.array_equal(a, b)


def test_hashing_embedder_honours_dim_and_returns_unit_vectors() -> None:
    embedder = HashingEmbedder(dim=32, seed=7)
    matrix = embedder.embed_documents(["alpha beta", "gamma delta"])
    assert matrix.shape == (2, 32)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    query = embedder.embed_query("alpha beta")
    assert query.shape == (32,)
    assert np.linalg.norm(query) == pytest.approx(1.0, abs=1e-5)


def test_hashing_embedder_scores_lexical_neighbours_above_strangers() -> None:
    embedder = HashingEmbedder(dim=512, seed=1337)
    matrix = embedder.embed_documents(
        [
            "the tenant rate limit is 600 requests per minute",
            "tenant rate limits are 600 requests each minute",
            "the on-call engineer is paged for a SEV-2 incident",
        ]
    )
    similar = float(matrix[0] @ matrix[1])
    unrelated = float(matrix[0] @ matrix[2])
    assert similar > unrelated
    assert similar > 0.5


def test_hashing_embedder_handles_empty_input() -> None:
    embedder = HashingEmbedder(dim=16)
    assert embedder.embed_documents([]).shape == (0, 16)
    assert embedder.embed_query("").shape == (16,)


def test_get_embedder_returns_the_hashing_default(tmp_path) -> None:
    embedder = get_embedder(make_settings(tmp_path))
    assert embedder.name == "hashing"
    assert embedder.dim == 256


# ---------------------------------------------------------------------------
# ingest_corpus
# ---------------------------------------------------------------------------


def write_corpus(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "gateway-overview.md").write_text(SAMPLE, encoding="utf-8")
    (corpus / "rate-limits.md").write_text(
        "# Rate Limits\n\n## Quotas\n\nEvery tenant starts with 600 requests per minute on the "
        "/v1/generate endpoint. Bursts above the limit are rejected with HTTP 429.\n",
        encoding="utf-8",
    )
    (corpus / "retention.md").write_text(
        "# Data Retention\n\n## Window\n\nIndex snapshots are retained for 90 days before "
        "deletion. The window applies to every region.\n",
        encoding="utf-8",
    )


def test_ingest_is_idempotent_and_force_reindexes(tmp_path) -> None:
    write_corpus(tmp_path)
    settings = make_settings(tmp_path)

    first = ingest_corpus(settings)
    assert first.documents_seen == 3
    assert first.documents_indexed == 3
    assert first.documents_skipped == 0
    assert first.chunks_written > 0
    assert first.embedder == "hashing"
    assert first.dim == 256
    assert first.chunk_tokens == settings.chunking.chunk_tokens
    assert first.chunk_overlap == settings.chunking.chunk_overlap

    second = ingest_corpus(settings)
    assert second.documents_indexed == 0
    assert second.documents_skipped == 3
    assert second.chunks_written == 0

    forced = ingest_corpus(settings, force=True)
    assert forced.documents_indexed == 3
    assert forced.documents_skipped == 0
    assert forced.chunks_written == first.chunks_written


def test_ingest_leaves_one_vector_per_chunk(tmp_path) -> None:
    write_corpus(tmp_path)
    settings = make_settings(tmp_path)
    report = ingest_corpus(settings)

    store, vectors = open_stores(settings)
    try:
        assert store.count_chunks() == report.chunks_written
        assert vectors.count() == store.count_chunks()
        assert vectors.meta()["embedder"] == "hashing"
        assert vectors.meta()["dim"] == 256
        for c in store.list_chunks():
            assert vectors.get(c.chunk_id) is not None
    finally:
        vectors.close()
        store.close()


def test_ingest_reindexes_a_changed_document(tmp_path) -> None:
    write_corpus(tmp_path)
    settings = make_settings(tmp_path)
    ingest_corpus(settings)

    (tmp_path / "corpus" / "retention.md").write_text(
        "# Data Retention\n\n## Window\n\nIndex snapshots are retained for 30 days before "
        "deletion. The window applies to every region.\n",
        encoding="utf-8",
    )
    report = ingest_corpus(settings)
    assert report.documents_indexed == 1
    assert report.documents_skipped == 2

    store, vectors = open_stores(settings)
    try:
        text = " ".join(c.text for c in store.list_chunks("retention"))
        assert "30 days" in text
        assert "90 days" not in text
    finally:
        vectors.close()
        store.close()
