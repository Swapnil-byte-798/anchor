"""Corpus loading, heading-aware chunking and indexing.

One chunker, no strategy switch. ``chunk_tokens`` and ``chunk_overlap`` are the only
knobs, which is exactly what makes the chunking half of the ablation grid a fair
comparison: the same code path runs six configurations and nothing else changes.

The property the rest of the system leans on is the char-offset contract. Every
chunk records ``char_start`` / ``char_end`` such that ``doc.text[char_start:char_end]``
IS the chunk text. Gold labels are verbatim answer spans resolved against chunk text
at eval time, so exact offsets are what keeps a label valid when the chunk size
changes underneath it.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from anchor.errors import IndexMismatchError, IngestError
from anchor.obs import get_logger, log_event
from anchor.settings import ChunkingSettings, Settings
from anchor.textutil import (
    count_tokens,
    make_chunk_id,
    make_doc_id,
    sha256_text,
    utc_now_iso,
)
from anchor.types import (
    Chunk,
    Document,
    DocumentStore,
    Embedder,
    IngestReport,
    LoadedDoc,
    VectorStore,
)

__all__ = ["load_corpus", "load_file", "chunk_document", "ingest_corpus"]

_LOG = get_logger("anchor.ingest")

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_SUPPORTED_SUFFIXES = (".md", ".txt")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _normalise_source(raw: str) -> str:
    """CRLF -> LF, trailing whitespace trimmed OUTSIDE fenced blocks only.

    Fenced code and tables are left byte-for-byte alone: a corpus that documents
    endpoint paths and shell snippets must not be silently reformatted.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line.rstrip())
            continue
        out.append(line if in_fence else line.rstrip())
    return "\n".join(out).strip("\n") + "\n"


def _extract_title(text: str, fallback: str) -> str:
    in_fence = False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _ATX_HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback.replace("-", " ").replace("_", " ").strip().title()


def load_file(path: str, corpus_dir: str) -> LoadedDoc:
    """Load a single corpus file into a :class:`LoadedDoc`."""
    file_path = Path(path)
    root = Path(corpus_dir)
    try:
        relative = file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = file_path.name
    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IngestError(f"{file_path} is not valid utf-8: {exc}") from exc
    except OSError as exc:
        raise IngestError(f"could not read {file_path}: {exc}") from exc

    text = _normalise_source(raw)
    # The digest is taken over the NORMALISED text, not the bytes on disk, so a
    # CRLF checkout on another machine does not look like a changed document and
    # force a pointless re-index.
    return LoadedDoc(
        doc_id=make_doc_id(relative),
        path=relative,
        title=_extract_title(text, file_path.stem),
        text=text,
        sha256=sha256_text(text),
    )


def load_corpus(corpus_dir: str) -> list[LoadedDoc]:
    """Recursively load every ``*.md`` / ``*.txt`` file, sorted by ``doc_id``."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise IngestError(f"corpus directory {corpus_dir!r} does not exist")
    paths: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        if any(part.startswith(".") for part in candidate.relative_to(root).parts):
            continue
        paths.append(candidate)
    docs = [load_file(str(p), corpus_dir) for p in paths]
    seen: dict[str, str] = {}
    for doc in docs:
        if doc.doc_id in seen:
            raise IngestError(
                f"duplicate doc_id {doc.doc_id!r} from {doc.path!r} and {seen[doc.doc_id]!r}"
            )
        seen[doc.doc_id] = doc.path
    return sorted(docs, key=lambda d: d.doc_id)


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        newline = text.find("\n", pos, end)
        stop = end if newline == -1 else newline
        spans.append((pos, stop))
        pos = stop + 1
    return spans


def _sections(text: str) -> list[tuple[str, int, int]]:
    """Split a document into ``(heading_path, body_start, body_end)`` sections.

    A chunk never crosses one of these boundaries, so ``heading_path`` is always a
    true statement about every sentence in the chunk.
    """
    sections: list[tuple[str, int, int]] = []
    stack: list[str] = []
    current_path = ""
    body_start = 0
    in_fence = False
    for line_start, line_end in _line_spans(text, 0, len(text)):
        line = text[line_start:line_end]
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _ATX_HEADING.match(line)
        if not match:
            continue
        sections.append((current_path, body_start, line_start))
        level = len(match.group(1))
        title = match.group(2).strip()
        del stack[level - 1 :]
        while len(stack) < level - 1:
            stack.append("")
        stack.append(title)
        current_path = " > ".join(p for p in stack if p)
        body_start = min(line_end + 1, len(text))
    sections.append((current_path, body_start, len(text)))
    return [(path, s, e) for path, s, e in sections if text[s:e].strip()]


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Blank-line separated blocks. A fenced block stays whole, blank lines included."""
    blocks: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = start
    in_fence = False
    for line_start, line_end in _line_spans(text, start, end):
        line = text[line_start:line_end]
        if _FENCE.match(line):
            in_fence = not in_fence
            if cur_start is None:
                cur_start = line_start
            cur_end = line_end
            continue
        if not line.strip() and not in_fence:
            if cur_start is not None:
                blocks.append((cur_start, cur_end))
                cur_start = None
            continue
        if cur_start is None:
            cur_start = line_start
        cur_end = line_end
    if cur_start is not None:
        blocks.append((cur_start, cur_end))
    return blocks


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence spans WITH offsets, using the same boundary rule as textutil."""
    body = text[start:end]
    cuts = [0]
    for match in _SENTENCE_BOUNDARY.finditer(body):
        cuts.append(match.end())
    cuts.append(len(body))
    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:], strict=False):
        s, e = start + a, start + b
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _overlap_start(text: str, start: int, end: int, overlap_tokens: int) -> int | None:
    """Offset of the first sentence of the trailing ``overlap_tokens`` of a chunk.

    Returns ``None`` when there is nothing to carry — which is what
    ``chunk_overlap == 0`` means, and that configuration is half of the ablation
    grid, so it has to be a real no-op rather than an approximation.
    """
    if overlap_tokens <= 0:
        return None
    spans = _sentence_spans(text, start, end)
    if len(spans) < 2:
        return None
    carried = 0
    chosen: int | None = None
    for sentence_start, sentence_end in reversed(spans):
        tokens = count_tokens(text[sentence_start:sentence_end])
        if carried and carried + tokens > overlap_tokens:
            break
        carried += tokens
        chosen = sentence_start
        if carried >= overlap_tokens:
            break
    if chosen is None or chosen <= start:
        return None
    return chosen


def chunk_document(doc: LoadedDoc, settings: ChunkingSettings) -> list[Chunk]:
    """Heading-aware, token-budgeted chunking with sentence-granular overlap.

    Rules, in order of precedence:

    1. A chunk NEVER crosses a heading boundary.
    2. Paragraphs are packed until the next one would exceed ``chunk_tokens``; a
       paragraph bigger than the budget on its own is split into sentences.
    3. ``chunk_overlap`` tokens of the previous chunk's TRAILING SENTENCES are
       carried into the next chunk of the same section (0 means none).
    4. A chunk under ``min_chunk_tokens`` is merged into the next chunk of the same
       section, or into the previous one when it is last. A section that produces a
       single short chunk emits it alone rather than losing it.

    Every chunk satisfies ``doc.text[char_start:char_end] == chunk.text``.
    """
    max_tokens = max(1, int(settings.chunk_tokens))
    overlap_tokens = max(0, int(settings.chunk_overlap))
    min_tokens = max(0, int(settings.min_chunk_tokens))
    text = doc.text

    spans: list[tuple[str, int, int]] = []
    for heading_path, body_start, body_end in _sections(text):
        units: list[tuple[int, int]] = []
        for para_start, para_end in _paragraph_spans(text, body_start, body_end):
            trimmed_start, trimmed_end = _trim(text, para_start, para_end)
            if trimmed_end <= trimmed_start:
                continue
            if count_tokens(text[trimmed_start:trimmed_end]) > max_tokens:
                units.extend(_sentence_spans(text, trimmed_start, trimmed_end))
            else:
                units.append((trimmed_start, trimmed_end))
        if not units:
            continue

        packed: list[tuple[int, int]] = []
        chunk_start: int | None = None
        chunk_end = 0
        acc = 0
        for unit_start, unit_end in units:
            unit_tokens = count_tokens(text[unit_start:unit_end])
            if chunk_start is not None and acc + unit_tokens > max_tokens:
                packed.append((chunk_start, chunk_end))
                carry = _overlap_start(text, chunk_start, chunk_end, overlap_tokens)
                if carry is None:
                    chunk_start = None
                    acc = 0
                else:
                    chunk_start = carry
                    acc = count_tokens(text[carry:chunk_end])
            if chunk_start is None:
                chunk_start = unit_start
                acc = 0
            chunk_end = unit_end
            acc += unit_tokens
        if chunk_start is not None:
            packed.append((chunk_start, chunk_end))

        merged: list[tuple[int, int]] = []
        for span_start, span_end in packed:
            if merged and count_tokens(text[merged[-1][0] : merged[-1][1]]) < min_tokens:
                merged[-1] = (merged[-1][0], span_end)
            else:
                merged.append((span_start, span_end))
        if len(merged) > 1 and count_tokens(text[merged[-1][0] : merged[-1][1]]) < min_tokens:
            tail = merged.pop()
            merged[-1] = (merged[-1][0], tail[1])

        spans.extend((heading_path, s, e) for s, e in merged)

    if not spans and text.strip():
        start, end = _trim(text, 0, len(text))
        spans = [("", start, end)]

    chunks: list[Chunk] = []
    for ordinal, (heading_path, span_start, span_end) in enumerate(spans):
        start, end = _trim(text, span_start, span_end)
        body = text[start:end]
        if not body:
            continue
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, start, end),
                doc_id=doc.doc_id,
                ordinal=ordinal,
                heading_path=heading_path,
                text=body,
                char_start=start,
                char_end=end,
                n_tokens=count_tokens(body),
                title=doc.title,
            )
        )
    # Re-number after any empty span was dropped so ordinals stay 0..n-1.
    for ordinal, chunk in enumerate(chunks):
        chunk.ordinal = ordinal
    return chunks


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------


def _revector_stored_chunks(
    store: DocumentStore,
    vectors: VectorStore,
    embedder: Embedder,
    batch: int,
) -> int:
    """Embed every stored chunk that currently has no vector; returns how many.

    This is what makes a reset safe. ``vectors.reset`` drops the whole matrix,
    including the vectors of documents this run skipped by digest, and the vector
    store has no per-id remove operation to do anything finer. Rebuilding from the
    chunk rows the store already holds restores exactly one vector per chunk
    without re-chunking unchanged documents.
    """
    pending = [chunk for chunk in store.list_chunks() if vectors.get(chunk.chunk_id) is None]
    for start in range(0, len(pending), batch):
        window = pending[start : start + batch]
        matrix = embedder.embed_documents([c.text for c in window])
        vectors.add([c.chunk_id for c in window], matrix)
    return len(pending)


def ingest_corpus(
    settings: Settings,
    *,
    corpus_dir: str | None = None,
    store: DocumentStore | None = None,
    vectors: VectorStore | None = None,
    embedder: Embedder | None = None,
    force: bool = False,
) -> IngestReport:
    """Load, chunk, embed and index the corpus. Idempotent by content digest.

    A document whose stored sha256 still matches is skipped outright; a changed one
    is deleted first and re-indexed, so running ingest twice is a no-op.

    Making that re-index a clean replacement rather than a merge takes a whole-index
    rebuild, and the reason is the chunk id. ``make_chunk_id`` hashes char offsets,
    so any edit that shifts offsets renames every later chunk; deleting the old chunk
    ROWS leaves their vectors behind, and :class:`~anchor.store.NumpyVectorStore` has
    no way to remove a vector. So whenever a run would replace an existing document
    (or the vector file is missing, or the embedder changed) the matrix is reset up
    front and rebuilt: re-indexed documents are embedded from their new chunks, and
    documents skipped by digest are re-embedded from their stored ones. The run ends
    on the invariant the rest of the system assumes — exactly one vector per chunk —
    and raises :class:`IndexMismatchError` rather than persisting an index that
    breaks it.
    """
    started = time.perf_counter()
    opened_store: DocumentStore | None = None
    opened_vectors: VectorStore | None = None
    if store is None or vectors is None:
        from anchor.store import open_stores

        fresh_store, fresh_vectors = open_stores(settings)
        if store is None:
            store, opened_store = fresh_store, fresh_store
        else:
            fresh_store.close()
        if vectors is None:
            vectors, opened_vectors = fresh_vectors, fresh_vectors
        else:
            fresh_vectors.close()
    if embedder is None:
        from anchor.embed import get_embedder

        embedder = get_embedder(settings)

    try:
        store.initialize()
        directory = corpus_dir or settings.paths.corpus_dir
        # `paths.corpus_exclude` is applied HERE and not in `load_corpus`, which
        # is also called by scripts/build_golden.py with its own exclusion list.
        # Keeping the filter on the ingest path means the served corpus and the
        # evaluated corpus are the same documents: corpus/README.md is a note
        # ABOUT the corpus, and indexing it made it retrievable as a seventh
        # document that the golden set had never heard of.
        excluded = frozenset(settings.paths.corpus_exclude)
        docs = [
            doc
            for doc in load_corpus(str(settings.resolve(directory)))
            if doc.doc_id not in excluded
        ]
        # An index built before the exclusion existed still holds the document.
        # Drop it rather than leaving it retrievable until someone runs --force.
        dropped_excluded = False
        for stale in excluded:
            if store.get_document(stale) is not None:
                store.delete_document(stale)
                dropped_excluded = True
                log_event(_LOG, logging.INFO, "ingest.excluded", doc_id=stale)

        existing_by_id = {doc.doc_id: store.get_document(doc.doc_id) for doc in docs}
        # Any document whose chunks are about to be deleted and rewritten orphans
        # its vectors: chunk ids carry char offsets, so an edit that shifts offsets
        # renames every later chunk and `add` cannot overwrite rows it can no longer
        # name. The vector store has no remove, so a rebuild is the only clean
        # replacement available.
        replaces_existing = dropped_excluded or any(
            (existing := existing_by_id[doc.doc_id]) is not None and existing.sha256 != doc.sha256
            for doc in docs
        )
        meta = vectors.meta()
        # The `count() != count_chunks()` term is what makes a plain re-run repair a
        # diverged index instead of dead-ending on the invariant check below. The
        # vector file can be deleted, truncated, or left stale by a run interrupted
        # after SQLite committed; in every one of those states no document looks
        # changed, so nothing else here would ask for a reset. There is no per-id
        # repair available — the vector store has no remove — so a rebuild is the
        # only fix, and `anchor ingest` should perform it rather than demand a
        # `--force` that nothing in the output suggests.
        needs_reset = (
            force
            or vectors.count() == 0
            or vectors.count() != store.count_chunks()
            or meta.get("embedder") != embedder.name
            or int(meta.get("dim", 0)) != embedder.dim
            or replaces_existing
        )
        # Hoisted OUT of the document loop. A run where every document is skipped by
        # digest still has to reset — the vector file can be missing or stale while
        # the SQLite index is intact — and that run reached neither the old in-loop
        # reset nor any `add`, so it persisted a zero-row matrix over a live index
        # and reported success.
        if needs_reset:
            vectors.reset(embedder.name, embedder.dim)

        batch = max(1, int(settings.embedding.batch_size))
        indexed = 0
        skipped = 0
        chunks_written = 0
        for doc in docs:
            existing = existing_by_id[doc.doc_id]
            if existing is not None and existing.sha256 == doc.sha256 and not force:
                skipped += 1
                log_event(
                    _LOG,
                    logging.DEBUG,
                    "ingest.skip",
                    doc_id=doc.doc_id,
                    reason="unchanged_sha256",
                )
                continue
            if existing is not None:
                store.delete_document(doc.doc_id)

            chunks = chunk_document(doc, settings.chunking)
            if not chunks:
                raise IngestError(f"document {doc.doc_id!r} produced no chunks")
            for start in range(0, len(chunks), batch):
                window = chunks[start : start + batch]
                matrix = embedder.embed_documents([c.text for c in window])
                vectors.add([c.chunk_id for c in window], matrix)
            store.upsert_chunks(chunks)
            store.upsert_document(
                Document(
                    doc_id=doc.doc_id,
                    path=doc.path,
                    title=doc.title,
                    sha256=doc.sha256,
                    n_chunks=len(chunks),
                    ingested_at=utc_now_iso(),
                )
            )
            indexed += 1
            chunks_written += len(chunks)
            log_event(
                _LOG,
                logging.INFO,
                "ingest.document",
                doc_id=doc.doc_id,
                path=doc.path,
                chunks=len(chunks),
                tokens=sum(c.n_tokens for c in chunks),
            )

        if needs_reset:
            # Documents skipped by digest lost their vectors to the reset above;
            # give them back from the chunk rows the store already holds.
            restored = _revector_stored_chunks(store, vectors, embedder, batch)
            if restored:
                log_event(_LOG, logging.INFO, "ingest.vectors_rebuilt", chunks=restored)

        # The invariant every consumer assumes and nothing used to check: one vector
        # per chunk. Raising here (instead of persisting) is what turns a silently
        # degraded dense leg into a visible failure — `search_dense` drops hits whose
        # chunk row is gone, so an index that breaks this returns fewer, worse
        # candidates and reports nothing.
        vector_count = vectors.count()
        chunk_count = store.count_chunks()
        if vector_count != chunk_count:
            raise IndexMismatchError(
                f"index is inconsistent after ingest: {vector_count} vectors for "
                f"{chunk_count} chunks. The vector file and the document store have "
                "diverged; re-run `anchor ingest --force` to rebuild."
            )
        vectors.persist()
        report = IngestReport(
            documents_seen=len(docs),
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_written=chunks_written,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            embedder=embedder.name,
            dim=embedder.dim,
            chunk_tokens=settings.chunking.chunk_tokens,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        log_event(
            _LOG,
            logging.INFO,
            "ingest.complete",
            documents_seen=report.documents_seen,
            documents_indexed=report.documents_indexed,
            documents_skipped=report.documents_skipped,
            chunks_written=report.chunks_written,
            embedder=report.embedder,
            dim=report.dim,
            chunk_tokens=report.chunk_tokens,
            chunk_overlap=report.chunk_overlap,
            duration_ms=round(report.duration_ms, 2),
        )
        return report
    finally:
        # Close only what this call opened; injected stores belong to the caller.
        if opened_vectors is not None:
            opened_vectors.close()
        if opened_store is not None:
            opened_store.close()
