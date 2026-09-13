"""Golden cases in, chunk ids out — and the reason the chunking ablation is honest.

A gold label in this repo is ``{doc_id, verbatim answer_span}``. It NEVER names a
chunk id. :func:`resolve_gold_chunks` turns a span into the set of chunk ids that
actually contain it, at EVAL TIME, against whatever index is currently built.

That single decision is what makes the chunking half of the ablation meaningful
rather than circular. If labels named chunk ids, then re-chunking at 256 or 1024
tokens would invalidate every label, and the only way to keep the grid running
would be to re-label per config — which is to say, to tune the labels until the
configuration you like wins. Because a label is a span, re-chunking changes only
which chunks contain it, never whether the label is correct. The same 43
questions, unmodified, are scored against all six chunk configurations.

The other half of the bargain: a span that resolves to ZERO chunks is a bug in
the labels, not a score of zero. It raises :class:`~anchor.errors.EvalError`
naming the case and the span, because a label that silently scores 0 would look
exactly like a retrieval failure and would quietly depress every recall number in
the repo.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from anchor.errors import EvalError
from anchor.textutil import normalize_text
from anchor.types import DocumentStore, EvalCase

__all__ = ["load_cases", "resolve_gold_chunks"]


def load_cases(path: str) -> list[EvalCase]:
    """Parse a golden-set JSONL file into :class:`EvalCase` objects.

    A malformed row raises :class:`EvalError` naming the 1-based LINE NUMBER, so a
    broken label is a 5-second fix rather than a pydantic traceback with no
    location in it.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise EvalError(f"golden cases file not found: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"could not read {path}: {exc}") from exc

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate_json(line)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
                for error in exc.errors()[:3]
            )
            raise EvalError(f"{path}:{line_number}: invalid case: {detail}") from None
        except ValueError as exc:  # not JSON at all
            raise EvalError(f"{path}:{line_number}: not valid JSON: {exc}") from None

        if case.id in seen:
            raise EvalError(f"{path}:{line_number}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)

    if not cases:
        raise EvalError(f"{path} contains no cases")
    return cases


def resolve_gold_chunks(case: EvalCase, store: DocumentStore) -> list[str]:
    """Resolve every gold span of ``case`` to the chunk ids that contain it.

    Both the span and each candidate chunk are put through
    :func:`anchor.textutil.normalize_text` — the ONE normaliser in the repo — so
    that punctuation, casing and whitespace differences between the corpus text
    and the label cannot cause a spurious miss. A span may legitimately resolve to
    more than one chunk when the chunker's overlap window duplicates it; every
    match counts, which is why retrieving either copy is scored as a hit.

    Returns an empty list for a case with no gold spans (the unanswerable slice).
    Raises :class:`EvalError` when a span that WAS supplied matches nothing.
    """
    resolved: list[str] = []
    for span in case.gold:
        needle = normalize_text(span.answer_span)
        if not needle:
            raise EvalError(
                f"case {case.id!r}: gold span for doc_id={span.doc_id!r} is empty after "
                "normalisation"
            )

        chunks = store.list_chunks(span.doc_id)
        if not chunks:
            raise EvalError(
                f"case {case.id!r}: doc_id={span.doc_id!r} has no chunks in the index — "
                "re-run `anchor ingest` (or check the doc_id on the label)"
            )

        matches = [chunk.chunk_id for chunk in chunks if needle in normalize_text(chunk.text)]
        if not matches:
            raise EvalError(
                f"case {case.id!r}: gold span does not occur in any chunk of "
                f"doc_id={span.doc_id!r}: {span.answer_span!r}. A label that resolves to zero "
                "chunks is a bug in the label, not a score of zero."
            )
        for chunk_id in matches:
            if chunk_id not in resolved:
                resolved.append(chunk_id)
    return resolved
