"""Ids, tokens and text normalisation.

These live together so no module can disagree about what a token, a sentence or
a chunk id is. In particular ``normalize_text`` is the ONE normaliser used to
decide whether a gold ``answer_span`` occurs inside a chunk — which is what
makes span-resolved gold labels (and therefore the chunking ablation) legitimate
rather than circular.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timezone

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WORD_RE = re.compile(r"\w+|[^\w\s]")

STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be but by for from has have how i if in into is it its of on or
    our that the their there these they this to was were what when where which who why will
    with you your do does did can could should would about also any""".split()
)

REFUSAL_TEXT = "I don't have enough supporting evidence in the indexed corpus to answer that."


def sha256_text(text: str) -> str:
    """Full hex sha256 of a string (utf-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_doc_id(relative_path: str) -> str:
    """Stable id from a corpus-relative path: 'data-retention-policy.md' -> 'data-retention-policy'."""
    stem = relative_path.replace("\\", "/")
    if "." in stem.rsplit("/", 1)[-1]:
        stem = stem.rsplit(".", 1)[0]
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in stem.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def make_chunk_id(doc_id: str, char_start: int, char_end: int) -> str:
    """Content-addressed chunk id: first 16 hex chars of sha256(doc:start:end)."""
    return sha256_text(f"{doc_id}:{char_start}:{char_end}")[:16]


def new_trace_id() -> str:
    """Sortable, unique trace id: <ms since epoch in hex>-<8 random hex>."""
    return f"{int(time.time() * 1000):012x}-{os.urandom(4).hex()}"


def utc_now_iso() -> str:
    """Current UTC timestamp, ISO-8601, seconds precision, 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def count_tokens(text: str) -> int:
    """Deterministic, dependency-free token estimate: words+punctuation scaled by 1.3.

    Every module MUST use this so chunk token counts are identical everywhere.
    """
    if not text:
        return 0
    return max(1, int(round(len(_WORD_RE.findall(text)) * 1.3)))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate on a word boundary so count_tokens(result) <= max_tokens."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def normalize_text(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed. The ONE normaliser
    used to decide whether a gold answer_span occurs inside a chunk."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens, stopwords retained."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Shared by grounding, the stub LLM, the judge and
    the golden-set builder so none of them can disagree about what a claim is."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]


def parse_citation_markers(text: str) -> list[str]:
    """Ordered, de-duplicated citation markers in a string: ['S1', 'S3']."""
    seen: list[str] = []
    for num in CITATION_PATTERN.findall(text):
        marker = f"S{num}"
        if marker not in seen:
            seen.append(marker)
    return seen


def strip_citation_markers(text: str) -> str:
    """Remove [Sn] markers and tidy the whitespace they leave behind."""
    return " ".join(CITATION_PATTERN.sub("", text).split())


def overlap_score(claim: str, evidence: str) -> float:
    """Fraction of the claim's content tokens present in the evidence.

    The single shared definition of lexical support: used by the abstain gate,
    the stub LLM, the grounding verifier and the heuristic judge.
    """
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 0.0
    evidence_set = set(content_tokens(evidence))
    return sum(1 for t in claim_tokens if t in evidence_set) / len(claim_tokens)


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over content tokens. Used to measure and CAP the lexical
    overlap between a generated golden question and its source sentence."""
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
