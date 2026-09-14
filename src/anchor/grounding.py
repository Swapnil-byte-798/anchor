"""Sentence-level grounding verification.

This is the step that turns "the model said something" into "the model said
something the retrieved evidence actually supports". Every sentence of an answer is
checked against the SPECIFIC blocks it cites — not against the whole context — so a
sentence cannot be laundered as grounded by sitting next to unrelated evidence that
happens to mention the same words.

Deliberately lexical and deliberately cheap: it runs on every request, offline, with
no model call. It is a verifier, not a judge; the judge seam
(:mod:`anchor.judge`) is a separate, validated scorer.
"""

from __future__ import annotations

import re

from anchor.settings import Settings
from anchor.textutil import (
    overlap_score,
    parse_citation_markers,
    split_sentences,
    strip_citation_markers,
)
from anchor.types import AnswerType, Citation, Claim, ContextBlock, GroundingResult

__all__ = ["check_grounding", "classify_answer"]

#: How much of a cited block is quoted back as evidence on a Citation.
_QUOTE_CHARS = 200

#: A claim boundary that the shared sentence splitter cannot see: a citation marker
#: sitting AFTER the full stop, as in "... deletion. [S1] Bursts are ... [S2]".
#: ``textutil._SENTENCE_SPLIT`` needs punctuation immediately before the space and a
#: capital immediately after it, and "] " satisfies neither, so without this the whole
#: answer collapses into one claim and per-sentence verification silently stops
#: happening. Both citation styles occur in practice — the stub emits the marker after
#: the stop, remote models usually put it before — so both must split.
_MARKER_CLAIM_BREAK = re.compile(r"(?<=\])\s+(?=[A-Z0-9(\"'])")


def _split_claims(answer_text: str) -> list[str]:
    """Sentences, then a second pass on citation-marker boundaries."""
    claims: list[str] = []
    for sentence in split_sentences(answer_text):
        claims.extend(part.strip() for part in _MARKER_CLAIM_BREAK.split(sentence) if part.strip())
    return claims


def check_grounding(
    answer_text: str, blocks: list[ContextBlock], settings: Settings
) -> GroundingResult:
    """Score each sentence of ``answer_text`` against the blocks it cites.

    A sentence with no citation markers scores 0.0 and is unsupported by
    construction — an uncited claim is exactly the failure mode this repo exists to
    measure, so it is never given the benefit of the doubt.

    Returns supported sentences in ``claims``, unsupported ones in
    ``dropped_claims``, and one :class:`Citation` per distinct marker used by a
    SUPPORTED claim. A citation attached only to a dropped sentence is not evidence
    of anything and is not reported.
    """
    threshold = float(settings.grounding.support_threshold)
    by_marker = {block.marker: block for block in blocks}

    sentences = _split_claims(answer_text)
    claims: list[Claim] = []
    dropped: list[Claim] = []
    uncited = 0
    used_markers: list[str] = []

    for sentence in sentences:
        markers = parse_citation_markers(sentence)
        bare = strip_citation_markers(sentence)
        if not markers:
            uncited += 1
            support = 0.0
        else:
            support = max(
                (
                    overlap_score(bare, by_marker[marker].text)
                    for marker in markers
                    if marker in by_marker
                ),
                default=0.0,
            )
        claim = Claim(
            text=sentence,
            citations=markers,
            supported=support >= threshold,
            support_score=round(support, 4),
        )
        if claim.supported:
            claims.append(claim)
            for marker in markers:
                if marker in by_marker and marker not in used_markers:
                    used_markers.append(marker)
        else:
            dropped.append(claim)

    citations = [
        Citation(
            marker=marker,
            chunk_id=by_marker[marker].chunk_id,
            doc_id=by_marker[marker].doc_id,
            title=by_marker[marker].title,
            heading_path=by_marker[marker].heading_path,
            quote=by_marker[marker].text[:_QUOTE_CHARS],
        )
        for marker in used_markers
    ]

    total = len(sentences)
    # The denominator is EVERY sentence, including the uncited ones. Excluding them
    # (``len(claims) / (total - uncited)``) would launder an answer that is mostly
    # uncited fabrication into a perfect score — a three-sentence fabrication beside
    # one cited sentence would report 1.0 instead of 0.25, and classify_answer would
    # return GROUNDED instead of REFUSED_LOW_SUPPORT. Do not "fix" this ratio.
    support_ratio = (len(claims) / total) if total else 0.0
    return GroundingResult(
        claims=claims,
        dropped_claims=dropped,
        citations=citations,
        support_ratio=round(support_ratio, 4),
        uncited_sentences=uncited,
    )


def classify_answer(support_ratio: float, settings: Settings) -> AnswerType:
    """Map a support ratio onto the answer taxonomy.

    An answer that falls below the partial band is NOT returned as a weak answer —
    it becomes a refusal. Refusing late, after generation, is a real guardrail: the
    gate cannot see what the generator will do with the context it was given.
    """
    if support_ratio >= settings.grounding.grounded_ratio:
        return AnswerType.GROUNDED
    if support_ratio >= settings.grounding.partial_ratio:
        return AnswerType.PARTIAL
    return AnswerType.REFUSED_LOW_SUPPORT
