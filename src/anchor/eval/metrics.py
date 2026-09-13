"""Every metric in this repo, implemented from scratch in one readable file.

There is no evaluation library here on purpose. A reviewer who wants to know what
``recall@5`` or ``nDCG@10`` means in these tables can read the twelve lines that
compute it, including the tie-breaks and the empty-input behaviour, rather than
work out which library version was pinned and which of its three definitions of
nDCG it implements.

Conventions that hold for every function below:

* ``retrieved`` is ALWAYS in rank order, best first, and may be shorter than ``k``.
* ``gold`` is a set-like list; duplicates in it are ignored.
* An empty input never raises: it returns 0.0 (the one documented exception is
  :func:`citation_precision_recall`, where "cited nothing and there was nothing to
  cite" is a perfect score, not a zero).
* Nothing here rounds. Rounding is the reporting layer's job.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from anchor.errors import EvalError
from anchor.textutil import normalize_text

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "normalize_answer",
    "alias_exact_match",
    "token_f1",
    "citation_precision_recall",
    "percentile",
    "cohens_kappa",
    "confusion_matrix",
]

_ARTICLES = frozenset({"a", "an", "the"})


# ==============================================================================
# Retrieval metrics
# ==============================================================================


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of the gold chunks that appear in the top ``k`` retrieved ids.

    With two gold chunks and one of them in the top k, this is 0.5 — not 1.0.
    Returns 0.0 when there is no gold (the unanswerable slice); callers average
    retrieval metrics over ANSWERABLE cases only, so that zero is never mixed in.
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    hits = sum(1 for chunk_id in set(retrieved[:k]) if chunk_id in gold_set)
    return hits / len(gold_set)


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of the top ``k`` retrieved slots that are gold.

    Divides by ``k`` rather than by the number of ids actually returned — the
    textbook definition. A short candidate list is penalised, which is correct:
    returning three results when five were asked for is a retrieval weakness.
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    hits = sum(1 for chunk_id in set(retrieved[:k]) if chunk_id in gold_set)
    return hits / k


def mrr(retrieved: list[str], gold: list[str]) -> float:
    """Reciprocal rank of the FIRST gold hit over the whole ranked list.

    1.0 when the top result is gold, 0.5 at rank 2, 0.0 when no gold chunk is
    retrieved at all.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Normalised discounted cumulative gain with BINARY relevance.

    ``DCG = sum(rel_i / log2(i + 1))`` over the top ``k`` ranks (i is 1-based), and
    the ideal DCG puts ``min(len(gold), k)`` relevant items in the top ranks. So a
    case with one gold chunk found at rank 3 scores ``(1/log2(4)) / (1/log2(2))``
    = 0.5, and finding it at rank 1 scores 1.0.
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0

    dcg = 0.0
    seen: set[str] = set()
    for position, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_set and chunk_id not in seen:
            seen.add(chunk_id)
            dcg += 1.0 / math.log2(position + 1)

    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold_set), k) + 1))
    return (dcg / ideal) if ideal > 0 else 0.0


# ==============================================================================
# Answer metrics
# ==============================================================================


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, punctuation out, articles out.

    Built on :func:`anchor.textutil.normalize_text` so that answer comparison and
    gold-span resolution cannot drift apart, with the articles dropped on top:
    "the 90 days" and "90 days" are the same answer.
    """
    return " ".join(token for token in normalize_text(text).split() if token not in _ARTICLES)


def _answer_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def alias_exact_match(answer: str, aliases: list[str]) -> bool:
    """True when a normalised alias occurs as a contiguous token run in ``answer``.

    Containment rather than string equality, because the generator here is
    EXTRACTIVE: it returns whole cited sentences, so an answer that is exactly the
    alias and nothing else would be the abnormal case. Matching on token runs, not
    substrings, is what stops the alias "4" from matching the "24" in "24 rows".
    """
    haystack = _answer_tokens(answer)
    if not haystack:
        return False
    for alias in aliases:
        needle = _answer_tokens(alias)
        if not needle:
            continue
        if len(needle) > len(haystack):
            continue
        for start in range(len(haystack) - len(needle) + 1):
            if haystack[start : start + len(needle)] == needle:
                return True
    return False


def token_f1(answer: str, aliases: list[str]) -> float:
    """SQuAD token-overlap F1 against the best-matching alias.

    Multiset (bag) intersection over normalised tokens, so a repeated token only
    counts as often as it appears in both. Returns 0.0 when there are no aliases
    or the answer is empty.
    """
    predicted = _answer_tokens(answer)
    if not predicted:
        return 0.0

    best = 0.0
    for alias in aliases:
        truth = _answer_tokens(alias)
        if not truth:
            continue
        shared = sum((Counter(predicted) & Counter(truth)).values())
        if shared == 0:
            continue
        precision = shared / len(predicted)
        recall = shared / len(truth)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def citation_precision_recall(cited: list[str], gold: list[str]) -> tuple[float, float]:
    """Precision and recall of the SET of cited doc_ids against the gold doc_ids.

    Document level, not chunk level: citing a different chunk of the right
    document is a correct citation, and holding the generator to the retriever's
    chunk boundaries would measure the chunker twice.

    Citing nothing when there was nothing to cite is ``(1.0, 1.0)`` — that is the
    correct behaviour on an unanswerable question, and scoring it zero would
    reward fabrication.
    """
    cited_set, gold_set = set(cited), set(gold)
    if not cited_set and not gold_set:
        return 1.0, 1.0
    shared = len(cited_set & gold_set)
    precision = shared / len(cited_set) if cited_set else 0.0
    recall = shared / len(gold_set) if gold_set else 0.0
    return precision, recall


# ==============================================================================
# Aggregation helpers
# ==============================================================================


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile. ``q`` is a FRACTION in [0, 1], not 0-100.

    ``percentile([10, 20], 0.5)`` is 15.0. Returns 0.0 for an empty list, so a
    latency column is never a crash on a run where every case errored.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    fraction = min(1.0, max(0.0, q))
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


# ==============================================================================
# Agreement metrics (judge validation)
# ==============================================================================


def cohens_kappa(a: list[str], b: list[str], labels: Sequence[str]) -> float:
    """Cohen's kappa: chance-corrected agreement between two raters.

    ``po`` is observed agreement — the diagonal of the confusion matrix over n.
    ``pe`` is the agreement expected by chance from the two raters' MARGINALS::

        pe = sum over labels L of  (count_a(L) / n) * (count_b(L) / n)
        kappa = (po - pe) / (1 - pe)

    Kappa is 1.0 on perfect agreement, 0.0 when the raters agree exactly as often
    as chance predicts, and negative when they agree LESS than chance. Returns 1.0
    when ``pe == 1`` (both raters used a single identical label, so chance already
    explains everything and the formula divides by zero) and 0.0 on empty input.

    Raises :class:`EvalError` on ragged input or an unknown label: a silently
    dropped row would move the number this repo reports about its own judge.
    """
    if len(a) != len(b):
        raise EvalError(f"cohens_kappa: ragged input, {len(a)} labels vs {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0

    allowed = list(labels)
    unknown = sorted({label for label in list(a) + list(b) if label not in allowed})
    if unknown:
        raise EvalError(f"cohens_kappa: labels outside {allowed}: {unknown}")

    agreements = sum(1 for left, right in zip(a, b, strict=True) if left == right)
    po = agreements / n

    count_a, count_b = Counter(a), Counter(b)
    pe = sum((count_a[label] / n) * (count_b[label] / n) for label in allowed)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def confusion_matrix(
    a: list[str], b: list[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """``{a_label: {b_label: count}}`` over the FULL label set.

    Every cell is present, including the zeros — a confusion matrix with holes in
    it invites the reader to assume the missing cells were never possible.
    """
    if len(a) != len(b):
        raise EvalError(f"confusion_matrix: ragged input, {len(a)} labels vs {len(b)}")
    allowed = list(labels)
    matrix: dict[str, dict[str, int]] = {row: {col: 0 for col in allowed} for row in allowed}
    for left, right in zip(a, b, strict=True):
        if left not in matrix or right not in matrix[left]:
            raise EvalError(
                f"confusion_matrix: labels outside {allowed}: ({left!r}, {right!r})"
            )
        matrix[left][right] += 1
    return matrix
