"""Every metric against a hand-computed fixture, plus the gate's threshold maths.

The expected value of each assertion is worked out in a comment above it. A
metric implementation that is only tested against itself is not tested: these
numbers were computed on paper first, which is the only way a sign error in an
nDCG discount ever gets caught.

Everything here is offline and runs in milliseconds — no index, no corpus, no
network. The store used by the gold-span resolution tests is a tiny fake that
implements the two methods those functions actually call.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import pytest

from anchor.errors import EvalError
from anchor.eval.cases import load_cases, resolve_gold_chunks
from anchor.eval.metrics import (
    alias_exact_match,
    citation_precision_recall,
    mrr,
    ndcg_at_k,
    normalize_answer,
    percentile,
    precision_at_k,
    recall_at_k,
    token_f1,
)
from anchor.eval.report import check_thresholds, load_thresholds, write_thresholds
from anchor.types import (
    Chunk,
    Document,
    EvalCase,
    GoldSpan,
    MetricReport,
    ThresholdSpec,
)

# ==============================================================================
# A store just big enough for resolve_gold_chunks
# ==============================================================================


class FakeStore:
    """Implements only what the eval layer calls. Deliberately not a real store."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    # -- the two methods under test -----------------------------------------
    def list_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        return [c for c in self._chunks if doc_id is None or c.doc_id == doc_id]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self._chunks if c.chunk_id == chunk_id), None)

    # -- the rest of the DocumentStore protocol, unused here ----------------
    def initialize(self) -> None: ...
    def upsert_document(self, doc: Document) -> None: ...
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    def get_document(self, doc_id: str) -> Document | None:
        return None

    def list_documents(self) -> list[Document]:
        return []

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [c for c in self._chunks if c.chunk_id in set(chunk_ids)]

    def neighbours(self, doc_id: str, ordinal: int, window: int = 1) -> list[Chunk]:
        return []

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        return []

    def delete_document(self, doc_id: str) -> int:
        return 0

    def count_documents(self) -> int:
        return len({c.doc_id for c in self._chunks})

    def count_chunks(self) -> int:
        return len(self._chunks)

    def close(self) -> None: ...


def _chunk(chunk_id: str, doc_id: str, text: str, ordinal: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=ordinal,
        text=text,
        char_start=0,
        char_end=len(text),
    )


# ==============================================================================
# Retrieval metrics
# ==============================================================================


def test_recall_at_k_counts_gold_found_not_slots_filled() -> None:
    retrieved = ["a", "b", "c", "d", "e", "g1"]
    # 2 gold, one of them ("c") inside the top 5 -> 1/2 = 0.5
    assert recall_at_k(retrieved, ["c", "g1"], 5) == 0.5
    # widen to 10 and both are found -> 2/2 = 1.0
    assert recall_at_k(retrieved, ["c", "g1"], 10) == 1.0
    # no gold at all -> 0.0 by documented convention (callers exclude these)
    assert recall_at_k(retrieved, [], 5) == 0.0
    # k <= 0 is 0.0, never a crash
    assert recall_at_k(retrieved, ["c"], 0) == 0.0


def test_precision_at_k_divides_by_k() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    # 1 gold hit in 5 slots -> 1/5 = 0.2
    assert precision_at_k(retrieved, ["c"], 5) == pytest.approx(0.2)
    # 2 gold hits in 5 slots -> 2/5 = 0.4
    assert precision_at_k(retrieved, ["c", "e"], 5) == pytest.approx(0.4)
    # a SHORT list is still divided by k: 1 hit, k=5 -> 0.2, not 0.5
    assert precision_at_k(["c", "z"], ["c"], 5) == pytest.approx(0.2)


def test_mrr_is_the_reciprocal_of_the_first_hit() -> None:
    assert mrr(["g", "b", "c"], ["g"]) == 1.0
    assert mrr(["a", "g", "c"], ["g"]) == 0.5
    # first hit at rank 4 -> 1/4; the second gold at rank 5 does not matter
    assert mrr(["a", "b", "c", "g", "h"], ["g", "h"]) == 0.25
    assert mrr(["a", "b"], ["g"]) == 0.0


def test_ndcg_at_10_on_a_five_item_ranking_worked_by_hand() -> None:
    # Ranking:  c1 c2 c3 c4 c5      gold = {c2, c5}, k = 10
    #
    #   DCG  = 1/log2(2+1) + 1/log2(5+1)
    #        = 1/1.5849625 + 1/2.5849625
    #        = 0.63092975  + 0.38685281   = 1.01778256
    #   IDCG = 1/log2(1+1) + 1/log2(2+1)          (2 gold, both at the top)
    #        = 1.0         + 0.63092975  = 1.63092975
    #   nDCG = 1.01778256 / 1.63092975   = 0.62405...
    ranking = ["c1", "c2", "c3", "c4", "c5"]
    expected = (1 / math.log2(3) + 1 / math.log2(6)) / (1.0 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, ["c2", "c5"], 10) == pytest.approx(expected)
    assert ndcg_at_k(ranking, ["c2", "c5"], 10) == pytest.approx(0.624050, abs=1e-6)

    # A perfect ranking is exactly 1.0 ...
    assert ndcg_at_k(["c2", "c5", "c1"], ["c2", "c5"], 10) == pytest.approx(1.0)
    # ... and one gold at rank 3 with k=10 is (1/log2(4)) / (1/log2(2)) = 0.5
    assert ndcg_at_k(["a", "b", "g"], ["g"], 10) == pytest.approx(0.5)
    # nothing retrieved -> 0.0
    assert ndcg_at_k([], ["g"], 10) == 0.0


# ==============================================================================
# Answer metrics
# ==============================================================================


def test_normalize_answer_drops_case_punctuation_and_articles() -> None:
    assert normalize_answer("The 90 days!") == "90 days"
    assert normalize_answer("$180 per month") == "180 per month"
    assert normalize_answer("  ") == ""


def test_alias_exact_match_is_a_token_run_not_a_substring() -> None:
    answer = "Gateway request logs are retained for 90 days. [S1]"
    assert alias_exact_match(answer, ["90 days"]) is True
    assert alias_exact_match(answer, ["ninety days", "90 days"]) is True
    # "9 days" is not a token run of "... 90 days ..."
    assert alias_exact_match(answer, ["9 days"]) is False
    # substring matching would wrongly fire here; token matching must not
    assert alias_exact_match("The report has 24 rows.", ["4"]) is False
    assert alias_exact_match("The report has 4 rows.", ["4"]) is True
    # empty inputs are False, never an exception
    assert alias_exact_match("", ["90 days"]) is False
    assert alias_exact_match(answer, []) is False
    assert alias_exact_match(answer, [""]) is False


def test_token_f1_takes_the_best_alias() -> None:
    # answer = "90 days" (2 tokens), alias "90 days" -> precision 1, recall 1 -> 1.0
    assert token_f1("90 days", ["90 days"]) == pytest.approx(1.0)
    # answer 4 tokens ("logs are kept 90 days" normalises to 5 tokens:
    # logs are kept 90 days); alias "90 days" (2 tokens); shared = 2
    #   precision = 2/5 = 0.4, recall = 2/2 = 1.0
    #   f1 = 2 * 0.4 * 1.0 / 1.4 = 0.571428...
    assert token_f1("Logs are kept 90 days", ["90 days"]) == pytest.approx(4 / 7)
    # the BEST alias wins: the second one matches exactly
    assert token_f1("90 days", ["completely different", "90 days"]) == pytest.approx(1.0)
    # no overlap at all
    assert token_f1("90 days", ["SEV-2"]) == 0.0
    # no aliases, or an empty answer
    assert token_f1("90 days", []) == 0.0
    assert token_f1("", ["90 days"]) == 0.0


def test_citation_precision_recall_treats_citing_nothing_correctly() -> None:
    # cited {a, b}, gold {a} -> precision 1/2, recall 1/1
    assert citation_precision_recall(["a", "b"], ["a"]) == (0.5, 1.0)
    # nothing to cite and nothing cited: the CORRECT behaviour on an
    # unanswerable question, so it is a perfect score, not a zero
    assert citation_precision_recall([], []) == (1.0, 1.0)
    # cited something when there was nothing to cite: precision 0
    assert citation_precision_recall(["a"], []) == (0.0, 0.0)
    # cited nothing when there WAS something: recall 0
    assert citation_precision_recall([], ["a"]) == (0.0, 0.0)


def test_percentile_interpolates_and_survives_small_lists() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.95) == 7.0
    # two points, q=0.5 -> exactly halfway between them
    assert percentile([10.0, 20.0], 0.5) == pytest.approx(15.0)
    # position = 0.5 * (5-1) = 2.0 -> the exact middle element
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == pytest.approx(3.0)
    # position = 0.95 * 4 = 3.8 -> 4 + 0.8 * (5 - 4) = 4.8
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == pytest.approx(4.8)
    # unsorted input is sorted first
    assert percentile([5.0, 1.0, 3.0], 0.0) == 1.0
    assert percentile([5.0, 1.0, 3.0], 1.0) == 5.0


# ==============================================================================
# Gold-span resolution — the mechanism the chunking ablation rests on
# ==============================================================================


def _case(span: str, doc_id: str = "doc-a") -> EvalCase:
    return EvalCase(
        id="c-1",
        question="how long?",
        type="factual",
        provenance="mechanical",
        gold=[GoldSpan(doc_id=doc_id, answer_span=span)],
    )


def test_resolve_gold_chunks_matches_through_normalisation() -> None:
    store = FakeStore(
        [
            _chunk("k1", "doc-a", "Retention. Request logs are retained for 90 days.", 0),
            _chunk("k2", "doc-a", "Archive storage is billed at $180 per month.", 1),
            _chunk("k3", "doc-b", "Request logs are retained for 90 days elsewhere.", 0),
        ]
    )
    # Punctuation and casing differ from the chunk text; normalisation bridges it.
    case = _case("request logs ARE retained, for 90 days")
    assert resolve_gold_chunks(case, store) == ["k1"]

    # The same span living in two chunks (an overlap window) resolves to both.
    overlapped = FakeStore(
        [
            _chunk("k1", "doc-a", "... retained for 90 days. Tail sentence.", 0),
            _chunk("k2", "doc-a", "Tail sentence. Retained for 90 days again.", 1),
        ]
    )
    assert resolve_gold_chunks(_case("retained for 90 days"), overlapped) == ["k1", "k2"]

    # A case with no gold spans (the unanswerable slice) resolves to nothing.
    unanswerable = EvalCase(
        id="u-1", question="?", type="unanswerable", provenance="entity_substituted"
    )
    assert resolve_gold_chunks(unanswerable, store) == []


def test_resolve_gold_chunks_raises_on_an_unresolvable_label() -> None:
    store = FakeStore([_chunk("k1", "doc-a", "Request logs are retained for 90 days.")])

    # A span that is nowhere in the document is a BUG IN THE LABEL and must fail
    # loudly — scoring it 0 would look exactly like a retrieval miss.
    with pytest.raises(EvalError) as excinfo:
        resolve_gold_chunks(_case("retained for 900 days"), store)
    message = str(excinfo.value)
    assert "c-1" in message and "900 days" in message

    # An unknown doc_id names itself too.
    with pytest.raises(EvalError) as excinfo:
        resolve_gold_chunks(_case("retained for 90 days", doc_id="nope"), store)
    assert "nope" in str(excinfo.value)


def test_load_cases_reports_the_line_number_of_a_bad_row(tmp_path: Any) -> None:
    good = (
        '{"id":"m-1","question":"how long?","type":"factual","provenance":"mechanical",'
        '"gold":[{"doc_id":"doc-a","answer_span":"90 days"}]}'
    )
    path = tmp_path / "qa.jsonl"
    path.write_text(f"{good}\n\n" + '{"id":"m-2","question":"","type":"nope"}\n', encoding="utf-8")

    with pytest.raises(EvalError) as excinfo:
        load_cases(str(path))
    # Blank lines are skipped, so the bad row is line 3 — and it must say so.
    assert ":3:" in str(excinfo.value)

    path.write_text(f"{good}\n{good}\n", encoding="utf-8")
    with pytest.raises(EvalError) as excinfo:
        load_cases(str(path))
    assert "duplicate" in str(excinfo.value)

    with pytest.raises(EvalError):
        load_cases(str(tmp_path / "missing.jsonl"))


# ==============================================================================
# The regression gate
# ==============================================================================


def _report(**metrics: float) -> MetricReport:
    return MetricReport(run_id="run-1", created_at="2026-01-01T00:00:00Z", metrics=dict(metrics))


def test_check_thresholds_passes_and_fails_in_the_right_direction() -> None:
    report = _report(
        recall_at_5=0.90,
        fabrication_rate_unanswerable=0.10,
        latency_p95_ms=1234.0,
        n_errors=3.0,
    )
    spec = ThresholdSpec(
        min={"recall_at_5": 0.85},
        max={"fabrication_rate_unanswerable": 0.15},
    )
    checks = {check.metric: check for check in check_thresholds(report, spec)}
    # 0.90 >= 0.85 and 0.10 <= 0.15: both pass
    assert checks["recall_at_5"].passed is True
    assert checks["fabrication_rate_unanswerable"].passed is True

    breached = ThresholdSpec(
        min={"recall_at_5": 0.95},  # 0.90 < 0.95 -> FAIL
        max={"fabrication_rate_unanswerable": 0.05},  # 0.10 > 0.05 -> FAIL
    )
    assert [check.passed for check in check_thresholds(report, breached)] == [False, False]

    # Latency and n_errors are machine-dependent and must never be gated, even
    # when a spec names them.
    noisy = ThresholdSpec(min={"latency_p95_ms": 1.0}, max={"n_errors": 0.0})
    assert check_thresholds(report, noisy) == []

    # A metric the spec names but the report does not carry scores 0.0, which
    # FAILS a floor: silence is not a pass.
    missing = check_thresholds(_report(), ThresholdSpec(min={"recall_at_5": 0.5}))
    assert missing[0].passed is False and missing[0].value == 0.0

    # An empty spec is legal and produces no checks.
    assert check_thresholds(report, ThresholdSpec()) == []


def test_write_thresholds_puts_the_floors_exactly_slack_below_the_run(tmp_path: Any) -> None:
    report = _report(
        recall_at_5=0.90,
        recall_at_10=0.95,
        mrr=0.80,
        groundedness=0.70,
        refusal_rate_unanswerable=0.60,
        fabrication_rate_unanswerable=0.20,
        false_refusal_rate_answerable=0.30,
    )
    path = tmp_path / "thresholds.yaml"
    spec = write_thresholds(report, str(path), 0.05)

    # Floors are observed - slack ...
    assert spec.min["recall_at_5"] == pytest.approx(0.85)
    assert spec.min["recall_at_10"] == pytest.approx(0.90)
    assert spec.min["mrr"] == pytest.approx(0.75)
    assert spec.min["refusal_rate_unanswerable"] == pytest.approx(0.55)
    # groundedness is deliberately NOT gated. On the shipped stub profile it is
    # algebraically 1 - false_refusal_rate_answerable — the stub quotes verbatim
    # out of the block it cites, so the verifier cannot fail it — and gating both
    # wrote the SAME constraint into this file twice under two different names.
    assert "groundedness" not in spec.min
    assert "groundedness" not in spec.max
    # ... and ceilings are observed + slack.
    assert spec.max["fabrication_rate_unanswerable"] == pytest.approx(0.25)
    assert spec.max["false_refusal_rate_answerable"] == pytest.approx(0.35)
    assert spec.generated_from_run == "run-1"

    # The file says, in words, that a human must not edit the numbers.
    text = path.read_text(encoding="utf-8")
    assert "NEVER EDIT" in text.upper()
    assert "GENERATED" in text.upper()

    # It round-trips through the loader unchanged.
    reloaded = load_thresholds(str(path))
    assert reloaded.min == spec.min
    assert reloaded.max == spec.max
    assert reloaded.slack == 0.05
    assert reloaded.generated_from_run == "run-1"

    # Bounds are clamped into [0, 1]: a metric observed at 0.0 cannot generate a
    # floor of -0.05.
    zeroed = write_thresholds(_report(recall_at_5=0.0), str(path), 0.05)
    assert zeroed.min["recall_at_5"] == 0.0


def test_load_thresholds_rejects_a_malformed_file(tmp_path: Any) -> None:
    path = tmp_path / "thresholds.yaml"

    path.write_text("", encoding="utf-8")
    with pytest.raises(EvalError):
        load_thresholds(str(path))

    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(EvalError):
        load_thresholds(str(path))

    path.write_text("min:\n  recall_at_5: not-a-number\n", encoding="utf-8")
    with pytest.raises(EvalError):
        load_thresholds(str(path))

    path.write_text("mim:\n  recall_at_5: 0.5\n", encoding="utf-8")
    with pytest.raises(EvalError) as excinfo:
        load_thresholds(str(path))
    assert "mim" in str(excinfo.value)

    with pytest.raises(EvalError):
        load_thresholds(str(tmp_path / "absent.yaml"))


def test_committed_thresholds_file_is_loadable_and_generated() -> None:
    """The file CI gates on must parse, and must say where it came from."""
    committed = Path(__file__).resolve().parents[1] / "eval" / "thresholds.yaml"
    spec = load_thresholds(str(committed))
    assert spec.generated_from_run, "thresholds.yaml must record the run that generated it"
    assert spec.slack > 0
    assert "recall_at_5" in spec.min
    assert "fabrication_rate_unanswerable" in spec.max
