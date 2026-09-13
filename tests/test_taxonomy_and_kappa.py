"""The failure taxonomy's four branches, and Cohen's kappa against textbook cases.

Two things are being defended here.

*   **Kappa is the number the judge-validation report leans on**, so it is checked
    against worked examples whose value is computed in the comment above the
    assertion — including the two degenerate cases (perfect agreement, and
    agreement exactly at chance) where the formula divides by something small.

*   **The taxonomy's rule ORDER is a contract.** The classes overlap, so the test
    pins each branch on a hand-built (EvalCase, CaseResult) pair, including the
    rule that an unanswerable case never reports ``retrieval_miss``.

All offline, all in milliseconds.
"""

from __future__ import annotations

import pytest

from anchor.errors import EvalError
from anchor.eval.metrics import cohens_kappa, confusion_matrix
from anchor.eval.taxonomy import classify_failure, render_taxonomy_report, select_worst
from anchor.types import (
    FAILURE_CLASSES,
    JUDGE_LABELS,
    AnswerType,
    CaseResult,
    EvalCase,
    FailureClass,
    FailureRecord,
    GoldSpan,
)

# ==============================================================================
# Cohen's kappa
# ==============================================================================


def test_cohens_kappa_textbook_2x2() -> None:
    # The standard worked example (Cohen 1960, as it appears in every textbook):
    #
    #                 B: yes   B: no
    #     A: yes        20       5      (25)
    #     A: no         10      15      (25)
    #                  (30)    (20)      n = 50
    #
    #   po = (20 + 15) / 50 = 0.70
    #   pe = (25/50)(30/50) + (25/50)(20/50) = 0.30 + 0.20 = 0.50
    #   kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohens_kappa(a, b, ("yes", "no")) == pytest.approx(0.40)


def test_cohens_kappa_3x3() -> None:
    # 10 items over three labels, paired position by position:
    #
    #     pos    1    2    3    4    5    6    7    8    9   10
    #     a      g    g    g    g    p    p    p    u    u    u   (4 g, 3 p, 3 u)
    #     b      g    g    g    p    p    p    u    u    u    u   (3 g, 3 p, 4 u)
    #     agree  y    y    y    .    y    y    .    y    y    y
    #
    #   diagonal = 3 (g,g) + 2 (p,p) + 3 (u,u) = 8
    #   po = 8 / 10 = 0.80
    #   pe = (4/10)(3/10) + (3/10)(3/10) + (3/10)(4/10)
    #      = 0.12 + 0.09 + 0.12 = 0.33
    #   kappa = (0.80 - 0.33) / (1 - 0.33) = 0.47 / 0.67 = 0.701492537...
    a = ["grounded"] * 4 + ["partial"] * 3 + ["ungrounded"] * 3
    b = ["grounded"] * 3 + ["partial"] * 3 + ["ungrounded"] * 4
    assert cohens_kappa(a, b, JUDGE_LABELS) == pytest.approx(0.47 / 0.67)
    assert cohens_kappa(a, b, JUDGE_LABELS) == pytest.approx(0.701493, abs=1e-6)


def test_cohens_kappa_edge_cases() -> None:
    labels = JUDGE_LABELS

    # Perfect agreement across more than one label: po = 1, kappa = 1.
    perfect = ["grounded", "partial", "ungrounded", "grounded"]
    assert cohens_kappa(perfect, list(perfect), labels) == pytest.approx(1.0)

    # BOTH raters used a single identical label: pe = 1, the formula divides by
    # zero, and the documented answer is 1.0 rather than a crash.
    assert cohens_kappa(["grounded"] * 5, ["grounded"] * 5, labels) == 1.0

    # Agreement exactly at chance: each rater says grounded half the time, and
    # they agree on half the items.
    #   a: g g u u     b: g u g u
    #   po = 2/4 = 0.5 ; pe = (2/4)(2/4) + (2/4)(2/4) = 0.5 ; kappa = 0
    a = ["grounded", "grounded", "ungrounded", "ungrounded"]
    b = ["grounded", "ungrounded", "grounded", "ungrounded"]
    assert cohens_kappa(a, b, labels) == pytest.approx(0.0)

    # Worse than chance is negative, not clamped.
    assert cohens_kappa(["grounded", "ungrounded"], ["ungrounded", "grounded"], labels) < 0

    # Empty input is 0.0, ragged input and unknown labels are loud errors.
    assert cohens_kappa([], [], labels) == 0.0
    with pytest.raises(EvalError):
        cohens_kappa(["grounded"], ["grounded", "partial"], labels)
    with pytest.raises(EvalError):
        cohens_kappa(["grounded"], ["excellent"], labels)


def test_confusion_matrix_has_every_cell_including_the_zeros() -> None:
    human = ["grounded", "grounded", "partial", "ungrounded"]
    judge = ["grounded", "partial", "partial", "partial"]
    matrix = confusion_matrix(human, judge, JUDGE_LABELS)

    # Full 3x3 shape: a matrix with holes invites the reader to assume the
    # missing cells were impossible.
    assert set(matrix) == set(JUDGE_LABELS)
    for row in matrix.values():
        assert set(row) == set(JUDGE_LABELS)

    assert matrix["grounded"]["grounded"] == 1
    assert matrix["grounded"]["partial"] == 1
    assert matrix["grounded"]["ungrounded"] == 0  # a present zero
    assert matrix["partial"]["partial"] == 1
    assert matrix["ungrounded"]["partial"] == 1
    assert sum(sum(row.values()) for row in matrix.values()) == len(human)

    with pytest.raises(EvalError):
        confusion_matrix(["grounded"], ["nope"], JUDGE_LABELS)


# ==============================================================================
# The taxonomy
# ==============================================================================


def _case(case_id: str = "c-1", case_type: str = "factual", gold: bool = True) -> EvalCase:
    return EvalCase(
        id=case_id,
        question="How long are request logs retained?",
        type=case_type,  # type: ignore[arg-type]
        provenance="mechanical" if case_type != "unanswerable" else "entity_substituted",
        gold=[GoldSpan(doc_id="doc-a", answer_span="90 days")] if gold else [],
        expected_answer_type=(
            AnswerType.REFUSED_NO_EVIDENCE if case_type == "unanswerable" else AnswerType.GROUNDED
        ),
    )


def _result(
    case_id: str = "c-1",
    *,
    case_type: str = "factual",
    gold_chunks: list[str] | None = None,
    retrieved: list[str] | None = None,
    packed: list[str] | None = None,
    abstained: bool = False,
    token_f1: float = 1.0,
    error: str | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        question="How long are request logs retained?",
        type=case_type,
        provenance="mechanical",
        gold_chunk_ids=gold_chunks if gold_chunks is not None else ["g1"],
        gold_doc_ids=["doc-a"] if case_type != "unanswerable" else [],
        retrieved_chunk_ids=retrieved if retrieved is not None else ["g1", "x1"],
        packed_chunk_ids=packed if packed is not None else ["g1"],
        abstained=abstained,
        token_f1=token_f1,
        error=error,
    )


def test_classify_failure_hits_each_of_the_four_branches() -> None:
    # (a) declared unanswerable -> genuinely_unanswerable, ALWAYS.
    assert (
        classify_failure(
            _case(case_type="unanswerable", gold=False),
            _result(case_type="unanswerable", gold_chunks=[], retrieved=["x1"], packed=["x1"]),
        )
        is FailureClass.GENUINELY_UNANSWERABLE
    )

    # (b) gold nowhere in the ranked list -> retrieval_miss.
    assert (
        classify_failure(_case(), _result(retrieved=["x1", "x2"], packed=["x1"]))
        is FailureClass.RETRIEVAL_MISS
    )

    # (c) retrieved but not packed -> wrong_chunk_retrieved (ranking or packing).
    assert (
        classify_failure(_case(), _result(retrieved=["x1", "g1"], packed=["x1"]))
        is FailureClass.WRONG_CHUNK_RETRIEVED
    )

    # (d) packed and still wrong (here: refused) -> context_ignored_by_generator.
    assert (
        classify_failure(_case(), _result(retrieved=["g1"], packed=["g1"], abstained=True))
        is FailureClass.CONTEXT_IGNORED_BY_GENERATOR
    )


def test_an_unanswerable_case_never_reports_retrieval_miss() -> None:
    # THE ordering rule. This case retrieved nothing gold — because there is no
    # gold to retrieve — and rule (b) in isolation would call it retrieval_miss.
    # Rule (a) must win, or the retriever gets blamed for the corpus.
    case = _case(case_type="unanswerable", gold=False)
    result = _result(case_type="unanswerable", gold_chunks=[], retrieved=[], packed=[])
    assert classify_failure(case, result) is FailureClass.GENUINELY_UNANSWERABLE

    # The same protection applies to an answerable case whose labels resolved to
    # nothing (a label bug): it is not a retrieval failure either.
    unresolved = _result(gold_chunks=[], retrieved=["x1"], packed=["x1"])
    assert classify_failure(_case(), unresolved) is FailureClass.GENUINELY_UNANSWERABLE


def test_select_worst_is_deterministic_and_ordered_by_severity() -> None:
    cases = {
        "err": _case("err"),
        "wrong-abstain": _case("wrong-abstain"),
        "lowf1": _case("lowf1"),
        "tie-b": _case("tie-b"),
        "tie-a": _case("tie-a"),
        "good": _case("good"),
    }
    results = [
        _result("good", token_f1=1.0),
        # identical on every key except the id: the tie-break must order them a, b
        _result("tie-b", token_f1=0.5),
        _result("tie-a", token_f1=0.5),
        _result("lowf1", token_f1=0.1),
        _result("wrong-abstain", token_f1=0.9, abstained=True),  # refused an answerable case
        _result("err", token_f1=0.0, error="boom"),
    ]

    order = [result.case_id for result in select_worst(cases, results, 6)]
    # errored first, then the wrong abstention, then ascending token_f1,
    # then the id tie-break (tie-a before tie-b), then the healthy case.
    assert order == ["err", "wrong-abstain", "lowf1", "tie-a", "tie-b", "good"]

    # Shuffling the input cannot change the output: the key is total.
    assert [r.case_id for r in select_worst(cases, list(reversed(results)), 6)] == order

    # worst_n truncates, and a non-positive n selects nothing.
    assert [r.case_id for r in select_worst(cases, results, 2)] == ["err", "wrong-abstain"]
    assert select_worst(cases, results, 0) == []


def test_render_taxonomy_report_covers_all_four_classes_and_worked_examples() -> None:
    distribution = {name: 0 for name in FAILURE_CLASSES}
    distribution["retrieval_miss"] = 1
    distribution["context_ignored_by_generator"] = 1

    records = [
        FailureRecord(
            case_id="m-0001",
            question="How long are request logs retained?",
            type="factual",
            failure_class=FailureClass.RETRIEVAL_MISS,
            gold_chunk_ids=["g1"],
            retrieved_chunk_ids=["x1", "x2"],
            packed_chunk_ids=["x1"],
            answer="I don't have enough supporting evidence.",
            expected_answer_type="grounded",
            got_answer_type="refused_no_evidence",
            note="why: none of the 1 gold chunk(s) appear in the candidates",
        ),
        FailureRecord(
            case_id="m-0002",
            question="What is the p95 budget?",
            type="factual",
            failure_class=FailureClass.CONTEXT_IGNORED_BY_GENERATOR,
            gold_chunk_ids=["g2"],
            retrieved_chunk_ids=["g2"],
            packed_chunk_ids=["g2"],
            answer="850 ms [S1]",
            expected_answer_type="grounded",
            got_answer_type="partial",
            note="why: a gold chunk was packed and the answer scored token_f1=0.30",
        ),
    ]

    report = render_taxonomy_report(distribution, records)

    # Every class appears in the distribution table, zeros included.
    for name in FAILURE_CLASSES:
        assert f"`{name}`" in report
    # Percentages are computed against the size of the worst set (2 here).
    assert "50%" in report
    # Both worked examples are present, with their question, note and answer.
    assert "m-0001" in report and "m-0002" in report
    assert "How long are request logs retained?" in report
    assert "850 ms [S1]" in report
    # A class with no example says so rather than silently vanishing.
    assert "_No case in the worst set fell into this class._" in report
    # And the honesty rule about unanswerable cases is stated in the report.
    assert "fabrication_rate_unanswerable" in report


def test_render_taxonomy_report_handles_an_empty_run() -> None:
    empty = {name: 0 for name in FAILURE_CLASSES}
    report = render_taxonomy_report(empty, [])
    assert "no results" in report.lower()
    for name in FAILURE_CLASSES:
        assert f"`{name}`" in report
