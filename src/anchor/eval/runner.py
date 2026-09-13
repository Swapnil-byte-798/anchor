"""Run the golden set and turn 43 cases into one :class:`MetricReport`.

What this module is careful about, in order of how much it matters:

1.  **The unanswerable slice is measured in both directions.** Refusing
    everything is not a win, so ``refusal_rate_unanswerable`` never appears
    without ``false_refusal_rate_answerable`` next to it, and
    ``fabrication_rate_unanswerable`` counts the specific failure that matters —
    an unanswerable question answered WITH at least one citation. That is
    confident fabrication, which is far worse than a wrong answer that cites
    nothing, and it is computed exactly that way.

2.  **Retrieval metrics are averaged over ANSWERABLE cases only.** There is no
    gold chunk for an unanswerable question, so including those cases would drag
    recall down by 10/43 for doing the right thing.

3.  **Every case is broken out by type and by provenance.** 9 of the 43 questions
    were written by the author; a reviewer must be able to see what the numbers
    look like without them, in every table, without asking.

4.  **One case can never abort the run.** Any exception is caught, recorded on
    ``CaseResult.error`` and counted in ``n_errors``.

5.  ``retrieval_only=True`` skips generation, grounding and judging entirely.
    That is the mode :mod:`anchor.eval.ablation` uses, and it is why 24 grid
    cells take seconds rather than half an hour.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from anchor.agent import AgentLoop, build_tools, get_planner
from anchor.context import AppContext, build_context
from anchor.errors import AnchorError
from anchor.eval.cases import load_cases, resolve_gold_chunks
from anchor.eval.metrics import (
    alias_exact_match,
    citation_precision_recall,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    token_f1,
)
from anchor.ingest import ingest_corpus
from anchor.obs import get_logger, log_event
from anchor.settings import Settings
from anchor.textutil import utc_now_iso
from anchor.types import (
    METRIC_KEYS,
    REFUSAL_TYPES,
    AgentResult,
    AnswerResult,
    CaseResult,
    EvalCase,
    MetricReport,
)

__all__ = ["run_eval"]

_LOG = get_logger("anchor.eval.runner")

# Tools whose presence in the step list is what `expected_tool` is asking about.
_FINAL_TOOL = "final_answer"


def run_eval(
    settings: Settings,
    *,
    cases_path: str | None = None,
    limit: int | None = None,
    ctx: AppContext | None = None,
    retrieval_only: bool = False,
) -> tuple[MetricReport, list[CaseResult]]:
    """Execute the golden set. Returns the aggregate report and every case result.

    ``ctx`` is accepted so the CLI can reuse one wired system across eval, gate
    and threshold generation; when it is omitted this builds (and closes) its own.
    The corpus is ingested automatically when the index is empty, so a fresh clone
    running ``anchor eval`` works without a separate ingest step.
    """
    owns_context = ctx is None
    context = ctx if ctx is not None else build_context(settings)

    try:
        _ensure_index(context, settings)
        cases = load_cases(cases_path or settings.paths.golden_path)
        if limit is not None and limit > 0:
            cases = cases[:limit]

        agent_loop: AgentLoop | None = None
        if not retrieval_only and any(case.type == "arithmetic" for case in cases):
            # The 4 arithmetic cases are the only ones that need a tool, so the
            # loop is built only when one of them is actually in the set.
            agent_loop = AgentLoop(
                ctx=context, tools=build_tools(context), planner=get_planner(context)
            )

        results: list[CaseResult] = []
        for case in cases:
            results.append(_run_case(case, context, settings, agent_loop, retrieval_only))

        report = _aggregate(cases, results, context, settings, retrieval_only)
        log_event(
            _LOG,
            logging.INFO,
            "eval.done",
            run_id=report.run_id,
            n_cases=report.n_cases,
            retrieval_only=retrieval_only,
            recall_at_5=report.metrics.get("recall_at_5", 0.0),
        )
        return report, results
    finally:
        if owns_context:
            context.close()


# ==============================================================================
# One case
# ==============================================================================


def _run_case(
    case: EvalCase,
    context: AppContext,
    settings: Settings,
    agent_loop: AgentLoop | None,
    retrieval_only: bool,
) -> CaseResult:
    """Execute one golden case. Never raises: a failure becomes ``error``."""
    result = CaseResult(
        case_id=case.id,
        question=case.question,
        type=case.type,
        provenance=case.provenance,
        gold_doc_ids=sorted({span.doc_id for span in case.gold}),
    )

    try:
        result.gold_chunk_ids = resolve_gold_chunks(case, context.store)
    except AnchorError as exc:
        # An unresolvable label is a bug in the labels. It is recorded loudly on
        # the case and counted in n_errors rather than scored as a retrieval miss,
        # which is what it would look like otherwise.
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    try:
        if retrieval_only:
            _run_retrieval_only(case, context, settings, result)
        else:
            _run_full(case, context, settings, agent_loop, result)
    except Exception as exc:  # one bad case must not kill 42 good ones
        result.error = f"{type(exc).__name__}: {exc}"
        log_event(_LOG, logging.WARNING, "eval.case_failed", case_id=case.id, error=str(exc))
    return result


def _run_retrieval_only(
    case: EvalCase, context: AppContext, settings: Settings, result: CaseResult
) -> None:
    """Retrieval, gate and packing only — no generation, grounding or judging."""
    started = time.perf_counter()
    retrieval = context.retriever.retrieve(case.question)
    result.latency_ms = (time.perf_counter() - started) * 1000.0

    cap = settings.retrieval.max_candidates_recorded
    result.retrieved_chunk_ids = [candidate.chunk_id for candidate in retrieval.candidates[:cap]]
    result.packed_chunk_ids = [block.chunk_id for block in retrieval.blocks]
    result.gate_passed = retrieval.gate.passed
    result.gate_lexical_overlap = retrieval.gate.lexical_overlap
    result.gate_top_share = retrieval.gate.top_share
    # The gate decision IS the abstention decision at this stage; nothing
    # downstream runs, so there is no low-support refusal to fold in.
    result.abstained = not retrieval.gate.passed


def _run_full(
    case: EvalCase,
    context: AppContext,
    settings: Settings,
    agent_loop: AgentLoop | None,
    result: CaseResult,
) -> None:
    """The whole request path, exactly as POST /v1/query or /v1/agent runs it."""
    answer: AnswerResult | AgentResult
    if case.type == "arithmetic" and agent_loop is not None:
        # The arithmetic cases are the reason the agent exists: they need a tool.
        answer = agent_loop.run(case.question, include_trace=True)
    else:
        answer = context.answer_service.answer(case.question, include_trace=True)

    result.answer = answer.answer
    result.answer_type = answer.answer_type
    result.abstained = answer.answer_type in REFUSAL_TYPES
    result.groundedness = answer.groundedness
    result.judge_label = answer.judge.label.value if answer.judge is not None else None
    result.latency_ms = answer.latency_ms
    result.cited_doc_ids = _unique([citation.doc_id for citation in answer.citations])

    trace = answer.trace
    if trace is not None:
        # Trace.candidates is already capped at retrieval.max_candidates_recorded
        # and is in final-rank order, which is exactly the ranked list to score.
        result.retrieved_chunk_ids = [candidate.chunk_id for candidate in trace.candidates]
        result.packed_chunk_ids = [block.chunk_id for block in trace.context]
        if trace.gate is not None:
            result.gate_passed = trace.gate.passed
            result.gate_lexical_overlap = trace.gate.lexical_overlap
            result.gate_top_share = trace.gate.top_share
        else:
            # No gate decision means retrieval never ran (a policy refusal).
            result.gate_passed = False

    if isinstance(answer, AgentResult):
        result.steps_used = answer.steps_used
        result.tool_used = _first_substantive_tool(answer)

    result.exact_match = alias_exact_match(result.answer, case.answer_aliases)
    result.token_f1 = token_f1(result.answer, case.answer_aliases)


def _first_substantive_tool(answer: AgentResult) -> str | None:
    """The first successful non-``final_answer`` tool the agent actually ran."""
    for step in answer.tool_steps:
        if step.ok and step.tool != _FINAL_TOOL:
            return step.tool
    return None


# ==============================================================================
# Aggregation
# ==============================================================================


def _aggregate(
    cases: list[EvalCase],
    results: list[CaseResult],
    context: AppContext,
    settings: Settings,
    retrieval_only: bool,
) -> MetricReport:
    by_id = {case.id: case for case in cases}
    fingerprint = settings.fingerprint()
    run_id = f"{utc_now_iso().replace('-', '').replace(':', '')}-{fingerprint}"

    notes: list[str] = []
    provider_note = (
        "llm=stub (offline extractive): these numbers measure the PIPELINE — retrieval, "
        "span resolution, gating, refusal and citation — not model quality."
        if context.llm.name == "stub"
        else f"llm={context.llm.name}: a real provider produced this run."
    )
    notes.append(provider_note)
    if context.llm.name == "stub" and not retrieval_only:
        notes.append(
            "token_f1 is structurally low under the extractive stub and should not be read as "
            "answer quality: the stub returns whole cited SENTENCES while the gold aliases are "
            "short spans ('$180'), so token overlap is diluted by every other word in the "
            "sentence. exact_match — does the alias appear verbatim in the answer — is the "
            "informative answer-side metric on this profile."
        )
    if retrieval_only:
        notes.append(
            "retrieval_only=True: generation, grounding and judging were skipped, so every "
            "answer-side metric below is 0.0 by construction."
        )
    errored = [result.case_id for result in results if result.error]
    if errored:
        notes.append(f"{len(errored)} case(s) errored and scored zero: {', '.join(errored)}")
    mechanical = [case for case in cases if case.provenance == "mechanical"]
    if mechanical:
        overlaps = [case.stem_overlap for case in mechanical]
        notes.append(
            f"mechanical questions (n={len(mechanical)}): mean lexical overlap with their source "
            f"sentence {sum(overlaps) / len(overlaps):.3f}, max {max(overlaps):.3f} "
            "(capped at 0.50 by the builder) — absolute recall is optimistic; trust the deltas."
        )

    tool_choice_n = sum(
        1
        for result in results
        if (case := by_id.get(result.case_id)) is not None and case.expected_tool
    )
    if not retrieval_only:
        notes.append(
            f"tool_choice_accuracy is scored over the {tool_choice_n} of {len(results)} cases "
            "that declare an expected_tool — NOT over all of them. At that denominator one case "
            f"moves the number by {1 / tool_choice_n:.3f}." if tool_choice_n
            else "tool_choice_accuracy has no scored cases: no golden case declares an "
            "expected_tool, so the reported 0.000 means 'not measured', not 'always wrong'."
        )
    # Re-checked every run rather than asserted once: is `groundedness` carrying
    # any information the abstain flag does not already carry?
    if not retrieval_only and results:
        degenerate = all(
            result.groundedness == (0.0 if result.abstained else 1.0) for result in results
        )
        if degenerate:
            notes.append(
                "groundedness is DEGENERATE on this run: it takes only the values 0.0 and 1.0 "
                "and is exactly `0.0 if abstained else 1.0` on every case, which makes it "
                "algebraically identical to 1 - false_refusal_rate_answerable over the "
                "answerable slice and means it carries no grounding information. The cause is a "
                "closed loop on the stub profile — the stub quotes sentences verbatim out of the "
                "block it then cites, so the verifier is scoring a sentence against its own "
                "source and cannot fail it. It is reported but NOT gated in eval/thresholds.yaml "
                "for exactly this reason; a real provider should make this note disappear."
            )

    return MetricReport(
        run_id=run_id,
        created_at=utc_now_iso(),
        config_fingerprint=fingerprint,
        llm_name=context.llm.name,
        embedder_name=context.embedder.name,
        judge_name=context.judge.name,
        retrieval_mode=settings.retrieval.mode,
        chunk_tokens=settings.chunking.chunk_tokens,
        chunk_overlap=settings.chunking.chunk_overlap,
        n_cases=len(results),
        tool_choice_n=tool_choice_n,
        metrics=_compute_metrics(by_id, results),
        by_type=_grouped(by_id, results, key=lambda result: result.type),
        by_provenance=_grouped(by_id, results, key=lambda result: result.provenance),
        notes=notes,
    )


def _compute_metrics(cases: dict[str, EvalCase], results: list[CaseResult]) -> dict[str, float]:
    """The EXACT :data:`anchor.types.METRIC_KEYS` key set, over the given results.

    Split of the denominators, stated once:

    * retrieval and answer metrics -> answerable cases (those with gold spans)
    * refusal / fabrication rates  -> the unanswerable slice
    * false-refusal rate           -> answerable cases
    * abstention accuracy          -> all cases
    * tool-choice accuracy         -> only cases carrying an ``expected_tool``
    """
    answerable = [r for r in results if not _is_unanswerable(cases.get(r.case_id), r)]
    unanswerable = [r for r in results if _is_unanswerable(cases.get(r.case_id), r)]
    scorable = [r for r in answerable if r.gold_chunk_ids]

    citation_scores = [
        citation_precision_recall(r.cited_doc_ids, r.gold_doc_ids) for r in answerable
    ]

    metrics: dict[str, float] = {
        "recall_at_5": _mean(
            [recall_at_k(r.retrieved_chunk_ids, r.gold_chunk_ids, 5) for r in scorable]
        ),
        "recall_at_10": _mean(
            [recall_at_k(r.retrieved_chunk_ids, r.gold_chunk_ids, 10) for r in scorable]
        ),
        "precision_at_5": _mean(
            [precision_at_k(r.retrieved_chunk_ids, r.gold_chunk_ids, 5) for r in scorable]
        ),
        "mrr": _mean([mrr(r.retrieved_chunk_ids, r.gold_chunk_ids) for r in scorable]),
        "ndcg_at_10": _mean(
            [ndcg_at_k(r.retrieved_chunk_ids, r.gold_chunk_ids, 10) for r in scorable]
        ),
        "exact_match": _mean([1.0 if r.exact_match else 0.0 for r in answerable]),
        "token_f1": _mean([r.token_f1 for r in answerable]),
        "citation_precision": _mean([precision for precision, _ in citation_scores]),
        "citation_recall": _mean([recall for _, recall in citation_scores]),
        "groundedness": _mean([r.groundedness for r in answerable]),
        "refusal_rate_unanswerable": _mean([1.0 if r.abstained else 0.0 for r in unanswerable]),
        # CONFIDENT FABRICATION: answered an unanswerable question AND cited
        # something. An unsupported guess with no citation is bad; a guess
        # wearing a citation is the failure this repo is built to catch.
        "fabrication_rate_unanswerable": _mean(
            [
                1.0 if (not r.abstained and r.cited_doc_ids) else 0.0
                for r in unanswerable
            ]
        ),
        "false_refusal_rate_answerable": _mean([1.0 if r.abstained else 0.0 for r in answerable]),
        "abstention_accuracy": _mean(
            [_abstention_correct(cases.get(r.case_id), r) for r in results]
        ),
        "tool_choice_accuracy": _tool_choice_accuracy(cases, results),
        "latency_p50_ms": percentile([r.latency_ms for r in results], 0.50),
        "latency_p95_ms": percentile([r.latency_ms for r in results], 0.95),
        "n_errors": float(sum(1 for r in results if r.error)),
    }

    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:  # pragma: no cover - guards a future edit, not a runtime path
        raise AssertionError(f"metric keys missing from the report: {missing}")
    return {key: round(metrics[key], 6) for key in METRIC_KEYS}


def _grouped(
    cases: dict[str, EvalCase],
    results: list[CaseResult],
    *,
    key: Callable[[CaseResult], str],
) -> dict[str, dict[str, float]]:
    """The full metric block, recomputed for each value of ``key``."""
    buckets: dict[str, list[CaseResult]] = {}
    for result in results:
        buckets.setdefault(key(result), []).append(result)
    return {name: _compute_metrics(cases, bucket) for name, bucket in sorted(buckets.items())}


def _is_unanswerable(case: EvalCase | None, result: CaseResult) -> bool:
    """Unanswerable by the CASE's declaration, never by what retrieval did."""
    if case is not None:
        return case.type == "unanswerable" or not case.gold
    return result.type == "unanswerable" or not result.gold_doc_ids


def _abstention_correct(case: EvalCase | None, result: CaseResult) -> float:
    """1.0 when the system abstained exactly when it was supposed to."""
    should_abstain = _is_unanswerable(case, result)
    return 1.0 if result.abstained == should_abstain else 0.0


def _tool_choice_accuracy(cases: dict[str, EvalCase], results: list[CaseResult]) -> float:
    """Fraction of ``expected_tool`` cases whose agent actually reached for it.

    Scored on ``tool_used`` — the first successful non-``final_answer`` tool — so a
    case that expected the calculator and got ``lookup_metric`` is wrong even if
    the final answer happened to be right.
    """
    scored = [
        (case, result)
        for result in results
        if (case := cases.get(result.case_id)) is not None and case.expected_tool
    ]
    if not scored:
        return 0.0
    return _mean(
        [1.0 if result.tool_used == case.expected_tool else 0.0 for case, result in scored]
    )


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _unique(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _ensure_index(context: AppContext, settings: Settings) -> None:
    """Ingest the corpus when the index is empty, so eval works on a fresh clone."""
    if context.store.count_chunks() > 0:
        return
    log_event(_LOG, logging.INFO, "eval.ingesting", reason="empty index")
    ingest_corpus(
        settings,
        store=context.store,
        vectors=context.vectors,
        embedder=context.embedder,
        force=False,
    )
