"""DECISIVE FEATURE 2 — the failure taxonomy: *what* failed, not just *that* it did.

"recall@5 is 0.81" tells a reviewer nothing about what to fix next. This module
takes the worst cases of a run and assigns each of them exactly one of four
classes, using facts already recorded on :class:`CaseResult` — gold chunk ids vs
retrieved vs packed vs the answer — so the classification is reproducible and
nobody's judgement is involved:

    retrieval_miss                the gold chunk is nowhere in the ranked list.
                                  Fix the retriever (or the chunking).
    wrong_chunk_retrieved         it WAS retrieved and then dropped before the
                                  prompt. Fix the ranking, MMR or the packer.
    context_ignored_by_generator  it was packed and the generator still got it
                                  wrong or refused. Fix the prompt, the stub's
                                  extraction rule or the grounding thresholds.
    genuinely_unanswerable        no correct retrieval was possible at all.

The report this writes (``eval/reports/failures.md``) is the artifact that shows
the author debugged a system rather than only built one: a distribution, and one
fully worked example per class with the gold span, the chunks that actually came
back and the answer that came out.
"""

from __future__ import annotations

import logging
from pathlib import Path

from anchor.eval.cases import load_cases
from anchor.eval.metrics import recall_at_k
from anchor.obs import get_logger, log_event
from anchor.settings import Settings
from anchor.types import (
    FAILURE_CLASSES,
    CaseResult,
    Chunk,
    DocumentStore,
    EvalCase,
    FailureClass,
    FailureRecord,
)

__all__ = ["classify_failure", "select_worst", "run_taxonomy", "render_taxonomy_report"]

_LOG = get_logger("anchor.eval.taxonomy")

_SNIPPET_CHARS = 160


# ==============================================================================
# Classification
# ==============================================================================


def classify_failure(case: EvalCase, result: CaseResult) -> FailureClass:
    """Assign exactly one failure class, from machine facts, in a fixed order.

    The ORDER is the contract, because the classes overlap:

    1.  ``genuinely_unanswerable`` — the case is declared unanswerable, or its
        labels resolved to no chunk at all. No correct retrieval was possible, so
        the failure lies in the corpus or the label, NOT in the retriever, and
        calling it a retrieval miss would slander a component that behaved
        correctly. An unanswerable case therefore never reports
        ``retrieval_miss``. Whether the system FABRICATED on these cases is a
        separate and more important number, reported by the runner as
        ``fabrication_rate_unanswerable`` rather than buried in this bucket.
    2.  ``retrieval_miss`` — no gold chunk anywhere in the recorded candidates.
    3.  ``wrong_chunk_retrieved`` — a gold chunk was retrieved but never reached
        the prompt: ranking, MMR or the token budget dropped it.
    4.  ``context_ignored_by_generator`` — a gold chunk WAS packed and the answer
        is still wrong, unsupported or a refusal. The evidence was there.
    """
    if case.type == "unanswerable" or not result.gold_chunk_ids:
        return FailureClass.GENUINELY_UNANSWERABLE

    gold = set(result.gold_chunk_ids)
    if not gold & set(result.retrieved_chunk_ids):
        return FailureClass.RETRIEVAL_MISS
    if not gold & set(result.packed_chunk_ids):
        return FailureClass.WRONG_CHUNK_RETRIEVED
    return FailureClass.CONTEXT_IGNORED_BY_GENERATOR


def select_worst(
    cases: dict[str, EvalCase], results: list[CaseResult], worst_n: int
) -> list[CaseResult]:
    """The ``worst_n`` cases by a deterministic severity key.

    Ordered by, in strict precedence: an execution error first, then a WRONG
    abstention decision (refused something answerable, or answered something
    unanswerable), then ascending answer quality (``token_f1``), then ascending
    ``recall@5``, then ``case_id`` so ties can never reorder between runs.
    """
    if worst_n <= 0:
        return []
    ordered = sorted(results, key=lambda result: _severity_key(cases.get(result.case_id), result))
    return ordered[:worst_n]


def _severity_key(case: EvalCase | None, result: CaseResult) -> tuple[int, int, float, float, str]:
    should_abstain = case.type == "unanswerable" if case is not None else not result.gold_doc_ids
    wrong_abstention = result.abstained != should_abstain
    return (
        0 if result.error else 1,
        0 if wrong_abstention else 1,
        result.token_f1,
        recall_at_k(result.retrieved_chunk_ids, result.gold_chunk_ids, 5),
        result.case_id,
    )


# ==============================================================================
# The report
# ==============================================================================


def run_taxonomy(
    settings: Settings,
    results: list[CaseResult],
    *,
    cases_path: str | None = None,
    worst_n: int | None = None,
    out_path: str | None = None,
) -> tuple[dict[str, int], list[FailureRecord]]:
    """Classify the worst cases and write ``eval/reports/failures.md``.

    Returns ``({failure_class: count}, records)``. The distribution always carries
    all four classes, zeros included — a class that never fires is information.
    """
    cases = {case.id: case for case in load_cases(cases_path or settings.paths.golden_path)}
    limit = worst_n if worst_n is not None and worst_n > 0 else settings.eval.worst_n
    worst = select_worst(cases, results, limit)

    # The store is opened read-only for the worked examples: a failure report
    # without the text of the chunk that came back instead is unreadable.
    from anchor.store import open_stores

    store, vectors = open_stores(settings)
    try:
        records: list[FailureRecord] = []
        for result in worst:
            case = cases.get(result.case_id)
            if case is None:  # a result from a different golden file; skip it
                continue
            records.append(_build_record(case, result, store))
    finally:
        vectors.close()
        store.close()

    distribution = {name: 0 for name in FAILURE_CLASSES}
    for record in records:
        distribution[record.failure_class.value] += 1

    destination = out_path or str(Path(settings.paths.reports_dir) / "failures.md")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_taxonomy_report(distribution, records), encoding="utf-8")
    log_event(
        _LOG,
        logging.INFO,
        "taxonomy.wrote",
        path=str(path),
        records=len(records),
        distribution=distribution,
    )
    return distribution, records


def _build_record(case: EvalCase, result: CaseResult, store: DocumentStore) -> FailureRecord:
    """One classified case, with everything a reader needs folded into ``note``."""
    failure_class = classify_failure(case, result)
    note_lines = [f"why: {_why(case, result, failure_class)}"]

    if case.gold:
        span = case.gold[0]
        note_lines.append(
            f'gold span (`{span.doc_id}`): "{_clip(span.answer_span)}" '
            f"-> chunk(s) {_ids(result.gold_chunk_ids)}"
        )
    else:
        note_lines.append("gold span: none — this question is labelled unanswerable.")

    note_lines.append("top-3 retrieved:")
    top3 = result.retrieved_chunk_ids[:3]
    if not top3:
        note_lines.append("  (nothing was retrieved)")
    for position, chunk_id in enumerate(top3, start=1):
        chunk = store.get_chunk(chunk_id)
        marker = " **[GOLD]**" if chunk_id in result.gold_chunk_ids else ""
        text = _clip(chunk.text) if chunk is not None else "(chunk not found in the index)"
        heading = _heading(chunk)
        note_lines.append(f'  {position}. `{chunk_id}`{marker} {heading} — "{text}"')

    packed = result.packed_chunk_ids
    if packed:
        packed_desc = ", ".join(
            f"[S{index}]=`{chunk_id}`"
            + (" **[GOLD]**" if chunk_id in result.gold_chunk_ids else "")
            for index, chunk_id in enumerate(packed, start=1)
        )
    else:
        packed_desc = "(nothing was packed — the gate refused before generation)"
    note_lines.append(f"packed context: {packed_desc}")

    if result.error:
        note_lines.append(f"error: {result.error}")

    return FailureRecord(
        case_id=result.case_id,
        question=result.question,
        type=result.type,
        failure_class=failure_class,
        gold_chunk_ids=list(result.gold_chunk_ids),
        retrieved_chunk_ids=list(result.retrieved_chunk_ids),
        packed_chunk_ids=list(packed),
        answer=result.answer,
        expected_answer_type=case.expected_answer_type.value,
        got_answer_type=result.answer_type.value if result.answer_type is not None else "",
        note="\n".join(note_lines),
    )


def _why(case: EvalCase, result: CaseResult, failure_class: FailureClass) -> str:
    """One machine-checkable sentence explaining the assignment."""
    if failure_class is FailureClass.GENUINELY_UNANSWERABLE:
        if case.type == "unanswerable":
            verdict = (
                "refused, which is correct" if result.abstained else "ANSWERED — a fabrication"
            )
            citations = len(result.cited_doc_ids)
            return (
                "the corpus provably cannot answer this (the subject entity occurs zero times "
                f"in it); the system {verdict}"
                + (f", citing {citations} document(s)" if citations else "")
            )
        return "the gold labels resolved to no chunk, so no correct retrieval was possible"
    if failure_class is FailureClass.RETRIEVAL_MISS:
        return (
            f"none of the {len(result.gold_chunk_ids)} gold chunk(s) appear anywhere in the "
            f"{len(result.retrieved_chunk_ids)} recorded candidates"
        )
    if failure_class is FailureClass.WRONG_CHUNK_RETRIEVED:
        return (
            "a gold chunk WAS retrieved but never reached the prompt — ranking, MMR or the "
            "token budget dropped it before packing"
        )
    quality = (
        "the system refused"
        if result.abstained
        else f"the answer scored token_f1={result.token_f1:.2f} against the gold aliases"
    )
    return f"a gold chunk was packed into the context and {quality}"


def render_taxonomy_report(distribution: dict[str, int], records: list[FailureRecord]) -> str:
    """The markdown a reviewer can read cold: distribution, then worked examples."""
    total = sum(distribution.values())
    lines = [
        "# Failure taxonomy",
        "",
        "> GENERATED by `make taxonomy` (`anchor taxonomy`). Do not edit.",
        "",
        f"The {total} worst cases of the most recent run, ranked by a deterministic severity",
        "key (errored first, then a wrong abstention decision, then ascending `token_f1`, then",
        "ascending recall@5, then case id). Each is assigned **exactly one** class from facts",
        "already on the case result — gold chunk ids vs retrieved vs packed vs the answer — so",
        "the assignment is reproducible rather than a judgement call.",
        "",
        "The order of the rules matters and is fixed: a case the corpus provably cannot answer",
        "is `genuinely_unanswerable` and never `retrieval_miss`, because no correct retrieval",
        "was available to miss. Whether the system fabricated on those cases is reported",
        "separately as `fabrication_rate_unanswerable`.",
        "",
        "## Distribution",
        "",
        "| failure class | n | % of the worst set | what it means you should fix |",
        "| --- | ---: | ---: | --- |",
    ]
    for name in FAILURE_CLASSES:
        count = distribution.get(name, 0)
        share = (100.0 * count / total) if total else 0.0
        lines.append(f"| `{name}` | {count} | {share:.0f}% | {_REMEDY[name]} |")

    if not records:
        lines.extend(["", "_No cases were selected: the run produced no results._", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "## Every case in the worst set",
            "",
            "| case | type | class | expected | got | gold retrieved | gold packed |",
            "| --- | --- | --- | --- | --- | :-: | :-: |",
        ]
    )
    for record in records:
        gold = set(record.gold_chunk_ids)
        retrieved = "yes" if gold & set(record.retrieved_chunk_ids) else "no"
        packed = "yes" if gold & set(record.packed_chunk_ids) else "no"
        if not gold:
            retrieved = packed = "n/a"
        lines.append(
            f"| `{record.case_id}` | {record.type} | `{record.failure_class.value}` "
            f"| {record.expected_answer_type} | {record.got_answer_type or '—'} "
            f"| {retrieved} | {packed} |"
        )

    lines.extend(["", "## One worked example per class", ""])
    for name in FAILURE_CLASSES:
        example = next((r for r in records if r.failure_class.value == name), None)
        if example is None:
            lines.extend(
                [f"### `{name}`", "", "_No case in the worst set fell into this class._", ""]
            )
            continue
        lines.extend(
            [
                f"### `{name}` — case `{example.case_id}`",
                "",
                f"**Question.** {example.question}",
                "",
                example.note,
                "",
                f"**Expected answer type.** `{example.expected_answer_type}` — "
                f"**got** `{example.got_answer_type or 'n/a'}`.",
                "",
                "**Answer produced.**",
                "",
                "```text",
                example.answer.strip() or "(no answer: the system refused or errored)",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


_REMEDY: dict[str, str] = {
    "retrieval_miss": "the retriever or the chunking: the evidence never came back at all",
    "wrong_chunk_retrieved": "ranking, MMR or the context packer: it came back and was dropped",
    "context_ignored_by_generator": "the prompt, the extraction rule or the grounding thresholds",
    "genuinely_unanswerable": "nothing in the retriever — check the refusal and fabrication rates",
}


def _ids(chunk_ids: list[str]) -> str:
    return ", ".join(f"`{chunk_id}`" for chunk_id in chunk_ids) if chunk_ids else "(none)"


def _heading(chunk: Chunk | None) -> str:
    if chunk is None:
        return ""
    label = chunk.heading_path or chunk.title or chunk.doc_id
    return f"({chunk.doc_id} / {label})"


def _clip(text: str, limit: int = _SNIPPET_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
