"""DECISIVE FEATURE 3 — is the judge worth anything?

FIRST, WHAT THIS DOES NOT VALIDATE. The ``groundedness`` column in the metrics
table is **not** produced by the judge. It comes from :mod:`anchor.grounding` —
the lexical per-sentence verifier — through ``AnswerResult.groundedness`` and
``CaseResult.groundedness``. The judge's verdict rides along on every response and
is recorded as ``CaseResult.judge_label``, but it feeds no key in
:data:`anchor.types.METRIC_KEYS` and no CI threshold. So this module validates the
JUDGE SEAM for its own sake; it does not underwrite any reported number, and
claiming that it did was exactly the sleight of hand this repo exists to refuse.

What it does: score the CONFIGURED judge (heuristic offline, LLM when a key
exists) against ``eval/golden/human_labels.jsonl`` — 30 frozen
``(question, answer, evidence)`` triples hand-labelled grounded/partial/
ungrounded, deliberately including fabricated answers and half-supported ones.

The triples are FROZEN TEXT. Nothing is retrieved and nothing is generated here,
so this number does not drift when the retriever changes, and it can be checked
in CI in milliseconds.

Two limits, stated here rather than buried:

1.  The labels come from ONE annotator — the author. Cohen's kappa therefore
    measures agreement between the judge and the author, NOT between the judge
    and the truth. A second annotator, or a disagreement-adjudication protocol,
    is what would be needed to call this validation in any strong sense.
2.  The kappa is measured OFF-DISTRIBUTION. On these 30 hand-written triples the
    judge emits all three labels; on the pipeline's own output it has emitted
    ``grounded`` on every answered case and nothing else. A classifier that looks
    discriminative on a validation set tells you nothing when it is constant in
    production — so when a golden run is on disk, the report prints the judge's
    label distribution over THAT run beside the validation numbers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from anchor.errors import EvalError
from anchor.eval.metrics import cohens_kappa, confusion_matrix
from anchor.judge import get_judge
from anchor.llm import get_llm
from anchor.obs import get_logger, log_event
from anchor.settings import Settings
from anchor.textutil import utc_now_iso
from anchor.types import JUDGE_LABELS, HumanLabel, JudgeAgreement

__all__ = ["load_human_labels", "validate_judge", "render_judge_report"]

_LOG = get_logger("anchor.eval.judge_validation")

# Below this, the judge is not strong enough to carry the groundedness number.
WEAK_KAPPA = 0.4

_DISAGREEMENT_PREFIX = "disagreement "


def load_human_labels(path: str) -> list[HumanLabel]:
    """Parse the frozen human-label JSONL. A bad row names its LINE NUMBER."""
    file_path = Path(path)
    if not file_path.is_file():
        raise EvalError(f"human labels file not found: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"could not read {path}: {exc}") from exc

    labels: list[HumanLabel] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label = HumanLabel.model_validate_json(line)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
                for error in exc.errors()[:3]
            )
            raise EvalError(f"{path}:{line_number}: invalid human label: {detail}") from None
        except ValueError as exc:
            raise EvalError(f"{path}:{line_number}: not valid JSON: {exc}") from None

        if label.id in seen:
            raise EvalError(f"{path}:{line_number}: duplicate label id {label.id!r}")
        seen.add(label.id)
        labels.append(label)

    if not labels:
        raise EvalError(f"{path} contains no human labels")
    return labels


def validate_judge(
    settings: Settings, *, labels_path: str | None = None, out_path: str | None = None
) -> JudgeAgreement:
    """Score the configured judge against the frozen labels; write the report.

    Every disagreement is kept, with the judge's own rationale, as a note on the
    returned :class:`JudgeAgreement` — a kappa with no visible disagreements is a
    number a reader has to take on faith.
    """
    labels = load_human_labels(labels_path or settings.paths.human_labels_path)

    # The judge is built exactly as the service builds it, so this validates what
    # actually runs — not a copy of it configured for the occasion.
    judge = get_judge(settings, get_llm(settings))

    human: list[str] = []
    predicted: list[str] = []
    notes: list[str] = []
    for label in labels:
        verdict = judge.score(label.question, label.answer, list(label.evidence))
        human.append(label.label.value)
        predicted.append(verdict.label.value)
        if verdict.label.value != label.label.value:
            notes.append(
                f"{_DISAGREEMENT_PREFIX}{label.id}: human={label.label.value} "
                f"judge={verdict.label.value} (score {verdict.score:.2f}) | "
                f"judge said: {verdict.rationale} | human said: {label.note}"
            )

    n = len(labels)
    agreements = sum(1 for left, right in zip(human, predicted, strict=True) if left == right)
    raw_agreement = agreements / n if n else 0.0
    kappa = cohens_kappa(human, predicted, JUDGE_LABELS)
    confusion = confusion_matrix(human, predicted, JUDGE_LABELS)

    per_label_recall: dict[str, float] = {}
    for name in JUDGE_LABELS:
        row = confusion.get(name, {})
        total = sum(row.values())
        per_label_recall[name] = (row.get(name, 0) / total) if total else 0.0

    notes.append(
        "the 30 labels come from ONE annotator (the author), so kappa measures "
        "judge-vs-author agreement, not judge-vs-truth."
    )
    notes.append(
        "this validates the JUDGE SEAM only. The `groundedness` metric in the README table "
        "comes from anchor.grounding, the lexical per-sentence verifier — it never passes "
        "through the judge. The judge's verdict is recorded as CaseResult.judge_label and "
        "currently feeds NO reported metric and NO CI gate."
    )
    distribution = _run_label_distribution(settings)
    if distribution:
        # Only the judge's own labels count: the refused cases carry no verdict.
        emitted = [name for name in JUDGE_LABELS if distribution.get(name)]
        if len(emitted) <= 1:
            notes.append(
                "OFF-DISTRIBUTION KAPPA: on the most recent golden run the judge emitted "
                f"{'only ' + emitted[0] if emitted else 'no label'} and nothing else. The kappa "
                "above was measured on hand-written triples where it does discriminate; a "
                "classifier that is constant on real output is not discriminating anything "
                "there, so do not read this kappa as a statement about production behaviour."
            )
    if kappa < WEAK_KAPPA:
        notes.append(
            f"WEAK JUDGE: kappa {kappa:.3f} is below {WEAK_KAPPA}. In plain words, this judge "
            "agrees with the annotator barely more than chance would. It is therefore too weak "
            "to be wired into any reported metric — which is one reason it is not."
        )

    agreement = JudgeAgreement(
        judge_name=judge.name,
        n=n,
        raw_agreement=round(raw_agreement, 6),
        cohens_kappa=round(kappa, 6),
        confusion=confusion,
        per_label_recall={key: round(value, 6) for key, value in per_label_recall.items()},
        run_label_distribution=distribution,
        notes=notes,
    )

    destination = out_path or str(Path(settings.paths.reports_dir) / "judge.md")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_judge_report(agreement), encoding="utf-8")
    log_event(
        _LOG,
        logging.INFO,
        "judge_validation.wrote",
        path=str(path),
        judge=judge.name,
        n=n,
        raw_agreement=agreement.raw_agreement,
        cohens_kappa=agreement.cohens_kappa,
    )
    return agreement


def _run_label_distribution(settings: Settings) -> dict[str, int]:
    """The judge's labels over the most recent golden run, if one is on disk.

    Read from ``<reports_dir>/results.jsonl`` — the stable copy written by every
    ``anchor eval``. Missing or unreadable is not an error: this is a diagnostic
    printed when it is available, and judge-check must still run on a clean
    checkout where no evaluation has been done yet.
    """
    path = Path(settings.paths.reports_dir) / "results.jsonl"
    if not path.is_file():
        return {}
    counts: dict[str, int] = {name: 0 for name in JUDGE_LABELS}
    counts["(no verdict — refused)"] = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label = json.loads(line).get("judge_label")
            if label in counts:
                counts[label] += 1
            elif label is None:
                counts["(no verdict — refused)"] += 1
    except (OSError, ValueError):
        return {}
    return counts


def render_judge_report(agreement: JudgeAgreement) -> str:
    """The markdown report: numbers, confusion, every disagreement, the caveat."""
    disagreements = [
        note[len(_DISAGREEMENT_PREFIX) :]
        for note in agreement.notes
        if note.startswith(_DISAGREEMENT_PREFIX)
    ]
    other_notes = [note for note in agreement.notes if not note.startswith(_DISAGREEMENT_PREFIX)]

    lines = [
        "# Judge validation",
        "",
        "> GENERATED by `make judge-check` (`anchor judge-check`). Do not edit.",
        "",
        f"Run at {utc_now_iso()}. Judge under test: **`{agreement.judge_name}`**, scored against",
        f"{agreement.n} FROZEN `(question, answer, evidence)` triples in",
        "`eval/golden/human_labels.jsonl` — 12 faithful answers, 9 answers with one unsupported",
        "clause, 9 fabrications. Nothing is retrieved or generated here, so this number does not",
        "move when retrieval changes.",
        "",
        "## Agreement",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| n | {agreement.n} |",
        f"| raw agreement | {agreement.raw_agreement:.3f} |",
        f"| Cohen's kappa | {agreement.cohens_kappa:.3f} |",
        "",
        "Kappa is chance-corrected: `(po - pe) / (1 - pe)`, where `pe` comes from the two",
        "raters' marginals. 1.0 is perfect agreement and 0.0 is exactly chance.",
        "",
        "## Confusion matrix (rows = human, columns = judge)",
        "",
        "| human \\ judge | " + " | ".join(f"`{name}`" for name in JUDGE_LABELS) + " | recall |",
        "| --- | " + " | ".join("---:" for _ in JUDGE_LABELS) + " | ---: |",
    ]
    for human_label in JUDGE_LABELS:
        row = agreement.confusion.get(human_label, {})
        cells = " | ".join(str(row.get(judge_label, 0)) for judge_label in JUDGE_LABELS)
        recall = agreement.per_label_recall.get(human_label, 0.0)
        lines.append(f"| `{human_label}` | {cells} | {recall:.3f} |")

    if agreement.run_label_distribution:
        total = sum(agreement.run_label_distribution.values())
        lines.extend(
            [
                "",
                "## The judge's labels on REAL pipeline output",
                "",
                "The confusion matrix above is measured on hand-written triples. This is what the",
                "same judge emitted over the most recent `anchor eval` run — the distribution it",
                "actually operates on. A judge that is constant here is not discriminating",
                "anything here, whatever its kappa on the validation set.",
                "",
                "| judge label | cases |",
                "| --- | ---: |",
            ]
        )
        for label, count in agreement.run_label_distribution.items():
            lines.append(f"| `{label}` | {count} |")
        emitted = [
            name for name in JUDGE_LABELS if agreement.run_label_distribution.get(name)
        ]
        if len(emitted) <= 1:
            lines.extend(
                [
                    "",
                    f"**The judge emitted one label across all {total} cases.** The kappa above is "
                    "therefore measured off-distribution and says nothing about how this judge "
                    "behaves in production.",
                ]
            )

    lines.extend(["", "## Disagreements", ""])
    if not disagreements:
        lines.append("_The judge matched the annotator on every triple._")
    else:
        lines.append(
            f"All {len(disagreements)} of them, with the judge's own rationale. This is the part "
            "worth reading: it is where the judge's failure mode is visible."
        )
        lines.append("")
        for item in disagreements:
            lines.append(f"* {item}")

    if other_notes:
        lines.extend(["", "## Notes", ""])
        for note in other_notes:
            lines.append(f"* {note}")

    lines.extend(
        [
            "",
            "## What this does and does not establish",
            "",
            "**It does not establish anything about the `groundedness` column.** That number is",
            "produced by `anchor.grounding`, the lexical per-sentence verifier, and never passes",
            "through the judge at all. The judge's verdict is returned on every response and",
            "recorded as `CaseResult.judge_label`; it feeds no key in `METRIC_KEYS`, no row of",
            "the README table and no floor in `eval/thresholds.yaml`. What is validated here is",
            "the judge seam, for its own sake.",
            "",
            "These labels were produced by **one annotator — the author of this repository**.",
            "Cohen's kappa above is therefore agreement between the judge and that one person,",
            "and it is **not** agreement between the judge and the truth. There was no second",
            "annotator, so there is no inter-annotator agreement to compare it against, and no",
            "adjudication protocol for the cases where a careful reader might disagree with the",
            "label. A high kappa here means the judge reproduces the author's reading of these",
            "30 triples; it does not mean the judge is right.",
            "",
            "And it is measured on a distribution the judge does not meet in practice: see the",
            "label distribution over the real run above. A validation set on which a classifier",
            "discriminates tells you nothing when the classifier is constant in production.",
            "",
            "It is shipped anyway because the alternative is worse. A judge that has never been",
            "measured this way is not evidence for anything at all, and a measured weak judge can",
            "at least be reported as weak — which is exactly what the note above does when kappa",
            f"falls below {WEAK_KAPPA}.",
            "",
        ]
    )
    return "\n".join(lines)
