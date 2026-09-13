"""The judge seam — and the honest caveat that comes with it.

WHAT THIS IS NOT. The judge does NOT produce the ``groundedness`` number reported
by this repo. That comes from :mod:`anchor.grounding`, the lexical per-sentence
verifier, and is carried on ``AnswerResult.groundedness``. A judge verdict is
attached to every answer as ``AnswerResult.judge`` and recorded by the eval runner
as ``CaseResult.judge_label``, and it currently feeds no key in
:data:`anchor.types.METRIC_KEYS`, no row of the README table and no CI threshold.
It is a second, independent opinion, reported and validated for its own sake.

A judge that has never been measured against human labels is not evidence — it is
a number-shaped opinion, and reporting one as a quality metric is the single most
common way a RAG project fools itself and its reader. The only thing that converts
a judge from decoration into instrumentation is agreement with labels someone
actually wrote down. This repo therefore ships those labels
(``eval/golden/human_labels.jsonl``, 30 frozen triples including deliberately
fabricated and half-supported answers) and the command that scores the configured
judge against them::

    make judge-check     ->  src/anchor/eval/judge_validation.py

which reports raw agreement, Cohen's kappa and the full 3x3 confusion matrix. Two
residual caveats, stated rather than buried: those labels come from ONE annotator,
so kappa measures judge-versus-author agreement rather than judge-versus-truth;
and on real pipeline output the offline heuristic judge has emitted a single label
for every answered case, so that kappa is measured off-distribution. The
validation report prints the run's label distribution so this is visible.

Both judges take ``evidence`` as ``list[str]`` rather than ``ContextBlock`` objects.
That is a design decision, not laziness: it is precisely what lets validation run
over frozen, hand-labelled text with no retrieval, no index and no live run — so the
judge's score cannot drift when the retriever changes underneath it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from anchor.llm import render_judge_messages
from anchor.settings import JudgeSettings, Settings
from anchor.textutil import overlap_score, split_sentences, strip_citation_markers
from anchor.types import Judge, JudgeLabel, JudgeVerdict, LLMProvider

__all__ = ["HeuristicJudge", "LLMJudge", "get_judge"]

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_LABELS = {label.value: label for label in JudgeLabel}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of a model reply, tolerating a ```json fence."""
    candidate = text.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class HeuristicJudge:
    """The offline default: deterministic, keyless, and therefore validatable.

    Scores each answer sentence by its best lexical overlap against any evidence
    string and averages. It is a weak judge by construction — it cannot detect a
    paraphrase and it cannot detect a true-but-unsupported claim that reuses the
    evidence's vocabulary. Its virtue is that it is fully specified, so its
    disagreements with the human labels are reproducible and reportable rather than
    mysterious. That is the whole argument: a weak judge whose error is measured
    beats a strong judge whose error is assumed.
    """

    def __init__(self, settings: JudgeSettings) -> None:
        self.name = "heuristic"
        self.settings = settings

    def score(self, question: str, answer: str, evidence: list[str]) -> JudgeVerdict:
        sentences = split_sentences(strip_citation_markers(answer))
        if not sentences or not answer.strip():
            return JudgeVerdict(
                label=JudgeLabel.UNGROUNDED,
                score=0.0,
                rationale="empty answer: nothing to support",
                judge=self.name,
            )
        if not evidence:
            return JudgeVerdict(
                label=JudgeLabel.UNGROUNDED,
                score=0.0,
                rationale="no evidence was supplied, so no claim can be supported",
                judge=self.name,
            )

        per_sentence = [
            max((overlap_score(sentence, item) for item in evidence), default=0.0)
            for sentence in sentences
        ]
        mean = sum(per_sentence) / len(per_sentence)
        weakest_index = min(range(len(per_sentence)), key=lambda i: (per_sentence[i], i))
        weakest = sentences[weakest_index]

        if mean >= self.settings.grounded_threshold:
            label = JudgeLabel.GROUNDED
        elif mean >= self.settings.partial_threshold:
            label = JudgeLabel.PARTIAL
        else:
            label = JudgeLabel.UNGROUNDED

        rationale = (
            f"mean sentence support {mean:.2f}; weakest sentence scored "
            f"{per_sentence[weakest_index]:.2f}: {weakest[:120]!r}"
        )
        return JudgeVerdict(
            label=label, score=round(mean, 4), rationale=rationale, judge=self.name
        )


class LLMJudge:
    """A model-backed judge, used only when a provider key is configured.

    An unparseable reply is mapped to ``ungrounded`` with the rationale
    ``judge_parse_failure`` rather than raising: a judge that crashes a request is
    worse than a judge that is conservatively wrong, and the failure is visible in
    the validation confusion matrix instead of hidden in a stack trace.
    """

    def __init__(self, settings: JudgeSettings, llm: LLMProvider) -> None:
        self.name = f"llm:{llm.name}"
        self.settings = settings
        self.llm = llm

    def score(self, question: str, answer: str, evidence: list[str]) -> JudgeVerdict:
        messages = render_judge_messages(question, answer, evidence)
        response = self.llm.complete(messages, temperature=0.0)
        parsed = _extract_json_object(response.text)
        if parsed is None:
            return JudgeVerdict(
                label=JudgeLabel.UNGROUNDED,
                score=0.0,
                rationale="judge_parse_failure",
                judge=self.name,
            )
        raw_label = str(parsed.get("label", "")).strip().lower()
        label = _LABELS.get(raw_label)
        if label is None:
            return JudgeVerdict(
                label=JudgeLabel.UNGROUNDED,
                score=0.0,
                rationale="judge_parse_failure",
                judge=self.name,
            )
        score = {JudgeLabel.GROUNDED: 1.0, JudgeLabel.PARTIAL: 0.5, JudgeLabel.UNGROUNDED: 0.0}[
            label
        ]
        rationale = str(parsed.get("rationale", "")).strip()[:500]
        return JudgeVerdict(label=label, score=score, rationale=rationale, judge=self.name)


def get_judge(settings: Settings, llm: LLMProvider) -> Judge:
    """Build the configured judge. ``heuristic`` is the offline default."""
    if settings.judge.provider == "llm":
        return LLMJudge(settings.judge, llm)
    return HeuristicJudge(settings.judge)
