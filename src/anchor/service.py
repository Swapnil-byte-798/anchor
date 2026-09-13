"""The request path: policy -> retrieve -> gate -> pack -> generate -> verify -> judge.

Two invariants are enforced structurally rather than by convention:

* **Generation without evidence is impossible.** If the gate passes but packing
  yields no blocks, this raises :class:`NoEvidenceError`. There is no code path in
  which the model is asked to answer from an empty context.
* **A refusal is a normal answer.** Every refusal returns a complete
  :class:`AnswerResult` with an ``answer_type`` in ``REFUSAL_TYPES`` and a machine
  readable ``refusal_reason``. Refusals are 200s; they are measured, not errored.
"""

from __future__ import annotations

import logging

from anchor.errors import NoEvidenceError, PolicyViolation
from anchor.grounding import check_grounding, classify_answer
from anchor.llm import PROMPT_VERSION, prompt_hash, render_answer_messages
from anchor.obs import TraceRecorder, get_logger, log_event
from anchor.retrieval import Retriever
from anchor.settings import Settings
from anchor.textutil import REFUSAL_TEXT, strip_citation_markers
from anchor.types import (
    AnswerResult,
    AnswerType,
    Judge,
    LLMProvider,
    Usage,
)

__all__ = ["check_question", "refusal_result", "AnswerService"]

_LOG = get_logger("anchor.service")


def check_question(question: str, settings: Settings) -> None:
    """Input policy. Raises :class:`PolicyViolation`; returns None when acceptable."""
    if not question or not question.strip():
        raise PolicyViolation("the question is empty")
    limit = int(settings.eval.max_question_chars)
    if len(question) > limit:
        raise PolicyViolation(
            f"the question is {len(question)} characters, over the {limit}-character limit"
        )


def refusal_result(
    reason: str, answer_type: AnswerType, trace_id: str, latency_ms: float
) -> AnswerResult:
    """A complete, well-formed refusal. Never a bare string, never an exception."""
    return AnswerResult(
        answer=REFUSAL_TEXT,
        answer_type=answer_type,
        claims=[],
        citations=[],
        groundedness=0.0,
        judge=None,
        refusal_reason=reason,
        trace_id=trace_id,
        latency_ms=round(latency_ms, 3),
        usage=Usage(),
        trace=None,
    )


def _is_refusal_text(text: str) -> bool:
    """True when the generator produced only the shared refusal sentence."""
    stripped = strip_citation_markers(text).strip()
    if stripped == REFUSAL_TEXT:
        return True
    return REFUSAL_TEXT in stripped and not stripped.replace(REFUSAL_TEXT, "").strip()


class AnswerService:
    """Orchestrates one question into one :class:`AnswerResult`."""

    def __init__(
        self, *, settings: Settings, retriever: Retriever, llm: LLMProvider, judge: Judge
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.judge = judge

    def new_recorder(self, question: str) -> TraceRecorder:
        """A trace stamped with the identity of every swappable component."""
        return TraceRecorder(
            question,
            settings=self.settings,
            llm_name=self.llm.name,
            embedder_name=self.retriever.embedder.name,
            judge_name=self.judge.name,
        )

    def answer(
        self,
        question: str,
        *,
        include_trace: bool = False,
        recorder: TraceRecorder | None = None,
        top_k: int | None = None,
    ) -> AnswerResult:
        """Answer a question, or refuse with a typed reason.

        ``recorder`` may be supplied by the agent loop so that a tool-mediated answer
        and its enclosing agent run share ONE trace rather than producing two
        disconnected ones.
        """
        rec = recorder or self.new_recorder(question)

        try:
            check_question(question, self.settings)
        except PolicyViolation as exc:
            rec.note(f"policy: {exc}")
            rec.finish(AnswerType.REFUSED_POLICY, 0.0)
            result = refusal_result(
                str(exc), AnswerType.REFUSED_POLICY, rec.trace_id, rec.elapsed_ms()
            )
            if include_trace:
                result.trace = rec.trace
            log_event(_LOG, logging.INFO, "answer.refused", reason="policy", detail=str(exc))
            return result

        retrieval = self.retriever.retrieve(question, recorder=rec, top_k=top_k)

        if not retrieval.gate.passed:
            rec.note(f"gate: {retrieval.gate.reason}")
            rec.finish(AnswerType.REFUSED_NO_EVIDENCE, 0.0)
            result = refusal_result(
                retrieval.gate.reason,
                AnswerType.REFUSED_NO_EVIDENCE,
                rec.trace_id,
                rec.elapsed_ms(),
            )
            if include_trace:
                result.trace = rec.trace
            log_event(
                _LOG,
                logging.INFO,
                "answer.refused",
                reason=retrieval.gate.reason,
                lexical_overlap=round(retrieval.gate.lexical_overlap, 4),
                top_share=round(retrieval.gate.top_share, 4),
            )
            return result

        blocks = retrieval.blocks
        if not blocks:
            # Structurally impossible to reach with a healthy packer; if it ever is
            # reached, failing loudly beats asking a model to answer from nothing.
            raise NoEvidenceError(
                "the gate passed but the context pack is empty; refusing to generate "
                "without evidence"
            )

        messages = render_answer_messages(question, blocks)
        rec.record_prompt(PROMPT_VERSION, prompt_hash())
        with rec.stage("generate"):
            response = self.llm.complete(
                messages,
                max_tokens=self.settings.llm.max_tokens,
                temperature=self.settings.llm.temperature,
                context=blocks,
            )
        rec.add_usage(response.usage)
        answer_text = response.text.strip()

        if not answer_text or _is_refusal_text(answer_text):
            rec.note("generator declined: no sentence cleared the support threshold")
            rec.finish(AnswerType.REFUSED_LOW_SUPPORT, 0.0)
            result = refusal_result(
                "generator_declined",
                AnswerType.REFUSED_LOW_SUPPORT,
                rec.trace_id,
                rec.elapsed_ms(),
            )
            result.usage = rec.trace.usage
            if include_trace:
                result.trace = rec.trace
            log_event(_LOG, logging.INFO, "answer.refused", reason="generator_declined")
            return result

        with rec.stage("verify"):
            grounding = check_grounding(answer_text, blocks, self.settings)
        answer_type = classify_answer(grounding.support_ratio, self.settings)

        if answer_type == AnswerType.REFUSED_LOW_SUPPORT:
            for dropped in grounding.dropped_claims:
                rec.note(f"dropped unsupported claim ({dropped.support_score}): {dropped.text}")
            rec.finish(answer_type, grounding.support_ratio)
            result = refusal_result(
                "low_grounding_support", answer_type, rec.trace_id, rec.elapsed_ms()
            )
            result.usage = rec.trace.usage
            if include_trace:
                result.trace = rec.trace
            log_event(
                _LOG,
                logging.INFO,
                "answer.refused",
                reason="low_grounding_support",
                support_ratio=grounding.support_ratio,
            )
            return result

        if answer_type == AnswerType.PARTIAL:
            # Return only what the evidence supports. Shipping the dropped clause
            # alongside a "partial" label would hand the caller the exact sentence
            # the verifier just rejected.
            for dropped in grounding.dropped_claims:
                rec.note(f"dropped unsupported claim ({dropped.support_score}): {dropped.text}")
            answer_text = " ".join(claim.text for claim in grounding.claims)

        with rec.stage("judge"):
            verdict = self.judge.score(question, answer_text, [b.text for b in blocks])

        rec.finish(answer_type, grounding.support_ratio)
        result = AnswerResult(
            answer=answer_text,
            answer_type=answer_type,
            claims=grounding.claims,
            citations=grounding.citations,
            # The LEXICAL VERIFIER's support ratio — not the judge's score. The two
            # are separate on purpose and `judge` below is the other one. Note that
            # on the stub profile this number is degenerate: the stub emits verbatim
            # sentences out of the block it cites, so the verifier scores a sentence
            # against its own source and gets 1.0. `anchor eval` re-checks that every
            # run and prints a note when it holds; it is why groundedness is reported
            # but not gated in eval/thresholds.yaml.
            groundedness=round(grounding.support_ratio, 4),
            judge=verdict,
            refusal_reason=None,
            trace_id=rec.trace_id,
            latency_ms=round(rec.elapsed_ms(), 3),
            usage=rec.trace.usage,
            trace=rec.trace if include_trace else None,
        )
        log_event(
            _LOG,
            logging.INFO,
            "answer.ok",
            answer_type=answer_type.value,
            groundedness=result.groundedness,
            citations=len(result.citations),
            judge_label=verdict.label.value,
        )
        return result
