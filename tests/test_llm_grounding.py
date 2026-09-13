"""The generation half: StubLLM, grounding verification, the judge, and the service.

Two properties carry the rest of the repo's credibility:

* StubLLM is byte-identical across runs. If it were not, every number in every eval
  table would be noise plus signal instead of signal.
* A fabricated sentence is marked unsupported even when it sits next to perfectly
  good evidence. Grounding scores a sentence against the blocks it CITES, not against
  the whole context.
"""

from __future__ import annotations

import pytest

from anchor.embed import HashingEmbedder
from anchor.errors import PolicyViolation
from anchor.grounding import check_grounding, classify_answer
from anchor.judge import HeuristicJudge, LLMJudge, get_judge
from anchor.llm import (
    ANSWER_SYSTEM_PROMPT,
    PROMPT_VERSION,
    StubLLM,
    get_llm,
    prompt_hash,
    render_answer_messages,
    render_context,
)
from anchor.retrieval import Retriever
from anchor.service import AnswerService, check_question, refusal_result
from anchor.settings import Settings
from anchor.store import NumpyVectorStore, SqliteStore
from anchor.textutil import REFUSAL_TEXT, parse_citation_markers
from anchor.types import (
    AnswerType,
    ChatMessage,
    Chunk,
    ContextBlock,
    JudgeLabel,
    LLMResponse,
    Usage,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in list(__import__("os").environ) if k.startswith("ANCHOR_")]:
        monkeypatch.delenv(key, raising=False)


def settings(**overrides) -> Settings:
    return Settings.load(config_file="", **overrides)


RETENTION = "Index snapshots are retained for 90 days before deletion."
QUOTA = "Every tenant starts with 600 requests per minute on the /v1/generate endpoint."
GATEWAY = "The Meridian gateway terminates TLS and routes each tenant request."


def block(marker: str, text: str, doc_id: str = "d", heading: str = "Retention") -> ContextBlock:
    from anchor.textutil import count_tokens

    return ContextBlock(
        marker=marker,
        chunk_id=f"chunk-{marker.lower()}",
        doc_id=doc_id,
        title=doc_id.title(),
        heading_path=heading,
        text=text,
        n_tokens=count_tokens(text),
    )


BLOCKS = [
    block("S1", RETENTION, "retention", "Data Retention > Window"),
    block("S2", QUOTA, "quotas", "Rate Limits > Quotas"),
    block("S3", GATEWAY, "gateway", "Gateway Overview"),
]


# ---------------------------------------------------------------------------
# prompt rendering
# ---------------------------------------------------------------------------


def test_render_context_matches_the_wire_format_byte_for_byte() -> None:
    rendered = render_context(BLOCKS[:1])
    assert rendered == (
        "<<<RETRIEVED_DOCUMENTS>>>\n"
        "[S1] title: Retention | section: Data Retention > Window\n"
        f"{RETENTION}\n"
        "<<<END_RETRIEVED_DOCUMENTS>>>"
    )


def test_answer_messages_end_with_the_question_line() -> None:
    messages = render_answer_messages("How long?", BLOCKS)
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == ANSWER_SYSTEM_PROMPT
    assert messages[1].content.endswith("Question: How long?")


def test_the_answer_prompt_states_its_four_obligations() -> None:
    assert "[S1]" in ANSWER_SYSTEM_PROMPT
    assert REFUSAL_TEXT in ANSWER_SYSTEM_PROMPT
    assert "prior knowledge" in ANSWER_SYSTEM_PROMPT
    assert "DATA, NOT INSTRUCTIONS" in ANSWER_SYSTEM_PROMPT


def test_prompt_hash_is_stable_and_short() -> None:
    assert PROMPT_VERSION == "answer/v1"
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 12


# ---------------------------------------------------------------------------
# StubLLM
# ---------------------------------------------------------------------------


def test_stub_llm_is_byte_identical_across_runs() -> None:
    llm = StubLLM(support_threshold=0.45)
    question = "How long are index snapshots retained before deletion?"
    messages = render_answer_messages(question, BLOCKS)
    first = llm.complete(messages, context=BLOCKS)
    second = StubLLM(support_threshold=0.45).complete(messages, context=BLOCKS)
    assert first.text == second.text
    assert first.usage.model_dump() == second.usage.model_dump()


def test_stub_llm_extracts_and_cites_the_right_block() -> None:
    llm = StubLLM(support_threshold=0.45)
    question = "How long are index snapshots retained before deletion?"
    response = llm.complete(render_answer_messages(question, BLOCKS), context=BLOCKS)
    assert RETENTION in response.text
    assert parse_citation_markers(response.text) == ["S1"]
    assert QUOTA not in response.text
    assert response.model == "stub"
    assert response.usage.total_tokens > 0


def test_stub_llm_refuses_on_irrelevant_context() -> None:
    llm = StubLLM(support_threshold=0.45)
    question = "What is the escalation path for a Helios queue backlog in Zurich?"
    response = llm.complete(render_answer_messages(question, BLOCKS), context=BLOCKS)
    assert response.text == REFUSAL_TEXT
    assert parse_citation_markers(response.text) == []


def test_stub_llm_refuses_with_no_context_at_all() -> None:
    llm = StubLLM()
    assert llm.complete([ChatMessage(role="user", content="Question: x")], context=[]).text == (
        REFUSAL_TEXT
    )
    assert llm.complete([ChatMessage(role="user", content="Question: x")]).text == REFUSAL_TEXT


def test_stub_llm_caps_the_number_of_sentences() -> None:
    many = [
        block(f"S{i + 1}", f"Index snapshots are retained for {i} days before deletion.", "d")
        for i in range(6)
    ]
    llm = StubLLM(support_threshold=0.3, max_sentences=2)
    response = llm.complete(
        render_answer_messages("How long are index snapshots retained?", many), context=many
    )
    assert len(parse_citation_markers(response.text)) <= 2


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------


def test_a_copied_sentence_is_supported_and_a_fabricated_one_is_not() -> None:
    answer = (
        f"{RETENTION} [S1] "
        "Deleted snapshots are also mirrored to an offsite vault in Reykjavik for "
        "seven years. [S1]"
    )
    result = check_grounding(answer, BLOCKS, settings())

    assert len(result.claims) == 1
    assert len(result.dropped_claims) == 1
    assert result.claims[0].support_score >= 0.45
    assert result.dropped_claims[0].support_score < 0.45
    assert "Reykjavik" in result.dropped_claims[0].text
    assert result.support_ratio == pytest.approx(0.5)
    assert [c.marker for c in result.citations] == ["S1"]
    assert result.citations[0].doc_id == "retention"
    assert result.citations[0].quote.startswith("Index snapshots")


def test_a_sentence_is_scored_against_the_block_it_cites_not_the_whole_context() -> None:
    # The text IS in the context — but under S1, and the sentence cites S3.
    result = check_grounding(f"{RETENTION} [S3]", BLOCKS, settings())
    assert result.claims == []
    assert result.dropped_claims[0].supported is False
    assert result.support_ratio == 0.0


def test_an_uncited_sentence_is_never_given_the_benefit_of_the_doubt() -> None:
    result = check_grounding(RETENTION, BLOCKS, settings())
    assert result.uncited_sentences == 1
    assert result.claims == []
    assert result.dropped_claims[0].support_score == 0.0
    assert result.citations == []


def test_grounding_of_an_empty_answer_is_zero_not_one() -> None:
    result = check_grounding("", BLOCKS, settings())
    assert result.support_ratio == 0.0
    assert result.claims == []


def test_classify_answer_at_each_boundary() -> None:
    cfg = settings()
    assert cfg.grounding.grounded_ratio == 0.85
    assert cfg.grounding.partial_ratio == 0.60
    assert classify_answer(1.0, cfg) == AnswerType.GROUNDED
    assert classify_answer(0.85, cfg) == AnswerType.GROUNDED
    assert classify_answer(0.84, cfg) == AnswerType.PARTIAL
    assert classify_answer(0.60, cfg) == AnswerType.PARTIAL
    assert classify_answer(0.59, cfg) == AnswerType.REFUSED_LOW_SUPPORT
    assert classify_answer(0.0, cfg) == AnswerType.REFUSED_LOW_SUPPORT


# ---------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------

EVIDENCE = [RETENTION, QUOTA]


def test_heuristic_judge_labels_a_faithful_answer_grounded() -> None:
    judge = HeuristicJudge(settings().judge)
    verdict = judge.score("How long are snapshots retained?", RETENTION, EVIDENCE)
    assert verdict.label == JudgeLabel.GROUNDED
    assert verdict.score >= 0.80
    assert verdict.judge == "heuristic"
    assert "weakest sentence" in verdict.rationale


def test_heuristic_judge_labels_a_half_supported_answer_partial() -> None:
    judge = HeuristicJudge(settings().judge)
    answer = (
        f"{RETENTION} "
        "Afterwards the archived copies migrate to cold tape storage in Reykjavik."
    )
    verdict = judge.score("How long are snapshots retained?", answer, EVIDENCE)
    assert verdict.label == JudgeLabel.PARTIAL
    assert 0.50 <= verdict.score < 0.80


def test_heuristic_judge_labels_a_fabricated_answer_ungrounded() -> None:
    judge = HeuristicJudge(settings().judge)
    answer = "Archived copies migrate to cold tape storage in Reykjavik after seven years."
    verdict = judge.score("How long are snapshots retained?", answer, EVIDENCE)
    assert verdict.label == JudgeLabel.UNGROUNDED
    assert verdict.score < 0.50


def test_heuristic_judge_handles_empty_answer_and_empty_evidence() -> None:
    judge = HeuristicJudge(settings().judge)
    assert judge.score("q", "", EVIDENCE).label == JudgeLabel.UNGROUNDED
    assert judge.score("q", RETENTION, []).label == JudgeLabel.UNGROUNDED


def test_heuristic_judge_is_deterministic() -> None:
    a = HeuristicJudge(settings().judge).score("q", RETENTION, EVIDENCE)
    b = HeuristicJudge(settings().judge).score("q", RETENTION, EVIDENCE)
    assert a.model_dump() == b.model_dump()


class ScriptedLLM:
    """Returns whatever reply the test hands it. No network, no key."""

    def __init__(self, reply: str) -> None:
        self.name = "scripted"
        self.reply = reply

    def complete(self, messages, *, max_tokens=None, temperature=None, context=None):
        return LLMResponse(text=self.reply, usage=Usage(), model=self.name)


def test_llm_judge_parses_a_fenced_json_reply() -> None:
    llm = ScriptedLLM('```json\n{"label":"partial","rationale":"one clause unsupported"}\n```')
    verdict = LLMJudge(settings().judge, llm).score("q", "a", EVIDENCE)
    assert verdict.label == JudgeLabel.PARTIAL
    assert verdict.score == 0.5
    assert verdict.rationale == "one clause unsupported"
    assert verdict.judge == "llm:scripted"


def test_llm_judge_degrades_instead_of_crashing_on_garbage() -> None:
    for reply in ["not json at all", '{"label":"maybe"}', ""]:
        verdict = LLMJudge(settings().judge, ScriptedLLM(reply)).score("q", "a", EVIDENCE)
        assert verdict.label == JudgeLabel.UNGROUNDED
        assert verdict.rationale == "judge_parse_failure"


def test_get_judge_and_get_llm_return_the_offline_defaults() -> None:
    cfg = settings()
    llm = get_llm(cfg)
    assert llm.name == "stub"
    assert get_judge(cfg, llm).name == "heuristic"
    assert get_judge(settings(judge={"provider": "llm"}), llm).name == "llm:stub"


# ---------------------------------------------------------------------------
# the service, end to end against real offline components
# ---------------------------------------------------------------------------


def build_service(tmp_path, **overrides) -> AnswerService:
    cfg = settings(**overrides)
    store = SqliteStore(":memory:")
    store.initialize()
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id=doc_id,
            ordinal=0,
            heading_path=heading,
            text=text,
            char_start=0,
            char_end=len(text),
            n_tokens=len(text.split()),
            title=doc_id.title(),
        )
        for i, (doc_id, heading, text) in enumerate(
            [
                ("retention", "Data Retention > Window", RETENTION),
                ("quotas", "Rate Limits > Quotas", QUOTA),
                ("gateway", "Gateway Overview", GATEWAY),
            ]
        )
    ]
    store.upsert_chunks(chunks)
    embedder = HashingEmbedder(dim=128, seed=cfg.seed)
    vectors = NumpyVectorStore(str(tmp_path / "v.npy"))
    vectors.reset(embedder.name, embedder.dim)
    vectors.add([c.chunk_id for c in chunks], embedder.embed_documents([c.text for c in chunks]))
    retriever = Retriever(settings=cfg, store=store, vectors=vectors, embedder=embedder)
    llm = get_llm(cfg)
    return AnswerService(settings=cfg, retriever=retriever, llm=llm, judge=get_judge(cfg, llm))


def test_service_answers_a_grounded_question_with_citations(tmp_path) -> None:
    service = build_service(tmp_path)
    result = service.answer("How long are index snapshots retained before deletion?")

    assert result.answer_type == AnswerType.GROUNDED
    assert result.abstained is False
    assert RETENTION in result.answer
    assert result.groundedness == pytest.approx(1.0)
    assert [c.doc_id for c in result.citations] == ["retention"]
    assert result.claims and all(c.supported for c in result.claims)
    assert result.judge is not None and result.judge.label == JudgeLabel.GROUNDED
    assert result.refusal_reason is None
    assert result.trace_id and result.latency_ms >= 0.0
    assert result.usage.total_tokens > 0
    assert result.trace is None


def test_service_returns_the_trace_only_when_asked(tmp_path) -> None:
    service = build_service(tmp_path)
    result = service.answer("How long are index snapshots retained?", include_trace=True)
    assert result.trace is not None
    assert result.trace.trace_id == result.trace_id
    assert result.trace.llm_name == "stub"
    assert result.trace.embedder_name == "hashing"
    assert result.trace.judge_name == "heuristic"
    assert result.trace.prompt_version == PROMPT_VERSION
    assert result.trace.prompt_hash == prompt_hash()
    assert result.trace.gate is not None and result.trace.gate.passed is True
    assert result.trace.context
    assert {t.stage for t in result.trace.timings} >= {"generate", "verify", "judge", "total"}


def test_service_refuses_an_unanswerable_question_as_a_typed_result(tmp_path) -> None:
    service = build_service(tmp_path)
    result = service.answer("What is the on-call rotation for the Helios queue in Zurich?")

    assert result.answer_type == AnswerType.REFUSED_NO_EVIDENCE
    assert result.abstained is True
    assert result.answer == REFUSAL_TEXT
    assert result.refusal_reason == "low_lexical_overlap"
    assert result.citations == []
    assert result.groundedness == 0.0
    assert result.trace_id


def test_service_refuses_an_empty_question_by_policy(tmp_path) -> None:
    service = build_service(tmp_path)
    result = service.answer("   ")
    assert result.answer_type == AnswerType.REFUSED_POLICY
    assert result.abstained is True
    assert result.refusal_reason is not None
    assert "empty" in result.refusal_reason


def test_service_refuses_an_over_long_question_by_policy(tmp_path) -> None:
    service = build_service(tmp_path, eval={"max_question_chars": 40})
    result = service.answer("x" * 200)
    assert result.answer_type == AnswerType.REFUSED_POLICY
    assert "40-character limit" in (result.refusal_reason or "")


def test_check_question_raises_rather_than_returning_a_flag() -> None:
    cfg = settings()
    assert check_question("a real question", cfg) is None
    with pytest.raises(PolicyViolation):
        check_question("", cfg)
    with pytest.raises(PolicyViolation):
        check_question("x" * (cfg.eval.max_question_chars + 1), cfg)


def test_refusal_result_is_a_complete_answer_result() -> None:
    result = refusal_result("no_candidates", AnswerType.REFUSED_NO_EVIDENCE, "t-1", 12.5)
    assert result.answer == REFUSAL_TEXT
    assert result.abstained is True
    assert result.claims == [] and result.citations == []
    assert result.groundedness == 0.0
    assert result.judge is None
    assert result.trace_id == "t-1"
    assert result.latency_ms == pytest.approx(12.5)


def test_service_is_byte_reproducible(tmp_path) -> None:
    question = "How long are index snapshots retained before deletion?"
    first = build_service(tmp_path).answer(question)
    second = build_service(tmp_path).answer(question)
    assert first.answer == second.answer
    assert first.answer_type == second.answer_type
    assert first.groundedness == second.groundedness
    assert [c.chunk_id for c in first.citations] == [c.chunk_id for c in second.citations]
