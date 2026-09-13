"""Tests for the foundation layer: settings, textutil and types.

These are the contracts every other module in the repo imports, so the
assertions here are deliberately concrete: hand-computed values written out in
comments rather than recomputed with the same function under test. If one of
these breaks, something further up is already wrong.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from anchor import __version__
from anchor.settings import Settings, get_settings
from anchor.textutil import (
    content_tokens,
    count_tokens,
    jaccard,
    make_chunk_id,
    make_doc_id,
    normalize_text,
    overlap_score,
    parse_citation_markers,
    split_sentences,
    strip_citation_markers,
    truncate_to_tokens,
)
from anchor.types import (
    FAILURE_CLASSES,
    JUDGE_LABELS,
    METRIC_KEYS,
    REFUSAL_TYPES,
    AnswerResult,
    AnswerType,
    Candidate,
    Chunk,
    FailureClass,
    JudgeLabel,
)

# =============================================================================
# fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No ANCHOR_* variable leaks in from the developer's shell or from another
    test, and the process-wide settings cache never crosses a test boundary."""
    for key in list(os.environ):
        if key.startswith("ANCHOR_"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def no_config(tmp_path) -> str:
    """A path that does not exist, so the YAML source is skipped entirely and
    Settings.load() yields the pure pydantic defaults."""
    return str(tmp_path / "does-not-exist.yaml")


# =============================================================================
# settings
# =============================================================================


def test_defaults_are_the_documented_zero_key_profile(no_config: str) -> None:
    """The default profile must need no key and touch no network: a hashing
    embedder, an extractive stub LLM and a heuristic judge."""
    s = Settings.load(config_file=no_config)

    assert s.app_name == "anchor"
    assert s.port == 8000
    assert s.seed == 1337

    # Zero-key, zero-network.
    assert s.llm.provider == "stub"
    assert s.llm.api_key is None
    assert s.embedding.provider == "hashing"
    assert s.embedding.dim == 256
    assert s.judge.provider == "heuristic"

    # Retrieval + chunking defaults (the ablation's starting point).
    assert s.retrieval.mode == "hybrid"
    assert s.retrieval.top_k == 5
    assert s.retrieval.rrf_k == 60
    assert s.chunking.chunk_tokens == 512
    assert s.chunking.chunk_overlap == 64
    assert s.chunking.min_chunk_tokens == 80

    # The scale-free abstain gate's one threshold.
    assert s.gate.enabled is True
    assert s.gate.min_lexical_overlap == 0.34
    assert not hasattr(s.gate, "min_top_share"), (
        "min_top_share was removed: it fired on zero of the 43 golden cases in "
        "every retrieval mode, and a gate feature that has never made a "
        "decision is decoration"
    )

    # The measurement layer.
    assert s.eval.worst_n == 12
    assert s.eval.threshold_slack == 0.05
    assert s.eval.ablation_modes == ["bm25", "dense", "hybrid", "hybrid_rerank"]
    assert s.eval.ablation_chunk_configs == [
        [256, 0],
        [256, 64],
        [512, 0],
        [512, 64],
        [1024, 0],
        [1024, 128],
    ]
    # 4 modes x 6 chunk configs = the 24 rows of eval/reports/ablation.md.
    assert len(s.eval.ablation_modes) * len(s.eval.ablation_chunk_configs) == 24

    assert s.agent.max_steps == 4
    assert s.agent.tools == ["search_docs", "calculator", "lookup_metric", "final_answer"]
    assert s.paths.golden_path == "eval/golden/qa.jsonl"
    assert s.paths.thresholds_path == "eval/thresholds.yaml"


def test_load_reads_a_yaml_file(tmp_path) -> None:
    cfg = tmp_path / "custom.yaml"
    cfg.write_text(
        "app_name: anchor\n"
        "retrieval:\n"
        "  mode: bm25\n"
        "  top_k: 9\n"
        "chunking:\n"
        "  chunk_tokens: 256\n",
        encoding="utf-8",
    )

    s = Settings.load(config_file=str(cfg))

    assert s.retrieval.mode == "bm25"
    assert s.retrieval.top_k == 9
    assert s.chunking.chunk_tokens == 256
    # Keys absent from the file keep their defaults (the file is merged, not
    # substituted) — otherwise a one-line override would blank the rest.
    assert s.chunking.chunk_overlap == 64
    assert s.llm.provider == "stub"


def test_env_var_beats_the_yaml_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Precedence is init kwargs > env > .env > YAML > defaults, so a container
    can be repointed at another provider without editing a file."""
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("retrieval:\n  mode: bm25\n  top_k: 9\n", encoding="utf-8")
    monkeypatch.setenv("ANCHOR_RETRIEVAL__MODE", "dense")

    s = Settings.load(config_file=str(cfg))

    assert s.retrieval.mode == "dense"  # env wins
    assert s.retrieval.top_k == 9  # the rest of the file still applies

    # ... and an init kwarg beats the env var in turn.
    s2 = Settings.load(config_file=str(cfg), retrieval={"mode": "hybrid_rerank"})
    assert s2.retrieval.mode == "hybrid_rerank"


def test_fingerprint_is_stable_and_changes_with_the_config(
    no_config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fingerprint is stamped on every Trace and MetricReport, so it has to
    be reproducible across loads and sensitive to anything that moves a number."""
    a = Settings.load(config_file=no_config).fingerprint()
    b = Settings.load(config_file=no_config).fingerprint()
    assert a == b
    assert len(a) == 12
    assert all(c in "0123456789abcdef" for c in a)

    monkeypatch.setenv("ANCHOR_RETRIEVAL__MODE", "bm25")
    changed = Settings.load(config_file=no_config).fingerprint()
    assert changed != a

    monkeypatch.setenv("ANCHOR_CHUNKING__CHUNK_TOKENS", "256")
    changed_again = Settings.load(config_file=no_config).fingerprint()
    assert changed_again != changed


def test_public_dict_never_leaks_a_secret(
    no_config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANCHOR_LLM__API_KEY", "sk-super-secret-value")
    monkeypatch.setenv("ANCHOR_EMBEDDING__API_KEY", "sk-another-secret")

    s = Settings.load(config_file=no_config)
    assert s.llm.api_key is not None
    assert s.llm.api_key.get_secret_value() == "sk-super-secret-value"

    public = s.public_dict()
    assert public["llm"]["api_key"] == "set"
    assert public["embedding"]["api_key"] == "set"

    blob = json.dumps(public)
    assert "sk-super-secret-value" not in blob
    assert "sk-another-secret" not in blob
    # The fingerprint is derived from public_dict, so it cannot leak either.
    assert "sk-" not in s.fingerprint()


def test_public_dict_reports_an_absent_key_as_none(no_config: str) -> None:
    public = Settings.load(config_file=no_config).public_dict()
    assert public["llm"]["api_key"] is None
    assert public["embedding"]["api_key"] is None


def test_resolve_is_relative_to_the_working_directory(no_config: str) -> None:
    s = Settings.load(config_file=no_config)
    assert str(s.resolve(s.paths.db_path)) == "data/index.db"


# =============================================================================
# textutil
# =============================================================================


def test_split_sentences() -> None:
    text = "The p95 latency is 240 ms. SEV-2 pages the on-call. Is that clear?"
    # Three terminators, each followed by whitespace then an uppercase start.
    assert split_sentences(text) == [
        "The p95 latency is 240 ms.",
        "SEV-2 pages the on-call.",
        "Is that clear?",
    ]

    # A lowercase continuation is NOT a boundary, so "90 days. see" stays one
    # sentence — abbreviations and decimals do not shatter a claim.
    assert split_sentences("Retention is 90 days. see the policy.") == [
        "Retention is 90 days. see the policy."
    ]

    # Newlines and runs of whitespace are collapsed before splitting.
    assert split_sentences("One.\n\n   Two.") == ["One.", "Two."]

    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_parse_citation_markers_is_ordered_and_deduplicated() -> None:
    text = "Logs are kept 90 days [S2]. The gateway [S1] and again [S2]."
    # Raw matches are 2, 1, 2 -> first-seen order, duplicates dropped.
    assert parse_citation_markers(text) == ["S2", "S1"]

    assert parse_citation_markers("no markers here") == []
    assert parse_citation_markers("[S10] then [S3]") == ["S10", "S3"]


def test_strip_citation_markers() -> None:
    # Markers removed, then whitespace re-collapsed: "A [S1] b" -> "A b".
    assert strip_citation_markers("A [S1] b [S2] c") == "A b c"

    # The marker-before-punctuation case leaves the punctuation stranded, which
    # is the documented behaviour: stripping is for comparison, not display.
    assert strip_citation_markers("Kept 90 days [S2]. See [S1] too.") == (
        "Kept 90 days . See too."
    )

    assert strip_citation_markers("nothing to strip") == "nothing to strip"


def test_overlap_score() -> None:
    evidence = "Gateway request logs are retained for 90 days."
    # content_tokens drops the stopwords "are"/"for":
    #   claim   -> {logs, retained, 90, days}          (4 tokens)
    #   evidence-> {gateway, request, logs, retained, 90, days}
    # all four present -> 4/4 = 1.0
    assert overlap_score("Logs are retained for 90 days", evidence) == 1.0

    # Swap the number: {logs, retained, 30, days}; "30" is absent -> 3/4 = 0.75
    assert overlap_score("Logs are retained for 30 days", evidence) == 0.75

    # A claim made entirely of stopwords has no content tokens -> 0.0, never a
    # free pass through the grounding check.
    assert content_tokens("the and of") == []
    assert overlap_score("the and of", evidence) == 0.0

    assert overlap_score("quantum tunnelling", evidence) == 0.0


def test_jaccard() -> None:
    question = "How long are logs retained?"
    source = "Logs are retained for 90 days."
    # content tokens: {long, logs, retained} and {logs, retained, 90, days}
    #   intersection = {logs, retained}                  -> 2
    #   union        = {long, logs, retained, 90, days}  -> 5
    #   2 / 5 = 0.4  (this is the metric the golden builder caps at <= 0.5)
    assert jaccard(question, source) == pytest.approx(0.4)

    assert jaccard(source, source) == 1.0
    assert jaccard("", "") == 0.0
    assert jaccard("alpha beta", "gamma delta") == 0.0


def test_count_tokens_counts_punctuation_and_scales_by_1_3() -> None:
    assert count_tokens("") == 0

    # "Retention is 90 days ." -> 5 matches; 5 * 1.3 = 6.5; Python rounds half
    # to even, so int(round(6.5)) == 6.
    assert count_tokens("Retention is 90 days.") == 6

    # "SEV" "-" "2" -> 3 matches; 3 * 1.3 = 3.9 -> 4.
    assert count_tokens("SEV-2") == 4

    # A single word can never be zero tokens.
    assert count_tokens("a") == 1

    # Monotonic in length, which is what the packer's budget arithmetic assumes.
    assert count_tokens("one two three four five") > count_tokens("one two")


def test_truncate_to_tokens_respects_the_budget_on_a_word_boundary() -> None:
    text = "The gateway retains request logs for ninety days before deletion occurs."

    assert truncate_to_tokens(text, 0) == ""
    assert truncate_to_tokens(text, -5) == ""

    # Under budget: returned unchanged, not re-joined or re-spaced.
    assert truncate_to_tokens(text, 1000) == text

    for budget in (1, 3, 5, 8, 12):
        out = truncate_to_tokens(text, budget)
        assert count_tokens(out) <= budget
        # Always a word-boundary prefix of the original.
        assert text.startswith(out)


def test_normalize_text_is_the_one_span_normaliser() -> None:
    # Case, punctuation and whitespace all collapse, so a gold answer_span can
    # be matched inside a chunk regardless of how the chunker cut the markdown.
    assert normalize_text("  Retention:  90 DAYS!! \n") == "retention 90 days"
    assert normalize_text("SEV-2") == "sev 2"
    assert normalize_text("") == ""
    # The property the eval harness depends on: a span normalises to a
    # substring of its containing text's normalisation.
    span = "retained for 90 days"
    doc = "Gateway request logs are **retained for 90 days**, then purged."
    assert normalize_text(span) in normalize_text(doc)


def test_make_chunk_id_is_deterministic_and_content_addressed() -> None:
    a = make_chunk_id("data-retention-policy", 0, 512)
    b = make_chunk_id("data-retention-policy", 0, 512)
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)

    # Any of the three inputs changing changes the id.
    assert make_chunk_id("data-retention-policy", 0, 513) != a
    assert make_chunk_id("data-retention-policy", 1, 512) != a
    assert make_chunk_id("oncall-runbook", 0, 512) != a


def test_make_doc_id() -> None:
    assert make_doc_id("data-retention-policy.md") == "data-retention-policy"
    assert make_doc_id("metrics.csv") == "metrics"
    assert make_doc_id("sub/dir/Gateway Overview.md") == "sub-dir-gateway-overview"
    assert make_doc_id("postmortem_2025_06_index__drift.md") == "postmortem-2025-06-index-drift"


# =============================================================================
# types
# =============================================================================


def _chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="data-retention-policy",
        ordinal=0,
        text="Gateway request logs are retained for 90 days.",
        char_start=0,
        char_end=46,
    )


def test_candidate_score_falls_back_rerank_rrf_dense_bm25() -> None:
    chunk = _chunk()

    # 1. rerank_score wins over everything, even a zero value.
    top = Candidate(chunk=chunk, rerank_score=0.0, rrf_score=9.9, dense_score=8.8, bm25_score=7.7)
    assert top.score == 0.0

    # 2. no rerank -> a NON-ZERO rrf_score wins.
    fused = Candidate(chunk=chunk, rrf_score=0.032, dense_score=8.8, bm25_score=7.7)
    assert fused.score == 0.032

    # 3. no rerank and rrf is zero (single-leg dense run) -> dense_score.
    dense = Candidate(chunk=chunk, dense_score=0.61, bm25_score=7.7)
    assert dense.score == 0.61

    # 4. bm25-only run -> bm25_score. This is the ablation BASELINE mode, so it
    #    has to produce a usable score with no other leg populated.
    lexical = Candidate(chunk=chunk, bm25_score=3.14)
    assert lexical.score == 3.14

    # Nothing populated at all is 0.0, never None and never an exception.
    assert Candidate(chunk=chunk).score == 0.0

    # The convenience accessor other modules sort and de-duplicate by.
    assert lexical.chunk_id == "c1"


def test_answer_result_abstained_is_true_for_every_refusal_type() -> None:
    for answer_type in (
        AnswerType.REFUSED_NO_EVIDENCE,
        AnswerType.REFUSED_LOW_SUPPORT,
        AnswerType.REFUSED_POLICY,
    ):
        result = AnswerResult(answer="...", answer_type=answer_type)
        assert result.abstained is True
        assert answer_type in REFUSAL_TYPES

    for answer_type in (AnswerType.GROUNDED, AnswerType.PARTIAL):
        result = AnswerResult(answer="...", answer_type=answer_type)
        assert result.abstained is False
        assert answer_type not in REFUSAL_TYPES

    assert len(REFUSAL_TYPES) == 3


def test_models_forbid_unknown_fields() -> None:
    """extra="forbid" everywhere: a typo'd field is a loud error, not a key that
    silently vanishes from a committed report artifact."""
    with pytest.raises(ValidationError):
        AnswerResult(answer="x", answer_type=AnswerType.GROUNDED, groundednes=1.0)


def test_metric_keys_membership() -> None:
    assert len(METRIC_KEYS) == 18
    assert len(set(METRIC_KEYS)) == len(METRIC_KEYS)

    # Retrieval quality — the ablation table and the CI gate read these.
    for key in ("recall_at_5", "recall_at_10", "precision_at_5", "mrr", "ndcg_at_10"):
        assert key in METRIC_KEYS

    # Answer quality and citation quality.
    for key in (
        "exact_match",
        "token_f1",
        "citation_precision",
        "citation_recall",
        "groundedness",
    ):
        assert key in METRIC_KEYS

    # The unanswerable slice: refusal vs confident fabrication, plus the cost of
    # over-refusing on questions that ARE answerable.
    for key in (
        "refusal_rate_unanswerable",
        "fabrication_rate_unanswerable",
        "false_refusal_rate_answerable",
        "abstention_accuracy",
    ):
        assert key in METRIC_KEYS

    for key in ("tool_choice_accuracy", "latency_p50_ms", "latency_p95_ms", "n_errors"):
        assert key in METRIC_KEYS


def test_failure_classes_membership() -> None:
    assert FAILURE_CLASSES == (
        "retrieval_miss",
        "wrong_chunk_retrieved",
        "context_ignored_by_generator",
        "genuinely_unanswerable",
    )
    # The tuple and the enum can never drift apart.
    assert tuple(c.value for c in FailureClass) == FAILURE_CLASSES
    assert len(FAILURE_CLASSES) == 4


def test_judge_labels_membership() -> None:
    assert JUDGE_LABELS == ("grounded", "partial", "ungrounded")
    assert tuple(label.value for label in JudgeLabel) == JUDGE_LABELS
    assert len(JUDGE_LABELS) == 3


def test_version_is_exported() -> None:
    assert __version__ == "0.1.0"
