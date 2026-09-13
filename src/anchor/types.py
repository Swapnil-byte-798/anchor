"""ALL shared pydantic models AND all Protocols. One import for every contract.

Every model declares ``model_config = ConfigDict(extra="forbid")`` so a typo in a
field name is a loud error rather than a silently ignored key. Field order and
defaults are part of the contract: other groups construct these models
positionally in tests and serialise them into committed report artifacts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class AnswerType(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    REFUSED_NO_EVIDENCE = "refused_no_evidence"
    REFUSED_LOW_SUPPORT = "refused_low_support"
    REFUSED_POLICY = "refused_policy"


REFUSAL_TYPES: frozenset[AnswerType] = frozenset(
    {
        AnswerType.REFUSED_NO_EVIDENCE,
        AnswerType.REFUSED_LOW_SUPPORT,
        AnswerType.REFUSED_POLICY,
    }
)


class JudgeLabel(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    UNGROUNDED = "ungrounded"


JUDGE_LABELS: tuple[str, ...] = ("grounded", "partial", "ungrounded")


class FailureClass(str, Enum):
    """The four buckets of eval/reports/failures.md. Exactly these, no others."""

    RETRIEVAL_MISS = "retrieval_miss"
    WRONG_CHUNK_RETRIEVED = "wrong_chunk_retrieved"
    CONTEXT_IGNORED_BY_GENERATOR = "context_ignored_by_generator"
    GENUINELY_UNANSWERABLE = "genuinely_unanswerable"


FAILURE_CLASSES: tuple[str, ...] = (
    "retrieval_miss",
    "wrong_chunk_retrieved",
    "context_ignored_by_generator",
    "genuinely_unanswerable",
)

CaseType = Literal["factual", "multi_hop", "arithmetic", "unanswerable"]
Provenance = Literal["mechanical", "entity_substituted", "authored"]
RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class LoadedDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    path: str
    title: str
    text: str
    sha256: str


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    path: str
    title: str
    sha256: str
    n_chunks: int = 0
    ingested_at: str = ""


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    doc_id: str
    ordinal: int
    heading_path: str = ""
    text: str
    char_start: int
    char_end: int
    n_tokens: int = 0
    title: str = ""


class Candidate(BaseModel):
    """A retrieval candidate carrying its provenance through every stage."""

    model_config = ConfigDict(extra="forbid")
    chunk: Chunk
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int | None = None
    selected_by_mmr: bool = False

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def score(self) -> float:
        """The score the gate and the packer read.

        Precedence: rerank_score, else rrf_score if non-zero, else the single-leg
        score (dense then bm25). Guarantees a usable score in EVERY mode.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score:
            return self.rrf_score
        if self.dense_score is not None:
            return self.dense_score
        return self.bm25_score or 0.0


class ContextBlock(BaseModel):
    """One labelled block of the prompt context: [S1], [S2], ..."""

    model_config = ConfigDict(extra="forbid")
    marker: str
    chunk_id: str
    doc_id: str
    title: str = ""
    heading_path: str = ""
    text: str
    n_tokens: int = 0


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    marker: str
    chunk_id: str
    doc_id: str
    title: str = ""
    heading_path: str = ""
    quote: str = ""


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    citations: list[str] = Field(default_factory=list)
    supported: bool = False
    support_score: float = 0.0


class GateDecision(BaseModel):
    """Scale-free by design: both features are ratios, so ONE threshold pair works
    across bm25 / dense / hybrid without retuning. That is what makes the
    ablation comparable across modes."""

    model_config = ConfigDict(extra="forbid")
    passed: bool
    reason: str = ""
    lexical_overlap: float = 0.0
    top_share: float = 0.0
    thresholds: dict[str, float] = Field(default_factory=dict)
    near_misses: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(default_factory=list)
    dropped_claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    support_ratio: float = 0.0
    uncited_sentences: int = 0


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: JudgeLabel
    score: float = 0.0
    rationale: str = ""
    judge: str = ""


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"


class ToolStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    observation: str = ""
    duration_ms: float = 0.0
    error: str | None = None


class PlannedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tool", "final"]
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class StageTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str
    duration_ms: float = 0.0


class Trace(BaseModel):
    """In-memory decision record. Returned inline when include_trace=True; the
    eval harness reads candidates/context off it. NOT persisted anywhere."""

    model_config = ConfigDict(extra="forbid")
    trace_id: str
    query: str
    created_at: str = ""
    config_fingerprint: str = ""
    llm_name: str = ""
    embedder_name: str = ""
    judge_name: str = ""
    retrieval_mode: str = ""
    chunk_tokens: int = 0
    chunk_overlap: int = 0
    prompt_version: str = ""
    prompt_hash: str = ""
    candidates: list[Candidate] = Field(default_factory=list)
    gate: GateDecision | None = None
    context: list[ContextBlock] = Field(default_factory=list)
    tool_steps: list[ToolStep] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    answer_type: AnswerType | None = None
    groundedness: float = 0.0
    notes: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    candidates: list[Candidate] = Field(default_factory=list)
    gate: GateDecision
    blocks: list[ContextBlock] = Field(default_factory=list)


class AnswerResult(BaseModel):
    """POST /v1/query response body."""

    model_config = ConfigDict(extra="forbid")
    answer: str
    answer_type: AnswerType
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    groundedness: float = 0.0
    judge: JudgeVerdict | None = None
    refusal_reason: str | None = None
    trace_id: str = ""
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    trace: Trace | None = None

    @property
    def abstained(self) -> bool:
        return self.answer_type in REFUSAL_TYPES


class AgentResult(BaseModel):
    """POST /v1/agent response body."""

    model_config = ConfigDict(extra="forbid")
    answer: str
    answer_type: AnswerType
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    groundedness: float = 0.0
    judge: JudgeVerdict | None = None
    refusal_reason: str | None = None
    trace_id: str = ""
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    tool_steps: list[ToolStep] = Field(default_factory=list)
    steps_used: int = 0
    budget_exhausted: bool = False
    planner: str = "rules"
    trace: Trace | None = None

    @property
    def abstained(self) -> bool:
        return self.answer_type in REFUSAL_TYPES


class IngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents_seen: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    duration_ms: float = 0.0
    embedder: str = ""
    dim: int = 0
    chunk_tokens: int = 0
    chunk_overlap: int = 0


# ---- API request/response models ---------------------------------------------


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    top_k: int | None = None
    include_trace: bool = False


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    max_steps: int | None = None
    include_trace: bool = False


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus_dir: str | None = None
    force: bool = False


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str
    detail: str = ""
    request_id: str = ""


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    version: str = ""
    llm: str = ""
    embedder: str = ""
    judge: str = ""
    dim: int = 0
    chunks: int = 0
    documents: int = 0


# ---- Evaluation models -------------------------------------------------------


class GoldSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    answer_span: str


class EvalCase(BaseModel):
    """One golden question. `gold` NEVER holds chunk ids — spans are resolved to
    chunks at eval time, which is what keeps the chunking ablation honest."""

    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    type: CaseType
    provenance: Provenance
    answer_aliases: list[str] = Field(default_factory=list)
    gold: list[GoldSpan] = Field(default_factory=list)
    expected_answer_type: AnswerType = AnswerType.GROUNDED
    expected_tool: str | None = None
    stem_overlap: float = 0.0
    source_doc_id: str = ""
    notes: str = ""


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    question: str
    type: str
    provenance: str
    answer: str = ""
    answer_type: AnswerType | None = None
    abstained: bool = False
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    packed_chunk_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_doc_ids: list[str] = Field(default_factory=list)
    cited_doc_ids: list[str] = Field(default_factory=list)
    groundedness: float = 0.0
    judge_label: str | None = None
    exact_match: bool = False
    token_f1: float = 0.0
    gate_passed: bool = True
    gate_lexical_overlap: float = 0.0
    gate_top_share: float = 0.0
    tool_used: str | None = None
    steps_used: int = 0
    latency_ms: float = 0.0
    error: str | None = None


# The EXACT key set of MetricReport.metrics. report.py, ablation.py, gate
# checking and the README table all index these strings; do not add or rename.
METRIC_KEYS: tuple[str, ...] = (
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "mrr",
    "ndcg_at_10",
    "exact_match",
    "token_f1",
    "citation_precision",
    "citation_recall",
    "groundedness",
    "refusal_rate_unanswerable",
    "fabrication_rate_unanswerable",
    "false_refusal_rate_answerable",
    "abstention_accuracy",
    "tool_choice_accuracy",
    "latency_p50_ms",
    "latency_p95_ms",
    "n_errors",
)


class MetricReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    created_at: str
    config_fingerprint: str = ""
    llm_name: str = ""
    embedder_name: str = ""
    judge_name: str = ""
    retrieval_mode: str = ""
    chunk_tokens: int = 0
    chunk_overlap: int = 0
    n_cases: int = 0
    #: How many of ``n_cases`` actually scored ``tool_choice_accuracy`` — i.e.
    #: carry an ``expected_tool``. It is NOT ``n_cases``, and a reader who assumes
    #: it is reads a 4-case sample as a 43-case measurement, so the table prints
    #: this denominator next to the metric.
    tool_choice_n: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    by_type: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_provenance: dict[str, dict[str, float]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AblationRow(BaseModel):
    """One cell of the strategy x chunking grid. bm25 is the baseline: for every
    chunk config, delta_recall_at_5_vs_bm25 is measured against the bm25 row of
    THAT SAME chunk config, and is 0.0 on the baseline row itself."""

    model_config = ConfigDict(extra="forbid")
    mode: str
    chunk_tokens: int
    chunk_overlap: int
    n_cases: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    is_baseline: bool = False
    delta_recall_at_5_vs_bm25: float = 0.0


class FailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    question: str
    type: str
    failure_class: FailureClass
    gold_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    packed_chunk_ids: list[str] = Field(default_factory=list)
    answer: str = ""
    expected_answer_type: str = ""
    got_answer_type: str = ""
    note: str = ""


class HumanLabel(BaseModel):
    """A FROZEN (question, answer, evidence) triple with a human groundedness
    label. Frozen text is why judge validation needs no live run."""

    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    answer: str
    evidence: list[str] = Field(default_factory=list)
    label: JudgeLabel
    note: str = ""


class JudgeAgreement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judge_name: str
    n: int = 0
    raw_agreement: float = 0.0
    cohens_kappa: float = 0.0
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    per_label_recall: dict[str, float] = Field(default_factory=dict)
    #: The judge's label distribution over the most recent golden run, when one is
    #: on disk. A kappa measured on hand-written triples where the judge emits all
    #: three labels says nothing about a judge that emits ONE label in production,
    #: so the operating distribution is printed next to the validation one.
    run_label_distribution: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ThresholdSpec(BaseModel):
    """Parsed eval/thresholds.yaml. Values are GENERATED from a real run."""

    model_config = ConfigDict(extra="forbid")
    min: dict[str, float] = Field(default_factory=dict)
    max: dict[str, float] = Field(default_factory=dict)
    generated_from_run: str = ""
    slack: float = 0.05


class ThresholdCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str
    direction: Literal["min", "max"]
    bound: float
    value: float
    passed: bool


# ---- Protocols ---------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Vectors are float32, L2-normalised: (n, dim) / (dim,)."""

    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


@runtime_checkable
class VectorStore(Protocol):
    def reset(self, embedder_name: str, dim: int) -> None: ...

    def add(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def get(self, chunk_id: str) -> np.ndarray | None: ...

    def count(self) -> int: ...

    def meta(self) -> dict[str, Any]: ...

    def persist(self) -> None: ...

    def load(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class DocumentStore(Protocol):
    def initialize(self) -> None: ...

    def upsert_document(self, doc: Document) -> None: ...

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None: ...

    def get_document(self, doc_id: str) -> Document | None: ...

    def list_documents(self) -> list[Document]: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...

    def list_chunks(self, doc_id: str | None = None) -> list[Chunk]: ...

    def neighbours(self, doc_id: str, ordinal: int, window: int = 1) -> list[Chunk]: ...

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]: ...

    def delete_document(self, doc_id: str) -> int: ...

    def count_documents(self) -> int: ...

    def count_chunks(self) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: list[ContextBlock] | None = None,
    ) -> LLMResponse:
        """Remote providers IGNORE `context` (it is already rendered into the
        messages). The offline StubLLM uses it for deterministic extractive
        answering. Documented, not hidden."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Scores an answer's groundedness against plain evidence strings. Taking
    `list[str]` (not ContextBlocks) is deliberate: it lets judge validation run
    over the frozen human-labelled rows with no retrieval."""

    name: str

    def score(self, question: str, answer: str, evidence: list[str]) -> JudgeVerdict: ...


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    args_model: type[BaseModel]

    def run(self, args: BaseModel) -> str: ...


@runtime_checkable
class Planner(Protocol):
    name: str

    def next_step(
        self, question: str, history: list[ToolStep], tools: dict[str, Tool]
    ) -> PlannedStep: ...
