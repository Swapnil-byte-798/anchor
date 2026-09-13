"""HTTP surface tests. Entirely offline: stub LLM, hashing embedder, heuristic judge.

The corpus here is tiny and written inline so the whole file is self-contained
and says out loud what it expects the retriever to find. It is NOT the real
corpus/ — these tests check the API contract, not retrieval quality; that is
what the golden set and the ablation are for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anchor.api import create_app
from anchor.settings import PathSettings, Settings
from anchor.types import JUDGE_LABELS, AnswerType

# ---------------------------------------------------------------------------
# A three-document corpus with deliberately distinct vocabulary per document.
# ---------------------------------------------------------------------------

RETENTION_MD = """# Data Retention Policy

## Scope

This policy covers every log stream produced by the Meridian gateway in the
primary region and in the failover region.

## Retention windows

Gateway request logs are retained for 90 days in the primary region.
Audit logs are retained for 365 days and are never deleted early.
Debug traces are retained for 7 days because they are large and rarely read.

## Deletion

Deletion runs nightly at 02:00 UTC and removes every record older than its
retention window. The deletion job writes a receipt to the audit stream so the
removal is itself auditable. A retention window may only be extended by the
platform owner, and every extension is recorded in the change log before the
next deletion run begins.
"""

RATE_LIMITS_MD = """# Rate Limits And Quotas

## Default limits

The default rate limit for the /v1/generate endpoint is 600 requests per minute
per tenant. Burst traffic above that limit receives HTTP 429 together with a
Retry-After header measured in seconds.

## Quotas

Each tenant receives a monthly quota of 5000000 tokens. A tenant that exceeds
its quota is throttled to 60 requests per minute until the next billing cycle
begins. Quota increases are approved by the capacity owner and take effect at
the start of the following day.

## Monitoring

The gateway exports a quota utilisation gauge every minute, and the capacity
owner reviews that gauge once a week.
"""

ONCALL_MD = """# On-call Runbook

## Severity levels

A SEV-1 incident requires a response within 5 minutes at any hour of the day. A
SEV-2 incident requires a response within 30 minutes during business hours. A
SEV-3 incident is handled on the next business day.

## Escalation

The primary on-call engineer acknowledges the page and escalates to the
secondary on-call engineer when there is no acknowledgement within 10 minutes.
The incident commander role is assumed by the secondary engineer for any SEV-1
incident lasting longer than one hour.

## Communication

Status updates are posted every 30 minutes for a SEV-1 incident and every two
hours for a SEV-2 incident.
"""

METRICS_CSV = """month,p95_latency_ms,error_rate_pct
2025-01,214,0.42
2025-02,209,0.38
2025-03,221,0.51
"""

GROUNDED_QUESTION = "How long are gateway request logs retained?"
# Plausible, on-domain English with zero lexical footprint in the corpus above.
OUT_OF_CORPUS_QUESTION = "What is the orbital eccentricity of the exoplanet Kepler 442b?"


def _write_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "data-retention-policy.md").write_text(RETENTION_MD, encoding="utf-8")
    (corpus / "rate-limits-and-quotas.md").write_text(RATE_LIMITS_MD, encoding="utf-8")
    (corpus / "oncall-runbook.md").write_text(ONCALL_MD, encoding="utf-8")
    (corpus / "metrics.csv").write_text(METRICS_CSV, encoding="utf-8")
    return corpus


def _settings_for(root: Path) -> Settings:
    corpus = root / "corpus"
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    # config_file="" pins the tests to the pydantic defaults: no YAML file is
    # layered in, so a change to config/default.yaml cannot silently move them.
    return Settings.load(
        config_file="",
        paths=PathSettings(
            data_dir=str(data),
            corpus_dir=str(corpus),
            db_path=str(data / "index.db"),
            vectors_path=str(data / "vectors.npy"),
            reports_dir=str(root / "reports"),
            metrics_csv=str(corpus / "metrics.csv"),
        ),
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> object:
    root = tmp_path_factory.mktemp("anchor-api")
    _write_corpus(root)
    app = create_app(_settings_for(root))
    with TestClient(app) as test_client:
        response = test_client.post("/v1/ingest", json={})
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["documents_indexed"] == 3
        assert report["chunks_written"] > 0
        yield test_client


# ---------------------------------------------------------------------------
# Liveness / readiness
# ---------------------------------------------------------------------------


def test_healthz_reports_the_offline_stack(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm"] == "stub"
    assert body["embedder"] == "hashing"
    assert body["dim"] == 256
    assert body["judge"] == "heuristic"
    assert response.headers["X-Request-Id"]


def test_readyz_is_ready_once_the_index_exists(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["documents"] == 3
    assert body["chunks"] > 0


def test_readyz_is_503_while_the_index_is_empty(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    app = create_app(_settings_for(tmp_path))
    with TestClient(app) as empty_client:
        response = empty_client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        # Liveness must stay green: the process is fine, the index is not.
        assert empty_client.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# The query path
# ---------------------------------------------------------------------------


def test_query_returns_a_grounded_answer_with_resolvable_citations(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": GROUNDED_QUESTION})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["answer_type"] in {AnswerType.GROUNDED.value, AnswerType.PARTIAL.value}
    assert "90 days" in body["answer"]
    assert body["citations"], "a grounded answer must cite something"
    assert body["groundedness"] > 0.0
    assert body["trace_id"]

    # The judge ran and produced one of the three labels.
    assert body["judge"] is not None
    assert body["judge"]["label"] in JUDGE_LABELS

    # Every citation must resolve to real, fetchable text. This is the property
    # that makes a citation checkable rather than decorative.
    for citation in body["citations"]:
        chunk = client.get(f"/v1/chunks/{citation['chunk_id']}")
        assert chunk.status_code == 200, citation
        assert chunk.json()["chunk_id"] == citation["chunk_id"]
        assert chunk.json()["doc_id"] == citation["doc_id"]


def test_out_of_corpus_question_refuses_with_a_200(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": OUT_OF_CORPUS_QUESTION})
    # A refusal is a successful response: the gate did its job.
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer_type"] == AnswerType.REFUSED_NO_EVIDENCE.value
    assert body["citations"] == []
    assert body["answer"]


def test_include_trace_returns_candidates_and_a_gate_decision(client: TestClient) -> None:
    response = client.post(
        "/v1/query", json={"question": GROUNDED_QUESTION, "include_trace": True}
    )
    assert response.status_code == 200
    trace = response.json()["trace"]
    assert trace is not None, "include_trace=True must populate the trace"
    assert trace["candidates"], "the trace must record what retrieval considered"
    assert trace["gate"] is not None
    assert trace["gate"]["passed"] is True
    assert 0.0 <= trace["gate"]["lexical_overlap"] <= 1.0
    assert trace["context"], "the packed context is part of the decision record"
    assert trace["embedder_name"] == "hashing"
    assert response.headers["X-Trace-Id"] == trace["trace_id"]


def test_trace_is_absent_by_default(client: TestClient) -> None:
    body = client.post("/v1/query", json={"question": GROUNDED_QUESTION}).json()
    assert body["trace"] is None


# ---------------------------------------------------------------------------
# Errors, all wearing the same envelope
# ---------------------------------------------------------------------------


def test_unknown_chunk_is_a_404_error_envelope(client: TestClient) -> None:
    response = client.get("/v1/chunks/not-a-real-chunk-id")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"error", "detail", "request_id"}
    assert body["request_id"]
    assert "not-a-real-chunk-id" in body["detail"]


def test_invalid_body_is_a_422_error_envelope(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": ""})
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error", "detail", "request_id"}
    assert body["error"] == "RequestValidationError"
    assert body["detail"]


def test_unknown_field_is_rejected(client: TestClient) -> None:
    # QueryRequest forbids extras, so a typo cannot be silently ignored.
    response = client.post("/v1/query", json={"question": "hello", "top_kk": 3})
    assert response.status_code == 422
    assert response.json()["error"] == "RequestValidationError"


# ---------------------------------------------------------------------------
# Ops surface
# ---------------------------------------------------------------------------


def test_metrics_exposes_counters_and_histograms(client: TestClient) -> None:
    client.post("/v1/query", json={"question": GROUNDED_QUESTION})
    body = client.get("/metrics").json()
    assert set(body) == {"counters", "histograms"}
    assert any(key.startswith("http_request_ms") for key in body["histograms"])
    assert any(key.startswith("query_answers") for key in body["counters"])


def test_system_config_is_redacted(client: TestClient) -> None:
    body = client.get("/v1/system/config").json()
    assert body["llm"]["provider"] == "stub"
    assert body["embedding"]["provider"] == "hashing"
    # A SecretStr must never leave the process as a value.
    assert body["llm"]["api_key"] in (None, "set")
    assert body["embedding"]["api_key"] in (None, "set")


def test_there_is_no_trace_endpoint(client: TestClient) -> None:
    # Declared non-goal, asserted so it cannot creep back in.
    paths = client.get("/openapi.json").json()["paths"]
    assert not [p for p in paths if p.rstrip("/").endswith("/traces")]
    assert "/v1/query" in paths and "/v1/agent" in paths
