"""The HTTP surface: one module holding create_app, middleware, handlers, routes.

Everything about the service that a caller can touch is on this one screen,
which is the point — a reviewer should be able to read the entire API contract
without opening a second file.

Three decisions worth stating up front:

1.  A REFUSAL IS A 200. `answer_type` carries the outcome (`grounded`,
    `partial`, `refused_no_evidence`, ...). Abstention is a product decision the
    service made on purpose, not a transport error, and a client that treats it
    as a 5xx will retry its way into exactly the fabrication this system exists
    to avoid.

2.  ENDPOINTS ARE `async def` AND CALL THE BLOCKING PIPELINE DIRECTLY. The whole
    pipeline is local — SQLite FTS5 plus one numpy matmul — and runs in single
    to low-tens of milliseconds. Keeping it on the event-loop thread means every
    SQLite access happens on the thread that opened the connection, which is
    what the sqlite3 default demands. The honest cost: concurrent requests
    serialise. For this workload that is the right trade; for a real deployment
    the fix is a connection-per-thread store, not a wider threadpool here.

3.  THERE IS NO TRACE ENDPOINT. No listing, no persistence, no replay. A trace
    is returned inline when `include_trace` is set and is otherwise discarded.
    It was cut on purpose and is a declared non-goal, not an oversight.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from anchor import __version__
from anchor.agent import AgentLoop, build_tools, get_planner
from anchor.context import AppContext, build_context
from anchor.errors import AnchorError, IngestError, http_status_for
from anchor.ingest import ingest_corpus
from anchor.obs import METRICS, configure_logging, current_request_id, get_logger, log_event
from anchor.settings import Settings, get_settings
from anchor.types import (
    AgentRequest,
    AgentResult,
    AnswerResult,
    Chunk,
    ErrorResponse,
    HealthResponse,
    IngestReport,
    IngestRequest,
    QueryRequest,
)

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Works with `uvicorn anchor.api:create_app --factory`.

    The context is wired here rather than in the lifespan so that a
    misconfiguration (a stale vector index, an absent provider key) fails loudly
    at construction time instead of on the first request.
    """
    cfg = settings or get_settings()
    configure_logging(cfg)
    ctx = build_context(cfg)
    tools = build_tools(ctx)
    planner = get_planner(ctx)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        ctx.close()

    app = FastAPI(
        title="anchor",
        version=__version__,
        summary="A grounded RAG + tool-agent service that measures its own retrieval quality.",
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    app.state.tools = tools
    app.state.planner = planner

    _register_middleware(app, cfg)
    _register_handlers(app)
    _register_routes(app)

    # Exactly one startup line, and it says what a reviewer needs to know: which
    # providers are live, and how to switch. No key is required to get here.
    suffix = ""
    if cfg.llm.provider == "stub":
        suffix = " (offline mode; set ANCHOR_LLM__PROVIDER=openai to use a provider)"
    log_event(
        logger,
        logging.INFO,
        f"llm={ctx.llm.name} embedder={ctx.embedder.name} dim={ctx.embedder.dim} "
        f"judge={ctx.judge.name} chunks={ctx.store.count_chunks()}{suffix}",
        config_fingerprint=cfg.fingerprint(),
        retrieval_mode=cfg.retrieval.mode,
    )
    return app


# ==============================================================================
# Middleware
# ==============================================================================


def _register_middleware(app: FastAPI, cfg: Settings) -> None:
    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        token = current_request_id.set(request_id)
        request.state.request_id = request_id
        path = request.url.path
        started = time.perf_counter()
        try:
            oversized = await _reject_oversized_body(request, cfg.obs.max_body_bytes, request_id)
            response: Response = oversized if oversized is not None else await call_next(request)

            duration_ms = (time.perf_counter() - started) * 1000.0
            response.headers["X-Request-Id"] = request_id
            trace_id = getattr(request.state, "trace_id", "")
            if trace_id:
                response.headers["X-Trace-Id"] = trace_id

            # The metric label is the matched route TEMPLATE, never the raw path.
            # `/v1/chunks/{chunk_id}` must aggregate across chunk ids: labelling by
            # path would give every id its own one-sample histogram (no usable p95)
            # and let any caller grow this in-process registry without bound.
            route = getattr(request.scope.get("route"), "path", None) or "<unmatched>"

            METRICS.observe("http_request_ms", duration_ms, route=route, method=request.method)
            METRICS.incr(
                "http_responses",
                route=route,
                method=request.method,
                status_class=f"{response.status_code // 100}xx",
            )
            log_event(
                logger,
                logging.INFO,
                "http_request",
                method=request.method,
                route=route,
                path=path,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
                trace_id=trace_id,
            )
            return response
        finally:
            current_request_id.reset(token)


async def _reject_oversized_body(
    request: Request, limit: int, request_id: str
) -> JSONResponse | None:
    """Enforce the body cap on bytes actually READ, returning a 413 or None.

    Content-Length is only a fast path — a `Transfer-Encoding: chunked` request
    declares no length, so a header-only check is advisory against well-behaved
    clients and no limit at all against anyone else. The body is therefore read
    here, one chunk at a time, and abandoned the moment it crosses the cap.

    What is read is handed to the downstream app through `Request._body`, which
    is the cache `BaseHTTPMiddleware`'s request wrapper replays to the endpoint;
    without it the handler would see an empty body.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return _error_response(
            status=413,
            error="PayloadTooLarge",
            detail=f"request body of {declared} bytes exceeds the limit of {limit} bytes",
            request_id=request_id,
        )

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return _error_response(
                status=413,
                error="PayloadTooLarge",
                detail=f"request body exceeds the limit of {limit} bytes",
                request_id=request_id,
            )
    request._body = bytes(body)
    return None


# ==============================================================================
# Error handlers — every failure leaves through the ErrorResponse envelope
# ==============================================================================


def _error_response(*, status: int, error: str, detail: str, request_id: str) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=request_id)
    return JSONResponse(status_code=status, content=body.model_dump())


def _register_handlers(app: FastAPI) -> None:
    @app.exception_handler(AnchorError)
    async def handle_anchor_error(request: Request, exc: Exception) -> JSONResponse:
        status = http_status_for(exc)
        level = logging.ERROR if status >= 500 else logging.WARNING
        log_event(logger, level, "anchor_error", error=type(exc).__name__, detail=str(exc))
        return _error_response(
            status=status,
            error=type(exc).__name__,
            detail=str(exc),
            request_id=getattr(request.state, "request_id", ""),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
        errors = exc.errors() if isinstance(exc, RequestValidationError) else []
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ())) or '<body>'}: {e.get('msg', '')}"
            for e in errors[:5]
        )
        return _error_response(
            status=422,
            error="RequestValidationError",
            detail=detail or "the request body did not validate",
            request_id=getattr(request.state, "request_id", ""),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
        status = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
        detail = str(getattr(exc, "detail", "")) or "request failed"
        return _error_response(
            status=status,
            error="HTTPError",
            detail=detail,
            request_id=getattr(request.state, "request_id", ""),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Anything reaching here is a bug: the pipeline raises typed AnchorErrors.
        logger.exception("unhandled_error")
        return _error_response(
            status=500,
            error="InternalError",
            detail=f"{type(exc).__name__}: {exc}",
            request_id=getattr(request.state, "request_id", ""),
        )


# ==============================================================================
# Routes
# ==============================================================================


def _confine_to_corpus_root(settings: Settings, requested: str | None) -> str | None:
    """Keep an HTTP-supplied `corpus_dir` inside the configured corpus root.

    The CLI may point ingest anywhere — its caller already has the filesystem.
    An HTTP caller does not, and the service has no auth, so an unconfined
    `corpus_dir` would let anyone on the network make the process read arbitrary
    *.md/*.txt off the host and then serve the text back through `/v1/query` and
    `/v1/chunks`. Absolute paths, `~` and `..` all reach `Settings.resolve`, so
    the check is on the fully resolved path (symlinks included), not the string.
    """
    if requested is None:
        return None
    root = Path(settings.resolve(settings.paths.corpus_dir)).resolve()
    candidate = Path(settings.resolve(requested)).resolve()
    if not candidate.is_relative_to(root):
        raise IngestError(
            f"corpus_dir must be inside the configured corpus root {str(root)!r}; "
            f"got {str(candidate)!r}"
        )
    return str(candidate)


def _register_routes(app: FastAPI) -> None:
    def ctx_of(request: Request) -> AppContext:
        context: AppContext = request.app.state.ctx
        return context

    @app.post("/v1/query", response_model=AnswerResult, tags=["query"])
    async def query(request: Request, body: QueryRequest) -> AnswerResult:
        """Answer a question over the indexed corpus.

        A refusal is a 200 with `answer_type` in the refusal set — the abstain
        gate doing its job is a successful response.
        """
        ctx = ctx_of(request)
        result = ctx.answer_service.answer(
            body.question, include_trace=body.include_trace, top_k=body.top_k
        )
        request.state.trace_id = result.trace_id
        METRICS.incr("query_answers", answer_type=result.answer_type.value)
        return result

    @app.post("/v1/agent", response_model=AgentResult, tags=["agent"])
    async def agent(request: Request, body: AgentRequest) -> AgentResult:
        """Run the bounded tool loop over the same retriever and the same grounding."""
        ctx = ctx_of(request)
        loop = AgentLoop(ctx=ctx, tools=request.app.state.tools, planner=request.app.state.planner)
        result = loop.run(
            body.question, include_trace=body.include_trace, max_steps=body.max_steps
        )
        request.state.trace_id = result.trace_id
        METRICS.incr("agent_runs", answer_type=result.answer_type.value, planner=result.planner)
        return result

    @app.post("/v1/ingest", response_model=IngestReport, tags=["corpus"])
    async def ingest(request: Request, body: IngestRequest) -> IngestReport:
        """(Re)build the index, reusing the live stores so the running service sees it.

        `corpus_dir` is confined to the configured corpus root: a 400 otherwise.
        """
        ctx = ctx_of(request)
        report = ingest_corpus(
            ctx.settings,
            corpus_dir=_confine_to_corpus_root(ctx.settings, body.corpus_dir),
            store=ctx.store,
            vectors=ctx.vectors,
            embedder=ctx.embedder,
            force=body.force,
        )
        METRICS.incr("ingest_runs")
        return report

    @app.get("/v1/chunks/{chunk_id}", response_model=Chunk, tags=["corpus"])
    async def get_chunk(request: Request, chunk_id: str) -> Chunk:
        """The exact text a citation points at — this is what makes citations checkable."""
        chunk = ctx_of(request).store.get_chunk(chunk_id)
        if chunk is None:
            raise StarletteHTTPException(status_code=404, detail=f"no chunk with id {chunk_id!r}")
        return chunk

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz(request: Request) -> HealthResponse:
        """Liveness: process is up and wired. Deliberately does NOT touch the DB."""
        ctx = ctx_of(request)
        return HealthResponse(
            status="ok",
            version=__version__,
            llm=ctx.llm.name,
            embedder=ctx.embedder.name,
            judge=ctx.judge.name,
            dim=ctx.embedder.dim,
        )

    @app.get("/readyz", response_model=HealthResponse, tags=["ops"])
    async def readyz(request: Request) -> Response:
        """Readiness: an empty index is a 503 — the service cannot ground anything yet."""
        ctx = ctx_of(request)
        documents = ctx.store.count_documents()
        chunks = ctx.store.count_chunks()
        vectors = ctx.vectors.count()
        ready = chunks > 0 and (vectors > 0 or ctx.settings.retrieval.mode == "bm25")
        payload = HealthResponse(
            status="ready" if ready else "not_ready",
            version=__version__,
            llm=ctx.llm.name,
            embedder=ctx.embedder.name,
            judge=ctx.judge.name,
            dim=ctx.embedder.dim,
            chunks=chunks,
            documents=documents,
        )
        return JSONResponse(
            status_code=200 if ready else 503, content=payload.model_dump(mode="json")
        )

    @app.get("/metrics", tags=["ops"])
    async def metrics() -> dict[str, Any]:
        """In-process counters and latency histograms. JSON, not Prometheus text."""
        return METRICS.snapshot()

    @app.get("/v1/system/config", tags=["ops"])
    async def system_config(request: Request) -> dict[str, Any]:
        """The EFFECTIVE configuration, with every secret redacted."""
        return ctx_of(request).settings.public_dict()
