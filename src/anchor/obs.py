"""Logging + in-process counters + the TraceRecorder.

Imports settings/types/textutil ONLY, so every layer can record without creating
an import cycle. The Trace is in-memory and returned inline when
``include_trace=True`` — nothing here writes a trace to disk.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from anchor.settings import Settings
from anchor.textutil import new_trace_id, utc_now_iso
from anchor.types import (
    AnswerType,
    Candidate,
    ContextBlock,
    GateDecision,
    StageTiming,
    ToolStep,
    Trace,
    Usage,
)

current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")
current_request_id: ContextVar[str] = ContextVar("current_request_id", default="")

_REDACT_KEYS = ("api_key", "authorization", "password", "secret", "token")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if current_trace_id.get():
            payload["trace_id"] = current_trace_id.get()
        if current_request_id.get():
            payload["request_id"] = current_request_id.get()
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload[key] = "***" if any(r in key.lower() for r in _REDACT_KEYS) else value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings) -> None:
    """Idempotent root-logger setup. Safe from api, cli and tests."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if settings.obs.log_json
        else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.obs.log_level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})


class MetricsRegistry:
    """Tiny thread-safe in-process counters/histograms behind GET /metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._hist: dict[str, list[float]] = {}

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        inner = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
        return f"{name}{{{inner}}}"

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            key = self._key(name, labels)
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            key = self._key(name, labels)
            bucket = self._hist.setdefault(key, [])
            bucket.append(value)
            if len(bucket) > 1000:
                del bucket[: len(bucket) - 1000]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hist: dict[str, dict[str, float]] = {}
            for key, values in self._hist.items():
                ordered = sorted(values)
                n = len(ordered)
                hist[key] = {
                    "count": float(n),
                    "sum": float(sum(ordered)),
                    "min": ordered[0],
                    "max": ordered[-1],
                    "p50": ordered[min(n - 1, int(0.50 * n))],
                    "p95": ordered[min(n - 1, int(0.95 * n))],
                }
            return {"counters": dict(self._counters), "histograms": hist}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._hist.clear()


METRICS = MetricsRegistry()


class TraceRecorder:
    """Collects one in-memory Trace. Never raises: recording must not break a
    request, and nothing is persisted."""

    def __init__(
        self,
        query: str,
        *,
        settings: Settings,
        llm_name: str = "",
        embedder_name: str = "",
        judge_name: str = "",
        trace_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.trace = Trace(
            trace_id=trace_id or new_trace_id(),
            query=query,
            created_at=utc_now_iso(),
            config_fingerprint=settings.fingerprint(),
            llm_name=llm_name,
            embedder_name=embedder_name,
            judge_name=judge_name,
            retrieval_mode=settings.retrieval.mode,
            chunk_tokens=settings.chunking.chunk_tokens,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        self._started = time.perf_counter()
        current_trace_id.set(self.trace.trace_id)

    @property
    def trace_id(self) -> str:
        return self.trace.trace_id

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.trace.timings.append(
                StageTiming(stage=name, duration_ms=(time.perf_counter() - start) * 1000.0)
            )

    def note(self, message: str) -> None:
        self.trace.notes.append(message)

    def record_candidates(self, candidates: list[Candidate], limit: int | None = None) -> None:
        cap = limit if limit is not None else self.settings.retrieval.max_candidates_recorded
        self.trace.candidates = list(candidates[:cap])

    def record_gate(self, gate: GateDecision) -> None:
        self.trace.gate = gate

    def record_context(self, blocks: list[ContextBlock]) -> None:
        self.trace.context = list(blocks)

    def record_prompt(self, version: str, prompt_hash_value: str) -> None:
        self.trace.prompt_version = version
        self.trace.prompt_hash = prompt_hash_value

    def record_tool_step(self, step: ToolStep) -> None:
        self.trace.tool_steps.append(step)

    def add_usage(self, usage: Usage) -> None:
        t = self.trace.usage
        self.trace.usage = Usage(
            prompt_tokens=t.prompt_tokens + usage.prompt_tokens,
            completion_tokens=t.completion_tokens + usage.completion_tokens,
            total_tokens=t.total_tokens + usage.total_tokens,
            cost_usd=round(t.cost_usd + usage.cost_usd, 8),
        )

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000.0

    def finish(self, answer_type: AnswerType, groundedness: float) -> Trace:
        self.trace.answer_type = answer_type
        self.trace.groundedness = round(groundedness, 4)
        self.trace.timings.append(StageTiming(stage="total", duration_ms=self.elapsed_ms()))
        return self.trace
