"""The bounded tool-agent behind POST /v1/agent.

Four tools, two planners and a loop whose every exit is typed.

The design constraint that shapes this module: an agent is only interesting if
it is *bounded*. Every way out of `AgentLoop.run` is enumerated — a planner
`final`, a `final_answer` tool call, a repeated (tool, args) pair, the step
limit, the wall clock, the token budget — and each one lands in the SAME
synthesis path, `ctx.answer_service.answer(...)`, so the grounding, refusal and
judge contract that governs POST /v1/query governs POST /v1/agent too. There is
no second, laxer answering path hiding behind the agent.

The second constraint: honesty about provenance. A number produced by the
calculator or by lookup_metric has no document citation and never will, so it is
folded into the result as a Claim with `supported=False`. The agent is not
allowed to launder a computed value into a cited one.

Dependency direction: this module sits beside `service` (context -> {service,
agent}) and imports downward only. `check_question` is imported from service
because the agent must apply the input policy BEFORE it spends any budget on
tools; service never imports agent, so this stays one-way.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import operator
import re
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from anchor.context import AppContext
from anchor.errors import BudgetExceeded, ConfigError, ToolError
from anchor.llm import render_plan_messages
from anchor.obs import TraceRecorder, get_logger, log_event
from anchor.service import check_question
from anchor.settings import Settings
from anchor.textutil import sha256_text, truncate_to_tokens
from anchor.types import (
    AgentResult,
    AnswerType,
    ChatMessage,
    Claim,
    LLMProvider,
    PlannedStep,
    Planner,
    Tool,
    ToolStep,
)

logger = get_logger(__name__)

# Observation strings are fed back to a planner and, for computed values, into a
# Claim. Both want a bounded string, so every tool truncates to this.
MAX_OBSERVATION_CHARS = 900
SEARCH_SNIPPET_CHARS = 400
NO_RESULTS = "NO_RESULTS"
# A tool value is spliced into the prose only when it is short enough to read as
# part of an answer; longer ones stay on the ToolStep and on their Claim.
ANSWER_FOLD_MAX_CHARS = 200
UNGROUNDED_TOOL_VALUE_NOTE = (
    "No supporting passage was retrieved, so this value comes from a tool and is "
    "not grounded in any indexed document."
)


# ==============================================================================
# Tool argument models
# ==============================================================================


class SearchDocsArgs(BaseModel):
    query: str = Field(description="Natural-language search query over the indexed corpus.")
    k: int = Field(default=5, ge=1, le=20, description="How many context blocks to return.")


class CalculatorArgs(BaseModel):
    expression: str = Field(
        description="A pure arithmetic expression, e.g. '(120 + 45) / 2'. "
        "Only + - * / // % ** and round/abs/min/max are available."
    )


class LookupMetricArgs(BaseModel):
    metric: str = Field(
        description="Column name from corpus/metrics.csv. Use 'list' to see the available names."
    )
    month: str | None = Field(
        default=None,
        description="Optional period filter, e.g. '2025-06' for one month or '2025' for a year.",
    )


class FinalAnswerArgs(BaseModel):
    answer: str = Field(description="The answer to return to the caller.")


# ==============================================================================
# Tools
# ==============================================================================


class SearchDocsTool:
    """Retrieval as a tool: the SAME Retriever the /v1/query path uses.

    The agent gets no privileged retrieval. If the abstain gate rejects the
    query the tool says NO_RESULTS rather than handing the planner a pile of
    weak chunks it will happily summarise anyway.
    """

    name: str = "search_docs"
    description: str = (
        "Search the indexed corpus and return the packed context blocks, "
        "or NO_RESULTS when nothing passes the abstain gate."
    )
    args_model: type[BaseModel] = SearchDocsArgs

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx

    def run(self, args: BaseModel) -> str:
        if not isinstance(args, SearchDocsArgs):  # pragma: no cover - guarded by the loop
            raise ToolError("search_docs was called with the wrong argument model")
        result = self._ctx.retriever.retrieve(args.query, top_k=args.k)
        if not result.gate.passed or not result.blocks:
            return NO_RESULTS
        lines: list[str] = []
        for block in result.blocks:
            snippet = " ".join(block.text.split())[:SEARCH_SNIPPET_CHARS]
            section = block.heading_path or block.title or block.doc_id
            lines.append(f"[{block.marker}] {section}: {snippet}")
        return _clip("\n".join(lines))


class CalculatorTool:
    """Arithmetic over an AST allowlist. Never `eval`, never `exec`.

    The threat here is not hypothetical: a planner is a text generator, and a
    text generator will eventually emit `__import__("os").system(...)`. So the
    expression is parsed, walked node by node against an allowlist, and rejected
    on the first thing that is not arithmetic. Depth, literal magnitude and
    exponent caps stop the other failure mode — a syntactically innocent
    expression like `[1]*10**9` or `9**9**9` that costs the process its memory.
    """

    name: str = "calculator"
    description: str = (
        "Evaluate a pure arithmetic expression (+ - * / // % **, round/abs/min/max). "
        "Rejects anything that is not arithmetic."
    )
    args_model: type[BaseModel] = CalculatorArgs

    # Wide enough for an aggregate over a two-year monthly series; the real
    # protections are the node allowlist and the depth/literal/exponent caps.
    MAX_EXPRESSION_CHARS = 600
    MAX_DEPTH = 64
    MAX_LITERAL = 1e12
    MAX_EXPONENT = 32

    _BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _CALLS: dict[str, Callable[..., Any]] = {
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
    }

    def run(self, args: BaseModel) -> str:
        if not isinstance(args, CalculatorArgs):  # pragma: no cover - guarded by the loop
            raise ToolError("calculator was called with the wrong argument model")
        expression = args.expression.strip()
        if not expression:
            raise ToolError("calculator requires a non-empty expression")
        if len(expression) > self.MAX_EXPRESSION_CHARS:
            raise ToolError(
                f"expression is {len(expression)} characters; "
                f"the limit is {self.MAX_EXPRESSION_CHARS}"
            )
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"could not parse expression {expression!r}: {exc.msg}") from exc
        value = self._eval(tree.body, depth=0)
        # The exact expression is preserved on ToolStep.args; a long one is
        # elided here so the observation stays readable as an answer fragment.
        shown = expression if len(expression) <= 80 else f"{expression[:77]}..."
        return f"{shown} = {_format_number(value)}"

    # -- the allowlisted evaluator ------------------------------------------
    def _eval(self, node: ast.AST, depth: int) -> Any:
        if depth > self.MAX_DEPTH:
            raise ToolError(f"expression nests deeper than {self.MAX_DEPTH} levels")

        if isinstance(node, ast.Constant):
            # bool is a subclass of int; reject it so `True + 1` is not arithmetic.
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError(f"only numeric literals are allowed, got {node.value!r}")
            if abs(node.value) > self.MAX_LITERAL:
                raise ToolError(f"literal {node.value!r} exceeds the magnitude cap")
            # int stays int: round(x, 2) rejects a float second argument, and
            # coercing everything to float would break it.
            return node.value

        if isinstance(node, ast.UnaryOp):
            unary_op = self._UNARY_OPS.get(type(node.op))
            if unary_op is None:
                raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")
            return unary_op(self._eval(node.operand, depth + 1))

        if isinstance(node, ast.BinOp):
            binary_op = self._BIN_OPS.get(type(node.op))
            if binary_op is None:
                raise ToolError(f"unsupported operator: {type(node.op).__name__}")
            left = self._eval(node.left, depth + 1)
            right = self._eval(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > self.MAX_EXPONENT:
                raise ToolError(f"exponent {right} exceeds the cap of {self.MAX_EXPONENT}")
            try:
                return binary_op(left, right)
            except ZeroDivisionError as exc:
                raise ToolError("division by zero") from exc
            except (OverflowError, ValueError) as exc:
                raise ToolError(f"arithmetic error: {exc}") from exc

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self._CALLS:
                raise ToolError("only round(), abs(), min() and max() may be called")
            if node.keywords:
                raise ToolError("keyword arguments are not allowed")
            call_args = [self._eval(arg, depth + 1) for arg in node.args]
            if not call_args:
                raise ToolError(f"{node.func.id}() needs at least one argument")
            try:
                return self._CALLS[node.func.id](*call_args)
            except (TypeError, ValueError) as exc:
                raise ToolError(f"{node.func.id}() rejected its arguments: {exc}") from exc

        raise ToolError(f"unsupported expression element: {type(node).__name__}")


class LookupMetricTool:
    """Reads corpus/metrics.csv with the stdlib csv module. No pandas, no magic.

    The column layout is discovered rather than assumed: the first column whose
    name looks like a period becomes the index and every other column is a
    metric. That keeps the tool working if the corpus CSV gains a column.
    """

    name: str = "lookup_metric"
    description: str = (
        "Look up a metric from the monthly metrics CSV. Omit `month` for the full series, "
        "pass '2025' for a year or '2025-06' for one month. Use metric='list' for the names."
    )
    args_model: type[BaseModel] = LookupMetricArgs

    def __init__(self, settings: Settings) -> None:
        self._path = settings.resolve(settings.paths.metrics_csv)

    def run(self, args: BaseModel) -> str:
        if not isinstance(args, LookupMetricArgs):  # pragma: no cover - guarded by the loop
            raise ToolError("lookup_metric was called with the wrong argument model")
        period_field, metric_fields, rows = _read_metrics_csv(self._path)

        requested = args.metric.strip()
        if requested.lower() in {"", "list", "metrics", "names"}:
            return "available metrics: " + ", ".join(metric_fields)

        field = _match_metric(requested, metric_fields)
        if field is None:
            raise ToolError(
                f"unknown metric {requested!r}; available metrics: {', '.join(metric_fields)}"
            )

        selected = rows
        if args.month:
            wanted = args.month.strip().lower()
            selected = [r for r in rows if r.get(period_field, "").strip().lower() == wanted]
            if not selected:
                selected = [
                    r for r in rows if r.get(period_field, "").strip().lower().startswith(wanted)
                ]
            if not selected:
                periods = [r.get(period_field, "") for r in rows]
                span = f"{periods[0]}..{periods[-1]}" if periods else "none"
                raise ToolError(f"no row for period {args.month!r}; the CSV covers {span}")

        pairs = [f"{r.get(period_field, '?')}={r.get(field, '')}" for r in selected]
        return _clip(f"{field}: " + ", ".join(pairs))


class FinalAnswerTool:
    """The loop's termination signal. Returns the answer verbatim.

    It exists as a real tool rather than a magic string so the planner has one
    uniform output shape and `tool_specs_text` can advertise how to stop.
    """

    name: str = "final_answer"
    description: str = "Finish the episode and return this answer to the caller."
    args_model: type[BaseModel] = FinalAnswerArgs

    def run(self, args: BaseModel) -> str:
        if not isinstance(args, FinalAnswerArgs):  # pragma: no cover - guarded by the loop
            raise ToolError("final_answer was called with the wrong argument model")
        return args.answer


def build_tools(ctx: AppContext) -> dict[str, Tool]:
    """Construct exactly the tools named in settings.agent.tools, in that order."""
    available: dict[str, Tool] = {
        "search_docs": SearchDocsTool(ctx),
        "calculator": CalculatorTool(),
        "lookup_metric": LookupMetricTool(ctx.settings),
        "final_answer": FinalAnswerTool(),
    }
    selected: dict[str, Tool] = {}
    for name in ctx.settings.agent.tools:
        if name not in available:
            raise ConfigError(
                f"agent.tools names unknown tool {name!r}; "
                f"available tools are: {', '.join(sorted(available))}"
            )
        selected[name] = available[name]
    return selected


def tool_specs_text(tools: dict[str, Tool]) -> str:
    """Compact tool catalogue for the LLM planner prompt."""
    lines: list[str] = []
    for tool in tools.values():
        schema = tool.args_model.model_json_schema()
        props = json.dumps(schema.get("properties", {}), separators=(",", ":"), sort_keys=True)
        required = ", ".join(schema.get("required", [])) or "none"
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  args={props} required={required}")
    return "\n".join(lines)


# ==============================================================================
# Planners
# ==============================================================================

# Phrases that mean "the caller wants a computed number", independent of whether
# a literal expression is present in the question.
_ARITHMETIC_PHRASES = ("how much", "total of", "difference between", "sum of", "average of")
# Aggregates the calculator can actually express once lookup_metric has the data.
_AGGREGATE_RE = re.compile(r"\b(total|sum|average|mean|combined)\b")
# A run of digits and arithmetic punctuation, e.g. "120 + 45" or "(3*4)/2".
_EXPRESSION_RE = re.compile(r"[0-9][0-9.,\s+\-*/%()]*[0-9)]")
# Unit suffixes carried by column names: p95_latency_ms is about "p95 latency".
_UNIT_TOKENS = frozenset(
    {"ms", "s", "sec", "secs", "pct", "percent", "usd", "eur", "gbp", "count", "n", "num"}
)
# Evidence that a question is asking about a metric SERIES rather than merely
# using the column's word in a sentence.
_TEMPORAL_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b\d{4}-\d{2}\b|\bper month\b|\bmonthly\b|\beach month\b"
    r"|\bover time\b|\btrend\b|\bjanuary|february|march|april|may|june|july|august"
    r"|september|october|november|december\b"
)
# "2025-06" and "2025-06-01" look like subtraction to the regex above.
_DATEISH_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# The right-hand side of lookup_metric's "period=value" pairs.
_SERIES_VALUE_RE = re.compile(r"=(-?\d+(?:\.\d+)?)")
_MEAN_RE = re.compile(r"\b(average|mean)\b")


class RulesPlanner:
    """The deterministic default: no model, no temperature, no surprises.

    This is why the zero-key demo and the agent tests are reproducible. The
    rules are evaluated in a fixed order and every branch sets `reason`, so a
    trace explains itself without anyone having to guess what a model was
    thinking.
    """

    name: str = "rules"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._metrics_path = settings.resolve(settings.paths.metrics_csv)

    def next_step(
        self, question: str, history: list[ToolStep], tools: dict[str, Tool]
    ) -> PlannedStep:
        lowered = question.lower()
        attempted = {step.tool for step in history}
        succeeded = {step.tool for step in history if step.ok}

        literal_expression = _extract_expression(question)
        needs_compute = bool(literal_expression) or any(p in lowered for p in _ARITHMETIC_PHRASES)
        metric = self._metric_in_question(lowered)

        # (1) Everything the question needs has been gathered -> answer.
        if "search_docs" in succeeded:
            compute_done = not needs_compute or "calculator" in succeeded
            metric_done = metric is None or "lookup_metric" in succeeded
            if compute_done and metric_done:
                return PlannedStep(
                    action="final",
                    reason="search_docs succeeded and every required computation is done",
                )

        # (2) A number has to be computed and the calculator has not run yet.
        if "calculator" in tools and "calculator" not in attempted:
            expression = literal_expression or self._derive_expression(lowered, history)
            if expression:
                return PlannedStep(
                    action="tool",
                    tool="calculator",
                    args={"expression": expression},
                    reason=(
                        "the question asks for a computed value and an expression is available"
                    ),
                )

        # (3) The question names a column of metrics.csv.
        if metric and "lookup_metric" in tools and "lookup_metric" not in attempted:
            period = _extract_period(question)
            args: dict[str, Any] = {"metric": metric}
            if period:
                # "in 2025" must narrow the series, or an average over it is
                # silently an average over the wrong span.
                args["month"] = period
            return PlannedStep(
                action="tool",
                tool="lookup_metric",
                args=args,
                reason=f"the question names the {metric!r} metric series",
            )

        # (4) Nothing has been retrieved yet — always ground before answering.
        if "search_docs" in tools and "search_docs" not in attempted:
            return PlannedStep(
                action="tool",
                tool="search_docs",
                args={"query": question, "k": self._settings.retrieval.top_k},
                reason="no retrieval has happened yet; ground the answer first",
            )

        # (5) Nothing left to try.
        return PlannedStep(action="final", reason="no rule applies; synthesising what we have")

    def _metric_in_question(self, lowered: str) -> str | None:
        """First metrics.csv column the question is actually ASKING about.

        Naive substring matching is too eager: a column called `requests` would
        fire on "the total of 120 + 45 requests", which is arithmetic and has
        nothing to do with the series. So a column matches when every one of its
        distinctive tokens (unit suffixes stripped) appears as a word, and a
        column distinctive in only ONE token additionally needs a temporal cue —
        a year, a month, "monthly" — that says a series is wanted.
        """
        temporal = _TEMPORAL_RE.search(lowered) is not None
        for field in _metric_field_names(self._metrics_path):
            parts = re.split(r"[^a-z0-9]+", field.lower())
            # The literal column name only counts when it is a distinctive
            # identifier; a one-word column like `requests` is just English.
            if len(parts) > 1 and field.lower() in lowered:
                return field
            core = [t for t in parts if t and t not in _UNIT_TOKENS]
            if not core:
                continue
            if not all(re.search(rf"\b{re.escape(token)}\b", lowered) for token in core):
                continue
            if len(core) >= 2 or temporal:
                return field
        return None

    def _derive_expression(self, lowered: str, history: list[ToolStep]) -> str:
        """Build a sum/average expression from a successful lookup_metric step.

        No mechanical rule can turn 'the median' into + - * /, so only the
        aggregates arithmetic can actually express are derived here.
        """
        if not _AGGREGATE_RE.search(lowered):
            return ""
        for step in reversed(history):
            if step.tool != "lookup_metric" or not step.ok:
                continue
            # lookup_metric emits "<metric>: <period>=<value>, ...", so read the
            # right-hand sides. Scraping every number would swallow the period
            # labels and produce nonsense.
            values = _SERIES_VALUE_RE.findall(step.observation)
            if len(values) < 2:
                return ""
            total = "(" + " + ".join(values) + ")"
            expression = f"{total} / {len(values)}" if _MEAN_RE.search(lowered) else total
            # Never propose a call that is already known to exceed the cap.
            if len(expression) > CalculatorTool.MAX_EXPRESSION_CHARS:
                return ""
            return expression
        return ""


class LLMPlanner:
    """Planning via a provider, with parse failure treated as a normal outcome.

    A model that emits malformed JSON must not take the process with it, so the
    contract here is: one call, one repair retry that shows the model its own
    parse error, then a typed give-up that lets the loop synthesise from
    whatever it already retrieved.
    """

    name: str = "llm"

    def __init__(self, settings: Settings, llm: LLMProvider) -> None:
        self._settings = settings
        self._llm = llm

    def next_step(
        self, question: str, history: list[ToolStep], tools: dict[str, Tool]
    ) -> PlannedStep:
        messages = render_plan_messages(question, history, tool_specs_text(tools))
        step, error = self._attempt(messages)
        if step is not None:
            return step

        # One repair retry: hand the model its own parse error back.
        messages = list(messages) + [
            ChatMessage(
                role="user",
                content=(
                    f"Your previous reply could not be parsed: {error}. "
                    "Reply with ONLY a JSON object: "
                    '{"action":"tool"|"final","tool":<name or null>,'
                    '"args":{},"reason":<short string>}'
                ),
            )
        ]
        step, error = self._attempt(messages)
        if step is not None:
            return step

        log_event(logger, logging.WARNING, "planner_parse_failure", error=error)
        return PlannedStep(action="final", reason="planner_parse_failure")

    def _attempt(self, messages: list[ChatMessage]) -> tuple[PlannedStep | None, str]:
        try:
            response = self._llm.complete(
                messages, temperature=0.0, max_tokens=self._settings.llm.max_tokens
            )
        except Exception as exc:  # a provider failure is a planning failure, not a crash
            return None, f"provider call failed: {exc}"
        try:
            payload = _parse_json_object(response.text)
        except ValueError as exc:
            return None, str(exc)
        try:
            return PlannedStep.model_validate(payload), ""
        except ValidationError as exc:
            return None, f"not a valid PlannedStep: {exc.errors()[:2]}"


def get_planner(ctx: AppContext) -> Planner:
    """RulesPlanner unless the config asks for the LLM planner."""
    if ctx.settings.agent.planner == "llm":
        return LLMPlanner(ctx.settings, ctx.llm)
    return RulesPlanner(ctx.settings)


# ==============================================================================
# The loop
# ==============================================================================


class AgentLoop:
    """A tool loop with an enumerated set of exits and one synthesis path."""

    def __init__(self, *, ctx: AppContext, tools: dict[str, Tool], planner: Planner) -> None:
        self._ctx = ctx
        self._tools = tools
        self._planner = planner
        self._settings = ctx.settings

    def run(
        self, question: str, *, include_trace: bool = False, max_steps: int | None = None
    ) -> AgentResult:
        # The input policy applies BEFORE any budget is spent on tools.
        check_question(question, self._settings)

        agent_cfg = self._settings.agent
        # A caller may lower the step budget but never raise it above the config.
        limit = agent_cfg.max_steps
        if max_steps is not None:
            limit = max(1, min(int(max_steps), agent_cfg.max_steps))

        recorder = self._ctx.answer_service.new_recorder(question)
        recorder.note(f"agent planner={self._planner.name} max_steps={limit}")
        started = time.perf_counter()
        history: list[ToolStep] = []
        seen: set[str] = set()
        budget_exhausted = False

        try:
            for step_number in range(1, limit + 1):
                self._check_budgets(started, recorder)
                planned = self._plan(question, history)

                if planned.action == "final":
                    recorder.note(f"planner chose final at step {step_number}: {planned.reason}")
                    break

                tool = self._tools.get(planned.tool or "")
                if tool is None:
                    history.append(
                        self._failed_step(
                            step_number,
                            planned.tool or "<none>",
                            planned.args,
                            f"UNKNOWN_TOOL: {planned.tool!r} is not enabled",
                            recorder,
                        )
                    )
                    continue

                fingerprint = _step_fingerprint(tool.name, planned.args)
                if fingerprint in seen:
                    recorder.note(f"loop detected on {tool.name}; forcing final")
                    break
                seen.add(fingerprint)

                step = self._invoke(step_number, tool, planned.args, recorder)
                history.append(step)

                if tool.name == "final_answer" and step.ok:
                    recorder.note("final_answer tool called; ending the episode")
                    break
            else:
                # The for-loop ran to its limit without anyone choosing to stop.
                budget_exhausted = True
                recorder.note(f"step budget of {limit} exhausted without a final step")
        except BudgetExceeded as exc:
            # Budgets are an expected outcome, not a 500.
            budget_exhausted = True
            recorder.note(f"budget exhausted: {exc}")

        return self._synthesise(
            question,
            history=history,
            recorder=recorder,
            include_trace=include_trace,
            budget_exhausted=budget_exhausted,
        )

    # -- internals -----------------------------------------------------------
    def _plan(self, question: str, history: list[ToolStep]) -> PlannedStep:
        """A planner that raises is a planner that is done."""
        try:
            return self._planner.next_step(question, list(history), self._tools)
        except Exception as exc:  # a bad planner must not kill the request
            log_event(logger, logging.WARNING, "planner_error", error=str(exc))
            return PlannedStep(action="final", reason=f"planner raised {type(exc).__name__}")

    def _check_budgets(self, started: float, recorder: TraceRecorder) -> None:
        agent_cfg = self._settings.agent
        elapsed = time.perf_counter() - started
        if elapsed > agent_cfg.wall_clock_budget_s:
            raise BudgetExceeded(
                f"agent wall clock budget of {agent_cfg.wall_clock_budget_s}s exceeded"
            )
        used = recorder.trace.usage.total_tokens
        if used > agent_cfg.max_total_tokens:
            raise BudgetExceeded(
                f"agent token budget of {agent_cfg.max_total_tokens} exceeded ({used} used)"
            )

    def _invoke(
        self, step_number: int, tool: Tool, raw_args: dict[str, Any], recorder: TraceRecorder
    ) -> ToolStep:
        """Validate, time and run one tool call. Never raises."""
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            # Invalid arguments are an observation the planner can learn from,
            # not a crash. This is the single most common real failure mode.
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()[:3]
            )
            return self._failed_step(
                step_number, tool.name, raw_args, f"INVALID_ARGS: {detail}", recorder
            )

        start = time.perf_counter()
        try:
            observation = tool.run(args)
            ok, error = True, None
        except ToolError as exc:
            observation, ok, error = f"TOOL_ERROR: {exc}", False, str(exc)
        except Exception as exc:  # an unforeseen tool bug is still an observation
            observation, ok, error = f"TOOL_ERROR: {exc}", False, f"{type(exc).__name__}: {exc}"
        duration_ms = (time.perf_counter() - start) * 1000.0

        # tool_timeout_s is enforced cooperatively: every tool here is local and
        # bounded, so an overrun is reported rather than pre-emptively killed —
        # a hard kill would cost a thread per call and buy nothing.
        if duration_ms > self._settings.agent.tool_timeout_s * 1000.0:
            recorder.note(f"{tool.name} took {duration_ms:.0f}ms, over the per-tool timeout")

        step = ToolStep(
            step=step_number,
            tool=tool.name,
            args=_jsonable(args.model_dump()),
            ok=ok,
            observation=_clip(observation),
            duration_ms=duration_ms,
            error=error,
        )
        recorder.record_tool_step(step)
        return step

    def _failed_step(
        self,
        step_number: int,
        tool_name: str,
        raw_args: dict[str, Any],
        observation: str,
        recorder: TraceRecorder,
    ) -> ToolStep:
        step = ToolStep(
            step=step_number,
            tool=tool_name,
            args=_jsonable(raw_args),
            ok=False,
            observation=observation,
            duration_ms=0.0,
            error=observation,
        )
        recorder.record_tool_step(step)
        return step

    def _synthesise(
        self,
        question: str,
        *,
        history: list[ToolStep],
        recorder: TraceRecorder,
        include_trace: bool,
        budget_exhausted: bool,
    ) -> AgentResult:
        """One synthesis path, shared with POST /v1/query.

        The RAG answer governs grounding, refusal and the judge verdict. Tool
        values are then folded in as explicitly UNSUPPORTED claims: a computed
        number has no document citation and pretending otherwise would be the
        exact dishonesty this repo is built to avoid.
        """
        result = self._ctx.answer_service.answer(
            question, include_trace=include_trace, recorder=recorder
        )

        answer = result.answer
        claims = list(result.claims)
        answer_type = result.answer_type
        refusal_reason = result.refusal_reason

        # Two SEPARATE decisions, and conflating them was a real bug: whether to
        # RECORD a computed value, and whether to SPLICE it into the prose.
        #
        # The record is unconditional. Every successful calculator/lookup_metric
        # step becomes an UNSUPPORTED claim, always — that is the invariant this
        # module is built on, and making it depend on what the prose happens to
        # contain is how a computed number goes missing from the result entirely.
        #
        # The splice is conditional, and it is conditioned on the tool's RESULT,
        # not on any number in the observation. Testing "does any summand appear
        # in the answer?" meant one coincidental match — the retrieved sentence
        # mentioning the December figure that is also the last term of the mean —
        # suppressed the whole fold, and the question was answered with a premise.
        computed = [
            step for step in history if step.ok and step.tool in {"calculator", "lookup_metric"}
        ]
        prefix: list[str] = []
        for position, step in enumerate(computed):
            text = step.observation
            recorder.note(f"folded {step.tool} value in with no citation (supported=False)")
            # `position`, not len(prefix): a long observation is claimed but not
            # spliced, and the claims must still keep the order the tools ran in.
            claims.insert(position, Claim(text=text, citations=[], supported=False))
            # Only short ones are spliced: "120 + 45 = 165" reads as part of an
            # answer, a 24-row series does not.
            if len(text) <= ANSWER_FOLD_MAX_CHARS and _result_missing_from(answer, step.tool, text):
                prefix.append(text)

        if prefix and answer_type in {
            AnswerType.REFUSED_NO_EVIDENCE,
            AnswerType.REFUSED_LOW_SUPPORT,
        }:
            # Retrieval found nothing, but a tool did produce a real value. The
            # honest label is partial: part of this is computed and uncited, and
            # none of it is document-grounded. Say so instead of pasting the
            # computed number in front of a refusal sentence.
            answer_type = AnswerType.PARTIAL
            refusal_reason = None
            answer = " ".join(prefix + [UNGROUNDED_TOOL_VALUE_NOTE]).strip()
            recorder.note("retrieval refused but a tool value exists; downgraded to partial")
        elif prefix:
            answer = " ".join(prefix + [answer]).strip()

        trace = result.trace
        if trace is None and include_trace:
            trace = recorder.trace

        return AgentResult(
            answer=answer,
            answer_type=answer_type,
            claims=claims,
            citations=list(result.citations),
            groundedness=result.groundedness,
            judge=result.judge,
            refusal_reason=refusal_reason,
            trace_id=result.trace_id or recorder.trace_id,
            latency_ms=result.latency_ms,
            usage=result.usage,
            tool_steps=history,
            steps_used=len(history),
            budget_exhausted=budget_exhausted,
            planner=self._planner.name,
            trace=trace,
        )


# ==============================================================================
# Module helpers
# ==============================================================================


def _clip(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Bound an observation so one tool cannot blow the planner's context."""
    if len(text) <= limit:
        return text
    return truncate_to_tokens(text[: limit + 40], max(1, limit // 5)) + " ..."


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    """Tool args land in a Trace that is serialised to JSON, so keep them plain."""
    return json.loads(json.dumps(data, default=str))


def _step_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    """Hash of (tool, sorted args) — the loop-detection key."""
    blob = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(f"{tool_name}|{blob}")[:16]


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a ```json fence and surrounding prose."""
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in the reply")
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _extract_expression(question: str) -> str:
    """Longest arithmetic-looking run in the question, or ''."""
    best = ""
    for match in _EXPRESSION_RE.finditer(question):
        chunk = match.group(0).strip()
        if not any(op in chunk for op in "+-*/%"):
            continue
        cleaned = chunk.replace(",", "").strip()
        if _DATEISH_RE.match(cleaned):
            continue  # "2025-06" is a period label, not a subtraction
        if len(_NUMBER_RE.findall(cleaned)) < 2:
            continue
        if len(cleaned) > len(best):
            best = cleaned
    return best


def _extract_period(question: str) -> str:
    """'2025-06' or '2025' from a question, whichever is present. '' if neither."""
    match = re.search(r"\b(20\d{2}-\d{2})\b", question)
    if match:
        return match.group(1)
    match = re.search(r"\b(20\d{2})\b", question)
    return match.group(1) if match else ""


def _result_value(tool: str, observation: str) -> str:
    """The value a tool step actually PRODUCED, as a string.

    A calculator observation is "<expression> = <result>", so the result is the
    right-hand side of the last ``=`` — every other number in it is an input.
    A lookup_metric observation is the series itself, so the whole string is the
    result and there is nothing to split off.
    """
    if tool == "calculator" and "=" in observation:
        return observation.rsplit("=", 1)[1].strip()
    return observation.strip()


def _result_missing_from(answer: str, tool: str, observation: str) -> bool:
    """True when the tool's RESULT does not already appear in the answer.

    Deliberately not "none of the observation's numbers appear": the summands of
    a mean are all in the observation, and one of them turning up by coincidence
    in a retrieved sentence must not suppress the computed mean.
    """
    value = _result_value(tool, observation)
    if not value:
        return False
    return value not in answer


def _read_metrics_csv(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    """(period column, metric columns, rows) from the metrics CSV."""
    if not path.is_file():
        raise ToolError(f"metrics CSV not found at {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [{k: (v or "") for k, v in row.items() if k} for row in reader]
    except OSError as exc:
        raise ToolError(f"could not read metrics CSV at {path}: {exc}") from exc
    if not fieldnames:
        raise ToolError(f"metrics CSV at {path} has no header row")
    period_field = next(
        (f for f in fieldnames if any(w in f.lower() for w in ("month", "date", "period", "year"))),
        fieldnames[0],
    )
    metric_fields = [f for f in fieldnames if f != period_field]
    if not metric_fields:
        raise ToolError(f"metrics CSV at {path} has no metric columns")
    return period_field, metric_fields, rows


def _metric_field_names(path: Path) -> list[str]:
    """Metric column names, or [] — the planner must never crash on a bad CSV."""
    try:
        return _read_metrics_csv(path)[1]
    except ToolError:
        return []


def _match_metric(requested: str, fields: list[str]) -> str | None:
    """Case- and punctuation-insensitive column match."""

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    target = norm(requested)
    for field in fields:
        if norm(field) == target:
            return field
    for field in fields:
        if target and target in norm(field):
            return field
    return None
