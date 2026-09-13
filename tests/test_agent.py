"""Agent tests: the bounds, the tools, and the deterministic planner.

An agent is only worth shipping if it is bounded, so the interesting tests here
are the ways it STOPS — the step budget, loop detection, and an invalid tool
call that becomes an observation instead of a crash. All offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from anchor.agent import (
    AgentLoop,
    CalculatorArgs,
    CalculatorTool,
    LookupMetricArgs,
    LookupMetricTool,
    RulesPlanner,
    _result_missing_from,
    _result_value,
    build_tools,
    tool_specs_text,
)
from anchor.context import AppContext, build_context
from anchor.errors import ConfigError, ToolError
from anchor.ingest import ingest_corpus
from anchor.settings import PathSettings, Settings
from anchor.types import PlannedStep, Tool, ToolStep

RETENTION_MD = """# Data Retention Policy

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
"""

METRICS_CSV = """month,p95_latency_ms,error_rate_pct
2025-01,214,0.42
2025-02,209,0.38
2025-03,221,0.51
"""

QUESTION = "How long are gateway request logs retained?"


def _settings_for(root: Path) -> Settings:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "data-retention-policy.md").write_text(RETENTION_MD, encoding="utf-8")
    (corpus / "rate-limits-and-quotas.md").write_text(RATE_LIMITS_MD, encoding="utf-8")
    (corpus / "metrics.csv").write_text(METRICS_CSV, encoding="utf-8")
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
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
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AppContext]:
    root = tmp_path_factory.mktemp("anchor-agent")
    context = build_context(_settings_for(root))
    ingest_corpus(
        context.settings,
        store=context.store,
        vectors=context.vectors,
        embedder=context.embedder,
    )
    try:
        yield context
    finally:
        context.close()


# ---------------------------------------------------------------------------
# Planner doubles. Each one exercises exactly one way out of the loop.
# ---------------------------------------------------------------------------


class _NeverStopsPlanner:
    """Always asks for another DIFFERENT search, so only the budget can stop it."""

    name = "never-stops"

    def __init__(self) -> None:
        self.calls = 0

    def next_step(
        self, question: str, history: list[ToolStep], tools: dict[str, Tool]
    ) -> PlannedStep:
        self.calls += 1
        return PlannedStep(
            action="tool",
            tool="search_docs",
            args={"query": f"{question} attempt {self.calls}", "k": 3},
            reason="deliberately never finishing",
        )


class _RepeatPlanner:
    """Always asks for the IDENTICAL call, so loop detection must stop it."""

    name = "repeat"

    def next_step(
        self, question: str, history: list[ToolStep], tools: dict[str, Tool]
    ) -> PlannedStep:
        return PlannedStep(
            action="tool",
            tool="search_docs",
            args={"query": "retention window", "k": 3},
            reason="the same call every time",
        )


class _ScriptedPlanner:
    """Replays a fixed script, repeating the last step once exhausted."""

    name = "scripted"

    def __init__(self, steps: list[PlannedStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next_step(
        self, question: str, history: list[ToolStep], tools: dict[str, Tool]
    ) -> PlannedStep:
        step = self._steps[min(self.calls, len(self._steps) - 1)]
        self.calls += 1
        return step


# ---------------------------------------------------------------------------
# The loop's exits
# ---------------------------------------------------------------------------


def test_step_budget_is_enforced(ctx: AppContext) -> None:
    planner = _NeverStopsPlanner()
    loop = AgentLoop(ctx=ctx, tools=build_tools(ctx), planner=planner)

    result = loop.run(QUESTION, max_steps=2)

    assert result.steps_used == 2, "the loop must stop at the requested step budget"
    assert result.budget_exhausted is True
    assert planner.calls == 2
    # Budget exhaustion is a typed result, not an exception and not a 500.
    assert result.answer
    assert result.answer_type is not None
    assert result.planner == "never-stops"


def test_max_steps_cannot_exceed_the_configured_ceiling(ctx: AppContext) -> None:
    planner = _NeverStopsPlanner()
    loop = AgentLoop(ctx=ctx, tools=build_tools(ctx), planner=planner)

    result = loop.run(QUESTION, max_steps=99)

    assert result.steps_used == ctx.settings.agent.max_steps
    assert result.budget_exhausted is True


def test_loop_detection_forces_a_final_step(ctx: AppContext) -> None:
    loop = AgentLoop(ctx=ctx, tools=build_tools(ctx), planner=_RepeatPlanner())

    result = loop.run(QUESTION, max_steps=4)

    # The first call runs; the identical second call is detected and ends the
    # episode, so the budget is never reached.
    assert result.steps_used == 1
    assert result.budget_exhausted is False
    assert result.tool_steps[0].tool == "search_docs"
    assert result.answer


def test_invalid_tool_args_become_an_observation_not_a_crash(ctx: AppContext) -> None:
    planner = _ScriptedPlanner(
        [
            PlannedStep(
                action="tool",
                tool="calculator",
                args={"expr": "1 + 1"},  # the field is `expression`
                reason="wrong argument name",
            ),
            PlannedStep(action="final", reason="giving up cleanly"),
        ]
    )
    loop = AgentLoop(ctx=ctx, tools=build_tools(ctx), planner=planner)

    result = loop.run(QUESTION, max_steps=4)

    assert result.steps_used == 1
    failed = result.tool_steps[0]
    assert failed.ok is False
    assert failed.observation.startswith("INVALID_ARGS")
    assert "expression" in failed.observation
    # The loop survived and still produced an answer through the normal path.
    assert result.answer
    assert planner.calls == 2


def test_unknown_tool_is_reported_rather_than_raised(ctx: AppContext) -> None:
    planner = _ScriptedPlanner(
        [
            PlannedStep(action="tool", tool="teleport", args={}, reason="not a real tool"),
            PlannedStep(action="final", reason="done"),
        ]
    )
    loop = AgentLoop(ctx=ctx, tools=build_tools(ctx), planner=planner)

    result = loop.run(QUESTION, max_steps=3)

    assert result.tool_steps[0].ok is False
    assert "UNKNOWN_TOOL" in result.tool_steps[0].observation
    assert result.answer


def test_search_docs_tool_returns_marked_blocks(ctx: AppContext) -> None:
    tools = build_tools(ctx)
    observation = tools["search_docs"].run(tools["search_docs"].args_model(query=QUESTION, k=3))
    assert observation != "NO_RESULTS"
    assert observation.startswith("[S1]")


def test_search_docs_tool_says_no_results_when_the_gate_rejects(ctx: AppContext) -> None:
    tools = build_tools(ctx)
    args = tools["search_docs"].args_model(query="orbital eccentricity of Kepler 442b", k=3)
    assert tools["search_docs"].run(args) == "NO_RESULTS"


# ---------------------------------------------------------------------------
# The rules planner
# ---------------------------------------------------------------------------


def test_rules_planner_routes_arithmetic_to_the_calculator(ctx: AppContext) -> None:
    planner = RulesPlanner(ctx.settings)
    tools = build_tools(ctx)

    step = planner.next_step("What is the total of 120 + 45 requests?", [], tools)

    assert step.action == "tool"
    assert step.tool == "calculator"
    assert "120" in step.args["expression"]
    assert "45" in step.args["expression"]
    assert step.reason


def test_rules_planner_grounds_a_plain_question_first(ctx: AppContext) -> None:
    planner = RulesPlanner(ctx.settings)
    step = planner.next_step(QUESTION, [], build_tools(ctx))
    assert step.action == "tool"
    assert step.tool == "search_docs"


def test_rules_planner_finishes_once_search_has_succeeded(ctx: AppContext) -> None:
    planner = RulesPlanner(ctx.settings)
    history = [ToolStep(step=1, tool="search_docs", ok=True, observation="[S1] ...")]
    step = planner.next_step(QUESTION, history, build_tools(ctx))
    assert step.action == "final"
    assert step.reason


def test_rules_planner_picks_up_a_metric_name(ctx: AppContext) -> None:
    planner = RulesPlanner(ctx.settings)
    step = planner.next_step("What was the p95 latency ms in March?", [], build_tools(ctx))
    assert step.action == "tool"
    assert step.tool == "lookup_metric"
    assert step.args["metric"] == "p95_latency_ms"


# ---------------------------------------------------------------------------
# Folding a computed value in: the RESULT decides, not "any number"
# ---------------------------------------------------------------------------


def test_a_coincidental_summand_no_longer_suppresses_the_computed_value() -> None:
    """The regression: fold on the RESULT, never on "any number in the observation".

    `_value_missing_from` returned True only when NONE of an observation's numbers
    appeared in the answer. The calculator's observation contains every summand,
    so one coincidental match — the retrieved sentence happening to mention the
    December figure, which is also the last term of the mean — suppressed the
    whole fold: the Claim was never inserted, the computed mean was never spliced,
    and an arithmetic question came back answered with a premise sentence,
    labelled grounded at groundedness 1.000.
    """
    observation = "(836 + 828 + 806 + 690) / 4 = 790"
    answer = "By December 2025 the gateway p95 had settled at 690 ms. [S4]"

    # The old rule saw 690 in the answer and concluded "the value is there".
    assert any(number in answer for number in ("836", "828", "806", "690"))
    # The new rule looks only at what the calculator actually PRODUCED.
    assert _result_value("calculator", observation) == "790"
    assert _result_missing_from(answer, "calculator", observation) is True
    # ... and is satisfied only by that result.
    assert _result_missing_from("the mean is 790", "calculator", observation) is False

    # A lookup_metric observation has no single result token: the series IS the
    # result, so the whole string is what gets tested.
    series = "p95_latency_ms: 2025-01=836, 2025-12=690"
    assert _result_value("lookup_metric", series) == series
    assert _result_missing_from(answer, "lookup_metric", series) is True


# ---------------------------------------------------------------------------
# Calculator: the allowlist is the whole point
# ---------------------------------------------------------------------------


def test_calculator_evaluates_arithmetic() -> None:
    tool = CalculatorTool()
    assert tool.run(CalculatorArgs(expression="(120 + 45) / 2")) == "(120 + 45) / 2 = 82.5"
    assert tool.run(CalculatorArgs(expression="round(10 / 3, 2)")).endswith("= 3.33")
    assert tool.run(CalculatorArgs(expression="max(3, 9) * 2")).endswith("= 18")


def test_calculator_rejects_dunder_import() -> None:
    tool = CalculatorTool()
    with pytest.raises(ToolError) as excinfo:
        tool.run(CalculatorArgs(expression='__import__("os").system("ls")'))
    assert "round()" in str(excinfo.value) or "unsupported" in str(excinfo.value)


def test_calculator_rejects_a_memory_bomb() -> None:
    tool = CalculatorTool()
    with pytest.raises(ToolError) as excinfo:
        tool.run(CalculatorArgs(expression="[1]*10**9"))
    assert "List" in str(excinfo.value)


def test_calculator_rejects_a_huge_exponent() -> None:
    tool = CalculatorTool()
    with pytest.raises(ToolError):
        tool.run(CalculatorArgs(expression="9 ** 9 ** 9"))


def test_calculator_rejects_names_and_attributes() -> None:
    tool = CalculatorTool()
    for expression in ("open", "x + 1", "(1).__class__", "print(1)"):
        with pytest.raises(ToolError):
            tool.run(CalculatorArgs(expression=expression))


def test_calculator_reports_division_by_zero_as_a_tool_error() -> None:
    with pytest.raises(ToolError, match="division by zero"):
        CalculatorTool().run(CalculatorArgs(expression="1/0"))


# ---------------------------------------------------------------------------
# lookup_metric
# ---------------------------------------------------------------------------


@pytest.fixture()
def metric_settings(tmp_path: Path) -> Settings:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(METRICS_CSV, encoding="utf-8")
    return Settings.load(config_file="", paths=PathSettings(metrics_csv=str(csv_path)))


def test_lookup_metric_returns_the_whole_series(metric_settings: Settings) -> None:
    observation = LookupMetricTool(metric_settings).run(LookupMetricArgs(metric="p95_latency_ms"))
    assert "2025-01=214" in observation
    assert "2025-03=221" in observation


def test_lookup_metric_filters_by_month(metric_settings: Settings) -> None:
    observation = LookupMetricTool(metric_settings).run(
        LookupMetricArgs(metric="p95_latency_ms", month="2025-02")
    )
    assert "209" in observation
    assert "214" not in observation


def test_lookup_metric_filters_by_year_prefix(metric_settings: Settings) -> None:
    observation = LookupMetricTool(metric_settings).run(
        LookupMetricArgs(metric="error_rate_pct", month="2025")
    )
    assert observation.count("=") == 3


def test_lookup_metric_lists_available_names(metric_settings: Settings) -> None:
    observation = LookupMetricTool(metric_settings).run(LookupMetricArgs(metric="list"))
    assert "p95_latency_ms" in observation
    assert "error_rate_pct" in observation


def test_lookup_metric_rejects_an_unknown_metric(metric_settings: Settings) -> None:
    with pytest.raises(ToolError) as excinfo:
        LookupMetricTool(metric_settings).run(LookupMetricArgs(metric="revenue"))
    message = str(excinfo.value)
    assert "unknown metric" in message
    # The error has to be actionable: it names what IS available.
    assert "p95_latency_ms" in message


def test_lookup_metric_rejects_a_missing_period(metric_settings: Settings) -> None:
    with pytest.raises(ToolError, match="no row for period"):
        LookupMetricTool(metric_settings).run(
            LookupMetricArgs(metric="p95_latency_ms", month="1999-01")
        )


def test_lookup_metric_without_a_csv_is_a_tool_error(tmp_path: Path) -> None:
    settings = Settings.load(
        config_file="", paths=PathSettings(metrics_csv=str(tmp_path / "absent.csv"))
    )
    with pytest.raises(ToolError, match="not found"):
        LookupMetricTool(settings).run(LookupMetricArgs(metric="anything"))


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def test_build_tools_honours_the_configured_tool_list(ctx: AppContext) -> None:
    settings = ctx.settings.model_copy(deep=True)
    settings.agent.tools = ["search_docs", "final_answer"]
    limited = build_tools(
        AppContext(
            settings=settings,
            store=ctx.store,
            vectors=ctx.vectors,
            embedder=ctx.embedder,
            llm=ctx.llm,
            judge=ctx.judge,
            retriever=ctx.retriever,
            answer_service=ctx.answer_service,
        )
    )
    assert list(limited) == ["search_docs", "final_answer"]


def test_build_tools_rejects_an_unknown_tool_name(ctx: AppContext) -> None:
    settings = ctx.settings.model_copy(deep=True)
    settings.agent.tools = ["search_docs", "teleport"]
    with pytest.raises(ConfigError, match="teleport"):
        build_tools(
            AppContext(
                settings=settings,
                store=ctx.store,
                vectors=ctx.vectors,
                embedder=ctx.embedder,
                llm=ctx.llm,
                judge=ctx.judge,
                retriever=ctx.retriever,
                answer_service=ctx.answer_service,
            )
        )


def test_tool_specs_text_advertises_every_tool(ctx: AppContext) -> None:
    specs = tool_specs_text(build_tools(ctx))
    for name in ("search_docs", "calculator", "lookup_metric", "final_answer"):
        assert f"- {name}:" in specs
    assert "expression" in specs  # the schema, not just the name
