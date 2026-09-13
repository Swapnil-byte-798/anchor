"""The `anchor` command line — the service AND the whole measurement layer.

Five of the commands below are the reason this repo exists:

    anchor eval         the golden set, with the unanswerable slice broken out
    anchor ablate       4 retrieval strategies x 6 chunk configs, bm25 = baseline
    anchor taxonomy     the worst cases, each assigned one failure class
    anchor judge-check  the configured judge against 30 frozen human labels
    anchor gate         the regression gate CI runs; exits 1 on a breach

Two rules this module exists to enforce:

*   NO NUMBER IS EVER TYPED BY HAND. `anchor thresholds` GENERATES
    eval/thresholds.yaml from a real run (observed minus the configured slack).
    A human editing that file by hand is the failure mode the whole design is
    built to prevent, so the only way to change it is to run the command.

*   FAILURES ARE ONE LINE, NOT A TRACEBACK. Every command wraps its body in
    `_guard()`: an AnchorError prints as `error: <Type>: <message>` and exits 1.
    A stack trace on a missing index tells a reviewer nothing they want to know.

Eval imports are deliberately lazy, inside the commands that need them, so
`anchor ingest` and `anchor serve` neither pay for nor depend on the eval
package importing cleanly.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Any, Iterator

import typer
from rich.console import Console
from rich.table import Table

from anchor import __version__
from anchor.agent import AgentLoop, build_tools, get_planner
from anchor.context import AppContext, build_context
from anchor.errors import AnchorError, EvalError
from anchor.ingest import ingest_corpus
from anchor.settings import Settings
from anchor.types import AgentResult, AnswerResult, CaseResult, Trace

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="anchor — a grounded RAG + agent service that measures its own retrieval quality.",
)
console = Console()
err_console = Console(stderr=True)

# Set by the top-level --config option; "" means "use the settings default".
_CONFIG_FILE: str = ""


# ==============================================================================
# Plumbing
# ==============================================================================


def _raw(text: str) -> None:
    """Print untrusted text: answers contain [S1] markers, which are rich markup."""
    console.print(text, markup=False, highlight=False)


def _load_settings(**overrides: Any) -> Settings:
    return Settings.load(_CONFIG_FILE or None, **overrides)


@contextmanager
def _guard() -> Iterator[None]:
    """Turn an AnchorError into one clean line and exit 1. No traceback."""
    try:
        yield
    except AnchorError as exc:
        err_console.print(f"[red]error[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from None


@contextmanager
def _context(**overrides: Any) -> Iterator[AppContext]:
    """Build a full AppContext for one command and always close it."""
    ctx = build_context(_load_settings(**overrides))
    try:
        yield ctx
    finally:
        ctx.close()


@app.callback()
def main(
    config: Annotated[
        str, typer.Option("--config", help="YAML config file (default: config/default.yaml).")
    ] = "",
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    """Shared options for every command."""
    global _CONFIG_FILE
    _CONFIG_FILE = config
    if version:
        console.print(f"anchor {__version__}")
        raise typer.Exit(0)


# ==============================================================================
# Service commands
# ==============================================================================


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", help="Bind address (default: settings.host).")
    ] = "",
    port: Annotated[int, typer.Option("--port", help="Port (default: settings.port).")] = 0,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on source changes.")] = False,
) -> None:
    """Run the HTTP API."""
    with _guard():
        settings = _load_settings()
        # uvicorn constructs the app itself (in a child process under --reload),
        # so the config choice travels through the environment variable that
        # settings.py already documents as the way to select a config file.
        if _CONFIG_FILE:
            os.environ["ANCHOR_CONFIG_FILE"] = _CONFIG_FILE
        import uvicorn

        uvicorn.run(
            "anchor.api:create_app",
            factory=True,
            host=host or settings.host,
            port=port or settings.port,
            reload=reload,
        )


@app.command()
def ingest(
    corpus_dir: Annotated[
        str, typer.Option("--corpus-dir", help="Corpus directory (default: settings.paths).")
    ] = "",
    force: Annotated[
        bool, typer.Option("--force", help="Re-index every document, ignoring hashes.")
    ] = False,
) -> None:
    """Chunk, embed and index the corpus into data/index.db + data/vectors.npy."""
    with _guard():
        with _context() as ctx:
            report = ingest_corpus(
                ctx.settings,
                corpus_dir=corpus_dir or None,
                store=ctx.store,
                vectors=ctx.vectors,
                embedder=ctx.embedder,
                force=force,
            )
        table = Table(title="ingest", show_header=True, header_style="bold")
        table.add_column("field")
        table.add_column("value", justify="right")
        for field, value in report.model_dump().items():
            rendered = f"{value:.1f}" if isinstance(value, float) else str(value)
            table.add_row(field, rendered)
        console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    trace: Annotated[
        bool, typer.Option("--trace", help="Show per-stage ranks and timings.")
    ] = False,
    top_k: Annotated[
        int, typer.Option("--top-k", help="Override retrieval.top_k (0 = settings).")
    ] = 0,
) -> None:
    """Answer one question over the indexed corpus."""
    with _guard():
        with _context() as ctx:
            result = ctx.answer_service.answer(
                question, include_trace=trace, top_k=top_k or None
            )
            _print_answer(result)
            if trace and result.trace is not None:
                _print_trace(result.trace)


@app.command()
def agent(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    trace: Annotated[
        bool, typer.Option("--trace", help="Show per-stage ranks and timings.")
    ] = False,
    max_steps: Annotated[
        int, typer.Option("--max-steps", help="Cap the step budget (0 = settings).")
    ] = 0,
) -> None:
    """Answer one question through the bounded tool loop."""
    with _guard():
        with _context() as ctx:
            loop = AgentLoop(ctx=ctx, tools=build_tools(ctx), planner=get_planner(ctx))
            result = loop.run(question, include_trace=trace, max_steps=max_steps or None)
            _print_answer(result)
            _print_tool_steps(result)
            if trace and result.trace is not None:
                _print_trace(result.trace)


# ==============================================================================
# The measurement layer
# ==============================================================================


@app.command("eval")
def eval_command(
    limit: Annotated[
        int, typer.Option("--limit", help="Only run the first N cases (0 = all).")
    ] = 0,
    cases: Annotated[
        str, typer.Option("--cases", help="Cases JSONL (default: settings.paths).")
    ] = "",
) -> None:
    """Run the golden set and write eval/reports/."""
    with _guard():
        from anchor.eval.report import render_metrics_table, render_refusal_sweep, write_report
        from anchor.eval.runner import run_eval

        with _context() as ctx:
            settings = ctx.settings
            report, results = run_eval(
                settings, cases_path=cases or None, limit=limit or None, ctx=ctx
            )
            out_path = write_report(report, results, settings.paths.reports_dir)

        console.print(f"[bold]run[/bold] {report.run_id}  n={report.n_cases}")
        _raw(render_metrics_table([report]))
        console.print()
        console.print("[bold]refusal-threshold sweep[/bold] (the operating point is a curve)")
        _raw(
            render_refusal_sweep(
                results,
                settings.eval.refusal_sweep_points,
                settings.gate.min_lexical_overlap,
            )
        )
        console.print(f"\nwrote {out_path}")


@app.command()
def ablate(
    modes: Annotated[
        str, typer.Option("--modes", help="Comma-separated modes (default: settings).")
    ] = "",
    chunk_configs: Annotated[
        str, typer.Option("--chunk-configs", help="e.g. '256x0,512x64' (default: settings).")
    ] = "",
    out: Annotated[
        str, typer.Option("--out", help="Output markdown path (default: reports_dir).")
    ] = "",
) -> None:
    """Sweep retrieval strategy x chunking. BM25-only is the declared baseline."""
    with _guard():
        from anchor.eval.ablation import render_ablation_table, run_ablation

        settings = _load_settings()
        parsed_modes = [m.strip() for m in modes.split(",") if m.strip()] or None
        parsed_configs = _parse_chunk_configs(chunk_configs) or None
        out_path = out or str(Path(settings.paths.reports_dir) / "ablation.md")
        rows = run_ablation(
            settings, modes=parsed_modes, chunk_configs=parsed_configs, out_path=out_path
        )
        _raw(render_ablation_table(rows))
        console.print(f"\nwrote {out_path}  ({len(rows)} rows)")


@app.command()
def taxonomy(
    worst_n: Annotated[
        int, typer.Option("--worst-n", help="How many worst cases (0 = settings).")
    ] = 0,
    cases: Annotated[
        str, typer.Option("--cases", help="Cases JSONL (default: settings.paths).")
    ] = "",
    reuse: Annotated[
        bool, typer.Option("--reuse/--fresh", help="Reuse the newest results.jsonl if present.")
    ] = True,
) -> None:
    """Classify the worst cases into the four failure classes."""
    with _guard():
        from anchor.eval.taxonomy import run_taxonomy

        settings = _load_settings()
        results = _load_results(settings) if reuse else None
        if results is None:
            from anchor.eval.runner import run_eval

            with _context() as ctx:
                _, results = run_eval(ctx.settings, cases_path=cases or None, ctx=ctx)
        else:
            console.print(f"[dim]reusing {len(results)} cached case results[/dim]")

        out_path = str(Path(settings.paths.reports_dir) / "failures.md")
        distribution, records = run_taxonomy(
            settings,
            results,
            cases_path=cases or None,
            worst_n=worst_n or None,
            out_path=out_path,
        )
        table = Table(title="failure taxonomy", show_header=True, header_style="bold")
        table.add_column("failure class")
        table.add_column("n", justify="right")
        for name, count in distribution.items():
            table.add_row(name, str(count))
        console.print(table)
        console.print(f"wrote {out_path}  ({len(records)} worked examples)")


@app.command("judge-check")
def judge_check(
    labels: Annotated[
        str, typer.Option("--labels", help="Human labels JSONL (default: settings.paths).")
    ] = "",
) -> None:
    """Score the configured judge against the frozen human labels."""
    with _guard():
        from anchor.eval.judge_validation import validate_judge

        settings = _load_settings()
        out_path = str(Path(settings.paths.reports_dir) / "judge.md")
        agreement = validate_judge(settings, labels_path=labels or None, out_path=out_path)

        console.print(
            f"[bold]judge[/bold] {agreement.judge_name}  n={agreement.n}  "
            f"raw agreement={agreement.raw_agreement:.3f}  "
            f"cohen's kappa={agreement.cohens_kappa:.3f}"
        )
        labels_seen = sorted(agreement.confusion)
        table = Table(title="confusion (rows = human, cols = judge)", header_style="bold")
        table.add_column("human \\ judge")
        for name in labels_seen:
            table.add_column(name, justify="right")
        for human in labels_seen:
            row = agreement.confusion.get(human, {})
            table.add_row(human, *[str(row.get(name, 0)) for name in labels_seen])
        console.print(table)
        for note in agreement.notes:
            console.print(f"[dim]{note}[/dim]")
        console.print(
            "[yellow]kappa is agreement with ONE annotator (the author), not with truth.[/yellow]"
        )
        console.print(f"wrote {out_path}")


@app.command()
def thresholds(
    limit: Annotated[
        int, typer.Option("--limit", help="Only run the first N cases (0 = all).")
    ] = 0,
) -> None:
    """GENERATE eval/thresholds.yaml from a real run. Never hand-type this file."""
    with _guard():
        from anchor.eval.report import write_thresholds
        from anchor.eval.runner import run_eval

        settings = _load_settings()
        with _context() as ctx:
            report, _ = run_eval(ctx.settings, limit=limit or None, ctx=ctx)
        spec = write_thresholds(
            report, settings.paths.thresholds_path, settings.eval.threshold_slack
        )

        table = Table(title="generated thresholds", header_style="bold")
        table.add_column("metric")
        table.add_column("direction")
        table.add_column("bound", justify="right")
        for metric, bound in sorted(spec.min.items()):
            table.add_row(metric, "min", f"{bound:.4f}")
        for metric, bound in sorted(spec.max.items()):
            table.add_row(metric, "max", f"{bound:.4f}")
        console.print(table)
        console.print(
            f"wrote {settings.paths.thresholds_path} from run {spec.generated_from_run} "
            f"(slack={spec.slack})"
        )


@app.command()
def gate(
    limit: Annotated[
        int, typer.Option("--limit", help="Only run the first N cases (0 = all).")
    ] = 0,
) -> None:
    """THE REGRESSION GATE. Runs the eval and exits 1 if any floor is breached."""
    with _guard():
        from anchor.eval.report import check_thresholds, load_thresholds
        from anchor.eval.runner import run_eval

        settings = _load_settings()
        path = settings.paths.thresholds_path
        if not Path(path).is_file():
            raise EvalError(f"{path} does not exist — generate it with `anchor thresholds`")
        spec = load_thresholds(path)

        with _context() as ctx:
            report, _ = run_eval(ctx.settings, limit=limit or None, ctx=ctx)
        checks = check_thresholds(report, spec)

        table = Table(title="regression gate", header_style="bold")
        table.add_column("metric")
        table.add_column("dir")
        table.add_column("bound", justify="right")
        table.add_column("value", justify="right")
        table.add_column("result")
        for check in checks:
            verdict = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
            table.add_row(
                check.metric, check.direction, f"{check.bound:.4f}", f"{check.value:.4f}", verdict
            )
        console.print(table)

        failed = [c for c in checks if not c.passed]
        if failed:
            names = ", ".join(c.metric for c in failed)
            err_console.print(f"[red]GATE FAILED[/red] {len(failed)} breached: {names}")
            raise typer.Exit(1)
        console.print(f"[green]GATE PASSED[/green] {len(checks)} checks against {path}")


# ==============================================================================
# Diagnostics
# ==============================================================================


@app.command()
def doctor(
    full: Annotated[
        bool, typer.Option("--full", help="Also dump the full redacted config as JSON.")
    ] = False,
) -> None:
    """Show the effective config, provider readiness and index health."""
    with _guard():
        # doctor deliberately does NOT use build_context: an embedder/index
        # mismatch is exactly the thing it exists to REPORT, and build_context
        # raises on it.
        from anchor.embed import get_embedder
        from anchor.store import open_stores

        settings = _load_settings()
        console.print(f"[bold]anchor {__version__}[/bold]  config={_CONFIG_FILE or '(default)'}")
        console.print(f"fingerprint={settings.fingerprint()}  seed={settings.seed}")

        summary = Table(title="effective configuration", header_style="bold")
        summary.add_column("setting")
        summary.add_column("value")
        for key, value in [
            ("llm.provider", settings.llm.provider),
            ("embedding.provider", f"{settings.embedding.provider} (dim={settings.embedding.dim})"),
            ("judge.provider", settings.judge.provider),
            ("retrieval.mode", settings.retrieval.mode),
            ("chunking", f"{settings.chunking.chunk_tokens}/{settings.chunking.chunk_overlap}"),
            ("gate", f"overlap>={settings.gate.min_lexical_overlap}"),
            ("agent.planner", f"{settings.agent.planner} (max_steps={settings.agent.max_steps})"),
            ("paths.db_path", settings.paths.db_path),
            ("paths.corpus_dir", settings.paths.corpus_dir),
        ]:
            summary.add_row(key, str(value))
        console.print(summary)

        providers = Table(title="provider readiness", header_style="bold")
        providers.add_column("slot")
        providers.add_column("provider")
        providers.add_column("usable")
        providers.add_column("why / how")
        for slot, name, usable, note in _provider_rows(settings):
            mark = "[green]yes[/green]" if usable else "[yellow]no[/yellow]"
            providers.add_row(slot, name, mark, note)
        console.print(providers)

        store, vectors = open_stores(settings)
        try:
            embedder = get_embedder(settings)
            meta = vectors.meta()
            index = Table(title="index health", header_style="bold")
            index.add_column("check")
            index.add_column("value")
            index.add_row("documents", str(store.count_documents()))
            index.add_row("chunks", str(store.count_chunks()))
            index.add_row("vectors", str(vectors.count()))
            index.add_row("vector meta", f"{meta.get('embedder')} dim={meta.get('dim')}")
            index.add_row("configured embedder", f"{embedder.name} dim={embedder.dim}")
            matches = vectors.count() == 0 or (
                meta.get("embedder") == embedder.name
                and int(meta.get("dim", 0)) == embedder.dim
            )
            mismatch = "[red]MISMATCH — run `anchor ingest --force`[/red]"
            index.add_row("match", "[green]ok[/green]" if matches else mismatch)
            console.print(index)
        finally:
            vectors.close()
            store.close()

        if full:
            _raw(json.dumps(settings.public_dict(), indent=2, sort_keys=True))


# ==============================================================================
# Rendering helpers
# ==============================================================================


def _print_answer(result: AnswerResult | AgentResult) -> None:
    console.print(f"[bold]answer_type[/bold] {result.answer_type.value}", end="  ")
    console.print(f"[bold]groundedness[/bold] {result.groundedness:.3f}", end="  ")
    if result.judge is not None:
        console.print(f"[bold]judge[/bold] {result.judge.label.value} ({result.judge.judge})")
    else:
        console.print()
    console.print()
    _raw(result.answer)
    console.print()

    if result.refusal_reason:
        console.print(f"[yellow]refusal reason[/yellow] {result.refusal_reason}")
    if result.citations:
        table = Table(title="citations", header_style="bold")
        table.add_column("marker")
        table.add_column("doc_id")
        table.add_column("section")
        table.add_column("chunk_id")
        for citation in result.citations:
            table.add_row(
                citation.marker, citation.doc_id, citation.heading_path or "-", citation.chunk_id
            )
        console.print(table)
    console.print(
        f"[dim]trace_id={result.trace_id} latency={result.latency_ms:.0f}ms "
        f"tokens={result.usage.total_tokens}[/dim]"
    )


def _print_tool_steps(result: AgentResult) -> None:
    table = Table(
        title=f"agent steps (planner={result.planner}, used={result.steps_used}, "
        f"budget_exhausted={result.budget_exhausted})",
        header_style="bold",
    )
    table.add_column("#", justify="right")
    table.add_column("tool")
    table.add_column("ok")
    table.add_column("ms", justify="right")
    table.add_column("observation")
    for step in result.tool_steps:
        observation = " ".join(step.observation.split())[:80]
        table.add_row(
            str(step.step),
            step.tool,
            "[green]yes[/green]" if step.ok else "[red]no[/red]",
            f"{step.duration_ms:.0f}",
            observation,
        )
    console.print(table)


def _print_trace(trace: Trace) -> None:
    if trace.gate is not None:
        gate_decision = trace.gate
        verdict = "[green]passed[/green]" if gate_decision.passed else "[red]failed[/red]"
        console.print(
            f"[bold]gate[/bold] {verdict}  lexical_overlap={gate_decision.lexical_overlap:.3f}  "
            f"top_share={gate_decision.top_share:.3f}  {gate_decision.reason}"
        )

    if trace.candidates:
        table = Table(title="candidates (per-stage ranks)", header_style="bold")
        for column in ("final", "chunk_id", "doc_id", "bm25", "dense", "rrf", "rerank", "mmr"):
            table.add_column(column, justify="right" if column != "doc_id" else "left")
        for candidate in trace.candidates:
            table.add_row(
                str(candidate.final_rank if candidate.final_rank is not None else "-"),
                candidate.chunk_id,
                candidate.chunk.doc_id,
                str(candidate.bm25_rank if candidate.bm25_rank is not None else "-"),
                str(candidate.dense_rank if candidate.dense_rank is not None else "-"),
                f"{candidate.rrf_score:.4f}",
                f"{candidate.rerank_score:.4f}" if candidate.rerank_score is not None else "-",
                "yes" if candidate.selected_by_mmr else "-",
            )
        console.print(table)

    if trace.context:
        table = Table(title="packed context", header_style="bold")
        table.add_column("marker")
        table.add_column("doc_id")
        table.add_column("tokens", justify="right")
        table.add_column("section")
        for block in trace.context:
            table.add_row(
                block.marker, block.doc_id, str(block.n_tokens), block.heading_path or "-"
            )
        console.print(table)

    if trace.timings:
        table = Table(title="timings", header_style="bold")
        table.add_column("stage")
        table.add_column("ms", justify="right")
        for timing in trace.timings:
            table.add_row(timing.stage, f"{timing.duration_ms:.1f}")
        console.print(table)

    for note in trace.notes:
        console.print(f"[dim]note: {note}[/dim]")


def _parse_chunk_configs(raw: str) -> list[tuple[int, int]]:
    """'256x0,512x64' -> [(256, 0), (512, 64)]."""
    configs: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise EvalError(f"bad chunk config {part!r}; expected TOKENSxOVERLAP, e.g. 512x64")
        tokens, _, overlap = part.partition("x")
        try:
            configs.append((int(tokens), int(overlap)))
        except ValueError as exc:
            raise EvalError(f"bad chunk config {part!r}: {exc}") from None
    return configs


def _load_results(settings: Settings) -> list[CaseResult] | None:
    """Read the newest results.jsonl, or None when there isn't a usable one."""
    path = Path(settings.paths.reports_dir) / "results.jsonl"
    if not path.is_file():
        return None
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        results = [CaseResult.model_validate_json(line) for line in lines]
    except (OSError, ValueError):
        return None
    return results or None


def _provider_rows(settings: Settings) -> list[tuple[str, str, bool, str]]:
    """Which providers this environment could actually use, and why not."""
    llm = settings.llm
    embedding = settings.embedding
    has_boto3 = find_spec("boto3") is not None
    rows: list[tuple[str, str, bool, str]] = [
        ("llm", "stub", True, "offline extractive stub — no key, no network"),
        (
            "llm",
            "openai",
            llm.api_key is not None,
            "set ANCHOR_LLM__API_KEY" if llm.api_key is None else "key configured",
        ),
        (
            "llm",
            "anthropic",
            llm.api_key is not None,
            "set ANCHOR_LLM__API_KEY" if llm.api_key is None else "key configured",
        ),
        (
            "llm",
            "azure_openai",
            llm.api_key is not None and bool(llm.azure_endpoint) and bool(llm.azure_deployment),
            "needs ANCHOR_LLM__API_KEY + azure_endpoint + azure_deployment",
        ),
        (
            "llm",
            "bedrock",
            has_boto3,
            "install the `bedrock` extra and configure AWS credentials"
            if not has_boto3
            else "boto3 present; AWS credentials resolved by boto3 itself",
        ),
        ("embedding", "hashing", True, f"deterministic, numpy-only, dim={embedding.dim}"),
        (
            "embedding",
            "openai",
            embedding.api_key is not None,
            "set ANCHOR_EMBEDDING__API_KEY" if embedding.api_key is None else "key configured",
        ),
        (
            "embedding",
            "azure_openai",
            embedding.api_key is not None and bool(embedding.azure_endpoint),
            "needs ANCHOR_EMBEDDING__API_KEY + azure_endpoint + azure_deployment",
        ),
        ("embedding", "bedrock", has_boto3, "install the `bedrock` extra + AWS credentials"),
        ("judge", "heuristic", True, "offline lexical judge — validated by `anchor judge-check`"),
        (
            "judge",
            "llm",
            llm.api_key is not None,
            "uses the configured LLM provider; needs its key",
        ),
    ]
    return rows
