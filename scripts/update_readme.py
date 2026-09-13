#!/usr/bin/env python3
"""Rewrite the README's metric table from the most recent run — or fail if it is stale.

    python scripts/update_readme.py --write    # regenerate the block in README.md
    python scripts/update_readme.py --check    # exit 1 with a diff if it is stale

This is the mechanism that makes a hand-typed number impossible to merge. CI runs
``--check`` on every push: it regenerates the block between ``<!-- EVAL:BEGIN -->``
and ``<!-- EVAL:END -->`` from ``eval/reports/latest.json`` and fails the build if
what is committed differs by a single character. There is no path by which a
number reaches the README except through a shipped command.

Two deliberate properties make that check meaningful rather than flaky:

*   **The block contains nothing machine-dependent.** No latency, no timestamp, no
    run id. Those live in the run JSON. A wall-clock number in a committed table
    would fail the build on a slower CI runner, and the usual fix for that is to
    make the check toothless.
*   **The block is a pure function of the run.** The same index and the same
    golden set produce the same table on any machine, because the default profile
    is a deterministic hashing embedder and an extractive stub with temperature 0.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from anchor.context import build_context
from anchor.errors import AnchorError, EvalError
from anchor.eval.cases import load_cases
from anchor.eval.report import render_metrics_table, render_refusal_sweep, write_report
from anchor.eval.runner import run_eval
from anchor.settings import Settings
from anchor.types import CaseResult, MetricReport

BEGIN = "<!-- EVAL:BEGIN -->"
END = "<!-- EVAL:END -->"


def build_block(
    report: MetricReport, results: list[CaseResult], settings: Settings
) -> str:
    """The markdown that lives between the two markers. Deterministic by design."""
    sweep = render_refusal_sweep(
        results, settings.eval.refusal_sweep_points, settings.gate.min_lexical_overlap
    )
    caption = _caption(settings)

    return "\n".join(
        [
            "",
            "<!-- Written by scripts/update_readme.py from eval/reports/latest.json.",
            "     Do not edit by hand: CI runs `--check` and fails when this block is",
            "     stale or hand-edited. -->",
            "",
            render_metrics_table([report]),
            "",
            caption,
            "",
            "#### Refusal-threshold sweep",
            "",
            sweep,
            "",
            "#### The other three reports",
            "",
            "[`eval/reports/ablation.md`](eval/reports/ablation.md) — 4 strategies x 6 chunk "
            "configs against the BM25 baseline · "
            "[`eval/reports/failures.md`](eval/reports/failures.md) — the worst cases, each "
            "assigned one failure class · "
            "[`eval/reports/judge.md`](eval/reports/judge.md) — the judge against 30 frozen "
            "human labels.",
            "",
        ]
    )


def _caption(settings: Settings) -> str:
    """The bias caption. The overlap cap is the biggest weakness in the set, so it
    is printed next to the numbers it inflates, not in a footnote."""
    try:
        cases = load_cases(settings.paths.golden_path)
    except AnchorError:
        return ""

    mechanical = [case for case in cases if case.provenance == "mechanical"]
    authored = [case for case in cases if case.provenance == "authored"]
    substituted = [case for case in cases if case.provenance == "entity_substituted"]
    if not mechanical:
        return ""

    overlaps = [case.stem_overlap for case in mechanical]
    mean = sum(overlaps) / len(overlaps)
    return (
        f"_n={len(cases)} golden cases: {len(mechanical)} mechanically generated from the corpus "
        f"(mean lexical overlap with their source sentence **{mean:.3f}**, max "
        f"**{max(overlaps):.3f}**, hard cap 0.500), {len(substituted)} entity-substituted and "
        f"provably unanswerable, {len(authored)} written by the author and flagged `authored` in "
        "every table above. The corpus is synthetic and self-written, so absolute scores say "
        "nothing about real customer documents — trust the relative deltas in the ablation "
        "table, and treat every rate here as a smoke-grade signal at this sample size._"
    )


def render_readme(current: str, block: str) -> str:
    """Replace the marked block, leaving every other byte of the README alone."""
    start = current.find(BEGIN)
    end = current.find(END)
    if start == -1 or end == -1 or end < start:
        raise EvalError(
            f"README.md must contain the markers {BEGIN} and {END} around the metrics table"
        )
    head = current[: start + len(BEGIN)]
    tail = current[end:]
    return f"{head}\n{block}\n{tail}"


def load_or_run(settings: Settings) -> tuple[MetricReport, list[CaseResult]]:
    """Read the most recent run, or execute one when there is nothing to read."""
    reports_dir = Path(settings.paths.reports_dir)
    report_path = reports_dir / "latest.json"
    results_path = reports_dir / "results.jsonl"

    if report_path.is_file() and results_path.is_file():
        report = MetricReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        lines = [
            line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        return report, [CaseResult.model_validate_json(line) for line in lines]

    print("no eval/reports/latest.json — running the evaluation first", file=sys.stderr)
    ctx = build_context(settings)
    try:
        report, results = run_eval(ctx.settings, ctx=ctx)
    finally:
        ctx.close()
    write_report(report, results, settings.paths.reports_dir)
    return report, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite (or verify) the README metric table from the latest eval run."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Rewrite the block in README.md.")
    mode.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 with a diff when the committed block is stale. Used by CI.",
    )
    parser.add_argument("--config", default="", help="YAML config file (default: settings).")
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(args.config or None)
        report, results = load_or_run(settings)
        readme_path = Path(settings.paths.readme_path)
        if not readme_path.is_file():
            raise EvalError(f"{readme_path} does not exist")

        current = readme_path.read_text(encoding="utf-8")
        updated = render_readme(current, build_block(report, results, settings))
    except AnchorError as exc:
        print(f"error {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.write:
        if updated == current:
            print(f"{readme_path} already up to date")
            return 0
        readme_path.write_text(updated, encoding="utf-8")
        print(f"wrote {readme_path} from run {report.run_id}")
        return 0

    if updated == current:
        print(f"{readme_path} is up to date with run {report.run_id}")
        return 0

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"{readme_path} (committed)",
        tofile=f"{readme_path} (regenerated from {report.run_id})",
    )
    sys.stdout.writelines(diff)
    print(
        f"\nSTALE: {readme_path} does not match the latest run. "
        "Run `make readme` and commit the result.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
