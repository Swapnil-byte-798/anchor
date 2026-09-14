"""DECISIVE FEATURE 1 — the ablation: 4 retrieval strategies x 6 chunk configs.

Why this is a real experiment rather than a table of numbers:

*   **BM25-only is the declared baseline.** It is printed FIRST in every block and
    ``delta_recall_at_5_vs_bm25`` is always measured against the bm25 row of the
    SAME chunk config. Comparing hybrid@512/64 against bm25@256/0 would move two
    variables at once and the delta would mean nothing.

*   **The labels survive re-chunking.** Gold is a verbatim answer span resolved to
    chunk ids at eval time (see :mod:`anchor.eval.cases`), so all six chunk
    configurations are scored against the same 43 unmodified questions. If labels
    named chunk ids, the chunking sweep would be circular by construction.

*   **The gate is scale-free, so the modes are comparable.** Its one feature is a
    ratio over the question's own tokens, which is what allows a single threshold
    to be used for bm25, dense and hybrid alike. Four separately-tuned gates would
    measure the tuning.

*   **Duplicate indexes are detected, not printed as findings.** Each chunk config
    is fingerprinted by the sha256 of its sorted chunk ids, and the report carries
    a loud banner when two configs collide — which is exactly what happened when
    the grid swept 512 and 1024 tokens over documents no larger than 360 tokens.

*   **It is retrieval-only.** No generation, no grounding, no judge: 6 re-ingests
    and 24 x 43 retrievals, which is seconds rather than half an hour, which is
    why the grid can be run on every interesting change instead of once.

Each chunk config is ingested ONCE into its own temporary data directory, then
every mode is run against that one index. The temp directories are removed
afterwards; the repo's real ``data/`` is never touched.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from anchor.context import build_context
from anchor.errors import EvalError
from anchor.eval.runner import run_eval
from anchor.ingest import ingest_corpus
from anchor.obs import get_logger, log_event
from anchor.settings import Settings
from anchor.textutil import sha256_text, utc_now_iso
from anchor.types import AblationRow, MetricReport

__all__ = ["run_ablation", "render_ablation_table"]

_LOG = get_logger("anchor.eval.ablation")

_BASELINE_MODE = "bm25"
_VALID_MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def run_ablation(
    settings: Settings,
    *,
    modes: list[str] | None = None,
    chunk_configs: list[tuple[int, int]] | None = None,
    out_path: str | None = None,
) -> list[AblationRow]:
    """Run the strategy x chunking grid and write the markdown report.

    Returns one :class:`AblationRow` per cell, ordered chunk config by chunk
    config with the bm25 baseline first inside each block.
    """
    selected_modes = list(modes or settings.eval.ablation_modes)
    unknown = [mode for mode in selected_modes if mode not in _VALID_MODES]
    if unknown:
        raise EvalError(f"unknown retrieval mode(s) {unknown}; expected some of {_VALID_MODES}")
    if _BASELINE_MODE not in selected_modes:
        raise EvalError(
            "bm25 is the declared baseline of this ablation and every delta is measured "
            "against it; it cannot be excluded from the grid"
        )

    configs = [tuple(pair) for pair in (chunk_configs or settings.eval.ablation_chunk_configs)]
    parsed: list[tuple[int, int]] = []
    for pair in configs:
        if len(pair) != 2:
            raise EvalError(f"chunk config {pair!r} must be (chunk_tokens, chunk_overlap)")
        parsed.append((int(pair[0]), int(pair[1])))

    rows: list[AblationRow] = []
    reports: list[MetricReport] = []
    fingerprints: dict[tuple[int, int], str] = {}
    for chunk_tokens, chunk_overlap in parsed:
        block_rows, block_reports, index_fingerprint = _run_chunk_config(
            settings, chunk_tokens, chunk_overlap, selected_modes
        )
        rows.extend(block_rows)
        reports.extend(block_reports)
        fingerprints[(chunk_tokens, chunk_overlap)] = index_fingerprint

    _apply_baseline_deltas(rows)

    destination = out_path or str(Path(settings.paths.reports_dir) / "ablation.md")
    _write_report(destination, rows, reports, settings, fingerprints)
    return rows


def _duplicate_index_groups(
    fingerprints: dict[tuple[int, int], str],
) -> list[list[tuple[int, int]]]:
    """Chunk configs that produced BYTE-IDENTICAL indexes, grouped.

    The grid used to sweep 256/512/1024 tokens over a corpus whose largest chunk
    at any setting is 360 tokens, so 512/0, 512/64, 1024/0 and 1024/128 all
    produced the same 42 chunks: four identical blocks presented as four distinct
    data points, and a generated conclusion about chunk size that was a tautology
    rather than a finding. A duplicate must never be silent again, so it is
    detected here and printed at the top of the report.
    """
    grouped: dict[str, list[tuple[int, int]]] = {}
    for config, digest in fingerprints.items():
        grouped.setdefault(digest, []).append(config)
    return sorted(
        (sorted(configs) for configs in grouped.values() if len(configs) > 1),
        key=lambda configs: configs[0],
    )


# ==============================================================================
# One chunk configuration
# ==============================================================================


def _run_chunk_config(
    settings: Settings, chunk_tokens: int, chunk_overlap: int, modes: list[str]
) -> tuple[list[AblationRow], list[MetricReport], str]:
    """Ingest ONCE at this chunk size, then run every mode against that index.

    Also returns a fingerprint of the index itself — the sha256 of its sorted
    chunk ids — so the caller can tell whether two chunk configurations in the
    grid are actually the same experiment.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"anchor-ablate-{chunk_tokens}-{chunk_overlap}-")
    rows: list[AblationRow] = []
    reports: list[MetricReport] = []
    index_fingerprint = ""
    try:
        base = _config_for(settings, temp_dir, chunk_tokens, chunk_overlap)
        ingest_report = ingest_corpus(base, force=True)
        index_fingerprint = _index_fingerprint(base)
        log_event(
            _LOG,
            logging.INFO,
            "ablate.ingested",
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            chunks=ingest_report.chunks_written,
            index_fingerprint=index_fingerprint,
        )

        # bm25 first so the baseline is computed before anything compares to it.
        ordered = [mode for mode in _VALID_MODES if mode in modes]
        for mode in ordered:
            cfg = _config_for(settings, temp_dir, chunk_tokens, chunk_overlap)
            cfg.retrieval.mode = mode  # type: ignore[assignment]
            ctx = build_context(cfg)
            try:
                report, _ = run_eval(cfg, ctx=ctx, retrieval_only=True)
            finally:
                ctx.close()
            reports.append(report)
            rows.append(
                AblationRow(
                    mode=mode,
                    chunk_tokens=chunk_tokens,
                    chunk_overlap=chunk_overlap,
                    n_cases=report.n_cases,
                    recall_at_5=report.metrics["recall_at_5"],
                    recall_at_10=report.metrics["recall_at_10"],
                    mrr=report.metrics["mrr"],
                    ndcg_at_10=report.metrics["ndcg_at_10"],
                    latency_p50_ms=report.metrics["latency_p50_ms"],
                    latency_p95_ms=report.metrics["latency_p95_ms"],
                    is_baseline=mode == _BASELINE_MODE,
                )
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return rows, reports, index_fingerprint


def _index_fingerprint(cfg: Settings) -> str:
    """sha256 over the sorted chunk ids of a freshly ingested index, 12 hex chars.

    A chunk id is derived from the document and the span it covers, so two
    configurations that agree on this produced the same chunks — and therefore
    the same experiment, however different their chunk_tokens look in the table.
    """
    from anchor.store import open_stores

    store, vectors = open_stores(cfg)
    try:
        chunk_ids = sorted(chunk.chunk_id for chunk in store.list_chunks())
    finally:
        store.close()
        vectors.close()
    return sha256_text("\n".join(chunk_ids))[:12]


def _config_for(
    settings: Settings, temp_dir: str, chunk_tokens: int, chunk_overlap: int
) -> Settings:
    """A deep copy of the settings pointed at a throwaway index."""
    cfg = settings.model_copy(deep=True)
    cfg.chunking.chunk_tokens = chunk_tokens
    cfg.chunking.chunk_overlap = chunk_overlap
    cfg.paths.data_dir = temp_dir
    cfg.paths.db_path = str(Path(temp_dir) / "index.db")
    cfg.paths.vectors_path = str(Path(temp_dir) / "vectors.npy")
    return cfg


def _apply_baseline_deltas(rows: list[AblationRow]) -> None:
    """Set ``delta_recall_at_5_vs_bm25`` against the bm25 row of the SAME config."""
    baselines = {
        (row.chunk_tokens, row.chunk_overlap): row.recall_at_5
        for row in rows
        if row.mode == _BASELINE_MODE
    }
    for row in rows:
        baseline = baselines.get((row.chunk_tokens, row.chunk_overlap))
        if baseline is None:  # pragma: no cover - guarded by the bm25 check above
            continue
        row.delta_recall_at_5_vs_bm25 = 0.0 if row.is_baseline else round(
            row.recall_at_5 - baseline, 6
        )


# ==============================================================================
# Rendering
# ==============================================================================


def render_ablation_table(rows: list[AblationRow]) -> str:
    """GitHub-markdown blocks, one per chunk config, bm25 printed first.

    The provider stamp (embedder, judge, config fingerprint) is written into the
    report file's preamble rather than here: :class:`AblationRow` is a shared
    contract and carries no provider fields, so a renderer that took them from
    somewhere else would be inventing them.
    """
    if not rows:
        return "_no ablation rows_"

    n_cases = max(row.n_cases for row in rows)
    modes = sorted({row.mode for row in rows})
    configs: list[tuple[int, int]] = []
    for row in rows:
        key = (row.chunk_tokens, row.chunk_overlap)
        if key not in configs:
            configs.append(key)

    lines: list[str] = [
        f"**{len(modes)} strategies x {len(configs)} chunk configs = {len(rows)} rows**, "
        f"n={n_cases} golden cases, retrieval only (no generation).",
        "",
        "`bm25` is the declared BASELINE and is printed first in every block. "
        "The Δ column is recall@5 minus the bm25 row **of the same chunk config**.",
    ]

    for chunk_tokens, chunk_overlap in configs:
        block = [row for row in rows if (row.chunk_tokens, row.chunk_overlap) == (
            chunk_tokens,
            chunk_overlap,
        )]
        block.sort(key=lambda row: (0 if row.is_baseline else 1, _mode_order(row.mode)))
        lines.extend(
            [
                "",
                f"#### chunk_tokens={chunk_tokens}, chunk_overlap={chunk_overlap}",
                "",
                "| mode | recall@5 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms "
                "| Δ recall@5 vs bm25 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in block:
            name = f"`{row.mode}` **(baseline)**" if row.is_baseline else f"`{row.mode}`"
            delta = "—" if row.is_baseline else f"{row.delta_recall_at_5_vs_bm25:+.3f}"
            lines.append(
                f"| {name} | {row.recall_at_5:.3f} | {row.recall_at_10:.3f} | {row.mrr:.3f} "
                f"| {row.ndcg_at_10:.3f} | {row.latency_p50_ms:.1f} | {row.latency_p95_ms:.1f} "
                f"| {delta} |"
            )

    return "\n".join(lines)


def _mode_order(mode: str) -> int:
    return _VALID_MODES.index(mode) if mode in _VALID_MODES else len(_VALID_MODES)


def _verdict(rows: list[AblationRow]) -> list[str]:
    """The finding, in prose, DERIVED from the rows — never typed by a human.

    Written so that a result the author would rather not have (the baseline
    winning) reads exactly as clearly as a result they would.
    """
    if not rows:
        return ["_No rows: nothing was run._"]

    challengers = [row for row in rows if not row.is_baseline]
    lines: list[str] = []

    if challengers:
        best = sorted(
            challengers,
            key=lambda row: (
                -row.delta_recall_at_5_vs_bm25,
                _mode_order(row.mode),
                row.chunk_tokens,
                row.chunk_overlap,
            ),
        )[0]
        delta = best.delta_recall_at_5_vs_bm25
        where = f"`{best.mode}` at {best.chunk_tokens}/{best.chunk_overlap}"
        if delta > 0:
            lines.append(
                f"**The best any strategy managed over its own BM25 baseline is "
                f"{delta:+.3f} recall@5** ({where}). At n=43 that is roughly {delta * 43:.0f} "
                "question(s): read it as a direction, not as a measurement."
            )
        else:
            lines.append(
                f"**No strategy beat the BM25 baseline in this run.** The best non-baseline "
                f"delta is {delta:+.3f} recall@5 ({where}); every other cell is worse. That is "
                "the honest result and it is left standing here rather than tuned away."
            )
            lines.append("")
            lines.append(
                "The most likely reason is a property of the BENCHMARK, not a defect in the "
                "dense leg: 24 of the 43 questions are mechanically derived from corpus "
                "sentences, and although the builder caps their lexical overlap with the source "
                "sentence at 0.5, they still speak the corpus's own vocabulary. Exact term "
                "matching is exactly what BM25 is best at. A dense retriever earns its place on "
                "paraphrase and synonymy, which this question set is deliberately constrained "
                "not to contain much of. Treat this table as evidence about the retrievers "
                "**on this set**, and as a reason to distrust anyone's absolute recall number, "
                "including this one."
            )

    n_configs = len({(row.chunk_tokens, row.chunk_overlap) for row in rows})
    if n_configs < 2:
        return lines

    by_mode: dict[str, list[float]] = {}
    for row in rows:
        by_mode.setdefault(row.mode, []).append(row.recall_at_5)
    spread = max((max(values) - min(values)) for values in by_mode.values())
    lines.append("")
    if spread < 1e-9:
        lines.append(
            f"**Chunking changed nothing.** recall@5 is identical across all {n_configs} chunk "
            "configurations for every strategy. On a six-document corpus whose gold spans are "
            "short and resolve into whichever chunk contains them, chunk size is simply not the "
            "binding constraint — which is worth knowing before spending a week tuning it."
        )
    else:
        lines.append(
            f"**Chunking moved recall@5 by at most {spread:.3f}** within a single strategy "
            f"across the {n_configs} configurations — the widest gap any one strategy shows "
            "between its best and worst chunk size."
        )
    return lines


def _write_report(
    path: str,
    rows: list[AblationRow],
    reports: list[MetricReport],
    settings: Settings,
    fingerprints: dict[tuple[int, int], str] | None = None,
) -> None:
    """Write eval/reports/ablation.md — prose first, then the grid."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    duplicates = _duplicate_index_groups(fingerprints or {})
    banner: list[str] = []
    if duplicates:
        banner = [
            "> ## ⚠ DUPLICATE INDEXES IN THIS GRID",
            ">",
            "> Two or more chunk configurations below produced a BYTE-IDENTICAL index, so the",
            "> blocks that share one are the same experiment printed more than once. Their rows",
            "> are not independent data points and any conclusion about chunk size drawn across",
            "> them is a tautology. Sweep sizes this corpus can actually resolve, or enlarge the",
            "> documents.",
            ">",
        ]
        for group in duplicates:
            configs = ", ".join(f"{tokens}/{overlap}" for tokens, overlap in group)
            banner.append(f"> * identical index: {configs}")
        banner.append("")

    # The provider names are read off a completed run because that is where the
    # resolved component identities live. The config fingerprint is NOT: every
    # report in `reports` was produced by a throwaway copy of the settings whose
    # paths.* point at a `tempfile.mkdtemp()` directory (see `_config_for`), and
    # `Settings.fingerprint()` hashes those paths, so stamping `reports[0]` gave a
    # fresh random digest on every run of an unchanged configuration — the exact
    # opposite of provenance. The caller's own settings are hashed instead, which
    # is the same value `anchor eval` stamps on its reports for the same config.
    sample = reports[0] if reports else None
    stamp = (
        f"embedder `{sample.embedder_name}` | judge `{sample.judge_name}` | "
        f"llm `{sample.llm_name}` (unused: retrieval only) | "
        f"base config fingerprint `{settings.fingerprint()}`"
        if sample is not None
        else "no runs"
    )

    body = [
        "# Ablation — retrieval strategy x chunking",
        "",
        "> GENERATED by `make ablate` (`anchor ablate`). Do not edit: every number here is",
        "> written by the command, and hand-editing one is the failure mode this repo is",
        "> built to prevent.",
        "",
        f"Run at {utc_now_iso()} — {stamp}.",
        "",
        *banner,
        "## Read the deltas, not the absolute column",
        "",
        "The absolute recall numbers below are **optimistic and not transferable**. The corpus",
        "is synthetic and self-written, and 24 of the 43 questions were mechanically derived",
        "from that same corpus (capped at 0.5 lexical overlap with their source sentence). A",
        "retriever scored against questions drawn from its own index will always look better",
        "than one facing real user language.",
        "",
        "What survives that bias is the **comparison**. Every row in a block faces exactly the",
        "same questions over exactly the same index, so the Δ column — recall@5 against the",
        "bm25 baseline of the same chunk config — is a like-for-like measurement of what the",
        "dense leg, the fusion and the reranker actually bought. That is the result of this",
        "table. With n=43, a delta of a few points is not a finding.",
        "",
        "## What this run actually found",
        "",
        *_verdict(rows),
        "",
        "## The grid",
        "",
        render_ablation_table(rows),
        "",
        "## Method",
        "",
        "* Each chunk config is ingested **once** into its own temporary index; every mode is",
        "  then run against that one index, so a mode difference is never a re-ingest",
        "  difference.",
        "* Gold labels are verbatim answer **spans** resolved to chunk ids at eval time, so the",
        "  same 43 questions are valid at every chunk size in the grid. Nothing is re-labelled",
        "  between rows.",
        "* Each ingested index is FINGERPRINTED (sha256 of its sorted chunk ids). If two chunk",
        "  configurations produce the same index they are the same experiment, and a banner at",
        "  the top of this report says so instead of letting the duplicate pass as a data point.",
        "* The **base config fingerprint** in the preamble is the fingerprint of the",
        "  configuration this command was invoked with — the same digest `anchor eval` stamps",
        "  on its reports — so two runs of an unchanged config print the same value. It covers",
        "  everything the grid holds fixed; the two things the grid varies are printed with the",
        "  rows themselves: the chunk config heads each block, the mode labels each row.",
        "* The abstain gate uses one ratio feature and therefore one threshold across all four",
        "  modes — no per-mode retuning, which is what keeps the rows comparable.",
        "* Retrieval only: no generation, no grounding, no judge.",
        "* Latency is wall-clock for retrieval alone on the machine that ran the command, so it",
        "  is comparable **within** this table and meaningless across machines.",
        "",
    ]
    destination.write_text("\n".join(body), encoding="utf-8")
    log_event(_LOG, logging.INFO, "ablate.wrote", path=str(destination), rows=len(rows))
