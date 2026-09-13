"""The test that keeps the golden set honest.

Nothing here ingests anything, opens a database or builds an index: every
assertion is made against the raw corpus files and the committed JSONL, which is
what lets it run in a second and fail for exactly one reason at a time.

Four properties are checked, and each one closes a specific way a
self-authored benchmark can lie:

1. LABELS RESOLVE. Every gold ``answer_span`` is a verbatim normalised
   substring of the document it claims to come from, and every ``doc_id`` is a
   real corpus file. Gold never names a chunk id, so changing chunk size or
   overlap cannot invalidate a label — which is what makes the chunking half of
   the ablation a real experiment rather than a self-fulfilling one.
2. UNANSWERABILITY IS PROVEN. Every case in the unanswerable slice carries an
   out-of-corpus entity whose marker token occurs zero times in the normalised
   corpus. Unanswerable is a checked property here, not the author's opinion.
3. DIVERGENCE IS ENFORCED. Every machine-derived case stays at or below the
   0.5 lexical-overlap cap against its own source sentence.
4. THE SET IS REPRODUCIBLE. Re-running the builder produces the committed file
   byte for byte, and the builder's transitive import set contains none of the
   retrieval-side modules. Together those two say that no question was ever
   quietly retuned to make a score go up.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anchor.textutil import jaccard, make_doc_id, normalize_text
from anchor.types import EvalCase, HumanLabel

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "corpus"
BUILDER = REPO_ROOT / "scripts" / "build_golden.py"
QA_PATH = REPO_ROOT / "eval" / "golden" / "qa.jsonl"
AUTHORED_PATH = REPO_ROOT / "eval" / "golden" / "authored_seed.jsonl"
HUMAN_LABELS_PATH = REPO_ROOT / "eval" / "golden" / "human_labels.jsonl"

# The composition the corpus plan commits to, by (type, provenance).
EXPECTED_COMPOSITION = {
    ("factual", "mechanical"): 24,
    ("unanswerable", "entity_substituted"): 10,
    ("multi_hop", "authored"): 5,
    ("arithmetic", "authored"): 4,
}

# The builder must never learn anything about how its questions will be answered.
FORBIDDEN_MODULES = frozenset(
    {
        "anchor.retrieval",
        "anchor.embed",
        "anchor.store",
        "anchor.llm",
        "anchor.judge",
        "anchor.service",
        "anchor.agent",
        "anchor.api",
        "anchor.cli",
    }
)

MAX_STEM_OVERLAP = 0.5


# ---------------------------------------------------------------------------
# fixtures — the corpus is read straight off disk, never through the index
# ---------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    """Environment for a subprocess, with the src layout importable.

    In CI the package is pip-installed and this changes nothing; in a bare
    checkout it is what lets the builder import ``anchor``.
    """
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


@pytest.fixture(scope="module")
def corpus_texts() -> dict[str, str]:
    """``doc_id -> normalised full text`` for every markdown file in corpus/."""
    texts: dict[str, str] = {}
    for path in sorted(CORPUS_DIR.rglob("*.md")):
        doc_id = make_doc_id(path.relative_to(CORPUS_DIR).as_posix())
        texts[doc_id] = normalize_text(path.read_text(encoding="utf-8"))
    assert texts, "corpus/ contains no markdown documents"
    return texts


@pytest.fixture(scope="module")
def corpus_blob(corpus_texts: dict[str, str]) -> str:
    """One padded normalised string holding the whole corpus, for occurrence checks."""
    return " " + " ".join(corpus_texts[doc_id] for doc_id in sorted(corpus_texts)) + " "


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    lines = QA_PATH.read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# 1. shape and composition
# ---------------------------------------------------------------------------


def test_every_line_is_a_valid_eval_case(cases: list[EvalCase]) -> None:
    assert len(cases) == sum(EXPECTED_COMPOSITION.values()) == 43
    ids = [case.id for case in cases]
    assert len(set(ids)) == len(ids), "duplicate case ids"
    questions = [case.question for case in cases]
    assert len(set(questions)) == len(questions), "duplicate questions"


def test_composition_is_exactly_as_documented(cases: list[EvalCase]) -> None:
    """24 mechanical / 10 entity-substituted / 5 multi-hop / 4 arithmetic.

    The README breaks every metric out by provenance, so this ratio is load
    bearing: it is what lets a reviewer see which numbers depend on questions
    the author wrote.
    """
    observed: dict[tuple[str, str], int] = {}
    for case in cases:
        key = (case.type, case.provenance)
        observed[key] = observed.get(key, 0) + 1
    assert observed == EXPECTED_COMPOSITION


def test_case_ids_follow_their_tier_prefix(cases: list[EvalCase]) -> None:
    prefix_by_provenance = {"mechanical": "m-", "entity_substituted": "u-", "authored": "a-"}
    for case in cases:
        assert case.id.startswith(prefix_by_provenance[case.provenance]), case.id


# ---------------------------------------------------------------------------
# 2. DEFENCE 3 — labels are spans, and they resolve
# ---------------------------------------------------------------------------


def test_every_gold_doc_id_is_a_real_corpus_file(
    cases: list[EvalCase], corpus_texts: dict[str, str]
) -> None:
    for case in cases:
        for gold in case.gold:
            assert gold.doc_id in corpus_texts, (
                f"{case.id}: gold doc_id {gold.doc_id!r} has no corpus file "
                f"(known: {sorted(corpus_texts)})"
            )


def test_every_answer_span_is_verbatim_in_its_document(
    cases: list[EvalCase], corpus_texts: dict[str, str]
) -> None:
    """A span that does not occur in its document resolves to zero chunks at
    eval time and takes the whole harness down with an EvalError. Catch it here,
    where the message can say which case and which sentence."""
    for case in cases:
        for gold in case.gold:
            needle = normalize_text(gold.answer_span)
            assert needle, f"{case.id}: empty answer_span"
            assert needle in corpus_texts[gold.doc_id], (
                f"{case.id}: answer_span is not a verbatim substring of "
                f"{gold.doc_id}\n  span: {gold.answer_span!r}"
            )


def test_gold_never_references_a_chunk_id(cases: list[EvalCase]) -> None:
    """GoldSpan has no chunk field at all; this asserts the JSON agrees."""
    for line in QA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for gold in json.loads(line)["gold"]:
            assert set(gold) == {"doc_id", "answer_span"}, gold


def test_multi_hop_cases_really_span_two_documents(cases: list[EvalCase]) -> None:
    for case in cases:
        if case.type != "multi_hop":
            continue
        doc_ids = {gold.doc_id for gold in case.gold}
        assert len(doc_ids) >= 2, f"{case.id} is labelled multi_hop but cites {doc_ids}"


# ---------------------------------------------------------------------------
# 3. DEFENCE 2 — the unanswerable slice is proven, not asserted
# ---------------------------------------------------------------------------


def test_unanswerable_cases_have_no_gold_and_expect_a_refusal(cases: list[EvalCase]) -> None:
    unanswerable = [case for case in cases if case.type == "unanswerable"]
    assert len(unanswerable) == 10
    for case in unanswerable:
        assert case.gold == [], f"{case.id} is unanswerable but carries gold spans"
        assert case.answer_aliases == [], f"{case.id} is unanswerable but carries an answer"
        assert case.expected_answer_type.value == "refused_no_evidence", case.id


def test_every_unanswerable_entity_is_absent_from_the_corpus(
    cases: list[EvalCase], corpus_blob: str
) -> None:
    """The property that makes the slice evidence rather than an assertion."""
    builder = _load_builder_module()
    markers = {marker: phrase for phrase, marker in builder.OUT_OF_CORPUS_ENTITIES}
    for case in (c for c in cases if c.type == "unanswerable"):
        used = [m for m in markers if f" {m} " in f" {normalize_text(case.question)} "]
        assert len(used) == 1, (
            f"{case.id} should carry exactly one out-of-corpus entity, found {used}"
        )
        marker = used[0]
        assert f" {marker} " not in corpus_blob, (
            f"{case.id} is not unanswerable: {markers[marker]!r} occurs in the corpus"
        )


# ---------------------------------------------------------------------------
# 4. DEFENCE 1 — the divergence cap is enforced, and it is reported
# ---------------------------------------------------------------------------


def test_machine_derived_cases_respect_the_divergence_cap(cases: list[EvalCase]) -> None:
    machine = [c for c in cases if c.provenance in {"mechanical", "entity_substituted"}]
    assert len(machine) == 34
    for case in machine:
        assert 0.0 <= case.stem_overlap <= MAX_STEM_OVERLAP, (
            f"{case.id} has stem_overlap {case.stem_overlap}, above the "
            f"{MAX_STEM_OVERLAP} cap — the builder should have discarded it"
        )


def test_recorded_stem_overlap_matches_a_fresh_measurement(
    cases: list[EvalCase], corpus_texts: dict[str, str]
) -> None:
    """``stem_overlap`` is printed in the README caption, so it has to be real.

    Recompute it for the mechanical tier from the question and its gold sentence
    and check the stored value was not simply written down.
    """
    for case in (c for c in cases if c.provenance == "mechanical"):
        assert len(case.gold) == 1, case.id
        measured = jaccard(case.question, case.gold[0].answer_span)
        assert measured == pytest.approx(case.stem_overlap, abs=5e-5), case.id


def test_authored_cases_are_flagged_and_quarantined(cases: list[EvalCase]) -> None:
    """Nine hand-written cases, all saying so, all carrying a note."""
    authored = [case for case in cases if case.provenance == "authored"]
    assert len(authored) == 9
    committed = {line for line in AUTHORED_PATH.read_text(encoding="utf-8").splitlines() if line}
    assert len(committed) == 9
    for case in authored:
        assert case.notes, f"{case.id} is hand written and must say what it tests"
        assert case.stem_overlap == 0.0, (
            f"{case.id} is authored: it has no mechanical stem, so stem_overlap is not defined"
        )


def test_arithmetic_cases_name_the_tool_they_should_route_to(cases: list[EvalCase]) -> None:
    for case in (c for c in cases if c.type == "arithmetic"):
        assert case.expected_tool in {"calculator", "lookup_metric"}, case.id


# ---------------------------------------------------------------------------
# 5. REGENERATION DETERMINISM — the artifact that proves nothing was retuned
# ---------------------------------------------------------------------------


def test_committed_golden_set_is_byte_identical_to_a_fresh_build(tmp_path: Path) -> None:
    """Re-run the builder and diff the bytes.

    If a question had ever been hand-edited to make a metric look better, this
    is the test that would catch it. It is the single most important assertion
    in the repository.
    """
    fresh = tmp_path / "qa.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--corpus",
            str(CORPUS_DIR),
            "--authored",
            str(AUTHORED_PATH),
            "--out",
            str(fresh),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        env=_child_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"builder failed:\n{result.stdout}\n{result.stderr}"
    assert fresh.read_bytes() == QA_PATH.read_bytes(), (
        "eval/golden/qa.jsonl is not reproducible from the committed corpus. "
        "Run `make golden` and commit the result."
    )


# ---------------------------------------------------------------------------
# 6. THE IMPORT CONSTRAINT — the builder cannot know about the retriever
# ---------------------------------------------------------------------------


def _load_builder_module():  # type: ignore[no-untyped-def]
    """Import scripts/build_golden.py by path, without polluting sys.path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("anchor_build_golden", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registering before execution is required: @dataclass resolves annotations
    # through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_declares_no_forbidden_import() -> None:
    """Static check: nothing retrieval-side is named anywhere in the source."""
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"), filename=str(BUILDER))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported & FORBIDDEN_MODULES == set(), (
        f"scripts/build_golden.py imports {sorted(imported & FORBIDDEN_MODULES)}; "
        "the golden set must be generated in ignorance of the retrieval design"
    )
    assert {name for name in imported if name.startswith("anchor")} == {
        "anchor.ingest",
        "anchor.textutil",
    }, f"unexpected anchor imports: {sorted(n for n in imported if n.startswith('anchor'))}"


def test_builder_pulls_in_no_forbidden_module_transitively() -> None:
    """Dynamic check, in a clean interpreter: what does importing it actually load?

    A static check alone would miss a forbidden module arriving through
    ``anchor.ingest``. A subprocess is used so an unrelated test that already
    imported the retriever cannot make this one pass by accident.
    """
    probe = (
        "import importlib.util, sys, json\n"
        f"spec = importlib.util.spec_from_file_location('bg', r'{BUILDER}')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['bg'] = module\n"
        "spec.loader.exec_module(module)\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('anchor'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=_child_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout.strip().splitlines()[-1]))
    assert loaded & FORBIDDEN_MODULES == set(), (
        f"importing the builder loads {sorted(loaded & FORBIDDEN_MODULES)}"
    )


# ---------------------------------------------------------------------------
# 7. the frozen human labels behind judge validation
# ---------------------------------------------------------------------------


def test_human_labels_are_frozen_and_balanced(corpus_blob: str) -> None:
    """30 rows, 12/9/9, every evidence string verbatim from the corpus.

    The text being frozen in the file is why `make judge-check` needs no live
    run and does not drift when retrieval changes.
    """
    lines = [line for line in HUMAN_LABELS_PATH.read_text(encoding="utf-8").splitlines() if line]
    rows = [HumanLabel.model_validate_json(line) for line in lines]
    assert len(rows) == 30
    assert len({row.id for row in rows}) == 30

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.label.value] = counts.get(row.label.value, 0) + 1
    assert counts == {"grounded": 12, "partial": 9, "ungrounded": 9}

    for row in rows:
        assert row.evidence, f"{row.id} has no evidence"
        assert row.note, f"{row.id} has no rationale"
        for evidence in row.evidence:
            assert f" {normalize_text(evidence)} " in corpus_blob, (
                f"{row.id}: evidence is not a verbatim corpus sentence\n  {evidence!r}"
            )
