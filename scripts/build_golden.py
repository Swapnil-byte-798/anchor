#!/usr/bin/env python3
"""Generate eval/golden/qa.jsonl mechanically from the RAW corpus text.

WHY THIS FILE EXISTS
--------------------
A candidate who writes both the corpus and the questions has measured nothing:
the questions inherit the corpus's vocabulary, and the retriever gets quietly
tuned until they pass. This script is the first of four defences against that.

It reads corpus text and nothing else. It has no import of — and no knowledge
of — the chunker, the embedder, the retriever, the abstain gate or the prompts,
so it CANNOT be tuned towards them even by accident. The forbidden set is:

    anchor.retrieval  anchor.embed  anchor.store
    anchor.llm        anchor.judge  anchor.service

``tests/test_golden_integrity.py`` asserts that this module's transitive import
set contains none of them, and asserts that re-running this script reproduces
the committed ``qa.jsonl`` byte for byte. That second assertion is the artifact
that proves no question was ever quietly retuned to make a score go up.

THE PIPELINE (five deterministic steps, no randomness anywhere)
---------------------------------------------------------------
(a) sentence-split each document section; keep sentences matching one of six
    typed fact extractors: duration, money, percent, latency_ms,
    count_threshold, identifier;
(b) the matched span becomes the answer; the rest of the sentence plus the
    document's heading path becomes the raw stem;
(c) LEXICAL DIVERGENCE — a fixed SYNONYM map rewrites the stem, the clause's
    leading auxiliary is fronted so the result reads as a question, and the
    stem is rendered through a template keyed by fact type;
(d) the case is DISCARDED unless jaccard(question, source_sentence) <= 0.5.
    The achieved overlap is stored on every case as ``stem_overlap``;
(e) gold = [{doc_id, answer_span: the verbatim source sentence}] — never a
    chunk id, because chunk ids would make the chunking ablation circular.

Then two more tiers:

  * ENTITY SUBSTITUTION builds the provably-unanswerable slice by swapping the
    subject of a mechanical case for an out-of-corpus entity, ASSERTING that
    the entity occurs zero times in the normalised corpus;
  * eval/golden/authored_seed.jsonl is merged in last, after re-verifying that
    every authored gold span really is a verbatim normalised substring of its
    document.

Usage:
    python scripts/build_golden.py                     # writes the default path
    python scripts/build_golden.py --out /tmp/qa.jsonl # what the test does
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# The ONLY two anchor imports permitted here, both of which are pure text:
# `load_corpus` reads files off disk and `textutil` defines tokens/sentences.
# Neither knows anything about how those files are later indexed or searched.
from anchor.ingest import load_corpus
from anchor.textutil import content_tokens, jaccard, normalize_text, split_sentences

# ---------------------------------------------------------------------------
# constants — everything below is fixed. There is no randomness in this script;
# SEED exists so that the one place an arbitrary choice is made (which
# out-of-corpus entity is assigned to which case) is reproducible and stated.
# ---------------------------------------------------------------------------

SEED = 1337

DEFAULT_CORPUS_DIR = "corpus"
DEFAULT_OUT_PATH = "eval/golden/qa.jsonl"
DEFAULT_AUTHORED_SEED_PATH = "eval/golden/authored_seed.jsonl"

# corpus/README.md documents the corpus rather than being part of it, so no
# question is ever generated from it. `paths.corpus_exclude` in
# config/default.yaml applies the SAME exclusion on the ingest path, so the
# served corpus and the evaluated corpus are the same six documents — they were
# not always, and the note about the corpus was retrievable as a seventh.
# Keep the two lists in step: this builder deliberately does not import
# Settings (it reads raw corpus text and nothing else), which is why the
# exclusion is spelled out in both places.
SKIP_DOC_IDS = frozenset({"readme"})

# The hard divergence cap from the corpus plan. A generated question that
# shares more than half its content tokens with the sentence it came from is
# not a question, it is a copy, and it is thrown away.
MAX_STEM_OVERLAP = 0.5

# Target composition. The builder fails loudly rather than shipping a short set.
N_MECHANICAL = 24
N_UNANSWERABLE = 10
N_AUTHORED_MULTI_HOP = 5
N_AUTHORED_ARITHMETIC = 4

MIN_CLAUSE_CONTENT_TOKENS = 3
MAX_CLAUSE_CONTENT_TOKENS = 12
MIN_SENTENCE_CONTENT_TOKENS = 6

# ---------------------------------------------------------------------------
# (a) the six typed fact extractors, in priority order
#
# Priority matters and is fixed: "850 ms" must be read as a latency and not as
# a count, "$4,200" as money and not as a bare number. The FIRST extractor that
# matches a sentence wins, and within it the FIRST match wins.
# ---------------------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
_SMALL_WORDS = r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"

_DURATION_UNITS = r"working\s+days?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|quarters?"
_COUNT_UNITS = (
    r"consecutive\s+failures|candidate\s+passages|concurrent\s+connections"
    r"|dimensional\s+vectors|requests?|calls?|documents?|connections?|jobs?"
    r"|passages?|tenants?|regions?|failures?|retries"
)

FACT_EXTRACTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("latency_ms", re.compile(rf"\b{_NUM}\s*ms\b")),
    ("percent", re.compile(rf"\b{_NUM}%")),
    ("money", re.compile(rf"\${_NUM}")),
    ("duration", re.compile(rf"\b(?:{_NUM}|{_SMALL_WORDS})\s+(?:{_DURATION_UNITS})\b")),
    # Digits only: "one tenant index" is not a threshold, "5,000 requests" is.
    ("count_threshold", re.compile(rf"\b{_NUM}\s+(?:{_COUNT_UNITS})\b")),
    (
        "identifier",
        re.compile(
            r"\bSEV-[1-4]\b|/v1/[a-z/]+|\b(?:us-east-1|eu-west-1|ap-south-1)\b|\btnt-\d{4}\b"
        ),
    ),
)

# ---------------------------------------------------------------------------
# (c) lexical divergence
#
# A fixed synonym map, applied ONE PASS over the original tokens so no
# replacement can feed another. Keys are lowercase; a token is only rewritten
# when it is purely alphabetic, which is what keeps SEV-2, /v1/generate,
# Retry-After and tnt-0471 intact.
# ---------------------------------------------------------------------------

SYNONYMS: dict[str, str] = {
    "acknowledge": "confirm",
    "acknowledged": "confirmed",
    "acknowledgement": "confirmation",
    "alert": "alarm",
    "approved": "authorised",
    "availability": "uptime",
    "billed": "charged",
    "billing": "invoicing",
    "budget": "allowance",
    "budgets": "allowances",
    "burst": "spike",
    "channel": "room",
    "concurrent": "simultaneous",
    "connections": "links",
    "daily": "per-day",
    "degraded": "impaired",
    "delete": "remove",
    "deleted": "removed",
    "deletion": "removal",
    "deploy": "release",
    "drain": "shed",
    "drift": "skew",
    "endpoint": "route",
    "endpoints": "routes",
    "engineer": "staffer",
    "escalate": "raise",
    "escalation": "handover",
    "exceed": "surpass",
    "exceeding": "surpassing",
    "exceeds": "surpasses",
    "expire": "lapse",
    "expires": "lapses",
    "freshness": "currency",
    "guarantee": "promise",
    "index": "catalogue",
    "indexes": "catalogues",
    "latency": "lag",
    "limit": "ceiling",
    "limited": "restricted",
    "limits": "ceilings",
    "log": "record",
    "logs": "records",
    "monthly": "per-month",
    "overnight": "nightly",
    "owner": "maintainer",
    "paged": "notified",
    "primary": "lead",
    "published": "documented",
    "queued": "buffered",
    "rebuild": "regenerate",
    "rebuilt": "regenerated",
    "region": "zone",
    "regions": "zones",
    "rejected": "refused",
    "rejects": "refuses",
    "retain": "keep",
    "retained": "kept",
    "retains": "keeps",
    "retention": "storage",
    "review": "check",
    "reviewed": "checked",
    "rollback": "revert",
    "rotation": "roster",
    "secondary": "backup",
    "severity": "priority",
    "shift": "stint",
    "sustained": "steady",
    "target": "goal",
    "team": "group",
    "tenant": "customer",
    "tenants": "customers",
    "threshold": "limit",
    "uplift": "boost",
    "window": "period",
    "windows": "periods",
}

# The clause's first verb is fronted only when it is an auxiliary, which turns
# "gateway request logs are retained for" into "are gateway request records kept
# for" — a readable question. When the first verb is a full verb instead, the
# clause is left alone and an elliptical template is used. One rule, two shapes,
# no part-of-speech tagger.
AUXILIARIES = frozenset(
    {"is", "are", "was", "were", "has", "have", "had", "must", "may", "can", "does", "do", "will"}
)

FULL_VERBS = frozenset(
    """accepts adds allows answers applies appears assumes began borrows carries closed closes
    completes confirmed covers declared demonstrates disagree draws drafts drops emits enters failed
    exceeds exists expires extends follows hands holds judged keeps lags links means moved opens
    owns points posts pulls raises ran reached rebuilds receives refills rejoins reported requires
    resurrect returns reviews rose runs satisfy scales sells sends serves sets settled signs
    starves starts stated stayed surprises survives takes tells tolerate uploaded went
    writes""".split()
)

# Fronting the auxiliary of a SUBORDINATE clause produces nonsense — "a tenant
# index that has not been rebuilt" must not become "has a tenant index that not
# been rebuilt". If any of these appears before the auxiliary, leave the clause
# in its own word order and use the elliptical template instead.
RELATIVIZERS = frozenset(
    {"that", "which", "who", "whose", "when", "where", "because", "if", "while", "whether", "so"}
)

# Conjunctions and complementisers genuinely dangle when a clause is cut short.
# Prepositions do NOT: English stranding makes "how long are records kept for?"
# better than "how long are records kept?", so they are deliberately kept.
_TRAILING_STRIP = frozenset(
    """and or but so then that which because while when where whether if both either
    neither""".split()
)

_DETERMINERS = frozenset(
    """the a an every each this that these those both all any some no another one""".split()
)

# Names that keep their capital when a clause is re-used as a question stem.
# Everything else that merely happens to start a sentence is lower-cased.
PROPER_NOUNS = frozenset({"meridian", "starter", "scale", "enterprise"})

_VOWEL_SOUND = re.compile(r"^[aeiou]", re.IGNORECASE)

# ... and the mirror image, for a clause taken from the far side of the span.
_LEADING_STRIP = frozenset(
    """and or but so then that which because while when where as per of to for at in on by with
    from within after before over under between into""".split()
)

# (b) per-type question templates. Two shapes: the first is used when the
# auxiliary was fronted, the second when the clause kept its own word order.
TEMPLATES: dict[str, str] = {
    "duration": "how long",
    "money": "what dollar amount",
    "percent": "what percentage",
    "latency_ms": "what latency in milliseconds",
    "count_threshold": "how many",
    "identifier": "which identifier",
}

FRONTED_TEMPLATE = "In the {heading} section, {interrogative} {clause}?"
ELLIPTICAL_TEMPLATE = "In the {heading} section: {clause} — {interrogative}?"

# ---------------------------------------------------------------------------
# (e) answer aliases
# ---------------------------------------------------------------------------

NUMBER_WORDS: dict[str, str] = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "14": "fourteen",
    "15": "fifteen",
    "20": "twenty",
    "24": "twenty-four",
    "25": "twenty-five",
    "30": "thirty",
    "40": "forty",
    "45": "forty-five",
    "50": "fifty",
    "60": "sixty",
    "72": "seventy-two",
    "90": "ninety",
    "100": "one hundred",
    "180": "one hundred and eighty",
    "400": "four hundred",
    "600": "six hundred",
}
WORD_NUMBERS: dict[str, str] = {word: digits for digits, word in NUMBER_WORDS.items()}

# ---------------------------------------------------------------------------
# DEFENCE 2 — the unanswerable slice.
#
# Each entity is a plausible, on-domain noun phrase that does not occur in the
# corpus. The marker is the one token whose absence is ASSERTED below: if it
# turns up anywhere in the normalised corpus the case is discarded, so
# unanswerability is a checked property of the corpus and not an opinion.
# ---------------------------------------------------------------------------

OUT_OF_CORPUS_ENTITIES: tuple[tuple[str, str], ...] = (
    ("the Helios queue", "helios"),
    ("the Zurich region", "zurich"),
    ("the Nimbus scheduler", "nimbus"),
    ("the Kestrel tier", "kestrel"),
    ("the Borealis cache", "borealis"),
    ("the Tallinn edge site", "tallinn"),
    ("the Obsidian planner", "obsidian"),
    ("the Marlowe connector", "marlowe"),
    ("the Quillon catalogue", "quillon"),
    ("the Saffron pipeline", "saffron"),
    ("the Vantage broker", "vantage"),
    ("the Perihelion shard", "perihelion"),
)


class GoldenBuildError(RuntimeError):
    """The golden set could not be built. Always names what to fix."""


# ---------------------------------------------------------------------------
# document structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    doc_id: str
    title: str
    heading: str
    text: str


_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_SKIP_LINE = re.compile(r"^\s*(?:[-*+>|]|\d+\.)\s")


def heading_label(heading: str) -> str:
    """Lower-case the section heading and drop a leading article.

    Without this "The bands" would render as "In the the bands section".
    """
    label = heading.strip().lower()
    for article in ("the ", "a ", "an "):
        if label.startswith(article):
            return label[len(article) :]
    return label


def split_sections(doc_id: str, text: str) -> list[Section]:
    """Split a markdown document into ``##`` sections.

    The heading used in the question is the SECTION leaf, deliberately not the
    full path including the document title: putting "Meridian Data Retention
    Policy" into every question would hand the retriever the answer document for
    free and quietly inflate recall.
    """
    title = ""
    heading = ""
    buffer: list[str] = []
    sections: list[Section] = []
    in_fence = False

    def flush() -> None:
        body = " ".join(line.strip() for line in buffer if line.strip())
        if body and heading:
            sections.append(Section(doc_id=doc_id, title=title, heading=heading, text=body))
        buffer.clear()

    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _SKIP_LINE.match(line):
            continue
        match = _ATX.match(line)
        if match:
            flush()
            level, label = len(match.group(1)), match.group(2).strip()
            if level == 1:
                title = label
                heading = ""
            else:
                heading = label
            continue
        buffer.append(line)
    flush()
    return sections


# ---------------------------------------------------------------------------
# (c) stem rewriting
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z]+")


def apply_synonyms(words: list[str]) -> list[str]:
    """One pass, alphabetic tokens only, so identifiers survive untouched."""
    out: list[str] = []
    for word in words:
        bare = word.strip(".,;:()'\"")
        if bare and _WORD.fullmatch(bare):
            replacement = SYNONYMS.get(bare.lower())
            if replacement is not None:
                out.append(word.replace(bare, replacement))
                continue
        out.append(word)
    return out


def strip_edges(words: list[str]) -> list[str]:
    """Drop dangling punctuation and function words from both ends of a clause."""
    trimmed = list(words)
    verb_budget = 1
    while trimmed:
        head = trimmed[0].strip(".,;:()'\"").lower()
        if head in _LEADING_STRIP or head == "":
            trimmed.pop(0)
            continue
        if head in FULL_VERBS and verb_budget:
            verb_budget -= 1
            trimmed.pop(0)
            continue
        break
    while trimmed:
        tail = trimmed[-1].strip(".,;:()'\"").lower()
        if tail in _TRAILING_STRIP or tail == "":
            trimmed.pop()
            continue
        break
    return [w.strip(".,;:") for w in trimmed if w.strip(".,;:")]


def fix_articles(clause: str) -> str:
    """Repair a/an after a synonym swap: "a allowance breach" -> "an allowance"."""
    words = clause.split()
    for index in range(len(words) - 1):
        lowered = words[index].lower()
        if lowered not in {"a", "an"}:
            continue
        vowel = bool(_VOWEL_SOUND.match(words[index + 1]))
        fixed = "an" if vowel else "a"
        words[index] = fixed.capitalize() if words[index][0].isupper() else fixed
    return " ".join(words)


def front_auxiliary(words: list[str]) -> tuple[list[str], bool]:
    """Move the clause's leading auxiliary to the front, if it has one.

    Returns ``(words, fronted)``. The auxiliary is only moved when it is the
    FIRST verb in the clause; "SEV-1 means the gateway is unavailable" keeps its
    order because ``means`` comes first, which stops the rule producing nonsense.
    """
    if words and words[0].strip(".,;:()'\"").lower() == "there":
        return words, False  # existential "there is ..." must not be inverted
    for index, word in enumerate(words):
        bare = word.strip(".,;:()'\"").lower()
        if bare in FULL_VERBS or bare in RELATIVIZERS:
            return words, False
        if bare in AUXILIARIES:
            if index == 0:
                return words, False
            reordered = [words[index], *words[:index], *words[index + 1 :]]
            return reordered, True
    return words, False


def cap_clause(words: list[str]) -> list[str]:
    """Keep at most MAX_CLAUSE_CONTENT_TOKENS content tokens, stopwords free."""
    kept: list[str] = []
    budget = MAX_CLAUSE_CONTENT_TOKENS
    for word in words:
        if content_tokens(word):
            if budget == 0:
                break
            budget -= 1
        kept.append(word)
    return kept


def build_clause(raw: str) -> tuple[str, bool] | None:
    """Rewrite one side of a split sentence into a question clause."""
    words = strip_edges(raw.split())
    if not words:
        return None
    words = cap_clause(apply_synonyms(words))
    # Lower-case a leading determiner BEFORE fronting, so "The published p95
    # budget" reads as "the documented p95 allowance" wherever it ends up.
    # Anything else — Meridian, Gateway, SEV-1, Retry-After — keeps its case.
    if words and words[0].istitle() and words[0].lower() not in PROPER_NOUNS:
        words[0] = words[0].lower()
    if not any(ch.isalpha() for ch in words[0]):
        return None  # a stem that opens on a bare date or number is not a question
    words, fronted = front_auxiliary(words)
    words = strip_edges(words)
    if len(content_tokens(" ".join(words))) < MIN_CLAUSE_CONTENT_TOKENS:
        return None
    clause = fix_articles(" ".join(words).strip())
    if not clause:
        return None
    return clause, fronted


def interrogative_for(fact_type: str, span: str) -> str:
    """The wh-phrase. ``count_threshold`` borrows the span's own unit so the
    question says "how many requests" rather than a bare "how many"."""
    base = TEMPLATES[fact_type]
    if fact_type == "count_threshold":
        unit = " ".join(w for w in span.split()[1:] if w)
        if unit:
            return f"{base} {unit}"
    return base


# ---------------------------------------------------------------------------
# (e) aliases
# ---------------------------------------------------------------------------


def build_aliases(span: str) -> list[str]:
    """The target span plus number-word variants and a comma-free variant."""
    aliases = [span]
    head, _, tail = span.partition(" ")
    digits = head.replace(",", "")
    if digits in NUMBER_WORDS:
        aliases.append(f"{NUMBER_WORDS[digits]} {tail}".strip())
    if head.lower() in WORD_NUMBERS:
        aliases.append(f"{WORD_NUMBERS[head.lower()]} {tail}".strip())
    if "," in span:
        aliases.append(span.replace(",", ""))
    if head != digits and digits in NUMBER_WORDS:
        aliases.append(f"{digits} {tail}".strip())
    out: list[str] = []
    for alias in aliases:
        if alias and alias not in out:
            out.append(alias)
    return out


# ---------------------------------------------------------------------------
# (a)-(d) the mechanical tier
# ---------------------------------------------------------------------------


@dataclass
class Mechanical:
    doc_id: str
    order: int
    question: str
    fact_type: str
    span: str
    sentence: str
    heading: str
    overlap: float
    clause: str
    fronted: bool


def extract_mechanical(sections: Iterable[Section]) -> tuple[list[Mechanical], int]:
    """Run steps (a)-(d) over every section. Returns (kept, discarded)."""
    kept: list[Mechanical] = []
    seen_questions: set[str] = set()
    discarded = 0
    order = 0
    for section in sections:
        for sentence in split_sentences(section.text):
            order += 1
            if len(content_tokens(sentence)) < MIN_SENTENCE_CONTENT_TOKENS:
                continue
            found = None
            for fact_type, pattern in FACT_EXTRACTORS:
                match = pattern.search(sentence)
                if match:
                    found = (fact_type, match)
                    break
            if found is None:
                continue
            fact_type, match = found
            span = match.group(0).strip()

            prefix, suffix = sentence[: match.start()], sentence[match.end() :]
            # Keep only the clause adjacent to the span: a colon or semicolon is
            # a hard boundary, so the unrelated half of a compound sentence never
            # ends up in the question.
            prefix = re.split(r"[:;]", prefix)[-1]
            suffix = re.split(r"[:;]", suffix)[0]
            side = prefix if len(content_tokens(prefix)) >= MIN_CLAUSE_CONTENT_TOKENS else suffix
            built = build_clause(side)
            if built is None:
                continue
            clause, fronted = built

            template = FRONTED_TEMPLATE if fronted else ELLIPTICAL_TEMPLATE
            question = template.format(
                heading=heading_label(section.heading),
                interrogative=interrogative_for(fact_type, span),
                clause=clause,
            )
            overlap = jaccard(question, sentence)
            if overlap > MAX_STEM_OVERLAP:
                discarded += 1
                continue
            if question in seen_questions:
                continue
            seen_questions.add(question)
            kept.append(
                Mechanical(
                    doc_id=section.doc_id,
                    order=order,
                    question=question,
                    fact_type=fact_type,
                    span=span,
                    sentence=sentence,
                    heading=section.heading,
                    overlap=round(overlap, 4),
                    clause=clause,
                    fronted=fronted,
                )
            )
    return kept, discarded


def _interleave_fact_types(bucket: list[Mechanical]) -> list[Mechanical]:
    """Reorder one document's candidates so the six fact types alternate.

    Without this a document full of durations contributes nothing but durations,
    and the money and latency extractors never reach the committed set.
    """
    order = [fact_type for fact_type, _ in FACT_EXTRACTORS]
    by_type: dict[str, list[Mechanical]] = {fact_type: [] for fact_type in order}
    for item in bucket:
        by_type[item.fact_type].append(item)
    out: list[Mechanical] = []
    depth = 0
    while len(out) < len(bucket):
        for fact_type in order:
            items = by_type[fact_type]
            if depth < len(items):
                out.append(items[depth])
        depth += 1
    return out


def select_spread(items: list[Mechanical], wanted: int) -> list[Mechanical]:
    """Take ``wanted`` cases, round-robin across documents in doc_id order.

    Deterministic and balanced on two axes: no document can dominate the set just
    because it happens to contain more numbers than the others, and no single
    fact type can crowd the others out within a document.
    """
    buckets: dict[str, list[Mechanical]] = {}
    for item in sorted(items, key=lambda m: (m.doc_id, m.order)):
        buckets.setdefault(item.doc_id, []).append(item)
    buckets = {doc_id: _interleave_fact_types(bucket) for doc_id, bucket in buckets.items()}
    chosen: list[Mechanical] = []
    cursor = 0
    while len(chosen) < wanted:
        progressed = False
        for doc_id in sorted(buckets):
            bucket = buckets[doc_id]
            if cursor < len(bucket):
                chosen.append(bucket[cursor])
                progressed = True
                if len(chosen) == wanted:
                    break
        if not progressed:
            break
        cursor += 1
    return chosen


# ---------------------------------------------------------------------------
# DEFENCE 2 — entity substitution
# ---------------------------------------------------------------------------


def swap_subject(clause: str, entity: str, fronted: bool) -> str:
    """Replace the clause's leading noun phrase with an out-of-corpus entity.

    The predicate is left untouched, so the question stays plausible and on
    domain — which is exactly what makes refusing it a real test rather than a
    nonsense-detector. When the clause has already had its auxiliary fronted the
    auxiliary is held back, so "are Gateway request records kept for" becomes
    "are the Quillon catalogue records kept for" and not "the Quillon catalogue
    records kept for".
    """
    words = clause.split()
    head = words[:1] if fronted else []
    body = words[1:] if fronted else words
    cut = 0
    for index, word in enumerate(body):
        if content_tokens(word):
            cut = index + 1
            break
    return fix_articles(" ".join([*head, *entity.split(), *body[cut:]]).strip())


def build_unanswerable(
    chosen: list[Mechanical], normalised_corpus: str, wanted: int
) -> list[tuple[Mechanical, str, str, str]]:
    """Return ``(source, question, entity, marker)`` for the unanswerable tier."""
    if not chosen:
        return []
    padded = f" {normalised_corpus} "
    # A stride coprime with the number of mechanical cases walks the whole list
    # without repeating, so the unanswerable slice spans every document rather
    # than clustering on whichever few happen to sit at the front.
    stride = 5
    picks = [chosen[(index * stride) % len(chosen)] for index in range(wanted)]
    out: list[tuple[Mechanical, str, str, str]] = []
    seen_questions: set[str] = set()
    for position, source in enumerate(picks):
        entity, marker = OUT_OF_CORPUS_ENTITIES[(SEED + position) % len(OUT_OF_CORPUS_ENTITIES)]
        # The assertion that makes unanswerability a checked property.
        if f" {normalize_text(marker)} " in padded:
            continue
        clause = swap_subject(source.clause, entity, source.fronted)
        template = FRONTED_TEMPLATE if source.fronted else ELLIPTICAL_TEMPLATE
        question = template.format(
            heading=heading_label(source.heading),
            interrogative=interrogative_for(source.fact_type, source.span),
            clause=clause,
        )
        if question in seen_questions:
            continue
        seen_questions.add(question)
        out.append((source, question, entity, marker))
        if len(out) == wanted:
            break
    return out


# ---------------------------------------------------------------------------
# DEFENCE 4 — the authored tier, quarantined and re-verified
# ---------------------------------------------------------------------------


def load_authored(path: Path, docs_by_id: dict[str, str]) -> list[dict[str, object]]:
    """Read the hand-written seed cases and re-verify every gold span."""
    if not path.is_file():
        raise GoldenBuildError(f"authored seed file {path} is missing")
    cases: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenBuildError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        case_id = case.get("id", f"<line {line_no}>")
        for gold in case.get("gold", []):
            doc_id = gold["doc_id"]
            if doc_id not in docs_by_id:
                raise GoldenBuildError(
                    f"authored case {case_id}: gold doc_id {doc_id!r} is not a corpus document "
                    f"(known: {', '.join(sorted(docs_by_id))})"
                )
            needle = normalize_text(gold["answer_span"])
            if needle not in docs_by_id[doc_id]:
                raise GoldenBuildError(
                    f"authored case {case_id}: answer_span is not a verbatim substring of "
                    f"{doc_id}. Copy the sentence out of corpus/{doc_id}.md exactly.\n"
                    f"  span: {gold['answer_span']!r}"
                )
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

# Field order is EvalCase's declaration order. Every field is written on every
# line, defaults included, so a reader never has to know what was omitted.
CASE_FIELDS = (
    "id",
    "question",
    "type",
    "provenance",
    "answer_aliases",
    "gold",
    "expected_answer_type",
    "expected_tool",
    "stem_overlap",
    "source_doc_id",
    "notes",
)


def case_to_line(case: dict[str, object]) -> str:
    ordered = {field: case[field] for field in CASE_FIELDS}
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)


def summarise(cases: list[dict[str, object]], discarded: int, candidates: int) -> str:
    """The composition table printed by `make golden`.

    The mean and max achieved overlap are printed here and nowhere else is the
    single biggest bias in the set allowed to stay a footnote.
    """
    rows: dict[tuple[str, str], list[float]] = {}
    for case in cases:
        key = (str(case["type"]), str(case["provenance"]))
        rows.setdefault(key, []).append(float(case["stem_overlap"]))
    lines = [
        "",
        f"golden set: {len(cases)} cases "
        f"({candidates} mechanical candidates, {discarded} discarded above "
        f"jaccard {MAX_STEM_OVERLAP})",
        "",
        f"  {'type':<14} {'provenance':<19} {'n':>3}  {'mean overlap':>12} {'max':>6}",
        f"  {'-' * 14} {'-' * 19} {'-' * 3}  {'-' * 12} {'-' * 6}",
    ]
    for (case_type, provenance), overlaps in sorted(rows.items()):
        mean = sum(overlaps) / len(overlaps)
        lines.append(
            f"  {case_type:<14} {provenance:<19} {len(overlaps):>3}  "
            f"{mean:>12.3f} {max(overlaps):>6.3f}"
        )
    machine_tiers = {"mechanical", "entity_substituted"}
    mechanical = [float(c["stem_overlap"]) for c in cases if c["provenance"] in machine_tiers]
    if mechanical:
        lines += [
            "",
            f"  machine-derived stem_overlap: mean {sum(mechanical) / len(mechanical):.3f}, "
            f"max {max(mechanical):.3f} (cap {MAX_STEM_OVERLAP})",
        ]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build(corpus_dir: str, authored_seed: str) -> tuple[list[dict[str, object]], str]:
    docs = [doc for doc in load_corpus(corpus_dir) if doc.doc_id not in SKIP_DOC_IDS]
    if not docs:
        raise GoldenBuildError(f"no usable documents found in {corpus_dir!r}")
    docs_by_id = {doc.doc_id: normalize_text(doc.text) for doc in docs}
    normalised_corpus = " ".join(docs_by_id[doc_id] for doc_id in sorted(docs_by_id))

    sections: list[Section] = []
    for doc in docs:
        sections.extend(split_sections(doc.doc_id, doc.text))

    candidates, discarded = extract_mechanical(sections)
    chosen = select_spread(candidates, N_MECHANICAL)
    if len(chosen) < N_MECHANICAL:
        raise GoldenBuildError(
            f"only {len(chosen)} mechanical cases survived the "
            f"jaccard <= {MAX_STEM_OVERLAP} divergence cap, need {N_MECHANICAL}. "
            "ADD FACTS TO THE CORPUS — never relax the cap."
        )

    cases: list[dict[str, object]] = []
    for index, item in enumerate(chosen, start=1):
        cases.append(
            {
                "id": f"m-{index:04d}",
                "question": item.question,
                "type": "factual",
                "provenance": "mechanical",
                "answer_aliases": build_aliases(item.span),
                "gold": [{"doc_id": item.doc_id, "answer_span": item.sentence}],
                "expected_answer_type": "grounded",
                "expected_tool": None,
                "stem_overlap": item.overlap,
                "source_doc_id": item.doc_id,
                "notes": f"fact type: {item.fact_type}; section: {item.heading}",
            }
        )

    substituted = build_unanswerable(chosen, normalised_corpus, N_UNANSWERABLE)
    if len(substituted) < N_UNANSWERABLE:
        raise GoldenBuildError(
            f"only {len(substituted)} entity-substituted cases could be proven unanswerable, "
            f"need {N_UNANSWERABLE}. Add entries to OUT_OF_CORPUS_ENTITIES."
        )
    for index, (source, question, entity, marker) in enumerate(substituted, start=1):
        cases.append(
            {
                "id": f"u-{index:04d}",
                "question": question,
                "type": "unanswerable",
                "provenance": "entity_substituted",
                "answer_aliases": [],
                "gold": [],
                "expected_answer_type": "refused_no_evidence",
                "expected_tool": None,
                "stem_overlap": round(jaccard(question, source.sentence), 4),
                "source_doc_id": source.doc_id,
                "notes": (
                    f"out-of-corpus entity: {entity} (marker {marker!r} occurs 0 times in the "
                    f"normalised corpus); derived from {source.doc_id}"
                ),
            }
        )

    authored = load_authored(Path(authored_seed), docs_by_id)
    by_type: dict[str, int] = {}
    for case in authored:
        by_type[str(case["type"])] = by_type.get(str(case["type"]), 0) + 1
    expected = {"multi_hop": N_AUTHORED_MULTI_HOP, "arithmetic": N_AUTHORED_ARITHMETIC}
    if by_type != expected:
        raise GoldenBuildError(f"{authored_seed} must hold exactly {expected}, found {by_type}")
    cases.extend(authored)

    return cases, summarise(cases, discarded, len(candidates))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the golden set mechanically from the raw corpus text. "
            "Reads corpus/ and eval/golden/authored_seed.jsonl; knows nothing "
            "about the retriever."
        )
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_DIR, help="corpus directory")
    parser.add_argument("--authored", default=DEFAULT_AUTHORED_SEED_PATH, help="authored seed file")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="output jsonl path")
    parser.add_argument("--quiet", action="store_true", help="suppress the composition table")
    args = parser.parse_args(argv)

    try:
        cases, summary = build(args.corpus, args.authored)
    except GoldenBuildError as exc:
        print(f"build_golden: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{case_to_line(case)}\n" for case in cases), encoding="utf-8")
    if not args.quiet:
        print(summary)
        print(f"wrote {len(cases)} cases to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
