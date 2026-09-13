# Architecture decisions

Ten decisions, each with the reason it was made and the cost it carries. A
decision recorded without its cost is marketing, so every entry below names what
it gives up.

One rule governs this file: **no measured number is typed here by hand.** Where a
result is relevant, this file names the command that produces it and the
artifact it lands in. The only figures written out are configuration values and
engineering estimates, and the estimates say so.

---

## ADR-001 — SQLite FTS5 plus a numpy matrix, not a vector database

**Decision.** Lexical retrieval is SQLite FTS5 with BM25; dense retrieval is a
float32 `(n, dim)` matrix in `data/vectors.npy` searched by a single matmul.

**Why.** The retrieval pipeline is the thing this repo exists to show. A vector
database would hide ranking, fusion and persistence behind a client library, and
a reviewer could no longer read the interesting code. Two files, no daemon, no
container dependency, and `docker compose up` works offline.

**The cost.** A brute-force scan is linear in corpus size. The rough engineering
crossover — an estimate, not a measurement — is around 10^5 chunks, where an
ANN index starts to matter. That is why `VectorStore` is a Protocol in
`types.py`: swapping the backend is one class, not a rewrite. The second backend
and its parity suite were cut as a non-goal (ADR-009).

---

## ADR-002 — Reciprocal rank fusion, not weighted score fusion

**Decision.** The hybrid mode fuses the BM25 and dense runs with RRF,
`1/(k + rank)`, `k = 60`.

**Why.** BM25 scores and cosine similarities live on different, unbounded,
corpus-dependent scales. Blending them by weight means normalising both, and
every normalisation choice (min-max over what window? z-score over which
population?) is a hidden tuning knob that has to be re-fitted whenever the
corpus changes. RRF reads ranks only, so it is scale-free and has one constant.

**The cost.** RRF discards score magnitude, so it cannot tell a runaway top hit
from a marginal one — a document ranked first by a hair fuses identically to one
ranked first by a mile. The abstain gate compensates by reading the raw score
mass (ADR-007).

---

## ADR-003 — A hashing embedder as the zero-key default

**Decision.** The default embedder hashes tokens into `dim = 256` signed buckets
and L2-normalises. No model is downloaded, ever.

**Why.** The repo must clone and run with no API key and no network. Any
sentence-transformer default would mean a multi-hundred-megabyte download on
first use, which makes CI slow, the container fat and the demo fragile. A
deterministic embedder also makes every run byte-reproducible, which is what
lets the ablation and the golden-set integrity test mean anything.

**The cost.** This is the honest one: a hashing embedder has no semantics. It
cannot match "retention window" to "how long we keep it". So the numbers this
repo reports measure the **pipeline** — retrieval, span resolution, gating,
refusal, citation — and not model quality. Switch `embedding.provider` to
`openai` and the absolute numbers move; the relative deltas are the point.

---

## ADR-004 — A deterministic extractive stub LLM, not a mock

**Decision.** `StubLLM` selects sentences from the packed context by lexical
overlap with the question and emits them with `[Sn]` citation markers.

**Why.** A mock returning a canned string would make every downstream
measurement meaningless: grounding, citation precision and the refusal rate
would all be scoring a fixture. The stub is a real, if primitive, generator — it
can only say what the retrieved context says, so it exercises the whole
verification path honestly and can genuinely fail when retrieval fails.

**The cost.** It cannot synthesise, paraphrase or perform a two-document join,
so `multi_hop` cases are answered poorly on the default profile and
`exact_match` is pessimistic. That is a floor on quality, not a ceiling on the
pipeline, and it is visible per-type in the eval report.

---

## ADR-005 — No agent framework

**Decision.** The tool loop in `agent.py` is written out: a planner, a tool
registry, a `for` loop with budgets, and a fingerprint set for loop detection.

**Why.** The interesting properties of an agent are its bounds — step limit,
wall clock, token budget, argument validation, repeat detection — and every one
of those is about thirty lines. Adopting a framework would add a large
dependency, bury those bounds in configuration, and make the one thing worth
reviewing the one thing a reviewer cannot see.

**The cost.** No streaming, no parallel tool calls, no ecosystem of prebuilt
integrations, and the `LLMPlanner` prompt/parse contract is hand-rolled — which
is why it tolerates a ```json fence, retries once with its own parse error, and
then gives up into a typed `final` rather than crashing.

---

## ADR-006 — An LLM judge seam, with mandatory human validation

**Decision.** `Judge` is a Protocol. `HeuristicJudge` is the offline default and
`LLMJudge` switches on when a key exists. Either way, `make judge-check` scores
the configured judge against 30 frozen human-labelled triples and reports raw
agreement, Cohen's kappa and a 3x3 confusion matrix.

**Why.** Say it plainly: **shipping an unvalidated judge would have been worse
than shipping no judge at all.** An unvalidated judge produces a groundedness
number that looks like evidence, travels into a README, and is not evidence. The
validation command is what converts the judge from a claim into a measurement.

**The cost.** The labels come from one annotator — the author. Kappa therefore
measures judge-versus-author agreement, not judge-versus-truth, and 30 rows is a
small sample. Both caveats are stated in the README and in
`eval/golden/PROTOCOL.md` rather than buried here.

---

## ADR-007 — A scale-free abstain gate

**Decision.** Two features decide whether to answer: the maximum lexical overlap
between the question and any of the top 5 chunks, and the top-1 share of the
top-5 score mass. Both are ratios in `[0, 1]`.

**Why.** A gate that thresholded raw scores would need re-tuning for every
retrieval mode, since BM25 scores, cosine similarities and RRF scores are not
comparable. One re-tuned threshold per mode would make the ablation circular —
each row would be measured against its own private operating point. Ratios mean
one threshold pair is valid for all four modes, so the rows compare.

**The cost.** Lexical overlap is a weak proxy for relevance and will refuse a
correctly-retrieved answer phrased in different words, inflating the false
refusal rate. The operating point is therefore never asserted: `anchor eval`
prints a refusal-threshold sweep so the whole curve is visible.

---

## ADR-008 — Chunk defaults of 512 tokens with 64 of overlap

**Decision.** `chunking.chunk_tokens = 512`, `chunking.chunk_overlap = 64`.

**Why.** Chunking is half the ablation grid, so this default is an empirical
claim and the authority for it is the committed table in
`eval/reports/ablation.md`, written by `make ablate` — it is deliberately not
reproduced here, because no number in this repo's docs is typed by hand. Read
the `hybrid @ 512/64` row against the `bm25 @ 512/64` baseline on the same chunk
config.

**The cost, and the reconciliation rule.** If the committed table shows a
different chunk config winning on recall@5, **say so rather than quietly
re-fitting the default** — and the default only stands on two stated grounds:
1024-token chunks buy recall by spending the 1400-token context budget on fewer,
longer blocks (worse citation precision, higher latency), and at `n = 43` a
difference of a few points is not a real difference. If a smaller config wins by
a real margin, the right response is to change the default and re-run the gate.

---

## ADR-009 — Non-goals, cut on purpose

Each of these was in an earlier design and was removed to keep the repo
readable. None is an oversight.

- **Trace persistence, `/v1/traces`, replay, the HTML trace viewer** — a trace is
  returned inline when `include_trace` is set and then discarded. Persistence
  added a schema, a retention story and an endpoint surface, and measured nothing.
- **A second `VectorStore` backend and its parity suite** — the Protocol seam
  proves the design; a second implementation only proved it twice (ADR-001).
- **Threshold calibration** — `eval/thresholds.yaml` is generated from an
  observed run minus a fixed slack. A calibration routine would be one more
  thing to trust and to validate.
- **PII redaction and injection defence** — both are real problems that deserve a
  real threat model and a red-team corpus. A token gesture at either would be
  security theatre in a portfolio repo.
- **Auth, multi-tenancy, rate limiting** — this service has no users to separate.
- **Streaming responses** — grounding verifies an answer after generation; the
  answer type is not known until it is complete, so streaming would mean
  streaming text that might still be refused.

---

## ADR-010 — No number is typed by hand, and CI enforces it

**Decision.** Every number in `README.md` and `eval/thresholds.yaml` is written
there by a shipped command: `scripts/update_readme.py --write` for the README
table, `anchor thresholds` for the floors, and `anchor ablate` / `anchor
taxonomy` / `anchor judge-check` for the reports under `eval/reports/`.

**Why.** A hand-typed metric is a number that was true once, on a machine that
no longer exists, for code that has since changed. It is the single easiest way
for a repo to lie without anyone intending to. Making the numbers generated
makes them falsifiable.

**Enforcement.** CI runs `anchor ingest`, `make eval`, then `make gate`, which
fails the build when a metric breaches its floor, and `make readme-check`, which
fails when the committed README table no longer matches a fresh run. CI also
regenerates `eval/golden/qa.jsonl` and fails on any diff, which is what proves
no golden question was quietly retuned to make a score go up.

**The cost.** Every genuine improvement costs a regeneration step, and a
threshold file generated from one run encodes that run's noise — which is what
`eval.threshold_slack` absorbs, at the price of a gate that is looser than the
observed numbers.
