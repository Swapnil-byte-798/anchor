# Anchor

**A grounded RAG + tool-agent service whose centrepiece is the measurement layer
that says where its retrieval actually fails.** Six synthetic documents, 43 golden
cases, five commands that produce every number on this page. Zero API keys, zero
network calls.

**Optional MCP add-on:** expose the existing document search, calculator and
metric lookup tools to an MCP host over local stdio. The normal API, CLI and
evaluation workflow are unchanged. See [the setup and limits](docs/MCP.md).

## The headline finding

> **The hybrid retriever this repo ships is worse than the 40 lines of SQLite FTS5
> it is built on top of, and its own ablation says so in bold.**

BM25-only scores **1.000 recall@5** in all six chunk configurations. The shipped
`hybrid` default scores **0.924** — **-0.076** against its own baseline. Across all
**24 ablation cells not one strategy beat BM25**; the best non-baseline cell in the
grid is `hybrid_rerank` at **-0.045**.

That result is generated, not written. `eval/reports/ablation.md` is produced by
`anchor ablate` and the verdict paragraph in it is machine-composed from the grid:

> "No strategy beat the BM25 baseline in this run. The best non-baseline delta is
> -0.045 recall@5 (`hybrid_rerank` at 96/0); every other cell is worse. That is the
> honest result and it is left standing here rather than tuned away."

**The most likely cause is a property of the benchmark, not a defect in the dense
leg.** 24 of the 43 questions are mechanically derived from corpus sentences, and
although the builder caps their lexical overlap with the source sentence at 0.5,
they still speak the corpus's own vocabulary. Exact term matching is what BM25 is
best at. A dense retriever earns its place on paraphrase and synonymy, which this
question set is deliberately constrained not to contain much of.

So the default stays `hybrid`, for paraphrase robustness this question set cannot
see. **That is a judgement the table does not support**, and it is stated here
rather than hidden. If you only take one thing from this repo, take that the
measurement was allowed to disagree with the build.

---

## 1. Results

The full 24-row grid, the per-case taxonomy and every judge disagreement are in
**[docs/RESULTS.md](docs/RESULTS.md)**. The summary across the grid:

| mode | recall@5 (min–max over 6 chunk configs) | recall@10 | MRR | Δ recall@5 vs bm25 |
| --- | ---: | ---: | ---: | ---: |
| `bm25` **(baseline)** | 1.000 – 1.000 | 1.000 | 0.955 – 0.970 | — |
| `dense` | 0.864 – 0.909 | 0.924 – 0.939 | 0.786 – 0.833 | -0.136 … -0.091 |
| `hybrid` **(shipped default)** | 0.909 – 0.924 | 0.970 – 1.000 | 0.803 – 0.833 | -0.091 … -0.076 |
| `hybrid_rerank` | 0.939 – 0.955 | 0.985 – 1.000 | 0.960 – 0.975 | -0.061 … -0.045 |

4 strategies x 6 chunk configs = 24 rows, n=43, retrieval only, no generation. The
Δ is measured against the BM25 row **of the same chunk config**, so it answers "what
did the dense leg buy me *here*" rather than mixing two variables.

Two things in that table are worth more than the headline:

* **`hybrid_rerank` edges BM25 on MRR in 4 of the 6 blocks** — 0.975 vs 0.970 in
  three of them, 0.960 vs 0.955 in the fourth — while losing on recall@5 in all six
  and on nDCG@10 in all six. The margin is 0.005 every time. At n=43 one case moving
  from rank 1 to rank 2 moves MRR by 0.012, so a 0.005 gap is smaller than a single
  case. It is not a finding. It is here because leaving it out would be
  cherry-picking in the other direction.
* **Chunking moved recall@5 by at most 0.045** within a single strategy across the
  whole grid. That is a weak result, and the grid's range — 96 to 144 tokens — is
  why it is even measurable. The corpus has 38 `##` sections and the longest is
  **161 tokens**; chunking is section-bounded, so every budget at or above 161
  produces the *same 38-chunk index*. 256/64, 512/64 and 1024/128 are byte-identical
  to the grid's 96/0 row. `anchor ablate` fingerprints every index it builds by the
  sha256 of its sorted chunk ids and banners the configurations that collide, because
  a duplicated index is not a second data point — run
  `anchor ablate --chunk-configs 96x0,256x64,512x64,1024x128` and the report opens
  with `identical index: 96/0, 256/64, 512/64, 1024/128` instead of four data points.
  A grid of 256/512/1024 would have produced four identical blocks and a tautology.

Note the second consequence: the **shipped 512/64 default is the grid's 96/0 row**.
The ablation covers the configuration that actually ships; it is just labelled by
the smallest budget that produces the same index.

### The shipped profile, end to end

Everything below is written by `make readme` from the most recent run.
`scripts/update_readme.py --check` runs in CI and fails the build if a single
character of it drifts from a fresh run — which is not hypothetical; it has caught a
real stale table in this repo's own history, after a corpus change that was never
re-evaluated.

<!-- EVAL:BEGIN -->

<!-- Written by scripts/update_readme.py from eval/reports/latest.json.
     Do not edit by hand: CI runs `--check` and fails when this block is
     stale or hand-edited. -->

**Profile.** llm `stub` | embedder `hashing` | judge `heuristic` | retrieval `hybrid` | chunking 512/64 | config fingerprint `66a15864436b` | n=43 cases

| metric | `hybrid` 512/64 | what it measures |
| --- | ---: | --- |
| **Retrieval** |  | |
| `recall_at_5` | 0.924 | gold chunk in the top 5 |
| `recall_at_10` | 0.985 | gold chunk in the top 10 |
| `precision_at_5` | 0.206 | gold share of the top 5 slots |
| `mrr` | 0.810 | reciprocal rank of the first gold hit |
| `ndcg_at_10` | 0.852 | binary-gain nDCG over the top 10 |
| **Answer** |  | |
| `exact_match` | 0.606 | an answer alias appears verbatim |
| `token_f1` | 0.108 | SQuAD token F1 against the best alias |
| `citation_precision` | 0.636 | cited documents that are gold |
| `citation_recall` | 0.621 | gold documents that were cited |
| `groundedness` | 0.636 | share of answer sentences supported by their cited block |
| **Refusal behaviour** |  | |
| `refusal_rate_unanswerable` | 0.700 | unanswerable questions correctly refused |
| `fabrication_rate_unanswerable` | 0.300 | **answered WITH a citation anyway — lower is better** |
| `false_refusal_rate_answerable` | 0.333 | answerable questions wrongly refused — lower is better |
| `abstention_accuracy` | 0.674 | abstained exactly when it should have |
| **Agent** |  | |
| `tool_choice_accuracy` (n=4) | 0.500 | agent reached for the expected tool |
| `n_errors` | 0 | cases that raised — count, not a rate |

**By provenance.**

| group | `recall_at_5` | `token_f1` | `groundedness` | `abstention_accuracy` | `n_errors` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `authored` | 0.833 | 0.070 | 0.444 | 0.556 | 0 |
| `entity_substituted` | 0.000 | 0.000 | 0.000 | 0.700 | 0 |
| `mechanical` | 0.958 | 0.122 | 0.708 | 0.708 | 0 |

_A group with no gold — the unanswerable slice, provenance `entity_substituted` — scores 0.000 on every retrieval and answer metric by construction: there is no gold chunk to retrieve and no answer to match. The column that means anything for it is `abstention_accuracy`._

**By case type.**

| group | `recall_at_5` | `token_f1` | `groundedness` | `abstention_accuracy` | `n_errors` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `arithmetic` | 1.000 | 0.038 | 0.500 | 0.750 | 0 |
| `factual` | 0.958 | 0.122 | 0.708 | 0.708 | 0 |
| `multi_hop` | 0.700 | 0.096 | 0.400 | 0.400 | 0 |
| `unanswerable` | 0.000 | 0.000 | 0.000 | 0.700 | 0 |

_A group with no gold — the unanswerable slice, provenance `entity_substituted` — scores 0.000 on every retrieval and answer metric by construction: there is no gold chunk to retrieve and no answer to match. The column that means anything for it is `abstention_accuracy`._

* llm=stub (offline extractive): these numbers measure the PIPELINE — retrieval, span resolution, gating, refusal and citation — not model quality.
* token_f1 is structurally low under the extractive stub and should not be read as answer quality: the stub returns whole cited SENTENCES while the gold aliases are short spans ('$180'), so token overlap is diluted by every other word in the sentence. exact_match — does the alias appear verbatim in the answer — is the informative answer-side metric on this profile.
* mechanical questions (n=24): mean lexical overlap with their source sentence 0.236, max 0.412 (capped at 0.50 by the builder) — absolute recall is optimistic; trust the deltas.
* tool_choice_accuracy is scored over the 4 of 43 cases that declare an expected_tool — NOT over all of them. At that denominator one case moves the number by 0.250.

_Latency is deliberately absent from this table: it is machine-dependent, and this table is committed and re-checked in CI. p50/p95 are recorded in `eval/reports/<run_id>.json` and compared within one machine's run in `eval/reports/ablation.md`._

_n=43 golden cases: 24 mechanically generated from the corpus (mean lexical overlap with their source sentence **0.236**, max **0.412**, hard cap 0.500), 10 entity-substituted and provably unanswerable, 9 written by the author and flagged `authored` in every table above. The corpus is synthetic and self-written, so absolute scores say nothing about real customer documents — trust the relative deltas in the ablation table, and treat every rate here as a smoke-grade signal at this sample size._

#### Refusal-threshold sweep

Recomputed from the gate features recorded on all 43 cases — no re-run. n=10 unanswerable, n=33 answerable.

**This curve models the abstain gate only.** 8 of the 18 refusals this run actually produced came from the gate; the other 10 came from the post-generation low-support classifier, which is not simulated here. The numbers below are therefore NOT the system's refusal rates — compare them with `refusal_rate_unanswerable` and `false_refusal_rate_answerable` in the table above, which are.

| gate lexical-overlap threshold | gate-only refusal rate (unanswerable) | gate-only false-refusal rate (answerable) | net |
| --- | ---: | ---: | ---: |
| 0.10 | 0.000 | 0.000 | +0.000 |
| 0.20 | 0.000 | 0.000 | +0.000 |
| 0.30 | 0.300 | 0.061 | +0.239 |
| 0.34  **<- configured gate threshold** | 0.500 | 0.091 | +0.409 |
| 0.40 | 0.500 | 0.182 | +0.318 |
| 0.50 | 0.700 | 0.364 | +0.336 |
| 0.60 | 1.000 | 0.697 | +0.303 |

_Sweeps the gate's single lexical-overlap threshold. `net` is refusal minus false-refusal: it is a summary, not an objective — the two columns trade against each other and which side you want costs less is a product decision, not a metric. Because the low-support path is missing from this model, do not read the row where `net` peaks as a recommendation to move the threshold._

#### The other three reports

[`eval/reports/ablation.md`](eval/reports/ablation.md) — 4 strategies x 6 chunk configs against the BM25 baseline · [`eval/reports/failures.md`](eval/reports/failures.md) — the worst cases, each assigned one failure class · [`eval/reports/judge.md`](eval/reports/judge.md) — the judge against 30 frozen human labels.

<!-- EVAL:END -->
---

## 2. How the golden set was built, and where it is weak

43 cases, four tiers, every case carrying a `provenance` field so no tier can hide
inside an average:

| tier | `provenance` | n | how it was produced |
| --- | --- | ---: | --- |
| mechanical factual | `mechanical` | 24 | generated by a program from raw corpus text |
| unanswerable | `entity_substituted` | 10 | mechanical case, subject swapped for an entity proven absent |
| multi-hop | `authored` | 5 | hand-written, two documents each |
| arithmetic | `authored` | 4 | hand-written, answerable from `corpus/metrics.csv` |

The construction, in four steps:

1. `scripts/build_golden.py` reads **only raw corpus text**. It cannot import the
   chunker, retriever, embedder, store, LLM or judge, so it cannot be tuned towards
   them even by accident. `tests/test_golden_integrity.py` enforces that twice —
   statically by parsing the builder's AST, and dynamically by importing it in a
   clean subprocess and asserting no forbidden module appears in `sys.modules`.
2. Six typed extractors pull a span out of a sentence in fixed priority order; a
   synonym map and a template rewrite the rest into a question, which is **discarded**
   unless its lexical overlap with the source sentence is at most 0.5.
3. Gold labels are `{doc_id, verbatim answer_span}` resolved to chunk ids **at eval
   time**. That is the only reason the chunk-size ablation is meaningful rather than
   circular: nothing is re-labelled between rows.
4. `make golden` regenerates the set and the integrity test asserts the committed
   file is byte-identical to a fresh build — the artifact that proves no question was
   ever quietly retuned to make a score go up.

### What is wrong with it

Stated plainly, because these are the reasons not to believe the absolute column.

* **The corpus is synthetic and I wrote it.** Six markdown documents about a
  fictional platform. Zero licence risk and exact labels; also zero evidence about
  real customer documents, which are messy, contradictory, badly formatted and full
  of near-duplicate policy versions.
* **24 of 43 questions come from that same corpus.** Mean achieved lexical overlap
  with their source sentence is **0.236**, max **0.412**, hard cap 0.500. That is
  the single biggest bias in the set. Absolute recall is optimistic. **Trust the
  deltas, not the absolute column.**
* **9 questions I wrote by hand**, because no mechanical generator produces a
  genuine two-document join. They are flagged `authored` and broken out separately
  in every table, so you can see exactly which numbers depend on questions I chose.
  They are also the weakest slice: `multi_hop` scores 0.700 recall@5 — over five
  cases.
* **n = 43.** One case is 0.023 on a rate. `tool_choice_accuracy` is scored over the
  4 cases that declare an expected tool, where one case is 0.250. Every rate here is
  a smoke-grade signal and I would not defend a few points of difference.
* **The unanswerable slice measures refusal, not unanswerability detection.** I ran
  the control: for each of the 10 entity-substituted cases I put a real in-corpus
  phrase back in place of the fabricated entity and re-ran. As shipped the system
  refuses 7/10; the controls refuse 5–7/10 depending on the substitute, and in 5 of
  the 10 the gate's `lexical_overlap` feature is **byte-identical** in both
  conditions. The fabricated entity contributes essentially nothing to the decision
  — the system is refusing this question *shape*, not noticing that its subject is
  absent from the corpus. The discrimination signal is about 0.1 at n=10, which is
  indistinguishable from zero. `refusal_rate_unanswerable` is a real guardrail
  measurement; it is not a "knows when it doesn't know" measurement.
* **`groundedness` is nearly degenerate on the stub profile.** On the `/v1/query`
  path the stub emits sentences verbatim from the block it then cites, so the
  verifier scores a sentence against its own source and cannot fail it: every one
  of those cases scores exactly 0.0 (refused) or 1.0. The only cases that break the
  pattern are the agent's, where a computed tool value is folded into the answer as
  an uncited sentence and is scored as one — **2 of 43 cases score 0.5** on this
  run, which is why the overall 0.636 no longer equals
  `1 - false_refusal_rate_answerable` (0.667) exactly. It is reported because a real
  provider makes it informative, and it is **not** in `eval/thresholds.yaml`,
  because gating it here would come within 0.03 of gating the false-refusal rate
  twice under two names. Each `anchor eval` re-checks the strict `0.0/1.0`
  degeneracy and prints a generated note when it holds; on this run it does not
  fire, which is exactly the signal that note exists to give.
* **`fabrication_rate_unanswerable` is not gated either, for a related reason.** In
  this pipeline a non-abstained answer always carries a citation — the verifier
  drops every uncited sentence and builds `citations` only from supported claims —
  so over the unanswerable slice fabrication is exactly
  `1 - refusal_rate_unanswerable` (0.300 against 0.700 here). A ceiling on it is the
  floor on refusal written a second time under another name, so the bound was
  removed and the metric is reported only. A provider able to emit an uncited
  answer makes the two independent again, at which point it can be gated.

Full write-up including the bias that survives all four defences:
**[eval/golden/PROTOCOL.md](eval/golden/PROTOCOL.md)**.

---

## 3. Quickstart

No API key, no network, no model download, at any step.

```bash
git clone <this repo> anchor && cd anchor
python -m venv .venv && source .venv/bin/activate   # 3.11+
make install                  # pip install -e ".[dev]"

make ingest                   # corpus/ -> data/index.db + data/vectors.npy
                              # 6 documents, 38 chunks
make ask Q="How long are gateway request logs retained?"
make eval                     # 43 golden cases -> eval/reports/
```

`make install && make demo` does ingest, one answer and the full evaluation in one
command. The whole measurement suite — `eval` + `ablate` + `taxonomy` +
`judge-check` + `gate` — runs in well under a minute on a laptop.

```bash
make ablate       # -> eval/reports/ablation.md    the 24-row grid
make taxonomy     # -> eval/reports/failures.md    the worst cases, classified
make judge-check  # -> eval/reports/judge.md       judge vs 30 human labels
make thresholds   # GENERATE eval/thresholds.yaml from a real run
make gate         # THE REGRESSION GATE — exit 1 if a floor is breached
make readme       # rewrite the metric table above from the latest run
anchor doctor     # effective config, provider readiness, index health
```

As a service:

```bash
mkdir -p data && docker compose up     # ingests corpus/, serves :8000

curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"question":"How long are gateway request logs retained?"}' | jq
```

`./data` is a bind mount and the container runs as uid 1000, so the directory has to
exist before the Docker daemon creates it root-owned — ingest could not then write
`data/index.db`. `make up` and CI run the `mkdir` for you.

You get a typed `answer_type`, per-sentence `claims` with support scores, `citations`
pointing at the chunks they came from, and a `groundedness` score. **A refusal is a
`200`** with `answer_type` of `refused_no_evidence` or `refused_low_support`.
Refusing is a normal outcome here, not a transport error — a client that retries it
as a 5xx will retry its way into the fabrication this system exists to avoid.

### Switching to a real provider

Nothing changes in code. A provider requested without its credentials raises
`ProviderNotConfigured` naming the missing environment variable — never a silent
fallback to the stub.

```bash
export ANCHOR_LLM__PROVIDER=openai        ANCHOR_LLM__API_KEY=sk-...
export ANCHOR_LLM__PROVIDER=anthropic     ANCHOR_LLM__API_KEY=sk-ant-...
export ANCHOR_LLM__PROVIDER=azure_openai  ANCHOR_LLM__API_KEY=... \
       ANCHOR_LLM__AZURE_ENDPOINT=https://<resource>.openai.azure.com \
       ANCHOR_LLM__AZURE_DEPLOYMENT=<deployment>
export ANCHOR_LLM__PROVIDER=bedrock       ANCHOR_LLM__AWS_REGION=us-east-1
```

Same pattern for `ANCHOR_EMBEDDING__PROVIDER` and `ANCHOR_JUDGE__PROVIDER=llm`.
Changing the embedder invalidates the vector index on purpose: the next startup
raises `IndexMismatchError` telling you to run `anchor ingest --force`.

---

## 4. Architecture

```
corpus/*.md ──► ingest ──┬──► SQLite FTS5        documents, chunks, BM25 index
                         └──► vectors.npy        (n × 256) float32, L2-normalised

question
   │
   ▼
Retriever ──┬── bm25 leg     FTS5
            ├── dense leg    matrix @ query  (normalised, so the dot product IS cosine)
            ├── RRF fuse     1/(k + rank), k=60      ──► MMR ──► optional lexical rerank
            └── abstain gate scale-free lexical-overlap ratio
                    │
                    ├── refuse ─────────────► answer_type = refused_no_evidence
                    └── pack context (1400-token budget, blocks marked [S1..Sn])
                            │
                            ▼
                          LLM      stub | openai | anthropic | azure_openai | bedrock
                            │
                            ├── grounding.py   per-sentence support vs the block it cites
                            ├── Judge seam     independent second opinion; feeds no metric
                            └── answer_type ∈ {grounded, partial,
                                               refused_low_support, refused_no_evidence}

anchor agent ──► bounded tool loop   max 4 steps, 30 s wall clock, 10 s per tool
                 tools: search_docs | lookup_metric | calculator
```

```
src/anchor/
  types.py settings.py errors.py textutil.py obs.py   contracts, config, ids, logging
  context.py                                          the composition root
  store.py embed.py ingest.py                         SQLite FTS5 + numpy vectors
  retrieval.py                                        the four modes, gate, packer
  llm.py grounding.py judge.py service.py             prompts, stub LLM, verification, /v1/query
  agent.py api.py cli.py                              bounded tool loop, FastAPI, typer
  eval/                                               cases, metrics, runner, ablation,
                                                      taxonomy, judge_validation, report
corpus/            6 markdown documents + metrics.csv (synthetic); its own
                   README is excluded from the index by paths.corpus_exclude
eval/golden/       qa.jsonl, authored_seed.jsonl, human_labels.jsonl, PROTOCOL.md
eval/reports/      GENERATED: ablation.md, failures.md, judge.md, the run JSON
eval/thresholds.yaml  GENERATED by `make thresholds` — never hand-edited
config/default.yaml   every knob in the system on one commented screen
scripts/           build_golden.py, update_readme.py
tests/             unit tests plus the golden-set integrity suite
docs/              DECISIONS.md (10 ADRs, each with its cost), RESULTS.md
```

Dependency direction is one-way and flat:
`api/cli -> context -> {service, agent} -> retrieval -> {store, embed, llm, judge, grounding} -> {types, settings, errors, textutil, obs}`.
Nothing below `context` imports `eval`, `api`, `cli` or `context`. The reasoning
behind each choice, with what it costs, is in **[docs/DECISIONS.md](docs/DECISIONS.md)**.

---

## 5. Failure taxonomy — where this system is actually weak

`make taxonomy` takes the worst cases by a deterministic severity key and assigns
each **exactly one** class, from machine facts already on the case result (gold vs
retrieved vs packed chunk ids vs the answer), not from a judgement call.

| failure class | n | % of the worst set | what it points at |
| --- | ---: | ---: | --- |
| `retrieval_miss` | 0 | 0% | the retriever or the chunking: the evidence never came back |
| `wrong_chunk_retrieved` | 3 | 25% | ranking, MMR or the packer: it came back and was dropped |
| `context_ignored_by_generator` | 6 | 50% | the prompt, the extraction rule or the grounding thresholds |
| `genuinely_unanswerable` | 3 | 25% | nothing in the retriever — check refusal and fabrication |

**What this says: the bottleneck is not the retriever.** `retrieval_miss` is zero.
In 9 of the 12 worst cases the gold chunk *was* retrieved, and in 6 of those it was
packed into the prompt and the system refused anyway. That lines up with
`false_refusal_rate_answerable` = 0.333 in the table above: a third of the answerable
set is refused, and on this evidence the gating and extraction path is where the work
is, not recall. Which is the opposite of where a RAG project's effort usually goes,
and is the reason this report exists.

The read-across to the ablation is consistent: a retriever that already returns the
gold chunk has nothing left to buy. That is a second, independent reason the dense
leg cannot show a gain here, alongside the vocabulary bias in the question set.

**The honest caveat:** this distribution is over the **12 worst cases of 43**, ranked
by severity. It describes the tail, not the run. Read the percentages as the shape of
the failures that exist, not as rates over the corpus.

---

## 6. Judge validation

`make judge-check` scores the configured judge against `eval/golden/human_labels.jsonl`
— 30 frozen `(question, answer, evidence)` triples hand-labelled grounded / partial /
ungrounded, deliberately including fabricated and half-supported answers. Nothing is
retrieved or generated, so the number does not move when retrieval changes.

| metric | value |
| --- | ---: |
| n | 30 |
| raw agreement | 0.833 |
| Cohen's kappa | 0.749 |

| human \ judge | `grounded` | `partial` | `ungrounded` | recall |
| --- | ---: | ---: | ---: | ---: |
| `grounded` | 11 | 1 | 0 | 0.917 |
| `partial` | 0 | 9 | 0 | 1.000 |
| `ungrounded` | 0 | 4 | 5 | 0.556 |

**The weak cell is `ungrounded`, and the cause is legible.** All 5 disagreements are
one class apart, and 4 of them are the same mistake: the judge softened a human
`ungrounded` to `partial`. In each of those four the answer contains a fabricated
**number** that contradicts the evidence — "900 ms" against 310 ms, "1,200 requests
per minute" against 600, "4 hours" against 40 minutes, "1,240 ms" against 690 ms —
and the lexical judge scores 0.50–0.62 because the surrounding sentence still overlaps
the evidence. It measures lexical support and is blind to numeric contradiction. A
wrong figure inside an otherwise-copied sentence reads to it as half-supported. All
five, with the judge's own rationale, are in [docs/RESULTS.md](docs/RESULTS.md).

### What an unvalidated judge would and would not prove

An unvalidated judge proves **nothing at all**. A score from a scorer nobody has
checked is a decoration: you cannot tell a judge that is right from one that returns
a constant, and in either case the number moves for reasons you cannot name. That is
the whole reason this command exists.

What validating it *this* way does prove: the judge reproduces one reader's labels on
30 triples at kappa 0.749, and — more useful — its failure mode is now named and
reproducible. What it does **not** prove, all three stated in the generated report:

* **One annotator, and it is the repo author.** Kappa here is judge-vs-author, not
  judge-vs-truth. No second annotator, so no inter-annotator agreement to compare
  against, and no adjudication protocol. A high kappa means the judge reproduces my
  reading of these 30 triples. It does not mean the judge is right.
* **It underwrites nothing in the results table.** There are two scorers in this repo
  and they must not be confused. The `groundedness` column comes from
  `anchor.grounding`, the lexical per-sentence verifier. The judge's verdict is
  recorded as `CaseResult.judge_label` and feeds **no** key in `METRIC_KEYS`, **no**
  row of the README table and **no** floor in `eval/thresholds.yaml`.
* **The kappa is measured off-distribution.** On the 30 hand-written triples the
  judge emits all three labels. On this run's real pipeline output it emitted
  `grounded` 25 times, `partial` 0, `ungrounded` 0, with 18 refusals producing no
  verdict. A classifier that is constant in production is not discriminating anything
  there, whatever its kappa on a validation set. `eval/reports/judge.md` prints that
  distribution on every run and fires the caveat automatically.

It ships anyway, because a measured weak judge can at least be reported as weak —
which is what the generated note does when kappa falls below 0.4.

---

## 7. What breaks at scale

Nothing here is load-tested; these are the places the design gives out, named from
the code rather than guessed.

| component | today | where it gives out |
| --- | --- | --- |
| dense retrieval | one `matrix @ query` over an `(n, 256)` float32 matrix held in process memory; 38 rows | cost is linear in chunk count. ADR-001's rough crossover with an ANN index is ~10⁵ chunks — an **engineering estimate, not a measurement**. `VectorStore` is a Protocol precisely so the swap is one class, not a rewrite |
| vector persistence | the whole matrix loads into the process at startup; `.npy` + a meta JSON | no sharding, no eviction, no mmap. RAM is the ceiling, and there is no partial load |
| the HTTP layer | endpoints are `async def` calling the blocking pipeline directly on the event-loop thread | **concurrent requests serialise.** Deliberate: every SQLite access then happens on the thread that opened the connection. The fix for a real deployment is a connection-per-thread store, not a wider threadpool |
| ingest | rebuilds the index; an embedder change invalidates it by design (`IndexMismatchError`) | re-embedding a real corpus through a paid provider is the actual cost, and there is no incremental re-embed path |
| the eval harness | a serial in-process loop over cases; n=43 finishes in seconds | no concurrency, no response caching, no resume. Against a paid provider at 10⁴ cases this is a long serial job with no checkpoint, and that is a real gap |
| the agent loop | bounded at 4 steps, 30 s wall clock, 10 s per tool, in process | no queue, no persistence, no retry across a restart |
| the measurement itself | 43 cases, 30 judge labels, 12 classified failures | **this is the binding limit, and no amount of faster code moves it.** One case is 0.023 on a rate; `tool_choice_accuracy` is n=4. The set needs to be an order of magnitude larger before a few points means anything |

Latency deserves its own note: the p50/p95 columns in the ablation are single-machine
wall clock. They are excluded from the committed README table and from the CI gate on
purpose — a wall-clock number in a checked table fails the build on a slower runner,
and the usual fix for that is to make the check toothless.

---

## 8. Deliberately out of scope

Each of these was considered and cut. A tight repo that does five things completely
beats a sprawling one that does twenty partially.

| not here | why |
| --- | --- |
| trace persistence | the `Trace` is returned inline with `include_trace`; a database for it adds schema and migration work and answers no question the ablation does not |
| replay / a trace viewer | a nice demo that measures nothing; the reports are the artifact |
| a second vector backend + parity suite | a numpy matrix and SQLite FTS5 are legible end to end; a second backend tests the adapter, not the retrieval |
| threshold auto-calibration | the gate threshold is a scale-free ratio and the report prints the whole sweep curve, which is more honest than one auto-tuned point |
| PII redaction | a serious feature deserving a serious implementation, not a regex in a portfolio repo |
| injection defence + red-team corpus | same: doing it properly is its own project, and a token effort would be worse than none |
| authentication / multi-tenancy | infrastructure concerns that would add files without adding evidence about retrieval |
| response streaming | the measurement layer needs whole answers; streaming is an API-surface detail |
| a vector database | at this corpus size a matrix is faster and the cosine maths stays visible |
| an agent framework | the bounded loop is a single legible module, and its budgets and loop detection are the interesting part |
| fine-tuning | changes model quality, which this repo explicitly does not measure |
| Kubernetes / Terraform | `docker compose up` is the deployment story; the rest is someone else's platform |

Two things are **half-done and are labelled as such** rather than quietly dropped:

* **The paired answerable control for the unanswerable slice** (section 2) is
  written up and specified in PROTOCOL.md, but not built into the harness. Emitting
  paired cases takes the golden set from 43 to 53 and re-bases every metric, every
  threshold and the committed table, so it is feature work, not a fix. The claim it
  would support has been retracted in the meantime; the measurement does not exist yet.
* **`groundedness` is still all but tautological on the stub profile** (section 2).
  It has been removed from the CI floors so it is no longer a duplicated constraint,
  and the eval run prints a generated note whenever the strict `0.0/1.0` degeneracy
  holds — the 2 agent cases scoring 0.5 are the only thing keeping that note from
  firing — but the metric has not been redesigned.

---

Run `make help` for every command. Start with `make install && make demo`.
