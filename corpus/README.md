# About this corpus

Every document in this directory is **synthetic**. It was written by the author
of this repository, for this repository, and describes a company called Meridian
that does not exist. Nothing here is copied, adapted or summarised from any
third-party source, so there is no licence question and no attribution to make.

## What is in here

Six markdown documents form a small, internally cross-referencing platform
handbook, plus one CSV that is the structured half the agent has to combine with
retrieval:

| file | what it is |
| --- | --- |
| `gateway-overview.md` | endpoints, latency budgets, regions, ownership |
| `rate-limits-and-quotas.md` | tiers, bursts, overage pricing, the batch quota |
| `incident-severity.md` | the SEV-1..SEV-4 bands and their response targets |
| `oncall-runbook.md` | rotation, first five minutes, escalation paths |
| `data-retention-policy.md` | how long each class of data is kept, and why |
| `postmortem-2025-06-index-drift.md` | a worked incident write-up |
| `metrics.csv` | 24 monthly rows, 2024-01 to 2025-12 |

The documents deliberately depend on each other. The 90-day request-log window
in the retention policy is the stated reason the drift in the postmortem could
not be reconstructed; the severity policy defines the bands the runbook
escalates on; the limits page defines the batch quota the postmortem's reindex
job exceeded. Those dependencies are what make a multi-hop question genuinely
multi-hop rather than a single lookup wearing a disguise.

## What being synthetic does to the numbers

Read this before quoting anything the evaluation prints.

1. **Absolute scores say nothing about real customer documents.** This corpus is
   clean, consistently formatted, free of contradictions, free of scanned
   tables, free of near-duplicate versions of the same policy, and roughly 3,000
   words in total. Real corpora are none of those things. A recall figure
   measured here does not transfer.
2. **Only the relative deltas are meaningful.** Because every configuration in
   `make ablate` is measured against the same corpus and the same 43 questions,
   the comparison between BM25, dense, hybrid and hybrid+rerank — and between
   the six chunk configurations — is a fair one. That comparison is the output
   worth reading. The absolute column next to it is not.
3. **The author wrote both the corpus and (some of) the questions.** That is the
   circularity problem, and `eval/golden/PROTOCOL.md` sets out the four separate
   defences the repository puts up against it, along with the bias that remains
   after all four.

## Editing this corpus

Changing any document changes the golden set, because the questions are derived
from the corpus text by `scripts/build_golden.py`. After an edit, run
`make golden` and commit the regenerated `eval/golden/qa.jsonl` alongside the
document. CI fails if the committed set is not reproducible from the committed
corpus.
