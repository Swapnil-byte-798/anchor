"""The measurement layer — the reason this repo is not a generic RAG service.

Five commands live in here, and four of them are the features the project is
actually about:

    anchor eval         run the 43-case golden set; write eval/reports/
                        (:mod:`anchor.eval.runner`, :mod:`anchor.eval.report`)
    anchor ablate       4 retrieval strategies x 6 chunk configs, retrieval only,
                        BM25-only as the declared baseline
                        (:mod:`anchor.eval.ablation`)
    anchor taxonomy     the worst cases, each assigned exactly one of four
                        failure classes from machine facts
                        (:mod:`anchor.eval.taxonomy`)
    anchor judge-check  the configured judge against 30 frozen human labels:
                        raw agreement, Cohen's kappa, 3x3 confusion
                        (:mod:`anchor.eval.judge_validation`)
    anchor gate         the CI regression gate; exits 1 when a floor in
                        eval/thresholds.yaml is breached
                        (:mod:`anchor.eval.report`)

Everything here imports :mod:`anchor.context` and below. Nothing below
``context`` imports anything from this package.
"""

from __future__ import annotations
