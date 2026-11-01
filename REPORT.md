# ClauseLens — Report

Status: **in progress**. Sections marked TODO are not yet written; all numbers
present are measured and reproducible from a committed config.

Deadline: Monday 2 November 2026, 17:00 IST.

---

## 0. Summary

TODO (write last). Must contain: the final system, headline test numbers with
confidence intervals, the number of test runs, and the one-sentence answer to
"does retrieval work here, and what actually drives the score?".

---

## 1. System design and alternatives

ClauseLens is a config-driven retrieval harness over LegalBench-RAG (CUAD subset,
194 queries, 29 documents). Every stage is swappable by YAML, and evaluation lives
outside the pipeline so ground truth cannot reach inference.

Pipeline: `chunker → retriever → optional reranker → abstention → ranked spans`.

Alternatives considered and rejected are recorded as ADRs in
[`docs/09-decisions.md`](docs/09-decisions.md). The decisions that shape the
results most:

- **ADR-005** — the paper's precision/recall definition *sums* per-snippet overlaps
  while the build plan said *union*; both are implemented, `paper` is the default.
- **ADR-008** — no LLM components (scope cut to protect the deadline).
- **ADR-010** — the query prefix leaks the document; routing is an ablation, not a
  hidden default.

---

## 2. Evaluation setup

| Item | Value |
|---|---|
| Benchmark | LegalBench-RAG, CUAD subset |
| Queries | 194 (140 train / 54 test), hash-fixed split |
| Documents | 29 |
| Ground-truth snippets | 272 |
| Metric | character precision/recall (paper semantics), doc-hit, span-hit, chunk-hit, MRR |
| Hardware | RTX 4060 Laptop (8 GB), Python 3.12.15 |

Offset integrity was verified before any modelling: every stored answer text equals
its character span, in both text and raw read modes, with zero mismatches
(`docs/02-data.md`).

---

## 3. Baseline (train)

`configs/baseline.yaml` — fixed 512-character chunks, no overlap, BM25
(`k1=1.5`, `b=0.75`), top-10, no reranking, no abstention.

| k | recall | precision | f1 | doc_hit | span_hit | chunk_hit | mrr |
|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.007 | 0.000 | 0.004 |
| 4 | 0.019 | 0.006 | 0.009 | 1.000 | 0.071 | 0.014 | 0.022 |
| 8 | 0.085 | 0.011 | 0.020 | 1.000 | 0.186 | 0.057 | 0.041 |
| 10 | 0.101 | 0.011 | 0.020 | 1.000 | 0.221 | 0.064 | 0.044 |
| 16 | 0.198 | 0.013 | 0.024 | 1.000 | 0.321 | 0.121 | 0.052 |

Index: 29 documents, 2396 chunks, 1,211,773 characters. Latency: 3.5 ms median,
4.6 ms p95 per query. Reproduce with `make eval CONFIG=configs/baseline.yaml`.

Three things this table already establishes:

1. **Document identification is solved** (`doc_hit@1 = 1.000`). Every query's
   `"Consider the <contract>"` prefix maps to exactly one document, so ranking
   failures are not "wrong contract" failures.
2. **Within-document ranking is the bottleneck.** The first chunk that overlaps
   the ground truth sits at rank 57–743 for the sampled queries inspected.
3. **Character precision is structurally low** (0.011 at k=10) because each
   returned chunk is 512 characters against a 394-character mean answer; this is
   the metric behaving as designed, and it is why chunk size is the centrepiece
   of the analysis.

Query-prefix ablation (`docs/05-retrieval.md`), train:

| Query text | recall@10 | recall@50 | recall@100 |
|---|---|---|---|
| full query | 0.101 | 0.375 | 0.465 |
| prefix stripped | 0.083 | 0.192 | 0.259 |

Stripping the prefix makes retrieval worse, confirming it carries routing
information rather than noise.

---

## 4. Iteration history

TODO. Generated from `logs/experiments.jsonl`, not transcribed.

---

## 5. Error analysis

TODO (at least 10 failures, grouped by cause, spanning ≥4 causes). The baseline
already provides candidates:

- **Wrong section** (`doc_hit` true, `span_hit` false): the contract's title/header
  chunks outrank the clause because the query prefix repeats the party names.
- **Partial span**: chunk boundaries cut through answers.
- **Extra context**: whole-chunk returns dilute character precision.

Each entry needs the query, the retrieved span, the expected span, and a
one-sentence diagnosis.

---

## 6. Limitations

See [`docs/12-risks-and-limitations.md`](docs/12-risks-and-limitations.md) for the
full list. The ones that most affect how these numbers should be read:

- One benchmark, one genre (US commercial contracts, English).
- The mini subset clusters queries onto 29 documents; it is not a random sample.
- The query prefix leaks the document, so prefix routing is not deployable as-is.
- No LLM components; whether an LLM would beat the best non-LLM pipeline here is
  untested.
- Small test set (54 answerable, ~13 unanswerable), so intervals are wide.

---

## 7. AI tool usage

TODO. State which components were AI-assisted and how they were verified. Design
decisions and rejected alternatives are recorded in `docs/09-decisions.md`.
