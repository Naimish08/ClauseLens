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

The grid in the next section shows that this baseline's 512-character chunks are
not the best size: 1024 characters nearly triples recall@10. That gap is the
improvement loop's first candidate change, not a retroactive edit to the baseline.

---

## 4. Chunking comparison (train)

24 configurations: 4 chunkers × sizes {256, 512, 1024} characters × overlap
{0%, 15%}, BM25 fixed, `k=10`, 140 train queries, 22 seconds of compute.
Figure: [`docs/figures/chunk_grid.png`](docs/figures/chunk_grid.png); full table
via `make grid` → `artifacts/grids/chunk_grid.csv`.

| chunker | size | overlap | chunks | mean len | containment | recall@10 | precision@10 | span_hit@10 |
|---|---|---|---|---|---|---|---|---|
| fixed | 256 | 0% | 4810 | 252 | 0.130 | 0.052 | 0.008 | 0.129 |
| fixed | 512 | 0% | 2396 | 506 | 0.401 | 0.101 | 0.011 | 0.221 |
| fixed | 1024 | 0% | 1203 | 1007 | 0.656 | 0.319 | 0.017 | 0.407 |
| recursive | 1024 | 15% | 1462 | 829 | 0.927 | 0.306 | 0.015 | 0.364 |
| sentence | 1024 | 15% | 1442 | 840 | 0.932 | **0.343** | 0.016 | 0.364 |
| clause | 1024 | 15% | 1759 | 689 | 0.927 | 0.256 | 0.015 | 0.300 |

**Span containment** is the fraction of ground-truth answers that lie fully inside
a single chunk. It is computed without any retrieval, and it is the ceiling that
recall cannot exceed.

1. **Size dominates every chunker comparison.** At 256 characters containment is
   0.13–0.35 and recall@10 cannot pass ~0.05; at 1024 characters containment is
   0.66–0.93 and recall reaches 0.18–0.34. Comparing chunkers at different sizes
   would mostly be measuring size.
2. **Sentence-1024 with 15% overlap is the best configuration** (recall@10 0.343),
   narrowly ahead of fixed-1024 (0.319) and recursive (0.306).
3. **Clause-aware chunking does not dominate.** It reaches 0.256 at 1024, below
   fixed and sentence at the same size. Contract headings are a poor proxy for
   where answers live, and dot-leader tables of contents create spurious
   boundaries. The hypothesis was reasonable; the data does not support it.
4. **Overlap is not free.** It raises recall at 1024 for recursive, sentence and
   clause, but *lowers* fixed-512 from 0.101 to 0.074: extra windows rescue
   straddling answers at large sizes and add near-duplicate competitors at small
   ones. Paper-mode scoring also double-counts shared characters (ADR-005).
5. **Precision sits at 0.007–0.017 for every chunker.** Chunk size sets
   precision; the chunker barely moves it. With a 317-character median answer,
   returning a 1,000-character chunk caps precision near 0.3 and returning ten of
   them caps it near 0.03 — the observed numbers.

The best recall (0.343) is still far below its containment ceiling (0.932), which
confirms that after size, **ranking is the bottleneck**, not chunking.

---

## 5. Abstention (train)

140 answerable train queries against 34 train negatives (the frozen set's train
split), BM25 over fixed-512 chunks. Totals: 46 negatives, split 34 train / 12
test by the same hash as the benchmark.
Figure: [`docs/figures/abstention_curves.png`](docs/figures/abstention_curves.png).

Operating points are **nested**: the threshold is chosen on four folds and applied
to the held-out fifth, so the reported numbers do not benefit from tuning on the
same queries. The selection rule was fixed in advance — lowest false abstention
among points with correct abstention ≥ 0.80, otherwise highest F1.

| policy | false abst. | 95% CI | correct abst. | 95% CI | abst. F1 |
|---|---|---|---|---|---|
| threshold | 0.079 | [0.036, 0.121] | 0.794 | [0.647, 0.912] | 0.853 |
| margin | 0.479 | [0.393, 0.564] | 0.794 | [0.647, 0.912] | 0.630 |
| category | 0.093 | [0.050, 0.143] | **0.853** | [0.735, 0.971] | **0.879** |
| classifier | **0.050** | [0.014, 0.086] | 0.794 | [0.647, 0.912] | 0.865 |

1. **The strawman survives.** A single global threshold on the BM25 top score
   reaches F1 0.853. The build plan expected fixed thresholds to do badly; on this
   data they do not, because unanswerable queries have systematically weaker
   lexical matches.
2. **Margin is the clear loser** (false abstention 0.479): in long contracts the
   top two chunks are usually close even when the answer is present.
3. **Category normalisation is nominally best** (F1 0.879) but its interval
   overlaps the global threshold's. "At least as good", not "better".
4. **The classifier trades recall for precision** (lowest false abstention 0.050,
   slightly lower correct abstention). No significant advantage.
5. **Type 3 hard negatives are where policies differ** — 0.50–0.70 correct versus
   1.00 for types 1, 2 and 4. With 10 type-3 train negatives that gap is not
   estimable.

With 34 negatives the intervals are wide enough that no ranking is defensible.
The report claims a working operating point, not a winning method.

### A confound worth reporting

Including `query_len` made the classifier look excellent — false abstention 0.000,
correct abstention 0.941, precision 1.000. Its largest coefficient was `query_len`
(−3.65), because the authored negatives are 51–151 characters against benchmark
queries of 134–243. The model had learned authorship, not answerability. The
feature was removed and a regression test now prevents reintroducing it. This is
the main risk of an authored negative set, and it was caught only by inspecting
coefficients.

---

## 6. Iteration history

TODO. Generated from `logs/experiments.jsonl`, not transcribed.

---

## 7. Error analysis

TODO (at least 10 failures, grouped by cause, spanning ≥4 causes). The baseline
already provides candidates:

- **Wrong section** (`doc_hit` true, `span_hit` false): the contract's title/header
  chunks outrank the clause because the query prefix repeats the party names.
- **Partial span**: chunk boundaries cut through answers.
- **Extra context**: whole-chunk returns dilute character precision.

Each entry needs the query, the retrieved span, the expected span, and a
one-sentence diagnosis.

---

## 8. Limitations

See [`docs/12-risks-and-limitations.md`](docs/12-risks-and-limitations.md) for the
full list. The ones that most affect how these numbers should be read:

- One benchmark, one genre (US commercial contracts, English).
- The mini subset clusters queries onto 29 documents; it is not a random sample.
- The query prefix leaks the document, so prefix routing is not deployable as-is.
- No LLM components; whether an LLM would beat the best non-LLM pipeline here is
  untested.
- Small test set (54 answerable, ~13 unanswerable), so intervals are wide.
- Character recall can be inflated by returning more text. The improvement loop
  therefore carries a text budget (at most 2.5× the baseline's mean returned
  characters); without it, the best "improvement" was a chunk size that returned
  twice as much of each contract (ADR-017).
- The authored negatives are shorter than benchmark queries. Query length was
  removed from the abstention classifier's features after it separated the classes
  by construction (ADR-006 section, "A confound worth reporting").
- Latency is gated by an absolute budget as well as a ratio (ADR-018). The budget
  is a judgement about what feels interactive, not a measurement.

---

## 9. AI tool usage

TODO. State which components were AI-assisted and how they were verified. Design
decisions and rejected alternatives are recorded in `docs/09-decisions.md`.
