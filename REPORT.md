# ClauseLens — Report

Every number is measured and reproducible from a committed config; the iteration
and comparison tables are generated from logs, not written by hand.

---

## 0. Summary

ClauseLens answers contract questions with ranked passages and exact character
offsets, or abstains explicitly when nothing is relevant enough. The final system is
fixed 1,024-character chunks, dense retrieval with gte-base, and a score-threshold
abstention policy.

**Headline, test split** (54 queries, never used for tuning):

| metric | baseline | final | delta | 95% CI |
|---|---|---|---|---|
| recall@10 | 0.127 | **0.401** | **+0.274** | [+0.146, +0.407] |
| precision@10 | 0.010 | **0.023** | +0.013 | [+0.004, +0.022] |
| mrr@10 | 0.046 | 0.179 | +0.132 | [+0.072, +0.199] |
| span_hit@10 | 0.204 | 0.481 | +0.278 | [+0.130, +0.426] |

On train, the shipped configuration reaches recall@10 **0.458** against 0.101
(+0.357 [+0.280, +0.431]). The train–test gap of 0.057 sits inside the interval, so
the loop did not visibly overfit the train split.

What actually drives the score, in order: **chunk size** (span containment sets the
recall ceiling), **dense versus lexical retrieval**, then everything else. Document
identification was already perfect at baseline (`doc_hit@1 = 1.000`); all the work
is ranking the right clause inside the right contract.

**Test-run accounting.** Four test evaluations, all after tuning was complete:
baseline (for the comparison), final before abstention was configured, final with
abstention, and a final abstention-only evaluation covering both directions. The
first final run was superseded by a functional requirement — shipping abstention —
not by a result. `logs/test_runs.log` records all four with reasons.

Main limitations: one benchmark of 29 contracts, authored negatives, and a loop that
is a local search over a hand-written rule table (Sections 9–10).

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
  while an early design note specified *union*; both are implemented, `paper` is the default.
- **ADR-008** — no LLM components (scope decision).
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
| Unanswerable queries | 46 authored (34 train / 12 test) |
| Test evaluations | 4, all after tuning (`logs/test_runs.log`) |
| Metric | character precision/recall (paper semantics), doc-hit, span-hit, chunk-hit, MRR |
| Hardware | RTX 4060 Laptop (8 GB), Python 3.12.15 |

Offset integrity was verified before any modelling: every stored answer text equals
its character span, in both text and raw read modes, with zero mismatches
(`docs/02-data.md`).

**Metric definitions.** Character precision is the total overlapping characters
divided by the total characters returned; character recall is the overlap divided
by the total ground-truth characters. Both are computed per query and
macro-averaged, so one long answer cannot outvote a short one. The implementation
follows the reference code exactly (`paper` mode), which sums per-snippet overlaps
rather than taking their union; the union variant exists because an early design
note and the paper disagree when retrieved chunks overlap (ADR-005), and every
table states which was used.

**Split discipline.** The train/test boundary is a SHA-256 of the query text and is
never re-derived. All development — chunking, retrieval, abstention, the loop — used
train only. Test evaluation requires an explicit flag and appends to
`logs/test_runs.log`; four runs were made in total, all after tuning, and Section 0
explains each.

**Uncertainty.** Every comparison is paired by query and reported with a 95%
percentile interval from 10,000 bootstrap resamples. A difference is called
significant only when the interval excludes zero. With 140 train queries, one query
flipping moves recall by about 0.007, which is the practical noise floor.

**Reproduction.** Each table is produced by a committed config and a `make` target:
`make eval`, `make grid`, `make retrieval-grid`, `make abstention`, `make loop`,
`make compare`. Run artefacts record the config hash, per-query metrics and
aggregates.

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

Evaluated on the **final config** (dense gte-base, fixed-1024): 140 answerable train
queries against 34 train negatives. Operating points are nested — thresholds chosen
on four folds, applied to the held-out fifth. Figure:
[`docs/figures/abstention_curves.png`](docs/figures/abstention_curves.png).

| policy | false abst. | 95% CI | correct abst. | 95% CI | abst. F1 |
|---|---|---|---|---|---|
| threshold | 0.286 | [0.207, 0.364] | 0.824 | [0.676, 0.941] | 0.765 |
| margin | 0.700 | [0.621, 0.771] | 0.824 | [0.706, 0.941] | 0.440 |
| category | 0.264 | [0.193, 0.336] | 0.824 | [0.676, 0.941] | **0.777** |
| classifier | 0.557 | [0.471, 0.636] | 0.765 | [0.618, 0.912] | 0.561 |

1. **Dense retrieval breaks score-based abstention.** The same protocol on the BM25
   baseline gave false abstention 0.079 at correct abstention 0.794. On dense
   retrieval it is 0.286 at 0.824. Dense always finds *some* similar chunk, so the
   top score no longer separates answerable from unanswerable queries. This is a
   real cost of the retrieval upgrade and it is the most transferable finding in
   this section.
2. **Margin is useless here** (false abstention 0.700): in a long contract the top
   two chunks are usually close whether or not the answer exists.
3. **The classifier is worse than a threshold** (0.557), despite seven features.
   With 34 negatives it overfits; the authored set is too small to support it.
4. **Category normalisation is nominally best and statistically indistinguishable**
   from the global threshold. "At least as good", not "better".
5. **The shipped operating point is deliberately conservative.** The config uses a
   threshold of 0.9018 on the cosine score, which on train abstains on 8.6% of
   answerable queries while correctly abstaining on 76.5% of negatives. On test the
   false-abstention rate was 14.8% — higher, so the threshold does not transfer
   perfectly across splits. That is stated rather than averaged away.
6. **Type 3 hard negatives remain the hard case** (0.70 correct); types 1, 2 and 4
   are near-saturated.

**Test-split abstention (final reporting).** This is the dedicated test evaluation
of abstention in both directions; the threshold is the fixed `0.9018` from
`configs/final.yaml`, with no selection on test, over the same 21-document test
index used by the retrieval run (`eval.run_abstention --split test`).

| direction | rate | 95% CI | n |
|---|---|---|---|
| false abstention (answerable) | 0.148 | [0.056, 0.241] | 54 |
| correct abstention (unanswerable) | 0.917 | [0.750, 1.000] | 12 |

The false-abstention rate reproduces the 14.8% (8/54) figure used in Section 5 and
the test results. Correct abstention is high overall, but the test negatives are
unevenly distributed (type 1: 1, type 2: 5, type 3: 4, type 4: 2). Per type, types
1, 2 and 4 are saturated and type 3 is 0.75; the interval on 12 points is wide, so
this is reported as an estimate, not a precise claim. It is also a different
composition from the 34 train negatives, which is why the train and test rates are
not directly comparable.

**Caveat:** A first version of the classifier looked near-perfect (false abstention
0.000) because `query_len` separated the classes by construction — the authored
negatives are shorter than benchmark queries. It was learning authorship. The
feature was removed and a regression test enforces the exclusion.

---

## 6. Retrieval comparison (train)

Chunking is held fixed at the best configuration from Section 4
(sentence-window, 1,024 characters, 15% overlap), 140 train queries. Figure:
[`docs/figures/retrieval_grid.png`](docs/figures/retrieval_grid.png); regenerate
with `make retrieval-grid`.

| retriever | recall@10 | precision@10 | mrr@10 | span_hit@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| BM25 | 0.343 | 0.016 | 0.099 | 0.364 | 3.8 | 6.4 |
| dense bge-base | 0.529 | 0.026 | 0.181 | 0.536 | 13.5 | 32.1 |
| **dense gte-base** | **0.580** | **0.028** | **0.221** | **0.586** | **10.8** | 17.7 |
| dense e5-large | 0.576 | 0.028 | 0.226 | 0.579 | 32.8 | 196.1 |
| hybrid RRF (bm25+bge) | 0.475 | 0.024 | 0.147 | 0.493 | 28.9 | 42.6 |
| hybrid weighted 2:1 (bm25) | 0.435 | 0.021 | 0.137 | 0.486 | 30.7 | 55.9 |
| hybrid + cross-encoder | 0.550 | 0.027 | 0.190 | 0.593 | 1,073 | 1,568 |

1. **Dense beats BM25 decisively** — by 0.19 to 0.24 recall@10. At this chunking
   size, lexical matching is not competitive, which is the opposite of the usual
   "BM25 is a strong legal baseline" expectation.
2. **Embedder size does not matter here.** gte-base (109M) scores 0.580 and
   e5-large (335M) scores 0.576 — a 3× parameter increase for no measurable gain,
   at 3× the query latency and roughly 5× the index time. That is a clean negative
   result for "bigger embeddings are better".
3. **Hybrid fusion is *worse* than its best member.** RRF of BM25 and bge scores
   0.475 against 0.529 for bge alone. Fusing a much weaker ranking dilutes a strong
   one. The loop reached hybrid first only because its rule table never tried dense
   on its own — a limitation of the rule ordering, discussed in Section 7.
4. **Weighting BM25 higher makes it worse still** (0.435), consistent with the
   dilution explanation.
5. **The reranker recovers some of the loss but not the lead.** 0.550 at 1,073 ms
   median, against 0.580 at 10.8 ms for gte-base alone. It also raises span_hit
   (0.593), so it improves *localisation* while ranking worse overall.
6. **Index cost is real.** BM25 indexes in 0.16 s; gte-base takes 45 s; e5-large
   takes 230 s. Retrieval speed and index speed are different budgets.

**Caveat:** The hybrid rows report near-zero index time because their embeddings
were cache hits from the earlier dense runs. That number is a cache measurement,
not the cost of building the index from scratch.

---

## 7. Iteration history

Generated from `logs/experiments.jsonl` by `make log-report` — never transcribed.
Fourteen iterations beyond the baseline, **2 accepted and 12 rejected**.

| # | change | failure addressed | recall@10 | Δ | accepted | why |
|---|---|---|---|---|---|---|
| 0 | baseline | n/a | 0.101 | — | yes | baseline |
| 1 | size: 512 → 1024 | wrong_section | 0.319 | +0.219 | yes | accepted |
| 2 | size: 1024 → 2048 | wrong_section | 0.493 | +0.174 | no | rejected: guardrail: mean retrieved 9901->19718 chars (cap 12518) |
| 3 | chunker: fixed → recursive | wrong_section | 0.255 | -0.064 | no | rejected: delta -0.064 < 0.01; won 2/5 folds |
| 4 | chunker: fixed → sentence | wrong_section | 0.301 | -0.018 | no | rejected: delta -0.018 < 0.01; CI includes 0 (-0.071, +0.034); won 2/5 folds |
| 5 | chunker: fixed → clause; max_size: 1024 | wrong_section | 0.180 | -0.140 | no | rejected: delta -0.139 < 0.01; won 0/5 folds |
| 6 | overlap: 0 → 153 | wrong_section | 0.316 | -0.004 | no | rejected: delta -0.004 < 0.01; CI includes 0 (-0.057, +0.051); won 2/5 folds |
| 7 | retriever: bm25 → dense | wrong_section | 0.496 | +0.177 | yes | accepted |
| 8 | size: 1024 → 2048 | wrong_section | 0.621 | +0.124 | no | rejected: guardrail: precision 0.025->0.017, mean retrieved 10046->19964 chars (cap 12518) |
| 9 | chunker: fixed → recursive | wrong_section | 0.459 | -0.037 | no | rejected: delta -0.037 < 0.01; CI includes 0 (-0.099, +0.023); won 2/5 folds |
| 10 | chunker: fixed → sentence | wrong_section | 0.522 | +0.026 | no | rejected: CI includes 0 (-0.029, +0.082) |
| 11 | chunker: fixed → clause; max_size: 1024 | wrong_section | 0.434 | -0.062 | no | rejected: delta -0.062 < 0.01; CI includes 0 (-0.136, +0.012); won 2/5 folds |
| 12 | overlap: 0 → 153 | wrong_section | 0.509 | +0.012 | no | rejected: CI includes 0 (-0.055, +0.078); won 3/5 folds |
| 13 | retriever: dense → hybrid | wrong_section | 0.424 | -0.072 | no | rejected: delta -0.072 < 0.01; won 1/5 folds |
| 14 | retriever: dense → rerank | wrong_section | 0.486 | -0.011 | no | rejected: delta -0.011 < 0.01; CI includes 0 (-0.071, +0.051); won 2/5 folds; guardrail: latency p95 12.5->1099.0 ms |

What the rejections establish:

- **Iteration 2 is the text budget working.** Doubling the chunk size again raised
  recall to 0.493, but only by returning 19,718 characters per query instead of
  9,901. Refused (ADR-017).
- **Iterations 3–6 and 9–12 are the chunker family losing.** At equal size, fixed
  chunks beat recursive, sentence and clause chunking, and overlap does not help.
  The independent chunking grid (Section 4) found the same ordering.
- **Iteration 8 is the closest call in the run.** At dense retrieval, 2,048-character
  chunks reached recall 0.620 — better than the accepted 0.496 — but with a 32%
  relative precision drop and 19,964 characters returned. Two guardrails refused it.
- **Iteration 13 shows hybrid is not additive.** After dense was accepted, adding
  BM25 by fusion *lost* 0.072 recall, matching the retrieval grid.
- **Iteration 14 is a genuine improvement the project declines to ship**: the
  cross-encoder cost 1,099 ms p95 for no significant gain.
- **Interaction limit, stated plainly.** Sentence chunking (+0.026) and overlap
  (+0.012) were each individually below the significance bar after dense, yet the
  grid shows the two combined reach 0.580. A one-change-at-a-time loop cannot take
  that step; Section 10 records it as a limitation rather than hiding it.

---

## 8. Baseline vs final

**Train** (140 queries), paired bootstrap, 10,000 resamples. The final column is the
shipped configuration, abstention included.

| metric | baseline | final | delta | 95% CI | significant |
|---|---|---|---|---|---|
| recall@10 | 0.101 | **0.458** | **+0.357** | [+0.280, +0.431] | yes |
| precision@10 | 0.011 | **0.022** | +0.011 | [+0.007, +0.016] | yes |
| f1@10 | 0.019 | 0.041 | +0.023 | [+0.014, +0.031] | yes |
| mrr@10 | 0.044 | 0.196 | +0.152 | [+0.115, +0.190] | yes |
| span_hit@10 | 0.221 | 0.557 | +0.336 | [+0.250, +0.421] | yes |
| chunk_hit@10 | 0.064 | 0.371 | +0.307 | [+0.229, +0.386] | yes |
| doc_hit@10 | 1.000 | 0.914 | −0.086 | [−0.136, −0.043] | yes |

**Test** (54 queries, four runs total, details in Section 0):

| metric | baseline | final | delta | 95% CI | significant |
|---|---|---|---|---|---|
| recall@10 | 0.127 | **0.401** | **+0.274** | [+0.146, +0.407] | yes |
| precision@10 | 0.010 | **0.023** | +0.013 | [+0.004, +0.022] | yes |
| mrr@10 | 0.046 | 0.179 | +0.132 | [+0.072, +0.199] | yes |
| span_hit@10 | 0.204 | 0.481 | +0.278 | [+0.130, +0.426] | yes |
| chunk_hit@10 | 0.093 | 0.333 | +0.241 | [+0.111, +0.370] | yes |

Two numbers need reconciling. The loop's last accepted state scores **0.496** on
train (Section 7); the shipped configuration scores **0.458** here. The difference is
abstention: enabling it for delivery abstains on 8.6% of answerable queries, which
costs about 0.038 recall and is the price of the functional requirement to return
nothing when nothing is relevant. That trade-off was chosen after the loop, using
train data only.

The other **decrease** is `doc_hit`, and it follows from the same decision: an
abstained query returns no document at all, so document hit falls from saturated to
0.914. Retrieval quality on the queries the system does answer is substantially
higher than the baseline's.

The train–test gap on the headline metric is 0.057, smaller than the width of the
confidence intervals, so the improvement is not a train-only artefact.

---

## 9. Error analysis

Regenerated on the shipped config with `make errors CONFIG=configs/final.yaml`.
140 train queries; every query is accounted for.

| cause | queries | share |
|---|---|---|
| extra_context | 53 | 38% |
| wrong_section | 50 | 36% |
| partial_span | 25 | 18% |
| false_abstention | 12 | 9% |

Twelve concrete failures, three per cause (full report with excerpts in
`artifacts/error_analysis.md`).

#### wrong_section

- *Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; What…*
  expected `HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGR` [16867:17117] “This Agreement and its application and interpretation will be governed exclusively by…”
  returned `HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGR` [19366:20008] “of any of the provisions of this Agreement shall be deemed or shall constitute a waiv…” — recall 0.000
- *Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; What is the expiration date of…*
  expected `THERAVANCEBIOPHARMA,INC_05_08_2020-EX-10.2-SER` [4382:4705] “The Executive's employment with the Company will commence on the Commencement Date an…”
  returned `THERAVANCEBIOPHARMA,INC_05_08_2020-EX-10.2-SER` [0:1021] “Exhibit 10.2 STRICTLY PRIVATE AND CONFIDENTIAL 1 April, 2020 THERAVANCE BIOPHARMA UK …” — recall 0.000
- *Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; Is there a clause preventing t…*
  expected `THERAVANCEBIOPHARMA,INC_05_08_2020-EX-10.2-SER` [25311:25713] “During the term of employment and for one (1) year thereafter, the Executive will not…”
  returned `THERAVANCEBIOPHARMA,INC_05_08_2020-EX-10.2-SER` [0:1021] “Exhibit 10.2 STRICTLY PRIVATE AND CONFIDENTIAL 1 April, 2020 THERAVANCE BIOPHARMA UK …” — recall 0.000

#### partial_span

- *Consider the Joint Venture Agreement between Tate & Lyle Fermentation Products Ltd. and Igene Biotechnology, Inc. for As…*
  expected `IGENEBIOTECHNOLOGYINC_05_13_2003-EX-1-JOINT VE` [23695:24217] “Except as set forth in Articles 7.1, 7.2, 8.1 and 8.2, it is explicitly agreed that n…”
  returned `IGENEBIOTECHNOLOGYINC_05_13_2003-EX-1-JOINT VE` [1022:2039] “20 21. MISCELLANEOUS PROVISIONS 20 22. DISPUTE RESOLUTION PROCEDURES; LIQUIDATION 23 …” — recall 0.025
- *Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; Is there a covenant not to…*
  expected `PapaJohnsInternationalInc_20190617_8-K_EX-10.1` [15881:16680] “PAPA JOHN'S shall not, during the Term or at any time thereafter, attack or challenge…”
  returned `PapaJohnsInternationalInc_20190617_8-K_EX-10.1` [0:1017] “Exhibit 10.1 ENDORSEMENT AGREEMENT THIS ENDORSEMENT AGREEMENT ("Agreement") is made a…” — recall 0.055
- *Consider the Gas Transportation Agreement between PennTex North Louisiana Operating, LLC and MRD Operating LLC; What is …*
  expected `PenntexMidstreamPartnersLp_20150416_S-1A_EX-10` [44720:44931] “This Agreement is entered into in the State of Texas and shall be governed, interpret…”
  returned `PenntexMidstreamPartnersLp_20150416_S-1A_EX-10` [0:1017] “Exhibit 10.4 Execution Version GAS TRANSPORTATION AGREEMENT This Gas Transportation A…” — recall 0.090

#### extra_context

- *Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile…*
  expected `GluMobileInc_20070319_S-1A_EX-10.09_436630_EX-` [598:989] “The rights granted hereunder shall be effective as of the Effective Date and shall ex…”
  returned `GluMobileInc_20070319_S-1A_EX-10.09_436630_EX-` [0:1023] “AMENDMENT NO. 3 Dated as of February 19, 2007 Reference is hereby made to that certai…” — recall 1.000
- *Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; What…*
  expected `HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGR` [5604:5739] “The term of this Agreement shall be one (1) year unless terminated earlier in accorda…”
  returned `HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGR` [19366:20008] “of any of the provisions of this Agreement shall be deemed or shall constitute a waiv…” — recall 1.000
- *Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is t…*
  expected `HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGR` [11380:11583] “GINARES agrees that the sole and exclusive remedy for money damages related to this A…”
  returned `HPILHOLDING_01_07_2015-EX-99.1-COOPERATION AGR` [19366:20008] “of any of the provisions of this Agreement shall be deemed or shall constitute a waiv…” — recall 1.000

#### false_abstention

- *Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Are there a…*
  expected `BICYCLETHERAPEUTICSPLC_03_10_2020-EX-10.11-SER` [33572:34034] “save that nothing in this paragraph 13.2 shall prevent you from holding (with the pri…”
  returned nothing (abstained) — recall 0.000
- *Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Does this c…*
  expected `BICYCLETHERAPEUTICSPLC_03_10_2020-EX-10.11-SER` [6235:6531] “On or as soon as practicable following the Effective Date, it is intended that you wi…”
  returned nothing (abstained) — recall 0.000
- *Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; How is inte…*
  expected `BICYCLETHERAPEUTICSPLC_03_10_2020-EX-10.11-SER` [44840:45359] “To the extent that such IPRs do not vest automatically in the Company by operation of…”
  returned nothing (abstained) — recall 0.000

**Diagnosis of the dominant patterns.**

- **`extra_context` is not a retrieval miss.** It means the ground truth was found
  but the returned chunk carried unrelated text. In a 1,024-character window around
  a 317-character median answer this is unavoidable; it is the precision cost of the
  chunk size that buys recall. Parent–child retrieval is the designed fix and was
  not built (`docs/04-chunking.md`).
- **`wrong_section` has one dominant mechanism: the header magnet.** The query
  restates the contract name and parties, so the document's title/cover chunk
  matches strongly (score 0.91–0.94) and outranks the actual clause. It appears in
  nearly every `wrong_section` example, across three different contracts.
- **`partial_span` is a boundary problem**, not a ranking problem: the right place
  was found and cut in half. That is what overlap was meant to fix, and on this data
  it did not pay for itself.
- **`false_abstention` is the price of the conservative threshold** (Section 5):
  12 of 140 train queries, 14.8% on test. The alternative is answering unanswerable
  queries, which the trade-off curves quantify.

---

## 10. Limitations and next steps

See [`docs/12-risks-and-limitations.md`](docs/12-risks-and-limitations.md) for the
full list. The ones that most affect how these numbers should be read:

- One benchmark, one genre (US commercial contracts, English).
- The mini subset clusters queries onto 29 documents; it is not a random sample.
- The query prefix leaks the document, so prefix routing is not deployable as-is.
- No LLM components; whether an LLM would beat the best non-LLM pipeline here is
  untested.
- Small test set (54 answerable, 12 unanswerable), so intervals are wide.
- **Abstention thresholds do not transfer to differently-phrased questions.** The
  shipped threshold is calibrated on benchmark queries, which name the contract;
  a bare clause question scores lower and is wrongly abstained. The demo config
  (`configs/serve.yaml`) lowers the threshold and is explicitly *not* the evaluated
  configuration.
- The improvement loop is a local search over a finite rule table. It re-opens
  rules after an accepted change (ADR-020) but cannot propose combinations no rule
  encodes; the retrieval grid is the broader check, and where they disagreed the
  report says which one changed the conclusion.
- Character recall can be inflated by returning more text. The improvement loop
  therefore carries a text budget (at most 2.5× the baseline's mean returned
  characters); without it, the best "improvement" was a chunk size that returned
  twice as much of each contract (ADR-017).
- The authored negatives are shorter than benchmark queries. Query length was
  removed from the abstention classifier's features after it separated the classes
  by construction (ADR-006 section, "A confound worth reporting").
- Latency is gated by an absolute budget as well as a ratio (ADR-018). The budget
  is a judgement about what feels interactive, not a measurement.

### Next steps

- **Parent–child retrieval.** Retrieve small spans and return the enclosing
  clause. It directly targets the dominant `extra_context` failure and the
  `partial_span` boundary cases, and is the most likely single improvement to
  precision without giving back recall.
- **Remove the query-prefix dependency.** Evaluate a semantic router or a
  document-classifier step on naturally phrased questions, then recalibrate
  abstention for that distribution. The benchmark prefix makes document routing
  nearly free; a deployed system will not have it.
- **Cross-reference and definition resolution at indexing time.** Resolve
  "as defined in Section 4.2" and carry defined terms into the chunk text. This is
  the part of the messy-contract problem the current chunkers do not address.
- **Measure an LLM reranker or query rewriter.** It is the main optional component
  not tested here. Any such experiment needs a cached, offline-reproducible run and
  a privacy note, because real contracts are confidential.
- **A larger, naturally phrased negative set.** The authored negatives are 46
  queries and are shorter than benchmark queries; the threshold and classifier
  would benefit from a larger set that does not encode authorship.
- **Token-based chunk sizes.** The current sizes are characters, which makes direct
  comparison with token-based papers awkward (ADR-013).

---

## 11. AI tool usage

**LLM use in the system: none.** No LLM is used for query rewriting, retrieval,
reranking, abstention or proposal in the improvement loop. The loop's proposer is a
deterministic rule table, so every accepted change is attributable to a named rule
and a config diff. This was a deliberate scope decision (ADR-008): it keeps the
pipeline auditable and avoids sending contract text off-machine. The cost is that
LLM components remain untested (Section 10).

AI coding assistants were used to write code, tests and documentation. No AI
component runs **inside** the system, and no API key or hosted model is required by
any evaluated configuration.

**How the output was controlled.**

1. Every number in this report is produced by committed code from a committed
   config, not written by hand. `make check` verifies that the required artefacts
   exist and that the experiment log is contiguous.
2. Metric behaviour is pinned by hand-computed fixtures, including the case where
   the paper's precision/recall definition exceeds 1.0 for overlapping chunks.
3. Data integrity is checked before any modelling: all 272 ground-truth spans match
   their stored answer text in both read modes.
4. Design decisions and rejected alternatives are recorded as ADRs
   (`docs/09-decisions.md`), including the ones that changed course.

**Corrections made during development.** These are the places where the first
answer looked fine and a check disagreed:

- An early design note referred to an upstream `mini()` function that does not
  exist; the selection is inline logic with unusual reseeding, and a "cleaner"
  reimplementation would have chosen a different 194 queries.
- A first abstention classifier appeared near-perfect (false abstention 0.000)
  because `query_len` separated the classes by construction — the authored
  negatives are shorter than benchmark queries. It was learning authorship, not
  answerability.
- The sentence chunker produced 125,131 chunks averaging 10 characters because
  dot-leader tables of contents were split and never re-packed.
- The first improvement-loop run accepted a chunk size that quadrupled the text
  returned per query; the guardrail was measuring an absolute precision drop that
  was meaningless near zero.
- A purely relative latency gate rejected a hybrid retriever that gained 0.090
  recall at 42 ms, because the baseline was 2.9 ms.

Each of these is a place where the first answer looked fine and a check disagreed.
The checks are in the repository.
