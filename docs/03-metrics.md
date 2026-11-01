# 03 — Metrics

Status: ✅ implemented (`src/metrics.py`, fixtures in `tests/test_metrics.py`).
Reference implementation: `legalbenchrag/run_benchmark.py` (`QAResult`).

## The primary pair: character precision and recall

For a single query, let

- `GT` = the set of ground-truth character spans, all in the same document
- `R@k` = the set of spans actually retrieved (the top `k` results)
- `overlap` = total characters covered by both `GT` and `R@k`

Then:

```
recall(q, k)    = overlap_length / total_gt_length
precision(q, k) = overlap_length / total_retrieved_length
```

and the reported numbers are the query-wise **means**:

```
Recall@k    = (1/|Q|) * Σ_q recall(q, k)
Precision@k = (1/|Q|) * Σ_q precision(q, k)
```

Notes that matter:

1. **It is macro-averaged per query, not concatenated.** A query with a 1,756-char
   answer does not outvote a query with a 67-char answer. Averaging over queries is
   what the reference code does (`avg_precision` / `avg_recall` over
   `qa_result_list`).
2. **Both are zero when the query has no ground truth** in the reference code
   (`if total_relevant_len == 0: return 0`). Ground-truth-free queries are the
   abstention set and are scored separately, not folded into this mean.
3. **Precision is bounded above by `gt_length / retrieved_length`.** If a query's
   answer is 317 chars and you retrieve one 2,000-char chunk that contains it,
   precision is at most 0.16. This is not a bug; it is the metric telling you
   that returning whole documents is not "retrieval".
4. Precision is *undefined-by-convention* (reported as 0) when nothing is
   retrieved. If you abstain correctly, you should not be punished by the
   answerable-query precision mean. Decision: excluded from the precision mean
   when the system abstains on an answerable query, and counted in the
   false-abstention rate instead. This is stated in the report.

## Sum versus union — a real discrepancy

The build plan says "take the **union** of the retrieved top-k spans". The
reference implementation does **not** union: it sums per-retrieved-snippet
lengths.

```python
for snippet in self.retrieved_snippets:
    total_retrieved_len += snippet.span[1] - snippet.span[0]
    for gt_snippet in self.qa_gt.snippets:
        if snippet.file_path == gt_snippet.file_path:
            ...  # adds pairwise overlap
```

For non-overlapping retrieved spans the two definitions agree. They diverge
exactly when retrieval returns **overlapping chunks** — which is precisely what
fixed-size chunking with `overlap > 0` does. In that case the sum-based
definition counts the duplicated characters twice in both numerator and
denominator.

Decision (ADR-005): implement the reference semantics as `mode="paper"` (default,
for comparability with published numbers) **and** provide `mode="union"` as an
option. Every results table states which was used. The chunking grid reports both
when `overlap > 0`, so the discrepancy is visible instead of hidden.

⚠️ **Measured consequence:** with overlapping retrieved chunks, `paper` recall can
exceed 1.0 — it is a ratio of summed per-snippet overlaps to a 100-character
ground truth, and duplicated characters are counted twice. The fixture
`test_overlapping_retrieved_spans_change_recall_between_modes` pins the exact
values (`paper` recall 1.2, `union` recall 1.0). A recall above 1.0 in a table is
therefore a signal that the table used `paper` mode with overlapping chunks, not a
bug in the harness. This is precisely why the `union` variant exists.

## Secondary metrics

All are computed per query over the same top-k list, then macro-averaged.

| Metric | Definition | Why it is here |
|---|---|---|
| `doc_hit@k` | 1 if any retrieved span is in a document containing a GT span | Separates "found the contract" from "found the clause". With 29 docs this saturates fast, which is itself informative. |
| `span_hit@k` | 1 if any retrieved span overlaps any GT span by ≥1 char | Lenient retrieval success; shows whether failure is ranking or bounding. |
| `chunk_hit@k` | 1 if any retrieved span **fully contains** a GT span | Strict; exposes chunk-too-small failures. |
| `mrr@k` | mean of `1/rank` of the first span that hits | Rank-sensitive; precision@k is not. |
| `precision@k`, `recall@k` | above | headline |
| `f1@k` | harmonic mean of the two means | single-number summary, secondary only |
| `latency_ms` | wall-clock per query, excluding one warm-up query | a reranker that costs 2s must justify itself |

`span_hit` vs `chunk_hit` is the pair that diagnoses "right place, wrong
granularity": high span-hit with low chunk-hit means chunks are too small to
contain a whole answer; high chunk-hit with low precision means chunks are too
large.

## k values

Report `k ∈ {1, 2, 4, 8, 16}` (the reference grid also uses 32 and 64; with
29 documents and ~3–6k chunks, `k > 16` mostly measures how much text you are
willing to dump, and precision collapses).

Primary optimisation target: **recall@10** (see `07-improvement-loop.md`).
Headline report metric: **character precision and recall at the same k**, because
that is what the paper reports and what the assignment asks for.

## Abstention metrics

Abstention is evaluated on two disjoint query sets and reported as a curve, never
as one number.

Let `A` = answerable (the 194 benchmark queries), `U` = unanswerable (the frozen
negative set).

| Metric | Definition |
|---|---|
| `false_abstention_rate` | fraction of `A` where the system abstains (want low) |
| `correct_abstention_rate` (on `U`) | fraction of `U` where the system abstains (want high) |
| `abstention_precision` | of all abstentions, fraction that were on `U` |
| `abstention_f1` | harmonic mean of `1 - false_abstention_rate` and `correct_abstention_rate` |

The trade-off curve plots `1 - false_abstention_rate` against
`correct_abstention_rate` while sweeping the policy's threshold. Methods are
compared as curves (AUC summarised only as a tiebreaker), because a single
threshold can always be tuned to look good on one side.

Per-type breakdown of `correct_abstention_rate` is mandatory, because the four
negative types have very different difficulty (`06-abstention.md`).

## Per-category breakdown

Category = the clause question after the first `;` (`derive_category`). 32
categories, support ranging from 24 queries to 1.

Reporting rules:

1. Always print `n` next to every category number.
2. Categories with `n < 5` are grouped as "other (n<5)" in the headline table and
   listed individually only in the appendix.
3. Do not rank categories by point estimate alone; a 1-query category scoring
   1.0 is noise.

## Statistical treatment

- **Paired bootstrap** for every headline comparison (baseline vs final, method A
  vs method B): resample queries with replacement `B = 10,000` times, report the
  mean difference and its 95% percentile interval. Paired because both systems
  answer the same queries.
- **Report intervals, not just p-values.** "Final beats baseline by +0.041 recall
  [0.012, 0.071]" is the claim format.
- **Within-train cross-validation** (5 folds by query hash) for the loop's
  accept/reject decision, so a change is not accepted on one lucky split.
- With 140 train queries the bootstrap interval is wide (roughly ±0.04 on recall
  near 0.5). The report says so rather than pretending to two decimal places.

## Metric unit tests (required before any results)

Hand-computed fixtures, not property tests alone:

1. Perfect retrieval of a single span → precision 1.0, recall 1.0.
2. Retrieve exactly half of the GT span → precision = 0.5 only if nothing else
   was retrieved; recall = 0.5.
3. Retrieve the whole GT span plus an equal amount of extra text → precision 0.5,
   recall 1.0.
4. Two non-overlapping GT spans, retrieve one → recall 0.5, precision 1.0.
5. Overlapping retrieved chunks → `paper` and `union` disagree; assert both
   values explicitly (this is the ADR-005 regression test).
6. Empty retrieval → both 0, no exception, no `ZeroDivisionError`.
7. Retrieval from the wrong document → 0 for both, but `doc_hit` 0 and
   `span_hit` 0.
8. Multi-document GT where only one document is retrieved → recall proportional
   to that document's share.

Anything that makes the headline numbers move must be pinnable by a fixture.
