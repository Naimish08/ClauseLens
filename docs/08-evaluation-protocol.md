# 08 — Evaluation protocol

Status: 📋 designed, `eval/` pending.

This document is the contract for how numbers are produced. If a number in the
report cannot be traced to a config file and a command here, it does not belong in
the report.

## Splits

| Split | Size | Use |
|---|---|---|
| train | 140 benchmark queries | all development, tuning, loop iterations |
| test | 54 benchmark queries | final reporting only |
| unanswerable train | ~70% of 40–50 negatives | abstention development |
| unanswerable test | ~30% of 40–50 negatives | final abstention reporting |

Sub-splits **within train** are allowed for cross-validation (5 folds, derived
from a hash of the query text so they are stable and cannot be reshuffled to
please a result). The train/test boundary is never re-derived.

## Test-set discipline

1. `eval/run_eval.py --split test` requires an explicit `--i-know-this-is-final`
   flag; without it the run aborts.
2. Every test run appends a line to `logs/test_runs.log`:
   `{timestamp, config, git_sha, commit_dirty, reason}`.
3. The report states the number of test runs and what each was for. If it was run
   more than once, say why (e.g. "the first run failed on a missing embedding
   cache; the second is reported").
4. No change is made after a test run without the report acknowledging it.

This is deliberately slightly annoying to do — the friction is the point.

## What is measured, per query

For each `k ∈ {1, 2, 4, 8, 16}`:

- character `recall@k` and `precision@k` (paper semantics; `union` variant
  available, see ADR-005)
- `doc_hit@k`, `span_hit@k`, `chunk_hit@k`, `mrr@k`
- latency (ms) for the query, per stage

For the abstention set: `abstained`, `correct`, `type`, plus the policy's score.

Outputs written per run:

```
artifacts/runs/{run_id}/
  config.yaml            exact config used (copy, not a reference)
  per_query.jsonl        one line per query with all metrics
  aggregate.json         means, counts, confidence intervals
  table.md               the markdown table for the report
  table.csv              the same, for plotting
  latency.json           per-stage timings
```

`run_id` is `{config_name}-{split}-{short_git_sha}-{timestamp}`, so a table is
always traceable.

## Aggregation

- Macro-average over queries (see `03-metrics.md`).
- Never average already-averaged tables.
- Group by clause category (`derive_category`) and always print `n`.
- Group by failure category for the error analysis, not for headline numbers.

## Comparisons and uncertainty

Every headline comparison is **paired** (same queries, two systems):

- paired bootstrap, `B = 10,000`, 95% percentile interval on the mean delta
- report the delta with its interval: `+0.041 [0.012, 0.071]`
- report the win/loss/tie counts as a sanity check (a +0.04 mean with 5 wins and
  40 losses would be suspicious)

Ablation table: baseline plus one component at a time, in the order chunking →
retrieval → rerank → routing → abstention. One row per cumulative step, plus rows
for the individual components added alone, so both "incremental" and "isolated"
effects are visible.

## Required tables for the report

1. **Chunking grid** — chunker × size × overlap, at fixed BM25, `k=10`: recall,
   precision, span-hit, chunk-hit. Plus the precision-vs-size plot.
2. **Retrieval comparison** — BM25, each dense model, hybrid RRF, weighted hybrid,
   hybrid + rerank. With latency.
3. **Routing ablation** — none / prefix / semantic, at the best retrieval config.
4. **Query preprocessing ablation** — full query vs stripped doc hint.
5. **Abstention** — per-policy curve, plus per-negative-type correct-abstention.
6. **Ablation table** — cumulative and isolated.
7. **Per-category table** — with `n` and the `n<5` grouping.
8. **Train vs test** — baseline and final, side by side, with query counts.
9. **Iteration history** — from `logs/experiments.jsonl`.
10. **Latency** — per method, median and p95.

## Error analysis requirements

At least **10 concrete failures**, grouped by cause, each with:

- the query text (or a clearly truncated form that does not identify a confidential party)
- the failure category from `07-improvement-loop.md`
- what was retrieved: document and character span, with the excerpt
- what should have been retrieved
- a one-sentence diagnosis ("BM25 ranked the definitions section first because the
  query term appears in three defined terms; the actual clause uses a synonym")

The 10 must span at least four distinct causes. Ten instances of `wrong_document`
is not an analysis.

## Reproducibility checklist

- [ ] every table row maps to a config file under `configs/`
- [ ] every table row has a recorded `run_id`
- [ ] seeds are in the config, not in the shell
- [ ] embeddings and rerank scores are cached to disk (keyed by model + config hash)
- [ ] `make eval` from a clean clone reproduces the baseline within float tolerance
- [ ] hardware and library versions are recorded (see `11-reproducibility.md`)

## Latency protocol

- Measure on the same machine, warmed up (one discarded warm-up query per config).
- Report median and p95 per query, plus index build time separately.
- Indexing time is not part of query latency but is reported, because it is the
  real cost of a reranker or a large embedder.
