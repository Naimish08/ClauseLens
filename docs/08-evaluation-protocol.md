# 08 — Evaluation protocol

Status: implemented. Every table in the report is produced by a command in this
document and a config under `configs/`.

This document is the contract for how numbers are produced. If a number in the
report cannot be traced to a config file and a command here, it does not belong in
the report.

## Splits

| Split | Size | Use |
|---|---|---|
| train | 140 benchmark queries | all development, tuning, loop iterations |
| test | 54 benchmark queries | final reporting only |
| unanswerable train | 34 negatives | abstention development |
| unanswerable test | 12 negatives | final abstention reporting |

Sub-splits **within train** are allowed for cross-validation (5 folds, derived
from a hash of the query text so they are stable and cannot be reshuffled to
please a result). The train/test boundary is never re-derived.

## Test-set discipline

1. `eval/run_eval.py --split test` and `eval/run_abstention.py --split test` both
   require an explicit `--i-know-this-is-final` flag; without it the run aborts.
2. Every test run appends a line to `logs/test_runs.log`:
   `{timestamp, config, git_sha, commit_dirty, reason}`.
3. The report states the number of test runs and what each was for. If it was run
   more than once, say why (e.g. "the first run failed on a missing embedding
   cache; the second is reported").
4. No change is made after a test run without the report acknowledging it.

This is deliberately slightly inconvenient to do — the friction is the point.

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

Ablation evidence comes from two places: the independent grids (chunking and
retrieval) and the loop's one-change-at-a-time record. The report presents the
grids as the controlled comparisons and the loop as the incremental history; it
does not claim a cumulative ablation that was not run.

## Tables in the report

1. **Chunking grid** — chunker × size × overlap, at fixed BM25, `k=10`: recall,
   precision, span-hit, chunk-hit. Plus the precision-vs-size plot.
2. **Retrieval comparison** — BM25, each dense model, hybrid RRF, weighted hybrid,
   hybrid + rerank. With latency.
3. **Query preprocessing ablation** — full query vs stripped document hint. This is
   the routing comparison in the report; a semantic router is not shipped (ADR-010).
4. **Abstention** — per-policy curve on train, plus per-negative-type correct
   abstention, and a fixed-threshold test-split result in both directions.
5. **Train vs test** — baseline and final, side by side, with query counts.
6. **Iteration history** — from `logs/experiments.jsonl`.
7. **Latency** — per method, median and p95.

## Error analysis requirements

At least **10 concrete failures**, grouped by cause, each with:

- the query text (or a clearly truncated form that does not reproduce a full answer)
- the failure category from `07-improvement-loop.md`
- what was retrieved: document and character span, with the excerpt
- what should have been retrieved
- a one-sentence diagnosis ("BM25 ranked the definitions section first because the
  query term appears in three defined terms; the actual clause uses a synonym")

The 10 must span at least four distinct causes. Ten instances of `wrong_document`
is not an analysis.

## Reproducibility checklist

- Every table row maps to a config file under `configs/`.
- Every table row has a recorded `run_id`.
- Seeds are in the config, not in the shell.
- Embeddings and rerank scores are cached to disk, keyed by model + config hash.
- `make setup && make data && make eval` reproduces the baseline within float
  tolerance; verified on 2026-11-01 in the recorded environment.
- Hardware and library versions are recorded (see `11-reproducibility.md`).

## Latency protocol

- Measure on the same machine, warmed up (one discarded warm-up query per config).
- Report median and p95 per query, plus index build time separately.
- Indexing time is not part of query latency but is reported, because it is the
  real cost of a reranker or a large embedder.
