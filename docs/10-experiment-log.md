# 10 — Experiment log

Status: 📋 schema defined, `loop/` pending.

`logs/experiments.jsonl` is the append-only record of every loop iteration,
accepted **and** rejected. It is committed to the repository; the git history plus
this file are the evidence that the loop did real work.

## Rules

1. Append-only. Never rewrite or delete a line — including lines from rejected or
   buggy iterations. If a run was invalid, add a new line with `"invalid": true`
   and a reason.
2. One JSON object per line, no trailing commas, UTF-8.
3. Written by `loop/run_loop.py` only. Hand-editing is a data-integrity bug.
4. Each line records the git commit it produced (or `null` for rejections, which
   produce no commit).

## Schema

```jsonc
{
  "iteration": 0,                       // 0 = baseline, then 1..N
  "timestamp": "2026-11-01T14:05:00Z",
  "accepted": true,
  "invalid": false,

  "proposal": {
    "proposer": "baseline",             // baseline | rule:<id> | manual
    "trigger": "n/a",                   // dominant failure category, or n/a
    "reason": "initial reference pipeline",
    "diff": {                           // config keys changed, old -> new
      "chunking.params.size": {"from": null, "to": 512}
    },
    "config_path": "configs/baseline.yaml"
  },

  "evaluation": {
    "split": "train",
    "n_queries": 140,
    "folds": 5,
    "primary_metric": "recall@10",
    "primary": 0.512,
    "primary_delta": null,              // null for the baseline
    "primary_ci95": null,               // [lo, hi] on the delta
    "guardrails": {
      "precision@10": 0.081,
      "false_abstention_rate": 0.0,
      "latency_ms_p95": 42.1
    },
    "secondary": {
      "mrr@10": 0.61,
      "doc_hit@10": 0.94,
      "span_hit@10": 0.88,
      "chunk_hit@10": 0.71
    }
  },

  "diagnosis": {
    "dominant_failure": "partial_span",
    "counts": {
      "wrong_document": 6,
      "wrong_section": 11,
      "partial_span": 23,
      "extra_context": 9,
      "false_abstention": 0,
      "missed_abstention": 0
    }
  },

  "decision": {
    "gate_effect_size": true,           // delta >= 0.01
    "gate_significance": true,          // 95% CI excludes 0
    "gate_guardrails": true,
    "notes": "accepted: +0.031 recall, precision -0.006"
  },

  "run_id": "baseline-train-a1b2c3d-20261101T140500",
  "git_commit": "a1b2c3d"
}
```

## Field notes

- **`iteration`** is sequential and never reused. A re-run of the same iteration
  gets a new line with `"rerun_of": <iteration>`.
- **`primary_delta` / `primary_ci95`** come from the paired bootstrap on the same
  fold assignment as the previous accepted iteration.
- **`diagnosis.counts`** count *queries*, and the categories are defined in
  `07-improvement-loop.md`. The counts should sum to the number of failing
  queries, not to `n_queries`.
- **`diff`** is the authoritative record of what changed. Reading the diff must be
  enough to reconstruct the candidate config without opening the YAML.
- **`git_commit`** is the commit that made the change permanent. Rejected
  iterations have `null`, because rejecting means nothing is committed but the log
  line.
- **`run_id`** links to `artifacts/runs/{run_id}/` for full per-query detail.

## Derived artefacts

`eval/tables.py` (or a small `scripts/log_report.py`) generates the report's
iteration-history table directly from this file:

```
| # | Change | Failure addressed | Recall@10 | Δ vs prev | Accepted | Why |
```

Generated, not transcribed. If the table in the report and the JSONL disagree, the
report is wrong.

## Why both accepted and rejected iterations are logged

The brief explicitly requires it, and it is the cheapest defence against the
impression of hindsight: a log containing only accepted changes looks like a plan
written after the fact. A log with, say, 8 iterations and 3 rejections shows the
search actually happened — and the rejections are themselves evidence that the
acceptance criterion has teeth.

## Integrity checks

`scripts/log_report.py --validate` checks:

1. `iteration` values are unique and contiguous from 0.
2. Every accepted iteration after 0 has exactly one `diff` key differing from its
   predecessor.
3. Every accepted iteration has a non-null `git_commit`.
4. Every rejected iteration has at least one failing gate.
5. `diagnosis.counts` sum equals the number of failing queries recorded in the
   run's `per_query.jsonl`.
