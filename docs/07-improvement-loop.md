# 07 — Improvement loop

Status: 📋 designed, `loop/` pending.

One command that evaluates on **train only**, diagnoses failures, proposes a
change, and accepts or rejects it by a statistical criterion. Every iteration —
accepted or rejected — is appended to `logs/experiments.jsonl`.

```
make loop CONFIG=configs/baseline.yaml
```

## Objective, fixed before iterating

- **Primary metric:** `recall@10` (character recall at k=10) on train.
- **Guardrails:** character precision@10 and false-abstention rate.
- **Constraint:** latency per query must not exceed 1.5× the baseline's.

Rationale for choosing recall as primary:

1. This is a *retrieval* task: a downstream reader can discard extra context, but
   cannot recover a clause that was never retrieved. Recall is the ceiling on
   answer quality.
2. Precision is directly gamed by returning less text, which is why it is a
   guardrail rather than the target.
3. Abstention must not be improved by abstaining more, which is why false
   abstention is a guardrail rather than a term in the objective.

The report justifies this choice explicitly and shows the precision/recall
trade-off for the accepted changes, so the objective cannot hide a precision
collapse.

## Step 1 — Evaluate

Run the current config on all 140 train queries, 5-fold cross-validation by query
hash. Store per-query results, not just aggregates — diagnosis needs them.

## Step 2 — Diagnose

Every failed query is assigned exactly one category, in this fixed order
(first match wins):

| Category | Rule | Signal |
|---|---|---|
| `wrong_document` | no retrieved span is in a GT document | routing / index coverage problem |
| `wrong_section` | right document, no overlap with any GT span | ranking within a document |
| `partial_span` | overlaps a GT span but does not fully contain it | chunk too small / boundary problem |
| `extra_context` | fully contains GT but retrieved far more text | chunk too large / precision problem |
| `false_abstention` | abstained on an answerable query | abstention policy too aggressive |
| `missed_abstention` | returned something for an unanswerable query | abstention policy too lax |

Also record, per failure: the top-1 score, the rank at which the first GT hit
appeared (if any), and the category (`derive_category`).

Aggregate the failures into a **dominant failure profile**: the counts per
category and the fraction of total loss attributable to each. Loss is weighted by
`1 - recall(q)`, so fixing ten near-complete queries counts less than fixing one
total miss.

## Step 3 — Propose

A rule table maps the dominant failure to candidate changes. This is deterministic
and auditable — no LLM (cut, ADR-008).

| Dominant failure | Candidate changes (in order) |
|---|---|
| `wrong_document` | enable `semantic` router; strip doc hint from query; increase `k`; check chunk coverage of the doc |
| `wrong_section` | increase `k`; try hybrid RRF; try a reranker; try a different embedder |
| `partial_span` | reduce chunk size; add parent expansion; reduce overlap |
| `extra_context` | reduce `k`; reduce chunk size; enable reranker to sharpen the top of the list |
| `false_abstention` | raise the abstention threshold; switch policy |
| `missed_abstention` | lower the threshold; add a feature to the classifier |

Each rule produces a **concrete config diff**, not prose. The proposer writes the
candidate to `configs/generated/iter_NN.yaml` and records which rule fired.

Stopping conditions: no rule fires, the proposal duplicates a previously rejected
one, or 8 iterations are reached.

## Step 4 — Accept or reject

Three gates, all must pass:

1. **Effect size:** mean `recall@10` improvement across CV folds ≥ 0.01 absolute.
2. **Statistical significance:** paired bootstrap over train queries
   (`B = 10,000`, resampling queries, paired by query) gives a 95% interval on the
   delta that excludes 0.
3. **Guardrails:** precision@10 must not drop by more than 0.02, false abstention
   must not rise by more than 0.02, and latency must stay within 1.5× baseline.

If all pass → accept: the candidate becomes the new current config. Otherwise →
reject, with the failing gate recorded.

⚠️ **Why not "the number went up":** with 140 queries, one query flipping can move
recall by ~0.007. An improvement of 0.01 is barely above that noise floor, which is
why the paired bootstrap is mandatory rather than decorative.

## Step 5 — Log

Append one JSON line per iteration to `logs/experiments.jsonl`. Schema is defined
in `10-experiment-log.md`. Both accepted and rejected iterations are logged, so
the report can show that the loop genuinely explored rather than narrating a
pre-decided path.

## Honesty guarantees

| Guarantee | Mechanism |
|---|---|
| Loop sees train only | `run_loop.py` refuses a config whose `data.split` is `test` |
| Ground truth never reaches inference | `pipeline.search(query: str)` has no access to spans |
| No silent test peeking | test evaluation appends to `logs/test_runs.log`; the report states the count |
| Proposals are reproducible | rule table is code, not an LLM; candidate configs are committed |
| Rejections are visible | JSONL contains `accepted: false` entries with the failing gate |
| Fold discipline | folds are derived from a hash of the query, fixed before the run |

## Minimum deliverable

At least **5 iterations beyond the baseline**, with a mix of accepted and rejected
outcomes. Five accepted iterations is a red flag for a too-lenient criterion, not
a success.

## Tests

1. Diagnoser: one fixture per failure category, each asserting the label.
2. Diagnoser precedence: a query that is both `partial_span` and `extra_context`
   is labelled by the documented order.
3. Proposer: every dominant failure category maps to at least one candidate, and
   candidates are valid configs (`validate_config` passes).
4. Acceptor: a synthetic +0.05 improvement with tight variance is accepted; a
   +0.005 improvement is rejected.
5. Acceptor: a large recall gain that halves precision is rejected by the
   guardrail.
6. Loop: a dry run with a stubbed evaluator writes exactly one JSONL line per
   iteration, including rejections.
7. Loop refuses `data.split: test`.
