# 06 — Abstention

Status: 🚧 `none` and `threshold` policies implemented and evaluated as the
baseline; margin, per-type and classifier pending, as is `data/unanswerable.jsonl`.

Abstention is treated as its own problem with its own dataset, not as a magic
score threshold. The comparison is a **trade-off curve**: false abstention on
answerable queries against correct abstention on unanswerable ones.

## The unanswerable set

Target **40–50 queries** (the checklist floor is 30; the design target is 40–50 so
per-type breakdowns are meaningful). Four deliberately different types:

| Type | Description | Why it is hard | Example shape |
|---|---|---|---|
| 1 | Clause type absent from *every* indexed contract | Should be easy; catches systems that always return something | "What is the cap on liability?" when no contract has such a clause |
| 2 | Contract not in the corpus | Tests routing / out-of-scope detection | "What is the governing law of the Acme 2024 MSA?" |
| 3 | Hard negative: clause type is real in the corpus, but **this** contract lacks it | Requires document-specific knowledge, not topic matching | Ask contract A about a clause only contracts B/C have |
| 4 | Near-miss vocabulary: related clause exists, the asked one does not | Lexical overlap defeats score thresholds | "termination for convenience" where only "termination for cause" exists |

Types 3 and 4 are where single-threshold systems fail, and the per-type breakdown
is the error-analysis centrepiece.

### Freezing protocol

1. Draft generated with BM25 near-miss mining, then **verified by a human against
   the actual contracts** — a negative is only valid if the clause really is
   absent.
2. Each entry records: `query`, `type` (1–4), `rationale`, `related_docs`,
   `related_clauses`, and its `split` from `assign_split`.
3. The set is committed **before** the improvement loop starts, in its own
   commit, and is never edited afterwards. The commit is the evidence.
4. It is split with the same `assign_split` function as the benchmark: ~30% test.
   With 45 queries that is ~13 test negatives — small, so test abstention numbers
   are reported with an interval, not three significant figures.
5. Validation: a script re-checks that no query in the set is answerable by any
   ground-truth span (exact and fuzzy containment), so a mislabelled negative
   cannot quietly poison the curve.

## The four policies

### 1. Global threshold on the top score

`abstain if top_score < τ`

- One parameter, easy to explain.
- Breaks down because scores are not comparable across query types or retrievers:
  a BM25 score of 8 and a cosine of 0.62 mean different things.
- ⚠️ This is the strawman the plan criticises. It must be included so the
  improvement is measurable, not assumed.

### 2. Margin rule

`abstain if top1 - top2 < τ` (optionally combined with a low absolute floor)

- Targets "I found several equally mediocre things" — the signature of an
  unanswerable query.
- Degrades when the system returns one high and several low scores by
  construction (e.g. a reranker that squashes the tail).

### 3. Per-type (per-category) threshold

A threshold calibrated separately for each clause category on train.

- Rationale: some clause questions ("governing law") are almost always present
  somewhere; others are rare and yield weak matches everywhere.
- ⚠️ Risk: 32 categories over 140 train queries means ~4 queries per category.
  Mitigation: calibrate per category where `n ≥ 8`, fall back to the global
  threshold elsewhere. Report how many categories fell back. This is honest and
  still tests the hypothesis.

### 4. Logistic regression on retrieval features

Features (all computable from retrieval output alone — no ground truth):

| Feature | Intuition |
|---|---|
| `top1_score`, `top2_score`, `top5_mean` | absolute confidence, retriever-normalised (rank-based or z-scored within query) |
| `score_gap_12` | margin |
| `doc_match_confidence` | router's belief (0 if no router) |
| `bm25_dense_agreement` | Jaccard of the two top-10 document/span sets |
| `n_above_floor` | how many results clear a low absolute floor |
| `retrieved_len_total` | how much text would be returned |
| `query_len`, `category` (one-hot for frequent categories) | query-type signal |

- Trained on train negatives (unanswerable, types 1–4) + train positives
  (answerable, label 0), class-weighted because negatives are the minority.
- **Leakage guard:** features must never include ground-truth overlap. The only
  labels used are "answerable / unanswerable", which is the task definition.
- Cross-validated within train; the reported curve uses out-of-fold predictions,
  not in-sample ones.

## Curve construction

For each policy, sweep its threshold (or decision boundary) and record:

- `1 - false_abstention_rate` on answerable train queries
- `correct_abstention_rate` on unanswerable train queries, overall **and per type**
- `abstention_precision` (fraction of abstentions that were correct)

Plot the curve; summarise with AUC **only** as a tiebreaker. Pick the operating
point on train by an explicit rule (e.g. "correct-abstention ≥ 0.80 with the
lowest false-abstention"), state the rule in advance, and apply it unchanged to
test.

## Interaction with answerable-query metrics

If the system abstains on an answerable query, that query is excluded from the
precision mean and counted in `false_abstention_rate`; recall is counted as 0
because nothing was returned. This asymmetry is deliberate: abstaining is not a
free pass, but it should not be scored as "returned nothing relevant, precision 0"
either. Recorded in ADR-006.

## Tests

1. Threshold policy: monotone behaviour — raising τ never decreases abstention.
2. Margin policy: identical top-1 and top-2 scores → abstain.
3. Per-type: a category with fewer than `min_support` train examples falls back to
   the global threshold.
4. Classifier: does not accept features containing ground-truth fields (assert on
   the feature-name whitelist).
5. Negatives validity: no unanswerable query's text is answerable by an existing
   span (fixture + real-set check).
6. Curve: the (0-false-abstention, 0-correct-abstention) and (1,1) corners are
   reachable by the lowest and highest thresholds respectively.
