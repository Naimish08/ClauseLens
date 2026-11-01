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

### Measured composition (2026-11-01, frozen)

`data/unanswerable.jsonl` contains **46 entries**, committed before any tuning:

| Type | Entries | Example |
|---|---|---|
| 1 | 12 | "Which agreement requires the licensor to place its source code in escrow…" |
| 2 | 10 | "What is the governing law of the Northwind Logistics Master Services Agreement?" |
| 3 | 14 | ROFR asked of an agency agreement that lacks it (two other contracts have one) |
| 4 | 10 | "Can Wildform terminate … for convenience?" where clause 9.2 allows only for-cause |

Split by the same `assign_split`: **34 train / 12 test**. Per type the test
support is uneven (type 1: 1, type 2: 5, type 3: 4, type 4: 2), so test-split
abstention is reported per type with `n` and no significance claims, and the
headline abstention comparison uses train.

Verification: every `absent_patterns` regex is checked against the claimed scope
(the whole corpus for types 1–2, the specific document for types 3–4) by
`make negatives`, and every type-4 entry must prove its near-miss neighbour is
present. `scripts/mine_negatives.py` produced the candidates.

⚠️ **The check earned its keep.** Three drafted entries were rejected and replaced:

- "tag-along rights" as a type-1 candidate — the IGENE joint venture has an express
  *Section 12.1 RIGHT OF FIRST REFUSAL; TAG-ALONG RIGHTS*. Replaced with a
  jury-trial-waiver query, which matches nothing anywhere.
- "termination for convenience" against deltathree — the contract grants a
  termination right exercisable "at any time" after a change of control. Replaced
  with the On2 support agreement, which allows termination only for uncured
  material breach.
- "insurance" against Dynamax — the contract *does* require insurance. Replaced
  with the PennTex gas transportation agreement.

A fourth near-miss exposed a regex pitfall: PennTex matched `/insur/` only because
of the word *insurrections* in a force-majeure clause. The stored patterns use
`insurance|insurer|insured`, not a substring match, and the burden is on the
pattern to be precise.

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
| ~~`query_len`~~ | **excluded**: separated the classes by construction (see below) |

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


### Measured results (2026-11-01)

`python -m eval.run_abstention --config configs/baseline.yaml` — BM25 over
fixed-512 chunks, 140 answerable train queries and 34 train negatives, about 2
seconds. Figure: [`figures/abstention_curves.png`](figures/abstention_curves.png).

The operating point rule was fixed before looking at the numbers: *the lowest
false-abstention rate among thresholds with correct abstention ≥ 0.80; if
unreachable, the highest abstention F1*. Fitted policies use out-of-fold
confidences, and the operating point itself is selected **nested** — chosen on the
other four folds and applied to the held-out fold — so the reported numbers do not
benefit from selecting the threshold on the same queries.

| policy | false abst. | 95% CI | correct abst. | 95% CI | abst. F1 |
|---|---|---|---|---|---|
| threshold | 0.079 | [0.036, 0.121] | 0.794 | [0.647, 0.912] | 0.853 |
| margin | 0.479 | [0.393, 0.564] | 0.794 | [0.647, 0.912] | 0.630 |
| category | 0.093 | [0.050, 0.143] | **0.853** | [0.735, 0.971] | **0.879** |
| classifier | **0.050** | [0.014, 0.086] | 0.794 | [0.647, 0.912] | 0.865 |

Per-type correct abstention at the selected points (threshold policy): type 1
1.00, type 2 1.00, type 3 0.50, type 4 1.00.

1. **The strawman is strong.** A single global threshold on the BM25 top score
   reaches F1 0.853, and nothing beats it by a margin that survives the interval.
   The build plan predicted that a fixed threshold would do badly; on this data it
   does not, because negative queries have systematically weaker lexical matches.
   That is a finding, not a failure of the experiment.
2. **Margin is clearly the worst** (FAR 0.479). It abstains whenever the top two
   results are close, which is common for answerable queries in long contracts.
3. **Category normalisation is nominally best** (F1 0.879, CAR 0.853) but its
   interval overlaps the threshold policy's. The honest claim is "at least as good
   as a global threshold", not "better".
4. **The classifier trades recall for precision.** Lowest false abstention (0.050)
   at slightly lower correct abstention. No significant advantage.
5. **Type 3 hard negatives are the discriminator.** Types 1, 2 and 4 are separated
   almost perfectly by score; the queries where the topic exists but the specific
   clause does not are where policies differ (0.50–0.70). With only 10 type-3
   train negatives, that difference is not estimable with confidence.
6. **All intervals are wide** (34 negatives). No ranking of these policies is
   defensible on this sample; the report says so rather than publishing a league
   table.

### A confound the first run exposed

With `query_len` in the feature vector, the classifier looked perfect: false
abstention 0.000, correct abstention 0.941, abstention precision 1.000. Inspecting
coefficients showed why — `query_len` had the largest weight by far (−3.65) and
the authored negatives are 51–151 characters against benchmark queries of 134–243.
The model had learned **who wrote the query**, not whether it was answerable.

`query_len` was removed from `FEATURE_NAMES`, and
`tests/test_abstain.py::test_query_length_is_not_a_model_feature` guards against
reintroducing it. After removal the classifier performs like the other policies.
This is the only defence that matters when negatives are authored rather than
sampled: check that the features cannot separate the classes by construction.

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
