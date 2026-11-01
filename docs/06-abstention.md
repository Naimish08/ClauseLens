# 06 — Abstention

Status: implemented and measured. All four policies and the 46-query frozen
unanswerable set are in place (`make abstention`).

Abstention is treated as its own problem with its own dataset, not as a magic
score threshold. The comparison is a **trade-off curve**: false abstention on
answerable queries against correct abstention on unanswerable ones.

## The unanswerable set

The set contains **46 queries**, above the checklist floor of 30 and large enough
for per-type breakdowns. Four deliberately different types:

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

**The check caught three drafting errors.** Three drafted entries were rejected and replaced:

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

1. Draft generated with BM25 near-miss mining, then **verified against the actual
   contracts** — a negative is only valid if the clause really is absent.
2. Each entry records: `query`, `type` (1–4), `rationale`, `related_docs`,
   `related_clauses`, and its `split` from `assign_split`.
3. The set is committed **before** the improvement loop starts, in its own
   commit, and is never edited afterwards. The commit is the evidence.
4. It is split with the same `assign_split` function as the benchmark: ~30% test.
   With 46 queries, 12 are in test — small, so test abstention numbers are
   reported with an interval, not three significant figures.
5. Validation: a script re-checks that no query in the set is answerable by any
   ground-truth span (exact and fuzzy containment), so a mislabelled negative
   cannot quietly poison the curve.

## The four policies

### 1. Global threshold on the top score

`abstain if top_score < τ`

- One parameter, easy to explain.
- Breaks down because scores are not comparable across query types or retrievers:
  a BM25 score of 8 and a cosine of 0.62 mean different things.
- This is the simple baseline that a score threshold represents. It is included so
  the improvement is measurable, not assumed.

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
- **Risk:** 32 categories over 140 train queries means ~4 queries per category.
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


### Measured results (2026-11-01, final config)

`python -m eval.run_abstention --config configs/final.yaml` — dense gte-base over
fixed-1024 chunks, 140 answerable train queries, 34 train negatives, ~20 seconds.
Nested operating points (threshold chosen on four folds, applied to the fifth):

| policy | false abst. | correct abst. | abst. F1 |
|---|---|---|---|
| threshold | 0.286 | 0.824 | 0.765 |
| margin | 0.700 | 0.824 | 0.440 |
| category | 0.264 | 0.824 | **0.777** |
| classifier | 0.557 | 0.765 | 0.561 |

1. **Dense retrieval breaks score-based abstention.** The same protocol on the BM25
   baseline gave false abstention 0.079 at correct abstention 0.794; on dense it is
   0.286 at 0.824. Dense retrieval always finds *some* similar chunk, so the top
   score no longer separates answerable from unanswerable queries. This is the most
   transferable finding in the section: an abstention policy calibrated on a lexical
   retriever does not transfer to a dense one.
2. **Margin is useless here** (0.700 false abstention): the top two chunks are
   usually close in a long contract regardless of whether the answer exists.
3. **The classifier lost to a plain threshold** (0.557) with 34 negatives; it
   overfits and is not shipped.
4. **Category normalisation is nominally best and statistically indistinguishable**
   from the global threshold.
5. **The shipped point is conservative on purpose**: threshold 0.9018 on cosine,
   giving 8.6% false abstention and 76.5% correct abstention on train. On test the
   false-abstention rate was 14.8%, so the threshold does not transfer perfectly
   across splits — stated rather than averaged away.
6. **Type 3 hard negatives remain the hard case** (0.70); types 1, 2 and 4 are
   near-saturated.
7. **Thresholds do not transfer to naturally-phrased questions.** Benchmark queries
   name the contract, which raises the top score; a bare clause question scores
   ~0.03–0.05 lower and is wrongly abstained by the same threshold. `configs/final.yaml`
   is the evaluated configuration; `configs/serve.yaml` lowers the threshold for the
   demo. This is a benchmark-style dependence, not a model failure.

**Caveat:** A first classifier looked near-perfect (false abstention 0.000) because
`query_len` separated the classes by construction: the authored negatives are 51–151
characters against benchmark queries of 134–243. It was learning who wrote the
query. The feature is excluded from `FEATURE_NAMES` and a regression test enforces
it.

### Test-split result (final reporting)

`python -m eval.run_abstention --config configs/final.yaml --split test
--i-know-this-is-final` — the fixed `0.9018` threshold from `configs/final.yaml`
applied to 54 answerable test queries and 12 test negatives, on the same
21-document test index used by the retrieval evaluation. No threshold is selected
on test.

| direction | rate | 95% CI | n |
|---|---|---|---|
| false abstention | 0.148 | [0.056, 0.241] | 54 |
| correct abstention | 0.917 | [0.750, 1.000] | 12 |

Per-type correct abstention: type 1 1.00 (n=1), type 2 1.00 (n=5), type 3 0.75
(n=4), type 4 1.00 (n=2). The interval on 12 negatives is wide; this is an
estimate, and the test composition differs from the 34 train negatives.

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
