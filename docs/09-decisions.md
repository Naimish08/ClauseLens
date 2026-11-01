# 09 — Decision log

Architecture decision records. Newest last. A decision that changes behaviour gets
an entry here in the same commit as the code.

Format: **Context** (the forces), **Decision**, **Consequences** (what becomes
easy, what becomes hard), **Alternatives rejected**.

---

## ADR-001 — Use Python 3.12 via `uv`, not the system Python 3.14

*Status: accepted · 2026-11-01*

**Context.** The machine's default interpreter is CPython 3.14.4. The project
needs `torch`, `sentence-transformers`, and (potentially) `faiss`. Wheel
availability for 3.14 is thin, and source builds of `torch` take tens of minutes
and can fail against CUDA 12.x headers.

**Decision.** Create the environment with `uv venv --python 3.12`
(3.12.15 is already installed) and pin `requires-python = ">=3.12,<3.13"`.

**Consequences.** Wheels resolve immediately; the environment is reproducible on
another machine via `uv`. Cost: the project cannot use 3.14-only syntax, which is
irrelevant here.

**Alternatives rejected.** System 3.14 (wheel risk); Conda (slower, heavier, no
benefit at this size).

---

## ADR-002 — Build an experiment platform, not a pipeline

*Status: accepted · 2026-11-01*

**Context.** The deliverable must compare chunkers, retrievers and abstention
policies, and must run a self-improvement loop that rewrites pipeline choices.
Hard-coded stages make every comparison a separate script, and separate scripts
drift.

**Decision.** Every stage is a protocol selected by YAML config. `pipeline.py` is
the only place that composes them. Evaluation lives outside the pipeline and only
ever receives `search(query: str)`.

**Consequences.** New stages cost three files (implementation, registry entry,
config). Comparisons are apples-to-apples by construction. The loop can propose
config diffs instead of patches. Cost: slightly more indirection than a script.

**Alternatives rejected.** Notebook-driven experiments (unreproducible, no shared
metric); one script per experiment (drift, no shared code path).

---

## ADR-003 — Evaluate the CUAD subset only

*Status: accepted · 2026-11-01*

**Context.** The downloaded archive contains CUAD, MAUD, PrivacyQA and
ContractNLI. The task specifies "CUAD-mini, 194 queries".

**Decision.** Load only `cuad.json`, capped at 194 queries via the reference
selection. The other benchmarks remain downloaded and unindexed.

**Consequences.** A single, coherent corpus; latency and index size stay small;
the report describes exactly one benchmark. Cost: no cross-domain generalisation
claim can be made — which is fine, because none is attempted.

**Alternatives rejected.** All four benchmarks (4× corpus, 4 benchmarks × 194
queries, and PrivacyQA policies are a different genre — dilutes the contract
focus within the time budget).

---

## ADR-004 — Read corpus files in `raw` mode, verify against `text` mode

*Status: accepted · 2026-11-01*

**Context.** The dataset card says to read "in Python's default text mode with UTF-8
so character offsets line up". Default text mode performs universal-newline
translation: `\r\n` → `\n`, which shifts every offset after the first line
ending. The upstream reference implementation uses plain `open()` (text mode), so
its published offsets were produced under that behaviour.

**Decision.** Read in `raw` mode (bytes → UTF-8 decode) for all indexing and
scoring. Run verification under **both** modes and report the result.

**Consequences.** Measured: all 272 snippets match in both modes and no corpus
file contains CRLF, so the modes coincide today. `raw` is still the safer default
because it cannot drift with platform settings, and the dual-mode check is a
regression test for any future corpus.

**Alternatives rejected.** Text mode only (silently wrong on CRLF input);
byte offsets (the benchmark JSON stores character offsets, and CUAD text is
ASCII-safe, so byte and character offsets coincide — but relying on that is an
unstated assumption).

---

## ADR-005 — Metric default follows the reference implementation, not the union variant

*Status: accepted · 2026-11-01*

**Context.** An early design note specified the **union** of retrieved spans. The
upstream `QAResult` code **sums** per-snippet lengths. The definitions agree for
disjoint retrieved spans and diverge for overlapping ones — which is exactly what
`overlap > 0` chunking produces.

**Decision.** Implement both. Default `mode="paper"` (sum-based) for comparability
with published numbers; expose `mode="union"`. Every table states which was used.
The chunking grid reports both wherever overlap > 0.

**Consequences.** Numbers are comparable to the paper, and the discrepancy is
quantified rather than hidden. Cost: one extra parameter and one extra regression
test.

**Alternatives rejected.** Union only (not comparable to the paper); paper only
(the union definition goes untested).

---

## ADR-006 — Abstention scoring is asymmetric

*Status: accepted · 2026-11-01*

**Context.** If a system abstains on an answerable query, should that query's
precision be 0, or excluded?

**Decision.** Excluded from the precision mean; counted in
`false_abstention_rate`; recall counted as 0.

**Consequences.** Abstaining is penalised once (through the abstention metric and
through recall) rather than three times. `abstention_f1` becomes the headline
abstention number, and precision remains comparable across systems that abstain at
different rates.

**Alternatives rejected.** Precision 0 for abstentions (double-counts the
penalty, and makes precision incomparable between an aggressive and a
conservative system).

---

## ADR-007 — Implement BM25 in-house

*Status: accepted · 2026-11-01*

**Context.** `rank_bm25` exists and is small, but scores cannot be mapped to
character spans and its tokeniser is fixed.

**Decision.** Implement Okapi BM25 (~50 lines) with configurable `k1`, `b`, and a
tokeniser tuned for legal text (stop-words kept).

**Consequences.** Full control of the span mapping; fewer dependencies; the
baseline is explainable line-by-line in a design review. Cost: the implementation
owns its bugs — mitigated by hand-computed score tests.

**Alternatives rejected.** `rank_bm25` (span mapping); `pyserini`/Lucene (JVM,
overkill, slower to iterate).

---

## ADR-008 — No LLM anywhere in the committed system

*Status: accepted · 2026-11-01 (scope cut)*

**Context.** The project was time-boxed, and the scope was fixed before
implementation. LLM components (query rewriting, LLM reranking, LLM proposal in the
loop) each add integration, caching, and evaluation cost.

**Decision.** Cut all LLM use. Document it as a limitation and as future work.
The loop's proposer is a deterministic rule table.

**Consequences.** The deliverable is complete and honest; every component is
explainable and can be modified in place during a design review; results are
reproducible offline. Cost: the report cannot claim LLM-related gains, and must
state so explicitly rather than omitting the topic.

**Alternatives rejected.** LLM reranker only (still needs the same eval and
caching infrastructure as everything else, and competes with the reranker slot);
LLM proposer in the loop (the rule table is more auditable, which the evaluation
rewards).

---

## ADR-009 — Clause categories come from the query template

*Status: accepted · 2026-11-01*

**Context.** The task asks for a per-CUAD-category breakdown. The downloaded
benchmark JSON has no category field. CUAD's official 41 categories are not
recoverable from the query text directly.

**Decision.** Use the text after the first `;` as the category. Measured: 32
distinct values, matching the clause question that generated each query.

**Consequences.** A per-category breakdown is possible with zero extra data.
Cost: it is *not* CUAD's official taxonomy, so the report must label it "clause
question type (32 categories)" and never claim official CUAD categories. Low
support (some categories have n=1) is reported with `n` and grouped below n=5.

**Alternatives rejected.** Re-deriving CUAD categories from the original CUAD
release (a data-join with uncertain version alignment, and the mini subset would
still have low support); mapping 32 questions to 41 categories by hand (a
subjective mapping that would need defending, for no analytic gain).

---

## ADR-010 — The document router is an ablation, not a default

*Status: accepted · 2026-11-01*

**Context.** All 194 queries begin with `"Consider the <contract>"`, and the 29
distinct prefixes map 1:1 to the 29 documents. Enabling prefix-based routing would
inflate every number by exploiting a benchmark artefact.

**Decision.** The default pipeline searches the whole corpus. Routing is measured
in three conditions — none, prefix, semantic — and reported as an ablation. The
report states that `prefix` routing is not deployable as-is.

**Consequences.** Headline numbers are defensible; the lever is still quantified
(a reviewer will ask, and the answer is a table). Cost: the best deployed
configuration looks weaker than the best possible one.

**Alternatives rejected.** Router always on (benchmark-gaming); router never on
(misses the single largest measured effect and a genuine deployment feature).

---

## ADR-011 — Documentation is a directory of topic files plus `AGENTS.md`

*Status: accepted · 2026-11-01*

**Context.** The engineering reasoning needs to be recorded, not just the code.

**Decision.** `docs/` holds one file per topic (architecture, data, metrics,
chunking, retrieval, abstention, loop, evaluation, decisions, log schema,
reproducibility, risks, glossary), indexed by `docs/README.md`. `AGENTS.md` at the
root states the contributor conventions.

**Consequences.** Changes are reviewable by topic; a reviewer can jump straight to
the relevant design. Cost: prose must be maintained alongside code, which is why
the documentation rules in `docs/README.md` require an ADR entry in the same
commit.

**Alternatives rejected.** One giant `DESIGN.md` (becomes stale in sections, hard
to review); no docs (the reasoning is part of the deliverable).

---

## ADR-012 — Data is downloaded, never committed

*Status: accepted · 2026-11-01*

**Context.** The archive is 90.6 MB; the corpus is 79 MB. Committing it bloats the
repository and duplicates upstream distribution.

**Decision.** `data/*` is gitignored except `data/unanswerable.jsonl`. The
download script records source URL, sha256, size and per-benchmark file counts in
`data/raw/MANIFEST.json`.

**Consequences.** Clone-and-run works with one command; the exact data version is
verifiable by hash. Cost: `make data` requires network on first run, and if
upstream ever changes the archive, the hash check makes the drift visible instead
of silent.

**Alternatives rejected.** Vendoring the 29 CUAD documents (partial corpus is
still copyrighted material with licence ambiguity); an HF dataset mirror
(none official exists, and a mirror can disappear).


---

## ADR-013 — Chunk sizes are specified in characters

*Status: accepted · 2026-11-01*

**Context.** The metric is character-based; the three embedders use different
tokenisers.

**Decision.** Config expresses `size` in characters; token counts are computed per
model only as a context-limit guard.

**Consequences.** The precision-vs-size plot is directly interpretable, and the
grid is a controlled comparison across models. Cost: `512` characters is not
`512` tokens, so the numbers are not directly comparable to papers that report
token chunk sizes — stated in the report.

**Alternatives rejected.** Token sizes (not comparable across the three models;
harder to reason about precision from the metric's own units).

---

## ADR-014 — `search(query, k)`; `top_k` is a serving default, not an evaluation cap

*Status: accepted · 2026-11-01*

**Context.** The first baseline run produced identical metrics at `k=10` and
`k=16`. Cause: the pipeline truncated results at `retrieval.top_k`, so evaluating
at a larger `k` silently re-measured the same ten spans.

**Decision.** `Pipeline.search(query, k=None)` returns at most `k` spans, defaulting
to `config.top_k`. Evaluation passes `max(cutoffs)` and scores prefixes of that one
ranking.

**Consequences.** Every reported `k` is measured on a real top-`k` list, and each
query costs one retrieval call instead of one per `k`. Serving still defaults to
`top_k`, so the app and CLI do not start dumping 16 passages because the report
measures recall@16.

**Alternatives rejected.** Setting `top_k` to the largest reported `k` (couples
serving behaviour to reporting choices); a separate evaluation-only search path
(two rankings, two behaviours to trust).

---

---

## ADR-016 — Reranking is implemented as a retriever wrapper, not a pipeline stage

*Status: accepted · 2026-11-01*

**Context.** A cross-encoder needs the chunk *text*, but `RetrievedSpan` carries only
offsets. Adding text to every span would bloat the hot path; adding a parallel
text-carrying pipeline stage would duplicate state.

**Decision.** `src/retrieval/rerank.py` implements `RerankRetriever(base, candidates,
model)`. It indexes the chunk text once, fetches `candidates` from the base
retriever, and reorders. Config expresses it as a retriever:
`retrieval.name: rerank` with a nested `base` spec.

**Consequences.** The pipeline interface is unchanged, so a reranker cannot change
how spans are produced — only their order. "Reranker adds N ms" is measurable
because the wrapper is timed like any retriever. Cost: `docs/01-architecture.md`'s
separate reranking stage is now unused, and the two-stage story is expressed as
composition rather than as a pipeline slot.

**Alternatives rejected.** Putting `text` on `RetrievedSpan` (memory and copies on
every query); a pipeline stage with a separate text index (duplicate state, two
things to keep in sync).

---

## ADR-017 — Guardrails are relative, and the loop has a text budget

*Status: accepted · 2026-11-01*

**Context.** The first full loop run accepted a chunk-size doubling from 1024 to
2048 characters: recall@10 rose from 0.315 to 0.511, mean retrieved text per query
rose from 9,809 to 19,568 characters, and the *absolute* precision guardrail
(no drop greater than 0.02) passed trivially because precision was 0.011 to begin
with. Left alone, the loop would learn to win recall by returning progressively
more of the document — the metric is telling the truth, but the optimisation is
not useful.

**Decision.** Three changes to the acceptance gate:

1. The precision guardrail is **relative**: precision may fall by at most 30% of
   its current value, not by an absolute amount.
2. A **text budget**: mean retrieved characters per answerable query may not exceed
   2.5× the *baseline's* mean. Measured against the baseline, not the previous
   iteration, so growth cannot compound.
3. Chunk-size and overlap changes are decoupled: ``size_up`` no longer silently
   turns on 15% overlap. One iteration changes one thing, so the effect is
   attributable.

**Consequences.** The loop now rejects the 1024→2048 step and settles at
fixed-1024, recall@10 0.315, precision 0.017, ~9.8k characters returned per query.
The rejected iteration stays in the log with the guardrail reason, which is itself
evidence that the gate has teeth. Cost: two more constants (0.30, 2.5) that must be
justified in the report rather than hidden.

**Alternatives rejected.** An absolute precision tolerance (meaningless near zero);
capping `k` instead of characters (a chunk-size increase defeats it); no cap and
report the trade-off (honest, but then the loop's "best" config returns whole
documents).

---

## ADR-018 — Latency is gated by an absolute budget, not only a ratio

*Status: accepted · 2026-11-01*

**Context.** The first loop run reached a hybrid BM25+dense retriever that raised
recall@10 from 0.319 to 0.409 and was rejected solely because p95 latency went from
2.9 ms to 42 ms — 14× the baseline, but far below human perception. A 1.5× ratio
applied to a millisecond-scale baseline makes any GPU component unacceptable on
principle, which is not the question the project is asking.

**Decision.** The latency gate passes when p95 is within
``max(250 ms, 1.5 × current p95)``. The relative term still catches regressions
from an already-slow configuration; the absolute budget sets the floor of what is
worth arguing about.

**Consequences.** Hybrid retrieval is accepted (42 ms is interactive) and the
cross-encoder reranker is still rejected (975 ms p95), which is the intended
distinction: a 40 ms gain in recall is free, a one-second one is not. The rejected
reranker stays in the log with its measured latency. Cost: one more constant to
justify, and a reader may reasonably ask why 250 ms — the report states that it is
a deliberate "still feels instant" threshold for a local interactive tool, not a
measurement.

**Alternatives rejected.** Pure ratio (rejects all GPU work); no latency gate and
report the trade-off (the loop would happily ship a one-second reranker); 1000 ms
budget (would accept the reranker without a justification argument).

---

## ADR-019 — The loop's rule table gained a dense-retriever step, and the hybrid builder was fixed

*Status: accepted · 2026-11-01*

**Context.** The retrieval grid (docs/05-retrieval.md) measured dense retrieval well
ahead of both BM25 and the BM25+dense hybrid: 0.580 recall@10 for gte-base against
0.475 for RRF fusion and 0.343 for BM25. The loop, whose rules tried
``retriever_hybrid`` but never a dense retriever on its own, therefore settled at
0.409 — a local optimum that the broader comparison shows is ~0.17 below what the
same train data supports.

**Decision.** Two changes:

1. ``retriever_dense`` is added to the rule table *before* ``retriever_hybrid``,
   using gte-base (best quality at the lowest dense latency in the grid).
2. ``_retriever_hybrid`` now fuses whichever family is missing with the current
   retriever. Previously, applying it after a dense switch would have built a
   hybrid of dense with itself.

**Consequences.** The loop now reaches dense retrieval and the final config matches
the best independently validated choice. The rule table was revised *after* seeing a
train-only comparison, which is disclosed in the report: this is design informed by
train evidence, not test leakage, and the test split remains untouched. Cost: the
loop is no longer a rule set fixed before any measurement, and the report must say
so rather than presenting hindsight as foresight.

**Alternatives rejected.** Keep the local optimum and document it (honest, but ships
a knowingly worse system and makes the loop look like a dead end); delete the hybrid
rule (loses the comparison that produced the finding).

---

## ADR-020 — Accepting a change re-opens previously rejected rules

*Status: accepted · 2026-11-01*

**Context.** With ADR-019 the loop reached dense retrieval (recall@10 0.496 from
0.319) and then stopped: the sentence chunker had been rejected at iteration 4
while the retriever was still BM25, and the loop never reconsidered it. The
retrieval grid shows sentence-1024-15% with gte-base at 0.580, well above the
loop's 0.496. Rejections are conditional on the system they were measured against,
and a later accepted change invalidates that condition.

**Decision.** On acceptance, clear the set of tried rules. Exact repeats remain
impossible because every proposed behaviour is hashed into ``tried_signatures``.
Rules whose builder now returns nothing (size cap reached, already the current
chunker) are skipped.

**Consequences.** The loop can re-test chunkers, overlap and fusion after the
retriever changes, which is where interaction effects live. It may spend iterations
re-confirming rejections; the log shows that, and the iteration cap bounds it.
Cost: the loop is less obviously monotone, so the iteration table matters more than
before.

**Alternatives rejected.** Never re-open (the local optimum above); re-open only
chunking rules (an arbitrary exception that hides the same problem elsewhere).
