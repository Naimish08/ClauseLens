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
ContractNLI. The brief describes "CUAD-mini, 194 queries".

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

**Context.** The build plan said to read "in Python's default text mode with UTF-8
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

## ADR-005 — Metric default follows the reference implementation, not the plan's wording

*Status: accepted · 2026-11-01*

**Context.** The build plan says to take the **union** of retrieved spans. The
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
(the plan's definition goes untested).

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
baseline is explainable line-by-line in the debrief. Cost: we own the bugs —
mitigated by hand-computed score tests.

**Alternatives rejected.** `rank_bm25` (span mapping); `pyserini`/Lucene (JVM,
overkill, slower to iterate).

---

## ADR-008 — No LLM anywhere in the committed system

*Status: accepted · 2026-11-01 (scope cut)*

**Context.** ~40 hours of wall clock against a plan estimating ~30 hours of work.
LLM components (query rewriting, LLM reranking, LLM proposal in the loop) each add
integration, caching, and evaluation cost.

**Decision.** Cut all LLM use. Document it as a limitation and as future work.
The loop's proposer is a deterministic rule table.

**Consequences.** The deliverable is complete and honest; every component is
explainable and modifies-in-place during the debrief; results are reproducible
offline. Cost: the report cannot claim LLM-related gains, and must say so
explicitly rather than omitting the topic.

**Alternatives rejected.** LLM reranker only (still needs the same eval and
caching infrastructure as everything else, and competes with the reranker slot);
LLM proposer in the loop (the rule table is more auditable, which the brief
rewards).

---

## ADR-009 — Clause categories come from the query template

*Status: accepted · 2026-11-01*

**Context.** The brief asks for a per-CUAD-category breakdown. The downloaded
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

**Context.** The project brief and the owner both require the engineering
reasoning to be recorded, not just the code.

**Decision.** `docs/` holds one file per topic (architecture, data, metrics,
chunking, retrieval, abstention, loop, evaluation, decisions, log schema,
reproducibility, risks, glossary), indexed by `docs/README.md`. `AGENTS.md` at the
root states the conventions any contributor — human or AI — must follow.

**Consequences.** Changes are reviewable by topic; a reviewer can jump straight to
the relevant design. Cost: prose must be maintained alongside code, which is why
the documentation rules in `docs/README.md` require an ADR entry in the same
commit.

**Alternatives rejected.** One giant `DESIGN.md` (becomes stale in sections, hard
to review); no docs (the brief explicitly grades the reasoning).

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
