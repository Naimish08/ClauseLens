# 13 — Glossary

Terms used precisely throughout the codebase and the report. Where a term has a
loose everyday meaning and a specific meaning here, the specific one wins.

**Abstention** — returning no passage and saying so, because nothing is relevant
enough. Distinct from "retrieval failed": abstention is a decision, failure is an
outcome.

**Answerable query** — one of the 194 CUAD-mini benchmark queries, which has at
least one ground-truth span.

**Unanswerable query** — a member of the authored negative set (`data/unanswerable.jsonl`),
with no ground-truth span anywhere in the indexed corpus.

**Assignment / split** — `assign_split(query)`: SHA-256 of the stripped query text,
mod 10, `< 3` → test. Fixed forever.

**Benchmark** — the CUAD subset of LegalBench-RAG, capped at 194 queries by the
reference selection. Sometimes called "CUAD-mini" or "the mini set".

**Chunk** — a contiguous range `[start, end)` of one corpus document, plus its
text. Invariant: `document.content[start:end] == chunk.text` (I3).

**Character precision@k** — total overlapping characters divided by total
retrieved characters, averaged over queries. Note `retrieved_length` is the sum of
returned chunk lengths, not the union (ADR-005).

**Character recall@k** — total overlapping characters divided by total
ground-truth characters, averaged over queries.

**Clause question type** — the text after the first `;` in a query. 32 distinct
values. *Not* an official CUAD category (ADR-009).

**Document hint / prefix** — the `"Consider the <contract>"` part of a query. Maps
1:1 to a corpus document in this benchmark (ADR-010).

**Corpus** — the 29 CUAD documents referenced by the mini queries. "The corpus"
never means all 462 downloaded CUAD files.

**CUAD** — *Contract Understanding Atticus Dataset*: 510 commercial contracts with
expert clause annotations, source of the queries and documents here.

**Hold-out / test set** — the 54 queries with `assign_split == "test"`. Evaluated
only for final reporting, with the count recorded in `logs/test_runs.log`.

**False abstention** — abstaining on an answerable query. Bad.

**Missed abstention** — returning a passage for an unanswerable query. Also bad,
and the failure mode of a system tuned only for recall.

**Fold** — one of five cross-validation partitions *within train*, derived from the
query-text hash. Used to make accept/reject decisions less fragile.

**Ground truth (GT)** — the annotated character span(s) for a query, from the
benchmark JSON. Never available to the pipeline at inference time.

**Hard negative** — an unanswerable query whose topic genuinely exists in the
corpus but not in the specified contract (negative type 3), or whose vocabulary
overlaps a real clause it is not asking about (type 4).

**Hybrid retrieval** — combining lexical and dense rankings, here with Reciprocal
Rank Fusion.

**Index** — the searchable representation of all chunks for one configuration.
Built by `Retriever.index()`; independent of any query.

**Latency** — per-query wall-clock time, excluding index build and one warm-up.

**Loop / improvement loop** — the one-command cycle: evaluate on train → diagnose →
propose → accept/reject → log.

**Macro-average** — mean of per-query metrics. The only averaging used for
headline numbers.

**MRR (mean reciprocal rank)** — mean of `1/rank` of the first retrieved span that
hits any ground truth. Zero if nothing hits.

**n** — the number of queries a reported statistic is computed over. Always printed
next to per-category numbers.

**Negative set** — the frozen collection of 40–50 unanswerable queries, with types
1–4.

**mini()** — *does not exist upstream.* The selection is inline logic in
`legalbenchrag/benchmark.py`; `src/data.py::select_mini` reproduces it. The name is
kept because the build plan used it.

**Pairwise/paired bootstrap** — resampling queries with replacement, with both
systems scored on the same resampled set, to get a confidence interval on the
*difference*.

**Partial span** — a failure where retrieval overlaps the ground truth but does not
contain all of it. Usually "chunk too small".

**Precision@k / Recall@k** — character-level (see above). Written with an explicit
`@k` whenever k matters.

**Reranker** — a cross-encoder that reorders a candidate list; consumes retrieval
output, never queries the index itself.

**Retrieved span** — a return value of `Retriever.search()`: `file_path`, `start`,
`end`, `score`, `chunk_id`. Always exactly a chunk span (I2).

**RRF (Reciprocal Rank Fusion)** — `Σ w_r / (k_rrf + rank_r)`; rank-based
combination that avoids score normalisation.

**Routed retrieval** — restricting search to a document chosen by a router, rather
than searching the whole corpus.

**Span hit vs chunk hit** — *span hit*: any overlap with GT. *Chunk hit*: a single
chunk fully contains a GT span. The pair diagnoses granularity problems.

**Spans must be disjoint** — within one query, ground-truth spans do not overlap
(asserted at load, mirroring the reference validator). Overlapping GT spans would
make the reference metric double-count.

**Test-run log** — `logs/test_runs.log`, an append-only record of every test
evaluation. The report states the count.

**Train vs test gap** — the difference between the final configuration's train and
test scores; a gap much larger than the bootstrap interval suggests overfitting to
train during the loop.

**Verification / `make verify`** — the offset integrity check: for all queries,
`document[start:end] == snippet.answer`, spans in bounds, spans disjoint,
documents present, no CRLF surprises.
