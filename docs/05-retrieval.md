# 05 — Retrieval

Status: 🚧 BM25, dense, hybrid and reranker implemented; measurement pending and
the document router not yet built.

## The document-routing lever

`02-data.md` measured that every one of the 194 queries starts with
`"Consider the <contract, parties>"` and that the 29 distinct prefixes map **1:1**
to the 29 corpus documents.

That means document identification is available in plain text, before any
retrieval. Two ways this matters:

1. **The honest baseline must not use it.** BM25/dense over all 29 documents is
   the real baseline; a router is an *additional* stage whose contribution is
   measured as an ablation, not baked into every number.
2. **The router is nearly free and should be attempted.** Restricting the search
   space to one document removes 28/29 of the distractors, which mostly improves
   precision (no cross-document near-duplicate clauses).

Three routing conditions are compared:

| Router | Input | Expected |
|---|---|---|
| `none` | search all 29 documents | baseline |
| `prefix` | string match of the `Consider the ...` hint against document metadata | near-perfect document ID; isolates whether the benchmark is "too easy" with the hint |
| `semantic` | embed the prefix, cosine against document embeddings | realistic deployment behaviour where no hint exists |

Because the hint is a benchmark artefact, the report must state clearly that
`prefix` routing is not deployable as-is, and that `semantic` routing is the
number to trust. This is exactly the kind of honesty the assignment is looking
for.

Optional second lever, measured as a query-preprocessing ablation: strip the
`"Consider the ...;"` prefix entirely and retrieve on the clause question alone.

### Measured baseline observations (train, 2026-11-01)

Fixed-512 chunks, BM25, 140 train queries, 29 documents, 2396 chunks:

| Query text | recall@10 | recall@50 | recall@100 |
|---|---|---|---|
| full query (prefix kept) | **0.101** | **0.375** | **0.465** |
| prefix stripped | 0.083 | 0.192 | 0.259 |

- `doc_hit@1 = 1.000`: the prefix alone identifies the correct document for every
  query, so document routing is not the bottleneck.
- Stripping the prefix makes retrieval clearly **worse**, which confirms the
  prefix is doing routing work rather than adding noise. The clause question on
  its own is too generic ("What is the expiration date of this contract?") to
  discriminate.
- The bottleneck is *within-document ranking*: the first overlapping chunk sits at
  rank 57–743, because the prefix's party names match the contract's title and
  header chunks, which then outrank the clause itself.

This is the measured case for the two-stage design: use the prefix (or a semantic
router) to pick the document, then rank clauses inside it.

## BM25

Implemented in-house (`src/retrieval/bm25.py`) rather than pulled from a package,
for three reasons:

1. We need scores mapped back to the exact chunk spans, so we control the
   index anyway.
2. `rank_bm25` has no notion of character offsets and its tokeniser is fixed.
3. BM25 is ~50 lines and must be explainable in the debrief.

Details:

- Okapi BM25, `k1 = 1.5`, `b = 0.75` (tunable in config).
- Tokenisation: lowercase, split on non-alphanumeric, drop tokens of length 1,
  keep stop-words (legal text is dense in "shall", "notwithstanding", "hereof" —
  removing them hurts).
- IDF uses the standard `log((N - df + 0.5)/(df + 0.5) + 1)`.
- Chunk text is scored; ties broken by `(file_path, start)`.
- Legal-specific query expansion is *not* done (no stemming, no synonyms) so the
  baseline stays honest.

Expected: strong. BM25 is famously competitive on legal text, and the report
should treat "dense must beat BM25 by a clear margin to justify its cost" as the
bar.

## Dense retrieval

Three embedders, chosen for a size spread rather than a leaderboard:

| Model | Params | VRAM (fp16) | Role |
|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | 109M | ~0.5 GB | base quality reference |
| `thenlper/gte-base` | 109M | ~0.5 GB | second base, different training mix |
| `intfloat/e5-large-v2` | 335M | ~1.3 GB | "does size matter" upper point |

All three fit comfortably on an 8 GB RTX 4060, and the 29-document corpus
(~3–6k chunks) indexes in minutes at most. Embeddings are cached to
`artifacts/embeddings/{model}/{config_hash}.npz` so the grid does not re-embed.

Notes:

- **Prefix conventions differ** and must be respected: `e5` requires
  `"query: "` / `"passage: "`, `bge` requires a query instruction
  (`"Represent this sentence for searching relevant passages: "`) for the
  retrieval task, `gte` needs none. Getting this wrong silently costs several
  points and would be a confusing "result".
- Normalise vectors; use cosine via inner product.
- Batch size chosen to fit 8 GB; no gradient, `torch.inference_mode()`.
- Exact search (numpy `matmul`) is used, not FAISS: 6k × 768 floats is ~18 MB, so
  approximate search would be a needless dependency and an accuracy question.

## Hybrid retrieval

Reciprocal Rank Fusion of BM25 and the best dense model:

```
score(d) = Σ_r  w_r / (k_rrf + rank_r(d))          k_rrf = 60
```

Two variants:

1. `hybrid_rrf` — equal weights `w = 1.0`, standard `k_rrf = 60`.
2. `hybrid_weighted` — tuned `w_bm25 : w_dense` over a small grid (`{0.25, 0.5,
   0.75, 1.0}`), tuned **on train only**.

RRF is used instead of score normalisation because BM25 and cosine scores live on
incomparable scales and normalisation is an extra, fragile choice.

## Reranking

Cross-encoder on the top candidates of the *best* hybrid configuration (not a full
sweep — time budget).

| Model | Notes |
|---|---|
| `BAAI/bge-reranker-base` | primary; ~1 GB fp16, fast on 20–50 candidates |

- Candidates `N ∈ {20, 50}`, output `k` = the reported k.
- Rerank text is truncated to the model's max length; the returned span stays the
  original chunk span, never the truncated one.
- Latency is measured and reported, because a reranker that adds ~2 s per query
  must justify itself against the recall it buys.

## Comparison table the code must produce

Rows = retrieval configs; columns = `recall@10`, `precision@10`, `mrr@10`,
`doc_hit@10`, `span_hit@10`, `latency_ms`, `n`. Grouped into: BM25, dense (3),
hybrid (2), hybrid + rerank, and routing ablations.

Every row is generated by `eval/run_eval.py` from a YAML config — no
hand-assembled numbers.

## Tests

1. BM25: exact expected scores on a 3-document fixture with hand-computed IDF.
2. BM25: a term appearing in every document contributes ~0 IDF but never
   negative.
3. Index/search round-trip: the returned span is always exactly one of the
   indexed chunk spans.
4. Dense: query prefix is applied for `e5` and `bge` and absent for `gte`
   (assert on the constructed input string, no model download needed).
5. RRF: hand-computed fusion on a fixture with known ranks; ties broken
   deterministically.
6. Router: `prefix` routing on the real mini set identifies the correct document
   for ≥95% of queries; `semantic` routing is evaluated, not asserted.
