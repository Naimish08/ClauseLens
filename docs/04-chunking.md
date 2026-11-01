# 04 — Chunking

Status: ✅ `fixed`, `recursive`, `sentence` and `clause` implemented, span-validated
and measured. Parent–child and contextual enrichment remain deferred.

Chunking is the single biggest lever on **character precision**, because
precision is `overlap / retrieved_length` and the median ground-truth answer is
**317 characters** (`02-data.md`). Returning one 2,000-character chunk around a
correct answer caps precision at ~16% no matter how good the retriever is.

## The span contract

Every chunker must satisfy, for every chunk it emits:

```python
document.content[chunk.start:chunk.end] == chunk.text
```

and chunks must be disjoint or explicitly flagged. Chunk IDs are
`f"{file_path}:{start}:{end}"`, so they are stable across runs and independent of
ingestion order.

Why this matters more than it sounds: the metric is defined on character spans of
the *original* document. If a chunker normalises whitespace, lowercases, or drops
headers, the chunk's own text no longer maps to a contiguous span and precision
becomes meaningless. All normalisation must therefore happen **inside** the
embedding representation, never in the text stored on the chunk.

## Units: characters, not tokens

Chunk sizes are specified in **characters**, with an approximate token count
recorded for the model's context limit. Reasons:

1. The metric is character-based; sizes in characters make the precision/size plot
   directly interpretable.
2. Tokenisers differ across the three embedders, so a token-based grid would not
   be a controlled comparison.
3. A character size can be validated against the span contract trivially.

The `max_tokens` guard is computed per model at index time and only warns; it
never truncates silently.

## Chunkers

| `name` | Behaviour | Key params | Hypothesis |
|---|---|---|---|
| `fixed` | Non-overlapping (or overlapping) fixed windows, snapped to the nearest whitespace within a tolerance | `size`, `overlap`, `snap` | Baseline. Best precision at small sizes, worst recall. |
| `recursive` | LangChain-style recursive split on `\n\n`, `\n`, `. `, ` ` until under `size` | `size`, `overlap` | Middle ground; no legal knowledge. |
| `sentence` | Groups whole sentences into windows of ~`size` chars | `size`, `max_sentences` | Never cuts mid-sentence; precision near fixed but cleaner boundaries. |
| `clause` | Regex section detection (numbered headings, `SECTION`/`ARTICLE`, all-caps title lines), one chunk per section, undersized sections coalesced and oversized ones split | `min_size`, `max_size` | Best structural match to how contracts actually answer questions. |

The clause chunker is written from scratch (not a library) because contract
formatting is adversarial: inconsistent numbering (`1.`, `1.1`, `(a)`, `Section
1.1`), all-caps headings, tables, and page headers injected mid-sentence.

Acceptance for the clause chunker: on the 29 CUAD-mini documents it must produce
chunks that (a) satisfy the span contract, (b) cover the document with <5%
uncovered characters, and (c) place at least 70% of the 272 ground-truth spans
entirely inside a single chunk. That third number is measurable *without* any
retrieval, and it is a cheap early signal — report it as "span containment".

## Overlap and the metric

`overlap > 0` is normally a recall aid, but it interacts with the metric: with
overlapping retrieved chunks, the reference metric double-counts duplicated
characters. See `03-metrics.md` and ADR-005. In the grid, `overlap` is therefore
`{0, 0.15}` only, and every overlapping configuration reports both the `paper`
and `union` metric so the effect is quantified rather than assumed.

## Grid

Sizes: `{256, 512, 1024}` characters; overlap `{0, 15%}`; chunkers
`{fixed, recursive, sentence, clause}`.

The centerpiece artefact is **precision vs chunk size** (and recall on a second
panel), per chunker, at a fixed retriever and `k`.

### Measured results (2026-11-01)

`make grid` runs all 24 variants (4 chunkers × 3 sizes × 2 overlaps) on the 140
train queries with BM25 in **22 seconds**. Full table:
`artifacts/grids/chunk_grid.csv`; figure: [`figures/chunk_grid.png`](figures/chunk_grid.png).

| chunker | size | overlap | chunks | mean len | containment | recall@10 | precision@10 | span_hit@10 |
|---|---|---|---|---|---|---|---|---|
| fixed | 1024 | 0% | 1203 | 1007 | 0.656 | **0.319** | 0.017 | 0.407 |
| fixed | 1024 | 15% | 1407 | 861 | 0.724 | 0.316 | 0.017 | 0.421 |
| recursive | 1024 | 15% | 1462 | 829 | 0.927 | 0.306 | 0.015 | 0.364 |
| sentence | 1024 | 15% | 1442 | 840 | 0.932 | **0.343** | 0.016 | 0.364 |
| clause | 1024 | 15% | 1759 | 689 | 0.927 | 0.256 | 0.015 | 0.300 |
| fixed | 512 | 0% | 2396 | 506 | 0.401 | 0.101 | 0.011 | 0.221 |
| recursive | 512 | 15% | 2913 | 416 | 0.755 | 0.097 | 0.009 | 0.207 |
| clause | 512 | 15% | 3104 | 390 | 0.755 | 0.099 | 0.009 | 0.179 |
| fixed | 256 | 0% | 4810 | 252 | 0.130 | 0.052 | 0.008 | 0.129 |
| recursive | 256 | 0% | 5444 | 223 | 0.354 | 0.040 | 0.008 | 0.093 |

Five findings, all measured:

1. **Recall is bounded by span containment, and size sets containment.**
   At 256 characters only 13–35% of ground-truth spans sit inside a single chunk,
   so recall@10 cannot exceed ~0.05. At 1024 characters containment reaches
   0.66–0.93 and recall jumps to 0.18–0.34. Any comparison of chunkers at a
   different size is really a comparison of size.
2. **Sentence-1024 with overlap is the best single configuration** (recall@10
   0.343), narrowly ahead of fixed-1024 (0.319) and recursive (0.306).
3. **Clause-aware chunking does not dominate.** It is respectable at 1024
   (0.256) but below fixed and sentence at equal size. Contracts' headings are a
   poor proxy for where CUAD answers live: answers often sit inside long sections
   whose heading is pages away, and dot-leader tables of contents produce spurious
   boundaries. The hypothesis behind clause-aware chunking was reasonable and is
   not supported by this data.
4. **Overlap helps at 1024 but hurts fixed-512** (0.101 → 0.074). With larger
   chunks a second window genuinely rescues straddling answers; with 512-char
   chunks it mostly injects near-duplicate competitors, and paper-mode scoring
   double-counts the shared characters (ADR-005). Overlap is not free.
5. **Precision stays at 0.007–0.017 for every chunker.** Chunk *size* sets
   precision; the chunker barely moves it. This is the metric behaving as
   designed and is the honest cost of returning whole chunks.

Even at the best containment, recall@10 is well below the containment ceiling,
which confirms `05-retrieval.md`: **ranking, not chunking, is the remaining
bottleneck** after size.

### A bug the grid exposed

The sentence and clause chunkers initially emitted ~10-character chunks (125,131
chunks at size 256) because contract tables of contents are full of dot leaders:
an oversized "sentence" was split, and the fragments were never re-packed. Both
now run `merge_spans` after `split_span`. Post-split packing is part of a chunker,
not an optimisation — without it, size parameters are decorative.

The failure was visible only because the grid reports `n_chunks` and mean chunk
length next to recall. A table of recall figures alone would have shown a
mysterious zero and no explanation.
## What is deliberately deferred

- **Parent–child retrieval** (retrieve small, return the enclosing section, or
  the reverse). It is the natural fix when span-hit is high but chunk-hit is high
  and precision is low. It is one of the highest-value next steps and is listed in
  `12-risks-and-limitations.md` as a documented omission, with the exact config
  change it would require.
- **Contextual enrichment** (prepend contract title, section heading, resolved
  definitions before embedding). Cheap to add inside `embed_text()` and likely
  helps dense retrieval on cross-referenced definitions; deferred only because it
  needs a definition resolver to be honest about claims.

## Tests

1. Span contract holds for every chunker on a synthetic document with CRLF, long
   lines, no trailing newline, and an empty document (must return `[]`, not
   crash).
2. No chunk exceeds `size` when `size` is specified as a hard cap; the clause
   chunker's `max_size` splits rather than truncates.
3. Coverage: concatenating all chunk spans (in order) covers ≥95% of the document.
4. Determinism: two runs produce identical chunk ID lists.
5. Overlap is exact: with `size=100, overlap=50`, adjacent chunks share exactly 50
   characters.
6. Sentence chunker never splits inside a sentence (assert on a fixture with
   abbreviations like "Inc." and numbered lists).
