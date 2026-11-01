# 04 — Chunking

Status: 🚧 `fixed` implemented and span-validated; `recursive`, `sentence`,
`clause`, parent–child and contextual enrichment pending.

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
| `clause` | Regex section detection (`^\s*(?:\d+(?:\.\d+)*|[A-Z][A-Z ]{3,})\s*[.):]?\s`), one chunk per numbered section/heading, with a size cap that splits oversized sections | `min_size`, `max_size`, `keep_heading` | Best structural match to how contracts actually answer questions. |

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
axis or panel), per chunker, at a fixed retriever and `k`. It answers: how much
recall are you buying per point of precision, and at what size does the curve
turn?

Expected shape (to be confirmed, not asserted): precision decreases monotonically
with size; recall rises then plateaus once chunks are larger than the answer plus
its context; the clause chunker dominates the frontier if CUAD answers align with
sections.

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
