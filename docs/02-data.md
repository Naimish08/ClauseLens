# 02 — Data

Status: ✅ implemented (`src/data.py`, `scripts/download_data.py`,
`scripts/verify_data.py`, `tests/test_data.py`).

All numbers in this document are **measured on the downloaded data** on
2026-11-01. Reproduce with `make data && make verify`.

## Provenance

| Item | Value |
|---|---|
| Benchmark | LegalBench-RAG (Pipitone & Houir Alami, 2024), CUAD subset |
| Paper | <https://arxiv.org/abs/2408.10343> |
| Upstream code | <https://github.com/zeroentropy-ai/legalbenchrag> |
| Data hosting | Dropbox folder linked from the upstream README (not on PyPI/HF) |
| Archive size | 90,591,976 bytes |
| Archive sha256 | `27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d` |
| Manifest | `data/raw/MANIFEST.json` (written by the download script) |

The archive contains **four** benchmarks (`cuad`, `maud`, `privacy_qa`,
`contractnli`) and 714 corpus files. Only CUAD is evaluated in this project; the
others are downloaded because they arrive in the same archive and are ignored.

## The reference selection, and why the plan's wording was wrong

The build plan described "the authors' `mini()` function". **No such function
exists upstream.** The actual logic lives inline in
`legalbenchrag/benchmark.py`:

```python
MAX_TESTS_PER_BENCHMARK = 194
SORT_BY_DOCUMENT = True

if len(tests) > MAX_TESTS_PER_BENCHMARK:
    if SORT_BY_DOCUMENT:
        tests = sorted(
            tests,
            key=lambda test: (
                random.seed(test.snippets[0].file_path),
                random.random(),
            )[1],
        )
    tests = tests[:MAX_TESTS_PER_BENCHMARK]
```

Three details matter and are easy to get wrong:

1. **It is per benchmark, not global.** CUAD is capped at 194 of its own 4042
   queries. The phrase "194 queries" refers to CUAD, not to the whole benchmark.
2. **The seeding is unusual.** `random.seed(path)` is called *inside the key
   function*, so each test re-seeds the global RNG from its first snippet's file
   path, then draws one float. The result is deterministic only because
   `sorted()` calls the key once per element in input order. A "cleaner"
   reimplementation (seed once, shuffle) produces a **different** subset.
3. **The cap is a document-clustering device.** Sorting by a path-derived random
   key groups queries onto few documents, so ingestion cost is small. That is why
   194 queries touch only 29 contracts.

`src/data.py::select_mini` reproduces this exactly, and
`tests/test_data.py::test_select_mini_is_a_prefix_of_the_reference_ordering`
re-derives the reference ordering verbatim.

**Consequence:** the mini set is *not* a uniform random sample of CUAD. Any claim
of "representative of CUAD" would be wrong. The report must call it "the
reference CUAD-mini subset".

## Splits

`assign_split` is specified as: SHA-256 of the stripped query text, hex digest
modulo 10, `< 3` → test, otherwise train.

```python
digest = hashlib.sha256(query_text.strip().encode("utf-8")).hexdigest()
return "test" if int(digest, 16) % SPLIT_MODULUS < TEST_BUCKETS else "train"
```

Measured on the 194 mini queries:

| Split | Queries |
|---|---|
| train | 140 |
| test | 54 |

The plan estimated ~135/59; the real split is 140/54 (72% / 28%). Either way,
"about 135 train queries" is close enough for the noise argument in
`07-improvement-loop.md`.

Properties to preserve:

- The hash is over the **stripped** text, so re-wrapping a query does not move
  it between splits.
- The split is a pure function of query text. It does not depend on order, seed,
  or the corpus. It must never be changed, because reported test numbers are
  measured against it.
- Sub-splits *within* train are allowed (cross-validation folds); re-splitting
  the train/test boundary is not.

## Corpus

| Item | Value |
|---|---|
| Documents referenced by the 194 queries | **29** |
| Ground-truth snippets | **272** |
| Queries with 1 / 2 / 3 / 4 / 5 / 6 / 7 snippets | 168 / 37 / 10 / 2 / 1 / 1 / 1 |
| GT span length: min / median / max / mean | 67 / 317 / 1756 / 394.4 chars |
| Documents containing CRLF | **0** |

Only the 29 referenced documents are loaded: `load_mini_cuad()` derives the path
set from the selected queries. Indexing all 462 CUAD contracts would change
nothing except latency — and would silently turn "document routing" into a much
harder problem than the benchmark intends.

⚠️ **The median answer is 317 characters.** This single fact drives the whole
chunking study: character precision is `overlap / retrieved_length`, so returning
one 2,000-character chunk around a correct 317-character answer scores roughly
16% precision even though the answer was found. See `03-metrics.md`.

## Offset verification findings

`make verify` checks, for both read modes, that
`document[start:end] == snippet.answer`, that spans are in bounds, that spans
within a query are disjoint, that no referenced document is missing, and that no
document contains CRLF.

Measured result on the 194 selected queries:

```
mode=raw  queries=194 snippets=272 docs=29 mismatches=0 out_of_bounds=0 crlf_docs=0 overlapping_gt_queries=0
mode=text queries=194 snippets=272 docs=29 mismatches=0 out_of_bounds=0 crlf_docs=0 overlapping_gt_queries=0
```

**Conclusion:** the published offsets are trustworthy, and because no corpus file
contains `\r\n`, the two decoding modes coincide on this corpus. The project
therefore reads in `raw` mode (bytes → UTF-8 decode), which cannot drift with
platform newline translation, and keeps the `text` mode in the verifier as a
regression check.

⚠️ **Trap avoided:** the plan said to read "in Python's default text mode with
UTF-8 so character offsets line up". Default text mode performs universal-newline
translation, so on a CRLF file every offset after the first line ending shifts.
It happens not to matter for this corpus, but the reasoning was wrong. See
`09-decisions.md` ADR-004.

⚠️ **Trap avoided:** the benchmark JSON **does** contain an `answer` field, even
though the reference code exposes it as a Pydantic `computed_field`. Offsets could
therefore be verified without the corpus. We verify against freshly read corpus
text instead, which also validates the download.

## Query structure

All 194 mini queries share one template:

```
Consider the <contract, parties>; <clause question>
```

Measured properties:

| Property | Value |
|---|---|
| Queries containing `;` | 194 / 194 |
| Distinct `<contract>` prefixes | **29** |
| Distinct `<clause question>` suffixes | **32** |
| Prefix → document mapping | **1:1** (every prefix identifies exactly one document) |
| Max queries per document | 18 |

Two consequences, both exploited later:

1. **The prefix is a document label in plain text.** A trivial router gets
   document identification nearly for free. Any comparison that ignores the
   prefix (baseline BM25 over the whole corpus) is *artificially* weak. The fix is
   to report both: with and without the prefix (ablation, `05-retrieval.md`).
2. **The suffix is a clause-type taxonomy.** The 32 distinct question strings are
   the category labels used for the per-category breakdown. They are *not*
   CUAD's official 41 categories: the mini subset does not cover all of them, and
   several have very low support (the largest has 24 queries, several have 1).
   Category-level numbers for low-support categories are reported with their `n`
   and never presented as reliable rankings.

`derive_category()` and `derive_doc_hint()` implement both extractions and are
unit-tested, including the no-semicolon fallback.

## Directory layout after `make data`

```
data/
  raw/legalbenchrag.zip      archived download (gitignored)
  raw/MANIFEST.json          url, hash, size, counts, timestamp
  benchmarks/*.json          all four benchmarks (only cuad.json is used)
  corpus/cuad/*.txt          462 files (only 29 are loaded)
  corpus/{maud,privacy_qa,contractnli}/
  unanswerable.jsonl         COMMITTED — the frozen negative set
```

`.gitignore` excludes `data/*` except `data/unanswerable.jsonl`, so the corpus
never enters git but the frozen negative set always does.

## Known traps

1. **Do not "fix" a span mismatch by editing spans.** Report it. A mismatch means
   the corpus or the dataset version differs, and silently repairing it
   invalidates every comparison against published numbers.
2. **Do not index all 462 CUAD contracts.** It changes the task and slows every
   experiment.
3. **Do not re-derive the mini subset with a different sort.** It changes which
   194 queries exist, and therefore every number.
4. **Do not shuffle the query list and reuse a cached index built from a different
   order.** Chunk IDs must be stable; the retriever must not depend on ingestion
   order.
