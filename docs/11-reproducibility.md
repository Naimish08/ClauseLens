# 11 — Reproducibility

Status: environment verified on 2026-11-01.

## Environment actually used

| Component | Version |
|---|---|
| OS | Linux 7.0.0-38-generic, glibc 2.43 |
| Python | 3.12.15 (managed by `uv`, not the system 3.14.4 — ADR-001) |
| `uv` | 0.13.3 |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.2 |
| GPU | NVIDIA RTX 4060 Laptop, 8188 MiB, driver 595.99.02 |

Optional stacks, installed only when a stage needs them:

```bash
uv pip install -e ".[ml]"    # torch, sentence-transformers, transformers
uv pip install -e ".[app]"   # streamlit
```

The dense stack was resolved to the versions installed by the pinned environment;
because dense scores can shift across major `torch`/`transformers` releases, the
embedding cache and per-query ranks are the reproducibility boundary rather than
the raw float values.

## From clean clone to first result

```bash
make setup      # uv venv --python 3.12 + core deps          (~20 s)
make data       # download + unpack LegalBench-RAG           (~11 s, 90.6 MB)
make verify     # offset integrity check                     (~0.1 s)
make test       # 251 tests                                  (~3 s)
make eval CONFIG=configs/baseline.yaml
```

No API keys and no accounts are required. BM25 configurations run offline after
`make data`; dense configurations download model weights on first use, then run
offline from the local cache.

## Determinism

| Source of nondeterminism | Handling |
|---|---|
| Split assignment | pure hash of the stripped query text; no seed involved |
| Mini-set selection | reproduces the reference reseeding exactly; verified by test |
| Fold assignment | derived from the query-text hash, so folds never move |
| Bootstrap resampling | explicit `seed` from the config |
| Tie-breaking in ranking | `(-score, file_path, start)` — total order, no reliance on dict order |
| Embedding backends | cached to disk keyed by `(model, chunk_config_hash)`; approximate GPU kernels can differ in the last float bits, which does not change ranks in practice but is noted |
| Python hash randomisation | never relied upon; no `set` iteration order affects output |

Determinism test: running the same config twice produces byte-identical
`per_query.jsonl` for BM25 configs. Dense configs are compared on ranks rather
than raw scores.

## Caching

```
artifacts/embeddings/{model_slug}/{chunk_config_hash}.npz   vectors + spans
artifacts/rerank/{config_hash}.jsonl                        reranker scores
artifacts/runs/{run_id}/                                    per-run outputs
```

Cache keys hash the *chunk configuration and corpus manifest hash*, so changing a
chunk size never silently reuses stale vectors. Clearing caches is always safe:
`make clean` removes `artifacts/`.

## Measured (2026-11-01)

Baseline config (`configs/baseline.yaml`, fixed-512, BM25, 140 train queries):

| Step | Measured |
|---|---|
| Parse + chunk + validate 29 documents | 2396 chunks from 1,211,773 chars |
| Index build (chunk + BM25) | included in the run below |
| Full eval run (chunk + index + 140 queries) | **0.74 s** wall clock |
| Full chunking grid (24 variants × 140 queries) | **22 s** wall clock |
| Abstention comparison, baseline (4 policies, 174 queries, 5 folds) | **1.9 s** wall clock |
| Abstention comparison, final dense config (4 policies, 174 queries, 5 folds) | **~20 s** wall clock |
| Dense index (gte-base, fixed-1024, 1203 train chunks) | **23.5 s** |
| Dense index (gte-base, sentence-1024, 1442 chunks; retrieval grid) | **45 s** |
| Dense query latency (gte-base) | median **10.8 ms**, p95 17.7 ms |
| Dense query latency (e5-large) | median 32.8 ms, p95 196 ms |
| Cross-encoder rerank (50 candidates) | median **1,073 ms** |
| Retrieval grid (7 configurations) | **~11 min** wall clock |
| Per-query latency (BM25) | median **3.5 ms**, p95 **4.6 ms** |
| Test suite | **251 tests, 3.0 s** |

Indexing dominates wall-clock time; query-time latency is milliseconds for BM25 and
tens of milliseconds for dense. Compute is therefore not the limiting cost of the
experiment — authoring the negatives and analysing failures are
(`12-risks-and-limitations.md`).

## Data version pinning

`data/raw/MANIFEST.json` records the archive URL, sha256
(`27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d`), size, and
per-benchmark file counts. If upstream replaces the archive, the hash changes and
the difference is visible before any number is produced.

## Reporting the environment

`REPORT.md` states the Python and library versions above, the GPU, and the total
wall-clock time of the final test run, so a reader can judge whether the claimed
latencies are plausible on their own hardware.
