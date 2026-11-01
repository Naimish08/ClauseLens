# 01 — Architecture

## Shape of the system

```
                    ┌──────────────────────────┐
  configs/*.yaml ──▶│      src/pipeline.py     │──▶ PipelineResult
                    │  (the only orchestrator) │    spans + scores + meta
                    └───────────┬──────────────┘
                                │
       ┌────────────┬───────────┼────────────┬─────────────┐
       ▼            ▼           ▼            ▼             ▼
   chunking/   retrieval/   retrieval/    abstain.py    (metrics
   chunker     retriever    reranker      policy        computed by
       │            │           │            │          eval/, not
       ▼            ▼           ▼            ▼          by the pipeline)
  list[Chunk]  list[Span]  list[Span]   bool + reason
```

Evaluation is **outside** the pipeline. `eval/run_eval.py` calls the pipeline and
scores the returned spans. The pipeline never sees ground truth, which is what
makes the "no leakage" claim checkable by reading one file.

## Layers and responsibilities

### 1. Data layer — `src/data.py` ✅

Owns all filesystem access to the benchmark and corpus. Nothing else reads a
corpus file. Provides:

- `load_benchmark()` → `list[Query]`
- `select_mini()` → the reference 194-query subset
- `assign_split()` → `"train"` / `"test"`, hash-fixed
- `cv_folds()` / `fold_of()` → hash-fixed folds within train
- `load_corpus()` → `dict[file_path, Document]` in a chosen decoding mode
- `verify_offsets()` → `VerificationReport`

**Why one module:** every downstream number depends on spans being interpreted
against identical text. Duplicating file reads is how offset bugs are born.

### 1b. Shared types — `src/types.py` ✅

`Chunk` and `RetrievedSpan` plus span utilities (`sort_and_merge_spans`,
`sort_retrieved`, `contains`). One definition so chunkers, retrievers, metrics and
the pipeline cannot disagree about shape or ordering.

### 1c. Metrics — `src/metrics.py` ✅

The single source of truth for scoring: character precision/recall in `paper` and
`union` modes, doc-hit, span-hit, chunk-hit, MRR, macro-aggregation, per-category
grouping, abstention summaries, and paired bootstrap. Nothing else computes a
score.

### 2. Chunking — `src/chunking/` ✅

`fixed`, `recursive`, `sentence` and `clause` are implemented, span-validated and
compared in the grid (`docs/04-chunking.md`).

Input: one `Document`. Output: `list[Chunk]`, each with an **exact character
span** into that document's text. A chunk is invalid if
`document.content[start:end] != chunk.text`.

Interface (all chunkers implement it):

```python
class Chunker(Protocol):
    name: str
    def chunk(self, document: Document) -> list[Chunk]: ...
```

### 3. Retrieval — `src/retrieval/` 🚧

`bm25` is implemented; dense, hybrid and rerank are pending
(`docs/05-retrieval.md`).

```python
class Retriever(Protocol):
    name: str
    def index(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, k: int) -> list[RetrievedSpan]: ...
```

`RetrievedSpan` carries `file_path`, `start`, `end`, `score`, and the originating
`chunk_id`. Retrievers are stateless with respect to queries; all state is built
by `index()`.

### 4. Reranking — `src/retrieval/rerank.py` 📋

```python
class Reranker(Protocol):
    name: str
    def rerank(self, query: str, candidates: list[RetrievedSpan], k: int) -> list[RetrievedSpan]: ...
```

Rerankers consume top-`N` candidates (20–50) and return top-`k`. They never see
ground truth.

### 5. Abstention — `src/abstain.py` 🚧

`none` and `threshold` are implemented; margin, per-type and classifier are
pending (`docs/06-abstention.md`).

```python
class Abstainer(Protocol):
    name: str
    def decide(self, features: QueryFeatures) -> AbstentionDecision: ...
```

`QueryFeatures` is computed from retrieval output only: top score, score gap
between rank 1 and 2, document-match confidence (if a router ran), BM25/dense
agreement, number of results above a low floor. A policy returns `abstain: bool`
plus a human-readable reason, which the app and CLI display.

### 6. Pipeline — `src/pipeline.py` ✅

The only component that knows about all stages. Builds stages from a config,
indexes once, exposes `search(query) -> PipelineResult`. No branching on
"which chunker is this" — stages are polymorphic.

### 7. Evaluation — `eval/` 🚧

Single-config evaluation with run artefacts and the test-run guard is implemented;
the comparison grids are pending (`docs/08-evaluation-protocol.md`).

Calls `pipeline.search()` for every query in a split, scores with `src/metrics.py`,
aggregates, and writes a results table (CSV + markdown) and a JSON detail file
for error analysis.

### 8. Loop — `loop/` 📋

Consumes an evaluation result, diagnoses failures, proposes config changes,
re-evaluates, and accepts or rejects them statistically. It writes only to
`configs/generated/` and `logs/experiments.jsonl`; it never edits `src/`.

## Config schema

A pipeline config is a YAML document. Example (baseline):

```yaml
name: baseline
seed: 13

data:
  benchmark: cuad
  mini_limit: 194
  read_mode: raw          # raw | text

chunking:
  name: fixed
  params: {size: 512, overlap: 0}

retrieval:
  name: bm25
  params: {k1: 1.5, b: 0.75}
  top_k: 10

rerank:
  name: none
  params: {}

abstain:
  name: threshold
  params: {threshold: 0.0}   # 0.0 = never abstain (baseline)

query:
  strip_doc_hint: false
```

Rules:

1. Every knob that the loop may change lives in the config. No `if config.name ==
   "fixed": ...` in `pipeline.py`.
2. Unknown keys are a hard error (`validate_config`), so a typo cannot silently
   fall back to a default.
3. Each config records its `seed`. Anything stochastic (subsampling, bootstrap,
   negative sampling) draws from it.
4. `configs/generated/` is for loop-produced configs; hand-written ones never go
   there.

## Invariants

These are the properties that must hold for results to mean anything. Each has a
test or an enforcement point.

| # | Invariant | Enforced by |
|---|---|---|
| I1 | Ground-truth spans match stored answer text | `scripts/verify_data.py`, `tests/test_data.py` |
| I2 | Every retrieved span is a real substring of a corpus document | pipeline-level assertion in eval |
| I3 | Every chunk satisfies `content[start:end] == text` | chunker tests |
| I4 | The pipeline never receives ground-truth spans | call signature: `search(query: str)` |
| I5 | The test split is evaluated only for final reporting | `logs/test_runs.log` + eval guard |
| I6 | Ties are broken deterministically | sort key `(-score, file_path, start)` |
| I7 | Same config + same seed ⇒ same output | determinism test on a small fixture |
| I8 | The unanswerable set is frozen once committed | `data/unanswerable.jsonl` + git history |

## Request flow (final target)

```
query
  │
  ├─ optional query preprocessing (strip "Consider the ...;" doc hint)
  │
  ├─ retriever.search(query, k=top_k)          → candidate spans
  ├─ optional reranker.rerank(query, cands)     → reordered spans
  ├─ feature computation (top score, gap, ...)  → QueryFeatures
  ├─ abstainer.decide(features)                 → abstain?
  │
  └─ PipelineResult(spans[:k], abstained, reason, timings)
```

Timings are recorded per stage (retrieval, rerank, abstention) because the report
must show latency per method.

## Extension points

Adding a chunker or retriever is deliberately a three-step task:

1. Implement the protocol in the right package, with `name` set to the config key.
2. Register it in that package's `__init__.py` registry dict.
3. Add a config file and a test with a tiny fixture document.

No other file changes. If a new stage needs more than that, the abstraction is
wrong and should be fixed rather than special-cased.
