# AGENTS.md

Conventions for anyone — human or AI — working in this repository. Read this
before changing code. It is authoritative when it conflicts with habit.

## What this project is

A config-driven clause-retrieval harness over LegalBench-RAG (CUAD subset, 194
queries), built to produce defensible comparison tables, not a demo. The product
is the evaluation, the loop, and the write-up.

Start with [`docs/00-overview.md`](docs/00-overview.md) and
[`docs/01-architecture.md`](docs/01-architecture.md).

## Commands

```bash
make setup      # uv venv --python 3.12 + core deps
make data       # download + unpack the benchmark (~90 MB, needs network once)
make verify     # verify ground-truth character offsets (must pass)
make negatives  # validate the frozen unanswerable set
make test       # pytest
make lint       # ruff
make eval CONFIG=configs/baseline.yaml
make loop CONFIG=configs/baseline.yaml
make grid               # chunking comparison + precision/size plot
make retrieval-grid     # retrieval comparison, chunking held fixed
make abstention         # abstention trade-off curves
make app        # streamlit
make cli ARGS="--config configs/final.yaml --query '...'"
```

Use `.venv/bin/python` (or `make`). Do **not** use the system `python3` — it is
3.14 and lacks the pinned environment (ADR-001).

## Hard rules

1. **Never change the split.** `assign_split` is part of the reported result. No
   re-seeding, no re-thresholding, no "just this once".
2. **Never edit `data/unanswerable.jsonl` after it is committed.** It is frozen by
   design; the git history is the evidence. Additions require a new file and an
   explanation in the report, not an edit.
3. **Never tune on test.** Evaluate test only with
   `--i-know-this-is-final`; every run appends to `logs/test_runs.log`; the report
   states the count.
4. **Ground truth must never be an inference input.** `pipeline.search()` takes a
   string. If a proposed change needs spans at query time, it is wrong.
5. **Never repair a span mismatch.** Report it. A "fixed" span invalidates every
   comparison.
6. **Never commit data, secrets, or caches.** `data/*` except
   `data/unanswerable.jsonl` is gitignored; so are `.env`, keys, `artifacts/`.
7. **Never make the pipeline depend on the query's document hint by default.**
   Routing is an ablation (ADR-010).
8. **Every behaviour change gets an ADR entry** in `docs/09-decisions.md`, in the
   same commit.

## Repository map

| Path | Owns |
|---|---|
| `src/data.py` | all corpus/benchmark reading, splits, offset verification |
| `src/metrics.py` | every metric; nothing else computes scores |
| `src/chunking/` | document → `list[Chunk]`, span-exact |
| `src/retrieval/` | `index()`/`search()`; bm25, dense, hybrid, rerank |
| `src/abstain.py` | abstention policies and the feature contract |
| `src/pipeline.py` | the only orchestrator; config → ranked spans |
| `eval/` | scoring, tables, run artefacts |
| `loop/` | diagnose, propose, accept, log |
| `app/` | streamlit + CLI, thin wrappers only |
| `scripts/` | one-shot operational scripts (download, verify, reports) |
| `docs/` | the engineering record |
| `configs/` | hand-written configs; `configs/generated/` for the loop |
| `tests/` | unit and fixture tests |

If a change does not fit one of those rows, it is probably in the wrong place.

## Invariants to preserve

See `docs/01-architecture.md` for the full table (I1–I8). The short version:

- spans are character offsets into UTF-8-decoded text read in `raw` mode
- every chunk satisfies `content[start:end] == text`
- retrieved spans are always exactly chunk spans
- ranking ties break on `(-score, file_path, start)`
- same config + same seed ⇒ same output

## Adding a stage

Three steps, no more:

1. Implement the protocol in the right package; set `name` to the config key.
2. Register it in that package's registry.
3. Add a config file and a test on a tiny fixture.

If it needs a fourth step, the abstraction is wrong — fix the abstraction instead
of special-casing.

## Code style

- Python 3.12, type hints on public functions, `from __future__ import annotations`.
- `ruff` with line length 100; keep the lint clean.
- Dataclasses for records, `Protocol` for stage interfaces.
- No global mutable state. Caches belong in `artifacts/` keyed by a config hash.
- Prefer stdlib. Every new dependency needs a reason in the ADR.
- Docstrings state *why* when the *what* is obvious from the signature.
- Never silently swallow an exception; a failed stage should say which stage.

## Testing expectations

- `make test` must pass before every commit.
- Metrics changes require a hand-computed fixture — not just a property test.
- Chunker changes require a span-contract test (`content[start:end] == text`).
- Loop changes require a test that rejections are logged.
- Integration tests that need data use `@requires_data` and skip cleanly on a
  fresh clone.

## Commit conventions

- One logical change per commit. Prefix: `data:`, `metrics:`, `chunk:`, `retrieval:`,
  `abstain:`, `loop:`, `eval:`, `app:`, `docs:`, `chore:`.
- The message says what changed and why in one or two lines.
- Docs and ADRs go in the same commit as the code they describe.
- The negative-set commit is separate and explicitly named, so the freeze is
  visible in the history.

## Documentation policy

- `docs/` is a set of topic files, indexed by `docs/README.md`. Do not create a
  second monolithic design document.
- Numbers in docs are either measured (say when and how) or labelled a target.
- Mark implementation state with the legend in `docs/README.md` (✅ 🚧 📋 ⚠️).
- Results and analysis belong in `REPORT.md`, never in `docs/`.

## AI-specific guidance

- Record AI-assisted changes in the report's AI-usage section, at the level of
  "which components", not a transcript.
- Do not paste the assignment brief into the repository.
- Cache any model output used in an evaluation; evaluations must be reproducible
  offline.
- If a generated change alters a headline number, re-run the full comparison grid
  rather than adjusting one cell.
- When unsure between two designs, write the trade-off into `docs/09-decisions.md`
  and pick the one that is easier to explain in a debrief.

## Known traps

- `select_mini` must reproduce the upstream reseeding exactly; "cleaner" versions
  pick a different 194 queries (`docs/02-data.md`).
- The benchmark JSON contains an `answer` field because the reference model exposes
  a Pydantic `computed_field`; do not assume spans are the only data.
- Default text-mode reads translate `\r\n`; use `read_mode: raw` unless you are
  deliberately testing the alternative (ADR-004).
- `e5` needs `"query: "` / `"passage: "` prefixes and `bge` needs a query
  instruction; omitting them silently costs points.
- Overlapping retrieved chunks make the `paper` and `union` metric variants
  disagree; report which one a table uses.
- With 140 train queries, one query flipping moves recall by ~0.007. Do not react
  to deltas smaller than the bootstrap interval.
