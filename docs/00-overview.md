# 00 — Overview

## What this is

ClauseLens answers natural-language questions about contracts by returning a
ranked list of passages with source document and exact character offsets — or by
returning nothing when no passage is relevant enough.

It is built as an **experiment platform**, not a pipeline. One config-driven code
path, one metric implementation, and a swappable stage at every step. The
deliverable is the comparison tables and the analysis, all produced by the same
harness.

Benchmark: [LegalBench-RAG](https://arxiv.org/abs/2408.10343), CUAD subset,
capped at 194 queries, scored at character-level precision and recall.

## Why this framing

A conventional pipeline looks like: chunk → embed → top-k → demo → report
saying "it works". Three problems with that:

1. **No comparison.** Without a fixed metric implementation and a fixed code path,
   "BM25 beat dense retrieval" is an anecdote about two different scripts.
2. **No error budget.** Character-level precision punishes large chunks brutally.
   Without a precision-vs-size plot, a score cannot be explained.
3. **No honesty mechanism.** Tuning against the test set is invisible unless the
   split and the run count are enforced by tooling.

The platform framing makes each of those structural rather than aspirational:
configs make comparisons real, the metric is tested once and reused, and the
split/test-run accounting is enforced in code.

## Scope

**In scope** (the committed critical path):

| Area | Included |
|---|---|
| Data | Download, offset verification, hash-fixed split, 29-document CUAD-mini corpus |
| Metrics | Character precision/recall at several k, doc-hit, span-hit, MRR, per-clause-type breakdown |
| Chunking | Fixed (multiple sizes/overlaps), recursive, sentence window, clause-aware |
| Retrieval | BM25, 3 dense embedders, hybrid RRF, cross-encoder rerank on the best hybrid |
| Abstention | Global threshold, margin rule, per-type threshold, logistic-regression classifier; trade-off curves |
| Loop | Diagnosis → proposal → statistical acceptance, ≥5 logged iterations |
| Interfaces | Streamlit app, CLI |
| Write-up | README, REPORT, 10+ diagnosed failures |

**Explicitly out of scope** (deliberate scope decisions, ADR-008):

- Any LLM: query rewriting, LLM reranking, LLM proposal in the loop
- `bge-m3` and `Qwen3-Embedding-0.6B` (keep the embedder study to three models)
- Optuna hyper-parameter search (a hand-specified grid is enough and is
  reproducible without a study database)
- Two-stage document router as a *submit* feature (it is measured as an
  ablation, because the query prefix makes it nearly free — see `05-retrieval.md`)
- Parent–child chunking and contextual enrichment (documented as next steps)

Cutting these is a deliberate, recorded choice (`09-decisions.md`, ADR-008), not
an omission. The report states it in Limitations.

## Success criteria

The project is complete when all of the following hold:

1. `make data && make verify` reproduces the corpus and confirms every
   ground-truth span, reporting mismatches rather than repairing them.
2. `make test` passes, including metric unit tests with hand-computed fixtures.
3. `make eval CONFIG=configs/baseline.yaml` produces the train table, and
   `configs/final.yaml` produces the final table, both from the same code path.
4. Abstention is measured in both directions on train, and once on test.
5. `logs/experiments.jsonl` contains a baseline plus at least five iterations,
   accepted *and* rejected, each with a reason and a proposer.
6. `REPORT.md` is readable in ~15 minutes and contains headline test numbers,
   iteration history, a 10-item error analysis, and limitations.
7. Every reported number is reproducible from a config file in the repository.

## Scope and time-box

The scope above was fixed before implementation, so that cuts are a recorded
decision rather than an accident. `12-risks-and-limitations.md` lists the
limitations that remain in the submitted system.

## Reader's guide

- **Want to run it?** `README.md`.
- **Want to judge it?** `REPORT.md`.
- **Want to understand the engineering?** This directory, in order.
- **Want to change a stage?** `01-architecture.md` then the stage's document,
  then `09-decisions.md`.
