# 12 — Risks and limitations

Status: live document. Reviewed at every commit that touches scope.

## Risk register

Legend: **P** = probability, **I** = impact, 1–3 each. Only P×I ≥ 4 needs a
mitigation that is actually scheduled.

| # | Risk | P | I | Score | Mitigation / status |
|---|---|---|---|---|---|
| R1 | Deadline (Mon 2 Nov, 17:00 IST) arrives with the report unfinished | 2 | 3 | 6 | Scope cut to P0 on day one (ADR-008). Report written incrementally from the first result, not at the end. |
| R2 | Dense stack install (`torch` + CUDA) eats hours or fails | 1 | 3 | 3 | Python 3.12 pins (ADR-001). BM25 grid is fully functional without it, so a failure degrades the comparison but does not block the deliverable. |
| R3 | Abstention curves are too noisy to interpret (≈13 test negatives) | 3 | 2 | 6 | Target 40–50 negatives (≈28–35 train); report per-type curves with `n` and intervals; never claim significance from 13 points. |
| R4 | Chunking grid explodes into too many runs | 2 | 2 | 4 | Fixed grid: 4 chunkers × 3 sizes × 2 overlaps, BM25 only, `k` held at 10. Dense runs only on the best 2–3 chunk configs. |
| R5 | Precision collapses for every configuration, making the table uninformative | 2 | 2 | 4 | Expected, not a failure: report precision-vs-size curve and span-hit/chunk-hit alongside, so the trade-off is visible rather than a wall of zeros. |
| R6 | Clause-aware chunker underperforms because CUAD formatting defeats the regex | 2 | 2 | 4 | Clause chunker is one row, not a dependency. If it underperforms, that is a reported finding with the span-containment diagnostic explaining why. |
| R7 | Loop accepts changes that are noise | 2 | 3 | 6 | Effect-size + paired bootstrap + guardrails (`07-improvement-loop.md`). Rejections are logged, so a lenient criterion would be visible. |
| R8 | Someone (human or AI) tunes against test | 2 | 3 | 6 | `--i-know-this-is-final` flag, `logs/test_runs.log`, report states the run count. |
| R9 | Negative set is mislabelled (a "negative" is actually answerable) | 3 | 3 | 9 | Human verification of every draft negative; automated re-check that no query is answerable by any ground-truth span; frozen commit before the loop. |
| R10 | Documentation drifts from code | 3 | 2 | 6 | ADR required in the same commit as any behaviour change (`docs/README.md`). Pseudocode in docs is marked as such. |
| R11 | Report overstates categories as "CUAD official" | 2 | 2 | 4 | Categories renamed to "clause question type" everywhere; ADR-009 records why. |
| R12 | Demo (app/CLI) is thin or broken at submission | 2 | 3 | 6 | Built last but kept intentionally thin: both are wrappers over `pipeline.search()`. Smoke test in CI-equivalent `make test`. |

## Limitations to state in the report

These are honest boundaries, not excuses. Each is written to be defensible in a
debrief.

1. **One benchmark, one genre.** CUAD only: commercial contracts, English,
   US-centric. No claim of generalisation to other legal documents, languages, or
   jurisdictions.

2. **The mini subset is not a random sample.** It is chosen to cluster queries onto
   few documents (`02-data.md`), so it over-represents the 29 contracts with many
   clause questions. Absolute scores should not be compared to full-CUAD results.

3. **The query prefix leaks the document.** Every query starts with
   `"Consider the <contract>"`. Prefix routing is therefore not deployable as-is;
   it is reported as an ablation, and the `semantic` router is the honest analogue
   (ADR-010).

4. **Character precision is structurally low.** Median answer is 317 characters;
   returning any useful context around it depresses precision. The metric is
   reported faithfully, but the system is not designed to maximise it at the cost
   of usefulness — the trade-off is shown, not hidden.

5. **No LLM components.** No query rewriting, no LLM reranking, no LLM-driven
   proposal (ADR-008). Whether an LLM would beat the best non-LLM pipeline here is
   an open question this project does not answer.

6. **Chunk sizes are in characters, not tokens.** Not directly comparable to
   papers reporting token-based chunk sizes (ADR-013).

7. **Abstention ground truth is synthetic.** The unanswerable set is authored, not
   drawn from a labelled corpus. It is frozen and typed, but it reflects one
   author's notion of "should abstain".

8. **Small test set.** 54 answerable and ~13 unanswerable test queries. Intervals
   are wide; the report gives them rather than three significant figures.

9. **No parent–child retrieval or contextual enrichment.** Designed
   (`04-chunking.md`) but not built. This is the single most likely source of
   further precision/recall gains.

10. **Latency measured on one laptop.** RTX 4060 Laptop, 8 GB. Absolute numbers
    are indicative; relative comparisons within the same run are the meaningful
    part.

11. **Cross-encoder reranking is evaluated on the best hybrid only**, not swept,
    for time. A different base retriever might interact differently with the same
    reranker.

12. **Evaluated at a single seed.** Bootstrap and CV address sampling noise, not
    seed sensitivity of dense models. Not measured; stated as a gap.

## Things deliberately not claimed

- No claim that any component is state of the art.
- No claim that the loop found an optimal configuration — it found an
  >noise-improving sequence within a fixed rule set.
- No claim that the abstention classifier generalises beyond the four authored
  negative types.
- No claim of statistical significance for any comparison whose interval crosses
  zero.
