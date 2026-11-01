# 12 — Risks and limitations

Status: reviewed at submission. Each risk below is closed, mitigated, or carried
forward as a limitation.

## Risk register

Legend: **P** = probability, **I** = impact, 1–3 each. The register records the
risks that could have invalidated the comparison, and how each was handled.

| # | Risk | P | I | Status | Outcome |
|---|---|---|---|---|---|
| R1 | The report is unfinished when work stops | 2 | 3 | Closed | Scope was fixed before implementation; the report was written from the first result onward. |
| R2 | Dense stack install (`torch` + CUDA) fails or consumes the budget | 1 | 3 | Closed | Python 3.12 pins (ADR-001). BM25 remains fully functional without the dense extra, so the comparison degrades rather than blocks. |
| R3 | Abstention curves are too noisy to interpret | 3 | 2 | Carried | 46 negatives (34 train / 12 test); test results are reported with intervals and per-type counts, and no significance is claimed from 12 points. |
| R4 | Chunking grid expands into too many runs | 2 | 2 | Closed | Fixed grid: 4 chunkers × 3 sizes × 2 overlaps, BM25 only, `k` held at 10. Dense runs only on the selected chunking. |
| R5 | Precision collapses for every configuration | 2 | 2 | Accepted | Expected: the median answer is 317 characters. Precision-vs-size, span-hit and chunk-hit are reported together so the trade-off is visible. |
| R6 | Clause-aware chunking underperforms | 2 | 2 | Closed | It did underperform. Reported as a negative result with the span-containment diagnostic explaining why. |
| R7 | The loop accepts changes that are noise | 2 | 3 | Mitigated | Effect size + paired bootstrap + guardrails (`07-improvement-loop.md`). Rejections are logged, so a lenient criterion would be visible. |
| R8 | A change is tuned against test | 2 | 3 | Mitigated | `--i-know-this-is-final` flag, `logs/test_runs.log`, and the run count stated in the report. |
| R9 | A "negative" query is actually answerable | 3 | 3 | Mitigated | Every draft negative was checked against the contracts; an automated re-check verifies that no query is answerable by a ground-truth span; the set is frozen in its own commit. |
| R10 | Documentation drifts from code | 3 | 2 | Mitigated | ADR required in the same commit as any behaviour change (`docs/README.md`). |
| R11 | Report overstates categories as "CUAD official" | 2 | 2 | Closed | Categories are described as clause question types; ADR-009 records why. |
| R12 | Demo (app/CLI) is thin or broken at submission | 2 | 3 | Closed | Both are thin wrappers over `pipeline.search()` and are covered by smoke tests in `make test`. |

## Limitations

These are honest boundaries, not excuses. Each is defensible in a design review.

1. **One benchmark, one genre.** CUAD only: commercial contracts, English,
   US-centric. No claim of generalisation to other legal documents, languages, or
   jurisdictions.

2. **The mini subset is not a random sample.** It is chosen to cluster queries onto
   few documents (`02-data.md`), so it over-represents the 29 contracts with many
   clause questions. Absolute scores should not be compared to full-CUAD results.

3. **The query prefix leaks the document.** Every benchmark query starts with
   `"Consider the <contract>"`. Prefix routing is therefore not deployable as-is;
   it is reported as an ablation, and a semantic router is the honest analogue
   (ADR-010).

4. **Character precision is structurally low.** The median answer is 317
   characters; returning any useful context around it depresses precision. The
   metric is reported faithfully, but the system is not designed to maximise it at
   the cost of usefulness — the trade-off is shown, not hidden.

5. **No LLM components.** No query rewriting, no LLM reranking, no LLM-driven
   proposal (ADR-008). Whether an LLM would beat the best non-LLM pipeline here is
   an open question this project does not answer.

6. **Chunk sizes are in characters, not tokens.** Not directly comparable to
   papers reporting token-based chunk sizes (ADR-013).

7. **Abstention ground truth is authored.** The unanswerable set is written for
   this project, not drawn from a labelled corpus. It is frozen and typed, but it
   reflects one author's notion of "should abstain".

8. **Small test set.** 54 answerable and 12 unanswerable test queries. Intervals
   are wide; the report gives them rather than three significant figures.

9. **No parent–child retrieval or contextual enrichment.** Designed
   (`04-chunking.md`) but not built. This is the most likely source of further
   precision and recall gains.

10. **The loop is a local search, not a global one.** It walks a finite rule table
    and applies one change at a time. Re-opening rules after an accepted change
    (ADR-020) lets it recover from stale rejections, but it cannot discover a
    combination no rule proposes. The independent grids are the broader check on
    the same train data, and where the two disagree the report says so.

11. **Latency was measured on one laptop.** RTX 4060 Laptop, 8 GB. Absolute
    numbers are indicative; relative comparisons within the same run are the
    meaningful part.

12. **Cross-encoder reranking is evaluated on one base retriever** (hybrid in the
    grid), not swept. A different base might interact differently with the same
    reranker.

13. **Rule ordering is design, and design can be wrong.** The first rule table put
    hybrid before dense, which cost recall until the retrieval grid exposed it
    (ADR-019). The ordering is judgement, informed by train measurements.

14. **Evaluated at a single seed.** Bootstrap and cross-validation address sampling
    noise, not seed sensitivity of dense models. Not measured; stated as a gap.

## Things deliberately not claimed

- No claim that any component is state of the art.
- No claim that the loop found an optimal configuration — it found a sequence that
  improved recall beyond noise within a fixed rule set.
- No claim that the abstention classifier generalises beyond the four authored
  negative types.
- No claim of statistical significance for any comparison whose interval crosses
  zero.
