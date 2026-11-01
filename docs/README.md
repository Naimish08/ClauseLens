# Documentation

This directory is the engineering record for ClauseLens. It explains *why* the
system is shaped the way it is, not just what it does — the code shows the what.

All documents describe the implemented system as submitted. The decision log
(`09-decisions.md`) records the historical alternatives and rejected options.

Read in order for a full understanding, or jump to the topic you need.

| Document | What it covers |
|---|---|
| [00-overview.md](00-overview.md) | Goal, scope, what is in and out, success criteria |
| [01-architecture.md](01-architecture.md) | System layers, module contracts, invariants, config schema |
| [02-data.md](02-data.md) | Dataset provenance, mini selection, splits, verified integrity findings |
| [03-metrics.md](03-metrics.md) | Exact metric definitions and how they differ from the paper's wording |
| [04-chunking.md](04-chunking.md) | Chunkers, span contract, overlap hazards, expected trade-offs |
| [05-retrieval.md](05-retrieval.md) | BM25, dense models, hybrid RRF, reranking, document routing |
| [06-abstention.md](06-abstention.md) | The four abstention policies and the unanswerable set protocol |
| [07-improvement-loop.md](07-improvement-loop.md) | Diagnosis taxonomy, proposal rules, statistical acceptance |
| [08-evaluation-protocol.md](08-evaluation-protocol.md) | Train/test discipline, test-run accounting, reporting tables |
| [09-decisions.md](09-decisions.md) | Numbered decision log (ADR): context, decision, consequences |
| [10-experiment-log.md](10-experiment-log.md) | Schema and conventions for `logs/experiments.jsonl` |
| [11-reproducibility.md](11-reproducibility.md) | Environment, commands, seeds, expected timings |
| [12-risks-and-limitations.md](12-risks-and-limitations.md) | Risk register and the honest limitations list |
| [13-glossary.md](13-glossary.md) | Terms used across the codebase and report |

## Documentation rules

1. A decision that changes behaviour gets an entry in `09-decisions.md` **in the
   same commit** as the code.
2. Numbers in these documents are either measured (say when/how) or labelled as
   estimates. Never write an unverified number as fact.
3. When an experiment contradicts a design assumption, update the design document
   rather than leaving stale prose.
4. Results and analysis belong in `REPORT.md`, not here. This directory records
   engineering reasoning; the report records findings.
