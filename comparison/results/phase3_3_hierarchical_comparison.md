# Phase 3.3 — Hierarchical Retrieval Validation

## Status

**NOT READY TO MERGE.** This document records validation status, not a claim that the full phase has passed acceptance.

## Intended architecture

```text
Query
  ↓
Rank compact section summaries
  ↓
Keep top_sections
  ↓
Rank pages only inside selected sections/subsections
  ↓
Keep top_pages_per_section
  ↓
Rank chunks only inside selected pages
  ↓
Deduplicate by canonical chunk_id
  ↓
Deterministic final ranking
  ↓
Canonical RetrievalResult[]
```

## Structural representation

- Canonical `DocumentChunk` objects remain the source of truth for text and provenance.
- A canonical chunk is stored once in the index.
- Page-to-chunk membership is represented separately through page child references. A chunk spanning multiple pages therefore has multiple page memberships but no conflicting single parent.
- `section_path` / `hierarchical_section_path` paths are represented as a section node and nested subsection nodes.
- Missing headings use an `Unsectioned` section node.
- Repeated identical headings can be distinguished only when the ingestion metadata supplies a stable structural identifier (`hierarchical_section_id`, `section_id`, `section_instance_id`, or `heading_id`). If such metadata is absent, the system does not invent a distinct structural identity from the title alone; this remains an ingestion limitation to resolve if the source data contains repeated-heading occurrences without identifiers.
- Original chunk provenance and metadata are preserved; hierarchy-specific node IDs use separate metadata keys.

## Benchmark inputs and status

The supplied Phase 3.3 ZIP does not contain `data/benchmarks/retrieval_corpus.json` or `data/benchmarks/retrieval_ground_truth.json`, which are the default inputs to `src/evaluation/run_hierarchical_benchmark.py`. Consequently, the real-corpus benchmark and same-query baseline comparison cannot be declared reproducible from this snapshot alone.

The existing `data/benchmarks/phase3_3_hierarchical_results.json` is historical output and must not be represented as a fresh run. Recover and verify the frozen corpus and ground truth before publishing new metrics. Do not edit frozen relevance labels to improve scores.

## Metrics to publish after a reproducible run

| Metric | Existing baseline | Hierarchical |
|---|---:|---:|
| Recall@5 | Not yet verified | Not yet rerun |
| Recall@10 | Not yet verified | Not yet rerun |
| MRR | Not yet verified | Not yet rerun |
| Mean / median latency | Not yet verified | Not yet rerun |
| Index construction time | Not yet verified | Not yet rerun |
| Memory measurement | Not yet verified | Python allocation metric available; process RSS still needed for a system-memory comparison |
| Section/page/chunk candidates | Not applicable or not instrumented in existing baseline | Instrumented by the hierarchical retriever |

## Remaining merge gates

1. Run Phase 3.3 unit/integration/benchmark tests in the intended project environment.
2. Run the complete regression suite and resolve collection/environment errors without broad skips.
3. Recover the exact frozen corpus and ground-truth inputs and verify their provenance.
4. Run the baseline and hierarchical method against identical queries and labels.
5. Include a real representative document integration test that is repository-relative and does not require a model download.
6. Record exact commands, outputs, environment/device, corpus identity, and benchmark methodology.
7. Review `git diff --check`, the final staged diff, and repository hygiene before opening a PR.

Phase 4 routing/orchestration, PageIndex refactoring, ColBERT refactoring, and changes to frozen ground truth are outside this phase.
