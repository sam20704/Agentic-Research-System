# Phase 3.3 — Hierarchical Retrieval Validation

## Status

**NOT READY TO MERGE.** The benchmark is now runnable from the supplied repository snapshot and compares against the frozen Phase 3.1 summary when the corpus, query, and relevance-label counts match. The measured retrieval quality is lower than both historical Phase 3.1 baselines, so this implementation should not be presented as a quality improvement.

## Intended architecture

```text
Query
  ↓
Score compact section/subsection summaries
  ↓
Keep top_sections
  ↓
Score pages only within selected sections/subsections
  ↓
Keep top_pages_per_section
  ↓
Score chunks only within selected pages
  ↓
Deduplicate by canonical chunk_id
  ↓
Deterministic final ranking
  ↓
Canonical RetrievalResult[]
```

Section selection necessarily scores each compact section/subsection summary. Progressive narrowing is demonstrated by the page and chunk stages: they score candidates only after parent selections. The benchmark now reports both retained candidate counts and actual nodes scored at each level; these values must not be conflated.

## Structural representation

- Canonical `DocumentChunk` objects remain the source of truth for text and provenance.
- A canonical chunk is stored once in the index.
- Page-to-chunk membership is represented separately through page child references. A chunk spanning multiple pages therefore has multiple page memberships but no conflicting single parent.
- `section_path` / `hierarchical_section_path` paths are represented as a section node and nested subsection nodes.
- Missing headings use an `Unsectioned` section node.
- Repeated identical headings can be distinguished only when ingestion metadata supplies a stable structural identifier (`hierarchical_section_id`, `section_id`, `section_instance_id`, or `heading_id`). Without such metadata, identical paths may be grouped together. The implementation cannot safely infer whether identical titles on different pages are one multi-page section or separate heading instances; this remains an ingestion limitation.
- Original chunk provenance and metadata are preserved; hierarchy-specific node IDs use separate metadata keys.

## Reproducible benchmark run

Command, from the repository root:

```bash
python -m src.evaluation.run_hierarchical_benchmark --output /tmp/phase3_3_hierarchical_results.json
```

Inputs present in the supplied snapshot:

- `data/benchmarks/retrieval_corpus.json` — 1,773 chunks
- `data/benchmarks/retrieval_ground_truth.json` — 20 queries, 186 relevance pairs
- `data/benchmarks/results/phase3_1_retrieval_summary.json` — historical Phase 3.1 summary, 1,773 chunks and the same 20 queries / 186 relevance pairs

One run in the review environment produced the following results. Timing and memory values are environment-dependent and should be rerun in the project's intended environment before publication.

| Metric | Hierarchical (this run) | Phase 3.1 BGE-M3 (historical) | Phase 3.1 ColBERT (historical) |
|---|---:|---:|---:|
| Recall@5 | 0.2063 | 0.3137 | 0.3189 |
| Recall@10 | 0.2814 | 0.5920 | 0.4904 |
| MRR | 0.6389 | 0.7500 | 0.8875 |
| Mean latency (ms) | 60.56 | 84.28 | 223.08 |
| Median latency (ms) | 57.13 | 82.19 | 145.87 |
| Index construction (ms) | 782.65 | Not available in summary | Not available in summary |
| Peak Python allocation (MiB) | 3.02 | Not the same memory metric | Not the same memory metric |
| Average section summaries scored | 8.00 | Not applicable / not recorded | Not applicable / not recorded |
| Average pages scored after section selection | 224.55 | Not recorded | Not recorded |
| Average chunks scored after page selection | 88.40 | Not recorded | Not recorded |
| Average chunk scoring reduction | 95.01% | Not recorded | Not recorded |

The Phase 3.1 quality values come from the committed historical summary, not a simultaneous rerun. The benchmark records this as a historical comparison and explicitly warns against treating historical latency or memory as a controlled head-to-head result. Hierarchical Recall@5, Recall@10, and MRR are below both recorded baselines in this run; this is a limitation, not a result to hide or explain away.

## Validation performed during this review

- `python -m pytest tests/test_hierarchical*.py tests/test_pageindex_navigation_evaluation.py -q` — **51 passed** after the benchmark-comparison changes.
- `python -m src.evaluation.run_hierarchical_benchmark --output /tmp/phase3_3_hierarchical_results.json` — completed and matched the historical Phase 3.1 dataset counts.
- `python -m pytest -q` — could not complete collection in the review container because project dependencies were not installed (`rank_bm25`, `sentence_transformers`, `transformers`, `qdrant_client`, and `ollama`). The review also found that `rank-bm25` was imported by the existing sparse retriever but was not declared in `requirements.txt`; it has now been added. Rerun the full suite after installing the declared requirements in the project's configured virtual environment.
- The supplied snapshot contains `.pageindex/Global_Supply_Chain_Report_DEC2025_EV_1.json`, and PageIndex evaluation paths are anchored to the repository root. The targeted PageIndex navigation evaluation tests passed in this environment.

## Remaining merge gates

1. Run the full suite in the configured project environment and resolve any failures without broad skips.
2. Decide how ingestion will provide stable identifiers for repeated headings when the source does not already include them; do not infer separate sections from titles alone.
3. Review the lower retrieval-quality metrics and decide whether the method meets the Phase 3.3 acceptance criteria. Do not modify frozen ground truth to improve results.
4. If publishing performance comparisons, rerun the baseline and hierarchical method under the same controlled environment and record exact commands, corpus identity, device, and methodology.
5. Run `git diff --check`, inspect the staged diff, and confirm no generated local artifacts were added.

Phase 4 routing/orchestration, PageIndex refactoring, ColBERT refactoring, and changes to frozen ground truth are outside this phase.
