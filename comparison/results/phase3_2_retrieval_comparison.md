# Phase 3.2 — Retrieval Comparison Results

## Benchmark Scope

The existing retrieval baseline was evaluated on the same 3 frozen Phase 3.2 navigation cases used for PageIndex evaluation.

**Frozen corpus:** 1,773 chunks  
**Queries:** 3

## Comparison

| Method | Recall@5 | MRR | Mean Query Latency |
|---|---:|---:|---:|
| BM25 + BGE-M3 + RRF + Qwen3-Reranker-0.6B | **0.5000** | **0.4167** | **1,317.19 ms** |
| PageIndex deterministic | 0.5000 | 0.6667 | ~1.46 ms |
| PageIndex Qwen | **1.0000** | **1.0000** | ~30,805 ms |

## Current Dense/Hybrid Baseline

**BM25 + BGE-M3 + RRF + Qwen3-Reranker-0.6B**

| Metric | Result |
|---|---:|
| Frozen corpus | 1,773 chunks |
| Queries | 3 |
| Recall@5 | **0.5000** |
| MRR | **0.4167** |
| Mean query latency | **1,317.19 ms** |
| BGE-M3 indexing time | **89,368.85 ms** |

### Per-Query Results

| Case | Recall@5 | MRR |
|---|---:|---:|
| `exact_structural_regional_trends` | 1.0000 | 1.0000 |
| `semantic_structural_ev_adoption_regions` | 0.0000 | 0.0000 |
| `multihop_regional_supply_chain` | 0.5000 | 0.2500 |

## PageIndex Deterministic

| Metric | Result |
|---|---:|
| Recall@5 | 0.5000 |
| MRR | 0.6667 |
| Mean query latency | ~1.46 ms |

## PageIndex Qwen Navigator

| Metric | Result |
|---|---:|
| Recall@5 | **1.0000** |
| MRR | **1.0000** |
| Mean query latency | ~30,805 ms |

## Interpretation

The existing dense/hybrid baseline achieved **0.5000 Recall@5** and **0.4167 MRR** on the Phase 3.2 cases.

The deterministic PageIndex navigator achieved the same Recall@5 while improving MRR and operating at substantially lower query latency.

The Qwen PageIndex navigator achieved **1.0000 Recall@5** and **1.0000 MRR** on these three cases, at substantially higher query latency.

These results are limited to the frozen Phase 3.2 benchmark cases and should not be interpreted as a general retrieval-quality conclusion beyond this benchmark.