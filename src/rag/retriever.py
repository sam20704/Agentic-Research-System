from src.rag.embedding import model
from src.rag.vectorstore import collection


DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 10
MIN_SIMILARITY = 0.15


def expand_query(query: str) -> str:
    """
    Light query expansion for policy/economy terminology.
    Keep this conservative to avoid retrieval drift.
    """
    expansion_terms = (
        " semiconductor policy incentives fiscal support subsidy "
        "government support manufacturing EV supply chain"
    )
    return query.strip() + expansion_terms


def decompose_query(query: str) -> list[str]:
    """
    Conservative query decomposition.
    Only decompose when the question is clearly multi-part.

    Returns a short list with the original query first.
    """
    q = query.strip()
    q_lower = q.lower()

    subqueries = [q]

    # Compare India vs Taiwan semiconductor policy
    if "compare" in q_lower and "india" in q_lower and "taiwan" in q_lower and "semiconductor" in q_lower:
        subqueries.extend([
            "India semiconductor policy",
            "Taiwan semiconductor policy"
        ])

    # Compare India vs global EV trends
    elif "compare" in q_lower and "india" in q_lower and "global" in q_lower and "ev" in q_lower:
        subqueries.extend([
            "India EV adoption trends",
            "global EV adoption trends"
        ])

    # Combined semiconductor + EV policy impact
    elif (
        ("analyze" in q_lower or "impact" in q_lower or "together" in q_lower)
        and "semiconductor" in q_lower
        and "ev" in q_lower
        and "india" in q_lower
    ):
        subqueries.extend([
            "India semiconductor policy industrial growth",
            "India EV policy industrial growth"
        ])

    # Contradictions / gaps
    elif any(word in q_lower for word in ["contradiction", "contradictions", "gap", "gaps"]):
        subqueries.extend([
            "policy goals and adoption trends India EV",
            "policy goals and adoption trends India semiconductor"
        ])

    # Risks in global supply chains
    elif (
        any(word in q_lower for word in ["risk", "risks"])
        and "supply chain" in q_lower
    ):
        subqueries.extend([
            "EV supply chain risks",
            "semiconductor supply chain risks"
        ])

    # Challenges in India semiconductor manufacturing
    elif (
        any(word in q_lower for word in ["challenge", "challenges"])
        and "india" in q_lower
        and "semiconductor" in q_lower
    ):
        subqueries.extend([
            "India semiconductor manufacturing challenges",
            "barriers to semiconductor manufacturing in India"
        ])

    # Remove duplicates preserving order
    seen = set()
    final_subqueries = []

    for sq in subqueries:
        key = sq.strip().lower()
        if key and key not in seen:
            seen.add(key)
            final_subqueries.append(sq.strip())

    return final_subqueries


def distance_to_similarity(distance):
    """
    Convert Chroma distance to a rough similarity score.
    Smaller distance = better match.
    """
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + distance)


def deduplicate_results(results, text_prefix_len=160):
    seen = set()
    unique = []

    for item in results:
        text = item["text"].strip()
        key = text[:text_prefix_len].lower()

        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def retrieve_for_single_query(query, fetch_k=DEFAULT_FETCH_K, source_filter=None):
    expanded_query = expand_query(query)
    query_embedding = model.encode([expanded_query])[0]

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": fetch_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if source_filter is not None:
        if isinstance(source_filter, str):
            query_kwargs["where"] = {"source": source_filter}
        elif isinstance(source_filter, dict):
            query_kwargs["where"] = source_filter

    results = collection.query(**query_kwargs)

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    if not documents or not documents[0]:
        return []

    docs = documents[0]
    metas = metadatas[0] if metadatas and metadatas[0] else [{} for _ in docs]
    dists = distances[0] if distances and distances[0] else [None for _ in docs]

    scored_results = []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = distance_to_similarity(dist)
        scored_results.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "similarity": sim,
            "source_query": query
        })

    return scored_results


def merge_and_rerank_results(all_results, original_query, top_k=DEFAULT_TOP_K, min_similarity=MIN_SIMILARITY):
    """
    Merge multi-query results conservatively.

    Ranking logic:
    - best similarity is primary signal
    - tiny bonus if seen in multiple subqueries
    - tiny bonus if retrieved by original query
    """
    if not all_results:
        return []

    grouped = {}

    for item in all_results:
        text_key = item["text"].strip().lower()

        if text_key not in grouped:
            grouped[text_key] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "best_similarity": item["similarity"],
                "best_distance": item["distance"],
                "matched_queries": {item["source_query"]}
            }
        else:
            grouped[text_key]["best_similarity"] = max(
                grouped[text_key]["best_similarity"],
                item["similarity"]
            )

            current_best_distance = grouped[text_key]["best_distance"]
            new_distance = item["distance"]

            if current_best_distance is None:
                grouped[text_key]["best_distance"] = new_distance
            elif new_distance is not None:
                grouped[text_key]["best_distance"] = min(current_best_distance, new_distance)

            grouped[text_key]["matched_queries"].add(item["source_query"])

    merged = []
    for _, item in grouped.items():
        multi_query_bonus = 0.01 * (len(item["matched_queries"]) - 1)
        original_query_bonus = 0.02 if original_query in item["matched_queries"] else 0.0
        final_score = item["best_similarity"] + multi_query_bonus + original_query_bonus

        merged.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "distance": item["best_distance"],
            "similarity": item["best_similarity"],
            "matched_queries": sorted(item["matched_queries"]),
            "query_match_count": len(item["matched_queries"]),
            "final_score": final_score
        })

    merged = deduplicate_results(merged)
    merged = [item for item in merged if item["similarity"] >= min_similarity]
    merged.sort(key=lambda x: x["final_score"], reverse=True)

    return merged[:top_k]


def retrieve(
    query,
    top_k=DEFAULT_TOP_K,
    fetch_k=DEFAULT_FETCH_K,
    min_similarity=MIN_SIMILARITY,
    source_filter=None,
    verbose=False,
    return_metadata=False,
):
    """
    Conservative multi-query retrieval.

    For simple questions:
    - effectively behaves close to the earlier strong baseline

    For a few clear multi-part question types:
    - adds 1–2 focused subqueries
    """
    subqueries = decompose_query(query)

    if verbose:
        print("Querying vector DB...")
        print(f"Collection size: {collection.count()}")
        print(f"Original query: {query}")
        print(f"Subqueries ({len(subqueries)}):")
        for i, sq in enumerate(subqueries, 1):
            print(f"  {i}. {sq}")
        print(f"Top K: {top_k}")
        print(f"Fetch K per query: {fetch_k}")
        print(f"Min similarity: {min_similarity}")
        if source_filter is not None:
            print(f"Source filter: {source_filter}")

    all_results = []
    for sq in subqueries:
        sub_results = retrieve_for_single_query(
            sq,
            fetch_k=fetch_k,
            source_filter=source_filter
        )
        all_results.extend(sub_results)

    final_results = merge_and_rerank_results(
        all_results,
        original_query=query,
        top_k=top_k,
        min_similarity=min_similarity
    )

    if verbose:
        if not final_results:
            print("\nNo chunks passed final filtering.")
        else:
            print("\nFinal merged + reranked results:")
            for i, item in enumerate(final_results, 1):
                preview = item["text"][:240].replace("\n", " ")
                print(f"\nRank {i}")
                print(f"Final score: {item['final_score']:.4f}")
                print(f"Similarity: {item['similarity']:.4f}")
                print(f"Distance: {item['distance']}")
                print(f"Matched queries: {item['matched_queries']}")
                print(f"Metadata: {item['metadata']}")
                print(f"Text: {preview}...")

    if not final_results:
        return []

    if return_metadata:
        return final_results

    return [item["text"] for item in final_results]