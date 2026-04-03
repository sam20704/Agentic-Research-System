import os
import json
import csv
import re
from typing import List, Dict, Any

from src.rag.retriever import retrieve
from src.rag.generator import generate_answer


# =========================================================
# Config
# =========================================================

TEST_SET_PATH = "data/benchmarks/test_set.json"
GROUND_TRUTH_PATH = "data/benchmarks/ground_truth.json"

RESULTS_JSON_PATH = "data/benchmarks/results/results.json"
RESULTS_CSV_PATH = "data/benchmarks/results/results.csv"
FAILURES_JSON_PATH = "data/benchmarks/results/failures.json"

REFUSAL_TEXT = "I don't have enough information to answer this."
TOP_K = 5


# =========================================================
# Utilities
# =========================================================

def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], path: str):
    ensure_parent_dir(path)

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id",
                "question",
                "parsed_answer",
                "support",
                "expected_route",
                "requires_web",
                "faithfulness",
                "completeness",
                "routing",
                "retrieval_recall_proxy",
                "citation",
                "constraint_compliance",
                "total",
                "normalized_score",
                "failure_tags"
            ])
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", normalize_text(text))


def lexical_overlap_score(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union else 0.0


def contains_refusal(answer: str) -> bool:
    return REFUSAL_TEXT.lower() in normalize_text(answer)


# =========================================================
# Answer Parsing
# =========================================================

def parse_answer_output(raw_answer: str) -> Dict[str, Any]:
    """
    Parses model output into:
    - raw_answer
    - parsed_answer
    - support
    - has_citation
    """
    if not raw_answer or not raw_answer.strip():
        return {
            "raw_answer": "",
            "parsed_answer": "",
            "support": "",
            "has_citation": False
        }

    text = raw_answer.strip()

    if contains_refusal(text):
        return {
            "raw_answer": text,
            "parsed_answer": REFUSAL_TEXT,
            "support": "",
            "has_citation": False
        }

    answer_match = re.search(
        r"Answer:\s*(.*?)(?:\n\s*Support:|$)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    support_match = re.search(
        r"Support:\s*(.*)",
        text,
        re.DOTALL | re.IGNORECASE
    )

    parsed_answer = answer_match.group(1).strip() if answer_match else text
    support = support_match.group(1).strip() if support_match else ""
    has_citation = bool(support)

    return {
        "raw_answer": text,
        "parsed_answer": parsed_answer,
        "support": support,
        "has_citation": has_citation
    }


# =========================================================
# Data Loading
# =========================================================

def load_test_set() -> List[Dict[str, Any]]:
    return load_json(TEST_SET_PATH)


def load_ground_truth() -> List[Dict[str, Any]]:
    return load_json(GROUND_TRUTH_PATH)


def index_ground_truth(ground_truth: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {item["id"]: item for item in ground_truth}


# =========================================================
# Scoring
# =========================================================

def score_faithfulness(parsed_answer: str, contexts: List[str]) -> int:
    """
    Heuristic groundedness score.
    Refusals are rewarded because they avoid hallucination.
    """
    if not parsed_answer or not parsed_answer.strip():
        return 0

    if contains_refusal(parsed_answer):
        return 5

    combined_context = "\n".join(contexts)
    overlap = lexical_overlap_score(parsed_answer, combined_context)

    if overlap >= 0.50:
        return 5
    elif overlap >= 0.30:
        return 4
    elif overlap >= 0.15:
        return 3
    elif overlap >= 0.08:
        return 2
    else:
        return 1


def score_completeness(parsed_answer: str, key_points: List[str], expected_route: str) -> int:
    """
    Baseline completeness using substring matching.
    """
    if not parsed_answer or not parsed_answer.strip():
        return 0

    if expected_route == "refuse":
        return 5 if contains_refusal(parsed_answer) else 0

    if contains_refusal(parsed_answer):
        return 0

    if not key_points:
        return 5

    answer_norm = normalize_text(parsed_answer)
    matches = 0

    for kp in key_points:
        if normalize_text(kp) in answer_norm:
            matches += 1

    ratio = matches / len(key_points)

    if ratio == 1.0:
        return 5
    elif ratio >= 0.75:
        return 4
    elif ratio >= 0.50:
        return 3
    elif ratio >= 0.25:
        return 2
    else:
        return 1


def score_routing(parsed_answer: str, expected_route: str) -> int:
    """
    Ground truth routes:
    - rag
    - hybrid
    - refuse

    For current evaluator:
    - rag/hybrid => should answer, not refuse
    - refuse => should refuse
    """
    if not parsed_answer or not parsed_answer.strip():
        return 0

    refusal = contains_refusal(parsed_answer)

    if expected_route == "refuse":
        return 5 if refusal else 0

    if expected_route in {"rag", "hybrid"}:
        return 5 if not refusal else 0

    return 0


def score_retrieval_recall_proxy(contexts: List[str], key_points: List[str], expected_route: str) -> float:
    """
    Retrieval recall proxy:
    checks how many expected key points appear in retrieved contexts.

    For refuse questions, return 1.0 because retrieval recall is not the target metric.
    """
    if expected_route == "refuse":
        return 1.0

    if not contexts:
        return 0.0

    if not key_points:
        return 1.0

    combined = normalize_text(" ".join(contexts))
    hits = 0

    for kp in key_points:
        if normalize_text(kp) in combined:
            hits += 1

    return round(hits / len(key_points), 3)


def score_citation(has_citation: bool, parsed_answer: str, expected_route: str) -> int:
    """
    Citation scoring:
    - refuse => no citation required, score 5 if correctly refused
    - rag/hybrid => citation required for non-refusal answer
    """
    if expected_route == "refuse":
        return 5 if contains_refusal(parsed_answer) else 0

    if contains_refusal(parsed_answer):
        return 0

    return 5 if has_citation else 0


def score_constraint_compliance(parsed_answer: str, must_not_claim: List[str]) -> int:
    """
    Checks whether forbidden phrases appear in the answer.
    Simple lexical baseline.
    """
    if not parsed_answer or not parsed_answer.strip():
        return 0

    if contains_refusal(parsed_answer):
        return 5

    answer_norm = normalize_text(parsed_answer)

    for forbidden in must_not_claim:
        if normalize_text(forbidden) in answer_norm:
            return 0

    return 5


def compute_total_score(
    faithfulness: int,
    completeness: int,
    routing: int,
    citation: int,
    constraint_compliance: int
) -> int:
    return faithfulness + completeness + routing + citation + constraint_compliance


def compute_normalized_score(total: int, max_score: int = 25) -> float:
    return round((total / max_score) * 100, 2)


# =========================================================
# Failure Analysis
# =========================================================

def classify_failure(
    parsed_answer: str,
    contexts: List[str],
    expected_route: str,
    requires_web: bool,
    scores: Dict[str, Any],
    has_citation: bool
) -> List[str]:
    failure_tags = []

    if not contexts:
        failure_tags.append("retrieval_empty")

    if expected_route in {"rag", "hybrid"} and scores.get("retrieval_recall_proxy", 0.0) < 0.3:
        failure_tags.append("bad_retrieval")

    if expected_route in {"rag", "hybrid"} and contains_refusal(parsed_answer):
        failure_tags.append("wrong_refusal")

    if expected_route == "refuse" and not contains_refusal(parsed_answer):
        failure_tags.append("missed_refusal")

    if scores.get("faithfulness", 0) <= 2 and not contains_refusal(parsed_answer):
        failure_tags.append("possible_hallucination")

    if scores.get("completeness", 0) <= 2 and expected_route in {"rag", "hybrid"}:
        failure_tags.append("incomplete_answer")

    if expected_route in {"rag", "hybrid"} and not has_citation and not contains_refusal(parsed_answer):
        failure_tags.append("missing_citations")

    if scores.get("constraint_compliance", 0) == 0:
        failure_tags.append("violated_must_not_claim")

    if requires_web and expected_route == "refuse" and not contains_refusal(parsed_answer):
        failure_tags.append("web_required_but_answered")

    return failure_tags


# =========================================================
# Evaluation
# =========================================================

def run_evaluation(top_k: int = TOP_K) -> List[Dict[str, Any]]:
    test_set = load_test_set()
    ground_truth = load_ground_truth()
    gt_index = index_ground_truth(ground_truth)

    results = []

    for test in test_set:
        qid = test["id"]
        question = test["question"]

        gt = gt_index.get(qid)
        if gt is None:
            print(f"Skipping Q{qid}: no matching ground truth found.")
            continue

        expected_route = gt.get("expected_route", "rag")
        key_points = gt.get("key_points", [])
        must_not_claim = gt.get("must_not_claim", [])
        requires_web = gt.get("requires_web", False)

        print(f"\nRunning Q{qid}: {question}")

        try:
            contexts = retrieve(question, top_k=top_k, verbose=False)

            if not contexts:
                print("⚠️ No retrieval for this query")

            raw_answer = generate_answer(question, contexts)
            parsed = parse_answer_output(raw_answer)

            parsed_answer = parsed["parsed_answer"]
            support = parsed["support"]
            has_citation = parsed["has_citation"]

            faithfulness = score_faithfulness(parsed_answer, contexts)
            completeness = score_completeness(parsed_answer, key_points, expected_route)
            routing = score_routing(parsed_answer, expected_route)
            retrieval_recall_proxy = score_retrieval_recall_proxy(contexts, key_points, expected_route)
            citation = score_citation(has_citation, parsed_answer, expected_route)
            constraint_compliance = score_constraint_compliance(parsed_answer, must_not_claim)

            scores = {
                "faithfulness": faithfulness,
                "completeness": completeness,
                "routing": routing,
                "retrieval_recall_proxy": retrieval_recall_proxy,
                "citation": citation,
                "constraint_compliance": constraint_compliance
            }

            total = compute_total_score(
                faithfulness=faithfulness,
                completeness=completeness,
                routing=routing,
                citation=citation,
                constraint_compliance=constraint_compliance
            )
            normalized = compute_normalized_score(total)

            failure_tags = classify_failure(
                parsed_answer=parsed_answer,
                contexts=contexts,
                expected_route=expected_route,
                requires_web=requires_web,
                scores=scores,
                has_citation=has_citation
            )

            result = {
                "id": qid,
                "question": question,
                "retrieved_contexts": contexts,
                "raw_answer": raw_answer,
                "parsed_answer": parsed_answer,
                "support": support,
                "ground_truth": {
                    "key_points": key_points,
                    "must_not_claim": must_not_claim,
                    "expected_route": expected_route,
                    "requires_web": requires_web
                },
                "scores": scores,
                "total": total,
                "normalized_score": normalized,
                "failure_tags": failure_tags
            }

            results.append(result)

            print(f"Parsed answer: {parsed_answer}")
            print(
                f"Scores -> faithfulness={faithfulness}, "
                f"completeness={completeness}, routing={routing}, "
                f"retrieval_recall_proxy={retrieval_recall_proxy}, "
                f"citation={citation}, constraint_compliance={constraint_compliance}, "
                f"total={total}, normalized={normalized}"
            )

            if failure_tags:
                print(f"Failure tags: {failure_tags}")

        except Exception as e:
            error_result = {
                "id": qid,
                "question": question,
                "retrieved_contexts": [],
                "raw_answer": "",
                "parsed_answer": "",
                "support": "",
                "ground_truth": {
                    "key_points": key_points,
                    "must_not_claim": must_not_claim,
                    "expected_route": expected_route,
                    "requires_web": requires_web
                },
                "scores": {
                    "faithfulness": 0,
                    "completeness": 0,
                    "routing": 0,
                    "retrieval_recall_proxy": 0.0,
                    "citation": 0,
                    "constraint_compliance": 0
                },
                "total": 0,
                "normalized_score": 0.0,
                "failure_tags": ["runtime_error"],
                "error": str(e)
            }

            results.append(error_result)
            print(f"Error while evaluating Q{qid}: {e}")

    return results


# =========================================================
# Reporting
# =========================================================

def build_csv_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for r in results:
        rows.append({
            "id": r.get("id"),
            "question": r.get("question"),
            "parsed_answer": r.get("parsed_answer"),
            "support": r.get("support"),
            "expected_route": r.get("ground_truth", {}).get("expected_route"),
            "requires_web": r.get("ground_truth", {}).get("requires_web"),
            "faithfulness": r.get("scores", {}).get("faithfulness"),
            "completeness": r.get("scores", {}).get("completeness"),
            "routing": r.get("scores", {}).get("routing"),
            "retrieval_recall_proxy": r.get("scores", {}).get("retrieval_recall_proxy"),
            "citation": r.get("scores", {}).get("citation"),
            "constraint_compliance": r.get("scores", {}).get("constraint_compliance"),
            "total": r.get("total"),
            "normalized_score": r.get("normalized_score"),
            "failure_tags": ", ".join(r.get("failure_tags", []))
        })

    return rows


def extract_failures(results: List[Dict[str, Any]], threshold: float = 70.0) -> List[Dict[str, Any]]:
    failures = []

    for r in results:
        score = r.get("normalized_score", 0.0)
        tags = r.get("failure_tags", [])
        if score < threshold or tags:
            failures.append(r)

    return failures


def print_summary(results: List[Dict[str, Any]]):
    if not results:
        print("\nNo results to summarize.")
        return

    total_questions = len(results)

    avg_total = sum(r.get("total", 0) for r in results) / total_questions
    avg_normalized = sum(r.get("normalized_score", 0.0) for r in results) / total_questions
    avg_faithfulness = sum(r.get("scores", {}).get("faithfulness", 0) for r in results) / total_questions
    avg_completeness = sum(r.get("scores", {}).get("completeness", 0) for r in results) / total_questions
    avg_routing = sum(r.get("scores", {}).get("routing", 0) for r in results) / total_questions
    avg_retrieval = sum(r.get("scores", {}).get("retrieval_recall_proxy", 0.0) for r in results) / total_questions
    avg_citation = sum(r.get("scores", {}).get("citation", 0) for r in results) / total_questions
    avg_constraint = sum(r.get("scores", {}).get("constraint_compliance", 0) for r in results) / total_questions

    failures = extract_failures(results)
    refusal_count = sum(1 for r in results if contains_refusal(r.get("parsed_answer", "")))

    route_counts = {"rag": 0, "hybrid": 0, "refuse": 0}
    for r in results:
        route = r.get("ground_truth", {}).get("expected_route")
        if route in route_counts:
            route_counts[route] += 1

    print("\n=== Evaluation Summary ===")
    print(f"Total questions: {total_questions}")
    print(f"Route counts: {route_counts}")
    print(f"Average total score: {avg_total:.2f} / 25")
    print(f"Average normalized score: {avg_normalized:.2f}%")
    print(f"Average faithfulness: {avg_faithfulness:.2f} / 5")
    print(f"Average completeness: {avg_completeness:.2f} / 5")
    print(f"Average routing: {avg_routing:.2f} / 5")
    print(f"Average retrieval_recall_proxy: {avg_retrieval:.2f}")
    print(f"Average citation: {avg_citation:.2f} / 5")
    print(f"Average constraint_compliance: {avg_constraint:.2f} / 5")
    print(f"Refusal count: {refusal_count}")
    print(f"Flagged failures: {len(failures)}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    results = run_evaluation(top_k=TOP_K)

    save_json(results, RESULTS_JSON_PATH)
    save_csv(build_csv_rows(results), RESULTS_CSV_PATH)

    failures = extract_failures(results)
    save_json(failures, FAILURES_JSON_PATH)

    print_summary(results)

    print("\nEvaluation complete.")
    print(f"Saved JSON results to: {RESULTS_JSON_PATH}")
    print(f"Saved CSV results to: {RESULTS_CSV_PATH}")
    print(f"Saved failure analysis to: {FAILURES_JSON_PATH}")