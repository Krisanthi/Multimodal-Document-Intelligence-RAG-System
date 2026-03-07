"""RAG System Evaluation. Computes retrieval and answer quality metrics."""

import json
import time
import logging
import sys
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level="WARNING")


def cosine_sim(a: List[float], b: List[float]) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def word_overlap(pred: str, ref: str) -> float:
    pred_words = set(pred.lower().split())
    ref_words = set(ref.lower().split())
    if not ref_words:
        return 0.0
    return len(pred_words & ref_words) / len(ref_words)


def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if s in relevant_sources)
    return hits / len(top_k)


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    top_k = retrieved_sources[:k]
    if not relevant_sources:
        return 0.0
    hits = sum(1 for s in relevant_sources if s in top_k)
    return hits / len(relevant_sources)


def mrr(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    for i, s in enumerate(retrieved_sources):
        if s in relevant_sources:
            return 1.0 / (i + 1)
    return 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_evaluation(test_file: str = None):
    from pipeline import RAGPipeline
    from ingestion.embedder import Embedder

    if test_file is None:
        test_file = str(PROJECT_ROOT / "evaluation" / "test_queries.json")

    with open(test_file) as f:
        test_data = json.load(f)

    pipe = RAGPipeline(lazy_init=True)
    embedder = Embedder()

    print("=" * 60)
    print("  DocIntel RAG - System Evaluation Report")
    print("=" * 60)
    print()

    total_chunks = pipe.os_client.get_doc_count()
    sources = pipe.os_client.get_sources()
    print(f"  Index: {total_chunks} chunks from {len(sources)} documents")
    print(f"  Test queries: {len(test_data['queries'])}")
    print()
    print("-" * 60)

    all_p5, all_p10 = [], []
    all_r5, all_r10 = [], []
    all_mrr_scores = []
    all_answer_overlap = []
    all_answer_semantic = []
    all_latencies = []

    for i, q in enumerate(test_data["queries"]):
        question = q["question"]
        expected_sources = q.get("relevant_sources", [])
        expected_keywords = q.get("expected_keywords", [])
        reference_answer = q.get("reference_answer", "")

        print(f"\n  Q{i+1}: {question}")

        start = time.time()
        result = pipe.query(question=question, top_k=10, rerank_top_k=5, search_mode="hybrid")
        latency = time.time() - start
        all_latencies.append(latency)

        answer = result.get("answer", "")
        chunks = result.get("retrieved_chunks", [])
        chunk_sources = [c.get("source", "") for c in chunks]

        # Retrieval metrics
        p5 = precision_at_k(chunk_sources, expected_sources, 5)
        p10 = precision_at_k(chunk_sources, expected_sources, 10)
        r5 = recall_at_k(chunk_sources, expected_sources, 5)
        r10 = recall_at_k(chunk_sources, expected_sources, 10)
        m = mrr(chunk_sources, expected_sources)

        all_p5.append(p5)
        all_p10.append(p10)
        all_r5.append(r5)
        all_r10.append(r10)
        all_mrr_scores.append(m)

        # Answer quality
        overlap = 0.0
        if expected_keywords:
            answer_lower = answer.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            overlap = hits / len(expected_keywords)
        all_answer_overlap.append(overlap)

        semantic = 0.0
        if reference_answer:
            ref_vec = embedder.embed_text(reference_answer)
            ans_vec = embedder.embed_text(answer)
            semantic = cosine_sim(ref_vec, ans_vec)
        all_answer_semantic.append(semantic)

        print(f"      P@5={p5:.2f}  R@5={r5:.2f}  MRR={m:.2f}  "
              f"Keyword={overlap:.2f}  Semantic={semantic:.2f}  "
              f"Latency={latency:.1f}s")

    print()
    print("=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)

    n = len(test_data["queries"])
    avg = lambda lst: sum(lst) / len(lst) if lst else 0

    avg_p5 = avg(all_p5)
    avg_p10 = avg(all_p10)
    avg_r5 = avg(all_r5)
    avg_r10 = avg(all_r10)
    avg_mrr = avg(all_mrr_scores)
    avg_f1_5 = f1_score(avg_p5, avg_r5)
    avg_keyword = avg(all_answer_overlap)
    avg_semantic = avg(all_answer_semantic)
    avg_latency = avg(all_latencies)

    print()
    print("  Retrieval Quality:")
    print(f"    Precision@5:      {avg_p5 * 100:.1f}%")
    print(f"    Precision@10:     {avg_p10 * 100:.1f}%")
    print(f"    Recall@5:         {avg_r5 * 100:.1f}%")
    print(f"    Recall@10:        {avg_r10 * 100:.1f}%")
    print(f"    F1@5:             {avg_f1_5 * 100:.1f}%")
    print(f"    MRR:              {avg_mrr * 100:.1f}%")
    print()
    print("  Answer Quality:")
    print(f"    Keyword Accuracy: {avg_keyword * 100:.1f}%")
    print(f"    Semantic Match:   {avg_semantic * 100:.1f}%")
    print()
    print("  Performance:")
    print(f"    Avg Latency:      {avg_latency:.2f}s")
    print(f"    Queries/sec:      {1/avg_latency:.2f}" if avg_latency > 0 else "    N/A")
    print()
    print("=" * 60)

    report = {
        "num_queries": n,
        "num_indexed_chunks": total_chunks,
        "num_documents": len(sources),
        "retrieval": {
            "precision_at_5": round(avg_p5, 4),
            "precision_at_10": round(avg_p10, 4),
            "recall_at_5": round(avg_r5, 4),
            "recall_at_10": round(avg_r10, 4),
            "f1_at_5": round(avg_f1_5, 4),
            "mrr": round(avg_mrr, 4),
        },
        "answer_quality": {
            "keyword_accuracy": round(avg_keyword, 4),
            "semantic_similarity": round(avg_semantic, 4),
        },
        "performance": {
            "avg_latency_seconds": round(avg_latency, 3),
        },
    }

    out_path = PROJECT_ROOT / "evaluation" / "results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved to: {out_path}")
    print()

    return report


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(test_file)
