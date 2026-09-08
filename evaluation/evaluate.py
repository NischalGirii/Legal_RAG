import os
import sys
import json
import time
import pickle
import argparse
from typing import List, Dict, Any

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_PATH,
    BM25_INDEX_PATH,
    CASE_INDEX_PATH,
    INGEST_METADATA_PATH,
    LLM_MODEL,
)
from src.text_processor import clean_devanagari_text, normalize_digits, clean_asr_transcript
from src.hybrid_search import (
    perform_hybrid_search,
    extract_query_identifiers,
    detect_query_intent,
    decision_exists,
    get_lookup_indexes,
)
from src.llm_generator import generate_nepali_answer, generate_comparison_answer
from src.sparse_index import retriever_from_pickle

sys.stdout.reconfigure(encoding='utf-8')

def load_benchmark_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(output_path: str = "evaluation/results.json", top_k: int = 5, run_llm: bool = True):
    dataset_path = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    queries = load_benchmark_dataset(dataset_path)
    print(f"Loaded {len(queries)} evaluation queries from {dataset_path}\n")

    # Load engines
    print("Loading search engines and indices...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = retriever_from_pickle(bm25_data)
    chunk_metadata = bm25_data["metadata"]
    get_lookup_indexes(chunk_metadata)

    metadata_info = None
    if os.path.exists(CASE_INDEX_PATH):
        try:
            with open(CASE_INDEX_PATH, "r", encoding="utf-8") as f:
                case_index_data = json.load(f)
            metadata_info = {"case_metadata": case_index_data}
        except Exception:
            metadata_info = None
    elif os.path.exists(INGEST_METADATA_PATH):
        try:
            with open(INGEST_METADATA_PATH, "r", encoding="utf-8") as f:
                ingest_data = json.load(f)
            metadata_info = {"case_metadata": ingest_data.get("case_metadata", {})}
        except Exception:
            metadata_info = None

    results_log = []
    
    doc_hit_1_count = 0
    doc_hit_3_count = 0
    doc_hit_5_count = 0
    mrr_total = 0.0
    
    valid_retrieval_queries = 0
    negative_queries_total = 0
    negative_queries_passed = 0
    
    fact_recall_total = 0.0
    llm_tested_count = 0

    print("=" * 80)
    print(f"{'ID':<32} | {'Doc Hit@1':<9} | {'Doc Hit@5':<9} | {'MRR':<6} | {'Facts':<7} | {'Status'}")
    print("=" * 80)

    for item in queries:
        qid = item["id"]
        raw_query = item["query"]
        expected_dec = item.get("expected_decision_no")
        expected_source = item.get("expected_source")
        expected_facts = item.get("expected_facts", [])
        is_not_found = item.get("is_not_found", False)

        # Preprocess
        cleaned_query = clean_devanagari_text(clean_asr_transcript(raw_query))
        identifiers = extract_query_identifiers(cleaned_query)
        intent = detect_query_intent(cleaned_query)

        # Multi-case check
        multi_cases = identifiers.get("multiple_decision_nos")
        current_case = None
        requested_dec = identifiers.get("decision_no")
        if requested_dec and decision_exists(requested_dec, chunk_metadata):
            current_case = {"case_id": f"decision_{requested_dec}", "decision_no": requested_dec}

        t0 = time.time()

        # Hybrid Search
        if intent in ("LIST_CASES", "COMPARISON") and not current_case:
            retrieval_case = None
        else:
            retrieval_case = current_case

        retrieved_items = perform_hybrid_search(
            query=cleaned_query,
            collection=collection,
            model=model,
            bm25=bm25,
            chunk_metadata=chunk_metadata,
            top_k=top_k,
            alpha=0.7,
            current_case=retrieval_case,
            identifiers=identifiers,
        )

        retrieval_time = time.time() - t0

        # LLM Generation
        llm_answer = ""
        if run_llm:
            if multi_cases and len(multi_cases) >= 2:
                llm_answer = generate_comparison_answer(
                    query=cleaned_query,
                    decision_numbers=multi_cases,
                    metadata_info=metadata_info,
                    chunk_metadata=chunk_metadata,
                    collection=collection,
                    model=model,
                    bm25=bm25,
                ) or ""
            else:
                llm_answer = generate_nepali_answer(
                    query=cleaned_query,
                    retrieved_items=retrieved_items,
                    model_name=LLM_MODEL,
                    current_case=current_case,
                    metadata_info=metadata_info,
                    comparison_mode=False,
                    chunk_metadata=chunk_metadata,
                ) or ""

        # Evaluation of Retrieval
        retrieved_docs = [r.get("source") for r in retrieved_items]
        retrieved_decs = [str(r.get("decision_no") or "") for r in retrieved_items]

        doc_hit_1 = False
        doc_hit_3 = False
        doc_hit_5 = False
        mrr = 0.0

        if not is_not_found:
            valid_retrieval_queries += 1
            exp_sources = [s.strip() for s in (expected_source or "").split(",") if s.strip()]
            exp_decs = [d.strip() for d in (expected_dec or "").split(",") if d.strip()]

            # Calculate rank
            for rank_idx, r_doc in enumerate(retrieved_docs, start=1):
                r_dec = retrieved_decs[rank_idx - 1] if rank_idx - 1 < len(retrieved_decs) else ""
                match = (expected_source == "ALL") or (r_doc in exp_sources) or (r_dec and r_dec in exp_decs)
                if match:
                    if mrr == 0.0:
                        mrr = 1.0 / rank_idx
                    if rank_idx == 1:
                        doc_hit_1 = True
                    if rank_idx <= 3:
                        doc_hit_3 = True
                    if rank_idx <= 5:
                        doc_hit_5 = True

            if doc_hit_1:
                doc_hit_1_count += 1
            if doc_hit_3:
                doc_hit_3_count += 1
            if doc_hit_5:
                doc_hit_5_count += 1
            mrr_total += mrr
        else:
            negative_queries_total += 1
            # Check if answer appropriately rejected
            rejection_indicators = ["अनुक्रमित छैन", "उपलब्ध छैन", "उल्लेख छैन", "जानकारी छैन", "माफ गर्नुहोस्"]
            rejected = any(ind in llm_answer for ind in rejection_indicators)
            if rejected:
                negative_queries_passed += 1

        # Evaluate Answer Facts
        fact_matches = 0
        normalized_answer = normalize_digits(llm_answer).lower()
        for fact in expected_facts:
            norm_fact = normalize_digits(fact).lower()
            if norm_fact in normalized_answer:
                fact_matches += 1

        fact_recall = (fact_matches / len(expected_facts)) if expected_facts else 1.0
        fact_recall_total += fact_recall
        llm_tested_count += 1

        status = "✅ PASS" if (doc_hit_5 and fact_recall >= 0.5) or (is_not_found and rejected) else "❌ FAIL"

        print(
            f"{qid:<32} | "
            f"{('YES' if doc_hit_1 else 'NO') if not is_not_found else 'N/A':<9} | "
            f"{('YES' if doc_hit_5 else 'NO') if not is_not_found else 'N/A':<9} | "
            f"{f'{mrr:.2f}' if not is_not_found else 'N/A':<6} | "
            f"{f'{fact_recall*100:.0f}%':<7} | "
            f"{status}"
        )

        results_log.append({
            "id": qid,
            "category": item["category"],
            "query": raw_query,
            "cleaned_query": cleaned_query,
            "intent": intent,
            "is_not_found": is_not_found,
            "expected_decision_no": expected_dec,
            "expected_source": expected_source,
            "expected_facts": expected_facts,
            "retrieved_sources": retrieved_docs,
            "retrieved_decisions": retrieved_decs,
            "doc_hit_1": doc_hit_1,
            "doc_hit_3": doc_hit_3,
            "doc_hit_5": doc_hit_5,
            "mrr": mrr,
            "fact_recall": fact_recall,
            "llm_answer": llm_answer,
            "retrieval_time_sec": retrieval_time,
        })

    # Summary
    doc_hit_1_rate = (doc_hit_1_count / valid_retrieval_queries) * 100 if valid_retrieval_queries else 0.0
    doc_hit_3_rate = (doc_hit_3_count / valid_retrieval_queries) * 100 if valid_retrieval_queries else 0.0
    doc_hit_5_rate = (doc_hit_5_count / valid_retrieval_queries) * 100 if valid_retrieval_queries else 0.0
    avg_mrr = (mrr_total / valid_retrieval_queries) if valid_retrieval_queries else 0.0
    avg_fact_recall = (fact_recall_total / llm_tested_count) * 100 if llm_tested_count else 0.0
    negative_rejection_rate = (negative_queries_passed / negative_queries_total) * 100 if negative_queries_total else 0.0

    summary = {
        "total_queries": len(queries),
        "valid_retrieval_queries": valid_retrieval_queries,
        "negative_queries": negative_queries_total,
        "doc_hit_1_rate": doc_hit_1_rate,
        "doc_hit_3_rate": doc_hit_3_rate,
        "doc_hit_5_rate": doc_hit_5_rate,
        "mean_reciprocal_rank": avg_mrr,
        "avg_fact_recall": avg_fact_recall,
        "negative_rejection_rate": negative_rejection_rate,
        "results": results_log,
    }

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Test Queries           : {len(queries)}")
    print(f"Document Hit Rate @ 1        : {doc_hit_1_rate:.1f}% ({doc_hit_1_count}/{valid_retrieval_queries})")
    print(f"Document Hit Rate @ 3        : {doc_hit_3_rate:.1f}% ({doc_hit_3_count}/{valid_retrieval_queries})")
    print(f"Document Hit Rate @ 5        : {doc_hit_5_rate:.1f}% ({doc_hit_5_count}/{valid_retrieval_queries})")
    print(f"Mean Reciprocal Rank (MRR)   : {avg_mrr:.4f}")
    print(f"Average Fact Recall (LLM)    : {avg_fact_recall:.1f}%")
    print(f"Negative Rejection Accuracy  : {negative_rejection_rate:.1f}% ({negative_queries_passed}/{negative_queries_total})")
    print("=" * 80)

    out_full = os.path.join(BASE_DIR, output_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    with open(out_full, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed benchmark results to: {out_full}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/baseline_results.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()
    run_evaluation(output_path=args.output, top_k=args.top_k, run_llm=not args.no_llm)
