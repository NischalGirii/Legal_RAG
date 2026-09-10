"""
Golden-set evaluation for the Nepali legal RAG pipeline.

Computes:
  • Recall@k     — is a chunk from the expected decision in the top-k?
  • Precision@k  — fraction of top-k chunks that belong to the expected case
  • Faithfulness — LLM-as-judge: is every claim in the answer supported by context?

Usage:
  python evaluate.py evaluation/golden_dataset.json --k 5
"""
import os
import sys
import json
import argparse
import pickle
from statistics import mean

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

from src.config import (
    CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_PATH, BM25_INDEX_PATH,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_FALLBACK, LLM_MODEL,
)
from src.sparse_index import retriever_from_pickle
from src.hybrid_search import perform_hybrid_search, extract_query_identifiers
from src.llm_generator import generate_nepali_answer, get_groq_client
from src.text_processor import clean_devanagari_text

load_dotenv()


def load_engines():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    bm25 = retriever_from_pickle(data)
    chunk_metadata = data["metadata"]
    try:
        ce = CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception:
        ce = CrossEncoder(CROSS_ENCODER_FALLBACK)
    return collection, model, bm25, chunk_metadata, ce


def recall_and_precision(results, golden, k):
    top = results[:k]
    exp_dec = str(golden.get("expected_decision_no") or "")
    exp_case = golden.get("expected_case_id")
    if not exp_dec and not exp_case:
        return None, None

    hits = 0
    for r in top:
        if exp_dec and str(r.get("decision_no") or "") == exp_dec:
            hits += 1
        elif exp_case and r.get("case_id") == exp_case:
            hits += 1

    recall = 1.0 if hits > 0 else 0.0
    precision = hits / max(len(top), 1)
    return recall, precision


FAITHFULNESS_PROMPT = """तपाईं एक कडा मूल्याङ्कनकर्ता हुनुहुन्छ। तल "प्रमाण" र "उत्तर" दिइएको छ।
प्रश्न: के उत्तरको प्रत्येक दाबी प्रमाणले समर्थन गर्दछ?

केवल "yes" वा "no" मा उत्तर दिनुहोस्।
- "yes" = उत्तरका सबै दाबी प्रमाणमा प्रत्यक्ष रूपमा समर्थित छन्।
- "no"  = उत्तरमा प्रमाण बाहिरको कुनै दाबी, अनुमान, वा मिति/नाम/दफा थपिएको छ।

प्रमाण:
{context}

उत्तर:
{answer}
"""


def judge_faithfulness(answer, context_chunks):
    client = get_groq_client()
    if not client or not answer:
        return None
    ctx = "\n\n".join(c.get("content", "")[:800] for c in context_chunks[:5])
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You output only 'yes' or 'no'."},
                {"role": "user", "content": FAITHFULNESS_PROMPT.format(context=ctx, answer=answer)},
            ],
            temperature=0,
            max_tokens=8,
        )
        verdict = (resp.choices[0].message.content or "").strip().lower()
        return 1.0 if verdict.startswith("yes") else 0.0
    except Exception as e:
        print(f"  [judge error] {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Path to golden_dataset.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--skip-faithfulness", action="store_true")
    args = ap.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        golden = json.load(f)

    print(f"Loading engines… ({len(golden)} queries, k={args.k})")
    collection, model, bm25, chunk_metadata, ce = load_engines()

    recalls, precisions, faithful = [], [], []
    for i, item in enumerate(golden, 1):
        query = clean_devanagari_text(item["query"])
        ids = extract_query_identifiers(query)

        results = perform_hybrid_search(
            query=query, collection=collection, model=model, bm25=bm25,
            chunk_metadata=chunk_metadata, top_k=15, alpha=0.7,
            identifiers=ids,
        )
        if results:
            pairs = [[query, r["content"]] for r in results]
            scores = ce.predict(pairs)
            for r, sc in zip(results, scores):
                r["score"] = float(sc)
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:args.k]

        r_at_k, p_at_k = recall_and_precision(results, item, args.k)
        if r_at_k is not None:
            recalls.append(r_at_k)
            precisions.append(p_at_k)

        answer = generate_nepali_answer(
            query=query, retrieved_items=results, model_name=LLM_MODEL,
            current_case=None, metadata_info=None, chunk_metadata=chunk_metadata,
        )
        answer = str(answer) if answer else ""

        f_score = None
        if not args.skip_faithfulness:
            f_score = judge_faithfulness(answer, results)
            if f_score is not None:
                faithful.append(f_score)

        print(
            f"[{i:02d}/{len(golden)}] R@{args.k}={r_at_k} P@{args.k}={p_at_k:.2f} "
            f"Faith={f_score}  Q: {query[:55]}…"
        )

    print("\n=== Summary ===")
    if recalls:
        print(f"Recall@{args.k}:      {mean(recalls):.3f}  ({sum(recalls):.0f}/{len(recalls)})")
        print(f"Precision@{args.k}:   {mean(precisions):.3f}")
    if faithful:
        print(f"Faithfulness:    {mean(faithful):.3f}  ({sum(faithful):.0f}/{len(faithful)})")


if __name__ == "__main__":
    main()