# test_rag.py
import sys
import os
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from src.text_processor import clean_devanagari_text, clean_and_repair_nepali_output
from src.hybrid_search import perform_hybrid_search
from src.llm_generator import generate_nepali_answer

sys.stdout.reconfigure(encoding='utf-8')

def run_test():
    print("--- 1. Testing OCR & ASCII Noise Cleaner ---")
    noisy_text = " , २०६३ को धारा ८८ मा GAH] व्यक्तिहरूलाई ... dal सृटढीकरणमा ~^\\| \ufffd"
    cleaned = clean_devanagari_text(noisy_text)
    repaired = clean_and_repair_nepali_output(noisy_text)
    print(f"Original Noise : {noisy_text}")
    print(f"Cleaned Text   : {cleaned}")
    print(f"Repaired Output: {repaired}")

    print("\n--- 2. Loading search engines ---")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="nepali_legal_docs")
    print(f"ChromaDB total count: {collection.count()}")

    model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")

    with open("./models/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    bm25 = bm25_data["bm25"]
    chunk_metadata = bm25_data["metadata"]

    test_queries = [
        "पूर्वपदाधिकारीहरूलाई सुविधा तथा सुरक्षा प्रदान गर्ने अध्यादेश",
        "नेपालको अन्तरिम संविधान २०६३ को धारा ८८ बमोजिम अध्यादेश"
    ]

    for i, raw_query in enumerate(test_queries, 1):
        print(f"\n==========================================")
        print(f"Query #{i}: {raw_query}")
        print(f"==========================================")
        cleaned_query = clean_devanagari_text(raw_query)

        search_results = perform_hybrid_search(
            query=cleaned_query,
            collection=collection,
            model=model,
            bm25=bm25,
            chunk_metadata=chunk_metadata,
            top_k=2,
            alpha=0.8
        )

        print(f"\nRetrieved Top {len(search_results)} chunks:")
        for r_idx, item in enumerate(search_results, 1):
            print(f"[{r_idx}] Score: {item['score']:.4f} | Source: {item['source']} (Page {item['page']} of {item.get('total_pages', '?')})")
            print(f"    Snippet         : {item['content'][:120]}...")
            print(f"    Parent Page Len : {len(item.get('page_text', ''))} chars")

        print("\n--- Generating LLM Response via Groq ---")
        answer = generate_nepali_answer(cleaned_query, search_results, model_name="openai/gpt-oss-20b")
        print("\n💡 AI Response:")
        print(answer)

if __name__ == "__main__":
    run_test()
