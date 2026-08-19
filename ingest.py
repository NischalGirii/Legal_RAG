import os
import sys
import re
import pickle
import json
import datetime
import fitz
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.ocr_engine import ocr_scanned_page, resolve_ocr_lang_flag
from src.text_processor import (
    is_valid_devanagari_text,
    clean_devanagari_text,
    chunk_text_by_sentences,
    char_ngram_tokenize,
    normalize_digits,
)


NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def collect_pdf_files(target_path: str) -> list[str]:
    if not os.path.exists(target_path):
        return []
    if os.path.isfile(target_path):
        return [target_path] if target_path.lower().endswith(".pdf") else []
    pdf_files = []
    for root, _, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return sorted(pdf_files)


def _first_match(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text, re.I | re.M)
        if m:
            return m.group(1).strip()
    return ""


def extract_case_metadata(page_text: str, file_name: str) -> dict:
    """Extract legal identity from the first pages of an NKP decision."""
    text = clean_devanagari_text(page_text)
    norm = normalize_digits(text)

    decision_no = _first_match([
        r"निर्णय\s*नं\.?\s*([0-9]+)",
        r"Decision\s*(?:No\.?|Number)\s*[:.-]?\s*([0-9]+)",
    ], norm)
    date = _first_match([
        r"फैसला\s*मिति\s*[:：\-]?\s*([0-9]{3,4}[./\-।][0-9]{1,2}[./\-।][0-9]{1,2})",
        r"आदेश\s*मिति\s*[:：\-]?\s*([0-9]{3,4}[./\-।][0-9]{1,2}[./\-।][0-9]{1,2})",
    ], norm)
    subject = _first_match([
        r"विषय\s*[ः:：-]?\s*([^\n|]+)",
    ], text)
    court = "सर्वोच्च अदालत" if "सर्वोच्च अदालत" in text else ""

    appellant = _first_match([
        r"पुनरावेदक/विपक्षी\s*[:：-]\s*([^\n]+)",
        r"निवेदक\s*[:：-]\s*([^\n]+)",
    ], text)
    respondent = _first_match([
        r"प्रत्यर्थी/निवेदक\s*[:：-]\s*([^\n]+)",
        r"विपक्षी\s*[:：-]\s*([^\n]+)",
    ], text)

    # Filename remains a secondary identifier, not the legal case number.
    stem = os.path.splitext(file_name)[0]
    case_id = f"decision_{decision_no}" if decision_no else f"file_{stem}"

    return {
        "case_id": case_id,
        "decision_no": decision_no,
        "date": date,
        "subject": subject,
        "court": court,
        "parties": {
            "appellant": appellant,
            "respondent": respondent,
        },
    }


def process_local_documents(target_path: str = None):
    if target_path is None:
        target_path = sys.argv[1] if len(sys.argv) > 1 else "./data/pdf"

    valid_files = collect_pdf_files(target_path)
    if not valid_files:
        print(f"No PDF files found at '{target_path}'.")
        return

    lang_flag = resolve_ocr_lang_flag()
    model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        chroma_client.delete_collection(name="nepali_legal_docs")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name="nepali_legal_docs",
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    chunk_metadata = []
    total_pages_processed = 0

    for file_idx, doc_path in enumerate(valid_files, 1):
        file_name = os.path.basename(doc_path)
        print(f"[{file_idx}/{len(valid_files)}] {file_name}", flush=True)
        try:
            doc = fitz.open(doc_path)
            total_pages = len(doc)
            total_pages_processed += total_pages

            # First two pages generally contain NKP case identity.
            header_text_parts = []
            for pno in range(min(2, total_pages)):
                p = doc[pno]
                native = p.get_text().strip()
                if not is_valid_devanagari_text(native, min_ratio=0.4):
                    native = ocr_scanned_page(p, lang_flag)
                header_text_parts.append(clean_devanagari_text(native))
            case_meta = extract_case_metadata("\n".join(header_text_parts), file_name)

            print(
                f"  case_id={case_meta['case_id']} decision={case_meta['decision_no']} "
                f"date={case_meta['date']} subject={case_meta['subject']}",
                flush=True,
            )

            for page_num in range(total_pages):
                page = doc[page_num]
                native_text = page.get_text().strip()
                if not is_valid_devanagari_text(native_text, min_ratio=0.4):
                    raw_text = ocr_scanned_page(page, lang_flag)
                else:
                    raw_text = native_text

                cleaned_page_text = clean_devanagari_text(raw_text)
                chunks = chunk_text_by_sentences(cleaned_page_text, max_chars=600, overlap_sentences=1)
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    searchable_chunk = (
                        f"[CASE_ID={case_meta['case_id']}] "
                        f"[DECISION_NO={case_meta['decision_no'] or 'UNKNOWN'}] "
                        f"[DATE={case_meta['date'] or 'UNKNOWN'}] "
                        f"[SUBJECT={case_meta['subject'] or 'UNKNOWN'}] "
                        f"[SOURCE={file_name}] [PAGE={page_num + 1}]\n{chunk}"
                    )
                    all_chunks.append(searchable_chunk)
                    chunk_metadata.append({
                        "source": file_name,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "content": chunk,
                        "searchable_chunk": searchable_chunk,
                        "page_text": cleaned_page_text,
                        **case_meta,
                    })
            doc.close()
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not all_chunks:
        print("No valid text chunks extracted.")
        return

    embeddings = model.encode(
        all_chunks, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    ids = [f"doc_chunk_{i}" for i in range(len(all_chunks))]
    metadatas = []
    for meta in chunk_metadata:
        metadatas.append({
            "source": meta["source"],
            "page": meta["page"],
            "total_pages": meta["total_pages"],
            "case_id": meta["case_id"],
            "decision_no": meta["decision_no"],
            "date": meta["date"],
            "subject": meta["subject"],
            "court": meta["court"],
            "appellant": meta["parties"].get("appellant", ""),
            "respondent": meta["parties"].get("respondent", ""),
        })

    batch_size = 2000
    for i in range(0, len(ids), batch_size):
        j = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:j],
            embeddings=embeddings[i:j],
            metadatas=metadatas[i:j],
            documents=all_chunks[i:j],
        )

    tokenized_corpus = [char_ngram_tokenize(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    os.makedirs("./models", exist_ok=True)
    with open("./models/bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": chunk_metadata}, f)

    summary = {
        "last_ingested": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(valid_files),
        "total_pages": total_pages_processed,
        "total_chunks": len(all_chunks),
        "files": [os.path.basename(f) for f in valid_files],
        "cases": sorted({m["case_id"] for m in chunk_metadata}),
        "case_metadata": {},
    }
    for m in chunk_metadata:
        summary["case_metadata"][m["case_id"]] = {
            "source": m["source"],
            "decision_no": m["decision_no"],
            "date": m["date"],
            "subject": m["subject"],
            "court": m["court"],
            "parties": m["parties"],
        }
    with open("./models/ingest_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Ingestion completed: {len(valid_files)} files, {len(all_chunks)} chunks")


if __name__ == "__main__":
    process_local_documents()