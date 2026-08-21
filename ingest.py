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


def extract_metadata_from_full_text(full_text: str, file_name: str) -> dict:
    """
    Extract case metadata from the entire document text.
    This is more robust than only scanning the header.
    """
    text = clean_devanagari_text(full_text)
    norm = normalize_digits(text)

    # ---- Decision Number ----
    decision_no = _first_match([
        r"निर्णय\s*नं\.?\s*([0-9]+)",
        r"Decision\s*(?:No\.?|Number)\s*[:.-]?\s*([0-9]+)",
    ], norm)

    # ---- Date ----
    # Look for both "फैसला मिति" and "आदेश मिति"
    date = _first_match([
        r"फैसला\s*मिति\s*[:：\-]?\s*([0-9]{3,4}[./\-।][0-9]{1,2}[./\-।][0-9]{1,2})",
        r"आदेश\s*मिति\s*[:：\-]?\s*([0-9]{3,4}[./\-।][0-9]{1,2}[./\-।][0-9]{1,2})",
    ], norm)

    # ---- Subject / Case Type ----
    # Try to extract from "विषय" line
    subject = _first_match([
        r"विषय\s*[ः:：-]?\s*([^\n|]+)",
    ], text)

    # If subject is still empty, try to get it from the first occurrence of "उत्प्रेषण", "परमादेश", etc.
    if not subject or subject == "हुने":
        case_type_keywords = ["उत्प्रेषण", "परमादेश", "बन्दीप्रत्यक्षीकरण", "अर्जी", "निवेदन"]
        for kw in case_type_keywords:
            if kw in text:
                subject = kw
                break

    # ---- Court ----
    court = "सर्वोच्च अदालत" if "सर्वोच्च अदालत" in text else ""

    # ---- Parties ----
    # Search the entire text for petitioner/respondent patterns
    appellant = _first_match([
        r"पुनरावेदक/विपक्षी\s*[:：-]\s*([^\n]+)",
        r"निवेदक\s*[:：-]\s*([^\n]+)",
        r"पुनरावेदक\s*[:：-]\s*([^\n]+)",
    ], text)
    respondent = _first_match([
        r"प्रत्यर्थी/निवेदक\s*[:：-]\s*([^\n]+)",
        r"विपक्षी\s*[:：-]\s*([^\n]+)",
        r"प्रत्यर्थी\s*[:：-]\s*([^\n]+)",
    ], text)

    # ---- Lawyers ----
    appellant_lawyer = _first_match([
        r"पुनरावेदकका\s*तर्फबाट\s*[:：-]?\s*([^\n]+)",
        r"निवेदकका\s*तर्फबाट\s*[:：-]?\s*([^\n]+)",
    ], text)
    respondent_lawyer = _first_match([
        r"प्रत्यर्थीका\s*तर्फबाट\s*[:：-]?\s*([^\n]+)",
        r"विपक्षीका\s*तर्फबाट\s*[:：-]?\s*([^\n]+)",
    ], text)

    # ---- Judges ----
    # Look for patterns that appear anywhere: "प्रधानन्यायाधीश", "न्यायाधीश"
    chief_justice = _first_match([
        r"सम्माननीय\s*(?:का\.मु\.\s*)?प्रधानन्यायाधीश\s*श्री\s*([^\n]+)",
        r"प्रधानन्यायाधीश\s*श्री\s*([^\n]+)",
    ], text)

    # For other judges, capture all lines containing "माननीय न्यायाधीश" or "न्यायाधीश श्री"
    judge_lines = re.findall(r"माननीय\s*न्यायाधीश\s*श्री\s*([^\n]+)", text)
    if not judge_lines:
        judge_lines = re.findall(r"न्यायाधीश\s*श्री\s*([^\n]+)", text)

    # Filter out OCR noise
    judge_lines = [line for line in judge_lines if "का.मु.प्" not in line]

    cleaned_judges = []
    for line in judge_lines:
        for part in re.split(r"\s*र\s*|\s*,\s*", line):
            part = part.strip()
            if part and len(part) > 2:
                part = re.sub(r",.*$", "", part)
                cleaned_judges.append(part)

    judges_list = []
    if chief_justice:
        judges_list.append(f"प्रधानन्यायाधीश {chief_justice}")
    judges_list.extend(cleaned_judges)
    judges_str = ", ".join(judges_list) if judges_list else ""

    # ---- Legal Provisions ----
    # Find all unique provisions mentions
    provisions = re.findall(r"(?:दफा|धारा)\s*[०-९0-9]+(?:\s*\([^)]+\))?", text)
    provisions = list(dict.fromkeys(provisions))  # unique
    provisions_str = ", ".join(provisions)

    # ---- Case ID ----
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
        "appellant_lawyer": appellant_lawyer,
        "respondent_lawyer": respondent_lawyer,
        "judges": judges_str,
        "provisions": provisions_str,
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

            # ---- Extract FULL document text for metadata ----
            full_text_parts = []
            for pno in range(total_pages):
                p = doc[pno]
                native = p.get_text().strip()
                if not is_valid_devanagari_text(native, min_ratio=0.4):
                    native = ocr_scanned_page(p, lang_flag)
                full_text_parts.append(clean_devanagari_text(native))
            full_text = "\n\n".join(full_text_parts)

            # Extract metadata from the entire text
            case_meta = extract_metadata_from_full_text(full_text, file_name)

            print(
                f"  case_id={case_meta['case_id']} decision={case_meta['decision_no']} "
                f"date={case_meta['date']} judges={case_meta['judges']} "
                f"provisions={case_meta['provisions']}",
                flush=True,
            )

            # Create a dedicated header summary chunk (page 0)
            header_summary = (
                f"CASE_ID: {case_meta['case_id']}\n"
                f"निर्णय नं.: {case_meta['decision_no']}\n"
                f"मिति: {case_meta['date']}\n"
                f"अदालत: {case_meta['court']}\n"
                f"विषय: {case_meta['subject']}\n"
                f"मुद्दाको प्रकार: {case_meta.get('case_type', '')}\n"
                f"न्यायाधीश: {case_meta['judges']}\n"
                f"पुनरावेदक/निवेदक: {case_meta['parties'].get('appellant', '')}\n"
                f"प्रत्यर्थी/विपक्षी: {case_meta['parties'].get('respondent', '')}\n"
                f"पुनरावेदकका कानून व्यवसायी: {case_meta['appellant_lawyer']}\n"
                f"प्रत्यर्थीका कानून व्यवसायी: {case_meta['respondent_lawyer']}\n"
                f"प्रमुख कानूनी प्रावधान: {case_meta['provisions']}"
            )
            all_chunks.append(header_summary)
            chunk_metadata.append({
                "source": file_name,
                "page": 0,  # header page
                "total_pages": total_pages,
                "content": header_summary,
                "is_header": True,
                **case_meta,
            })

            # Process each page for regular chunks
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
                        f"[COURT={case_meta['court'] or 'UNKNOWN'}] "
                        f"[CASE_TYPE={case_meta.get('case_type', '') or 'UNKNOWN'}] "
                        f"[JUDGES={case_meta['judges'] or 'UNKNOWN'}] "
                        f"[APPELLANT={case_meta['parties'].get('appellant', 'UNKNOWN')}] "
                        f"[RESPONDENT={case_meta['parties'].get('respondent', 'UNKNOWN')}] "
                        f"[APPELLANT_LAWYER={case_meta['appellant_lawyer'] or 'UNKNOWN'}] "
                        f"[RESPONDENT_LAWYER={case_meta['respondent_lawyer'] or 'UNKNOWN'}] "
                        f"[PROVISIONS={case_meta['provisions'] or 'UNKNOWN'}] "
                        f"[SOURCE={file_name}] [PAGE={page_num + 1}]\n{chunk}"
                    )
                    all_chunks.append(searchable_chunk)
                    chunk_metadata.append({
                        "source": file_name,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "content": chunk,
                        "is_header": False,
                        **case_meta,
                    })
            doc.close()
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not all_chunks:
        print("No valid text chunks extracted.")
        return

    # ---- Embedding and Indexing ----
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
            "appellant_lawyer": meta.get("appellant_lawyer", ""),
            "respondent_lawyer": meta.get("respondent_lawyer", ""),
            "judges": meta.get("judges", ""),
            "provisions": meta.get("provisions", ""),
            "is_header": meta.get("is_header", False),
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

    # ---- Save metadata summary ----
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
        cid = m["case_id"]
        if cid not in summary["case_metadata"]:
            summary["case_metadata"][cid] = {
                "source": m["source"],
                "decision_no": m["decision_no"],
                "date": m["date"],
                "subject": m["subject"],
                "court": m["court"],
                "parties": m["parties"],
                "judges": m.get("judges", ""),
                "appellant_lawyer": m.get("appellant_lawyer", ""),
                "respondent_lawyer": m.get("respondent_lawyer", ""),
                "provisions": m.get("provisions", ""),
            }
    with open("./models/ingest_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Ingestion completed: {len(valid_files)} files, {len(all_chunks)} chunks")


if __name__ == "__main__":
    process_local_documents()