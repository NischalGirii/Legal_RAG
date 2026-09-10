import os
import sys
import argparse
import pickle
import json
import datetime
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_PATH,
    BM25_INDEX_PATH,
    CASE_INDEX_PATH,
    INGEST_METADATA_PATH,
    EXTRACT_CACHE_DIR,
    CHUNK_DOCS_PATH,
    EMBED_BATCH_SIZE,
    CHROMA_UPSERT_BATCH,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    PRAKARAN_MAX_CHARS,
    PRAKARAN_OVERLAP_CHARS,
    METADATA_LLM_MODEL,
    INGEST_WORKERS,
    ensure_models_dir,
)
from src.ocr_engine import ocr_scanned_page, resolve_ocr_lang_flag
from src.text_processor import (
    is_valid_devanagari_text,
    clean_devanagari_text,
    chunk_text_by_sentences,
    char_ngram_tokenize,
    normalize_digits,
    chunk_by_prakaran,
)
from src.hybrid_search import reset_lookup_indexes
from src.sparse_index import SparseBM25

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_groq_client = None
_llm_lock = threading.Lock()


def get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp")
_SUPPORTED_UPLOAD_EXTS = (".pdf", ".txt") + _IMAGE_EXTS


def collect_source_files(target_path: str) -> list[str]:
    if not os.path.exists(target_path):
        return []
    if os.path.isfile(target_path):
        return [target_path] if target_path.lower().endswith((".pdf", ".txt")) else []
    source_files = []
    for root, _, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith((".pdf", ".txt")):
                source_files.append(os.path.join(root, file))
    return sorted(source_files)


def file_fingerprint(path: str) -> dict:
    stat = os.stat(path)
    return {
        "name": os.path.basename(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def _first_match(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text, re.I | re.M)
        if m:
            return m.group(1).strip()
    return ""


def extract_metadata_with_llm(full_text: str, file_name: str) -> dict:
    groq_client = get_groq_client()
    norm = normalize_digits(full_text)
    decision_no = _first_match([
        r"निर्णय\s*नं\.?\s*([0-9]+)",
        r"निर्णय\s*([0-9]{4})",
        r"Decision\s*(?:No\.?|Number)\s*[:.-]?\s*([0-9]+)",
    ], norm)
    if not decision_no:
        match = re.search(r"nkp[_\s-]*([0-9]+)[_\s-]*", file_name, re.I)
        if match:
            decision_no = match.group(1)

    date = _first_match([
        r"फैसला\s*मिति\s*[:：\-]?\s*([0-9]{3,4}[./\-][0-9]{1,2}[./\-][0-9]{1,2})",
        r"आदेश\s*मिति\s*[:：\-]?\s*([0-9]{3,4}[./\-][0-9]{1,2}[./\-][0-9]{1,2})",
        r"मिति\s*[:：\-]?\s*([0-9]{3,4}[./\-][0-9]{1,2}[./\-][0-9]{1,2})",
    ], norm)

    subject = _first_match([
        r"विषय\s*[ः:：-]\s*([^\n|।]+)",
        r"मुद्दाको\s*प्रकार\s*[ः:：-]\s*([^\n|।]+)",
    ], full_text)
    if not subject or subject == "हुने":
        case_types = ["उत्प्रेषण", "परमादेश", "बन्दीप्रत्यक्षीकरण", "नागरिकता", "अंशबण्डा", "सम्पत्ति रोक्का"]
        for ct in case_types:
            if ct in full_text:
                subject = ct
                break

    court = "सर्वोच्च अदालत" if "सर्वोच्च अदालत" in full_text else ""

    appellant = _first_match([
        r"(?:पुनरावेदक|निवेदक)\s*[ः:：-]\s*([^\n]+)",
        r"(?:पुनरावेदक/विपक्षी)\s*[ः:：-]\s*([^\n]+)",
        r"(?:पुनरावेदक\s*M)\s*([^\n]+)",
    ], full_text)
    respondent = _first_match([
        r"(?:प्रत्यर्थी|विपक्षी)\s*[ः:：-]\s*([^\n]+)",
        r"(?:प्रत्यर्थी/निवेदक)\s*[ः:：-]\s*([^\n]+)",
        r"(?:प्रत्यर्थी\s*M)\s*([^\n]+)",
    ], full_text)

    appellant_lawyer = _first_match([
        r"(?:पुनरावेदक|निवेदक)का\s*(?:तर्फबाट|कानून व्यवसायी)\s*[ः:：-]?\s*([^\n]+)",
        r"पुनरावेदक/विपक्षीका\s*तर्फबाट\s*[ः:：-]?\s*([^\n]+)",
    ], full_text)
    respondent_lawyer = _first_match([
        r"(?:प्रत्यर्थी|विपक्षी)का\s*(?:तर्फबाट|कानून व्यवसायी)\s*[ः:：-]?\s*([^\n]+)",
        r"प्रत्यर्थी/निवेदकका\s*तर्फबाट\s*[ः:：-]?\s*([^\n]+)",
    ], full_text)

    chief_justice = _first_match([
        r"(?:सम्माननीय\s*)?(?:का\.मु\.\s*)?प्रधानन्यायाधीश\s*श्री\s*([^\n]+)",
        r"प्रधानन्यायाधीश\s*श्री\s*([^\n]+)",
    ], full_text)
    other_judges = re.findall(r"माननीय\s*न्यायाधीश\s*श्री\s*([^\n]+)", full_text)
    if not other_judges:
        other_judges = re.findall(r"न्यायाधीश\s*श्री\s*([^\n]+)", full_text)
    other_judges = [j.strip() for j in other_judges if "का.मु." not in j and len(j) > 2]
    judges = []
    if chief_justice:
        judges.append(f"प्रधानन्यायाधीश {chief_justice.strip()}")
    judges.extend(other_judges)
    judges_str = ", ".join(judges) if judges else ""

    provisions = re.findall(r"(?:दफा|धारा|अ\.बं\.)\s*[०-९0-9]+(?:\s*\([^)]+\))?", full_text)
    provisions = list(dict.fromkeys(provisions))
    provisions_str = ", ".join(provisions)

    final_order = ""
    legal_principle = ""
    precedents = ""

    # Send Head (first 4,000 chars) + Tail (last 4,000 chars) for complete context
    if groq_client and len(full_text) > 200:
        if len(full_text) > 8000:
            sample_text = f"{full_text[:4000]}\n\n[...बीचको व्यहोरा...]\n\n{full_text[-4000:]}"
        else:
            sample_text = full_text

        prompt = f"""तलको नेपाली कानूनी फैसलाको प्रारम्भिक खण्ड र अन्तिम (फैसला/ठहर) खण्डको पाठ दिइएको छ।
यसलाई ध्यानपूर्वक पढेर ३ वटा कुराहरू निकाल्नुहोस्। उत्तर JSON format मा मात्र दिनुहोस्:
{{
  "final_order": "सर्वोच्च अदालतको अन्तिम आदेश वा फैसलाको मुख्य निष्कर्ष (जस्तै: आदेश सदर, पुनरावेदन खारेज, परमादेश जारी, आदि)",
  "legal_principle": "यस फैसलामा प्रतिपादन गरिएको मुख्य कानूनी सिद्धान्त वा नजिर",
  "precedents": "फैसलामा उल्लेख भएका अघिल्ला नजिरहरू"
}}
यदि कुनै विवरण स्पष्ट छैन भने खाली स्ट्रिङ "" दिनुहोस्।

फैसलाको पाठ:
{sample_text}
"""
        for attempt in range(3):
            try:
                with _llm_lock:
                    response = groq_client.chat.completions.create(
                        model=METADATA_LLM_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a legal metadata extraction assistant. You must output only a valid JSON object without any extra conversational text."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2048,  # Provides headroom for CoT reasoning tokens
                    )
                raw_text = response.choices[0].message.content or ""
                # Safely extract the JSON object, ignoring any preceding <think> tags or thoughts
                match = re.search(r"\{[\s\S]*\}", raw_text)
                clean_json = match.group(0) if match else raw_text.strip()

                result = json.loads(clean_json)
                final_order = result.get("final_order", "")
                legal_principle = result.get("legal_principle", "")
                precedents = result.get("precedents", "")
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"[LLM extraction] Attempt {attempt + 1} failed for {file_name}: {e} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"[LLM extraction] Failed for {file_name} after 3 attempts: {e}")

    # Fallback to tail of document for final order
    if not final_order:
        tail = full_text[-3500:]
        if "सदर हुने ठहर्छ" in tail or "सदर" in tail:
            final_order = "आदेश सदर"
        elif "खारेज हुने ठहर्छ" in tail or "खारेज" in tail:
            final_order = "पुनरावेदन/रिट खारेज"
        elif "परमादेश जारी" in tail:
            final_order = "परमादेश जारी"
        elif "बदर हुने" in tail or "बदर" in tail:
            final_order = "बदर"

    if not legal_principle:
        match = re.search(r"(?:सिद्धान्त|प्रतिपादन)\s*[:：]\s*([^\n।]+)", full_text)
        if match:
            legal_principle = match.group(1).strip()
    if not precedents:
        match = re.search(r"(?:नजीर|पूर्व\s*निर्णय)\s*[:：]\s*([^\n।]+)", full_text)
        if match:
            precedents = match.group(1).strip()

    case_id = f"decision_{decision_no}" if decision_no else f"file_{os.path.splitext(file_name)[0]}"

    return {
        "case_id": case_id,
        "decision_no": decision_no,
        "date": date,
        "subject": subject,
        "court": court,
        "parties": {"appellant": appellant, "respondent": respondent},
        "appellant_lawyer": appellant_lawyer,
        "respondent_lawyer": respondent_lawyer,
        "judges": judges_str,
        "provisions": provisions_str,
        "case_type": subject if subject else "",
        "final_order": final_order,
        "legal_principle": legal_principle,
        "precedents": precedents,
    }


def _cache_file_path(file_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name)
    return os.path.join(EXTRACT_CACHE_DIR, f"{safe}.json")


def load_extract_cache(file_name: str, fingerprint: dict) -> dict | None:
    path = _cache_file_path(file_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        fp = cached.get("fingerprint") or {}
        if fp.get("mtime") == fingerprint["mtime"] and fp.get("size") == fingerprint["size"]:
            return cached
    except Exception:
        return None
    return None


def save_extract_cache(file_name: str, fingerprint: dict, pages: list[str], case_meta: dict) -> None:
    ensure_models_dir()
    path = _cache_file_path(file_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"fingerprint": fingerprint, "pages": pages, "case_meta": case_meta},
            fh,
            ensure_ascii=False,
        )


def _header_summary(case_meta: dict) -> str:
    return (
        f"CASE_ID: {case_meta['case_id']}\n"
        f"निर्णय नं.: {case_meta['decision_no']}\n"
        f"मिति: {case_meta['date']}\n"
        f"अदालत: {case_meta['court']}\n"
        f"विषय: {case_meta['subject']}\n"
        f"मुद्दाको प्रकार: {case_meta.get('case_type', '')}\n"
        f"न्यायाधीश: {case_meta['judges']}\n"
        f"पुनरावेदक/विपक्षी: {case_meta['parties'].get('appellant', '')}\n"
        f"प्रत्यर्थी/निवेदक: {case_meta['parties'].get('respondent', '')}\n"
        f"पुनरावेदकका कानून व्यवसायी: {case_meta['appellant_lawyer']}\n"
        f"प्रत्यर्थीका कानून व्यवसायी: {case_meta['respondent_lawyer']}\n"
        f"प्रमुख कानूनी प्रावधान: {case_meta['provisions']}\n"
        f"अन्तिम आदेश / फैसला: {case_meta['final_order']}\n"
        f"मुख्य कानूनी सिद्धान्त: {case_meta['legal_principle']}\n"
        f"अघिल्ला नजिरहरू: {case_meta['precedents']}"
    )


def _searchable_prefix(case_meta: dict, file_name: str, page: int, prakaran: str | None = None) -> str:
    extra = f" [PRAKARAN={prakaran}]" if prakaran else ""
    return (
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
        f"[FINAL_ORDER={case_meta['final_order'] or 'UNKNOWN'}] "
        f"[LEGAL_PRINCIPLE={case_meta['legal_principle'] or 'UNKNOWN'}] "
        f"[PRECEDENTS={case_meta['precedents'] or 'UNKNOWN'}] "
        f"[SOURCE={file_name}] [PAGE={page}]{extra}\n"
    )


def _append_page_chunks(all_chunks, chunk_metadata, file_name, page_num, total_pages, cleaned_page_text, case_meta):
    # Prioritise section-aware (prakaran) chunking with a larger budget so
    # complete legal arguments stay intact. Fall back to sentence chunking
    # only when the page has no numbered sections.
    prakaran_chunks = chunk_by_prakaran(
        cleaned_page_text,
        max_chars=PRAKARAN_MAX_CHARS,
        overlap_chars=PRAKARAN_OVERLAP_CHARS,
    )
    if prakaran_chunks:
        for chunk_text, p_no in prakaran_chunks:
            if not chunk_text.strip():
                continue
            all_chunks.append(_searchable_prefix(case_meta, file_name, page_num, p_no) + chunk_text)
            chunk_metadata.append({
                "source": file_name, "page": page_num, "total_pages": total_pages,
                "content": chunk_text, "is_header": False,
                "prakaran_no": p_no, **case_meta,
            })
        return
    chunks = chunk_text_by_sentences(cleaned_page_text, max_chars=CHUNK_MAX_CHARS, overlap_sentences=2)
    for chunk in chunks:
        if not chunk.strip():
            continue
        all_chunks.append(_searchable_prefix(case_meta, file_name, page_num) + chunk)
        chunk_metadata.append({
            "source": file_name, "page": page_num, "total_pages": total_pages,
            "content": chunk, "is_header": False, **case_meta,
        })

def _extract_image_text(doc_path: str, lang_flag: str) -> str:
    try:
        doc = fitz.open(doc_path)
        page = doc[0]
        native = page.get_text().strip()
        if not is_valid_devanagari_text(native, min_ratio=0.1):
            native = ocr_scanned_page(page, lang_flag)
        doc.close()
        return clean_devanagari_text(native)
    except Exception:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(doc_path).convert("RGB")
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="PDF")
            buf.seek(0)
            doc = fitz.open(stream=buf.read(), filetype="pdf")
            page = doc[0]
            native = ocr_scanned_page(page, lang_flag)
            doc.close()
            return clean_devanagari_text(native)
        except Exception as e:
            print(f"  [image OCR fallback failed] {e}")
            return ""


def extract_pages(doc_path: str, lang_flag: str) -> tuple[list[str], dict]:
    file_name = os.path.basename(doc_path)
    fingerprint = file_fingerprint(doc_path)
    cached = load_extract_cache(file_name, fingerprint)
    if cached:
        print(f"  cache hit ({file_name})", flush=True)
        return cached["pages"], cached["case_meta"]

    pages = []
    if file_name.lower().endswith(".pdf"):
        doc = fitz.open(doc_path)
        for pno in range(len(doc)):
            native = doc[pno].get_text("text").strip()
            if not is_valid_devanagari_text(native, min_ratio=0.3):
                native = ocr_scanned_page(doc[pno], lang_flag)
            pages.append(clean_devanagari_text(native))
        doc.close()
    elif file_name.lower().endswith(_IMAGE_EXTS):
        text = _extract_image_text(doc_path, lang_flag)
        pages = [text] if text.strip() else [""]
    else:
        with open(doc_path, "r", encoding="utf-8") as fh:
            pages = [clean_devanagari_text(fh.read())]

    case_meta = extract_metadata_with_llm("\n\n".join(pages), file_name)
    save_extract_cache(file_name, fingerprint, pages, case_meta)
    return pages, case_meta


def chunks_from_pages(file_name: str, pages: list[str], case_meta: dict) -> tuple[list[str], list[dict]]:
    all_chunks = []
    chunk_metadata = []
    total_pages = len(pages)
    header_summary = _header_summary(case_meta)
    all_chunks.append(header_summary)
    chunk_metadata.append({
        "source": file_name, "page": 0, "total_pages": total_pages,
        "content": header_summary, "is_header": True, **case_meta,
    })
    for page_num, cleaned_page_text in enumerate(pages, start=1):
        _append_page_chunks(all_chunks, chunk_metadata, file_name, page_num, total_pages, cleaned_page_text, case_meta)
    return all_chunks, chunk_metadata


def chroma_row(meta: dict, chunk_index: int) -> dict:
    return {
        "source": meta["source"],
        "page": meta["page"],
        "total_pages": meta["total_pages"],
        "case_id": meta["case_id"],
        "decision_no": meta.get("decision_no") or "",
        "date": meta.get("date") or "",
        "subject": meta.get("subject") or "",
        "court": meta.get("court") or "",
        "appellant": meta.get("parties", {}).get("appellant", ""),
        "respondent": meta.get("parties", {}).get("respondent", ""),
        "appellant_lawyer": meta.get("appellant_lawyer") or "",
        "respondent_lawyer": meta.get("respondent_lawyer") or "",
        "judges": meta.get("judges") or "",
        "provisions": meta.get("provisions") or "",
        "final_order": meta.get("final_order") or "",
        "legal_principle": meta.get("legal_principle") or "",
        "precedents": meta.get("precedents") or "",
        "is_header": bool(meta.get("is_header", False)),
        "prakaran_no": str(meta.get("prakaran_no") or ""),
        "chunk_index": int(chunk_index),
    }


def load_chunk_documents() -> list[str] | None:
    if not os.path.exists(CHUNK_DOCS_PATH):
        return None
    documents = []
    with open(CHUNK_DOCS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def write_chunk_documents(documents: list[str]) -> None:
    ensure_models_dir()
    with open(CHUNK_DOCS_PATH, "w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_existing_index() -> tuple[list, list] | None:
    if not os.path.exists(BM25_INDEX_PATH):
        return [], []
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    metadata = data.get("metadata") or []
    documents = load_chunk_documents()
    if documents is None:
        documents = data.get("documents")
    if documents is None:
        return None
    if len(documents) != len(metadata):
        return None
    return metadata, documents


def persist_indexes(chunk_metadata: list, documents: list, fingerprints: list, total_pages: int, file_names: list[str]):
    ensure_models_dir()
    tokenized_corpus = [char_ngram_tokenize(chunk) for chunk in documents]
    sparse = SparseBM25.build(tokenized_corpus)
    write_chunk_documents(documents)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"sparse": sparse, "metadata": chunk_metadata}, f)

    case_index = {}
    for meta in chunk_metadata:
        dec_no = meta.get("decision_no")
        if dec_no and dec_no not in case_index:
            case_index[dec_no] = {
                "case_id": meta["case_id"],
                "source": meta["source"],
                "decision_no": dec_no,
                "date": meta.get("date", ""),
                "subject": meta.get("subject", ""),
                "court": meta.get("court", ""),
                "parties": meta.get("parties", {}),
                "judges": meta.get("judges", ""),
                "provisions": meta.get("provisions", ""),
                "final_order": meta.get("final_order", ""),
                "legal_principle": meta.get("legal_principle", ""),
                "precedents": meta.get("precedents", ""),
                "appellant_lawyer": meta.get("appellant_lawyer", ""),
                "respondent_lawyer": meta.get("respondent_lawyer", ""),
            }
    with open(CASE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(case_index, f, ensure_ascii=False, indent=2)

    summary = {
        "last_ingested": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(file_names),
        "total_pages": total_pages,
        "total_chunks": len(documents),
        "files": file_names,
        "file_fingerprints": fingerprints,
        "cases": sorted({m["case_id"] for m in chunk_metadata}),
        "case_metadata": {},
    }
    for m in chunk_metadata:
        cid = m["case_id"]
        if cid not in summary["case_metadata"]:
            summary["case_metadata"][cid] = {
                "source": m["source"],
                "decision_no": m.get("decision_no", ""),
                "date": m.get("date", ""),
                "subject": m.get("subject", ""),
                "court": m.get("court", ""),
                "parties": m.get("parties", {}),
                "judges": m.get("judges", ""),
                "appellant_lawyer": m.get("appellant_lawyer", ""),
                "respondent_lawyer": m.get("respondent_lawyer", ""),
                "provisions": m.get("provisions", ""),
                "final_order": m.get("final_order", ""),
                "legal_principle": m.get("legal_principle", ""),
                "precedents": m.get("precedents", ""),
            }
    with open(INGEST_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    reset_lookup_indexes()


def upsert_embeddings(collection, model, documents: list[str], metadatas: list[dict], start_index: int):
    ids = [f"doc_chunk_{start_index + i}" for i in range(len(documents))]
    for i in range(0, len(documents), EMBED_BATCH_SIZE):
        j = min(i + EMBED_BATCH_SIZE, len(documents))
        embeddings = model.encode(
            documents[i:j],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=EMBED_BATCH_SIZE,
        ).tolist()
        for k in range(i, j, CHROMA_UPSERT_BATCH):
            end = min(k + CHROMA_UPSERT_BATCH, j)
            local_start = k - i
            local_end = end - i
            collection.upsert(
                ids=ids[k:end],
                embeddings=embeddings[local_start:local_end],
                metadatas=metadatas[k:end],
                documents=documents[k:end],
            )


def get_or_create_collection(client, rebuild: bool):
    if rebuild:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        return client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def process_local_documents(target_path: str = None, rebuild: bool = False):
    if target_path is None:
        target_path = DATA_DIR

    valid_files = collect_source_files(target_path)
    if not valid_files:
        print(f"No .pdf or .txt files found at '{target_path}'.")
        return

    current_fps = {os.path.basename(p): file_fingerprint(p) for p in valid_files}
    loaded = None if rebuild else load_existing_index()
    if loaded is None:
        print("Legacy BM25 pickle has no stored documents; rebuilding corpus.")
        rebuild = True
        existing_meta, existing_docs = [], []
    else:
        existing_meta, existing_docs = loaded
    previous_summary = {}
    if os.path.exists(INGEST_METADATA_PATH) and not rebuild:
        try:
            with open(INGEST_METADATA_PATH, "r", encoding="utf-8") as fh:
                previous_summary = json.load(fh)
        except Exception:
            previous_summary = {}

    previous_fps = {fp["name"]: fp for fp in previous_summary.get("file_fingerprints", [])}
    indexed_sources = {m.get("source") for m in existing_meta}

    changed = [
        name for name, fp in current_fps.items()
        if name in previous_fps and (
            previous_fps[name].get("mtime") != fp["mtime"] or previous_fps[name].get("size") != fp["size"]
        )
    ]
    removed = [name for name in previous_fps if name not in current_fps]
    if changed or removed:
        print(f"Indexed files changed ({changed}) or removed ({removed}); rebuilding corpus from extract cache.")
        rebuild = True
        existing_meta, existing_docs = [], []

    new_files = [
        path for path in valid_files
        if rebuild or os.path.basename(path) not in indexed_sources
    ]
    if not rebuild and not new_files:
        print("Knowledge base is up to date. Pass --rebuild to force a full reindex.")
        return

    lang_flag = resolve_ocr_lang_flag()
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = get_or_create_collection(chroma_client, rebuild=rebuild)

    new_chunks = []
    new_metadata = []
    pages_processed = 0
    workers = max(1, INGEST_WORKERS)
    print(f"Extracting {len(new_files)} file(s) with {workers} worker(s)...", flush=True)

    def _extract_one(doc_path: str):
        file_name = os.path.basename(doc_path)
        pages, case_meta = extract_pages(doc_path, lang_flag)
        chunks, metas = chunks_from_pages(file_name, pages, case_meta)
        return file_name, pages, case_meta, chunks, metas

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_one, path) for path in new_files]
        extracted = []
        for path, fut in zip(new_files, futures):
            try:
                extracted.append(fut.result())
            except Exception as e:
                print(f"Error processing {os.path.basename(path)}: {e}")

    for file_idx, (file_name, pages, case_meta, chunks, metas) in enumerate(extracted, 1):
        pages_processed += len(pages)
        print(
            f"[{file_idx}/{len(new_files)}] {file_name} "
            f"case_id={case_meta['case_id']} decision={case_meta['decision_no']} "
            f"chunks={len(chunks)}",
            flush=True,
        )
        new_chunks.extend(chunks)
        new_metadata.extend(metas)

    if not new_chunks and not existing_meta:
        print("No valid text chunks extracted.")
        return

    start_index = 0 if rebuild else len(existing_meta)
    chroma_metas = [chroma_row(meta, start_index + i) for i, meta in enumerate(new_metadata)]
    if new_chunks:
        upsert_embeddings(collection, model, new_chunks, chroma_metas, start_index)

    all_metadata = existing_meta + new_metadata
    all_documents = existing_docs + new_chunks
    fingerprints = [current_fps[os.path.basename(p)] for p in valid_files]
    total_pages = previous_summary.get("total_pages", 0) if not rebuild else 0
    total_pages = pages_processed if rebuild else total_pages + pages_processed
    persist_indexes(
        all_metadata,
        all_documents,
        fingerprints,
        total_pages,
        [os.path.basename(p) for p in valid_files],
    )
    print(f"✅ Ingestion completed: {len(valid_files)} files, {len(all_documents)} chunks "
          f"({len(new_chunks)} new)")


def process_uploaded_file(doc_path: str) -> dict:
    """Incrementally ingest a single user-uploaded file (PDF, TXT, or image)."""
    if not os.path.isfile(doc_path):
        raise FileNotFoundError(f"Uploaded file not found: {doc_path}")

    file_name = os.path.basename(doc_path)
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in _SUPPORTED_UPLOAD_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"[upload] Processing '{file_name}'...", flush=True)

    lang_flag = resolve_ocr_lang_flag()
    pages, case_meta = extract_pages(doc_path, lang_flag)
    chunks, metas = chunks_from_pages(file_name, pages, case_meta)

    if not chunks:
        print(f"[upload] No text extracted from '{file_name}'.")
        return {
            "file_id": file_name,
            "file_name": file_name,
            "case_id": case_meta["case_id"],
            "chunks_added": 0,
        }

    loaded = load_existing_index()
    if loaded is None:
        existing_meta, existing_docs = [], []
    else:
        existing_meta, existing_docs = loaded

    start_index = len(existing_meta)

    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = get_or_create_collection(chroma_client, rebuild=False)
    chroma_metas = [chroma_row(m, start_index + i) for i, m in enumerate(metas)]
    upsert_embeddings(collection, model, chunks, chroma_metas, start_index)

    all_meta = existing_meta + metas
    all_docs = existing_docs + chunks

    previous_summary: dict = {}
    if os.path.exists(INGEST_METADATA_PATH):
        try:
            with open(INGEST_METADATA_PATH, "r", encoding="utf-8") as fh:
                previous_summary = json.load(fh)
        except Exception:
            previous_summary = {}

    fp = file_fingerprint(doc_path)
    existing_fps = list(previous_summary.get("file_fingerprints", []))
    existing_fps = [f for f in existing_fps if f.get("name") != file_name]
    existing_fps.append(fp)

    existing_file_names = [f for f in previous_summary.get("files", []) if f != file_name]
    existing_file_names.append(file_name)

    total_pages = previous_summary.get("total_pages", 0) + len(pages)

    persist_indexes(
        all_meta,
        all_docs,
        existing_fps,
        total_pages,
        existing_file_names,
    )

    print(f"[upload] ✅ '{file_name}' — {len(chunks)} chunks added (total corpus: {len(all_docs)}).")
    return {
        "file_id": file_name,
        "file_name": file_name,
        "case_id": case_meta["case_id"],
        "chunks_added": len(chunks),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest NKP documents into the hybrid index.")
    parser.add_argument("path", nargs="?", default=None, help="Folder or file to ingest (default: ./data)")
    parser.add_argument("--rebuild", action="store_true", help="Wipe Chroma/BM25 and reindex all files")
    args = parser.parse_args()
    process_local_documents(args.path, rebuild=args.rebuild)


if __name__ == "__main__":
    main()