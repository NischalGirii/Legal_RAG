import os
import sys
import io
import pickle
import uuid
import json
import traceback
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types
from groq import Groq
import edge_tts

from src.llm_generator import generate_nepali_answer, generate_comparison_answer
from src.hybrid_search import (
    perform_hybrid_search,
    extract_query_identifiers,
    detect_query_intent,
    decision_exists,
    snap_decision_number,
)
from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_PATH,
    BM25_INDEX_PATH,
    CASE_INDEX_PATH,
    INGEST_METADATA_PATH,
    LLM_MODEL,
    VOICE_LANGUAGE,
    GEMINI_STT_MODEL,
)
from src.sparse_index import retriever_from_pickle
from src.text_processor import clean_devanagari_text, clean_asr_transcript, clean_text_for_tts

load_dotenv()

app = FastAPI(title="Legal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Clients ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---- Load search engines ----
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = retriever_from_pickle(bm25_data)
    chunk_metadata = bm25_data["metadata"]
    print("✅ Search engines loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load knowledge base: {e}")
    sys.exit(1)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
session_cases: Dict[str, dict] = {}

# ---- Load Global Case Metadata for LIST_CASES ----
METADATA_INFO = None
if os.path.exists(CASE_INDEX_PATH):
    try:
        with open(CASE_INDEX_PATH, "r", encoding="utf-8") as f:
            case_index_data = json.load(f)
        METADATA_INFO = {"case_metadata": case_index_data}
        print(f"✅ Loaded metadata for {len(case_index_data)} cases from {CASE_INDEX_PATH}.")
    except Exception as e:
        print(f"⚠️ Could not load CASE_INDEX_PATH: {e}")
elif os.path.exists(INGEST_METADATA_PATH):
    try:
        with open(INGEST_METADATA_PATH, "r", encoding="utf-8") as f:
            ingest_data = json.load(f)
        METADATA_INFO = {"case_metadata": ingest_data.get("case_metadata", {})}
        print("✅ Loaded metadata from INGEST_METADATA_PATH.")
    except Exception as e:
        print(f"⚠️ Could not load INGEST_METADATA_PATH: {e}")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    case_id: Optional[str] = None
    decision_no: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

class TranscribeResponse(BaseModel):
    transcript: str

def build_current_case(case_id: Optional[str], decision_no: Optional[str]) -> Optional[dict]:
    dec = snap_decision_number(decision_no or (case_id.replace("decision_", "") if case_id else None))
    if dec:
        return {"case_id": f"decision_{dec}", "decision_no": dec}
    return None

def infer_case_from_keywords(query: str) -> Optional[dict]:
    q = query.lower()
    if any(kw in q for kw in ["प्रहरी", "पुलिस", "police", "नियमावली", "३० वर्ष", "30 years", "२०४९"]):
        dec = "9099"
        if decision_exists(dec, chunk_metadata):
            return {"case_id": f"decision_{dec}", "decision_no": dec}
    if any(kw in q for kw in ["अध्यादेश", "पूर्व पदाधिकारी", "भूतपूर्व", "रोक्का", "सम्पत्ति"]):
        dec = "9100"
        if decision_exists(dec, chunk_metadata):
            return {"case_id": f"decision_{dec}", "decision_no": dec}
    return None

@app.get("/api/config")
async def get_config():
    return {
        "voice_live": False,
        "voice_language": VOICE_LANGUAGE,
        "stt_model": GEMINI_STT_MODEL,
    }

# ---- Native Neural Nepali TTS ----
@app.get("/api/tts")
async def text_to_speech(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        spoken_text = clean_text_for_tts(text)
        voice = "ne-NP-SagarNeural"
        communicate = edge_tts.Communicate(spoken_text, voice=voice)

        audio_stream = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.extend(chunk["data"])

        return Response(content=bytes(audio_stream), media_type="audio/mpeg")
    except Exception as e:
        print(f"❌ [TTS Error]: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        raw_query = clean_asr_transcript(request.message)
        query = clean_devanagari_text(raw_query)
        sess_id = request.session_id or str(uuid.uuid4())

        identifiers = extract_query_identifiers(query)
        intent = detect_query_intent(query)

        # 1. Multi-case comparison check
        multi_cases = identifiers.get("multiple_decision_nos")
        if multi_cases and len(multi_cases) >= 2:
            comparison_reply = generate_comparison_answer(
                query=query,
                decision_numbers=multi_cases,
                metadata_info=METADATA_INFO,
                chunk_metadata=chunk_metadata,
                collection=collection,
                model=model,
                bm25=bm25,
            )
            if comparison_reply:
                return ChatResponse(reply=comparison_reply)

        # 2. Dynamic Case Resolution (Bypasses sticky session for global queries)
        if intent in ("LIST_CASES", "COMPARISON", "GREETING"):
            current_case = None
        else:
            query_case = None
            if identifiers.get("decision_no"):
                dec = identifiers["decision_no"]
                if decision_exists(dec, chunk_metadata):
                    query_case = {"case_id": f"decision_{dec}", "decision_no": dec}

            if query_case is None:
                query_case = infer_case_from_keywords(query)

            # If user asks about a new case, override the session context
            if query_case:
                current_case = query_case
                session_cases[sess_id] = query_case
            else:
                current_case = build_current_case(request.case_id, request.decision_no)
                if current_case is None and sess_id in session_cases:
                    current_case = session_cases[sess_id]

        # 3. Hybrid search
        results = perform_hybrid_search(
            query=query,
            collection=collection,
            model=model,
            bm25=bm25,
            chunk_metadata=chunk_metadata,
            top_k=15,
            alpha=0.7,
            current_case=current_case,
            identifiers=identifiers,
        )

        # 4. Unscoped fallback retry if scoped search yielded zero results
        if not results and current_case:
            print(f"🔄 Scoped search in {current_case.get('case_id')} empty. Retrying unscoped search...")
            results = perform_hybrid_search(
                query=query,
                collection=collection,
                model=model,
                bm25=bm25,
                chunk_metadata=chunk_metadata,
                top_k=15,
                alpha=0.7,
                current_case=None,
                identifiers=identifiers,
            )

        # 5. Cross-Encoder reranking
        if results:
            pairs = [[query, r["content"]] for r in results]
            scores = cross_encoder.predict(pairs)
            for r, sc in zip(results, scores):
                r["score"] = float(sc)
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:5]

        # 6. Pass real METADATA_INFO to answer generator
        answer = generate_nepali_answer(
            query=query,
            retrieved_items=results,
            model_name=LLM_MODEL,
            current_case=current_case,
            metadata_info=METADATA_INFO,
            comparison_mode=False,
            detected_numbers=None,
            stream=False,
        )

        if hasattr(answer, "__iter__") and not isinstance(answer, str):
            full = "".join(chunk.choices[0].delta.content or "" for chunk in answer if hasattr(chunk, "choices") and chunk.choices)
            answer = full

        return ChatResponse(reply=str(answer) if answer else "क्षमा गर्नुहोस्, उत्तर उत्पन्न गर्न सकिएन।")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        if not audio_bytes or len(audio_bytes) < 1000:
            return TranscribeResponse(transcript="")

        raw_mime = audio.content_type or "audio/webm"
        clean_mime = raw_mime.split(";")[0].strip().lower()
        if clean_mime not in ["audio/webm", "audio/mp4", "audio/wav", "audio/ogg", "audio/mp3", "audio/mpeg"]:
            clean_mime = "audio/webm"

        print(f"[Transcribe] Processing {len(audio_bytes)} bytes ({clean_mime})...")

        # 1. Gemini STT with candidate part inspection
        if client:
            try:
                response = client.models.generate_content(
                    model=GEMINI_STT_MODEL,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime),
                        "यो अडियोलाई शुद्ध नेपालीमा ट्रान्सक्राइब गर्नुहोस्। निर्णय नं. ९०९९ वा ९१०० जस्ता अंक प्रष्ट लेख्नुहोस्।",
                    ],
                )
                raw_text = (response.text or "").strip()
                if not raw_text and response.candidates:
                    for cand in response.candidates:
                        if cand.content and cand.content.parts:
                            for p in cand.content.parts:
                                if hasattr(p, "text") and p.text:
                                    raw_text += p.text
                                elif hasattr(p, "audio_transcription") and p.audio_transcription:
                                    raw_text += getattr(p.audio_transcription, "text", str(p.audio_transcription))

                if raw_text.strip():
                    transcript = clean_asr_transcript(raw_text.strip())
                    print(f"✅ [Transcribe ({GEMINI_STT_MODEL})] Result: '{transcript}'")
                    return TranscribeResponse(transcript=transcript)
            except Exception as gemini_err:
                print(f"⚠️ Gemini STT ({GEMINI_STT_MODEL}) skipped: {gemini_err}")

        # 2. Groq Whisper fallback
        if groq_client:
            try:
                file_obj = io.BytesIO(audio_bytes)
                file_obj.name = "audio.webm"
                transcription = groq_client.audio.transcriptions.create(
                    file=file_obj,
                    model="whisper-large-v3",
                    language="ne",
                    prompt="नेपाली कानून, सर्वोच्च अदालत, निर्णय नं. ९०९९, निर्णय नं. ९१००, प्रहरी नियमावली २०४९, ३० वर्षे सेवा अवधि, न्यायाधीश, निवेदक, विपक्षी, फैसला मिति",
                    response_format="text",
                )
                transcript = clean_asr_transcript(str(transcription).strip())
                print(f"✅ [Transcribe (Groq Whisper)] Result: '{transcript}'")
                return TranscribeResponse(transcript=transcript)
            except Exception as groq_err:
                print(f"❌ Groq Whisper error: {groq_err}")

        raise HTTPException(status_code=500, detail="All transcription services failed.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok"}