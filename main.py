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

# ---- Load global case metadata at startup so LIST_CASES / decision-existence
#      checks always have the full picture instead of relying on None. ----
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
    session_id: str

class TranscribeResponse(BaseModel):
    transcript: str

def build_current_case(case_id: Optional[str], decision_no: Optional[str]) -> Optional[dict]:
    if case_id:
        return {"case_id": case_id, "decision_no": decision_no or case_id.replace("decision_", "")}
    if decision_no:
        return {"case_id": f"decision_{decision_no}", "decision_no": decision_no}
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
        "voice_live": False,  # Uses the robust multi-turn VAD pipeline
        "voice_language": VOICE_LANGUAGE,
        "stt_model": GEMINI_STT_MODEL,
    }

# ---- Native Neural Nepali TTS (Strips Markdown for Natural Speech) ----
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
        # The frontend must persist and resend this session_id on every
        # subsequent request (see ChatResponse.session_id below) — otherwise
        # a fresh uuid is minted each turn and session_cases (sticky case
        # tracking) never actually persists across the conversation.
        sess_id = request.session_id or str(uuid.uuid4())

        # Pass the sticky case's case_id so extract_query_identifiers' own
        # pronoun-resolution ("यो", "उक्त", "सो", ...) can actually fire —
        # previously this parameter was never supplied, so that logic was
        # dead code and pronoun references silently fell through to the
        # unconditional session_cases fallback instead.
        active_case_id = (session_cases.get(sess_id) or {}).get("case_id")
        identifiers = extract_query_identifiers(query, active_case_id=active_case_id)
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
                return ChatResponse(reply=comparison_reply, session_id=sess_id)

        # 2. Resolve the requested decision number, if any, with ASR/typo
        #    tolerance: snap a near-miss like "91099" -> "9099" instead of
        #    bluntly rejecting it (this is exactly the "audio mis-hears the
        #    number" problem — see snap_decision_number's edit-distance
        #    tolerance in hybrid_search.py).
        requested_dec = identifiers.get("decision_no")
        requested_dec_is_explicit = bool(requested_dec) and not identifiers.get("_inferred_as_standalone")
        if requested_dec:
            requested_dec = snap_decision_number(requested_dec, chunk_metadata)

        # Only reject immediately for an EXPLICIT, high-confidence reference
        # (a literal "निर्णय नं. XXXX" pattern) that still doesn't exist
        # after snapping. A WEAKLY inferred number (extract_query_identifiers'
        # fallback: "any case-keyword nearby + exactly one bare number") is
        # not reliable enough to short-circuit on its own — e.g. "प्रहरी सेवा
        # २०४९ ... फैसला" contains "२०४९", the *regulation year*, not a
        # decision number, and infer_case_from_keywords below correctly
        # resolves that query to a real case (9099) via the word "प्रहरी".
        # Rejecting on the weak number guess alone, before keyword inference
        # gets a chance, previously broke that working case entirely.
        if (
            requested_dec_is_explicit
            and intent not in ("LIST_CASES", "COMPARISON")
            and not decision_exists(requested_dec, chunk_metadata)
        ):
            available = []
            if METADATA_INFO and METADATA_INFO.get("case_metadata"):
                available = sorted(set(
                    str(info.get("decision_no"))
                    for info in METADATA_INFO["case_metadata"].values()
                    if info.get("decision_no")
                ))
            extra = f" उपलब्ध निर्णय नं.: {', '.join(available)}।" if available else ""
            return ChatResponse(
                reply=f"निर्णय नं. {requested_dec} यस ज्ञानकोषमा अनुक्रमित छैन।{extra}",
                session_id=sess_id,
            )

        # 3. Bypass sticky session lock for genuinely global queries.
        #    - LIST_CASES / COMPARISON: explicit "list/compare everything" intent.
        #    - LEGAL_PROVISION: "X ऐन/धारा भनेको के हो?" is a general-law
        #      question, not a question about whatever case was last
        #      discussed — silently scoping it to the sticky case only
        #      produces a false "not found" for content that may not even
        #      relate to that case.
        #    - CROSS_CASE_CUES: explicit "among your decisions..." survey
        #      questions must search everything, not just the sticky case.
        #      Without this, such a question was observed to get scoped to
        #      whichever single case was last discussed, and the LLM then
        #      produced a plausible-sounding but factually wrong answer
        #      about *that* case instead of correctly identifying a
        #      different, actually-matching case elsewhere in the corpus.
        CROSS_CASE_CUES = ["मध्ये", "सबै निर्णयमा", "सबैमा", "कुन निर्णयमा", "कुनकुन निर्णयमा"]
        if intent in ("LIST_CASES", "COMPARISON") or intent == "LEGAL_PROVISION" or any(cue in query for cue in CROSS_CASE_CUES):
            current_case = None
        else:
            # Resolve the *real* case_id from METADATA_INFO rather than
            # assuming "decision_{no}" — ingest.py only uses that convention
            # when it successfully read the decision number off the PDF;
            # otherwise it falls back to a "file_<name>" case_id, and
            # guessing wrong here silently breaks the Chroma `where` filter
            # downstream (perform_hybrid_search's decision-number fallback
            # covers this too, but resolving it correctly at the source is
            # cheaper and keeps session_cases accurate for later turns).
            query_case = None
            if requested_dec and decision_exists(requested_dec, chunk_metadata):
                case_info = (METADATA_INFO or {}).get("case_metadata", {}).get(requested_dec)
                real_case_id = case_info.get("case_id") if case_info else None
                query_case = {
                    "case_id": real_case_id or f"decision_{requested_dec}",
                    "decision_no": requested_dec,
                }

            if query_case is None:
                query_case = infer_case_from_keywords(query)

            # A weakly-inferred number that matched neither a real decision
            # nor any keyword-based case: now it's safe to tell the user
            # it isn't indexed, since we gave keyword inference a fair shot.
            if query_case is None and requested_dec and not requested_dec_is_explicit:
                available = []
                if METADATA_INFO and METADATA_INFO.get("case_metadata"):
                    available = sorted(set(
                        str(info.get("decision_no"))
                        for info in METADATA_INFO["case_metadata"].values()
                        if info.get("decision_no")
                    ))
                extra = f" उपलब्ध निर्णय नं.: {', '.join(available)}।" if available else ""
                return ChatResponse(
                    reply=f"निर्णय नं. {requested_dec} यस ज्ञानकोषमा अनुक्रमित छैन।{extra}",
                    session_id=sess_id,
                )

            if query_case is None and identifiers.get("_inferred_from_context") and active_case_id:
                # Explicit pronoun reference ("यो", "उक्त", "सो", ...) back
                # to the case already being discussed.
                query_case = session_cases.get(sess_id)

            if query_case:
                current_case = query_case
                session_cases[sess_id] = query_case
            else:
                current_case = build_current_case(request.case_id, request.decision_no)
                if current_case is None and sess_id in session_cases:
                    current_case = session_cases[sess_id]

        # 3. First hybrid search attempt (scoped to current_case if applicable)
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

        # 4. Unscoped fallback retry if scoped search yielded zero results —
        #    prevents a wrong/eager case-lock from silently producing "no info".
        if not results and current_case:
            print(f"🔄 Scoped search in {current_case.get('case_id')} empty. Retrying unscoped search across entire corpus...")
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

        # 6. Pass real METADATA_INFO and chunk_metadata through.
        answer = generate_nepali_answer(
            query=query,
            retrieved_items=results,
            model_name=LLM_MODEL,
            current_case=current_case,
            metadata_info=METADATA_INFO,
            comparison_mode=False,
            detected_numbers=None,
            stream=False,
            chunk_metadata=chunk_metadata,
        )

        if hasattr(answer, "__iter__") and not isinstance(answer, str):
            full = "".join(chunk.choices[0].delta.content or "" for chunk in answer if hasattr(chunk, "choices") and chunk.choices)
            answer = full

        reply_text = str(answer) if answer else "क्षमा गर्नुहोस्, उत्तर उत्पन्न गर्न सकिएन।"
        return ChatResponse(reply=reply_text, session_id=sess_id)
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

        # 1. Gemini STT — force text-only output and inspect all part types,
        #    since some responses return only an `audio_transcription` part
        #    rather than a plain `text` part (response.text alone can miss it).
        if client:
            try:
                response = client.models.generate_content(
                    model=GEMINI_STT_MODEL,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime),
                        "यो अडियोलाई शुद्ध नेपालीमा ट्रान्सक्राइब गर्नुहोस्। निर्णय नं. ९०९९ वा ९१०० जस्ता अंक प्रष्ट लेख्नुहोस्।",
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT"],
                        temperature=0,
                    ),
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
                else:
                    print(f"⚠️ Gemini STT ({GEMINI_STT_MODEL}) returned no usable text; falling back to Groq.")
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