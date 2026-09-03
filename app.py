import os
import pickle
import json
import re
import datetime
import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_PATH,
    BM25_INDEX_PATH,
    INGEST_METADATA_PATH,
    LLM_MODEL,
)
from src.text_processor import clean_devanagari_text, normalize_digits
from src.hybrid_search import (
    perform_hybrid_search,
    detect_query_intent,
    extract_query_identifiers,
    get_lookup_indexes,
    decision_exists,
)
from src.llm_generator import generate_nepali_answer, generate_comparison_answer
from src.sparse_index import retriever_from_pickle

st.set_page_config(page_title="Nepali Legal Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ Nepali Legal Assistant & Document Chatbot")
st.caption("AI-Powered Legal Search & Case-Aware Question Answering over NKP Documents")

groq_key_present = bool(os.environ.get("GROQ_API_KEY"))
if not groq_key_present:
    st.warning("⚠️ GROQ_API_KEY is not set in your environment.")

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "नमस्कार! म तपाईंको नेपाली कानूनी सहायक हुँ। निर्णय नं., मुद्दा, पक्षकार, दफा वा NKP PDF का बारेमा प्रश्न सोध्नुहोस्।"
    }]
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "last_mentioned_case" not in st.session_state:
    st.session_state.last_mentioned_case = None
if "case_history" not in st.session_state:
    st.session_state.case_history = []
if "last_compared" not in st.session_state:
    st.session_state.last_compared = None

FOLLOWUP_PRONOUNS = {"यो", "उक्त", "सो", "तो", "त्यो", "यस", "उस", "ती", "तिनी", "उनी"}
CASE_REFERENCE_KEYWORDS = {"मुद्दा", "फैसला", "निर्णय", "केस"}

def resolve_followup(query: str, current_case: dict, last_mentioned: dict) -> tuple[str, dict]:
    if not query:
        return query, current_case

    identifiers = extract_query_identifiers(
        query,
        active_case_id=current_case.get("case_id") if current_case else None,
    )
    if identifiers.get("decision_no") and len(identifiers.get("decision_no", "").split(",")) >= 2:
        return query, current_case
    if identifiers.get("decision_no"):
        dec = identifiers["decision_no"]
        if decision_exists(dec, chunk_metadata):
            new_case = {
                "case_id": f"decision_{dec}",
                "decision_no": dec,
            }
            return query, new_case
        return query, current_case

    tokens = set(query.lower().split())
    if any(w in tokens for w in (FOLLOWUP_PRONOUNS | CASE_REFERENCE_KEYWORDS)):
        target_case = current_case or last_mentioned
        if target_case:
            case_id = target_case.get("case_id", "अज्ञात")
            decision_no = target_case.get("decision_no", "अज्ञात")
            prefix = f"मुद्दा {case_id} (निर्णय नं. {decision_no}) को बारेमा"
            return f"{prefix}: {query}", target_case

    return query, current_case

# ── Load resources (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_search_engines():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    metadata_info = {}
    if os.path.exists(INGEST_METADATA_PATH):
        with open(INGEST_METADATA_PATH, "r", encoding="utf-8") as meta_f:
            metadata_info = json.load(meta_f)
    chunk_metadata = bm25_data["metadata"]
    get_lookup_indexes(chunk_metadata)
    return collection, model, retriever_from_pickle(bm25_data), chunk_metadata, metadata_info

try:
    collection, model, bm25, chunk_metadata, metadata_info = load_search_engines()
except Exception as e:
    st.error(f"❌ Knowledge base not found ({e}). Run `python ingest.py` first to rebuild the case-aware index.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Chatbot Settings")
top_k = st.sidebar.slider("Top chunks / evidence:", 3, 12, 5)
alpha = st.sidebar.slider("Vector ↔ BM25 weight (α):", 0.0, 1.0, 0.50, 0.05,
                           help="1.0 = pure dense vector, 0.0 = pure BM25 keyword")
use_streaming = st.sidebar.toggle("⚡ Stream LLM response", value=True)

st.sidebar.divider()
st.sidebar.header("📚 Knowledge Base")
st.sidebar.caption(f"🟢 ChromaDB vectors: {collection.count()}")
st.sidebar.caption(f"🧩 Indexed chunks: {len(chunk_metadata)}")
st.sidebar.caption(f"📄 Documents: {metadata_info.get('total_files', 0)}")
st.sidebar.caption(f"⚖️ Cases: {len(metadata_info.get('case_metadata', {}))}")
st.sidebar.caption(f"🤖 LLM: `{LLM_MODEL}` via Groq")

if st.session_state.current_case:
    st.sidebar.info(f"**Current Case:** {st.session_state.current_case.get('case_id', 'Unknown')}")
else:
    st.sidebar.info("**Current Case:** None (ask about a case to set context)")

st.sidebar.divider()
if st.sidebar.button("🔄 Reload DB", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.current_case = None
    st.session_state.last_mentioned_case = None
    st.session_state.case_history = []
    st.session_state.last_compared = None
    st.rerun()

if st.sidebar.button("💾 Export Chat", use_container_width=True):
    lines = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{msg['content']}\n")
    export_text = "\n---\n".join(lines)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        label="📥 Download .txt",
        data=export_text.encode("utf-8"),
        file_name=f"legal_chat_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True,
    )

def _score_badge(score: float) -> str:
    if score >= 0.5:
        color = "green"
    elif score >= 0.25:
        color = "orange"
    else:
        color = "red"
    return f":{color}[Score `{score:.4f}`]"

def render_citations(sources: list):
    if not sources:
        return
    with st.expander(f"📚 Source Citations ({len(sources)} chunks)"):
        for rank, item in enumerate(sources, 1):
            badge = _score_badge(item.get("score", 0))
            st.markdown(
                f"**#{rank}** — `{item.get('source')}` | "
                f"निर्णय नं. `{item.get('decision_no', 'N/A')}` | "
                f"पृष्ठ `{item.get('page', '?')}` | {badge}"
            )
            st.info(item.get("content", ""))

# ── Render existing messages ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_citations(message["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("कानूनी विषय, फैसला, निर्णय नं. वा PDF को बारेमा सोध्नुहोस्..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    cleaned_query = clean_devanagari_text(prompt)

    identifiers = extract_query_identifiers(
        cleaned_query,
        active_case_id=st.session_state.current_case.get("case_id") if st.session_state.current_case else None,
    )

    numbers = identifiers.get("multiple_decision_nos") or []
    numbers = [normalize_digits(str(n)) for n in numbers]
    intent_early = detect_query_intent(cleaned_query)
    dual_followup = any(w in cleaned_query for w in ["दुवै", "यी दुई", "दुबै मुद्दा"])
    if (not numbers or len(numbers) < 2) and dual_followup and st.session_state.last_compared:
        numbers = list(st.session_state.last_compared)
    comparison_mode = intent_early == "COMPARISON" or len(numbers) >= 2
    answer_case = st.session_state.current_case
    if comparison_mode and len(numbers) >= 2:
        search_case = None
        st.session_state.last_compared = numbers
    else:
        comparison_mode = False
        search_case = st.session_state.current_case

    answer_case = st.session_state.current_case
    if not comparison_mode and identifiers.get("decision_no"):
        dec = identifiers["decision_no"]
        answer_case = {
            "case_id": f"decision_{dec}",
            "decision_no": dec,
        }
        if decision_exists(dec, chunk_metadata):
            st.session_state.current_case = answer_case
            st.session_state.case_history.append((answer_case, prompt))
            search_case = answer_case
        else:
            search_case = None

    if not (identifiers.get("decision_no") and not decision_exists(identifiers.get("decision_no"), chunk_metadata)):
        resolved_query, new_case = resolve_followup(
            cleaned_query,
            st.session_state.current_case,
            st.session_state.last_mentioned_case,
        )
        if new_case != st.session_state.current_case and not comparison_mode:
            st.session_state.current_case = new_case
            st.session_state.case_history.append((new_case, prompt))
            search_case = new_case
            answer_case = new_case
    else:
        resolved_query = cleaned_query

    if not st.session_state.current_case and st.session_state.last_mentioned_case and not comparison_mode:
        if not (identifiers.get("decision_no") and not decision_exists(identifiers.get("decision_no"), chunk_metadata)):
            st.session_state.current_case = st.session_state.last_mentioned_case
            search_case = st.session_state.last_mentioned_case
            answer_case = st.session_state.last_mentioned_case

    final_query = resolved_query

    with st.chat_message("assistant"):
        intent = detect_query_intent(final_query)
        st.caption(
            f"Query type: `{intent}`"
            + (f" | Case: `{answer_case.get('case_id')}`" if answer_case else "")
        )

        final_answer = ""
        search_results = []

        if comparison_mode and numbers:
            with st.spinner("दुवै मुद्दाको तुलना गर्दै..."):
                result = generate_comparison_answer(
                    query=final_query,
                    decision_numbers=numbers,
                    metadata_info=metadata_info,
                    chunk_metadata=chunk_metadata,
                    collection=collection,
                    model=model,
                    bm25=bm25,
                    top_k=5,
                    alpha=alpha,
                )
            if result:
                final_answer = result
                st.markdown(final_answer)
            else:
                final_answer = "तुलना गर्न मिल्ने जानकारी फेला परेन।"
                st.warning(final_answer)
            search_results = []
        else:
            unknown_decision = bool(
                identifiers.get("decision_no")
                and not decision_exists(identifiers["decision_no"], chunk_metadata)
            )
            if unknown_decision:
                search_results = []
            else:
                with st.spinner("कानूनी कागजात खोजिँदैछ..."):
                    search_results = perform_hybrid_search(
                        query=final_query,
                        collection=collection,
                        model=model,
                        bm25=bm25,
                        chunk_metadata=chunk_metadata,
                        top_k=top_k,
                        alpha=alpha,
                        current_case=search_case,
                        identifiers=identifiers,
                    )

            with st.spinner("निर्णयका प्रमाणमा आधारित उत्तर तयार हुँदैछ..."):
                result = generate_nepali_answer(
                    query=final_query,
                    retrieved_items=search_results,
                    current_case=answer_case,
                    metadata_info=metadata_info,
                    comparison_mode=comparison_mode,
                    detected_numbers=numbers if comparison_mode else None,
                    stream=use_streaming and groq_key_present,
                )

            if use_streaming and groq_key_present and hasattr(result, "__iter__") and not isinstance(result, str):
                final_answer = st.write_stream(
                    chunk.choices[0].delta.content or ""
                    for chunk in result
                    if chunk.choices and chunk.choices[0].delta.content
                )
            else:
                final_answer = result if isinstance(result, str) else str(result)
                st.markdown(final_answer)

            decision_matches = re.findall(r"निर्णय नं\.\s*(\d+)", final_answer or "")
            if decision_matches:
                last_dec = decision_matches[-1]
                st.session_state.last_mentioned_case = {
                    "case_id": f"decision_{last_dec}",
                    "decision_no": last_dec,
                }
            else:
                st.session_state.last_mentioned_case = None

            if search_results:
                render_citations(search_results)
            elif not unknown_decision:
                st.warning("सान्दर्भिक प्रमाण फेला परेन।")

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer or "",
        "sources": search_results if not comparison_mode else [],
    })