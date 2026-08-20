import os
import pickle
import json
import re
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.text_processor import clean_devanagari_text
from src.hybrid_search import perform_hybrid_search, detect_query_intent, extract_query_identifiers
from llm_generator import generate_nepali_answer

st.set_page_config(page_title="Nepali Legal Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ Nepali Legal Assistant & Document Chatbot")
st.caption("AI-Powered Legal Search & Case-Aware Question Answering over NKP Documents")

if not os.environ.get("GROQ_API_KEY"):
    st.warning("⚠️ GROQ_API_KEY is not set in your environment.")

# Session state
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

FOLLOWUP_PRONOUNS = {"यो", "उक्त", "सो", "तो", "त्यो", "यस", "उस", "ती", "तिनी", "उनी"}
CASE_REFERENCE_KEYWORDS = {"मुद्दा", "फैसला", "निर्णय", "केस"}

def resolve_followup(query: str, current_case: dict, last_mentioned: dict) -> tuple[str, dict]:
    if not query:
        return query, current_case

    identifiers = extract_query_identifiers(query)
    new_case = None
    if identifiers.get("decision_no"):
        new_case = {"case_id": f"decision_{identifiers['decision_no']}", "decision_no": identifiers['decision_no']}
    elif identifiers.get("source"):
        new_case = {"case_id": identifiers["source"], "source": identifiers["source"]}
    elif identifiers.get("source_stem"):
        new_case = {"case_id": identifiers["source_stem"], "source": identifiers["source_stem"]}

    if new_case:
        return query, new_case

    tokens = set(query.lower().split())
    if any(w in tokens for w in (FOLLOWUP_PRONOUNS | CASE_REFERENCE_KEYWORDS)):
        target_case = current_case or last_mentioned
        if target_case:
            case_id = target_case.get("case_id", "अज्ञात")
            decision_no = target_case.get("decision_no", "अज्ञात")
            prefix = f"मुद्दा {case_id} (निर्णय नं. {decision_no}) को बारेमा"
            resolved = f"{prefix}: {query}"
            return resolved, target_case

    return query, current_case

@st.cache_resource
def load_search_engines():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="nepali_legal_docs")
    model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
    with open("./models/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    metadata_info = {}
    if os.path.exists("./models/ingest_metadata.json"):
        with open("./models/ingest_metadata.json", "r", encoding="utf-8") as meta_f:
            metadata_info = json.load(meta_f)
    return collection, model, bm25_data["bm25"], bm25_data["metadata"], metadata_info

try:
    collection, model, bm25, chunk_metadata, metadata_info = load_search_engines()
except Exception:
    st.error("❌ Knowledge base not found. Run `python ingest.py` first to rebuild the case-aware index.")
    st.stop()

st.sidebar.header("⚙️ Chatbot Settings")
top_k = st.sidebar.slider("Top chunks / evidence:", 3, 12, 5)
alpha = st.sidebar.slider("Vector ↔ BM25 weight:", 0.0, 1.0, 0.15, 0.05)

st.sidebar.divider()
st.sidebar.header("📚 Knowledge Base")
st.sidebar.caption(f"🟢 ChromaDB vectors: {collection.count()}")
st.sidebar.caption(f"🧩 Indexed chunks: {len(chunk_metadata)}")
st.sidebar.caption(f"📄 Documents: {metadata_info.get('total_files', 0)}")
st.sidebar.caption(f"⚖️ Cases: {len(metadata_info.get('case_metadata', {}))}")

if st.session_state.current_case:
    st.sidebar.info(f"**Current Case:** {st.session_state.current_case.get('case_id', 'Unknown')}")
else:
    st.sidebar.info("**Current Case:** None (ask a new case)")

if st.sidebar.button("🔄 Reload DB", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.current_case = None
    st.session_state.last_mentioned_case = None
    st.session_state.case_history = []
    st.rerun()

def render_citations(sources):
    with st.expander("📚 Source Document Citations"):
        for rank, item in enumerate(sources, 1):
            st.markdown(
                f"**Evidence #{rank}** — `{item.get('source')}` | "
                f"निर्णय नं. `{item.get('decision_no', 'N/A')}` | "
                f"पृष्ठ `{item.get('page', '?')}` | Score `{item.get('score', 0):.4f}`"
            )
            st.info(item.get("content", ""))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_citations(message["sources"])

if prompt := st.chat_input("कानूनी विषय, फैसला, निर्णय नं. वा PDF को बारेमा सोध्नुहोस्..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    cleaned_query = clean_devanagari_text(prompt)
    current_case_dict = st.session_state.current_case
    last_mentioned = st.session_state.last_mentioned_case
    resolved_query, new_case = resolve_followup(cleaned_query, current_case_dict, last_mentioned)

    if new_case != current_case_dict:
        st.session_state.current_case = new_case
        st.session_state.case_history.append((new_case, prompt))
        if new_case == last_mentioned:
            st.session_state.last_mentioned_case = None
        # No rerun – answer immediately

    final_query = resolved_query

    with st.chat_message("assistant"):
        intent = detect_query_intent(final_query)
        st.caption(f"Query type: `{intent}`" +
                   (f" | Case: `{st.session_state.current_case.get('case_id')}`" if st.session_state.current_case else ""))

        with st.spinner("कानूनी कागजात खोजिँदैछ..."):
            search_results = perform_hybrid_search(
                query=final_query,
                collection=collection,
                model=model,
                bm25=bm25,
                chunk_metadata=chunk_metadata,
                top_k=top_k,
                alpha=alpha,
            )

        with st.spinner("निर्णयका प्रमाणमा आधारित उत्तर तयार हुँदैछ..."):
            final_answer = generate_nepali_answer(
                query=final_query,
                retrieved_items=search_results,
                current_case=st.session_state.current_case,
                metadata_info=metadata_info  # <-- added this parameter
            )

        decision_matches = re.findall(r'निर्णय नं\.\s*(\d+)', final_answer)
        if decision_matches:
            last_dec = decision_matches[-1]
            st.session_state.last_mentioned_case = {
                "case_id": f"decision_{last_dec}",
                "decision_no": last_dec
            }
        else:
            st.session_state.last_mentioned_case = None

        st.markdown(final_answer)
        if search_results:
            render_citations(search_results)
        else:
            st.warning("सान्दर्भिक प्रमाण फेला परेन।")

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "sources": search_results,
    })