import os
import pickle
import json
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


@st.cache_resource
def load_search_engines():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="nepali_legal_docs")
    model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
    with open("./models/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    metadata_info = {}
    if os.path.exists("./models/ingest_metadata.json"):
        try:
            with open("./models/ingest_metadata.json", "r", encoding="utf-8") as meta_f:
                metadata_info = json.load(meta_f)
        except Exception:
            pass
    return collection, model, bm25_data["bm25"], bm25_data["metadata"], metadata_info


try:
    collection, model, bm25, chunk_metadata, metadata_info = load_search_engines()
except Exception:
    st.error("❌ Knowledge base not found. Run `python ingest.py` first to rebuild the case-aware index.")
    st.stop()

st.sidebar.header("⚙️ Chatbot Settings")
top_k = st.sidebar.slider("Top chunks / evidence:", 3, 12, 5)
alpha = st.sidebar.slider("Vector ↔ BM25 weight:", 0.0, 1.0, 0.75, 0.05)

st.sidebar.divider()
st.sidebar.header("📚 Knowledge Base")
st.sidebar.caption(f"🟢 ChromaDB vectors: {collection.count()}")
st.sidebar.caption(f"🧩 Indexed chunks: {len(chunk_metadata)}")
st.sidebar.caption(f"📄 Documents: {metadata_info.get('total_files', 0)}")
st.sidebar.caption(f"⚖️ Cases: {len(metadata_info.get('case_metadata', {}))}")

if st.sidebar.button("🔄 Reload DB", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "नमस्कार! म तपाईंको नेपाली कानूनी सहायक हुँ। निर्णय नं., मुद्दा, पक्षकार, दफा वा NKP PDF का बारेमा प्रश्न सोध्नुहोस्।"
    }]


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
    intent = detect_query_intent(cleaned_query)
    identifiers = extract_query_identifiers(cleaned_query)

    with st.chat_message("assistant"):
        st.caption(f"Query type: `{intent}`" + (f" | Case identifier: `{identifiers}`" if identifiers else ""))
        with st.spinner("कानूनी कागजात खोजिँदैछ..."):
            search_results = perform_hybrid_search(
                query=cleaned_query,
                collection=collection,
                model=model,
                bm25=bm25,
                chunk_metadata=chunk_metadata,
                top_k=top_k,
                alpha=alpha,
            )

        with st.spinner("निर्णयका प्रमाणमा आधारित उत्तर तयार हुँदैछ..."):
            final_answer = generate_nepali_answer(cleaned_query, search_results)

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