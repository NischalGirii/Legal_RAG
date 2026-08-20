# ⚖️ Nepali Legal Assistant & Document Chatbot (Hybrid RAG)

An AI-powered Retrieval-Augmented Generation (RAG) system for indexing, searching, and synthesizing information from **Nepal Supreme Court precedents (Nepal Law Patrika - NKP)** and legal statutes.

Built with:

- **ChromaDB**
- **SentenceTransformers**
- **BM25**
- **PyMuPDF**
- **OpenCV + Tesseract OCR**
- **Groq LLM** (`openai/gpt-oss-20b`)
- **Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red)](https://streamlit.io/)

---

## 🌟 Key Features

### 🔎 Case-Aware Hybrid Retrieval

Combines multiple retrieval strategies:

- Dense vector search using `paraphrase-multilingual-MiniLM-L12-v2`
- Sparse keyword search using `BM25Okapi`
- Exact decision/case-number matching
- Case identifier boosting with a `+0.65` score boost

This helps prevent retrieval from mixing information between unrelated legal cases.

### 📝 OCR Fallback for Scanned PDFs

Automatically detects pages with insufficient extracted text and falls back to OCR using:

- OpenCV image preprocessing
- Tesseract OCR
- Nepali / Devanagari language support

Supported OCR language configurations include:

```text
nep
script/Devanagari
```

### 🧠 Query Intent Detection

The system identifies the user's query intent and adapts retrieval and generation accordingly.

Supported intents:

- `CASE_SUMMARY`
- `CASE_LOOKUP`
- `LEGAL_PROVISION`

### 🛡️ Strict Case Boundary Grounding

The generation layer enforces strict case separation through system-prompt constraints.

This helps prevent:

- Cross-case contamination
- Incorrect attribution of facts
- Mixing decisions from different cases
- Unsupported legal claims

### 🖥️ Interactive Streamlit Dashboard

The Streamlit interface provides:

- Legal document chatbot
- Configurable hybrid retrieval weights (`α`)
- Retrieval diagnostics
- Citation inspection
- Vector index statistics
- Parent-page context expansion

---

# 📁 Project Architecture

```text
nepali_rag_project/
│
├── data/
│   ├── pdf/                 # Raw NKP court decision PDFs
│   └── txt/                 # Extracted raw text files
│
├── models/                  # BM25 index, metadata, case summaries, embeddings
│   ├── paraphrase-multilingual-MiniLM-L12-v2/
│   ├── bm25_index.pkl
│   ├── case_summaries.json
│   └── ingest_metadata.json
│
├── chroma_db/               # Persistent ChromaDB vector database
│
├── src/                     # Core modular components
│   ├── hybrid_search.py     # Sparse (BM25) + Dense retrieval & case reranking
│   ├── llm_generator.py     # Groq API integration & structured grounding prompts
│   ├── ocr_engine.py        # OpenCV image preprocessing & Tesseract OCR
│   └── text_processor.py    # Devanagari cleaning, normalization, & chunking
│
├── app.py                   # Streamlit web application interface
├── ingest.py                # PDF ingestion and vector/sparse indexing pipeline
├── query_analyzer.py        # Intent detection and query classification router
├── test_rag.py              # CLI diagnostic test harness
├── pyproject.toml           # Project dependency configuration
├── uv.lock                  # uv package manager lockfile
├── .env                     # Environment variables (Groq API Key)
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

# 🚀 Getting Started

## 1. Prerequisites

### Python

Python **3.10 or higher** is recommended.

Check your version:

```bash
python --version
```

### Tesseract OCR

Tesseract OCR must be installed with Nepali/Devanagari language support.

### Windows

Install Tesseract OCR and add it to your system `PATH`.

### macOS

```bash
brew install tesseract
brew install tesseract-lang
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-nep
```

### Groq API Key

A Groq API key is required for LLM generation.

---

# ⚙️ Installation

## 2. Clone the Repository

```bash
git clone https://github.com/nischalgirii/nepali-legal-rag.git
cd nepali-legal-rag
```

## 3. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Important:** Never commit your `.env` file or API keys to GitHub.

Make sure `.gitignore` contains:

```gitignore
.env
.venv/
__pycache__/
chroma_db/
data/pdf/
models/
```

---

# 📚 Data Ingestion & Indexing

Place your legal PDF documents inside:

```text
./data/pdf/
```

Then run:

```bash
python ingest.py
```

The ingestion pipeline performs the following operations:

1. Extracts native Nepali/Devanagari text from PDFs using **PyMuPDF**.
2. Detects pages with insufficient extracted text.
3. Falls back to **OpenCV + Tesseract OCR** for scanned pages.
4. Extracts metadata such as:
   - Decision number
   - Decision date
   - Subject
   - Appellant
   - Respondent
5. Cleans and normalizes Nepali text.
6. Splits documents into searchable chunks.
7. Generates embeddings using **SentenceTransformers**.
8. Stores embeddings in **ChromaDB**.
9. Builds the **BM25** sparse-search index.
10. Saves ingestion metadata for later retrieval.

---

# 🧠 Retrieval Pipeline

The system uses a hybrid retrieval architecture combining semantic and lexical retrieval.

```text
                         User Query
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Query Intent      │
                  │     Detection       │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
            Dense Vector Search    BM25 Search
          SentenceTransformers    Keyword Search
                    │                 │
                    └────────┬────────┘
                             ▼
                    Hybrid Score Fusion
                             │
                             ▼
                  Case / Decision Boost
                             │
                             ▼
                    Case Boundary Filter
                             │
                             ▼
                       Top Documents
                             │
                             ▼
                           Groq LLM
                             │
                             ▼
                       Final Answer
```

This architecture improves retrieval for:

- Semantic questions
- Exact legal terminology
- Decision numbers
- Case identifiers
- Nepali legal phrases
- Statutory references

---

# 🔬 Hybrid Retrieval

The final retrieval score combines dense similarity and BM25 relevance.

Conceptually:

```text
Hybrid Score =
    α × Dense Similarity
    +
    (1 - α) × BM25 Score
```

Exact decision or case identifiers receive an additional boost:

```text
Case Match Boost = +0.65
```

This gives exact legal identifiers higher priority when the user explicitly references a case or decision number.

---

# 🛡️ Grounding & Hallucination Control

The chatbot is designed to keep information separated by legal case.

The generation layer uses strict **case-boundary grounding** so that facts retrieved from one case are not incorrectly attributed to another case.

The model is instructed to:

- Use retrieved documents as the primary source of truth.
- Avoid mixing information from different cases.
- Preserve case-specific facts.
- Avoid unsupported claims.
- Respect the detected query intent.
- Indicate when the retrieved documents do not contain sufficient information.

---

# 🖥️ Running the Chatbot

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The dashboard provides access to:

- Legal document search
- Question answering
- Retrieval diagnostics
- Citation inspection
- Hybrid-search configuration
- Index statistics

---

# 🧪 Testing

Run the CLI diagnostic test suite:

```bash
python test_rag.py
```

The test harness can be used to inspect:

- Retrieval ranking
- Hybrid search behavior
- Case identifier matching
- Query intent detection
- Prompt grounding
- Generated responses

---

# 💡 Sample Queries

## Case Summary

```text
मुद्दा नम्बर ४ को संक्षिप्त विवरण दिनुहोस्
```

## Decision Lookup

```text
निर्णय नं. ९८७६ फैसला मिति र मुख्य आदेश के हो?
```

## Legal Provision

```text
नेपालको अन्तरिम संविधान २०६३ को धारा ३२ बमोजिमको रिट
```

---

# 🔧 Technology Stack

| Component | Purpose |
|---|---|
| **Python** | Core application development |
| **SentenceTransformers** | Semantic/vector retrieval |
| **BM25Okapi** | Sparse keyword retrieval |
| **ChromaDB** | Persistent vector storage |
| **PyMuPDF** | PDF text extraction |
| **OpenCV** | OCR image preprocessing |
| **Tesseract OCR** | Nepali/Devanagari OCR |
| **Groq** | LLM generation |
| **Streamlit** | Web interface |

---

# ☁️ Streamlit Cloud Deployment

For Streamlit Cloud deployment, add your API key through:

```text
App Settings → Secrets
```

Use:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Then configure `llm_generator.py` to read the key from `st.secrets` when running on Streamlit Cloud.

---

# ⚠️ Limitations

- OCR quality depends on the quality, resolution, and layout of scanned PDFs.
- Nepali legal terminology may contain OCR-specific spelling errors.
- Retrieval quality depends on:
  - Chunk size
  - Chunk overlap
  - Embedding model
  - BM25 configuration
  - Hybrid-search weighting
- Poorly scanned documents may produce incomplete or noisy text.
- LLM output quality depends on the quality of the retrieved context.

---

# 🔒 Security

Do not commit sensitive credentials or generated indexes to GitHub.

At minimum, add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
chroma_db/
models/
data/pdf/
```

---

# 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

# 👨‍💻 Author

**Nischal Giri**

GitHub: [@nischalgirii](https://github.com/nischalgirii)

---

## ⭐ Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.
