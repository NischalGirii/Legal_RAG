# ⚖️ Nepali Legal Assistant & Document Chatbot (Local RAG)

An AI-powered Retrieval-Augmented Generation (RAG) system engineered for indexing, searching, and synthesizing information from Nepal Supreme Court precedents (Nepal Law Patrika - NKP) and legal statutes.

Built using **ChromaDB**, **SentenceTransformers**, **BM25**, **PyMuPDF**, **OpenCV + Tesseract OCR**, and **Groq LLM** (`openai/gpt-oss-20b`).

---

## 🌟 Key Features

- **Case-Aware Hybrid Retrieval**: Combines dense vector search (`paraphrase-multilingual-MiniLM-L12-v2`) with sparse keyword matching (`BM25Okapi`) and exact decision/case-number boosting (+0.65 score boost) to prevent case-mixing.
- **OCR Fallback for Scanned PDFs**: Automatically detects low-quality scanned pages and applies OpenCV adaptive thresholding and PyTesseract Devanagari OCR (`nep` / `script/Devanagari`).
- **Query Intent Detection**: Identifies whether a user is requesting a `CASE_SUMMARY`, `CASE_LOOKUP`, or `LEGAL_PROVISION`, and dynamically adapts retrieval depth and LLM prompt structures.
- **Strict Case Boundary Grounding**: Enforces strict `CASE BOUNDARY` separation in the LLM system prompt to prevent hallucination and cross-contamination of facts across different legal cases.
- **Interactive Streamlit Dashboard**: Offers configurable hybrid weights (`α`), citation inspection, vector index statistics, and parent-page context expansion.

---

## 📁 Project Architecture

```text
├── data/
│   └── pdf/                # Raw NKP court decision PDFs
├── models/                 # BM25 index (.pkl) and ingestion metadata (.json)
├── chroma_db/              # Persistent ChromaDB vector database
├── src/
│   ├── text_processor.py   # Devanagari cleaning, digit normalization, sentence chunking
│   ├── ocr_engine.py       # OpenCV preprocessing & PyTesseract OCR fallback
│   ├── hybrid_search.py    # Dense/Sparse fusion & case identifier reranking
│   └── llm_generator.py    # Groq API integration & structured system prompt
├── app.py                  # Streamlit web application dashboard
├── ingest.py               # Data ingestion, OCR processing, chunking & indexing pipeline
├── test_rag.py             # CLI test harness for retrieval and prompt verification
├── .env.example            # Environment variables template
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python**: 3.10 or higher
- **Tesseract OCR**: Installed on your system with Devanagari/Nepali language packs (`nep`, `script/Devanagari`).
  - **Windows**: Install Tesseract OCR and add it to your system `PATH`.
- **Groq API Key**: Obtain an API key from the Groq Cloud Console.

### 2. Installation

Clone the repository and create a virtual environment:

```bash
# Clone repository
git clone https://github.com/nischalgirii/nepali-legal-rag.git
cd nepali-legal-rag

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate on Windows CMD
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Important:** Never commit your `.env` file or API keys to GitHub.

---

## 📚 Data Ingestion & Indexing

Place your PDF legal documents in:

```text
./data/pdf/
```

Then run:

```bash
python ingest.py
```

The ingestion pipeline will:

1. Parse native Devanagari text from PDFs using PyMuPDF.
2. Detect pages with insufficient extracted text.
3. Fall back to OpenCV preprocessing and Tesseract OCR for scanned pages.
4. Extract relevant metadata such as decision number, date, subject, appellant, and respondent.
5. Clean and normalize extracted Nepali text.
6. Split documents into searchable chunks.
7. Generate embeddings using SentenceTransformers.
8. Store embeddings in ChromaDB.
9. Build the BM25 sparse-search index.
10. Save ingestion metadata for later retrieval.

---

## 🧠 Retrieval Pipeline

The system uses a **hybrid retrieval architecture**:

```text
                    User Query
                        │
                        ▼
              ┌──────────────────┐
              │ Query Intent      │
              │ Detection         │
              └────────┬─────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Dense Vector Search     BM25 Search
      SentenceTransformers    Keyword Search
             │                   │
             └─────────┬─────────┘
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
                Groq LLM Response
```

This approach improves retrieval for both:

- Semantic questions
- Exact legal terminology
- Decision numbers
- Case identifiers
- Nepali legal phrases

---

## 🛡️ Grounding & Hallucination Control

The chatbot is designed to keep information separated by legal case.

The generation layer uses strict **case-boundary grounding** so that information retrieved from one case is not incorrectly attributed to another case.

The system should answer based on retrieved legal documents rather than relying on unsupported general knowledge.

---

## 🖥️ Running the Chatbot

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The dashboard provides access to the legal document chatbot and retrieval diagnostics.

---

## 🧪 Testing

Run the CLI diagnostic test suite:

```bash
python test_rag.py
```

This can be used to inspect:

- Retrieval ranking
- Hybrid search behavior
- Case identifier matching
- Prompt grounding
- Generated responses

---

## 💡 Sample Queries

### Case Summary

```text
मुद्दा नम्बर ४ को संक्षिप्त विवरण दिनुहोस्
```

### Decision Lookup

```text
निर्णय नं. ९८७६ फैसला मिति र मुख्य आदेश के हो?
```

### Legal Provision

```text
नेपालको अन्तरिम संविधान २०६३ को धारा ३२ बमोजिमको रिट
```

---

## 🔧 Configuration

The hybrid retrieval system can be tuned according to the dataset.

Important components include:

| Component | Purpose |
|---|---|
| SentenceTransformers | Semantic/vector retrieval |
| BM25Okapi | Sparse keyword retrieval |
| ChromaDB | Persistent vector storage |
| PyMuPDF | PDF text extraction |
| OpenCV | OCR image preprocessing |
| Tesseract OCR | Nepali/Devanagari OCR |
| Groq | LLM generation |
| Streamlit | Web interface |

---

## ⚠️ Limitations

- OCR quality depends on the quality and layout of scanned PDFs.
- Legal terminology may contain OCR-specific spelling errors.
- Retrieval quality depends on chunking, embedding quality, and BM25 configuration.
- The system is a document-grounded research assistant and should not be treated as a substitute for professional legal advice.
- A Groq API key is required for the LLM generation component.

---

## 👤 Author

**Nischal Giri**

*Bachelor of Computer Applications (BCA) Student*

Kathmandu, Nepal

GitHub: https://github.com/nischalgirii

---

## 📄 License

Add the project's preferred license here before publishing the repository publicly.
#   L e g a l _ R A G  
 