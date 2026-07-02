# ⚖️ LegalMind AI — RAG-Based Legal Document Intelligence

## 📌 Overview

LegalMind AI is a production-grade **Retrieval-Augmented Generation (RAG)** platform built specifically for Indian advocates and law firms.

It allows lawyers to upload legal documents — contracts, agreements, property deeds, court notices — and interact with them through natural language chat, auto-generated FAQs, and multi-document comparison.

Every answer is generated **strictly from the uploaded document content** with page-level citations — zero hallucination, zero external knowledge. Built with LangChain, Qdrant Cloud, Supabase Auth, Groq LLMs, and Streamlit.

---

## 🚀 Live Demo

**Streamlit App:**  
*Coming soon — deployment in progress*

---

## ✨ Features

### 📂 Multi-Format Document Support
- PDF (.pdf)
- Word (.docx, .doc)
- Text (.txt, .md)
- CSV (.csv)
- Excel (.xlsx, .xls)
- HTML (.html, .htm)
- Images (.png, .jpg, .jpeg, .webp)

### 💬 Chat with Documents (Strict RAG)
- Ask questions in natural language
- Answers come **only** from uploaded documents
- Page-level source citations on every answer
- Retrieval confidence scoring (Strong / Moderate / Low)
- Per-document persistent chat history
- Real-time response streaming

### ❓ Auto FAQ Generation
- Generates 6 FAQs automatically after upload
- Adapts to document type (contract, policy, report, legal notice)
- FAQs saved per document — regenerate anytime
- Strictly based on document content only

### 🔍 Multi-Document Comparison
- Compare 2–5 documents simultaneously
- Full structured analysis:
  - Executive Summary
  - Document A Overview
  - Document B Overview
  - Similarities
  - Differences
  - Key Findings
  - Conclusion
- Specific question mode — ask targeted comparison questions
- Previous comparison results saved per session

### 📂 Private Document Library
- Each user gets their own isolated document library
- Documents persist across sessions
- Upload once — access forever
- Permanent delete with full cleanup

### 🔒 User Authentication
- Secure email + password login via Supabase Auth
- Each advocate sees only their own documents
- Row Level Security enforced
- Session management with logout

### 📊 Confidence-Based Retrieval
- Retrieval confidence score on every answer
- Junk chunk filtering during ingestion and retrieval
- Source chunk inspection toggle
- Warning shown for low confidence answers

### 🖼️ Vision AI for Images
- Reads screenshots, charts, scanned documents
- Extracts all visible text and visual information
- Converts image content into searchable knowledge

---

## 🏗️ Architecture

### Key Components

- **Auth Layer** — Supabase Auth with per-user data isolation
- **Document Processing Layer** — Extracts content from all file types
- **Embedding Layer** — BAAI/bge-small-en-v1.5 via HuggingFace
- **Vector Database** — Qdrant Cloud (per-document collections)
- **Retrieval Layer** — Semantic search with junk chunk filtering
- **Generation Layer** — Groq Llama 3.1 70B with strict RAG prompt
- **Library Layer** — Per-user JSON document registry

### RAG Pipeline Flow

```text
User Login (Supabase Auth)
         │
         ▼
Upload Document
(PDF, Word, CSV, Excel, HTML, Images)
         │
         ▼
Text Extraction
         │
         ▼
Chunking + Junk Filtering
(500 tokens, 50 overlap)
         │
         ▼
BGE Embeddings Generation
(BAAI/bge-small-en-v1.5)
         │
         ▼
Qdrant Cloud Vector Store
(per-document collection)
         │
         ▼
User Question
         │
         ▼
Semantic Similarity Search
(top-16 → filter → top-8)
         │
         ▼
Strict RAG Prompt
(no external knowledge allowed)
         │
         ▼
Groq Llama 3.1 70B
         │
         ▼
Answer + Page Citations + Confidence Score
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11 | Core programming language |
| Streamlit | Web application framework |
| LangChain | RAG pipeline framework |
| Groq | LLM inference (llama-3.1-70b-versatile) |
| Qdrant Cloud | Cloud vector database |
| Supabase | User authentication + Row Level Security |
| HuggingFace Embeddings | Semantic embeddings |
| BAAI/bge-small-en-v1.5 | Embedding model |
| Pandas | Excel/CSV processing |
| BeautifulSoup | HTML parsing |
| Docx2txt | Word document extraction |
| fpdf2 | PDF generation |
| Python Dotenv | Environment management |

---

## 🚀 Deployment

This application is deployed using **Streamlit Community Cloud**.

### 📌 Deployment Steps

- Connected GitHub repository to Streamlit Cloud
- Configured **Streamlit Secrets** for secure API key storage
- Enabled **auto-deployment** from the `main` branch
- Every push to GitHub automatically updates the live app

---

## 📂 Project Structure

```text
LegalMind-AI/
│
├── app.py                  # Main Streamlit application and RAG workflow
├── ingest.py               # CLI document ingestion script (optional)
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Excludes sensitive and generated files
├── .env                    # Environment variables (not committed)
│
├── .streamlit/
│   └── config.toml         # Streamlit configuration
│
├── docs/
│   ├── LegalMind_AI_Documentation.html   # Full technical documentation
│   ├── LegalMind_AI_Report.html          # Project portfolio page
│   └── LegalMind_AI_Pitch.md             # Short project pitch
│
└── uploads/                # Temporary uploaded documents (auto-created)
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/gowthamreddy-s/LegalMind-AI.git
cd LegalMind-AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Supabase Setup

Run this SQL in your Supabase SQL Editor:

```
sql
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT NOT NULL,
    full_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name)
    VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'full_name');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
```

### Run Application

```bash
streamlit run app.py
```

---

## 📄 Supported File Types

| Category | Extensions |
|----------|-----------|
| PDF | .pdf |
| Word | .docx, .doc |
| Text | .txt, .md |
| CSV | .csv |
| Excel | .xlsx, .xls |
| HTML | .html, .htm |
| Images | .png, .jpg, .jpeg, .webp |

---

## 🔄 Workflow

1. User signs up / logs in via Supabase Auth
2. Upload a legal document through the sidebar
3. Click Index Document — text extracted, chunked, embedded
4. Vectors stored in Qdrant Cloud (per-document collection)
5. Select document from library — chat, FAQ, or compare
6. Ask a question — semantic search retrieves top chunks
7. Strict RAG prompt sent to Groq Llama 3.1 70B
8. Answer streamed with page citations and confidence score

---

## 🎯 Key Capabilities

- Strict Retrieval-Augmented Generation (RAG)
- Zero Hallucination Architecture
- Per-User Document Isolation
- Semantic Vector Search (Qdrant Cloud)
- Multi-Document Comparison
- Auto FAQ Generation
- Vision-Based Document Understanding
- Source Attribution with Page Citations
- Confidence Scoring
- Real-Time Streaming Responses
- Persistent Chat History per Document

---

## 🔒 Security

- API keys managed via environment variables and Streamlit Secrets
- Supabase Row Level Security — users see only their own data
- Per-user document library isolation
- Sensitive credentials excluded via `.gitignore`
- No hardcoded secrets anywhere

---

## 🚀 Production Roadmap

- [ ] Hybrid Search (BM25 + Semantic Vector Search)
- [ ] FastAPI backend
- [ ] Next.js + React frontend
- [ ] OpenAI GPT-4o integration
- [ ] AWS S3 file storage
- [ ] Razorpay payment integration
- [ ] Mobile app
- [ ] Indian legal corpus fine-tuning
- [ ] Multi-language support (Hindi, Telugu, Tamil)

---

## 📚 Learning Outcomes

**Through this project, I gained practical experience in:**

- Building production-grade RAG applications for legal domain
- Implementing strict no-hallucination RAG architecture
- Integrating Qdrant Cloud for scalable vector storage
- Implementing per-user data isolation with Supabase Auth
- Building multi-document comparison pipelines
- Working with LangChain — loaders, splitters, prompts, LCEL chains
- Integrating Groq LLMs for fast inference and streaming
- Processing multiple document formats including images via Vision AI
- Implementing confidence-based retrieval with junk chunk filtering
- Deploying production AI applications on Streamlit Cloud

---

## 📖 Documentation

Full technical documentation available in the `docs/` folder:

- `LegalMind_AI_Documentation.html` — Complete technical documentation
- `LegalMind_AI_Report.html` — Project portfolio and showcase
- `LegalMind_AI_Pitch.md` — Short project pitch

---

## 👨‍💻 Author

**Gowtham Reddy S**

MSc Data Science | AI Engineer | Generative AI & Legal Tech

**LinkedIn:** https://www.linkedin.com/in/gowtham-reddy-s-9797a625a

---

⭐ If you found this project useful, consider giving it a star.
