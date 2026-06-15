# 🧠 DocuMind AI — Multi-Format RAG Document Assistant

## 📌 Overview

DocuMind AI is an intelligent Retrieval-Augmented Generation (RAG) application that allows users to upload documents and interact with them using natural language.

The application supports multiple document formats including PDFs, Word documents, spreadsheets, HTML files, text files, and images. Uploaded content is processed, embedded, stored in a vector database, and retrieved through semantic search to generate accurate, source-grounded responses.

Built with LangChain, ChromaDB, Hugging Face Embeddings, Groq LLMs, and Streamlit, DocuMind AI provides fast, citation-based answers with confidence scoring and real-time response streaming.

---

## 🚀 Live Demo

**Streamlit App:**  
https://multiformat-rag-intelligence.streamlit.app

---

## ✨ Features

### 📂 Multi-Format Document Support
- PDF (.pdf)
- Word (.docx, .doc)
- Text (.txt)
- Markdown (.md)
- CSV (.csv)
- Excel (.xlsx, .xls)
- HTML (.html, .htm)
- Images (.png, .jpg, .jpeg, .webp)

### 🔍 Intelligent Retrieval-Augmented Generation (RAG)
- Automatic document chunking
- Semantic embeddings using BGE Small
- ChromaDB vector storage
- Similarity-based retrieval
- Context-aware question answering
- Source-grounded responses

### 🖼️ Vision AI for Images
- Reads screenshots, charts, diagrams, and scanned documents
- Extracts visible text and visual information
- Converts image content into searchable knowledge

### 📊 Confidence-Based Retrieval
- Query rewriting for improved search accuracy
- Retrieval confidence scoring
- Strong / Moderate / Weak confidence indicators
- Source chunk inspection

### 💬 Interactive Chat Experience
- Real-time response streaming
- Session-based conversation memory
- Source citations
- Modern Streamlit interface
- Document information dashboard

---

## 🏗️ Architecture

### Key Components

- **Document Processing Layer** – Extracts content from PDFs, Word documents, spreadsheets, HTML files, text files, and images.
- **Embedding Layer** – Converts document chunks into vector embeddings using BAAI/bge-small-en-v1.5.
- **Vector Database** – Stores embeddings in ChromaDB for efficient semantic retrieval.
- **Retrieval Layer** – Rewrites user queries and retrieves the most relevant document chunks.
- **Generation Layer** – Uses Groq Llama models to generate context-aware answers with source citations.

### RAG Pipeline Flow

```text
User Uploads Document
         │
         ▼
Document Processing
(PDF, Word, CSV, Excel, HTML, Images)
         │
         ▼
Text Chunking
         │
         ▼
Embeddings Generation
(BAAI/bge-small-en-v1.5)
         │
         ▼
ChromaDB Vector Store
         │
         ▼
User Question
         │
         ▼
Query Rewriting
         │
         ▼
Similarity Search
         │
         ▼
Relevant Chunks Retrieved
         │
         ▼
Groq Llama Model
         │
         ▼
Answer + Citations + Confidence Score
```
---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python 3.11 | Core programming language |
| Streamlit | Web application framework |
| LangChain | RAG pipeline framework |
| Groq | LLM inference (Llama 3.1) |
| ChromaDB | Vector database |
| Hugging Face Embeddings | Semantic embeddings |
| BAAI/bge-small-en-v1.5 | Embedding model |
| Pandas | Excel processing |
| BeautifulSoup | HTML parsing |
| Docx2txt | Word document extraction |
| Python Dotenv | Environment management |
| docx2txt, pypdf | File parsing |

---

## 🚀 Deployment

This application is deployed using **Streamlit Community Cloud**.

### 📌 Deployment Steps:

- Connected GitHub repository to Streamlit Cloud  
- Configured **Streamlit Secrets** for secure API key storage (`GROQ_API_KEY`)  
- Enabled **auto-deployment** from the `main` branch  
- Every push to GitHub automatically updates the live app

---

## 📂 Project Structure

```text
DocuMind-AI/
│
├── app.py               # Main Streamlit application and RAG workflow
├── ingest.py            # Document ingestion and vector database creation
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
├── .gitignore           # Excludes sensitive and generated files
├── .env                 # Environment variables (not committed)
│
├── uploads/             # Temporary uploaded documents
│
└── chroma_db/           # Persistent ChromaDB vector storage
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd DocuMind-AI
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
```

### Run Application

```bash
streamlit run app.py
```

---

## 📄 Supported File Types

| Category | Extensions |
|-----------|-----------|
| PDF | .pdf |
| Word | .docx, .doc |
| Text | .txt, .md |
| CSV | .csv |
| Excel | .xlsx, .xls |
| HTML | .html, .htm |
| Images | .png, .jpg, .jpeg, .webp |

---

## 🔄 Workflow

1. Upload a document through the interface
2. Content is extracted and processed
3. Text is split into semantic chunks
4. Chunks are converted into vector embeddings
5. Embeddings are stored in ChromaDB
6. User submits a question
7. Query is rewritten for better retrieval
8. Relevant chunks are retrieved
9. Context is sent to Groq LLM
10. Response is generated with citations and confidence scores

---

## 🎯 Key Capabilities

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Document Question Answering
- Multi-Format File Processing
- Vision-Based Document Understanding
- Source Attribution
- Confidence Scoring
- Real-Time Streaming Responses

---

## 🔒 Security

- API keys managed using environment variables
- Sensitive credentials excluded via `.gitignore`
- No hardcoded secrets

---

## 🚀 Future Enhancements

- Persistent chat memory
- Hybrid Search (BM25 + Vector Search)
- Web Search Integration
- Agentic RAG Workflows
- LangGraph Integration
- User Authentication
- Cloud Vector Database Support

---

## 📚 Learning Outcomes

**Through this project, I gained practical experience in:**

- Building end-to-end Retrieval-Augmented Generation (RAG) applications.
- Implementing semantic search using vector embeddings and ChromaDB.
- Working with LangChain components such as document loaders, text splitters, prompts, and LCEL chains.
- Integrating Groq LLMs for fast inference and response generation.
- Processing multiple document formats including PDFs, Word files, spreadsheets, HTML pages, and images.
- Implementing query rewriting and confidence-based retrieval.
- Developing interactive AI applications using Streamlit.
- Managing embeddings, vector databases, and document retrieval pipelines.
- Deploying production-ready AI applications on Streamlit Cloud.

---

## 👨‍💻 Author

**Gowtham Reddy S**

MSc Data Science | AI/ML Engineer | Generative AI & Agentic AI Enthusiast

**LinkedIn:** https://www.linkedin.com/in/gowtham-reddy-s-9797a625a

---

⭐ If you found this project useful, consider giving it a star.