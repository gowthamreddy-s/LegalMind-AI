import streamlit as st
import os, base64, uuid
import pandas as pd
from dotenv import load_dotenv
from groq import Groq as GroqClient
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, BSHTMLLoader
)
from langchain_core.documents import Document

load_dotenv()

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #f5f3ff; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #2d2a6e 60%, #1e1b4b 100%);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #c7d2fe !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5b4fc !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p { color: #a5b4fc !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"]   { color: #e0e7ff !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.08);
    border: 1.5px dashed #818cf8;
    border-radius: 10px;
}
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #c4b5fd !important;
    background: white !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}
[data-testid="stExpander"] {
    border: 1px solid #e0e7ff !important;
    border-radius: 10px !important;
    background: white !important;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Supported file types ───────────────────────────────────────────
SUPPORTED_TYPES = {
    "pdf":  "📄 PDF document",
    "docx": "📝 Word document",
    "doc":  "📝 Word document",
    "txt":  "📃 Text file",
    "md":   "📃 Markdown",
    "csv":  "📊 CSV spreadsheet",
    "xlsx": "📊 Excel spreadsheet",
    "xls":  "📊 Excel spreadsheet",
    "html": "🌐 HTML file",
    "htm":  "🌐 HTML file",
    "png":  "🖼️ Image",
    "jpg":  "🖼️ Image",
    "jpeg": "🖼️ Image",
    "webp": "🖼️ Image",
}

# ── Cache resources ────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

# ── Load vectorstore by collection name ───────────────────────────
# Takes collection_name as argument so cache busts when name changes
@st.cache_resource
def load_vectorstore(_collection_name="documents"):
    if not os.path.exists("chroma_db"):
        return None
    try:
        return Chroma(
            persist_directory="chroma_db",
            embedding_function=load_embedder(),
            collection_name=_collection_name
        )
    except Exception:
        return None

embedder = load_embedder()
llm      = load_llm()
groq_raw = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

# ── Load any file type ─────────────────────────────────────────────
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()

    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path).load()

    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8").load()

    elif ext in [".html", ".htm"]:
        # Try UTF-8 first, fall back to latin-1 which never fails
        try:
            return BSHTMLLoader(file_path, open_encoding="utf-8").load()
        except Exception:
            return BSHTMLLoader(file_path, open_encoding="latin-1").load()

    elif ext == ".csv":
        return CSVLoader(file_path).load()

    elif ext in [".xlsx", ".xls"]:
    # Read all sheets
        xl = pd.ExcelFile(file_path)
        all_text = []

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            df = df.fillna("")  # replace NaN with empty string

            # Convert each row to "Column: Value, Column: Value" format
            # This makes it readable for the AI unlike raw table format
            sheet_text = f"Sheet: {sheet_name}\n"
            sheet_text += f"Columns: {', '.join(df.columns.astype(str))}\n\n"

            for _, row in df.iterrows():
                row_text = " | ".join(
                    f"{col}: {val}"
                    for col, val in zip(df.columns, row)
                    if str(val).strip()
                )
                if row_text:
                    sheet_text += row_text + "\n"

            all_text.append(sheet_text)

        full_text = "\n\n".join(all_text)
        return [Document(
            page_content=full_text,
            metadata={"source": file_path, "type": "excel"}
        )]

    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] \
                     else f"image/{ext.strip('.')}"

        response = groq_raw.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_data}"}
                    },
                    {
                        "type": "text",
                        "text": """Describe this image in complete detail.
Include ALL visible text, numbers, tables, charts, diagrams, and labels.
If it is a document or screenshot, transcribe the text exactly.
Be thorough — this description will be used to answer questions."""
                    }
                ]
            }]
        )
        description = response.choices[0].message.content
        return [Document(
            page_content=f"[Image described by AI vision]:\n\n{description}",
            metadata={"source": file_path, "type": "image"}
        )]

    else:
        return TextLoader(file_path, encoding="utf-8").load()


# ── Ingest — uses unique collection name to avoid Windows lock ─────
# Windows locks chroma_db files while ChromaDB has an open connection.
# Deleting a locked folder causes PermissionError.
# Solution: never delete — create a new collection with a unique name
# each time. Old collections stay on disk harmlessly.
def ingest_file(file_path):
    docs = load_document(file_path)
    if not docs:
        return 0, 0

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    ).split_documents(docs)

    # Unique collection name — no deletion needed, no Windows lock error
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    st.session_state.collection_name = collection_name

    Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory="chroma_db",
        collection_name=collection_name
    )

    # Clear cache so load_vectorstore picks up new collection
    load_vectorstore.clear()
    return len(chunks), len(docs)


# ── Helpers ────────────────────────────────────────────────────────
def to_confidence(distance):
    return round(max(0, (1 - distance / 2) * 100), 1)

def rewrite_query(question):
    chain = ChatPromptTemplate.from_messages([
        ("system", "Rewrite as a short keyword search query for a document. Return ONLY the query, nothing else."),
        ("human", "{question}")
    ]) | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()

def confidence_bar(score):
    color = "#10b981" if score > 70 else "#f59e0b" if score > 40 else "#ef4444"
    label = "Strong" if score > 70 else "Moderate" if score > 40 else "Weak"
    st.markdown(f"""
    <div style="margin:10px 0 6px">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px">
        <span style="font-size:12px;color:#6b7280;font-weight:500">Retrieval confidence</span>
        <span style="font-size:12px;font-weight:600;color:{color}">{score:.1f}% — {label} match</span>
      </div>
      <div style="background:#e5e7eb;border-radius:6px;height:7px;overflow:hidden">
        <div style="background:linear-gradient(90deg,{color}88,{color});
                    width:{score}%;height:100%;border-radius:6px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def source_card(i, doc, score):
    color = "#10b981" if score > 70 else "#f59e0b" if score > 40 else "#ef4444"
    page  = doc.metadata.get("page", "—")
    text  = doc.page_content[:280].replace("\n", " ")
    st.markdown(f"""
    <div style="border:1px solid #e5e7eb;border-left:4px solid {color};
                border-radius:10px;padding:14px 16px;margin:6px 0;background:white;
                box-shadow:0 1px 4px rgba(0,0,0,0.05)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-weight:600;color:#1e1b4b;font-size:13px">Source {i+1}</span>
        <span style="background:{color}18;color:{color};font-size:11px;
                     padding:2px 10px;border-radius:20px;font-weight:600">
          {score:.0f}% · Page {page}
        </span>
      </div>
      <p style="font-size:12px;color:#6b7280;margin:0;font-family:monospace;
                line-height:1.6;overflow:hidden;display:-webkit-box;
                -webkit-line-clamp:3;-webkit-box-orient:vertical">{text}…</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem">
      <div style="font-size:2.5rem">🧠</div>
      <div style="font-size:1.1rem;font-weight:600;color:#e0e7ff;margin-top:4px">DocuMind AI</div>
      <div style="font-size:11px;color:#818cf8;margin-top:2px">LangChain + Groq</div>
    </div>
    <hr style="border-color:#3730a3;margin:1rem 0">
    """, unsafe_allow_html=True)

    with st.expander("Supported file types", expanded=False):
        st.markdown("""
        <div style="font-size:12px;line-height:2;color:#a5b4fc">
        📄 PDF &nbsp;·&nbsp; 📝 Word (.docx)<br>
        📊 CSV &nbsp;·&nbsp; 📊 Excel (.xlsx)<br>
        📃 TXT &nbsp;·&nbsp; 📃 Markdown<br>
        🌐 HTML &nbsp;·&nbsp; 🖼️ Images (JPG/PNG)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Upload Document")
    uploaded = st.file_uploader(
        "Drop any file here",
        type=list(SUPPORTED_TYPES.keys()),
        label_visibility="collapsed"
    )

    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join(
        "uploads",
        f"uploaded_{uploaded.name}"
    )
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
        

        ext   = os.path.splitext(uploaded.name)[1].lower().strip(".")
        label = SUPPORTED_TYPES.get(ext, "📁 File")
        spinner_msg = "Reading image with vision AI..." \
                      if ext in ["png","jpg","jpeg","webp"] \
                      else "Processing with LangChain..."

        with st.spinner(spinner_msg):
            n_chunks, n_pages = ingest_file(save_path)
        os.remove(save_path)

        if n_chunks:
            st.success(f"✅ {n_chunks} chunks indexed")
            st.session_state.doc_name = uploaded.name
            st.session_state.doc_type = label
            st.session_state.messages = []
        else:
            st.error("Failed to process file.")

    # Use collection name from session to load correct vectorstore
    collection_name = st.session_state.get("collection_name", "documents")
    vectorstore = load_vectorstore(collection_name)

    if vectorstore:
        st.markdown("<hr style='border-color:#3730a3;margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown("### Document Info")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Chunks", vectorstore._collection.count())
        with c2:
            st.metric("Model", "BGE")

        if "doc_name" in st.session_state:
            doc_type = st.session_state.get("doc_type", "📁")
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.15);border-radius:8px;
                        padding:8px 12px;margin:6px 0">
              <span style="font-size:12px;color:#a5b4fc">
                {doc_type} — {st.session_state.doc_name}
              </span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#3730a3;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("### Settings")
    n_results    = st.slider("Chunks to retrieve", 3, 15, 10)
    show_chunks  = st.toggle("Show source chunks", value=False)
    show_rewrite = st.toggle("Show rewritten query", value=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <hr style='border-color:#3730a3;margin:1rem 0'>
    <div style="font-size:11px;color:#6366f1;text-align:center">
      Skill 3 · LangChain · Agentic AI path
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#9333ea 100%);
            padding:1.8rem 2rem;border-radius:16px;margin-bottom:1.5rem;
            box-shadow:0 8px 32px rgba(99,102,241,0.3)">
  <h1 style="color:white;margin:0;font-size:1.9rem;font-weight:600;letter-spacing:-0.02em">
    🧠 DocuMind AI
  </h1>
  <p style="color:rgba(255,255,255,0.75);margin:0.4rem 0 0;font-size:0.95rem">
    Chat with PDF · Word · CSV · Excel · HTML · Images — powered by RAG + LangChain
  </p>
</div>
""", unsafe_allow_html=True)

# Load vectorstore using current collection name
collection_name = st.session_state.get("collection_name", "documents")
vectorstore = load_vectorstore(collection_name)

if not vectorstore:
    st.markdown("""
    <div style="background:white;border:1.5px dashed #c4b5fd;border-radius:14px;
                padding:3rem;text-align:center;margin-top:2rem">
      <div style="font-size:3rem;margin-bottom:1rem">📂</div>
      <div style="font-size:1.1rem;font-weight:600;color:#4f46e5;margin-bottom:0.5rem">
        No document loaded
      </div>
      <div style="font-size:0.9rem;color:#6b7280;margin-bottom:1rem">
        Upload a file in the sidebar or run <code>python ingest.py</code>
      </div>
      <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap">
        <span style="background:#ede9fe;color:#5b21b6;padding:4px 12px;border-radius:20px;font-size:12px">📄 PDF</span>
        <span style="background:#ede9fe;color:#5b21b6;padding:4px 12px;border-radius:20px;font-size:12px">📝 Word</span>
        <span style="background:#ede9fe;color:#5b21b6;padding:4px 12px;border-radius:20px;font-size:12px">📊 CSV</span>
        <span style="background:#ede9fe;color:#5b21b6;padding:4px 12px;border-radius:20px;font-size:12px">📊 Excel</span>
        <span style="background:#ede9fe;color:#5b21b6;padding:4px 12px;border-radius:20px;font-size:12px">🌐 HTML</span>
        <span style="background:#ede9fe;color:#5b21b6;padding:4px 12px;border-radius:20px;font-size:12px">🖼️ Image</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────
question = st.chat_input("Ask anything about your document...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # STEP 1: Rewrite query
    search_query = rewrite_query(question)
    if show_rewrite:
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:#ede9fe;padding:5px 12px;border-radius:20px;margin-bottom:8px">
          <span style="font-size:12px;color:#5b21b6">
            🔍 Search query: <em>{search_query}</em>
          </span>
        </div>""", unsafe_allow_html=True)

    # STEP 2: Retrieve with confidence scores
    docs_with_scores = vectorstore.similarity_search_with_score(
        search_query, k=n_results
    )
    docs   = [d for d, _ in docs_with_scores]
    scores = [to_confidence(s) for _, s in docs_with_scores]
    avg    = round(sum(scores) / len(scores), 1) if scores else 0

    # STEP 3: Confidence bar + warning
    confidence_bar(avg)
    if avg < 35:
        st.warning("⚠️ Low confidence — document may not contain a strong answer.")

    # STEP 4: Source chunks
    if show_chunks:
        with st.expander(f"📚 {len(docs)} sources · avg {avg:.1f}%"):
            for i, (doc, score) in enumerate(zip(docs, scores)):
                source_card(i, doc, score)

    # STEP 5: Build context with citations
    context = "\n\n".join(
        f"[Source {i+1}] Page {doc.metadata.get('page','—')}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # STEP 6: LangChain LCEL chain
    chain = ChatPromptTemplate.from_messages([
        ("system", """Answer using ONLY the sources below.
Cite inline as [Source N]. Be thorough and include all relevant details.
If not found say exactly: "I couldn't find that in the document."
Never use outside knowledge.

{context}"""),
        ("human", "{question}")
    ]) | llm | StrOutputParser()

    # STEP 7: Stream answer
    with st.chat_message("assistant"):
        reply = st.write_stream(
            chain.stream({"context": context, "question": question})
        )
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
          <span style="font-size:11px;background:#ede9fe;color:#5b21b6;
                       padding:2px 10px;border-radius:20px">
            {'🟢' if avg>70 else '🟡' if avg>40 else '🔴'} {avg:.1f}% confidence
          </span>
          <span style="font-size:11px;background:#f0fdf4;color:#166534;
                       padding:2px 10px;border-radius:20px">
            {len(docs)} sources
          </span>
          <span style="font-size:11px;background:#faf5ff;color:#6b21a8;
                       padding:2px 10px;border-radius:20px">
            LangChain + Groq
          </span>
        </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": reply})