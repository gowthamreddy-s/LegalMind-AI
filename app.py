import streamlit as st
import os, base64, uuid, json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq as GroqClient
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, BSHTMLLoader
)
from langchain_core.documents import Document
from supabase import create_client, Client

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Page config — MUST BE FIRST STREAMLIT COMMAND ──────────────────
st.set_page_config(
    page_title="LegalMind AI — Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Qdrant client ───────────────────────────────────────────────────
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )

qdrant_client = get_qdrant_client()

# ── Supabase client ─────────────────────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    return create_client(url, key)

supabase = get_supabase()

# ── CSS (keep your existing style, just update colors/name) ─────────
st.markdown("""
<style>
.stApp { background: #f0f4ff; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #94a3b8 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(29,78,216,0.4) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(29,78,216,0.06);
    border: 1.5px dashed #3b82f6;
    border-radius: 10px;
}
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #93c5fd !important;
    background: white !important;
}
#MainMenu, footer { visibility: hidden; }
.doc-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #1d4ed8;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
}
.doc-card-active {
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-left: 4px solid #1d4ed8;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# AUTH — Login / Signup
# ══════════════════════════════════════════════════════════════════════
def show_auth_page():
    st.markdown("""
    <div style="max-width:400px;margin:4rem auto">
      <div style="text-align:center;margin-bottom:2rem">
        <div style="font-size:3rem">⚖️</div>
        <h1 style="font-size:1.8rem;font-weight:700;color:#1e3a8a;margin:0">
          LegalMind AI
        </h1>
        <p style="color:#64748b;margin-top:0.4rem">
          Legal Document Intelligence
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            st.markdown("#### Welcome back")
            email    = st.text_input("Email", key="login_email",
                                     placeholder="your@email.com")
            password = st.text_input("Password", type="password",
                                     key="login_password",
                                     placeholder="Your password")

            if st.button("Login", use_container_width=True, type="primary"):
                if email and password:
                    try:
                        res = supabase.auth.sign_in_with_password({
                            "email": email,
                            "password": password
                        })
                        st.session_state.user    = res.user
                        st.session_state.session = res.session
                        st.success("✅ Logged in successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {str(e)}")
                else:
                    st.warning("Please enter email and password.")

        with tab2:
            st.markdown("#### Create account")
            full_name    = st.text_input("Full Name", key="signup_name",
                                          placeholder="Advocate Ramesh Kumar")
            email_signup = st.text_input("Email", key="signup_email",
                                          placeholder="your@email.com")
            password_signup = st.text_input("Password", type="password",
                                             key="signup_password",
                                             placeholder="Min 6 characters")

            if st.button("Create Account", use_container_width=True,
                         type="primary"):
                if full_name and email_signup and password_signup:
                    if len(password_signup) < 6:
                        st.warning("Password must be at least 6 characters.")
                    else:
                        try:
                            res = supabase.auth.sign_up({
                                "email": email_signup,
                                "password": password_signup,
                                "options": {
                                    "data": {"full_name": full_name}
                                }
                            })
                            st.success("✅ Account created! Please login.")
                        except Exception as e:
                            st.error(f"Signup failed: {str(e)}")
                else:
                    st.warning("Please fill all fields.")


# ── Check auth state ─────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user    = None
    st.session_state.session = None

# Show login page if not logged in
if not st.session_state.user:
    show_auth_page()
    st.stop()

# Get current user
current_user    = st.session_state.user
current_user_id = current_user.id
current_email   = current_user.email

# ── Constants ────────────────────────────────────────────────────────
CHROMA_DIR   = "chroma_db"
UPLOAD_DIR   = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

SUPPORTED_TYPES = {
    "pdf":"📄 PDF","docx":"📝 Word","doc":"📝 Word",
    "txt":"📃 Text","md":"📃 Markdown","csv":"📊 CSV",
    "xlsx":"📊 Excel","xls":"📊 Excel",
    "html":"🌐 HTML","htm":"🌐 HTML",
    "png":"🖼️ Image","jpg":"🖼️ Image",
    "jpeg":"🖼️ Image","webp":"🖼️ Image",
}

# ── Document Library (Supabase PostgreSQL)  ────────
def load_library() -> dict:
    try:
        user = st.session_state.get("user")
        if not user:
            return {}
        result = supabase.table("documents")\
            .select("*")\
            .eq("user_id", user.id)\
            .execute()
        lib = {}
        for doc in result.data:
            lib[doc["id"]] = {
                "id": doc["id"],
                "name": doc["name"],
                "file_type": doc["file_type"],
                "chunk_count": doc["chunk_count"],
                "page_count": doc["page_count"],
                "collection_name": doc["collection_name"],
                "uploaded_at": doc["uploaded_at"],
                "faqs": doc["faqs"] or []
            }
        return lib
    except Exception as e:
        print(f"Error loading library: {e}")
        return {}

def save_library(lib: dict):
    pass  # Now handled by Supabase directly

def add_to_library(doc_id, name, file_type, chunk_count, page_count, collection_name):
    try:
        user = st.session_state.get("user")
        if not user:
            return
        supabase.table("documents").insert({
            "id": doc_id,
            "user_id": user.id,
            "name": name,
            "file_type": file_type,
            "chunk_count": chunk_count,
            "page_count": page_count,
            "collection_name": collection_name,
        }).execute()
    except Exception as e:
        print(f"Error adding to library: {e}")

def delete_from_library(doc_id):
    try:
        lib = load_library()
        if doc_id in lib:
            col_name = lib[doc_id]["collection_name"]
            try:
                qdrant_client.delete_collection(col_name)
            except Exception:
                pass
        supabase.table("documents")\
            .delete()\
            .eq("id", doc_id)\
            .execute()
    except Exception as e:
        print(f"Error deleting: {e}")

# ── Cache resources ──────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",   # upgraded model
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
        max_tokens=2048,                    # 0 = consistent legal answers
    )

def get_vectorstore(collection_name: str):
    try:
        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=load_embedder(),
            content_payload_key="page_content",
        )
    except Exception:
        return None

embedder = load_embedder()
llm      = load_llm()
groq_raw = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

# ── File loaders (keep your exact working code) ──────────────────────
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path).load()
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8").load()
    elif ext in [".html", ".htm"]:
        try:
            return BSHTMLLoader(file_path, open_encoding="utf-8").load()
        except Exception:
            return BSHTMLLoader(file_path, open_encoding="latin-1").load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(file_path)
        all_text = []
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name).fillna("")
            sheet_text = f"Sheet: {sheet_name}\n"
            sheet_text += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
            for _, row in df.iterrows():
                row_text = " | ".join(
                    f"{col}: {val}" for col, val in zip(df.columns, row) if str(val).strip()
                )
                if row_text:
                    sheet_text += row_text + "\n"
            all_text.append(sheet_text)
        return [Document(page_content="\n\n".join(all_text),
                         metadata={"source": file_path, "type": "excel"})]
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        media_type = f"image/{'jpeg' if ext in ['.jpg','.jpeg'] else ext.strip('.')}"
        response = groq_raw.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                {"type": "text",
                 "text": "Describe this image in complete detail. Include ALL visible text, numbers, tables, charts. If it is a document, transcribe the text exactly."}
            ]}]
        )
        return [Document(
            page_content=f"[Image AI description]:\n\n{response.choices[0].message.content}",
            metadata={"source": file_path, "type": "image"}
        )]
    else:
        return TextLoader(file_path, encoding="utf-8").load()


# ── Ingest into named collection ─────────────────────────────────────
def ingest_file(file_path, original_name):
    import time
    from qdrant_client.models import PointStruct

    docs = load_document(file_path)
    if not docs:
        return None

    # Smaller chunks for faster upload
    raw_chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    ).split_documents(docs)

    # Filter junk chunks
    chunks = [
        c for c in raw_chunks
        if len(c.page_content.strip()) > 100
        and len(set(c.page_content.split())) > 10
    ]

    print(f"   Total chunks after filter: {len(chunks)}")

    doc_id          = str(uuid.uuid4())[:8]
    collection_name = f"legal_{doc_id}"

    # Create Qdrant collection
    try:
        qdrant_client.get_collection(collection_name)
        print(f"   Collection exists: {collection_name}")
    except Exception:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        print(f"   Created collection: {collection_name}")

    # Generate embeddings
    print("   Generating embeddings...")
    texts     = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vectors   = embedder.embed_documents(texts)
    print(f"   Embeddings done: {len(vectors)}")

    # Upload in small batches with retry
    batch_size    = 10
    total_batches = (len(chunks) - 1) // batch_size + 1
    max_retries   = 3

    for i in range(0, len(chunks), batch_size):
        batch_texts    = texts[i:i + batch_size]
        batch_vectors  = vectors[i:i + batch_size]
        batch_metadata = metadatas[i:i + batch_size]

        points = [
            PointStruct(
                id=i + idx,
                vector=vec,
                payload={
                    "page_content": txt,
                    "page": meta.get("page", meta.get("page_label", "")),
                    "source": meta.get("source", ""),
                    "metadata": meta
                }
            )
            for idx, (txt, vec, meta) in enumerate(
                zip(batch_texts, batch_vectors, batch_metadata)
            )
        ]

        # Retry logic
        for attempt in range(max_retries):
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    timeout=120
                )
                print(f"   ✅ Batch {i//batch_size + 1}/{total_batches}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️ Batch {i//batch_size + 1} failed, retrying in 5s... ({attempt+1}/{max_retries})")
                    time.sleep(5)
                else:
                    print(f"   ❌ Batch {i//batch_size + 1} failed after {max_retries} attempts: {e}")
                    raise e

        # Small delay between batches to avoid rate limiting
        time.sleep(0.5)

    ext        = os.path.splitext(original_name)[1].lower().strip(".")
    file_type  = SUPPORTED_TYPES.get(ext, "📁 File")
    page_count = len(set(
        d.metadata.get("page", 0) for d in docs
    ))

    add_to_library(doc_id, original_name, file_type,
                   len(chunks), page_count, collection_name)

    print(f"   🎉 Done: {len(chunks)} chunks uploaded to Qdrant")
    return doc_id

# ── Helpers ──────────────────────────────────────────────────────────
def to_confidence(distance):
    return round(max(0, (1 - distance / 2) * 100), 1)

def confidence_bar(score):
    color = "#10b981" if score > 70 else "#f59e0b" if score > 40 else "#ef4444"
    label = "Strong" if score > 70 else "Moderate" if score > 40 else "Weak"
    st.markdown(f"""
    <div style="margin:8px 0">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:12px;color:#64748b;font-weight:500">Retrieval confidence</span>
        <span style="font-size:12px;font-weight:600;color:{color}">{score:.1f}% — {label}</span>
      </div>
      <div style="background:#e2e8f0;border-radius:6px;height:6px">
        <div style="background:{color};width:{score}%;height:100%;border-radius:6px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def generate_chat_pdf(messages, doc_name):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "LegalMind AI - Chat History", fill=True, ln=True, align="C")

    # Document name
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Document: {doc_name}", ln=True, align="C")
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
             ln=True, align="C")
    pdf.ln(6)

    # Messages
    for msg in messages:
        if msg["role"] == "user":
            pdf.set_fill_color(239, 246, 255)
            pdf.set_text_color(30, 58, 138)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, "YOU", fill=True, ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, msg["content"].encode('latin-1',
                          errors='replace').decode('latin-1'))
            pdf.ln(2)
        else:
            pdf.set_fill_color(240, 253, 244)
            pdf.set_text_color(22, 101, 52)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, "LEGALMIND AI", fill=True, ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, msg["content"].encode('latin-1',
                          errors='replace').decode('latin-1'))
            pdf.ln(4)

    return pdf.output()


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    library = load_library()

    # ── User info + logout ───────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.15);border-radius:8px;
                padding:8px 12px;margin-bottom:8px">
      <div style="font-size:12px;color:#94a3b8">Logged in as</div>
      <div style="font-size:13px;font-weight:600;color:#e2e8f0">
        {current_email}
      </div>
    </div>""", unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state.user    = None
        st.session_state.session = None
        st.session_state.active_doc_id = None
        st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
      <div style="font-size:2rem">⚖️</div>
      <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-top:4px">LegalMind AI</div>
      <div style="font-size:11px;color:#64748b;margin-top:2px">RAG · Groq · LangChain</div>
    </div>
    <hr style="border-color:#334155;margin:0.8rem 0">
    """, unsafe_allow_html=True)

    # ── Upload ──────────────────────────────────────────────────────
    st.markdown("### Upload Document")
    uploaded = st.file_uploader(
        "Drop any file",
        type=list(SUPPORTED_TYPES.keys()),
        label_visibility="collapsed"
    )

    if uploaded is not None:
        st.session_state.pending_file_name  = uploaded.name
        st.session_state.pending_file_bytes = uploaded.getbuffer().tobytes()

    if "pending_file_name" in st.session_state:
        st.info(f"📄 Ready to index: {st.session_state.pending_file_name}")

        if st.button("⚡ Index Document", use_container_width=True):
            file_name  = st.session_state.pending_file_name
            file_bytes = st.session_state.pending_file_bytes
            save_path  = os.path.join(UPLOAD_DIR, file_name)

            with open(save_path, "wb") as f:
                f.write(file_bytes)

            ext = os.path.splitext(file_name)[1].lower().strip(".")
            msg = "Reading image with vision AI..." \
                  if ext in ["png","jpg","jpeg","webp"] \
                  else f"Indexing {file_name}..."

            with st.spinner(msg):
                doc_id = ingest_file(save_path, file_name)

            os.remove(save_path)

            del st.session_state.pending_file_name
            del st.session_state.pending_file_bytes

            if doc_id:
                st.success("✅ Indexed successfully")
                st.session_state.active_doc_id = doc_id
                st.session_state.messages      = []
                st.session_state.mode          = "chat"
                library = load_library()
            else:
                st.error("Failed to process file.")

    # ── Document Library ────────────────────────────────────────────
    st.markdown("<hr style='border-color:#334155;margin:0.8rem 0'>", unsafe_allow_html=True)
    doc_count = len(library)
    st.markdown(f"### My Documents &nbsp; `{doc_count}`", unsafe_allow_html=True)

    if not library:
        st.markdown("""
        <div style="font-size:12px;color:#475569;text-align:center;padding:12px 0">
          No documents yet.<br>Upload one above.
        </div>""", unsafe_allow_html=True)
    else:
        active_id = st.session_state.get("active_doc_id")
        for doc_id, doc in library.items():
            is_active = doc_id == active_id
            icon      = doc["file_type"].split()[0]

            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                if st.button(
                    f"{icon} {doc['name'][:22]}{'…' if len(doc['name'])>22 else ''}",
                    key=f"sel_{doc_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.active_doc_id = doc_id
                    st.session_state.mode          = "chat"
                    st.rerun()
            with col2:
                if st.button("💬", key=f"clr_{doc_id}", help="Clear chat history"):
                    st.session_state[f"messages_{doc_id}"] = []
                    st.rerun()
            with col3:
                if st.button("🗑", key=f"del_{doc_id}", help="Delete document"):
                    if active_id == doc_id:
                        st.session_state.active_doc_id = None
                    st.session_state[f"messages_{doc_id}"] = []
                    delete_from_library(doc_id)
                    st.rerun()

    # ── Mode selector ────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#334155;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown("### Mode")
    mode = st.radio(
        "Select mode",
        ["💬 Chat", "❓ FAQ", "🔍 Compare"],
        label_visibility="collapsed",
        key="mode_radio"
    )
    st.session_state.mode = mode.split()[1].lower()

    # ── Settings ─────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#334155;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown("### Settings")
    n_results   = st.slider("Chunks to retrieve", 3, 12, 6)
    show_chunks = st.toggle("Show source chunks", value=False)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    
# ══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a8a 0%,#1d4ed8 50%,#2563eb 100%);
            padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.2rem;
            box-shadow:0 6px 24px rgba(29,78,216,0.25)">
  <h1 style="color:white;margin:0;font-size:1.7rem;font-weight:600">
    ⚖️ LegalMind AI
  </h1>
  <p style="color:rgba(255,255,255,0.7);margin:0.3rem 0 0;font-size:0.88rem">
    Strict RAG · Answers only from your documents · No hallucination
  </p>
</div>
""", unsafe_allow_html=True)

library    = load_library()
active_id  = st.session_state.get("active_doc_id")
active_doc = library.get(active_id) if active_id else None
mode       = st.session_state.get("mode", "chat")

# ── No doc selected ─────────────────────────────────────────────────
if not active_doc:
    st.markdown("""
    <div style="background:white;border:1.5px dashed #93c5fd;border-radius:14px;
                padding:3rem;text-align:center;margin-top:1rem">
      <div style="font-size:3rem;margin-bottom:1rem">📂</div>
      <div style="font-size:1.1rem;font-weight:600;color:#1d4ed8;margin-bottom:0.5rem">
        No document selected
      </div>
      <div style="font-size:0.88rem;color:#64748b">
        Upload a document or select one from the sidebar
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Active doc header ────────────────────────────────────────────────
st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
            padding:10px 16px;margin-bottom:1rem;display:flex;align-items:center;gap:10px">
  <span style="font-size:1.2rem">{active_doc['file_type'].split()[0]}</span>
  <div>
    <div style="font-weight:600;color:#0f172a;font-size:14px">{active_doc['name']}</div>
    <div style="font-size:11px;color:#64748b">
      {active_doc['chunk_count']} chunks · {active_doc.get('page_count',0)} pages
       · Uploaded {active_doc['uploaded_at']}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

vs = get_vectorstore(active_doc["collection_name"])
if not vs:
    st.error("Vector store not found. Please re-upload this document.")
    st.stop()



# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# MODE: CHAT
# ══════════════════════════════════════════════════════════════════════
if mode == "chat":
    chat_key = f"messages_{active_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    messages = st.session_state[chat_key]

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about this legal document...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state[chat_key].append({"role": "user", "content": question})

        # Retrieve chunks
        docs_scores = vs.similarity_search_with_score(question, k=16)

        # Filter junk chunks (table headers, repeated legends)
        filtered = [
            (d, s) for d, s in docs_scores
            if len(d.page_content.strip()) > 150
            and len(set(d.page_content.split())) > 15
        ]
        if not filtered:
            filtered = docs_scores
        filtered = filtered[:8]

        docs   = [d for d, _ in filtered]
        scores = [to_confidence(s) for _, s in filtered]
        avg    = round(sum(scores) / len(scores), 1) if scores else 0
        
        confidence_bar(avg)
        if avg < 25:
            st.warning("⚠️ Low confidence — this topic may not be in the document.")

        if show_chunks:
            with st.expander(f"📚 {len(docs)} source chunks · avg {avg:.1f}%"):
                for i, (doc, score) in enumerate(zip(docs, scores)):
                    color = "#10b981" if score>70 else "#f59e0b" if score>40 else "#ef4444"
                    page  = doc.metadata.get("page") or doc.metadata.get("page_label") or "—"
                    st.markdown(f"""
                    <div style="border-left:3px solid {color};padding:8px 12px;
                                margin:6px 0;background:#f8fafc;border-radius:4px">
                      <span style="font-size:11px;color:{color};font-weight:600">
                        Source {i+1} · Page {page} · {score:.0f}%
                      </span>
                      <p style="font-size:12px;color:#475569;margin:4px 0 0;
                                font-family:monospace">{doc.page_content[:300]}…</p>
                    </div>""", unsafe_allow_html=True)

        context = "\n\n".join(
            f"[Source {i+1}] Page {doc.metadata.get('page', '—')}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

        chain = ChatPromptTemplate.from_messages([
            ("system", """You are a legal document analysis assistant.

STRICT RULES:

IMPORTANT:
The document may contain repeated headers, footers, OCR noise, table legends,
abbreviations, watermarks, or text such as:
"HC High Court C Core P Periphery"
Ignore these completely unless directly relevant to the question.

YOUR RULES:

1. Answer ONLY using the document sources provided.
2. Do NOT use external knowledge, legal knowledge, assumptions, or inference.
3. Provide a detailed answer when information exists in the document.
4. Use bullet points for multiple findings.
5. Do NOT repeat the same information.
6. Cite every important statement using:
   [Source 1, Page 4]
7. If information is partially available, answer only the available portion and state what is missing.
8. If information is not found, reply exactly:
   "Not available in this document."
9. If multiple sources contain relevant information, combine them into a single answer with citations.
10. Ignore OCR noise, page numbers, repeated headers, repeated footers, scanning artifacts, and watermark text.
11. Format answers professionally using paragraphs and bullet points where appropriate.


SOURCES:
{context}"""),
            ("human", "{question}")
        ]) | llm | StrOutputParser()

        with st.chat_message("assistant"):
            reply = st.write_stream(
                chain.stream({"context": context, "question": question})
            )
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
              <span style="font-size:11px;background:#eff6ff;color:#1d4ed8;
                           padding:2px 10px;border-radius:20px">
                {'🟢' if avg>70 else '🟡' if avg>40 else '🔴'} {avg:.1f}% confidence
              </span>
              <span style="font-size:11px;background:#f0fdf4;color:#166534;
                           padding:2px 10px;border-radius:20px">{len(docs)} sources</span>
              <span style="font-size:11px;background:#faf5ff;color:#6b21a8;
                           padding:2px 10px;border-radius:20px">Strict RAG · No hallucination</span>
            </div>""", unsafe_allow_html=True)

        st.session_state[chat_key].append({"role": "assistant", "content": reply})

# ══════════════════════════════════════════════════════════════════════
# MODE: FAQ
# ══════════════════════════════════════════════════════════════════════
elif mode == "faq":
    st.markdown("### ❓ Auto-generated FAQs")
    st.markdown(f"*Based strictly on content in **{active_doc['name']}***")

    # Check if FAQs already generated
    saved_faqs = active_doc.get("faqs", [])

    if saved_faqs:
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"**{len(saved_faqs)} FAQs generated**")
        with col2:
            if st.button("🔄 Redo", help="Regenerate FAQs"):
                supabase.table("documents")\
                    .update({"faqs": faqs})\
                    .eq("id", active_id)\
                    .execute()
                st.rerun()

        for i, faq in enumerate(saved_faqs):
            with st.expander(f"Q{i+1}: {faq['question']}"):
                st.markdown(faq["answer"])
                st.markdown(f"""
                <span style="font-size:11px;color:#64748b">
                📌 {faq.get('source','Document')}
                </span>""", unsafe_allow_html=True)

    else:
        st.info("No FAQs generated yet for this document.")

        if st.button("✨ Generate FAQs from this document", use_container_width=True):
            with st.spinner("Reading document and generating FAQs..."):

                # Get top chunks as context
                all_chunks = vs.similarity_search(
                    "document summary key topics important information",
                    k=10
                )
                context = "\n\n".join(
                    f"[Chunk {i+1}]:\n{doc.page_content}"
                    for i, doc in enumerate(all_chunks)
                )

                faq_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a legal document analyst.
Read the document excerpts below and generate exactly 6 FAQs.

STRICT RULES:
1. Use ONLY information present in the document.
2. Do NOT use external knowledge.
3. Determine the document type automatically.
4. If it is a report, generate report-related FAQs.
5. If it is a policy, generate policy-related FAQs.
6. If it is a contract, generate contract-related FAQs.
7. Questions should cover the most important information found in the document.
8. Return ONLY valid JSON.

Required format:
[
  {{
    "question": "Question text",
    "answer": "Answer text",
    "source": "Page X"
  }}
]

Return ONLY the JSON array. No markdown. No explanation.

DOCUMENT:
{context}"""),
                    ("human", "Generate 6 FAQs from this legal document.")
                ]) | llm | StrOutputParser()

                raw = faq_prompt.invoke({"context": context})

                try:
                    # Clean response and parse JSON
                    raw_clean = raw.strip()
                    if raw_clean.startswith("```"):
                        raw_clean = raw_clean.split("```")[1]
                        if raw_clean.startswith("json"):
                            raw_clean = raw_clean[4:]
                    faqs = json.loads(raw_clean.strip())[:6]

                    # Save to library
                    lib = load_library()
                    lib[active_id]["faqs"] = faqs
                    save_library(lib)
                    st.rerun()

                except Exception as e:
                    st.error(f"Could not parse FAQs: {str(e)}")
                    st.code(raw)
# ══════════════════════════════════════════════════════════════════════
# MODE: COMPARE
# ══════════════════════════════════════════════════════════════════════
elif mode == "compare":
    st.markdown("### 🔍 Compare Documents")

    library = load_library()
    if len(library) < 2:
        st.warning("You need at least 2 documents to compare. Upload more documents.")
        st.stop()

    doc_names = {doc_id: doc["name"] for doc_id, doc in library.items()}

    # Multi-select — 2 or more documents
    selected_ids = st.multiselect(
        "Select documents to compare (2 or more)",
        options=list(doc_names.keys()),
        format_func=lambda x: doc_names[x],
        default=list(doc_names.keys())[:2],
        max_selections=5
    )

    if len(selected_ids) < 2:
        st.info("Please select at least 2 documents.")
        st.stop()

    # Auto FAQ summary for each selected doc
    st.markdown("---")
    st.markdown("#### 📋 Document Summaries")
    
    summary_cols = st.columns(len(selected_ids))
    vectorstores = {}

    for i, doc_id in enumerate(selected_ids):
        doc      = library[doc_id]
        vs_temp  = get_vectorstore(doc["collection_name"])
        vectorstores[doc_id] = vs_temp

        with summary_cols[i]:
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;
                        border-radius:8px;padding:10px;margin-bottom:8px">
              <div style="font-weight:600;font-size:13px;color:#1e293b;
                          margin-bottom:4px">{doc['name'][:30]}</div>
              <div style="font-size:11px;color:#64748b">
                {doc['chunk_count']} chunks · {doc.get('page_count',0)} pages
              </div>
            </div>""", unsafe_allow_html=True)

            # Show existing FAQs or generate button
            saved_faqs = doc.get("faqs", [])
            if saved_faqs:
                st.markdown(f"**Top questions:**")
                for faq in saved_faqs[:2]:
                    st.markdown(f"""
                    <div style="font-size:11px;color:#475569;background:white;
                                border-left:3px solid #3b82f6;padding:6px 8px;
                                margin:3px 0;border-radius:3px">
                      {faq['question']}
                    </div>""", unsafe_allow_html=True)
            else:
                if st.button(f"Generate FAQs", key=f"faq_{doc_id}"):
                    with st.spinner(f"Generating FAQs for {doc['name'][:20]}..."):
                        chunks = vs_temp.similarity_search(
                            "main purpose parties obligations terms conditions", k=8
                        )
                        context = "\n\n".join(
                            f"[Chunk {i+1}]:\n{d.page_content}"
                            for i, d in enumerate(chunks)
                        )
                        faq_chain = ChatPromptTemplate.from_messages([
                            ("system", """Generate 4 FAQs from this document.
Return ONLY a JSON array:
[{{"question":"...","answer":"...","source":"..."}}]
Use ONLY document content. No external knowledge."""),
                            ("human", f"Document:\n{context}")
                        ]) | llm | StrOutputParser()

                        raw = faq_chain.invoke({})
                        try:
                            raw_clean = raw.strip()
                            if "```" in raw_clean:
                                raw_clean = raw_clean.split("```")[1]
                                if raw_clean.startswith("json"):
                                    raw_clean = raw_clean[4:]
                            faqs = json.loads(raw_clean.strip())[:4]
                            lib  = load_library()
                            lib[doc_id]["faqs"] = faqs
                            save_library(lib)
                            st.rerun()
                        except Exception:
                            st.error("Could not generate FAQs.")

    # Show previous comparison if exists
    compare_key = f"compare_result_{active_id}"
    if compare_key in st.session_state:
        prev = st.session_state[compare_key]
        with st.expander("📊 Previous comparison result", expanded=False):
            st.markdown(f"**Question:** {prev['question']}")
            st.markdown(prev["result"])
            if st.button("🗑 Clear saved comparison"):
                del st.session_state[compare_key]
                st.rerun()

    # Comparison question
    st.markdown("---")
    st.markdown("#### 🤖 Ask a Comparison Question")

    if "compare_q" not in st.session_state:
        st.session_state.compare_q = ""

    question = st.text_input(
        "What do you want to compare? (optional)",
        value=st.session_state.compare_q,
        placeholder="e.g. What are the key differences?"
    )

    # Update session when user types
    st.session_state.compare_q = question
    
    # If no question typed use default
    if st.button("🔍 Compare All", use_container_width=True):
        compare_question = st.session_state.compare_q.strip() \
        if st.session_state.compare_q.strip() \
        else "complete comparison of both documents including purpose, key topics, similarities, differences, findings and conclusion"
        
        # Retrieve from ALL selected documents
        all_contexts = []
        for doc_id in selected_ids:
            doc    = library[doc_id]
            vs_tmp = vectorstores[doc_id]
            if not vs_tmp:
                continue
            chunks = vs_tmp.similarity_search(compare_question, k=5)
            context_text = "\n".join(
                f"[{doc['name'][:20]}-{i+1}] "
                f"Page {c.metadata.get('page','—')}: {c.page_content}"
                for i, c in enumerate(chunks)
            )
            all_contexts.append(
                f"DOCUMENT: {doc['name']}\n{context_text}"
            )
        combined_context = "\n\n---\n\n".join(all_contexts)

        doc_list = "\n".join(
            f"- {library[d]['name']}" for d in selected_ids
        )

        is_specific_question = st.session_state.compare_q.strip() != ""

        if is_specific_question:
            system_prompt = f"""You are a legal document analysis expert.
You have excerpts from {len(selected_ids)} documents:
{doc_list}

STRICT RULES:
1. Answer ONLY using the excerpts provided.
2. Do NOT use external knowledge.
3. Do not infer tone, intent, audience, or purpose unless explicitly stated.
4. Answer the user's specific question for EACH document separately.
5. Label each document clearly — Document A, Document B etc.
6. Use bullet points whenever possible.
7. Be concise and direct.
8. If info not found say "Not found in [document name]"

DOCUMENTS:
{combined_context}"""

        else:
            system_prompt = f"""You are a legal document comparison expert.
You have excerpts from {len(selected_ids)} documents:
{doc_list}

STRICT RULES:
1. Do not infer tone, intent, audience, or purpose unless explicitly stated in the excerpts.
2. Do NOT use external knowledge.
3. Create these sections:

## Executive Summary
## Document A Overview
## Document B Overview
## Similarities
## Differences
## Key Findings
## Conclusion

4. Clearly identify which document each finding comes from.
5. If information exists in one document but not the other, explicitly state it.
6. Use bullet points whenever possible.
7. Keep the comparison concise and professional.
8. If information is missing, explicitly state it.

If info not found in a document say "Not found in [document name]"

DOCUMENTS:
{combined_context}"""

        compare_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ]) | llm | StrOutputParser()

        # Show source chunks per document
        with st.expander(f"📚 Retrieved chunks per document — {len(selected_ids)} docs"):
            chunk_cols = st.columns(len(selected_ids))
            for i, doc_id in enumerate(selected_ids):
                doc    = library[doc_id]
                vs_tmp = vectorstores[doc_id]
                chunks = vs_tmp.similarity_search(compare_question, k=5)
                with chunk_cols[i]:
                    st.markdown(f"**{doc['name'][:25]}**")
                    for chunk in chunks:
                        st.markdown(f"""
                        <div style="background:#f8fafc;border-left:3px solid #3b82f6;
                                    padding:6px 8px;margin:4px 0;border-radius:3px;
                                    font-size:11px;color:#334155">
                          {chunk.page_content[:200]}…
                        </div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Comparison Result")
        with st.spinner(f"Comparing {len(selected_ids)} documents..."):
            result = compare_prompt.invoke({"question": compare_question})
            st.markdown(result)

            # Save result to session
            compare_key = "last_compare_result"
            st.session_state[compare_key] = {
                "question": compare_question,
                "result": result,
                "docs": [library[d]["name"] for d in selected_ids]
            }