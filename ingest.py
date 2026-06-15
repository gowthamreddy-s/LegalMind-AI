import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document          # ← fixed import
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader,
    TextLoader, BSHTMLLoader
)
import pandas as pd
import shutil, os, sys

load_dotenv()

# ── Load any file type ─────────────────────────────────────────────
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"   File type: {ext}")

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()

    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path).load()

    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8").load()

    elif ext in [".html", ".htm"]:
        return BSHTMLLoader(file_path).load()

    elif ext == ".csv":
        return CSVLoader(file_path).load()

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        return [Document(
            page_content=text,
            metadata={"source": file_path, "type": "excel"}
        )]

    else:
        print(f"   Unknown type — trying as plain text")
        return TextLoader(file_path, encoding="utf-8").load()


# ── Main ingest ────────────────────────────────────────────────────
def ingest(file_path="document.pdf"):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading: {file_path}")
    docs = load_document(file_path)
    print(f"{len(docs)} pages/sections loaded")

    print("Splitting into chunks...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    ).split_documents(docs)
    print(f"{len(chunks)} chunks created")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    print("Storing in ChromaDB...")
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print(f"\nDone. {len(chunks)} chunks stored.")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"
    ingest(path)