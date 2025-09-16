# app.py
import os
import streamlit as st
from typing import List
import tempfile
import pdfplumber
from PyPDF2 import PdfReader

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_docs"
EMBED_MODEL_NAME = "mxbai-embed-large"  # Ollama embedding model name
LLM_MODEL_NAME = "llama3.2"             # Ollama LLM model name

def extract_text(pdf_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name
        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        os.unlink(tmp_path)
        return "\n\n".join(text_parts)
    except Exception:
        reader = PdfReader(pdf_bytes)
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])

def split_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

@st.cache_resource
def get_embeddings_llm():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    return embeddings, llm

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📚 PDF Chatbot (Ollama + Chroma)")

embeddings, llm = get_embeddings_llm()

uploaded_files = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file.read())
        if text.strip():
            all_docs.extend(split_text(text))

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    retriever = vectordb.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    query = st.text_input("Ask a question about your documents")
    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)
        st.subheader("Answer")
        st.write(answer)