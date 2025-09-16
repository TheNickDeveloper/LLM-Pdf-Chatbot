# 📚 PDF Chatbot (Ollama + Chroma)

This project is a **chatbot web application** that allows users to upload one or more PDF documents and ask natural language questions about their content. It leverages **large language models (LLMs)** and **vector embeddings** to provide accurate, context-aware answers from the uploaded files.

---

## 🪴 Application UIUX

![image](https://github.com/TheNickDeveloper/Pdf-Chatbot/blob/main/application_uiux.png)

---


## 🚀 How It Works

1. **Upload PDFs**  
   Users can upload one or more PDF files through the Streamlit interface.

2. **Text Extraction**  
   The application extracts text from each PDF using `pdfplumber` (with `PyPDF2` as a fallback).

3. **Text Splitting**  
   Extracted text is split into smaller chunks using `RecursiveCharacterTextSplitter`, making it more manageable for embeddings and retrieval.

4. **Vector Embeddings & Storage**  
   Each text chunk is converted into embeddings using **Ollama’s embedding model** (`mxbai-embed-large`).  
   The embeddings are stored in a **Chroma vector database** for efficient similarity search.

5. **Question Answering**  
   - User inputs a query in natural language.  
   - Relevant document chunks are retrieved from Chroma.  
   - The **Ollama LLM** (`llama3.2`) processes the retrieved chunks and generates an answer.  
   - The answer is displayed in the Streamlit app.

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** – for building the interactive web UI.
- **[LangChain](https://www.langchain.com/)** – to handle text splitting, retrieval, and chaining the LLM with the database.
- **[Ollama](https://ollama.ai/)** – for running the LLM (`llama3.2`) and embedding model (`mxbai-embed-large`) locally.
- **[Chroma](https://www.trychroma.com/)** – as the vector database for storing embeddings and enabling similarity search.
- **[pdfplumber](https://github.com/jsvine/pdfplumber)** & **[PyPDF2](https://pypi.org/project/pypdf2/)** – for extracting text from PDFs.

---

## ▶️ Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Typical dependencies include:

- **streamlit
- **langchain
- **langchain-ollama
- **chromadb
- **pdfplumber
- **PyPDF2

### 2. Run Ollama

Make sure you have Ollama installed and the required models (llama3.2 and mxbai-embed-large) available.

### 3. Start the App
```bash
streamlit run app.py
```

### 4. Usage

- Upload one or more PDFs.

- Type your question into the input box.

- Get AI-powered answers directly from your documents.

---

## 📌 Example Use Cases

- Academic research paper Q&A

- Legal document analysis

- Business report summarization

- Personal knowledge management
