# 🔒 Private RAG Assistant

A fully local, privacy-first Retrieval-Augmented Generation (RAG) assistant for querying PDF documents and YouTube transcripts — without sending any data to external cloud APIs.

## 🎯 What It Does

Upload one or more PDFs (research papers, notes, study material) and/or a YouTube video link, then ask natural-language questions about that content. The assistant answers strictly from the provided documents, with source citations for every answer — all running entirely on your own machine.

## 🤔 Why This Project

Researchers, students, and professionals often can't paste confidential documents into tools like ChatGPT or Claude, since that data leaves their control and goes to an external server. This project solves that — the data never leaves your device, at any point in the pipeline.

## 🏗️ Architecture

User uploads PDF/YouTube → Chunking → Embeddings → Vector DB (Chroma) → Question → Retrieval (top-k matching chunks) → LLM (Ollama/Llama 3.2) → Answer

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama running Llama 3.2 (1B) — fully local |
| Embeddings | HuggingFace all-MiniLM-L6-v2 — local, no API |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| PDF Loading | PyPDFLoader (supports multiple files) |
| YouTube Loading | YoutubeLoader (native captions) |
| UI | Streamlit |

## 🚀 How to Run Locally

1. Clone this repo: git clone https://github.com/SharanMahadevan36/private-rag-assistant.git then cd private-rag-assistant

2. Create and activate a virtual environment: python -m venv venv then venv\Scripts\activate

3. Install dependencies: pip install -r requirements.txt

4. Install Ollama from ollama.com and pull the model: ollama pull llama3.2:1b

5. Run the app: streamlit run app.py

## 💡 Key Engineering Decisions

- Switched from Whisper-based audio transcription to YouTube's native captions — Whisper transcription was too slow on CPU-only hardware; captions load in seconds and cover most educational content.
- Chose Llama 3.2 (1B) over larger models — significantly faster inference on CPU with minimal RAM usage, while still producing accurate answers when grounded in retrieved context.
- Source citations for every answer — each response shows exactly which document chunk it came from, so answers are verifiable rather than a black box.
- Multi-PDF support — users can upload and query across several documents at once, all indexed into a single vector store.

## 📸 Demo

private-rag-assistant-demo.gif

## 📄 License

MIT
