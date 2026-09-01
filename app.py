import os
import tempfile
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Private RAG Assistant",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #060709 !important;
        color: #f3f4f6 !important;
    }

    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1180px !important;
    }

    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.4rem;
    }
    .brand-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .nav-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        padding: 5px 12px;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    .eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        color: #f59e0b;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .hero-title .accent {
        color: #f59e0b;
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 60%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        color: #9ca3af;
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }

    .showcase-card {
        background: radial-gradient(circle at 80% 20%, rgba(245, 158, 11, 0.05) 0%, rgba(13, 15, 20, 0.95) 70%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 20px;
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.6);
        position: relative;
    }
    .card-badge {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .card-badge span {
        color: #f59e0b;
    }
    .card-subtext {
        color: #9ca3af;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 2px;
        margin-bottom: 14px;
    }
    .card-metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 12px;
    }
    .metric-pill {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 12px;
        text-align: left;
    }
    .metric-num {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    .metric-lbl {
        font-size: 0.68rem;
        color: #6b7280;
        text-transform: uppercase;
        margin-top: 2px;
    }

    .bento-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 1.8rem;
    }
    .bento-item {
        background: #0d0f14;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 18px;
        transition: all 0.2s ease;
    }
    .bento-item:hover {
        border-color: rgba(245, 158, 11, 0.3);
    }
    .bento-icon {
        font-size: 1.3rem;
        margin-bottom: 8px;
    }
    .bento-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .bento-text {
        font-size: 0.8rem;
        color: #808898;
        line-height: 1.4;
    }

    .source-panel {
        background: rgba(13, 15, 20, 0.8);
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.82rem;
        color: #9ca3af;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 3px solid #f59e0b;
    }
    .source-label {
        font-weight: 700;
        color: #f59e0b;
        margin-right: 6px;
    }

    .chat-welcome {
        text-align: center;
        color: #6b7280;
        padding: 40px 20px;
        font-size: 0.95rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #090b0e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Cache the sentence-transformer embeddings in memory."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm(model_name="llama3.2:1b"):
    """Cache local Ollama LLM connection."""
    return OllamaLLM(model=model_name, temperature=0.3)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "stats" not in st.session_state:
    st.session_state.stats = {}

# ============================================================
# TOP BRAND BAR
# ============================================================
st.markdown("""
<div class="top-nav">
    <div class="brand-logo">
        PRIVATE <span style="color:#f59e0b;">RAG</span> <span style="color:#ffffff;">ASSISTANT</span>
    </div>
    <div class="nav-status">
        <span class="status-dot"></span> 100% Local Engine
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📂 Knowledge Source")
    st.caption("Upload one or more PDFs, or a YouTube transcript, to embed securely on-device.")

    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Processed entirely in local RAM — you can select multiple PDFs at once"
    )
    youtube_url = st.text_input("YouTube Video URL (Captions)", placeholder="https://youtube.com/watch?v=...")

    process_btn = st.button("🚀 Build Vector Index", use_container_width=True, type="primary")

    if st.session_state.processed and st.session_state.stats:
        stats = st.session_state.stats
        st.markdown(f"""
        <div style="background: #0d0f14; border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 10px; padding: 12px; margin-top: 14px;">
            <div style="font-size: 0.7rem; color: #f59e0b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">Active Vector Index</div>
            <div style="font-size: 0.85rem; color: #e5e7eb; line-height: 1.5;">
                📄 <b>{stats.get('pdf_files', 0)}</b> PDF file(s), <b>{stats.get('pdf_pages', 0)}</b> pages<br>
                🎥 <b>{'Active' if stats.get('yt_loaded') else 'None'}</b> Video Transcript<br>
                🧩 <b>{stats.get('chunks', 0)}</b> Vector Chunks
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🛠️ Quick Actions")

        if st.button("⚡ Generate Summary", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Give me a full, detailed summary of this document, covering all the major points in at least 6-8 sentences."
            })
            st.rerun()

        if st.session_state.messages:
            chat_export = "\n\n".join([f"**{m['role'].upper()}**:\n{m['content']}" for m in st.session_state.messages])
            st.download_button(
                label="📥 Export Chat (.txt)",
                data=chat_export,
                file_name="private_rag_chat.txt",
                mime="text/plain",
                use_container_width=True
            )

        if st.button("🗑️ Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with st.expander("ℹ️ About This Project (Architecture)"):
        st.markdown("""
        **🎯 What:**
        A fully local, privacy-first Retrieval-Augmented Generation (RAG) assistant for querying PDF documents and YouTube transcripts — without sending data to any external cloud API.

        **🤔 Why:**
        Built for researchers, students, and professionals who want to ask questions about their own documents — research papers, personal notes, study material — without uploading that content to a third-party AI service.

        **⚙️ Technical Pipeline:**
        1. **Ingestion:** `PyPDFLoader` & `YoutubeLoader` extract text/captions (supports multiple PDFs at once).
        2. **Chunking:** `RecursiveCharacterTextSplitter` (900 chars, 150 overlap).
        3. **Embeddings:** `all-MiniLM-L6-v2` runs locally via HuggingFace.
        4. **Vector Store:** ChromaDB in-memory index for fast similarity lookup.
        5. **Inference:** Local `Ollama` (`Llama 3.2 1B`) prompted strictly on retrieved chunks.

        **💡 Key Engineering Trade-offs:**
        - Switched from Whisper-based audio transcription to YouTube's native captions for faster loading.
        - Selected `Llama 3.2 1B` over larger models for responsive local CPU inference with minimal RAM usage.
        """)

# ============================================================
# PROCESSING - handles multiple PDFs + optional YouTube link
# ============================================================
if process_btn:
    if not uploaded_files and not youtube_url.strip():
        st.sidebar.error("⚠️ Please provide at least one PDF or a YouTube URL.")
    else:
        progress_bar = st.progress(0, text="Starting...")
        all_docs = []
        pdf_pages_count = 0
        pdf_files_count = 0
        yt_loaded = False

        if uploaded_files:
            pdf_files_count = len(uploaded_files)
            for idx, pdf_file in enumerate(uploaded_files):
                pct = 10 + int((idx / len(uploaded_files)) * 40)
                progress_bar.progress(pct, text=f"📄 Parsing PDF {idx+1}/{len(uploaded_files)}: {pdf_file.name}...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_pdf_path = tmp_file.name

                try:
                    loader = PyPDFLoader(tmp_pdf_path)
                    pdf_docs = loader.load()
                    pdf_pages_count += len(pdf_docs)
                    all_docs.extend(pdf_docs)
                finally:
                    if os.path.exists(tmp_pdf_path):
                        os.remove(tmp_pdf_path)

        progress_bar.progress(55, text="🎥 Checking YouTube link...")
        if youtube_url.strip():
            try:
                yt_loader = YoutubeLoader.from_youtube_url(youtube_url.strip(), add_video_info=True)
                yt_docs = yt_loader.load()
                if yt_docs:
                    yt_loaded = True
                    all_docs.extend(yt_docs)
            except Exception as e:
                st.sidebar.warning(f"Could not load YouTube transcript: {e}")

        if not all_docs:
            progress_bar.empty()
            st.sidebar.error("❌ No valid text found.")
        else:
            progress_bar.progress(65, text="✂️ Splitting content into chunks (900 chars, 150 overlap)...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
            chunked_docs = splitter.split_documents(all_docs)

            progress_bar.progress(80, text="🧠 Generating embeddings with MiniLM-L6-v2...")
            embeddings = get_embedding_model()

            progress_bar.progress(92, text="💾 Building local ChromaDB vector index...")
            vector_store = Chroma.from_documents(
                documents=chunked_docs,
                embedding=embeddings
            )

            progress_bar.progress(100, text="✅ Done!")

            st.session_state.vector_store = vector_store
            st.session_state.processed = True
            st.session_state.messages = []
            st.session_state.stats = {
                "pdf_files": pdf_files_count,
                "pdf_pages": pdf_pages_count,
                "yt_loaded": yt_loaded,
                "chunks": len(chunked_docs)
            }

            progress_bar.empty()
            st.sidebar.success("✅ Ready to answer your questions!")
            st.rerun()

# ============================================================
# LANDING PAGE (before any document is processed)
# ============================================================
if not st.session_state.processed:
    col_left, col_right = st.columns([1.25, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="eyebrow">Local RAG Engine // Offline Inference</div>
        <h1 class="hero-title">Private Answers <br><span class="accent">from Your Own Documents.</span></h1>
        <p class="hero-desc">
            Query your research papers, notes, and YouTube transcripts entirely on your local machine. Zero external API calls, zero data leaving your device.
        </p>
        """, unsafe_allow_html=True)

        st.info("👈 **Upload one or more PDFs (or enter a YouTube URL) in the sidebar to build your vector database.**")

    with col_right:
        st.markdown("""
        <div class="showcase-card">
            <div class="card-badge">Local <span>Stack</span></div>
            <div class="card-subtext">Open-Source Engine Architecture</div>
            <div style="height: 1px; background: rgba(255,255,255,0.06); margin: 10px 0;"></div>
            <div class="card-metric-row">
                <div class="metric-pill">
                    <div class="metric-num">Llama 3.2 (1B)</div>
                    <div class="metric-lbl">Local CPU LLM</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-num">ChromaDB</div>
                    <div class="metric-lbl">Vector Database</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-num">MiniLM-L6-v2</div>
                    <div class="metric-lbl">HF Embeddings</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-num">LangChain</div>
                    <div class="metric-lbl">RAG Orchestration</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bento-grid">
        <div class="bento-item">
            <div class="bento-icon">🔒</div>
            <div class="bento-title">Local Privacy</div>
            <div class="bento-text">Files stay in local memory. Built for research papers, personal notes, and study material.</div>
        </div>
        <div class="bento-item">
            <div class="bento-icon">⚡</div>
            <div class="bento-title">Zero API Costs</div>
            <div class="bento-text">Powered entirely by local weights with no token metering or subscription costs.</div>
        </div>
        <div class="bento-item">
            <div class="bento-icon">📑</div>
            <div class="bento-title">Precise Attribution</div>
            <div class="bento-text">Every generated response attributes exact chunk previews and document origin pages.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CHAT INTERFACE (after processing)
# ============================================================
else:
    chat_box = st.container()

    with chat_box:
        if not st.session_state.messages:
            st.markdown("""
            <div class="chat-welcome">
                💬 Your document(s) are indexed and ready.<br>
                Ask a specific question, or try "Give me a summary" for a full overview.
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"📑 View Retrieved Context ({len(msg['sources'])} citations)"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"""
                            <div class="source-panel">
                                <span class="source-label">Source {i} [{src['origin']}]:</span>
                                <span>{src['preview']}</span>
                            </div>
                            """, unsafe_allow_html=True)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and len(st.session_state.messages) % 2 != 0:
        latest_query = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document context..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                results = retriever.invoke(latest_query)
                retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in results])

                prompt_template = ChatPromptTemplate.from_template("""
                You are a knowledgeable assistant answering questions about the document(s) provided below.

                Guidelines:
                - Answer using the context provided. If the user asks for a summary, overview, or general explanation, synthesize the key points across the ENTIRE context — don't just quote one line or one chunk.
                - Match your answer length to what is asked: a simple factual question gets 1-2 sentences; a request for "explain", "summary", "details", or "overview" gets a full paragraph or bullet points; a request for a specific number of points or lines (e.g. "give 5 points", "in 3 lines", "20 lines") gets exactly that.
                - If the user specifies an exact number of lines or sentences (e.g. "in 2 lines", "3 sentences"), your entire answer must contain EXACTLY that many lines or sentences — no more, no less.
                - If no length is specified for a general/summary question, default to a thorough paragraph (5-8 sentences) or a bulleted list covering the main points.
                - Only say "I cannot find this information in the uploaded documents" if the context truly has nothing relevant to the question.
                Context:
                {context}

                Question:
                {question}

                Answer:
                """)

                formatted_prompt = prompt_template.format(
                    context=retrieved_context if retrieved_context else "No context found.",
                    question=latest_query
                )

                llm = get_llm()
                raw_response = llm.invoke(formatted_prompt)

            msg_placeholder = st.empty()
            accumulated = ""
            for token in raw_response.split(" "):
                accumulated += token + " "
                msg_placeholder.markdown(accumulated + "▌")
                time.sleep(0.015)
            msg_placeholder.markdown(accumulated.strip())

            sources_info = []
            for doc in results:
                origin = "PDF" if "page" in doc.metadata else "YouTube"
                preview = doc.page_content[:180].replace("\n", " ") + "..."
                sources_info.append({"origin": origin, "preview": preview})

            if sources_info:
                with st.expander(f"📑 View Retrieved Context ({len(sources_info)} citations)"):
                    for i, src in enumerate(sources_info, 1):
                        st.markdown(f"""
                        <div class="source-panel">
                            <span class="source-label">Source {i} [{src['origin']}]:</span>
                            <span>{src['preview']}</span>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": accumulated.strip(),
            "sources": sources_info
        })

    user_query = st.chat_input("Ask a question about your indexed documents...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.rerun()