from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# STEP 1A: Load the PDF (user provides path)
# ============================================================
pdf_path = input("Enter the path to your PDF file: ")

loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")

# ============================================================
# STEP 1B: Load YouTube video (optional) - fetch existing captions
# ============================================================
youtube_url = input("Enter a YouTube link (or press Enter to skip): ")

youtube_docs = []
if youtube_url.strip() != "":
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        youtube_docs = loader.load()
        print(f"YouTube transcript loaded: {len(youtube_docs)} document(s)")
    except Exception as e:
        print(f"Could not load YouTube captions: {e}")
        print("Continuing with PDF only.")

# Combine PDF pages + YouTube transcript into one list
all_docs = pages + youtube_docs
print(f"Total combined documents: {len(all_docs)}")

# ============================================================
# STEP 2: Chunking - split all documents into smaller pieces
# ============================================================
chunk_size = 1024
chunk_overlap = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunked_docs = splitter.split_documents(all_docs)
print(f"Total chunks created: {len(chunked_docs)}")

# ============================================================
# STEP 3: Embeddings - load local embedding model
# ============================================================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ============================================================
# STEP 4: Vector Store - embed all chunks and store in Chroma
# ============================================================
vector_store = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
print("Vector store created and saved successfully!")

# ============================================================
# STEP 5: Connect to local Ollama LLM
# ============================================================
llm = OllamaLLM(model="llama3")

# ============================================================
# STEP 6: Prompt template - tells the LLM how to answer
# ============================================================
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the provided document to answer this."

Context:
{context}

Question:
{question}

Answer:
""")

# ============================================================
# STEP 7: Loop - ask questions repeatedly until user exits
# ============================================================
while True:
    query = input("\nAsk your question (or type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve top matching chunks for this question
    results = vector_store.similarity_search(query, k=3)

    # Combine retrieved chunks into one context block
    retrieved_context = "\n\n".join([doc.page_content for doc in results])

    # Fill the prompt template with context + question
    final_prompt = prompt_template.format(context=retrieved_context, question=query)

    # Send to Ollama and get the answer
    answer = llm.invoke(final_prompt)

    print("=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(answer)