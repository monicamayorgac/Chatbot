# -*- coding: utf-8 -*-
"""
Biomedical Advisor Chatbot
A RAG-based chatbot for biomedical engineering documents using Streamlit and Google Gemini.

Author: Monica Mayorga
Project: Seminar Biomedical Engineering
"""

import streamlit as st
import os
import tempfile
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================
# Option 1: Put your API key directly here (simple but less secure)
# Option 2: Leave empty and use environment variable GOOGLE_API_KEY
# Option 3: Leave empty and use .streamlit/secrets.toml

API_KEY = ""  # <-- PUT YOUR KEY HERE IF YOU WANT (e.g., "AIzaSy...")

# Other settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"
LLM_TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 5
# =============================================================================

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Biomedical Advisor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    .stats-card {
        background-color: #e8f4ea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Biomedical Advisor")
st.caption("Your AI assistant for biomedical engineering documents")


# --- API KEY MANAGEMENT ---
def get_api_key():
    """
    Get API key from multiple sources in order of priority:
    1. Hardcoded in config (API_KEY variable above)
    2. Streamlit secrets (for deployment)
    3. Environment variable
    4. User input (fallback)
    """
    # Priority 1: Hardcoded key
    if API_KEY:
        return API_KEY
    
    # Priority 2: Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
    except Exception:
        pass
    
    # Priority 3: Environment variable
    env_key = os.environ.get('GOOGLE_API_KEY')
    if env_key:
        return env_key
    
    # Priority 4: Session state (user input)
    return st.session_state.get('user_api_key', '')


def setup_api_key(key):
    """Set up the API key in the environment."""
    if key:
        os.environ['GOOGLE_API_KEY'] = key
        st.session_state['user_api_key'] = key
        return True
    return False


# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'messages': [],
        'vector_db': None,
        'processed_files': set(),
        'file_hashes': {},
        'embeddings_model': None,
        'processing_complete': False,
        'total_chunks': 0,
        'total_pages': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --- HELPER FUNCTIONS ---
def compute_file_hash(file_content):
    """Compute MD5 hash of file content for caching."""
    return hashlib.md5(file_content).hexdigest()


def get_cached_embeddings():
    """Get or create cached embeddings model."""
    if st.session_state.embeddings_model is None:
        with st.spinner("Loading embedding model (one-time setup)..."):
            st.session_state.embeddings_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    return st.session_state.embeddings_model


def load_single_pdf(file_data):
    """Load a single PDF file. Used for parallel processing."""
    file_content, file_name = file_data
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Set proper metadata
        for doc in docs:
            doc.metadata["source"] = file_name
            doc.metadata["file_hash"] = compute_file_hash(file_content)
        
        return docs, file_name, len(docs)
    finally:
        os.remove(tmp_path)


def process_documents_parallel(files, progress_bar, status_text):
    """Process multiple PDFs in parallel for better performance."""
    all_docs = []
    total_pages = 0
    
    # Prepare file data for parallel processing
    file_data_list = []
    for file in files:
        file_content = file.getvalue()
        file_hash = compute_file_hash(file_content)
        
        # Skip already processed files (by hash)
        if file_hash in st.session_state.file_hashes:
            status_text.text(f"Skipping {file.name} (already processed)")
            continue
            
        file_data_list.append((file_content, file.name))
        st.session_state.file_hashes[file_hash] = file.name
    
    if not file_data_list:
        status_text.text("All files already processed!")
        return [], 0
    
    # Process PDFs in parallel
    status_text.text(f"Loading {len(file_data_list)} PDF(s) in parallel...")
    
    with ThreadPoolExecutor(max_workers=min(4, len(file_data_list))) as executor:
        futures = {executor.submit(load_single_pdf, fd): fd[1] for fd in file_data_list}
        
        completed = 0
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                docs, name, pages = future.result()
                all_docs.extend(docs)
                total_pages += pages
                completed += 1
                progress_bar.progress(completed / len(file_data_list) * 0.5)
                status_text.text(f"Loaded {name} ({pages} pages)")
            except Exception as e:
                st.warning(f"Error loading {file_name}: {str(e)}")
    
    return all_docs, total_pages


def setup_knowledge_base(files):
    """Set up the knowledge base with parallel processing and caching."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Load PDFs in parallel
    all_docs, total_pages = process_documents_parallel(files, progress_bar, status_text)
    
    if not all_docs:
        progress_bar.empty()
        return st.session_state.vector_db
    
    # Step 2: Split documents
    status_text.text("Splitting documents into chunks...")
    progress_bar.progress(0.6)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    splits = splitter.split_documents(all_docs)
    
    # Step 3: Create/update vector store
    status_text.text(f"Indexing {len(splits)} chunks...")
    progress_bar.progress(0.8)
    
    embeddings = get_cached_embeddings()
    
    # If we have an existing DB, add to it; otherwise create new
    if st.session_state.vector_db is not None:
        st.session_state.vector_db.add_documents(splits)
        vectorstore = st.session_state.vector_db
    else:
        vectorstore = Chroma.from_documents(splits, embeddings)
    
    # Update stats
    st.session_state.total_chunks += len(splits)
    st.session_state.total_pages += total_pages
    
    progress_bar.progress(1.0)
    status_text.text(f"Successfully indexed {len(splits)} chunks from {total_pages} pages!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return vectorstore


# --- RAG LOGIC ---
def manual_rag_response(user_question, db, history):
    """Generate response using RAG (Retrieve, Augment, Generate)."""
    
    # A. Retrieve relevant documents
    docs = db.similarity_search(user_question, k=4)
    
    # B. Build context
    context_parts = []
    sources_info = []
    
    for i, doc in enumerate(docs):
        source_name = doc.metadata.get('source', 'Unknown')
        page_num = doc.metadata.get('page', 0) + 1
        content = doc.page_content.strip()
        
        context_parts.append(f"[Document {i+1} - {source_name}, Page {page_num}]:\n{content}")
        sources_info.append({
            'source': source_name,
            'page': page_num,
            'preview': content[:150] + "..." if len(content) > 150 else content
        })
    
    context_text = "\n\n".join(context_parts)
    
    # C. Create prompt with system instructions
    # =========================================================================
    # SYSTEM PROMPT EXPLANATION:
    # This tells the AI HOW to behave. It's like giving instructions to an employee.
    # Without this, the AI would just be a generic assistant.
    # With this, it knows it should:
    #   - Act as a biomedical expert
    #   - Use the document context provided
    #   - Be precise and professional
    #   - Admit when it doesn't know something
    # =========================================================================
    system_prompt = """You are a knowledgeable Biomedical Engineering advisor. Your role is to:
1. Answer questions accurately based on the provided document context
2. Explain complex biomedical concepts clearly
3. Reference specific sections when relevant
4. Acknowledge when information is not available in the documents

Guidelines:
- Be precise and professional
- Use technical terminology appropriately
- Provide examples when helpful
- If the context doesn't contain relevant information, say so honestly
"""

    final_prompt = f"""{system_prompt}

CONTEXT FROM DOCUMENTS:
{context_text}

RECENT CONVERSATION:
{history}

USER QUESTION: {user_question}

Please provide a helpful, accurate response based on the document context above."""

    # D. Generate response with retry logic
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_retries=MAX_RETRIES
    )
    
    raw_response = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(final_prompt)
            
            # Handle different response formats
            if isinstance(response.content, list):
                for part in response.content:
                    if isinstance(part, dict) and 'text' in part:
                        raw_response += part['text']
                    elif isinstance(part, str):
                        raw_response += part
            else:
                raw_response = str(response.content)
            break
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return "Rate limit reached. Please wait a moment and try again.", []
            else:
                return f"Error generating response: {error_str}", []
    
    return raw_response, sources_info


def format_sources(sources_info):
    """Format source citations for display."""
    if not sources_info:
        return ""
    
    unique_sources = {}
    for src in sources_info:
        key = f"{src['source']}_p{src['page']}"
        if key not in unique_sources:
            unique_sources[key] = src
    
    formatted = "\n\n---\n**Sources Referenced:**\n"
    for src in unique_sources.values():
        formatted += f"- **{src['source']}** (Page {src['page']})\n"
    
    return formatted


# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    
    # API Key section
    api_key = get_api_key()
    
    if not api_key:
        st.warning("API Key Required")
        user_key = st.text_input(
            "Enter Google API Key",
            type="password",
            help="Get your key from https://makersuite.google.com/app/apikey"
        )
        if user_key:
            setup_api_key(user_key)
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key configured")
        if st.button("Change API Key"):
            st.session_state['user_api_key'] = ''
            os.environ.pop('GOOGLE_API_KEY', None)
            st.rerun()
    
    st.divider()
    
    # Document upload section
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        accept_multiple_files=True,
        type="pdf",
        help="Upload your biomedical engineering documents (PDF format)"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear All", use_container_width=True)
    
    # Statistics
    if st.session_state.total_chunks > 0:
        st.divider()
        st.header("Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Total Pages", st.session_state.total_pages)
        col2.metric("Total Chunks", st.session_state.total_chunks)
        st.caption(f"Files processed: {len(st.session_state.file_hashes)}")
    
    # Clear functionality
    if clear_btn:
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.processed_files = set()
        st.session_state.file_hashes = {}
        st.session_state.total_chunks = 0
        st.session_state.total_pages = 0
        st.success("Cleared all data!")
        st.rerun()


# --- PROCESS DOCUMENTS ---
if process_btn and uploaded_files:
    if not get_api_key():
        st.error("Please configure your Google API Key first!")
    else:
        setup_api_key(get_api_key())
        try:
            st.session_state.vector_db = setup_knowledge_base(uploaded_files)
            st.session_state.processing_complete = True
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")


# --- MAIN CHAT INTERFACE ---
# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    if st.session_state.vector_db:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Build conversation history
                history_str = "\n".join([
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.messages[-6:]
                ])
                
                # Get response
                answer, sources = manual_rag_response(
                    prompt,
                    st.session_state.vector_db,
                    history_str
                )
                
                # Format and display
                full_response = answer + format_sources(sources)
                st.markdown(full_response)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload and process documents first using the sidebar.")


# --- FOOTER ---
st.divider()
st.caption("Biomedical Advisor | Powered by Google Gemini & LangChain")
