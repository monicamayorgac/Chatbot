# -*- coding: utf-8 -*-
# streamlit run seminar_bot.py

import streamlit as st
import os
import tempfile
import time

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical Advisor", page_icon="🤖")
st.title("Biomedical Advisor")
st.caption("Your AI assistant for biomedical engineering documents")

# --- API KEY FUNCTION ---
def get_api_key():
    """Get API key from secrets file, environment, or session state."""
    try:
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
    except:
        pass
    if os.environ.get('GOOGLE_API_KEY'):
        return os.environ.get('GOOGLE_API_KEY')
    return st.session_state.get('GOOGLE_API_KEY', '')

# --- 1. SETUP API KEY ---
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = get_api_key()

with st.sidebar:
    st.header("Settings")
    
    current_key = get_api_key()
    if current_key:
        st.success("API Key configured")
        os.environ["GOOGLE_API_KEY"] = current_key
        st.session_state.GOOGLE_API_KEY = current_key
    else:
        api_key = st.text_input("Google API Key", type="password")
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    process_btn = st.button("Process Documents")

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# NEW: Store document names for multi-doc retrieval
if "document_names" not in st.session_state:
    st.session_state.document_names = []

# --- 3. PROCESS DOCUMENTS ---
def setup_knowledge_base(files):
    status_text = st.empty()
    status_text.text("Reading PDFs...")
    
    all_text = []
    doc_names = []
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
            
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = file.name
            
        all_text.extend(docs)
        doc_names.append(file.name)
        os.remove(tmp_path)
    
    # Store document names
    st.session_state.document_names = doc_names
            
    status_text.text("Splitting Text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_text)
    
    status_text.text("💾 Indexing Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embeddings)
    
    status_text.text(f"Ready! Indexed {len(doc_names)} document(s)")
    return vectorstore

if process_btn and uploaded_files and st.session_state.GOOGLE_API_KEY:
    with st.spinner("Analyzing..."):
        try:
            st.session_state.vector_db = setup_knowledge_base(uploaded_files)
        except Exception as e:
            st.error(f"Error processing: {e}")

# --- 4. RAG LOGIC (IMPROVED: retrieves from ALL documents) ---
def manual_rag_response(user_question, db, history, doc_names):
    
    # Get chunks from EACH document (2 chunks per document)
    all_docs = []
    
    if len(doc_names) > 1:
        # Multiple documents: get chunks from each one
        chunks_per_doc = max(2, 8 // len(doc_names))  # Distribute chunks, minimum 2 per doc
        
        for doc_name in doc_names:
            # Search with filter for this specific document
            try:
                docs = db.similarity_search(
                    user_question,
                    k=chunks_per_doc,
                    filter={"source": doc_name}
                )
                all_docs.extend(docs)
            except:
                # If filter doesn't work, fall back to regular search
                pass
        
        # If filtering didn't work, do a larger general search
        if not all_docs:
            all_docs = db.similarity_search(user_question, k=15)
    else:
        # Single document: just get top chunks
        all_docs = db.similarity_search(user_question, k=1)
    
    # Build context organized by document
    context_by_doc = {}
    sources_set = set()
    
    for doc in all_docs:
        source_name = doc.metadata.get('source', 'Unknown')
        page_num = doc.metadata.get('page', 0) + 1
        content = doc.page_content
        
        if source_name not in context_by_doc:
            context_by_doc[source_name] = []
        context_by_doc[source_name].append(f"[Page {page_num}]: {content}")
        sources_set.add(f"{source_name} (Page {page_num})")
    
    # Format context clearly by document
    context_text = ""
    for doc_name, chunks in context_by_doc.items():
        context_text += f"\n\n=== DOCUMENT: {doc_name} ===\n"
        for chunk in chunks:
            context_text += f"{chunk}\n"
    
    # Prompt
    final_prompt = f"""
    You are a helpful advisor. Answer based on the context below.
    The context contains excerpts from {len(doc_names)} document(s): {', '.join(doc_names)}
    
    When asked to summarize or compare, make sure to address ALL documents provided.
    
    CONTEXT: {context_text}
    CHAT HISTORY: {history}
    USER QUESTION: {user_question}
    """
    
    # Generate with Retry
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    
    raw_response = "Error"
    for attempt in range(3):
        try:
            response = llm.invoke(final_prompt)
            
            if isinstance(response.content, list):
                raw_response = ""
                for part in response.content:
                    if isinstance(part, dict) and 'text' in part:
                        raw_response += part['text']
                    elif isinstance(part, str):
                        raw_response += part
            else:
                raw_response = str(response.content)
            break
        except Exception as e:
            if "429" in str(e):
                time.sleep(5)
            else:
                return f"Error: {str(e)}"

    # Format Citations
    if sources_set:
        raw_response += "\n\n---\n**📚 Sources Used:**\n"
        for source in sorted(sources_set):
            raw_response += f"* {source}\n"
            
    return raw_response

# --- 5. CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_db:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
                answer = manual_rag_response(
                    prompt, 
                    st.session_state.vector_db, 
                    history_str,
                    st.session_state.document_names
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("⚠️ Please upload documents first.")

# --- FOOTER ---
st.divider()
st.caption("Biomedical Advisor | Powered by Google Gemini & LangChain")