
#appi key"AIzaSyCVGMgBvJ8UFkhnsPszSh-eLattoT8_GsM"
#/Users/monicamayorga/Documents/Seminar_proyect/Sammy/bin/streamlit run /Users/monicamayorga/Documents/Seminar_proyect/seminar_bot.py
#streamlit run seminar_bot.py
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

# --- 1. SETUP API KEY ---
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = ""

with st.sidebar:
    st.header("Settings")
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

# --- 3. PROCESS DOCUMENTS ---
def setup_knowledge_base(files):
    status_text = st.empty()
    status_text.text("📂 Reading PDFs...")
    
    all_text = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
            
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # FIX: Force real filename
        for doc in docs:
            doc.metadata["source"] = file.name
            
        all_text.extend(docs)
        os.remove(tmp_path)
            
    status_text.text("✂️ Splitting Text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_text)
    
    status_text.text("💾 Indexing Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embeddings)
    
    status_text.text("✅ Ready!")
    return vectorstore

if process_btn and uploaded_files and st.session_state.GOOGLE_API_KEY:
    with st.spinner("Analyzing..."):
        try:
            st.session_state.vector_db = setup_knowledge_base(uploaded_files)
        except Exception as e:
            st.error(f"Error processing: {e}")

# --- 4. RAG LOGIC (With CLEAN TEXT Filter) ---
def manual_rag_response(user_question, db, history):
    # A. Retrieve
    docs = db.similarity_search(user_question, k=3)
    
    # B. Augment
    context_text = ""
    sources_set = set()
    
    for i, doc in enumerate(docs):
        source_name = doc.metadata.get('source', 'Unknown')
        page_num = doc.metadata.get('page', 0) + 1
        content = doc.page_content
        context_text += f"\n[Chunk from {source_name}]: {content}\n"
        sources_set.add(f"{source_name} (Page {page_num})")
    
    # C. Prompt
    final_prompt = f"""
    You are a helpful advisor. Answer based on the context below.
    CONTEXT: {context_text}
    CHAT HISTORY: {history}
    USER QUESTION: {user_question}
    """
    
    # D. Generate with Retry
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
                return f"❌ Error: {str(e)}"

    # E. Format Citations Cleanly
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
                answer = manual_rag_response(prompt, st.session_state.vector_db, history_str)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("ERROR! ⚠️ Please upload documents first.")

