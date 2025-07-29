# app.py (Definitive Final Version with Smart Routing LLM Support and API Key Fix)
import streamlit as st

# Force clear all caches on startup
st.cache_data.clear()
st.cache_resource.clear()
from rag_system import RAGSystem
import asyncio
import os

# --- Explicitly set API keys from Streamlit secrets into os.environ ---
# This ensures they are available before any LLM clients try to access them.
# The `os.environ.get` will read from Streamlit's internal secrets mechanism.
google_api_key_from_secrets = st.secrets.get("GOOGLE_API_KEY")
if google_api_key_from_secrets:
    os.environ["GOOGLE_API_KEY"] = google_api_key_from_secrets
    print("DEBUG: GOOGLE_API_KEY set in os.environ from st.secrets.")
else:
    print("WARNING: GOOGLE_API_KEY not found in st.secrets.")

openai_api_key_from_secrets = st.secrets.get("OPENAI_API_KEY")
if openai_api_key_from_secrets:
    os.environ["OPENAI_API_KEY"] = openai_api_key_from_secrets
    print("DEBUG: OPENAI_API_KEY set in os.environ from st.secrets.")
else:
    print("WARNING: OPENAI_API_KEY not found in st.secrets.")

@st.cache_resource
def load_rag_system():
    """
    Loads the RAG system and stores it in cache.
    RAGSystem now internally handles LLM initialization and routing.
    """
    print(f"--- FIRST TIME INITIALIZATION OF RAG SYSTEM (cached) ---")
    try:
        rag_system_instance = RAGSystem()
        print(f"--- RAG SYSTEM INITIALIZED SUCCESSFULLY (cached) ---")
        return rag_system_instance
    except Exception as e:
        st.error(f"Error during RAGSystem initialization: {e}")
        print(f"ERROR: RAGSystem initialization failed: {e}")
        raise e

# --- Core App Setup ---
st.set_page_config(page_title="AI Strategy Assistant", layout="wide")
st.title("🤖 AI Strategy Assistant")

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

with st.spinner(f"Initializing knowledge base (using smart routing for LLMs)..."):
    rag_system = load_rag_system()
st.success(f"Knowledge base initialized!")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Sidebar UI ---
with st.sidebar:
    st.header("About")
    st.write("An AI assistant for strategic planning and operations for a children's home.")
    
    if st.button("Clear Chat & Session File"):
        st.session_state.messages = []
        st.session_state.uploaded_filename = None
        rag_system.clear_session()
        st.rerun()

    st.header("📄 Session Documents")
    uploaded_doc = st.file_uploader("Upload a PDF for this session", type=["pdf"])
    
    st.header("🖼️ Image Analysis")
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

# --- File Processing Logic ---
if uploaded_doc:
    if st.session_state.uploaded_filename != uploaded_doc.name:
        with st.spinner(f"Processing '{uploaded_doc.name}' for this session..."):
            file_bytes = uploaded_doc.getvalue()
            rag_system.process_uploaded_file(file_bytes)
            st.session_state.uploaded_filename = uploaded_doc.name
        st.success(f"'{uploaded_doc.name}' loaded for this session!")
        st.rerun() 

if uploaded_image:
    with st.sidebar:
        st.image(uploaded_image, caption="Image for Analysis", use_column_width=True)


# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            st.markdown(message["content"]["answer"])
            with st.expander("Show Sources Consulted"):
                st.write(message["content"]["sources_text"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about strategic planning or operations for a children's home..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever = rag_system.get_current_retriever()
            source_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in source_docs])
            image_bytes = uploaded_image.getvalue() if uploaded_image else None

            result = rag_system.query(
                user_question=prompt, 
                context_text=context_text,
                source_docs=source_docs,
                image_bytes=image_bytes
            )
            answer = result["answer"]
            source_documents = result["source_documents"]
            
            unique_sources = set()
            for source in source_documents:
                source_name = source.metadata.get('source', 'Unknown')
                page_info = f", page {source.metadata.get('page', '')}" if source.metadata.get('page', '') else ""
                unique_sources.add(f"- {os.path.basename(source_name)}{page_info}")
            sources_text = "\n".join(sorted(list(unique_sources))) if unique_sources else "No specific sources were consulted."
            
            st.markdown(answer)
            with st.expander("Show Sources Consulted"):
                st.write(sources_text)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": {"answer": answer, "sources_text": sources_text}
            })
