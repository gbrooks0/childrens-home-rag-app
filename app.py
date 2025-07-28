# app.py (Definitive Final Version with All Syntax Errors Corrected and Logic Improved)

import streamlit as st
from rag_system import RAGSystem
import asyncio
import os

@st.cache_resource
def load_rag_system():
    """Loads the RAG system and stores it in cache."""
    print("--- FIRST TIME INITIALIZATION ---")
    return RAGSystem()

# --- Core App Setup ---
st.set_page_config(page_title="AI Strategy Assistant", layout="wide")
st.title("🤖 AI Strategy Assistant")

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

with st.spinner("Initializing knowledge base..."):
    rag_system = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Sidebar UI ---
with st.sidebar:
    st.header("About")
    st.write("An AI assistant for strategic planning for a children's home.")
    
    if st.button("Clear Chat & Session File"):
        st.session_state.messages = []
        st.session_state.uploaded_filename = None
        rag_system.clear_session()
        st.rerun()

    st.header("📄 Session Documents")
    uploaded_doc = st.file_uploader("Upload a PDF for this session", type=["pdf"])
    
    st.header("🖼️ Image Analysis")
    uploaded_image = st.file_uploader("Upload an image to analyze", type=["png", "jpg", "jpeg"])

# --- File Processing Logic ---
if uploaded_doc:
    if st.session_state.uploaded_filename != uploaded_doc.name:
        with st.spinner(f"Processing '{uploaded_doc.name}'..."):
            file_bytes = uploaded_doc.getvalue()
            rag_system.process_uploaded_file(file_bytes)
            st.session_state.uploaded_filename = uploaded_doc.name
            
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

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the appropriate retriever (main or ensemble)
            retriever = rag_system.get_current_retriever()
            
            # Retrieve source documents based on the prompt
            # THIS IS WHERE 'source_docs' IS DEFINED, PREVIOUSLY 'docs'
            source_docs = retriever.invoke(prompt)
            
            # Join the page content of the retrieved documents to form the context text
            # 'source_docs' is used here instead of the undefined 'docs'
            context_text = "\n\n".join([doc.page_content for doc in source_docs])
            
            # Prepare image bytes if an image was uploaded
            image_bytes = uploaded_image.getvalue() if uploaded_image else None

            # Query the RAG system
            result = rag_system.query(
                user_question=prompt, 
                context_text=context_text,
                source_docs=source_docs, # Pass the retrieved docs
                image_bytes=image_bytes
            )
            answer = result["answer"]
            source_documents = result["source_documents"] # This will be the same as source_docs

            # Process unique sources for display
            unique_sources = set()
            for source in source_documents:
                source_name = source.metadata.get('source', 'Unknown')
                page_info = f", page {source.metadata.get('page', '')}" if source.metadata.get('page', '') else ""
                unique_sources.add(f"- {os.path.basename(source_name)}{page_info}")
            sources_text = "\n".join(sorted(list(unique_sources))) if unique_sources else "No specific sources were consulted."
            
            # Display the answer and sources
            st.markdown(answer)
            with st.expander("Show Sources Consulted"):
                st.write(sources_text)
            
            # Append the assistant's response to the chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": {"answer": answer, "sources_text": sources_text}
            })
