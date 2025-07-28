# app.py (Definitive Final Version with Lazy Initialization)

import streamlit as st
from rag_system import RAGSystem
import asyncio
import os

# --- THIS IS THE KEY ARCHITECTURAL CHANGE ---
# The load_rag_system function is GONE. We will initialize the system inside the chat logic.

# --- Core App Setup ---
st.set_page_config(page_title="AI Strategy Assistant", layout="wide")
st.title("🤖 AI Strategy Assistant")

# Set up the asyncio event loop
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Initialize session state for chat history ONLY
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Sidebar UI (Unchanged) ---
with st.sidebar:
    st.header("About")
    st.write("An AI assistant for strategic planning for a children's home.")
    if st.button("Clear Chat & Session File"):
        st.session_state.messages = []
        st.session_state.uploaded_filename = None
        # If the RAG system exists in the session, clear its session-specific parts
        if "rag_system" in st.session_state:
            st.session_state.rag_system.clear_session()
        st.rerun()
    st.header("📄 Session Documents")
    uploaded_doc = st.file_uploader("Upload a PDF for this session", type=["pdf"])
    st.header("🖼️ Image Analysis")
    uploaded_image = st.file_uploader("Upload an image to analyze", type=["png", "jpg", "jpeg"])

# --- Main Chat Interface ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # --- LAZY INITIALIZATION HAPPENS HERE ---
            # Initialize the RAG system ONCE, the first time a question is asked.
            if "rag_system" not in st.session_state:
                with st.spinner("Starting AI engine for the first time..."):
                    st.session_state.rag_system = RAGSystem()
            
            # File processing now happens right before the query
            if uploaded_doc and st.session_state.uploaded_filename != uploaded_doc.name:
                with st.spinner(f"Processing '{uploaded_doc.name}'..."):
                    file_bytes = uploaded_doc.getvalue()
                    st.session_state.rag_system.process_uploaded_file(file_bytes)
                    st.session_state.uploaded_filename = uploaded_doc.name
            
            # --- The rest of the logic is now safe ---
            current_retriever = st.session_state.rag_system.get_current_retriever()
            docs = current_retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            image_bytes = uploaded_image.getvalue() if uploaded_image else None

            result = st.session_state.rag_system.query(
                user_question=prompt, 
                context_text=context_text,
                image_bytes=image_bytes
            )
            answer = result["answer"]
            
            st.markdown(answer)
            # Add the plain answer to the message history for display
            st.session_state.messages.append({"role": "assistant", "content": answer})
