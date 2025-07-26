# app.py (Corrected Polished Version with Image Uploader)

import streamlit as st
from rag_system import RAGSystem
import asyncio
import os
from langchain_core.messages import HumanMessage
import base64

# Helper Function for Caching
@st.cache_resource
def load_rag_system():
    """Loads the RAG system and stores it in cache."""
    print("--- FIRST TIME INITIALIZATION ---")
    return RAGSystem()

# Core App Setup
st.set_page_config(page_title="AI Strategy Assistant", layout="wide")
st.title("🤖 AI Strategy Assistant")

# Set up the asyncio event loop
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load the RAG system from the cache
with st.spinner("Initializing knowledge base for the first time..."):
    rag_system = load_rag_system()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Sidebar UI with Image Uploader Restored ---
with st.sidebar:
    st.header("About")
    st.write(
        "This AI assistant uses a knowledge base of official documents to answer "
        "questions about strategic planning for a children's home."
    )
    
    if st.button("Clear Chat & Session Files"):
        st.session_state.messages = []
        st.session_state.uploaded_filename = None
        rag_system.update_retriever(None) 
        rag_system.clear_memory()
        # We need to clear the file uploaders too, which is done by rerunning
        st.rerun()

    st.header("📄 Session Documents")
    uploaded_doc = st.file_uploader(
        "Upload a PDF to discuss in this session", type=["pdf"]
    )
    
    # --- IMAGE UPLOADER IS BACK ---
    st.header("🖼️ Image Analysis")
    uploaded_image = st.file_uploader(
        "Upload an image to analyze with your question", type=["png", "jpg", "jpeg"]
    )

# --- File Processing Logic ---
if uploaded_doc:
    if st.session_state.uploaded_filename != uploaded_doc.name:
        with st.spinner(f"Processing '{uploaded_doc.name}'..."):
            file_bytes = uploaded_doc.getvalue()
            session_retriever = rag_system.process_uploaded_file(file_bytes)
            rag_system.update_retriever(session_retriever)
            st.session_state.uploaded_filename = uploaded_doc.name

# Display the uploaded image in the sidebar if it exists
if uploaded_image:
    with st.sidebar:
        st.image(uploaded_image, caption="Image for Analysis", use_column_width=True)

# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # This is the original, more direct logic that we need
            
            # 1. Get the current retriever (ensemble or main)
            current_retriever = rag_system.chain.retriever
            docs = current_retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Construct the prompt message (text or multimodal)
            prompt_template = f"""Use the following context to answer the question. Be expansive and detailed.
            Context: {context_text}
            Question: {prompt}"""

            image_bytes = uploaded_image.getvalue() if uploaded_image else None
            if image_bytes:
                print("Constructing multimodal prompt for LLM.")
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                message_content = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_template},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                )
            else:
                print("Constructing text-only prompt for LLM.")
                message_content = prompt_template

            # 3. Invoke the LLM directly
            # We bypass the conversational chain here to manually insert the image
            # The memory is still being updated by the chain's definition
            response = rag_system.llm.invoke(message_content)
            answer = response.content
            
            # 4. Display the answer and sources
            st.markdown(answer)
            with st.expander("Show Sources Consulted"):
                unique_sources = set()
                for source in docs:
                    source_name = source.metadata.get('source', 'Unknown')
                    page_info = f", page {source.metadata['page'] + 1}" if "page" in source.metadata else ""
                    unique_sources.add(f"- {os.path.basename(source_name)}{page_info}")
                
                sources_text = "\n".join(sorted(list(unique_sources))) if unique_sources else "No specific sources were consulted."
                st.write(sources_text)
            
            # 5. Manually update the conversational memory
            # This is a necessary step since we bypassed the chain.invoke()
            rag_system.memory.save_context({"question": prompt}, {"answer": answer})
            
            full_response_for_history = f"{answer}\n\n**Sources:**\n{sources_text}"
            st.session_state.messages.append({"role": "assistant", "content": full_response_for_history})
