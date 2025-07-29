# app.py (Enhanced Version with Robust Error Handling and Better UX)
import streamlit as st
from rag_system import RAGSystem
import asyncio
import os
import traceback

# --- Page Configuration (must be first Streamlit command) ---
st.set_page_config(
    page_title="AI Strategy Assistant for Children's Homes", 
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Key Setup with Better Error Handling ---
def setup_api_keys():
    """Setup API keys from Streamlit secrets with proper error handling."""
    api_keys_status = {"google": False, "openai": False}
    
    try:
        google_api_key = st.secrets.get("GOOGLE_API_KEY")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            api_keys_status["google"] = True
            print("DEBUG: GOOGLE_API_KEY configured successfully.")
        else:
            st.warning("⚠️ Google API key not found in secrets. Gemini features may be limited.")
            
        openai_api_key = st.secrets.get("OPENAI_API_KEY")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            api_keys_status["openai"] = True
            print("DEBUG: OPENAI_API_KEY configured successfully.")
        else:
            st.warning("⚠️ OpenAI API key not found in secrets. ChatGPT features may be limited.")
            
    except Exception as e:
        st.error(f"Error configuring API keys: {str(e)}")
        print(f"ERROR: API key configuration failed: {e}")
    
    return api_keys_status

# --- Initialize API Keys ---
api_status = setup_api_keys()

@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system with comprehensive error handling."""
    print("--- INITIALIZING RAG SYSTEM (cached) ---")
    
    try:
        rag_system_instance = RAGSystem()
        print("--- RAG SYSTEM INITIALIZED SUCCESSFULLY ---")
        return rag_system_instance
        
    except FileNotFoundError as e:
        error_msg = "❌ Knowledge base not found. Please ensure the FAISS index is properly deployed."
        st.error(error_msg)
        print(f"ERROR: {error_msg} Details: {e}")
        st.stop()
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize the knowledge base: {str(e)}"
        st.error(error_msg)
        print(f"ERROR: RAG System initialization failed: {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        st.stop()

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    if "rag_system_ready" not in st.session_state:
        st.session_state.rag_system_ready = False

# --- Main App Header ---
st.title("🏠 AI Strategy Assistant for Children's Homes")
st.markdown("*Expert guidance on strategic planning, operations, and Ofsted compliance*")

# --- Async Event Loop Setup ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Initialize Session State ---
initialize_session_state()

# --- Load RAG System ---
if not st.session_state.rag_system_ready:
    with st.spinner("🔄 Initializing AI knowledge base with smart routing..."):
        rag_system = load_rag_system()
        st.session_state.rag_system_ready = True
    
    # Show API status
    status_cols = st.columns(2)
    with status_cols[0]:
        if api_status["google"]:
            st.success("✅ Gemini AI Ready")
        else:
            st.error("❌ Gemini AI Unavailable")
    with status_cols[1]:
        if api_status["openai"]:
            st.success("✅ ChatGPT Ready")  
        else:
            st.error("❌ ChatGPT Unavailable")
            
    if not any(api_status.values()):
        st.error("⚠️ No AI models available. Please check your API key configuration.")
        st.stop()
else:
    rag_system = load_rag_system()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("📋 Session Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_filename = None
            rag_system.clear_session()
            st.success("Session reset!")
            st.rerun()

    st.header("📄 Session Documents")
    st.markdown("Upload a PDF to add context for this conversation:")
    uploaded_doc = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        help="This document will be used to enhance responses for this session only."
    )
    
    if st.session_state.uploaded_filename:
        st.success(f"📁 Active: {st.session_state.uploaded_filename}")
    
    st.header("🖼️ Image Analysis")
    st.markdown("Upload an image for multimodal analysis:")
    uploaded_image = st.file_uploader(
        "Choose an image", 
        type=["png", "jpg", "jpeg"],
        help="Images can be analyzed alongside your questions."
    )

# --- File Processing with Enhanced Error Handling ---
if uploaded_doc:
    if st.session_state.uploaded_filename != uploaded_doc.name:
        try:
            with st.spinner(f"📖 Processing '{uploaded_doc.name}'..."):
                file_bytes = uploaded_doc.getvalue()
                rag_system.process_uploaded_file(file_bytes)
                st.session_state.uploaded_filename = uploaded_doc.name
            st.success(f"✅ '{uploaded_doc.name}' loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to process '{uploaded_doc.name}': {str(e)}")
            print(f"ERROR: File processing failed: {e}")

# --- Image Display ---
if uploaded_image:
    with st.sidebar:
        st.image(uploaded_image, caption="Image for Analysis", use_column_width=True)

# --- Main Chat Interface ---
# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            st.markdown(message["content"]["answer"])
            if message["content"].get("sources_text"):
                with st.expander("📚 View Sources", expanded=False):
                    st.markdown(message["content"]["sources_text"])
        else:
            st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask about children's home strategy, operations, or Ofsted requirements..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Analyzing your question..."):
                # Retrieve relevant documents
                retriever = rag_system.get_current_retriever()
                source_docs = retriever.invoke(prompt)
                context_text = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Handle image safely
                image_bytes = None
                if uploaded_image:
                    try:
                        image_bytes = uploaded_image.getvalue()
                    except Exception as e:
                        st.warning(f"⚠️ Could not process image: {str(e)}")

                # Query the RAG system
                result = rag_system.query(
                    user_question=prompt, 
                    context_text=context_text,
                    source_docs=source_docs,
                    image_bytes=image_bytes
                )
                
                answer = result["answer"]
                source_documents = result["source_documents"]

                # Display the response
                st.markdown(answer)
                
                # Format and display sources
                if source_documents:
                    unique_sources = set()
                    for source in source_documents:
                        source_name = source.metadata.get('source', 'Unknown')
                        page_info = f", page {source.metadata.get('page', '')}" if source.metadata.get('page', '') else ""
                        unique_sources.add(f"• {os.path.basename(source_name)}{page_info}")
                    
                    sources_text = "\n".join(sorted(list(unique_sources)))
                    
                    with st.expander("📚 View Sources", expanded=False):
                        st.markdown("**Sources consulted for this response:**")
                        st.markdown(sources_text)
                else:
                    sources_text = "No specific sources were consulted."
                
                # Store in session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": {
                        "answer": answer, 
                        "sources_text": sources_text
                    }
                })
                
        except Exception as e:
            error_message = f"❌ I encountered an error while processing your question: {str(e)}"
            st.error(error_message)
            print(f"ERROR: Query processing failed: {e}")
            print(f"TRACEBACK: {traceback.format_exc()}")
            
            # Store error message in chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I apologize, but I encountered an error processing your question. Please try rephrasing your question or contact support if the issue persists."
            })

# --- Footer ---
with st.container():
    st.markdown("---")
    st.markdown(
        "*This AI assistant provides guidance based on available documentation. "
        "Always verify important decisions with current regulations and professional advice.*"
    )