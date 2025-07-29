# Add this to your existing app.py file

# At the top of your app.py, add this import after your existing imports:
try:
    from compliance_analyzer import ComplianceAnalyzer, create_compliance_interface
    COMPLIANCE_FEATURES_AVAILABLE = True
except ImportError:
    COMPLIANCE_FEATURES_AVAILABLE = False
    st.sidebar.warning("Compliance analyzer not available. Add compliance_analyzer.py to enable advanced features.")

# Add this to your sidebar (after your existing sidebar content):
st.sidebar.header("🔧 Analysis Mode")

if COMPLIANCE_FEATURES_AVAILABLE:
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Type:",
        ["📝 Standard RAG Q&A", "🔍 Compliance Analysis"],
        help="Standard Q&A uses your existing RAG system, Compliance Analysis adds enhanced image analysis"
    )
else:
    analysis_mode = "📝 Standard RAG Q&A"
    st.sidebar.info("Only Standard RAG Q&A available. Add compliance_analyzer.py for more features.")

# Add this where you want the main interface (replace your current main interface):
if analysis_mode == "📝 Standard RAG Q&A":
    # Your existing interface code goes here - NO CHANGES NEEDED
    # Just copy your current interface implementation
    
    st.header("📝 Standard RAG Q&A")
    
    # Your existing question input
    user_question = st.text_area("Ask a question about children's home standards:")
    
    # Your existing file upload (if you have it)
    uploaded_file = st.file_uploader("Upload a file (optional)", type=['pdf'])
    
    if uploaded_file:
        # Your existing file processing code
        pass
    
    # Your existing query button and processing
    if st.button("Ask Question"):
        if user_question:
            with st.spinner("Getting answer..."):
                # Your existing query logic
                retriever = st.session_state.rag_system.get_current_retriever()
                docs = retriever.get_relevant_documents(user_question)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                result = st.session_state.rag_system.query(
                    user_question=user_question,
                    context_text=context,
                    source_docs=docs
                )
                
                # Your existing result display
                st.write("**Answer:**")
                st.write(result["answer"])
                
                # Your existing source documents display (if you have it)
                if result.get("source_documents"):
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.write(f"**Source {i+1}:**")
                            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

elif analysis_mode == "🔍 Compliance Analysis":
    # New compliance analysis interface
    create_compliance_interface()

# Optional: Add system status in sidebar
if COMPLIANCE_FEATURES_AVAILABLE:
    with st.sidebar.expander("📊 System Status"):
        st.write("✅ Standard RAG System: Active")
        st.write("✅ Compliance Analyzer: Active")
        
        if hasattr(st.session_state, 'rag_system'):
            try:
                health = st.session_state.rag_system.get_system_health()
                st.write(f"✅ Models Available: {sum(health['models'].values())}/4")
            except:
                st.write("✅ RAG System: Running")
else:
    with st.sidebar.expander("📊 System Status"):
        st.write("✅ Standard RAG System: Active")
        st.write("❌ Compliance Analyzer: Not Available")

# Optional: Add quick help
with st.sidebar.expander("❓ Help"):
    st.write("**Standard RAG Q&A:**")
    st.write("• Ask questions about children's home standards")
    st.write("• Upload PDFs for context")
    st.write("• Get answers from your knowledge base")
    
    if COMPLIANCE_FEATURES_AVAILABLE:
        st.write("**Compliance Analysis:**")
        st.write("• Upload images for visual compliance assessment")
        st.write("• Get detailed risk analysis and action plans")
        st.write("• Professional compliance reporting")

# Optional: Add footer with version info
st.sidebar.markdown("---")
st.sidebar.caption("🏠 Children's Home RAG System")
if COMPLIANCE_FEATURES_AVAILABLE:
    st.sidebar.caption("✨ Enhanced with Compliance Analysis")

# Add this at the very bottom of your app.py file for debugging
if st.sidebar.checkbox("🔧 Debug Mode"):
    st.subheader("🔧 Debug Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**System Status:**")
        st.write(f"RAG System Loaded: {hasattr(st.session_state, 'rag_system')}")
        st.write(f"Compliance Features: {COMPLIANCE_FEATURES_AVAILABLE}")
        
        if hasattr(st.session_state, 'rag_system'):
            st.write("RAG System Type:", type(st.session_state.rag_system).__name__)
    
    with col2:
        st.write("**Session State Keys:**")
        for key in st.session_state.keys():
            st.write(f"• {key}")

# Example of how your complete app.py structure should look:
"""
# Complete app.py structure example:

import streamlit as st
import os
import base64

# Your existing imports
from rag_system import RAGSystem

# New import for compliance features
try:
    from compliance_analyzer import ComplianceAnalyzer, create_compliance_interface
    COMPLIANCE_FEATURES_AVAILABLE = True
except ImportError:
    COMPLIANCE_FEATURES_AVAILABLE = False

# Your existing page config and title
st.set_page_config(page_title="Children's Home RAG System", page_icon="🏠", layout="wide")
st.title("🏠 Children's Home Compliance Assistant")

# Your existing RAG system initialization
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()

# Sidebar configuration
st.sidebar.header("🔧 Analysis Mode")

if COMPLIANCE_FEATURES_AVAILABLE:
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Type:",
        ["📝 Standard RAG Q&A", "🔍 Compliance Analysis"]
    )
else:
    analysis_mode = "📝 Standard RAG Q&A"

# Main interface based on mode
if analysis_mode == "📝 Standard RAG Q&A":
    # Your existing RAG interface
    st.header("📝 Ask Questions About Children's Home Standards")
    
    user_question = st.text_area("Enter your question:")
    
    if st.button("Get Answer"):
        if user_question:
            # Your existing RAG logic
            pass

elif analysis_mode == "🔍 Compliance Analysis":
    # New compliance interface
    create_compliance_interface()
"""