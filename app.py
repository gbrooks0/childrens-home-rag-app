# app.py - Complete Enhanced Version with All Features

# CRITICAL: SQLite3 Fix MUST be first, before any other imports
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("DEBUG: pysqlite3 imported and set as default sqlite3 module.")
except ImportError:
    print("DEBUG: pysqlite3 not found, falling back to system sqlite3.")

# Now safe to import other modules
import streamlit as st
import os
import re
from rag_system import RAGSystem

# Safe import for compliance features
try:
    from compliance_analyzer import ComplianceAnalyzer, create_compliance_interface
    COMPLIANCE_FEATURES_AVAILABLE = True
except ImportError:
    COMPLIANCE_FEATURES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Children's Home Management System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Common question patterns for auto-complete
COMMON_QUESTIONS = [
    "How do we prepare for an Ofsted inspection?",
    "What are the requirements for staff training?",
    "How do we develop a care plan for a new child?",
    "What's the best approach to managing challenging behavior?",
    "How do we implement safeguarding procedures?",
    "What documentation is required for children's homes?",
    "How do we support children with trauma?",
    "What are the fire safety requirements?",
    "How do we manage staff retention and recruitment?",
    "What budget planning considerations should we include?",
    "How do we create therapeutic environments?",
    "What are the education support requirements?",
    "How do we handle complaints and concerns?",
    "What's required for medication management?",
    "How do we support care leavers transitioning?",
    "What are the room standards and personalisation requirements?",
    "How do we manage relationships with social workers?",
    "What training is required for new staff?",
    "How do we implement positive behavior support?",
    "What are the requirements for recording and reporting?"
]

def get_contextual_tip(current_input=""):
    """Provide contextual tips based on what user is typing."""
    tips = [
        "💡 **Tip:** Include specific details like ages, timelines, or current challenges for more targeted advice",
        "💡 **Tip:** Mention your current Ofsted rating if asking about improvements or inspections",
        "💡 **Tip:** Include the number of children and staff when asking about operational questions",
        "💡 **Tip:** Specify if you're asking about a particular child's needs or general policy guidance",
        "💡 **Tip:** Mention any deadlines or urgent timescales to get prioritized recommendations"
    ]
    
    # Context-aware tips based on input content
    if "ofsted" in current_input.lower() or "inspection" in current_input.lower():
        return "💡 **Tip:** Mention your current rating and inspection timeline for more specific preparation guidance"
    elif "child" in current_input.lower() or "young person" in current_input.lower():
        return "💡 **Tip:** Include the child's age, specific needs, or challenges for more personalized guidance"
    elif "staff" in current_input.lower():
        return "💡 **Tip:** Specify staff roles, experience levels, or specific training needs for targeted advice"
    elif "budget" in current_input.lower() or "cost" in current_input.lower():
        return "💡 **Tip:** Include your home size, capacity, or specific financial challenges for relevant guidance"
    else:
        return tips[len(current_input) % len(tips)]

def get_question_suggestions(input_text):
    """Get auto-complete suggestions based on user input."""
    if len(input_text) < 3:
        return []
    
    input_lower = input_text.lower()
    suggestions = []
    
    for question in COMMON_QUESTIONS:
        if any(word in question.lower() for word in input_lower.split()):
            suggestions.append(question)
        if len(suggestions) >= 5:
            break
    
    return suggestions

# Initialize RAG system
@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the RAG system."""
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG System: {e}")
        return None

if 'rag_system' not in st.session_state:
    with st.spinner("Initializing system..."):
        st.session_state.rag_system = initialize_rag_system()

if st.session_state.rag_system is None:
    st.error("❌ System initialization failed. Please refresh the page.")
    st.stop()

# Enhanced CSS for clean design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    
    .contextual-tip {
        background-color: #f8f9fa;
        border-left: 3px solid #17a2b8;
        padding: 0.75rem;
        border-radius: 0 5px 5px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .tab-container {
        margin: 2rem 0;
    }
    
    .quick-action-grid {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Clean header - HIGHEST POSSIBLE POSITION
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🏠 Children's Home Management System")
st.markdown("*Strategic, operational & compliance guidance for residential care*")
st.markdown('</div>', unsafe_allow_html=True)

# Clean tab interface immediately after title
st.markdown('<div class="tab-container">', unsafe_allow_html=True)
mode_tab1, mode_tab2 = st.tabs(["💬 Ask Questions", "📷 Analyze Images"])
st.markdown('</div>', unsafe_allow_html=True)

with mode_tab1:
    # Check if we have a result to display
    if 'current_result' in st.session_state and st.session_state.current_result:
        # DISPLAY ANSWER AT TOP - replacing question area
        st.subheader("🧠 Expert Guidance")
        st.write(st.session_state.current_result["answer"])
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Ask New Question", type="primary"):
                # Clear result and show question form again
                del st.session_state.current_result
                if 'current_question' in st.session_state:
                    st.session_state.current_question = ""
                st.rerun()
        
        with col2:
            if st.button("📋 View Key Actions"):
                st.session_state.show_actions = not st.session_state.get('show_actions', False)
        
        with col3:
            if st.button("📚 View Sources"):
                st.session_state.show_sources = not st.session_state.get('show_sources', False)
        
        # Show additional info if requested
        if st.session_state.get('show_actions', False):
            st.markdown("---")
            st.markdown("**📋 Key Actions:**")
            st.markdown("• Review with your management team\n• Consider implementation timeline\n• Monitor and evaluate outcomes")
        
        if st.session_state.get('show_sources', False) and st.session_state.current_result.get("source_documents"):
            st.markdown("---")
            st.markdown(f"**📚 Sources ({len(st.session_state.current_result['source_documents'])} references):**")
            for i, doc in enumerate(st.session_state.current_result["source_documents"], 1):
                with st.expander(f"Source {i}"):
                    preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    st.write(preview)
    
    else:
        # SHOW QUESTION FORM - only when no result
        st.subheader("💬 Ask Your Question")
        
        # Initialize session state for question
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        # Handle quick action selection
        if hasattr(st.session_state, 'quick_question'):
            st.session_state.current_question = st.session_state.quick_question
            delattr(st.session_state, 'quick_question')
        
        # Question input with keyboard shortcut
        user_question = st.text_area(
            "Describe your situation or question:",
            value=st.session_state.current_question,
            placeholder="Start typing your question... (e.g., 'How do we prepare for...')  |  Press Ctrl+Enter to submit",
            height=120,
            key="question_input",
            on_change=None
        )
        
        # Add JavaScript for Ctrl+Enter functionality
        st.markdown("""
        <script>
        function addKeyboardShortcut() {
            const textArea = document.querySelector('textarea[aria-label="Describe your situation or question:"]');
            if (textArea && !textArea.hasEventListener) {
                textArea.hasEventListener = true;
                textArea.addEventListener('keydown', function(e) {
                    if (e.ctrlKey && e.key === 'Enter') {
                        e.preventDefault();
                        // Find the guidance button and click it
                        const buttons = document.querySelectorAll('button');
                        for (let button of buttons) {
                            if (button.textContent.includes('Get Expert Guidance')) {
                                button.click();
                                break;
                            }
                        }
                    }
                });
            }
        }
        
        // Run after a short delay to ensure DOM is loaded
        setTimeout(addKeyboardShortcut, 100);
        
        // Also run when Streamlit reruns
        if (window.streamlitReady) {
            addKeyboardShortcut();
        } else {
            window.addEventListener('load', addKeyboardShortcut);
        }
        </script>
        """, unsafe_allow_html=True)
        
        # Get guidance button - ALWAYS VISIBLE with question
        if st.button("🧠 Get Expert Guidance", type="primary", use_container_width=True):
            if user_question and user_question.strip():
                # Enhanced loading with progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Initialize
                    status_text.text("🔍 Analyzing your question...")
                    progress_bar.progress(20)
                    
                    # Step 2: Retrieve relevant documents
                    status_text.text("📚 Searching knowledge base...")
                    progress_bar.progress(40)
                    retriever = st.session_state.rag_system.get_current_retriever()
                    docs = retriever.get_relevant_documents(user_question)
                    
                    # Step 3: Prepare context
                    status_text.text("🧠 Preparing expert context...")
                    progress_bar.progress(60)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Step 4: Generate response with appropriate prompt based on question type
                    status_text.text("💭 Generating expert guidance...")
                    progress_bar.progress(80)
                    
                    # Determine if this is a factual/informational question or strategic guidance request
                    question_lower = user_question.lower()
                    
                    # Factual question indicators
                    factual_indicators = [
                        "what are", "what is", "list", "define", "explain", "describe",
                        "standards", "requirements", "regulations", "according to",
                        "how many", "which", "who", "when", "where"
                    ]
                    
                    # Strategic question indicators  
                    strategic_indicators = [
                        "how do we", "how can we", "what should we", "help us",
                        "strategy", "implement", "improve", "develop", "plan",
                        "guidance", "advice", "recommend", "approach", "best practice"
                    ]
                    
                    is_factual = any(indicator in question_lower for indicator in factual_indicators)
                    is_strategic = any(indicator in question_lower for indicator in strategic_indicators)
                    
                    if is_factual and not is_strategic:
                        # Provide comprehensive factual information
                        enhanced_question = f"""
                        Question: {user_question}
                        
                        Please provide a comprehensive and detailed factual response that covers:
                        1. Complete listing of all relevant standards/requirements
                        2. Specific regulation numbers where applicable
                        3. Clear explanation of what each standard covers
                        4. Key requirements and expectations for each
                        5. Important details that practitioners need to know
                        6. Any interconnections between different standards
                        
                        Format as clear, informative content with proper structure and detail. Focus on accuracy and completeness of factual information.
                        """
                    else:
                        # Provide strategic guidance
                        enhanced_question = f"""
                        As a senior children's home management consultant, provide comprehensive strategic and operational guidance for:
                        
                        Question: {user_question}
                        
                        Please provide:
                        1. Strategic considerations and business implications
                        2. Practical implementation steps with timelines
                        3. Risk management and mitigation strategies  
                        4. Child-centered impact assessment
                        5. Resource requirements and cost considerations
                        6. Success metrics and monitoring approaches
                        7. Specific actionable recommendations
                        
                        Format your response as professional management guidance for children's home leaders.
                        """
                    
                    result = st.session_state.rag_system.query(
                        user_question=enhanced_question,
                        context_text=context,
                        source_docs=docs
                    )
                    
                    # Step 5: Complete
                    status_text.text("✅ Guidance ready!")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store result and trigger rerun to show answer
                    if result and result.get("answer"):
                        st.session_state.current_result = result
                        st.session_state.current_question = user_question
                        st.rerun()
                    else:
                        st.error("❌ Sorry, I couldn't generate a response. Please try rephrasing your question.")
                        
                except Exception as e:
                    # Clear progress indicators on error
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.error("❌ An error occurred while processing your question.")
                    
                    # Show helpful error information
                    with st.expander("🔧 Error Details"):
                        st.code(f"Error type: {type(e).__name__}")
                        st.code(f"Error message: {str(e)}")
                        
                        st.markdown("**💡 Try these solutions:**")
                        st.markdown("""
                        • **Refresh the page** and try again
                        • **Rephrase your question** with more specific details
                        • **Check your internet connection**
                        • **Try a shorter, simpler question first**
                        • **Use one of the Quick Action buttons** instead
                        """)
            else:
                st.warning("⚠️ Please enter a question to receive guidance")
        
        # Auto-complete suggestions - only show when typing
        if user_question and len(user_question) > 3:
            suggestions = get_question_suggestions(user_question)
            if suggestions:
                st.markdown("**💡 Suggested completions:**")
                for i, suggestion in enumerate(suggestions):
                    if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                        st.session_state.current_question = suggestion
                        st.rerun()
        
        # Contextual help
        tip = get_contextual_tip(user_question)
        st.markdown(f'<div class="contextual-tip">{tip}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Action Buttons - Below question input
        st.subheader("🚀 Quick Actions")
        st.markdown("*Or choose from these common scenarios:*")
        
        quick_actions = {
            "🚨 Inspection Prep": "We have an upcoming Ofsted inspection. What should we focus on to ensure we're fully prepared and can demonstrate our improvements?",
            "💰 Budget Planning": "Help us develop an effective budget strategy for the upcoming financial year while maintaining quality of care.",
            "👥 Staff Issues": "We're experiencing staff retention challenges. What strategies can we implement to improve staff satisfaction and reduce turnover?",
            "🏠 New Admission": "We're admitting a new child to our home. What processes and considerations should we prioritize for a successful transition?",
            "📋 Policy Review": "We need to review and update our policies. What are the current best practices and regulatory requirements we should include?",
            "🎯 Quality Improvement": "We want to enhance our quality of care and move towards outstanding. What key areas should we focus on?"
        }
        
        # Display quick actions in a clean grid
        st.markdown('<div class="quick-action-grid">', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (action, question) in enumerate(quick_actions.items()):
            with cols[i % 3]:
                if st.button(action, key=f"quick_{i}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Optional document upload - simplified
        with st.expander("📎 Add Supporting Documents (Optional)"):
            uploaded_file = st.file_uploader(
                "Upload relevant documents for context",
                type=['pdf', 'docx', 'txt'],
                help="Upload policies, reports, or other documents for more tailored guidance"
            )
            
            if uploaded_file:
                st.success(f"✅ {uploaded_file.name} uploaded")
                try:
                    st.session_state.rag_system.process_uploaded_file(uploaded_file.read())
                    st.info("📚 Document integrated for this session")
                except Exception as e:
                    st.error(f"Failed to process: {e}")

with mode_tab2:
    # Visual analysis high on page
    if not COMPLIANCE_FEATURES_AVAILABLE:
        st.error("❌ Visual analysis features are currently being upgraded.")
        st.info("💡 Use the 'Ask Questions' tab for comprehensive guidance.")
    else:
        st.subheader("📷 Visual Compliance Analysis")
        st.markdown("Upload facility images for comprehensive assessment and insights.")
        
        # Initialize compliance analyzer
        if 'compliance_analyzer' not in st.session_state:
            st.session_state.compliance_analyzer = ComplianceAnalyzer(st.session_state.rag_system)
        
        # Smart image analysis prompts
        st.markdown("**🎯 What would you like to focus on?**")
        
        analysis_focus_options = {
            "🛡️ Safety Check": "Analyze this image for health and safety compliance, focusing on fire safety, hazards, and emergency procedures.",
            "📋 Compliance Review": "Conduct a comprehensive compliance assessment covering all regulatory requirements and standards.",
            "🏠 Environment Assessment": "Evaluate this space for quality of environment, personalisation, and child experience.",
            "🍎 Food Safety Focus": "Analyze this kitchen/dining area for food hygiene, safety, and nutritional considerations.",
            "💚 Wellbeing Spaces": "Assess how well this environment supports children's emotional wellbeing and therapeutic needs."
        }
        
        selected_focus = None
        cols = st.columns(3)
        for i, (focus_type, prompt) in enumerate(analysis_focus_options.items()):
            with cols[i % 3]:
                if st.button(focus_type, key=f"focus_{i}"):
                    selected_focus = prompt
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload facility image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload photos of any facility area for comprehensive analysis"
        )
        
        if uploaded_image:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(uploaded_image, caption="Image for Analysis", use_container_width=True)
            
            with col2:
                st.markdown("**Analysis Coverage:**")
                coverage_areas = [
                    "🛡️ Health & Safety",
                    "🍎 Food & Nutrition", 
                    "🤝 Dignity & Respect",
                    "🏠 Personalisation",
                    "💚 Wellbeing Spaces",
                    "🔒 Safeguarding",
                    "🏢 Environment Quality"
                ]
                
                for area in coverage_areas:
                    st.write(f"• {area}")
            
            # Custom analysis focus
            custom_focus = st.text_area(
                "Custom analysis focus (optional):",
                value=selected_focus or "",
                placeholder="e.g., 'Focus on therapeutic environment and child comfort' or leave blank for general analysis",
                height=60
            )
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                # Enhanced loading for image analysis
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Prepare analysis
                    status_text.text("📷 Processing uploaded image...")
                    progress_bar.progress(25)
                    
                    question = custom_focus if custom_focus.strip() else "Analyze this image comprehensively for compliance, quality, and child experience."
                    image_bytes = uploaded_image.read()
                    
                    # Step 2: Initialize analyzer
                    status_text.text("🧠 Initializing compliance analysis...")
                    progress_bar.progress(50)
                    
                    # Step 3: Conduct analysis
                    status_text.text("🔍 Conducting comprehensive visual analysis...")
                    progress_bar.progress(75)
                    
                    result = st.session_state.compliance_analyzer.analyze_image_compliance(question, image_bytes)
                    
                    # Step 4: Complete
                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    if result:
                        # Clean results display
                        st.markdown("---")
                        st.subheader("📊 Analysis Results")
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Score", f"{result.overall_compliance_score}%")
                        
                        with col2:
                            st.metric("Areas Reviewed", len(result.category_distribution))
                        
                        with col3:
                            st.metric("Findings", result.total_issues)
                        
                        with col4:
                            critical_count = result.risk_distribution.get("CRITICAL", 0)
                            st.metric("Priority Items", critical_count)
                        
                        # Priority actions
                        if result.priority_actions:
                            st.subheader("🎯 Priority Actions")
                            for i, action in enumerate(result.priority_actions, 1):
                                st.write(f"{i}. {action}")
                        
                        # Positive observations
                        if result.positive_observations:
                            st.subheader("✅ Strengths Identified")
                            for observation in result.positive_observations:
                                st.success(f"• {observation}")
                        
                        # Strategic recommendations
                        if result.recommendations:
                            st.subheader("💡 Recommendations")
                            for rec in result.recommendations:
                                st.write(f"• {rec}")
                    else:
                        st.error("❌ Analysis failed to produce results. Please try again with a different image.")
                        
                except Exception as e:
                    # Clear progress indicators on error
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.error("❌ Image analysis failed.")
                    
                    # Show helpful error information  
                    with st.expander("🔧 Error Details"):
                        st.code(f"Error: {str(e)}")
                        st.markdown("**💡 Try these solutions:**")
                        st.markdown("""
                        • **Check image format** - use JPG, PNG, or JPEG
                        • **Ensure image is clear** and well-lit
                        • **Try a smaller image** (under 10MB)
                        • **Refresh the page** and try again
                        • **Use a different image** of the same area
                        """)
        else:
            st.info("👆 Upload an image to begin visual analysis")

# Minimal sidebar with just essential status
with st.sidebar:
    st.markdown("### 📊 System Status")
    st.success("✅ Guidance System: Online")
    st.success("✅ Knowledge Base: Current")
    if COMPLIANCE_FEATURES_AVAILABLE:
        st.success("✅ Visual Analysis: Available")
    else:
        st.info("ℹ️ Visual Analysis: Upgrading")
    
    # Quick help
    with st.expander("❓ Quick Help"):
        st.markdown("""
        **Ask Questions:**
        • Use quick actions for common scenarios
        • Be specific about your situation
        • Include relevant context and timelines
        
        **Visual Analysis:**
        • Upload clear, well-lit images
        • Use focus options for targeted analysis
        • Review all findings and recommendations
        """)

# Clean footer
st.markdown("---")
st.markdown("🏠 **Children's Home Management System** - Professional guidance for residential care excellence")
