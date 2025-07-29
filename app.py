# app.py - FINAL COMPLETE VERSION WITH ALL FIXES
# Includes all enhancements and fixes discussed in our conversation

# CRITICAL: SQLite3 Fix MUST be first, before any other imports
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("DEBUG: pysqlite3 imported and set as default sqlite3 module.")
except ImportError:
    print("DEBUG: pysqlite3 not found, falling back to system sqlite3.")

# Safe imports after SQLite fix
import streamlit as st
import os
import re
import time
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

# ENHANCED FEATURES: Auto-complete question database
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
    "What are the requirements for recording and reporting?",
    "What are the quality standards for children's homes according to Ofsted?",
    "How do we improve our Ofsted rating from Good to Outstanding?",
    "What are the staffing requirements for children's homes?",
    "How do we handle financial management and budgeting?",
    "What policies and procedures do we need?"
]

def get_contextual_tip(current_input=""):
    """ENHANCED FEATURE: Provide dynamic contextual tips based on user input."""
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
    elif "draft" in current_input.lower() or "write" in current_input.lower():
        return "💡 **Tip:** Provide context about the purpose, audience, and key points you want to include"
    else:
        return tips[len(current_input) % len(tips)]

def get_question_suggestions(input_text):
    """ENHANCED FEATURE: Auto-complete suggestions based on user input."""
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

# ENHANCED INITIALIZATION with proper error handling
if 'rag_system' not in st.session_state:
    with st.spinner("Initializing comprehensive guidance system..."):
        st.session_state.rag_system = initialize_rag_system()

if st.session_state.rag_system is None:
    st.error("❌ System initialization failed. Please refresh the page.")
    st.stop()

# ENHANCED CSS for professional appearance
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
    
    .model-attribution {
        font-size: 0.8rem;
        color: #6c757d;
        font-style: italic;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# CLEAN HEADER - Positioned at very top as requested
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🏠 Children's Home Management System")
st.markdown("*Strategic, operational & compliance guidance for residential care*")
st.markdown('</div>', unsafe_allow_html=True)

# CLEAN TAB INTERFACE - Immediately after title as requested
st.markdown('<div class="tab-container">', unsafe_allow_html=True)
mode_tab1, mode_tab2 = st.tabs(["💬 Ask Questions", "📷 Analyze Images"])
st.markdown('</div>', unsafe_allow_html=True)

with mode_tab1:
    # ENHANCED FEATURE: Answer replacement at top of page
    if 'current_result' in st.session_state and st.session_state.current_result:
        # DISPLAY ANSWER AT TOP - replacing question area
        st.subheader("🧠 Expert Guidance")
        
        # ENHANCED FEATURE: Show which AI model was used
        if 'model_used' in st.session_state and st.session_state.model_used:
            st.markdown(f'<div class="model-attribution">Response generated by {st.session_state.model_used}</div>', unsafe_allow_html=True)
        
        st.write(st.session_state.current_result["answer"])
        
        # CLEAN ACTION BUTTONS
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Ask New Question", type="primary"):
                # Clear result and show question form again
                del st.session_state.current_result
                if 'current_question' in st.session_state:
                    st.session_state.current_question = ""
                if 'model_used' in st.session_state:
                    del st.session_state.model_used
                st.rerun()
        
        with col2:
            if st.button("📋 View Key Actions"):
                st.session_state.show_actions = not st.session_state.get('show_actions', False)
        
        with col3:
            if st.button("📚 View Sources"):
                st.session_state.show_sources = not st.session_state.get('show_sources', False)
        
        # ENHANCED FEATURE: Progressive disclosure of additional information
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
        # ENHANCED QUESTION FORM - Only when no result
        st.subheader("💬 Ask Your Question")
        
        # Initialize session state for question
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        # ENHANCED FEATURE: Handle quick action selection
        if hasattr(st.session_state, 'quick_question'):
            st.session_state.current_question = st.session_state.quick_question
            delattr(st.session_state, 'quick_question')
        
        # ENHANCED QUESTION INPUT with keyboard shortcut support
        user_question = st.text_area(
            "Describe your situation or question:",
            value=st.session_state.current_question,
            placeholder="Start typing your question... (e.g., 'How do we prepare for...')  |  Press Ctrl+Enter to submit",
            height=120,
            key="question_input"
        )
        
        # ENHANCED FEATURE: Robust Ctrl+Enter functionality
        st.components.v1.html("""
        <script>
        function setupKeyboardShortcut() {
            // Find the textarea by looking for the specific placeholder text
            const textAreas = document.querySelectorAll('textarea');
            let targetTextArea = null;
            
            for (let textarea of textAreas) {
                if (textarea.placeholder && textarea.placeholder.includes('Start typing your question')) {
                    targetTextArea = textarea;
                    break;
                }
            }
            
            if (targetTextArea && !targetTextArea.hasCtrlEnterListener) {
                targetTextArea.hasCtrlEnterListener = true;
                
                targetTextArea.addEventListener('keydown', function(e) {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        // Find and click the Get Expert Guidance button
                        const buttons = document.querySelectorAll('button');
                        for (let button of buttons) {
                            const buttonText = button.textContent || button.innerText;
                            if (buttonText.includes('Get Expert Guidance')) {
                                console.log('Keyboard shortcut triggered - clicking button');
                                button.click();
                                return;
                            }
                        }
                        
                        // Alternative: try finding by button attributes
                        const primaryButtons = document.querySelectorAll('button[kind="primary"]');
                        if (primaryButtons.length > 0) {
                            console.log('Fallback: clicking primary button');
                            primaryButtons[0].click();
                        }
                    }
                });
                
                console.log('Ctrl+Enter listener added successfully');
            }
        }
        
        // Try multiple times to ensure it works across Streamlit reruns
        setTimeout(setupKeyboardShortcut, 100);
        setTimeout(setupKeyboardShortcut, 500);
        setTimeout(setupKeyboardShortcut, 1000);
        
        // Set up observer for when Streamlit reruns the page
        const observer = new MutationObserver(function(mutations) {
            setupKeyboardShortcut();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        </script>
        """, height=0)
        
        # MAIN SUBMIT BUTTON - Always visible as requested
        button_clicked = st.button("🧠 Get Expert Guidance", type="primary", use_container_width=True, key="guidance_button")
        
        # ENHANCED PROCESSING with improved loading indicators and dual model support
        if button_clicked:
            if user_question and user_question.strip():
                # ENHANCED FEATURE: Improved loading indicators
                progress_container = st.empty()
                status_container = st.empty()
                
                try:
                    # Step 1: Initialize with visible progress
                    with progress_container.container():
                        progress_bar = st.progress(0)
                    with status_container.container():
                        st.info("🔍 Analyzing your question...")
                    time.sleep(0.5)
                    
                    # Step 2: Retrieve relevant documents
                    with progress_container.container():
                        progress_bar = st.progress(40)
                    with status_container.container():
                        st.info("📚 Searching knowledge base...")
                    
                    retriever = st.session_state.rag_system.get_current_retriever()
                    docs = retriever.get_relevant_documents(user_question)
                    time.sleep(0.5)
                    
                    # Step 3: Prepare context
                    with progress_container.container():
                        progress_bar = st.progress(60)
                    with status_container.container():
                        st.info("🧠 Preparing expert context...")
                    
                    context = "\n\n".join([doc.page_content for doc in docs])
                    time.sleep(0.5)
                    
                    # Step 4: ENHANCED FEATURE - Smart response type detection
                    with progress_container.container():
                        progress_bar = st.progress(80)
                    with status_container.container():
                        st.info("💭 Generating expert guidance...")
                    
                    # ENHANCED FEATURE: Three response types based on question intent
                    question_lower = user_question.lower()
                    
                    # Creation/drafting requests - user wants actual deliverable
                    creation_indicators = [
                        "draft", "write", "create", "develop a", "make a", "compose",
                        "help me write", "help me draft", "help me create", "generate a",
                        "design a", "build a", "prepare a", "produce a"
                    ]
                    
                    # Factual/informational questions - user wants information
                    factual_indicators = [
                        "what are", "what is", "list", "define", "explain", "describe",
                        "standards", "requirements", "regulations", "according to",
                        "how many", "which", "who", "when", "where", "tell me about"
                    ]
                    
                    # Strategic guidance requests - user wants strategic advice
                    strategic_indicators = [
                        "how do we", "how can we", "what should we", "help us develop",
                        "strategy for", "approach to", "improve our", "best practices for",
                        "guidance on", "advice on", "recommend", "suggestions for"
                    ]
                    
                    is_creation_request = any(indicator in question_lower for indicator in creation_indicators)
                    is_factual = any(indicator in question_lower for indicator in factual_indicators)
                    is_strategic = any(indicator in question_lower for indicator in strategic_indicators)
                    
                    if is_creation_request:
                        # User wants something created/drafted
                        enhanced_question = f"""
                        The user is asking you to create/draft something specific. Please provide the actual deliverable they requested, not strategic advice about how to create it.
                        
                        Request: {user_question}
                        
                        Please create the specific item requested (email, policy, plan, document, etc.) in a ready-to-use format. Focus on providing the actual content they can use directly, not guidance about how to create it.
                        
                        Format the response as the finished deliverable with appropriate structure, professional language, and complete content.
                        """
                    elif is_factual and not is_strategic:
                        # User wants factual information
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
                        # User wants strategic guidance
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
                    
                    # ENHANCED FEATURE: Use dual-model system from rag_system.py
                    result = st.session_state.rag_system.query(
                        user_question=enhanced_question,
                        context_text=context,
                        source_docs=docs
                    )
                    
                    # Step 5: Complete with model attribution
                    with progress_container.container():
                        progress_bar = st.progress(100)
                    
                    # ENHANCED FEATURE: Show which model was used
                    if result and result.get("metadata") and result["metadata"].get("llm_used"):
                        used_model = result["metadata"]["llm_used"]
                        with status_container.container():
                            st.success(f"✅ Response generated by {used_model}")
                    else:
                        with status_container.container():
                            st.success("✅ Guidance ready!")
                    
                    time.sleep(1)  # Let users see completion
                    
                    # Clear progress indicators
                    progress_container.empty()
                    status_container.empty()
                    
                    # ENHANCED FEATURE: Store result and show at top
                    if result and result.get("answer"):
                        st.session_state.current_result = result
                        st.session_state.current_question = user_question
                        # Store which model was used for display
                        if result.get("metadata") and result["metadata"].get("llm_used"):
                            st.session_state.model_used = result["metadata"]["llm_used"]
                        st.rerun()
                    else:
                        st.error("❌ Sorry, I couldn't generate a response. This might indicate an issue with both Gemini and ChatGPT models.")
                        st.info("💡 The system attempts to use both Gemini and ChatGPT with smart routing. If both fail, please check your API configuration or try again later.")
                        
                except Exception as e:
                    # ENHANCED ERROR HANDLING
                    progress_container.empty()
                    status_container.empty()
                    
                    st.error("❌ An error occurred while processing your question.")
                    
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
        
        # ENHANCED FEATURE: Auto-complete suggestions
        if user_question and len(user_question) > 3:
            suggestions = get_question_suggestions(user_question)
            if suggestions:
                st.markdown("**💡 Suggested completions:**")
                for i, suggestion in enumerate(suggestions):
                    if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                        st.session_state.current_question = suggestion
                        st.rerun()
        
        # ENHANCED FEATURE: Contextual help
        tip = get_contextual_tip(user_question)
        st.markdown(f'<div class="contextual-tip">{tip}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ENHANCED FEATURE: Quick Action Buttons
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
        
        # ENHANCED FEATURE: Document upload
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
    # ENHANCED VISUAL ANALYSIS TAB
    if not COMPLIANCE_FEATURES_AVAILABLE:
        st.error("❌ Visual analysis features are currently being upgraded.")
        st.info("💡 Use the 'Ask Questions' tab for comprehensive guidance.")
    else:
        st.subheader("📷 Visual Compliance Analysis")
        st.markdown("Upload facility images for comprehensive assessment and insights.")
        
        # Initialize compliance analyzer
        if 'compliance_analyzer' not in st.session_state:
            st.session_state.compliance_analyzer = ComplianceAnalyzer(st.session_state.rag_system)
        
        # ENHANCED FEATURE: Smart image analysis prompts
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
                # FIXED: use_container_width instead of deprecated use_column_width
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
                # ENHANCED loading for image analysis
                progress_container = st.empty()
                status_container = st.empty()
                
                try:
                    # Step 1: Prepare analysis
                    with progress_container.container():
                        progress_bar = st.progress(25)
                    with status_container.container():
                        st.info("📷 Processing uploaded image...")
                    
                    question = custom_focus if custom_focus.strip() else "Analyze this image comprehensively for compliance, quality, and child experience."
                    image_bytes = uploaded_image.read()
                    time.sleep(0.5)
                    
                    # Step 2: Initialize analyzer
                    with progress_container.container():
                        progress_bar = st.progress(50)
                    with status_container.container():
                        st.info("🧠 Initializing compliance analysis...")
                    time.sleep(0.5)
                    
                    # Step 3: Conduct analysis
                    with progress_container.container():
                        progress_bar = st.progress(75)
                    with status_container.container():
                        st.info("🔍 Conducting comprehensive visual analysis...")
                    
                    result = st.session_state.compliance_analyzer.analyze_image_compliance(question, image_bytes)
                    
                    # Step 4: Complete
                    with progress_container.container():
                        progress_bar = st.progress(100)
                    with status_container.container():
                        st.success("✅ Analysis complete!")
                    
                    time.sleep(1)
                    progress_container.empty()
                    status_container.empty()
                    
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
                    progress_container.empty()
                    status_container.empty()
                    
                    st.error("❌ Image analysis failed.")
                    
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

# Add any additional functionality or initialization here if needed
