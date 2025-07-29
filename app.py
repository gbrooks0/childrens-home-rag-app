# app.py - Fixed Version with Correct Function Order

import streamlit as st
import os
import base64
from rag_system import RAGSystem

# Safe import for compliance features
try:
    from compliance_analyzer import ComplianceAnalyzer, create_compliance_interface
    COMPLIANCE_FEATURES_AVAILABLE = True
except ImportError:
    COMPLIANCE_FEATURES_AVAILABLE = False

# DEFINE FUNCTIONS FIRST (before they're called)
def get_placeholder_text(category):
    """Get appropriate placeholder text based on selected category."""
    placeholders = {
        "🎯 Strategic Planning & Development": "e.g., We're looking to expand from 6 to 12 bed capacity over the next 2 years. What strategic considerations should we factor in, and how do we ensure we maintain quality while growing?",
        "💼 Operations & Management": "e.g., We're struggling with staff retention and high turnover is affecting continuity of care. What operational strategies can we implement to improve staff satisfaction and reduce turnover?",
        "📋 Compliance & Regulatory": "e.g., We have an Ofsted inspection coming up in 3 months. We're currently rated 'Good' but want to achieve 'Outstanding'. What specific areas should we focus on and how do we demonstrate our improvements?",
        "👤 Child-Centered Care & Support": "e.g., We have a 14-year-old resident with complex trauma who is struggling with emotional regulation and school attendance. What therapeutic approaches and support strategies would be most effective?",
        "👥 Staff Management & Development": "e.g., We want to develop a comprehensive training program for new residential care workers. What should be included, and how do we balance theoretical knowledge with practical skills development?",
        "💰 Financial Management & Planning": "e.g., We're developing our budget for next year and placement fees haven't increased in line with our rising costs. How do we maintain financial sustainability while ensuring quality of care?",
        "🏢 Facilities & Environment": "e.g., We're renovating our communal areas and want to create spaces that feel homely but also meet all safety and regulatory requirements. What design principles should guide our approach?",
        "📊 Quality Assurance & Improvement": "e.g., We want to implement a robust quality assurance system that goes beyond minimum compliance. What metrics should we track and how do we create a culture of continuous improvement?",
        "🤝 Stakeholder Relations & Communication": "e.g., We're having challenges with communication between our home, local authority commissioners, and social workers. How can we improve collaboration and ensure everyone is aligned on care goals?"
    }
    
    return placeholders.get(category, "e.g., Describe your situation, challenge, or question about running your children's home. Include relevant context such as current circumstances, specific challenges you're facing, and what outcomes you're hoping to achieve.")

# Page configuration
st.set_page_config(
    page_title="Children's Home Management System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize RAG system FIRST
@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the RAG system."""
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG System: {e}")
        return None

if 'rag_system' not in st.session_state:
    with st.spinner("Initializing comprehensive guidance system..."):
        st.session_state.rag_system = initialize_rag_system()

if st.session_state.rag_system is None:
    st.error("❌ Failed to initialize system. Please refresh the page.")
    st.stop()

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        padding: 2rem 0;
        border-bottom: 3px solid #2E8B57;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        text-align: center;
    }
    
    .expertise-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e6e9ef;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .category-header {
        color: #2E8B57;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .example-questions {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E8B57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🏠 Children's Home Management System")
st.markdown("**Comprehensive Strategic, Operational & Compliance Guidance**")
st.markdown("*Expert assistance for all aspects of running a successful children's residential care home*")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.header("🎛️ System Navigation")
    
    # Mode selection
    analysis_mode = st.radio(
        "Choose your guidance type:",
        ["💼 Strategic & Operational Guidance", "🔍 Visual Compliance Analysis"],
        help="Select the type of assistance you need"
    )
    
    # System capabilities overview
    with st.expander("🎯 System Capabilities"):
        st.markdown("""
        **Strategic Planning:**
        • Business development & growth strategies
        • Financial planning & budget management
        • Quality improvement initiatives
        
        **Operations Management:**
        • Daily operational procedures
        • Staff management & development
        • Resource allocation & efficiency
        
        **Compliance & Standards:**
        • Ofsted inspection preparation
        • Regulatory compliance guidance
        • Policy development & implementation
        
        **Child-Centered Care:**
        • Individual care planning
        • Therapeutic approaches
        • Educational support strategies
        """)
    
    # System status
    with st.expander("📊 System Status"):
        st.success("✅ Strategic Guidance: Online")
        st.success("✅ Operational Support: Active")
        st.success("✅ Compliance Database: Current")
        if COMPLIANCE_FEATURES_AVAILABLE:
            st.success("✅ Visual Analysis: Available")
        else:
            st.info("ℹ️ Visual Analysis: Upgrading")

# Main content based on selected mode
if analysis_mode == "💼 Strategic & Operational Guidance":
    
    st.header("💼 Strategic & Operational Management Guidance")
    st.markdown("Get expert advice on strategy, operations, compliance, and child-centered care delivery.")
    
    # Guidance categories with examples
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced question input
        guidance_category = st.selectbox(
            "What area do you need guidance on?",
            [
                "General Inquiry",
                "🎯 Strategic Planning & Development", 
                "💼 Operations & Management",
                "📋 Compliance & Regulatory",
                "👤 Child-Centered Care & Support",
                "👥 Staff Management & Development",
                "💰 Financial Management & Planning",
                "🏢 Facilities & Environment",
                "📊 Quality Assurance & Improvement",
                "🤝 Stakeholder Relations & Communication"
            ]
        )
        
        user_question = st.text_area(
            "Describe your situation or question:",
            placeholder=get_placeholder_text(guidance_category),
            height=140,
            help="Provide as much context as possible for the most relevant guidance"
        )
    
    with col2:
        st.markdown('<div class="example-questions">', unsafe_allow_html=True)
        st.markdown("**💡 Example Questions by Category:**")
        
        example_categories = {
            "🎯 Strategic": [
                "How do we develop a 5-year growth strategy?",
                "What are the key performance indicators for children's homes?",
                "How do we improve our Ofsted rating to Outstanding?"
            ],
            "💼 Operations": [
                "How do we optimize staff rotas for better coverage?",
                "What's the best approach to managing challenging behavior?",
                "How do we implement new therapeutic approaches?"
            ],
            "📋 Compliance": [
                "How do we prepare for an Ofsted inspection?",
                "What documentation is required for care planning?",
                "How do we ensure we meet all safeguarding requirements?"
            ],
            "👤 Child Care": [
                "How do we support a child with trauma history?",
                "What's best practice for education support?",
                "How do we handle transition planning for care leavers?"
            ]
        }
        
        for category, questions in example_categories.items():
            if st.button(f"{category}", key=f"cat_{category}"):
                st.session_state.show_examples = category
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show example questions if category selected
    if hasattr(st.session_state, 'show_examples'):
        category = st.session_state.show_examples
        st.markdown(f"**Example questions for {category}:**")
        questions = example_categories.get(category, [])
        for i, q in enumerate(questions):
            if st.button(q, key=f"example_q_{i}"):
                st.session_state.selected_question = q
                delattr(st.session_state, 'show_examples')
    
    # Handle example question selection
    if hasattr(st.session_state, 'selected_question'):
        user_question = st.session_state.selected_question
        delattr(st.session_state, 'selected_question')
    
    # Document upload for additional context
    with st.expander("📎 Upload Supporting Documents (Optional)"):
        st.markdown("Upload relevant documents to provide additional context for more tailored guidance:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Useful document types:**")
            st.markdown("""
            • Current policies and procedures
            • Ofsted inspection reports
            • Care plans and assessments
            • Staff meeting minutes
            • Financial reports or budgets
            • Training records
            """)
        
        with col2:
            uploaded_file = st.file_uploader(
                "Choose documents",
                type=['pdf', 'docx', 'txt'],
                help="Upload any relevant documents for context-specific guidance"
            )
            
            if uploaded_file:
                st.success(f"✅ Document uploaded: {uploaded_file.name}")
                try:
                    st.session_state.rag_system.process_uploaded_file(uploaded_file.read())
                    st.info("📚 Document integrated into guidance system for this session")
                except Exception as e:
                    st.error(f"Failed to process document: {e}")
    
    # Get guidance button
    if st.button("🧠 Get Expert Guidance", type="primary", use_container_width=True):
        if user_question and user_question.strip():
            with st.spinner("Analyzing your situation and accessing comprehensive knowledge base..."):
                try:
                    # Enhanced query with category context
                    enhanced_question = f"""
                    Category: {guidance_category}
                    
                    Question: {user_question}
                    
                    Please provide comprehensive guidance covering:
                    1. Strategic considerations and implications
                    2. Operational best practices and implementation steps
                    3. Compliance and regulatory requirements
                    4. Child-centered approaches and impact considerations
                    5. Risk management and mitigation strategies
                    6. Success metrics and monitoring approaches
                    """
                    
                    retriever = st.session_state.rag_system.get_current_retriever()
                    docs = retriever.get_relevant_documents(enhanced_question)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    result = st.session_state.rag_system.query(
                        user_question=enhanced_question,
                        context_text=context,
                        source_docs=docs
                    )
                    
                    # Display comprehensive results
                    st.markdown("---")
                    st.subheader("🧠 Expert Strategic & Operational Guidance")
                    
                    # Format the response for better readability
                    response_text = result["answer"]
                    
                    # Add category-specific insights badge
                    if guidance_category != "General Inquiry":
                        st.info(f"🎯 **Specialized Guidance:** {guidance_category}")
                    
                    st.write(response_text)
                    
                    # Additional insights section
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Key Considerations:**")
                        st.markdown("""
                        • Strategic alignment with long-term goals
                        • Impact on children's outcomes and experiences
                        • Regulatory compliance requirements
                        • Resource and budget implications
                        """)
                    
                    with col2:
                        st.markdown("**⚡ Next Steps:**")
                        st.markdown("""
                        • Review the guidance with your senior management team
                        • Consider pilot implementation where appropriate
                        • Schedule follow-up reviews and monitoring
                        • Document decisions and rationale
                        """)
                    
                    # Source documents
                    if result.get("source_documents"):
                        with st.expander(f"📚 Knowledge Sources ({len(result['source_documents'])} expert sources referenced)"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.markdown(f"**Expert Source {i}:**")
                                preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                                st.write(preview)
                                if i < len(result["source_documents"]):
                                    st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Error processing your request: {e}")
                    st.write("Please try rephrasing your question or contact support.")
        else:
            st.warning("⚠️ Please describe your situation or question to receive expert guidance")

elif analysis_mode == "🔍 Visual Compliance Analysis":
    # Image Analysis Interface
    if not COMPLIANCE_FEATURES_AVAILABLE:
        st.error("❌ Visual compliance analysis features are currently being upgraded.")
        st.info("💡 Use Strategic & Operational Guidance mode for comprehensive support, or contact your administrator for visual analysis access.")
    else:
        st.header("🔍 Visual Compliance Analysis")
        st.markdown("Upload facility images for comprehensive compliance assessment and operational insights.")
        
        # Initialize compliance analyzer
        if 'compliance_analyzer' not in st.session_state:
            st.session_state.compliance_analyzer = ComplianceAnalyzer(st.session_state.rag_system)
        
        uploaded_image = st.file_uploader(
            "Upload facility image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload photos for comprehensive operational and compliance analysis"
        )
        
        if uploaded_image:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(uploaded_image, caption="Facility Image for Analysis", use_column_width=True)
            
            with col2:
                st.markdown("**Comprehensive Analysis Coverage:**")
                analysis_areas = [
                    "🛡️ Health & Safety Compliance",
                    "🍎 Food Service & Nutrition", 
                    "🤝 Dignity & Respect Standards",
                    "🏠 Environment & Personalisation",
                    "📚 Educational Support Spaces",
                    "💚 Therapeutic & Wellbeing Areas",
                    "🔒 Safeguarding & Security",
                    "🏢 Facilities & Maintenance",
                    "💼 Operational Efficiency"
                ]
                
                for area in analysis_areas:
                    st.write(f"• {area}")
            
            analysis_focus = st.text_area(
                "Specific analysis focus (optional):",
                placeholder="e.g., 'Evaluate this space for operational efficiency and child experience' or leave blank for comprehensive analysis",
                height=80
            )
            
            if st.button("🔍 Conduct Comprehensive Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing image for strategic, operational, and compliance insights..."):
                    try:
                        question = analysis_focus if analysis_focus.strip() else "Conduct a comprehensive analysis of this facility image covering strategic, operational, compliance, and child experience considerations."
                        image_bytes = uploaded_image.read()
                        
                        result = st.session_state.compliance_analyzer.analyze_image_compliance(question, image_bytes)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Comprehensive Facility Analysis")
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Assessment", f"{result.overall_compliance_score}%")
                        
                        with col2:
                            st.metric("Areas Reviewed", len(result.category_distribution))
                        
                        with col3:
                            st.metric("Findings", result.total_issues)
                        
                        with col4:
                            critical_count = result.risk_distribution.get("CRITICAL", 0)
                            st.metric("Priority Items", critical_count)
                        
                        # Results sections
                        if result.priority_actions:
                            st.subheader("🎯 Priority Actions")
                            for i, action in enumerate(result.priority_actions, 1):
                                st.write(f"{i}. {action}")
                        
                        if result.positive_observations:
                            st.subheader("✅ Positive Observations")
                            for observation in result.positive_observations:
                                st.success(f"• {observation}")
                        
                        if result.recommendations:
                            st.subheader("💡 Strategic Recommendations")
                            for rec in result.recommendations:
                                st.write(f"• {rec}")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
        else:
            st.info("👆 Upload a facility image to begin comprehensive analysis")

# Footer
st.markdown("---")
st.markdown("🏠 **Children's Home Management System** - Your comprehensive partner for strategic excellence, operational efficiency, and outstanding care delivery")
if COMPLIANCE_FEATURES_AVAILABLE:
    st.markdown("✨ *Enhanced with AI-powered visual analysis and compliance assessment*")