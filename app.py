# app.py - STREAMLINED VERSION WITH PERFORMANCE OPTIMIZATION

# STEP 1: Set debug mode control FIRST
import os
import sys
import warnings
import tempfile
import hashlib
from pathlib import Path
import streamlit as st
import time
import datetime
from PIL import Image
import io
# Optional imports for document processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import psutil
    SYSTEM_MONITOR = True
except ImportError:
    SYSTEM_MONITOR = False

# Set default debug mode to FALSE unless explicitly enabled
if 'DEBUG_MODE' not in os.environ:
    os.environ['DEBUG_MODE'] = 'false'

# Helper function for conditional debug logging
# Global flags to prevent repeated debug messages
_SQLITE_LOGGED = False
_ENV_SETUP_LOGGED = False

def debug_log(message, once_only=False, key=None):
    """Only print debug messages if debug mode is enabled, with optional one-time logging"""
    if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
        if once_only and key:
            # Use session state to track what we've already logged
            if 'debug_logged' not in st.session_state:
                st.session_state.debug_logged = set()
            
            if key not in st.session_state.debug_logged:
                print(f"DEBUG: {message}")
                st.session_state.debug_logged.add(key)
        elif not once_only:
            print(f"DEBUG: {message}")

# STEP 2: Suppress ALL Google/gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'poll'

# Suppress Python warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*absl.*")
warnings.filterwarnings("ignore", message=".*gRPC.*")
warnings.filterwarnings("ignore", module="langchain")

# STEP 3: SQLite3 Fix with SILENT logging

def setup_sqlite_with_clean_logging():
    """Setup SQLite with one-time logging only"""
    global _SQLITE_LOGGED
    
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        if not _SQLITE_LOGGED:
            debug_log("Using pysqlite3", once_only=True, key="sqlite_pysqlite")
            _SQLITE_LOGGED = True
    except ImportError:
        if not _SQLITE_LOGGED:
            debug_log("Using system sqlite3", once_only=True, key="sqlite_system")
            _SQLITE_LOGGED = True

# STEP 4: Regular imports
import streamlit as st
import time
import datetime

# Import RAG system
from rag_system import RAGSystem

# Safe import for compliance features
try:
    from compliance_analyzer import ComplianceAnalyzer, create_compliance_interface
    COMPLIANCE_FEATURES_AVAILABLE = True
except ImportError:
    COMPLIANCE_FEATURES_AVAILABLE = False
    debug_log("Compliance features not available")

# Page configuration
st.set_page_config(
    page_title="Children's Home Management System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== AUTHENTICATION SYSTEM =====

def get_tester_credentials():
    """Get tester credentials with fallback system"""
    try:
        # Try Streamlit Cloud secrets first
        testers = st.secrets.get("testers", {})
        if testers:
            return testers
    except Exception as e:
        print(f"DEBUG: Secrets access failed: {e}")
    
    # Fallback for local development
    return {
        "DEMO001": {
            "password": "DemoAccess2024!",
            "name": "Demo Tester",
            "email": "demo@example.com",
            "expires": "2025-12-31"
        },
        "TEST001": {
            "password": "TestAccess456!",
            "name": "Beta Tester 1", 
            "email": "tester1@example.com",
            "expires": "2025-12-31"
        }
    }

def check_session_valid():
    """Check if current session is still valid"""
    if not st.session_state.get('authenticated_tester', False):
        return False
    
    # Check session timeout (2 hours)
    session_start = st.session_state.get('session_start', 0)
    current_time = time.time()
    session_duration = current_time - session_start
    
    if session_duration > 7200:  # 2 hours
        st.session_state['authenticated_tester'] = False
        return False
    
    return True

def log_tester_activity(tester_id, action, details=""):
    """Log tester activity with completely clean output"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "tester_id": tester_id or "anonymous",
            "action": action,
            "details": details
        }
        
        # Store in session state
        if 'activity_log' not in st.session_state:
            st.session_state['activity_log'] = []
        
        st.session_state['activity_log'].append(log_entry)
        
        # Only log the most important user actions - nothing else
        critical_actions = [
            'successful_login', 'failed_login', 'logout',
            'system_error', 'feedback_submitted'
        ]
        
        # Only show critical actions or when in debug mode
        if action in critical_actions:
            print(f"TESTER: {timestamp} - {tester_id or 'anonymous'} - {action}")
            if action == 'system_error' and details:
                print(f"  └─ {details}")
        elif os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            print(f"DEBUG: {timestamp} - {tester_id or 'anonymous'} - {action}")
            if details:
                print(f"  └─ {details}")
        
    except Exception as e:
        # Silent fallback - don't spam with log errors
        pass

def validate_and_authenticate(tester_id, access_code):
    """Validate credentials and set up authenticated session"""
    try:
        if not tester_id or not access_code:
            return False
        
        # Get valid credentials
        valid_testers = get_tester_credentials()
        
        # Check if tester exists and password matches
        if tester_id not in valid_testers:
            return False
        
        tester_info = valid_testers[tester_id]
        if tester_info.get("password") != access_code:
            return False
        
        # Check expiration
        expiry_date = tester_info.get("expires", "2099-12-31")
        try:
            expiry = datetime.datetime.strptime(expiry_date, "%Y-%m-%d").date()
            if datetime.date.today() > expiry:
                return False
        except:
            pass  # Allow access if date parsing fails
        
        # Set up authenticated session
        st.session_state['authenticated_tester'] = True
        st.session_state['tester_id'] = tester_id
        st.session_state['tester_info'] = tester_info
        st.session_state['session_start'] = time.time()
        
        # Log successful authentication
        log_tester_activity(tester_id, "successful_login", f"User: {tester_info.get('name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Authentication failed: {e}")
        return False

def show_login_interface():
    """Streamlined login interface"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 600px;
        margin: 2rem auto;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .login-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="login-header">
        <h1>🏠 Children's Home Management System</h1>
        <h3>Beta Testing Portal</h3>
        <p>Secure access for authorized testers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form
    st.markdown("### 🔐 Tester Login")
    
    tester_id = st.text_input(
        "Tester ID:",
        placeholder="e.g., DEMO001",
        help="Your unique tester identifier"
    )
    
    access_code = st.text_input(
        "Access Code:",
        type="password",
        placeholder="Enter your access code",
        help="Secure password provided with your Tester ID"
    )
    
    # Login button
    if st.button("🚀 Start Testing Session", type="primary", use_container_width=True):
        if validate_and_authenticate(tester_id, access_code):
            st.success("✅ Authentication successful!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please verify your Tester ID and Access Code.")
            log_tester_activity(tester_id or "unknown", "failed_login", "Invalid credentials")
    
    # Info section
    with st.expander("ℹ️ Testing Information"):
        st.markdown("""
        **Session Details:**
        - Duration: 2 hours per session
        - Auto-logout after inactivity
        - Activity logging for improvement
        
        **Need Access?**
        Contact the administrator to receive your testing credentials.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_session_info():
    """Show session information in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 Session Info")
        
        tester_info = st.session_state.get('tester_info', {})
        st.write(f"**Tester:** {tester_info.get('name', 'Unknown')}")
        st.write(f"**ID:** {st.session_state.get('tester_id', 'N/A')}")
        
        # Session timer
        session_start = st.session_state.get('session_start', time.time())
        elapsed = time.time() - session_start
        remaining = max(0, 7200 - elapsed)
        
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        st.write(f"**Time remaining:** {hours}h {minutes}m")
        
        # Logout button
        if st.button("🚪 End Session", type="secondary"):
            log_tester_activity(st.session_state.get('tester_id', 'unknown'), "logout")
            
            # Clear session
            for key in ['authenticated_tester', 'tester_id', 'tester_info', 'session_start']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()

def require_authentication():
    """Main authentication function"""
    if check_session_valid():
        show_session_info()
        return True
    
    # Clear invalid session data
    st.session_state['authenticated_tester'] = False
    show_login_interface()
    return False

# ===== ENVIRONMENT SETUP =====

def setup_environment_variables():
    """Set up environment variables with one-time logging"""
    global _ENV_SETUP_LOGGED
    
    try:
        # Direct access to simple secrets format (Streamlit Cloud compatible)
        openai_key = st.secrets.get("OPENAI_API_KEY")
        google_key = st.secrets.get("GOOGLE_API_KEY")
        
        # Set OpenAI key
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            if not _ENV_SETUP_LOGGED:
                debug_log("OpenAI API key configured", once_only=True, key="openai_setup")
        
        # Set Google key with additional gRPC suppression
        if google_key:
            os.environ['GOOGLE_API_KEY'] = google_key
            # Additional Google/gRPC environment variables
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
            os.environ['GRPC_POLL_STRATEGY'] = 'poll'
            os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
            if not _ENV_SETUP_LOGGED:
                debug_log("Google API key configured", once_only=True, key="google_setup")
        
        # Mark as logged
        _ENV_SETUP_LOGGED = True
        
        # Verify at least one key is available
        if not (openai_key or google_key):
            debug_log("WARNING: No API keys found in secrets")
            return False
            
        return True
        
    except Exception as e:
        debug_log(f"Environment setup failed: {e}")
        return False

# ===== RAG SYSTEM INITIALIZATION =====

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with one-time logging"""
    debug_log("Initializing RAG system...", once_only=True, key="rag_init_start")
    
    # Fix asyncio issues silently
    try:
        import asyncio
        
        # Additional threading environment variables for gRPC
        os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
        os.environ['GRPC_POLL_STRATEGY'] = 'poll'
        
        # Standard asyncio fixes
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        # Windows-specific policy
        if sys.platform.startswith('win'):
            if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        debug_log("Asyncio configured", once_only=True, key="asyncio_config")
        
    except Exception as e:
        debug_log(f"Asyncio configuration failed: {e}")
    
    try:
        # Try to initialize RAG system
        rag_system = RAGSystem()
        debug_log("RAG system initialized successfully", once_only=True, key="rag_init_success")
        return rag_system
        
    except Exception as e:
        debug_log(f"RAG system initialization failed: {e}")
        
        # Provide helpful error information only in debug mode
        error_msg = str(e).lower()
        if "api" in error_msg or "key" in error_msg:
            debug_log("HINT: Check API key configuration")
        elif "import" in error_msg or "module" in error_msg:
            debug_log("HINT: Check dependencies installation")
        elif "connection" in error_msg or "network" in error_msg:
            debug_log("HINT: Check internet connection")
        
        return None

# ===== PERFORMANCE MODE SELECTOR =====

def add_performance_mode_selector():
    """Add performance mode selector to sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚡ Performance Settings")
        
        mode_descriptions = {
            "fast": "🚀 Fast (2-4 docs, ~5s response)",
            "balanced": "⚖️ Balanced (3-7 docs, ~8s response)", 
            "comprehensive": "🔍 Comprehensive (5-12 docs, ~15s response)"
        }
        
        current_mode = st.session_state.get('performance_mode', 'balanced')
        
        selected_mode = st.selectbox(
            "Response Mode:",
            ["fast", "balanced", "comprehensive"],
            index=["fast", "balanced", "comprehensive"].index(current_mode),
            format_func=lambda x: mode_descriptions[x],
            help="Choose speed vs comprehensiveness trade-off"
        )
        
        st.session_state['performance_mode'] = selected_mode
        
        # Show mode description
        if selected_mode == "fast":
            st.caption("⚡ Optimized for speed. Best for simple, factual questions.")
        elif selected_mode == "balanced":
            st.caption("⚖️ Good balance of speed and completeness. Recommended for most queries.")
        else:
            st.caption("🔍 Maximum thoroughness. Best for complex strategic questions.")
        
        return selected_mode


# ===== QUESTION HANDLING FUNCTIONS =====

def show_question_form():
    """Display the question input form (always visible when no answer)"""
    st.subheader("💬 Ask Your Question")
    
    # Initialize question state
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Handle quick question selection
    if hasattr(st.session_state, 'quick_question'):
        st.session_state.current_question = st.session_state.quick_question
        delattr(st.session_state, 'quick_question')
    
    # Question input
    user_question = st.text_area(
        "Describe your situation or question:",
        value=st.session_state.current_question,
        placeholder="Start typing your question... (e.g., 'How do we prepare for an Ofsted inspection?')",
        height=120,
        key="question_input"
    )
    
    # Submit button
    if st.button("🧠 Get Expert Guidance", type="primary", use_container_width=True):
        if user_question and user_question.strip():
            # Store the question that was asked
            st.session_state['last_asked_question'] = user_question
            process_question(user_question)
        else:
            st.warning("⚠️ Please enter a question to receive guidance")
    
    # Quick actions
    show_quick_actions()
    
    # Contextual tip
    tip = get_contextual_tip(user_question)
    st.markdown(f'<div class="contextual-tip">{tip}</div>', unsafe_allow_html=True)

def show_quick_actions():
    """Display quick action buttons"""
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    quick_actions = {
        "🚨 Inspection Prep": "We have an upcoming Ofsted inspection. What should we focus on to ensure we're fully prepared?",
        "💰 Budget Planning": "Help us develop an effective budget strategy for the upcoming financial year.",
        "👥 Staff Issues": "We're experiencing staff retention challenges. What strategies can we implement?",
        "🏠 New Admission": "We're admitting a new child. What processes should we prioritize for a successful transition?",
        "📋 Policy Review": "We need to review and update our policies. What current best practices should we include?",
        "🎯 Quality Improvement": "We want to enhance our care quality and move towards outstanding. What should we focus on?"
    }
    
    cols = st.columns(3)
    for i, (action, question) in enumerate(quick_actions.items()):
        with cols[i % 3]:
            if st.button(action, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_question = question
                log_tester_activity(
                    st.session_state.get('tester_id', 'unknown'),
                    "quick_action_used",
                    f"Action: {action}"
                )
                st.rerun()

def get_contextual_tip(current_input=""):
    """Enhanced contextual tips including performance tips"""
    if not current_input:
        return "💡 **Tip:** Use the ⚡ Performance Settings in the sidebar - Fast mode for quick answers, Comprehensive for detailed analysis!"
    
    input_lower = current_input.lower()
    
    # Check for potentially slow queries
    if len(current_input) > 200:
        return "⚡ **Performance Tip:** Long questions work best in Comprehensive mode. For faster responses, try Fast mode."
    
    if "ofsted" in input_lower or "inspection" in input_lower:
        return "💡 **Tip:** Inspection questions work great in Balanced mode for good speed and thorough coverage."
    elif any(word in input_lower for word in ["comprehensive", "detailed", "analysis", "strategy"]):
        return "🔍 **Performance Tip:** Complex questions get better results in Comprehensive mode."
    elif any(word in input_lower for word in ["what is", "define", "list"]):
        return "🚀 **Performance Tip:** Simple questions work great in Fast mode for quicker responses!"
    else:
        tips = [
            "💡 **Tip:** Adjust ⚡ Performance Settings based on your time/detail needs",
            "⚡ **Speed Tip:** Fast mode gives good answers in under 5 seconds",
            "🔍 **Quality Tip:** Use Comprehensive mode for complex questions"
        ]
        return tips[len(current_input) % len(tips)]

def handle_question_tab():
    """Handle the question asking tab - FIXED VERSION that always shows question form"""
    
    # If there's a current result, show it first
    if 'current_result' in st.session_state and st.session_state.current_result:
        show_current_result_inline()
        
        # Add spacing and section for new question
        st.markdown("---")
        st.markdown("### 🆕 Ask Another Question")
    
    # ALWAYS show the question form (this was missing!)
    show_question_form()

def show_current_result_inline():
    """Display the current result with clean interface"""
    
    # Show the question that was asked
    if 'last_asked_question' in st.session_state:
        st.markdown("### 🤔 Your Question:")
        st.info(st.session_state['last_asked_question'])
    
    st.subheader("🧠 Expert Guidance")
    
    # Show model attribution if available
    if 'model_used' in st.session_state and st.session_state.model_used:
        st.caption(f'*Response by {st.session_state.model_used}*')
    
    # The main answer - clean and prominent
    st.write(st.session_state.current_result["answer"])
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Ask New Question", type="primary"):
            clear_current_result()
    
    with col2:
        if st.button("✏️ Edit & Resubmit"):
            # Keep result but show question form for editing
            st.session_state.show_edit_form = True
    
    with col3:
        if st.button("📋 Key Actions"):
            st.session_state.show_actions = not st.session_state.get('show_actions', False)
    
    with col4:
        if st.button("📚 Sources"):
            st.session_state.show_sources = not st.session_state.get('show_sources', False)
    
    # Show edit form if requested
    if st.session_state.get('show_edit_form', False):
        st.markdown("---")
        show_compact_question_form()
    
    # Progressive disclosure
    if st.session_state.get('show_actions', False):
        st.markdown("---")
        st.markdown("**📋 Recommended Actions:**")
        st.markdown("""
        • **Review** these recommendations with your management team
        • **Plan implementation** with realistic timelines  
        • **Monitor progress** and evaluate outcomes regularly
        """)
    
    # CLEAN SOURCES SECTION (Option 4 - Expandable Clean List)
    if st.session_state.get('show_sources', False) and st.session_state.current_result.get("source_documents"):
        st.markdown("---")
        doc_count = len(st.session_state.current_result["source_documents"])
        st.markdown(f"**📚 Reference Sources** ({doc_count} documents)")
        
        # Improved document identification
        source_types = []
        for i, doc in enumerate(st.session_state.current_result["source_documents"], 1):
            # Look at more content and check multiple indicators
            content = doc.page_content[:500].lower()  # Increased from 150 to 500
            
            # More comprehensive document type detection
            if any(keyword in content for keyword in ["children's homes regulations", "regulation", "the children's homes"]):
                doc_type = "Children's Homes Regulations"
            elif any(keyword in content for keyword in ["quality standards", "standard", "quality framework"]):
                doc_type = "Quality Standards Framework"
            elif any(keyword in content for keyword in ["ofsted", "inspection", "inspector", "inspection framework"]):
                doc_type = "Ofsted Inspection Guidelines"
            elif any(keyword in content for keyword in ["guide to", "guidance", "guide", "april 2015"]):
                doc_type = "Official Guidance Document"
            elif any(keyword in content for keyword in ["childrenscommissioner", "commissioner", "cco-"]):
                doc_type = "Children's Commissioner Report"
            elif any(keyword in content for keyword in ["safeguarding", "protection", "harm"]):
                doc_type = "Safeguarding Guidelines"
            elif any(keyword in content for keyword in ["education", "skills", "learning"]):
                doc_type = "Education Standards"
            elif any(keyword in content for keyword in ["wellbeing", "well-being", "health"]):
                doc_type = "Health & Wellbeing Standards"
            elif "http" in content or "www" in content or ".pdf" in content:
                doc_type = "Online Resource/Report"
            else:
                # Try to extract a meaningful title from the beginning
                first_line = doc.page_content.split('\n')[0].strip()
                if len(first_line) > 10 and len(first_line) < 100:
                    doc_type = first_line
                else:
                    doc_type = "Regulatory Documentation"
            
            source_types.append(f"**{i}.** {doc_type}")
        
        # Display as clean numbered list
        st.markdown("\n".join(source_types))
        
        # Optional: Add expandable details
        with st.expander("📋 View Source Details", expanded=False):
            for i, doc in enumerate(st.session_state.current_result["source_documents"], 1):
                st.markdown(f"**Source {i}:**")
                
                # Show first line as title if it looks like a title
                first_line = doc.page_content.split('\n')[0].strip()
                if len(first_line) > 10 and len(first_line) < 200:
                    st.markdown(f"*{first_line}*")
                
                # Show preview
                preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                st.text(preview)
                
                if i < len(st.session_state.current_result["source_documents"]):
                    st.markdown("---")

def show_compact_question_form():
    """Show a compact question form for editing/new questions"""
    st.markdown("### ✏️ Edit Your Question")
    
    # Pre-fill with the last asked question
    current_question = st.session_state.get('last_asked_question', '')
    
    user_question = st.text_area(
        "Modify your question or ask a new one:",
        value=current_question,
        height=100,
        key="edit_question_input"
    )
    
    # Buttons in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧠 Get New Answer", type="primary", use_container_width=True):
            if user_question and user_question.strip():
                st.session_state['last_asked_question'] = user_question
                st.session_state.show_edit_form = False  # Hide the edit form
                process_question(user_question)
            else:
                st.warning("⚠️ Please enter a question")
    
    with col2:
        if st.button("❌ Cancel Edit", use_container_width=True):
            st.session_state.show_edit_form = False
    
    with col3:
        if st.button("🔄 Start Fresh", use_container_width=True):
            clear_current_result()

def clear_current_result():
    """Clear current result and return to question form"""
    for key in ['current_result', 'model_used', 'show_actions', 'show_sources', 'last_asked_question', 'show_edit_form']:
        if key in st.session_state:
            del st.session_state[key]
    # Clear the current_question to start fresh
    st.session_state.current_question = ""

# ===== END QUESTION HANDLING FUNCTIONS =====

# ===== QUESTION PROCESSING =====

def process_question(user_question):
    """Process the user's question with clean, minimal progress indicators"""
    # Get performance mode from user selection
    performance_mode = st.session_state.get('performance_mode', 'balanced')
    
    start_time = time.time()
    
    log_tester_activity(
        st.session_state.get('tester_id', 'unknown'),
        "question_submitted",
        f"Mode: {performance_mode}, Question: {user_question[:50]}..."
    )
    
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # Clean progress indicators - no mode references
        with progress_container.container():
            progress_bar = st.progress(20)
        with status_container.container():
            st.info("🔍 Analyzing your question...")
        time.sleep(0.3)
        
        with progress_container.container():
            progress_bar = st.progress(60)
        with status_container.container():
            st.info("📚 Gathering relevant information...")
        
        # Use adaptive RAG system if available
        if hasattr(st.session_state.rag_system, 'query_with_performance_mode'):
            result = st.session_state.rag_system.query_with_performance_mode(
                user_question, 
                performance_mode
            )
        else:
            # Fallback to standard processing
            retriever = st.session_state.rag_system.get_current_retriever()
            docs = retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in docs])
            enhanced_question = enhance_question_based_on_intent(user_question)
            
            result = st.session_state.rag_system.query(
                user_question=enhanced_question,
                context_text=context,
                source_docs=docs
            )
        
        with progress_container.container():
            progress_bar = st.progress(100)
        with status_container.container():
            st.success("✅ Response ready!")
        
        time.sleep(1)
        progress_container.empty()
        status_container.empty()
        
        if result and result.get("answer"):
            st.session_state.current_result = result
            st.session_state.current_question = user_question
            
            # Store model info if available
            if result.get("metadata") and result["metadata"].get("llm_used"):
                st.session_state.model_used = result["metadata"]["llm_used"]
            
            total_time = time.time() - start_time
            log_tester_activity(
                st.session_state.get('tester_id', 'unknown'),
                "successful_response",
                f"Mode: {performance_mode}, Time: {total_time:.1f}s"
            )

            st.rerun()
            
        else:
            st.error("❌ Sorry, I couldn't generate a response.")
            log_tester_activity(
                st.session_state.get('tester_id', 'unknown'),
                "failed_response",
                f"Mode: {performance_mode}, No response generated"
            )
            
    except Exception as e:
        total_time = time.time() - start_time
        progress_container.empty()
        status_container.empty()
        
        st.error("❌ An error occurred while processing your question.")
        log_tester_activity(
            st.session_state.get('tester_id', 'unknown'),
            "system_error",
            f"Mode: {performance_mode}, Error: {str(e)[:100]}"
        )
        
        with st.expander("🔧 Error Details"):
            st.code(f"Error: {str(e)}")
            st.markdown("**💡 Try refreshing the page or rephrasing your question.**")

def enhance_question_based_on_intent(user_question):
    """Enhance question based on detected intent"""
    question_lower = user_question.lower()
    
    # Creation/drafting requests
    creation_indicators = ["draft", "write", "create", "develop a", "make a", "compose"]
    
    # Factual/informational questions
    factual_indicators = ["what are", "what is", "list", "define", "explain", "describe"]
    
    # Strategic guidance requests
    strategic_indicators = ["how do we", "how can we", "what should we", "strategy for"]
    
    is_creation = any(indicator in question_lower for indicator in creation_indicators)
    is_factual = any(indicator in question_lower for indicator in factual_indicators)
    is_strategic = any(indicator in question_lower for indicator in strategic_indicators)
    
    if is_creation:
        return f"""
        Create the specific deliverable requested by the user. Provide actual content they can use directly.
        
        Request: {user_question}
        
        Format as a ready-to-use deliverable with professional structure and complete content.
        """
    elif is_factual and not is_strategic:
        return f"""
        Question: {user_question}
        
        Provide comprehensive factual information including:
        1. Complete standards/requirements
        2. Specific regulations and numbers
        3. Clear explanations and details
        4. Key requirements for practitioners
        
        Focus on accuracy and completeness.
        """
    else:
        return f"""
        As a senior children's home consultant, provide strategic guidance for:
        
        Question: {user_question}
        
        Include:
        1. Strategic considerations
        2. Implementation steps with timelines
        3. Risk management strategies
        4. Child-centered impact assessment
        5. Resource requirements
        6. Success metrics
        7. Actionable recommendations
        """

# ============================================================================
# ENHANCED DOCUMENT ANALYSIS FUNCTIONS
# ============================================================================

def handle_document_analysis_tab():
    """Enhanced document analysis tab - FIXED to hide interface after analysis"""
    
    # Check if we have document analysis results
    if 'document_analysis_result' in st.session_state and st.session_state.document_analysis_result:
        show_document_analysis_result()
        
        # Add spacing and section for new analysis
        st.markdown("---")
        st.markdown("### 🆕 Analyze New Documents")
        
        # Show button to clear results and start new analysis
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Analyze Different Documents", type="primary", use_container_width=True, key="doc_analyze_different"):
                clear_document_analysis_result()
                st.rerun()
        
        with col2:
            if st.button("🔄 Start Fresh", use_container_width=True, key="doc_start_fresh"):
                clear_document_analysis_result()
                st.rerun()
        
        # REMOVED: Don't show the upload interface in expander when results exist
        # This was causing the interface to still be visible
        
        return  # Exit early - don't show upload interface
    
    # Show normal interface if no results (moved to top)
    show_document_upload_interface()

def show_document_upload_interface():
    """Show the document upload and analysis interface - FIXED KEYS"""
    st.subheader("📄 Document Analysis & Processing")
    st.markdown("Upload documents for comprehensive analysis, extraction, and Q&A.")
    
    # 1. CUSTOM ANALYSIS FOCUS - with unique key prefix
    custom_focus = st.text_area(
        "🎯 Custom Analysis Focus (optional):",
        placeholder="e.g., 'Focus on safeguarding policies and compliance requirements'",
        height=80,
        help="Specify what aspects you want the analysis to focus on",
        key="doc_tab_custom_focus"  # Added doc_tab_ prefix
    )
    
    # 2. Analysis type selection - with unique key prefix
    analysis_type = st.selectbox(
        "📋 Analysis Type:",
        [
            "📊 Comprehensive Analysis",
            "✅ Compliance Check", 
            "📝 Content Summary",
            "🔍 Key Information Extraction",
            "❓ Document Q&A",
            "📈 Policy Gap Analysis"
        ],
        help="Choose the type of analysis you need",
        key="doc_tab_analysis_type"  # Added doc_tab_ prefix
    )
    
    # 3. File upload - with unique key prefix
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True,
        help="Supports PDF, Word, Text, and Markdown files",
        key="doc_tab_uploader"  # Added doc_tab_ prefix
    )
    
    # Show the analyze button immediately after the file uploader
    if uploaded_files:
        # Place a horizontal line for visual separation
        st.markdown("---")
        
        # THE ANALYZE BUTTON - with unique key prefix
        if st.button("🔍 Analyze Documents", type="primary", use_container_width=True, key="doc_tab_analyze_btn"):
            analyze_documents(
                uploaded_files, 
                analysis_type, 
                custom_focus,
                {
                    'metadata': True,
                    'tables': True,
                    'contradictions': False,
                    'compare': len(uploaded_files) > 1
                }
            )
        
        st.markdown("---")
        
        # Show file details below the button
        st.markdown("### 📁 Uploaded Files")
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{file.name}** ({file.size/1024:.1f} KB)")
            with col2:
                st.write(file.type)
            with col3:
                st.write(f"File {i+1}")

def show_document_analysis_result():
    """Display document analysis results with clean interface"""
    
    result = st.session_state.document_analysis_result
    
    # Show what was analyzed
    st.markdown("### 📄 Document Analysis Complete")
    
    if 'analysis_metadata' in st.session_state:
        metadata = st.session_state.analysis_metadata
        st.info(f"✅ Analyzed {metadata.get('file_count', 1)} document(s) using {metadata.get('analysis_type', 'Analysis')}")
    
    # Main analysis result
    st.subheader("📋 Analysis Report")
    st.write(result['answer'])
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Key Findings"):
            st.session_state.show_doc_findings = not st.session_state.get('show_doc_findings', False)
    
    with col2:
        if st.button("📄 Generate Report"):
            generate_document_report(result)
    
    with col3:
        if st.button("📚 Source Details"):
            st.session_state.show_doc_sources = not st.session_state.get('show_doc_sources', False)
    
    # Progressive disclosure
    if st.session_state.get('show_doc_findings', False):
        st.markdown("---")
        st.markdown("**📋 Key Findings Summary:**")
        st.markdown("""
        • **Review** the analysis findings with your team
        • **Implement** recommended improvements
        • **Monitor** compliance gaps identified
        • **Follow up** on action items specified
        """)
    
    if st.session_state.get('show_doc_sources', False) and 'analysis_metadata' in st.session_state:
        st.markdown("---")
        metadata = st.session_state.analysis_metadata
        if metadata.get('processed_docs'):
            st.markdown("**📁 Analyzed Documents:**")
            for i, doc in enumerate(metadata['processed_docs'], 1):
                st.write(f"**{i}.** {doc['filename']} ({doc['word_count']:,} words)")


def analyze_documents(files, analysis_type, custom_focus, options):
    """Process and analyze uploaded documents"""
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # Step 1: Process files
        with progress_container.container():
            progress_bar = st.progress(20)
        with status_container.container():
            st.info("📄 Processing uploaded documents...")
        
        processed_docs = []
        for file in files:
            doc_data = process_single_document(file)
            if doc_data:
                processed_docs.append(doc_data)
        
        # Step 2: Combine and analyze
        with progress_container.container():
            progress_bar = st.progress(60)
        with status_container.container():
            st.info("🧠 Performing comprehensive analysis...")
        
        # Create analysis prompt based on type
        analysis_prompt = create_document_analysis_prompt(
            analysis_type, custom_focus, processed_docs, options
        )
        
        # Get response from RAG system
        result = st.session_state.rag_system.query_with_performance_mode(
            analysis_prompt, 
            st.session_state.get('performance_mode', 'balanced')
        )
        
        # Step 3: Complete
        with progress_container.container():
            progress_bar = st.progress(100)
        with status_container.container():
            st.success("✅ Analysis complete!")
        
        time.sleep(1)
        progress_container.empty()
        status_container.empty()
        
        st.session_state.document_analysis_result = result
        st.session_state.analysis_metadata = {
            'analysis_type': analysis_type,
            'file_count': len(files),
            'processed_docs': processed_docs,
            'custom_focus': custom_focus,
            'options': options
        }
        
        # Log activity
        log_tester_activity(
            st.session_state.get('tester_id', 'unknown'),
            "document_analysis_completed",
            f"Type: {analysis_type}, Files: {len(files)}"
        )
        
        # Refresh to show results
        st.rerun()
        
    except Exception as e:
        progress_container.empty()
        status_container.empty()
        st.error("❌ Document analysis failed.")
        
        with st.expander("🔧 Error Details"):
            st.code(f"Error: {str(e)}")

def clear_document_analysis_result():
    """Clear document analysis results - UPDATED for new keys"""
    for key in ['document_analysis_result', 'analysis_metadata', 'show_doc_findings', 'show_doc_sources']:
        if key in st.session_state:
            del st.session_state[key]


def process_single_document(file):
    """Process a single document and extract content"""
    try:
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            # Use your existing PDF processing
            content = extract_pdf_content(file)
        elif file_extension == 'docx':
            content = extract_docx_content(file)
        elif file_extension in ['txt', 'md']:
            content = file.read().decode('utf-8')
        else:
            return None
        
        return {
            'filename': file.name,
            'content': content,
            'size': file.size,
            'type': file_extension,
            'word_count': len(content.split()),
            'char_count': len(content)
        }
        
    except Exception as e:
        st.warning(f"Failed to process {file.name}: {str(e)}")
        return None

def extract_pdf_content(file):
    """Extract content from PDF file"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
        return content
    except ImportError:
        # Fallback to your existing method
        return "PDF processing requires PyPDF2. Using alternative method..."

def extract_docx_content(file):
    """Extract content from Word document"""
    try:
        import docx
        doc = docx.Document(file)
        content = ""
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
        return content
    except ImportError:
        return "Word document processing requires python-docx library."

def create_document_analysis_prompt(analysis_type, custom_focus, docs, options):
    """Create specialized prompt based on analysis type"""
    
    # Combine all document content
    combined_content = "\n\n".join([
        f"DOCUMENT: {doc['filename']}\n{doc['content']}" 
        for doc in docs
    ])
    
    base_context = f"Analyze the following documents:\n\n{combined_content}\n\n"
    
    if "Comprehensive Analysis" in analysis_type:
        prompt = f"""{base_context}
Provide a comprehensive analysis including:
1. **Executive Summary** - Key findings and overview
2. **Content Analysis** - Main themes, topics, and structure
3. **Compliance Assessment** - Regulatory alignment and gaps
4. **Risk Analysis** - Potential issues and concerns
5. **Recommendations** - Actionable next steps
6. **Key Metrics** - Important data points and statistics
        """
    
    elif "Compliance Check" in analysis_type:
        prompt = f"""{base_context}
Perform a detailed compliance analysis:
1. **Regulatory Standards** - Which standards are addressed
2. **Compliance Gaps** - Missing or insufficient coverage
3. **Risk Areas** - Potential compliance vulnerabilities
4. **Required Actions** - Steps to achieve full compliance
5. **Priority Level** - Urgency of each finding
        """
    
    elif "Content Summary" in analysis_type:
        prompt = f"""{base_context}
Create a structured content summary:
1. **Key Points** - Main messages and findings
2. **Action Items** - Required tasks and responsibilities
3. **Deadlines** - Important dates and timelines
4. **Stakeholders** - People and roles mentioned
5. **Resources** - Required materials and support
        """
    
    elif "Key Information Extraction" in analysis_type:
        prompt = f"""{base_context}
Extract key information in structured format:
1. **Important Dates** - All dates and deadlines
2. **Key Personnel** - Names, roles, and responsibilities
3. **Financial Information** - Costs, budgets, and financial data
4. **Requirements** - Mandatory actions and standards
5. **Contact Information** - Phone numbers, emails, addresses
        """
    
    elif "Document Q&A" in analysis_type:
        prompt = f"""{base_context}
Based on these documents, provide answers to common questions:
1. **What are the main objectives?**
2. **Who is responsible for what?**
3. **What are the key deadlines?**
4. **What resources are needed?**
5. **What are the success criteria?**
6. **What are the potential risks?**
        """
    
    else:  # Policy Gap Analysis
        prompt = f"""{base_context}
Analyze for policy gaps and improvements:
1. **Current Coverage** - What policies exist
2. **Missing Policies** - What's not covered
3. **Outdated Content** - What needs updating
4. **Best Practice Gaps** - Industry standard comparisons
5. **Implementation Issues** - Practical challenges
        """
    
    # Add custom focus if provided
    if custom_focus:
        prompt += f"\n\nSpecial Focus: {custom_focus}"
    
    # Add options-based instructions
    if options.get('metadata'):
        prompt += "\n\nInclude document metadata analysis (file info, structure, etc.)"
    if options.get('tables'):
        prompt += "\n\nExtract and analyze any tables or structured data"
    if options.get('contradictions'):
        prompt += "\n\nIdentify any contradictions or conflicts between documents"
    if options.get('compare') and len(docs) > 1:
        prompt += "\n\nCompare documents and highlight differences/similarities"
    
    return prompt

def display_document_analysis_results(result, analysis_type, docs, options):
    """Display comprehensive analysis results"""
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", len(docs))
    
    with col2:
        total_words = sum(doc['word_count'] for doc in docs)
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        total_size = sum(doc['size'] for doc in docs)
        st.metric("Total Size", f"{total_size/1024:.1f} KB")
    
    with col4:
        response_time = result.get('metadata', {}).get('response_time_ms', 0)
        st.metric("Analysis Time", f"{response_time/1000:.1f}s")
    
    # Main analysis result
    st.markdown("### 📋 Analysis Report")
    st.write(result['answer'])
    
    # Document details
    with st.expander("📁 Document Details", expanded=False):
        for doc in docs:
            st.markdown(f"**{doc['filename']}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Type: {doc['type'].upper()}")
            with col2:
                st.write(f"Words: {doc['word_count']:,}")
            with col3:
                st.write(f"Size: {doc['size']/1024:.1f} KB")
            
            # Show content preview
            if st.button(f"Preview {doc['filename']}", key=f"preview_{doc['filename']}"):
                st.text_area(
                    f"Content Preview - {doc['filename']}",
                    doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content'],
                    height=200,
                    key=f"content_{doc['filename']}"
                )
            st.markdown("---")

def display_system_health():
    """Display system health and performance information"""
    st.subheader("⚙️ System Health & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 AI Models Status")
        
        # Check RAG system health
        if st.session_state.get('rag_system'):
            try:
                health = st.session_state.rag_system.get_system_health()
                models = health.get('models', {})
                
                for model, status in models.items():
                    status_icon = "✅" if status else "❌"
                    st.write(f"{status_icon} {model}")
                    
            except Exception as e:
                st.error(f"❌ Health check failed: {e}")
        else:
            st.error("❌ RAG System not available")
        
        # API Keys status
        st.markdown("### 🔑 API Configuration")
        openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        google_key = bool(os.environ.get("GOOGLE_API_KEY"))
        
        st.write(f"{'✅' if openai_key else '❌'} OpenAI API")
        st.write(f"{'✅' if google_key else '❌'} Google API")
    
    with col2:
        st.markdown("### 📊 Performance Metrics")
        
        # Get performance stats
        if hasattr(st.session_state.rag_system, 'get_performance_stats'):
            try:
                perf_stats = st.session_state.rag_system.get_performance_stats()
                if isinstance(perf_stats, dict) and perf_stats.get('total_queries', 0) > 0:
                    st.metric("Total Queries", perf_stats['total_queries'])
                    st.metric("Avg Response Time", f"{perf_stats.get('avg_total_time', 0):.1f}s")
                    st.metric("Cache Hits", perf_stats.get('cache_hits', 0))
                else:
                    st.info("No performance data available yet")
            except Exception as e:
                st.warning(f"Performance stats unavailable: {e}")
        
        # System resources
        if SYSTEM_MONITOR:
            st.markdown("### 💾 System Resources")
            try:
                memory_percent = psutil.virtual_memory().percent
                st.metric("Memory Usage", f"{memory_percent:.1f}%")
            except Exception as e:
                st.warning(f"Memory monitoring unavailable: {e}")
        
        # Session info
        st.markdown("### ⏱️ Session Info")
        session_start = st.session_state.get('session_start', time.time())
        session_duration = int((time.time() - session_start) / 60)
        st.metric("Session Duration", f"{session_duration} min")
    
    # Additional system information
    with st.expander("🔍 Detailed System Information"):
        if st.session_state.get('rag_system'):
            try:
                health = st.session_state.rag_system.get_system_health()
                st.json(health)
            except Exception as e:
                st.error(f"Detailed health check failed: {e}")


# ============================================================================
# SIMPLIFIED ENHANCED IMAGE ANALYSIS FUNCTIONS  
# ============================================================================

def handle_image_tab():
    """Enhanced image analysis tab - FIXED to hide interface after analysis"""
    
    # CHECK FOR EXISTING RESULTS FIRST
    if 'image_analysis_result' in st.session_state and st.session_state.image_analysis_result:
        show_enhanced_image_results()
        
        # Add spacing and section for new analysis
        st.markdown("---")
        st.markdown("### 🆕 Analyze New Images")
        
        # Buttons for new analysis with unique keys
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📷 Analyze Different Images", type="primary", use_container_width=True, key="enh_analyze_different"):
                clear_image_analysis_result()
                st.rerun()
        
        with col2:
            if st.button("🔄 Start Fresh", use_container_width=True, key="enh_start_fresh"):
                clear_image_analysis_result()
                st.rerun()
        
        # REMOVED: Don't show upload interface in expander when results exist
        # This was causing the interface to still be visible
        
        return  # Exit early - don't show upload interface
    
    # Show upload interface (moved to top)
    show_simplified_upload_interface()

def show_simplified_upload_interface():
    """Simplified upload interface - FIXED KEYS"""
    st.subheader("📷 Visual Analysis")
    st.markdown("Upload images for clear, actionable analysis and compliance checking.")
    
    # 1. CUSTOM FOCUS - with unique key prefix
    custom_focus = st.text_area(
        "🎯 Additional Focus (optional):",
        placeholder="e.g., 'Pay special attention to the dining area setup' or leave blank",
        height=60,
        key="img_tab_custom_focus"  # Added img_tab_ prefix
    )
    
    # 2. Analysis focus selection - with unique key prefix
    analysis_focus = st.selectbox(
        "🎯 Analysis Focus:",
        [
            "🛡️ Safety & Compliance Check",
            "🏠 Environment Quality Assessment", 
            "🍎 Food & Kitchen Safety Review",
            "💚 Wellbeing & Atmosphere Evaluation",
            "📋 General Facility Inspection"
        ],
        help="Choose what you want to focus on",
        key="img_tab_analysis_select"  # Added img_tab_ prefix
    )
    
    # 3. Detail level - with unique key prefix
    detail_level = st.radio(
        "📊 Report Style:",
        ["🚀 Quick Summary (Key points only)", "📋 Structured Report (Recommended)", "🔍 Detailed Analysis"],
        index=1,
        key="img_tab_detail_select"  # Added img_tab_ prefix
    )
    
    # 4. Image upload - with unique key prefix
    if 'image_uploader_key' not in st.session_state:
        st.session_state.image_uploader_key = 0
    
    uploaded_images = st.file_uploader(
        "Upload Images (1-3 images recommended)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload facility photos for analysis",
        key=f"img_tab_uploader_{st.session_state.image_uploader_key}"  # Added img_tab_ prefix
    )
    
    # Show the analyze button immediately after the file uploader
    if uploaded_images:
        # Place a horizontal line for visual separation
        st.markdown("---")
        
        # THE ANALYZE BUTTON - with unique key prefix
        if st.button("🔍 Analyze Images", type="primary", use_container_width=True, key="img_tab_analyze_btn"):
            analyze_images_simplified(uploaded_images, analysis_focus, detail_level, custom_focus)
        
        st.markdown("---")
        
        # Show image preview below the button
        st.markdown("### 📁 Images Ready for Analysis")
        st.info(f"📷 {len(uploaded_images)} image(s) uploaded and ready for analysis.")
        
        # Show images in expandable section
        with st.expander("👁️ Preview Uploaded Images", expanded=False):
            cols = st.columns(min(3, len(uploaded_images)))
            for i, image in enumerate(uploaded_images):
                with cols[i % 3]:
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)
    

def analyze_images_simplified(images, analysis_focus, detail_level, custom_focus):
    """Simplified image analysis with cleaner prompts"""
    
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # Progress tracking
        with progress_container.container():
            progress_bar = st.progress(20)
        with status_container.container():
            st.info("📷 Processing uploaded images...")
        
        # Create simplified analysis prompt
        prompt = create_simplified_analysis_prompt(analysis_focus, detail_level, custom_focus, len(images))
        
        with progress_container.container():
            progress_bar = st.progress(60)
        with status_container.container():
            st.info("🔍 Conducting visual analysis...")
        
        # Process first image (or combine if multiple)
        image_data = images[0]
        image_data.seek(0)
        original_bytes = image_data.read()
        compressed_bytes = compress_image_for_analysis(original_bytes)
        
        # Get analysis result
        result = st.session_state.rag_system.query_with_performance_mode(
            prompt,
            st.session_state.get('performance_mode', 'balanced'),
            compressed_bytes
        )
        
        with progress_container.container():
            progress_bar = st.progress(100)
        with status_container.container():
            st.success("✅ Analysis complete!")
        
        time.sleep(1)
        progress_container.empty()
        status_container.empty()
        
        if result and result.get('answer'):
            # Store results
            st.session_state.image_analysis_result = result
            st.session_state.image_metadata = {
                'image_count': len(images),
                'primary_image': images[0].name,
                'analysis_focus': analysis_focus,
                'detail_level': detail_level,
                'custom_focus': custom_focus
            }
            
            # Clear uploader for next use
            st.session_state.image_uploader_key += 1
            
            # Log activity
            log_tester_activity(
                st.session_state.get('tester_id', 'unknown'),
                "enhanced_image_analysis_completed",
                f"Images: {len(images)}, Focus: {analysis_focus}"
            )
            
            # CRITICAL FIX: Tell the app to stay on image tab after rerun
            st.session_state.stay_on_image_tab = True

            # Refresh to show results
            st.rerun()
        else:
            st.error("❌ Analysis failed to produce results. Please try again.")
            
    except Exception as e:
        progress_container.empty()
        status_container.empty()
        
        st.error("❌ Image analysis failed.")
        
        with st.expander("🔧 Error Details"):
            st.code(f"Error: {str(e)}")

def create_simplified_analysis_prompt(analysis_focus, detail_level, custom_focus, image_count):
    """Create clean, focused analysis prompts"""
    
    # Base analysis instructions based on focus
    focus_instructions = {
        "🛡️ Safety & Compliance Check": "Focus on health and safety compliance, fire safety, hazard identification, and regulatory requirements.",
        "🏠 Environment Quality Assessment": "Evaluate the quality of the physical environment, comfort, cleanliness, maintenance, and child-friendly features.",
        "🍎 Food & Kitchen Safety Review": "Examine food preparation areas, hygiene standards, equipment safety, and food safety compliance.",
        "💚 Wellbeing & Atmosphere Evaluation": "Assess how the environment supports children's emotional wellbeing, dignity, privacy, and positive atmosphere.",
        "📋 General Facility Inspection": "Comprehensive visual inspection covering safety, quality, compliance, and child experience."
    }
    
    base_instruction = focus_instructions.get(analysis_focus, "Provide a comprehensive facility analysis.")
    
    # Adjust detail level
    if "Quick Summary" in detail_level:
        format_instruction = """
Provide a CONCISE analysis in this format:
• **Key Strengths:** 2-3 positive observations
• **Areas for Attention:** 2-3 priority improvements 
• **Compliance Status:** Brief compliance assessment
• **Next Steps:** 1-2 immediate actions needed
        """
    elif "Detailed Analysis" in detail_level:
        format_instruction = """
Provide a COMPREHENSIVE analysis in this format:
• **Overview:** What you observe in the image(s)
• **Compliance Assessment:** Detailed regulatory evaluation
• **Safety Analysis:** Specific safety observations and concerns
• **Quality Evaluation:** Environment quality and standards
• **Positive Practices:** Good examples identified
• **Improvement Areas:** Specific recommendations with rationale
• **Action Plan:** Prioritized steps for improvement
        """
    else:  # Structured Report (default)
        format_instruction = """
Provide a STRUCTURED analysis in this format:
• **Summary:** Brief overview of what you observe
• **Key Findings:** 3-4 main observations (both positive and areas for improvement)
• **Compliance Notes:** Important regulatory considerations
• **Recommendations:** 2-3 actionable next steps
        """
    
    # Add custom focus if provided
    custom_instruction = f"\n\nAdditional Focus: {custom_focus}" if custom_focus.strip() else ""
    
    # Complete prompt
    return f"""
Analyze this image for a children's home facility with focus on: {base_instruction}

{format_instruction}

{custom_instruction}

Be specific, practical, and focus on actionable insights for facility managers.
    """

def show_enhanced_image_results():
    """Display enhanced image analysis results with clean interface"""
    
    result = st.session_state.image_analysis_result
    metadata = st.session_state.get('image_metadata', {})
    
    # Show what was analyzed
    st.markdown("### 📷 Visual Analysis Complete")
    
    analysis_info = f"✅ Analyzed {metadata.get('image_count', 1)} image(s)"
    if metadata.get('primary_image'):
        analysis_info += f" • Primary: {metadata.get('primary_image')}"
    analysis_info += f" • Focus: {metadata.get('analysis_focus', 'General')}"
    
    st.info(analysis_info)
    
    # Show model attribution if available
    if result.get('metadata') and result['metadata'].get('llm_used'):
        st.caption(f'*Analysis by {result["metadata"]["llm_used"]}*')
    
    # Main analysis result - CLEAN DISPLAY
    st.subheader("📋 Visual Analysis Report")
    st.write(result['answer'])
    
    # Action buttons with unique keys
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Analyze New Images", type="primary", key="enh_new_analysis"):
            clear_image_analysis_result()
    
    with col2:
        if st.button("📄 Download Report", key="enh_download_report"):
            generate_enhanced_image_report(result, metadata)
    
    with col3:
        if st.button("📊 Analysis Details", key="enh_show_details"):
            st.session_state.show_enh_details = not st.session_state.get('show_enh_details', False)
    
    # Progressive disclosure for details
    if st.session_state.get('show_enh_details', False):
        st.markdown("---")
        st.markdown("**📊 Analysis Details:**")
        if metadata:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Images analyzed:** {metadata.get('image_count', 1)}")
                st.write(f"**Detail level:** {metadata.get('detail_level', 'Standard')}")
            with col2:
                st.write(f"**Analysis focus:** {metadata.get('analysis_focus', 'General')}")
                if metadata.get('custom_focus'):
                    st.write(f"**Custom focus:** {metadata.get('custom_focus')}")

def clear_image_analysis_result():
    """Clear image analysis results and FORCE file uploader reset - UPDATED"""
    # Clear all result data
    for key in ['image_analysis_result', 'image_metadata', 'show_enh_details']:
        if key in st.session_state:
            del st.session_state[key]
    
    # CRITICAL FIX: Force file uploader to reset by incrementing key
    if 'image_uploader_key' not in st.session_state:
        st.session_state.image_uploader_key = 0
    st.session_state.image_uploader_key += 1

def generate_enhanced_image_report(result, metadata):
    """Generate downloadable enhanced image analysis report"""
    report_content = f"""
# Visual Analysis Report
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Images Analyzed:** {metadata.get('image_count', 1)}
**Primary Image:** {metadata.get('primary_image', 'Unknown')}
**Analysis Focus:** {metadata.get('analysis_focus', 'General')}
**Detail Level:** {metadata.get('detail_level', 'Standard')}

## Analysis Results
{result['answer']}

## Additional Information
- **Custom Focus:** {metadata.get('custom_focus', 'None specified')}
- **Generated by:** Children's Home Management System
- **Analysis Mode:** {st.session_state.get('performance_mode', 'balanced')}

---
*Report generated by Children's Home Management System*
    """
    
    st.download_button(
        label="📥 Download Complete Report",
        data=report_content,
        file_name=f"visual_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key="enh_download_btn"
    )

# ============================================================================
# HELPER FUNCTIONS FOR REPORTS
# ============================================================================

def generate_document_report(result):
    """Generate downloadable document analysis report"""
    report_content = f"""
# Document Analysis Report
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** {st.session_state.analysis_metadata.get('analysis_type', 'Unknown')}

## Analysis Results
{result['answer']}

---
*Report generated by Children's Home Management System*
    """
    
    st.download_button(
        label="📥 Download Document Report",
        data=report_content,
        file_name=f"document_analysis_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def generate_image_report(result):
    """Generate downloadable image analysis report"""
    report_content = f"""
# Image Analysis Report
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Image:** {st.session_state.image_metadata.get('image_name', 'Unknown')}

## Visual Analysis Results
{result['answer']}

---
*Report generated by Children's Home Management System*
    """
    
    st.download_button(
        label="📥 Download Image Report",
        data=report_content,
        file_name=f"image_analysis_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


def generate_analysis_report(results, analysis_type, images):
    """Generate a comprehensive analysis report"""
    st.markdown("### 📄 Comprehensive Analysis Report")
    
    report_content = f"""
# Image Analysis Report
**Analysis Type:** {analysis_type}  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Images Analyzed:** {len(images)}

## Executive Summary
{generate_executive_summary(results)}

## Detailed Findings
{format_detailed_findings(results)}

## Recommendations
{generate_recommendations(results)}

## Next Steps
Based on this analysis, consider the following actions:
1. Address any critical safety issues immediately
2. Develop improvement plan for identified gaps
3. Schedule follow-up assessment in 3-6 months
4. Document changes and improvements made

---
*Report generated by Children's Home Management System*
    """
    
    st.markdown(report_content)
    
    # Download button
    st.download_button(
        label="📥 Download Report",
        data=report_content,
        file_name=f"image_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def generate_executive_summary(results):
    """Generate executive summary from results"""
    # This would analyze the results and create a summary
    return "Analysis completed successfully. Key findings and recommendations have been identified."

def format_detailed_findings(results):
    """Format detailed findings from results"""
    # This would format the analysis results
    return "Detailed findings have been documented for each analyzed image."

def generate_recommendations(results):
    """Generate recommendations from results"""
    # This would extract recommendations from the analysis
    return "Specific recommendations have been provided based on the analysis findings."

def compress_image_for_analysis(image_bytes, max_size_kb=30):
    """Compress image to stay under the AI model size limit"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = rgb_image
        
        # Start with reasonable size and quality
        quality = 85
        max_dimension = 800
        
        # Resize if too large
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Compress until under size limit
        while True:
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_bytes = output.getvalue()
            
            if len(compressed_bytes) <= max_size_kb * 1024:
                return compressed_bytes
            
            # Reduce quality or resize further
            if quality > 60:
                quality -= 10
            else:
                current_size = image.size
                new_size = (int(current_size[0] * 0.8), int(current_size[1] * 0.8))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                quality = 75
            
            # Safety check
            if quality < 30 and max(image.size) < 200:
                break
        
        return compressed_bytes
        
    except Exception as e:
        print(f"Image compression failed: {e}")
        return image_bytes

# ===== SESSION MANAGEMENT =====

def show_session_summary():
    """Enhanced session summary with performance stats"""
    with st.sidebar:
        if st.session_state.get('activity_log'):
            st.markdown("---")
            st.markdown("### 📊 Session Summary")
            
            activities = st.session_state.get('activity_log', [])
            question_count = len([a for a in activities if a['action'] == 'question_submitted'])
            feedback_count = len([a for a in activities if a['action'] == 'feedback_submitted'])
            
            st.write(f"**Questions asked:** {question_count}")
            st.write(f"**Feedback items:** {feedback_count}")
            
            session_minutes = int((time.time() - st.session_state.get('session_start', time.time())) / 60)
            st.write(f"**Session time:** {session_minutes} minutes")
            
            # Performance stats if available
            if hasattr(st.session_state.rag_system, 'get_performance_stats'):
                try:
                    perf_stats = st.session_state.rag_system.get_performance_stats()
                    if isinstance(perf_stats, dict) and perf_stats.get('total_queries', 0) > 0:
                        st.markdown("**⚡ Performance:**")
                        avg_time = perf_stats.get('avg_total_time', 0)
                        cache_hits = perf_stats.get('cache_hits', 0)
                        st.write(f"Avg response: {avg_time:.1f}s")
                        if cache_hits > 0:
                            st.write(f"Cache hits: {cache_hits}")
                except Exception:
                    pass
            
            with st.expander("📋 Activity Log"):
                for activity in activities[-10:]:
                    st.text(f"{activity['timestamp']}: {activity['action']}")

# ===== MAIN APPLICATION =====

def main():
    """Main application function"""
    # Setup SQLite with clean logging
    setup_sqlite_with_clean_logging()

    # Authentication check FIRST - before any other UI elements
    if not require_authentication():
        st.stop()  # This stops execution if not authenticated
    
    # Environment setup
    if not setup_environment_variables():
        st.error("❌ Failed to configure API keys")
        st.info("💡 Check your .streamlit/secrets.toml file")
        with st.expander("🔧 Configuration Help"):
            st.code("""
# .streamlit/secrets.toml
[api_keys]
openai = "your-openai-api-key"
google = "your-google-api-key"
            """)
        st.stop()
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.rag_system = initialize_rag_system()
    
    if st.session_state.rag_system is None:
        st.error("❌ System initialization failed")
        st.info("💡 Please refresh the page or contact support")
        st.stop()
    
    # CSS styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    .testing-banner {
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        border-radius: 5px;
    }
    .contextual-tip {
        background-color: #f8f9fa;
        border-left: 3px solid #17a2b8;
        padding: 0.75rem;
        border-radius: 0 5px 5px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Beta testing banner
    st.markdown("""
    <div class="testing-banner">
    🧪 BETA TESTING VERSION - Your feedback helps improve this system!
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🏠 Children's Home Management System")
    st.markdown("*Strategic, operational & compliance guidance for residential care*")
    
    tester_info = st.session_state.get('tester_info', {})
    if tester_info:
        st.markdown(f"*Welcome, {tester_info.get('name', 'Beta Tester')}!*")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add performance selector
    add_performance_mode_selector()
    
    # Feedback mechanism in sidebar
    with st.sidebar:
        st.markdown("### 📝 Beta Feedback")
        
        feedback_type = st.selectbox(
            "Feedback Type:",
            ["💡 Suggestion", "🐛 Bug Report", "👍 Positive", "❓ Question"]
        )
        
        feedback_text = st.text_area(
            "Your Feedback:",
            placeholder="Share thoughts, report bugs, or suggest improvements...",
            height=100
        )
        
        if st.button("📤 Submit Feedback", type="secondary"):
            if feedback_text.strip():
                log_tester_activity(
                    st.session_state.get('tester_id', 'unknown'),
                    "feedback_submitted",
                    f"{feedback_type}: {feedback_text[:100]}..."
                )
                st.success("✅ Feedback submitted!")
            else:
                st.warning("Please enter feedback before submitting")
    
    # Debug controls (only for specific testers)
    with st.sidebar:
        if st.session_state.get('tester_id') in ['DEMO001', 'TEST001']:
            st.markdown("---")
            st.markdown("### 🔧 Debug Controls")
            
            current_debug = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
            debug_mode = st.checkbox(
                "🔍 Debug Mode", 
                value=current_debug,
                help="Show detailed system logs in terminal"
            )
            
            # Update environment variable
            if debug_mode != current_debug:
                if debug_mode:
                    os.environ['DEBUG_MODE'] = 'true'
                    st.success("🔍 Debug logging enabled")
                else:
                    os.environ['DEBUG_MODE'] = 'false'
                    st.info("🔇 Debug logging disabled")
            
            if debug_mode:
                st.caption("📝 Debug messages will appear in terminal")
            else:
                st.caption("🔇 Only critical events logged")
    
    # Show session summary
    show_session_summary()
    
# ============================================================================
# UPDATED MAIN TAB STRUCTURE
# ============================================================================

    # Initialize tab state
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0

    # Check if we should stay on image tab after analysis
    if 'stay_on_image_tab' in st.session_state and st.session_state.stay_on_image_tab:
        st.session_state.current_tab = 2  # Image tab index
        del st.session_state.stay_on_image_tab

    # Create tabs with state management
    tab_names = ["💬 Ask Questions", "📄 Analyze Documents", "📷 Analyze Images", "⚙️ System Health"]

    # Use the radio button value directly, don't fight with session state
    selected_tab = st.radio(
        "Select Tab:",
        options=list(range(len(tab_names))),
        format_func=lambda x: tab_names[x],
        index=st.session_state.current_tab,
        horizontal=True,
        key="tab_selector",
        label_visibility="collapsed"
    )

    # Always update session state to match the radio button
    st.session_state.current_tab = selected_tab

    # Display content based on selected tab
    if selected_tab == 0:
        handle_question_tab()
    elif selected_tab == 1:
        handle_document_analysis_tab()
    elif selected_tab == 2:
        handle_image_tab()
    elif selected_tab == 3:
        display_system_health()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
    🧪 <strong>Thank you for beta testing!</strong> Your feedback helps improve this system.<br>
    Use the feedback form in the sidebar to report issues or suggestions.
    </div>
    """, unsafe_allow_html=True)



# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    # Track page loads
    if 'page_loads' not in st.session_state:
        st.session_state['page_loads'] = 0
        log_tester_activity(
            st.session_state.get('tester_id', 'unknown'),
            "session_started",
            "Application loaded"
        )
    
    st.session_state['page_loads'] += 1
    
    # Run main application
    main()
        
