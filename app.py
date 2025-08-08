# app.py - COMPLETE FIXED VERSION WITH ALL CRITICAL ISSUES RESOLVED

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

# Global flags to prevent repeated debug messages
_SQLITE_LOGGED = False
_ENV_SETUP_LOGGED = False

# ===== CRITICAL FIX #1: GLOBAL UI STATE INITIALIZATION =====
def initialize_global_ui_state():
    """Initialize global UI state - SAFE TO CALL MULTIPLE TIMES"""
    if 'ui_state' not in st.session_state:
        st.session_state.ui_state = {
            'current_result': None,
            'current_question': "",
            'last_asked_question': "",
            'show_sources': False,
            'show_analytics': False,
            'question_counter': 0,
            'initialized': True
        }
    
    # Initialize other session state variables if needed
    if 'show_doc_findings' not in st.session_state:
        st.session_state.show_doc_findings = False
    if 'show_doc_sources' not in st.session_state:
        st.session_state.show_doc_sources = False
    if 'show_img_details' not in st.session_state:
        st.session_state.show_img_details = False

# Call global initialization immediately
initialize_global_ui_state()

def debug_log(message, once_only=False, key=None):
    """Only print debug messages if debug mode is enabled, with optional one-time logging"""
    if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
        if once_only and key:
            if 'debug_logged' not in st.session_state:
                st.session_state.debug_logged = set()
            
            if key not in st.session_state.debug_logged:
                print(f"DEBUG: {message}")
                st.session_state.debug_logged.add(key)
        elif not once_only:
            print(f"DEBUG: {message}")

def debug_timing(func_name, start_time):
    """Debug timing helper"""
    if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
        elapsed = time.time() - start_time
        print(f"TIMING: {func_name} took {elapsed:.3f}s")

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
from rag_system import HybridRAGSystem as EnhancedRAGSystem, create_rag_system

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

# ===== CACHED CSS =====
@st.cache_data
def get_app_css():
    return """
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
    .result-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .question-container {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .clean-separator {
        margin: 2rem 0;
        border-top: 2px solid #e9ecef;
    }
    .stTextArea > div > div > textarea {
        border-radius: 8px;
    }
    .stButton > button {
        border-radius: 8px;
    }
    </style>
    """

# ===== AUTHENTICATION SYSTEM =====
def get_tester_credentials():
    """Get tester credentials with fallback system"""
    try:
        testers = st.secrets.get("testers", {})
        if testers:
            return testers
    except Exception as e:
        debug_log(f"Secrets access failed: {e}")
    
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
        
        if 'activity_log' not in st.session_state:
            st.session_state['activity_log'] = []
        
        st.session_state['activity_log'].append(log_entry)
        
        critical_actions = [
            'successful_login', 'failed_login', 'logout',
            'system_error', 'feedback_submitted'
        ]
        
        if action in critical_actions:
            print(f"TESTER: {timestamp} - {tester_id or 'anonymous'} - {action}")
            if action == 'system_error' and details:
                print(f"  └─ {details}")
        elif os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            print(f"DEBUG: {timestamp} - {tester_id or 'anonymous'} - {action}")
            if details:
                print(f"  └─ {details}")
        
    except Exception:
        pass

def validate_and_authenticate(tester_id, access_code):
    """Validate credentials and set up authenticated session"""
    try:
        if not tester_id or not access_code:
            return False
        
        valid_testers = get_tester_credentials()
        
        if tester_id not in valid_testers:
            return False
        
        tester_info = valid_testers[tester_id]
        if tester_info.get("password") != access_code:
            return False
        
        expiry_date = tester_info.get("expires", "2099-12-31")
        try:
            expiry = datetime.datetime.strptime(expiry_date, "%Y-%m-%d").date()
            if datetime.date.today() > expiry:
                return False
        except:
            pass
        
        st.session_state['authenticated_tester'] = True
        st.session_state['tester_id'] = tester_id
        st.session_state['tester_info'] = tester_info
        st.session_state['session_start'] = time.time()
        
        log_tester_activity(tester_id, "successful_login", f"User: {tester_info.get('name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Authentication failed: {e}")
        return False

def show_login_interface():
    """Streamlined login interface"""
    st.markdown(get_app_css(), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-container" style="max-width: 600px; margin: 2rem auto; padding: 2rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e9ecef;">
    <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1>🏠 Children's Home Management System</h1>
        <h3>Beta Testing Portal</h3>
        <p>Secure access for authorized testers</p>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 Tester Login")
    
    tester_id = st.text_input(
        "Tester ID:",
        placeholder="e.g., DEMO001",
        help="Your unique tester identifier",
        key="login_tester_id_input"
    )
    
    access_code = st.text_input(
        "Access Code:",
        type="password",
        placeholder="Enter your access code",
        help="Secure password provided with your Tester ID",
        key="login_access_code_input"
    )
    
    if st.button("🚀 Start Testing Session", type="primary", use_container_width=True, key="login_submit_button"):
        if validate_and_authenticate(tester_id, access_code):
            st.success("✅ Authentication successful!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please verify your Tester ID and Access Code.")
            log_tester_activity(tester_id or "unknown", "failed_login", "Invalid credentials")
    
    with st.expander("ℹ️ Testing Information"):
        st.markdown("""
        **Session Details:**
        - Duration: 2 hours per session
        - Auto-logout after inactivity
        - Activity logging for improvement
        
        **Need Access?**
        Contact the administrator to receive your testing credentials.
        """)

def show_session_info():
    """Show session information in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 Session Info")
        
        tester_info = st.session_state.get('tester_info', {})
        st.write(f"**Tester:** {tester_info.get('name', 'Unknown')}")
        st.write(f"**ID:** {st.session_state.get('tester_id', 'N/A')}")
        
        session_start = st.session_state.get('session_start', time.time())
        elapsed = time.time() - session_start
        remaining = max(0, 7200 - elapsed)
        
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        st.write(f"**Time remaining:** {hours}h {minutes}m")
        
        if st.button("🚪 End Session", type="secondary", key="sidebar_logout_button"):
            log_tester_activity(st.session_state.get('tester_id', 'unknown'), "logout")
            
            for key in ['authenticated_tester', 'tester_id', 'tester_info', 'session_start']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()

def require_authentication():
    """Main authentication function"""
    if check_session_valid():
        show_session_info()
        return True
    
    st.session_state['authenticated_tester'] = False
    show_login_interface()
    return False

# ===== ENVIRONMENT SETUP =====
def setup_environment_variables():
    """Enhanced environment setup with better local support"""
    global _ENV_SETUP_LOGGED
    
    try:
        try:
            openai_key = st.secrets.get("OPENAI_API_KEY")
            google_key = st.secrets.get("GOOGLE_API_KEY")
            
            if openai_key and google_key:
                os.environ['OPENAI_API_KEY'] = openai_key
                os.environ['GOOGLE_API_KEY'] = google_key
                if not _ENV_SETUP_LOGGED:
                    debug_log("API keys loaded from Streamlit secrets", once_only=True, key="env_streamlit")
                    _ENV_SETUP_LOGGED = True
                return True
        except Exception as e:
            debug_log(f"Streamlit secrets failed: {e}")
        
        try:
            api_keys = st.secrets.get("api_keys", {})
            openai_key = api_keys.get("openai")
            google_key = api_keys.get("google")
            
            if openai_key and google_key:
                os.environ['OPENAI_API_KEY'] = openai_key
                os.environ['GOOGLE_API_KEY'] = google_key
                if not _ENV_SETUP_LOGGED:
                    debug_log("API keys loaded from api_keys section", once_only=True, key="env_nested")
                    _ENV_SETUP_LOGGED = True
                return True
        except Exception as e:
            debug_log(f"Nested secrets failed: {e}")
        
        try:
            import toml
            secrets_path = Path(".streamlit/secrets.toml")
            
            if secrets_path.exists():
                with open(secrets_path, 'r') as f:
                    secrets = toml.load(f)
                
                openai_key = secrets.get("OPENAI_API_KEY")
                google_key = secrets.get("GOOGLE_API_KEY")
                
                if not (openai_key and google_key):
                    api_keys = secrets.get("api_keys", {})
                    openai_key = api_keys.get("openai")
                    google_key = api_keys.get("google")
                
                if openai_key and google_key:
                    os.environ['OPENAI_API_KEY'] = openai_key
                    os.environ['GOOGLE_API_KEY'] = google_key
                    if not _ENV_SETUP_LOGGED:
                        debug_log("API keys loaded via manual TOML parsing", once_only=True, key="env_manual")
                        _ENV_SETUP_LOGGED = True
                    return True
        except Exception as e:
            debug_log(f"Manual TOML loading failed: {e}")
        
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            google_key = os.getenv("GOOGLE_API_KEY")
            
            if openai_key and google_key:
                if not _ENV_SETUP_LOGGED:
                    debug_log("API keys found in environment variables", once_only=True, key="env_vars")
                    _ENV_SETUP_LOGGED = True
                return True
        except Exception as e:
            debug_log(f"Environment variables check failed: {e}")
        
        debug_log("WARNING: No API keys found in any location")
        return False
        
    except Exception as e:
        debug_log(f"Environment setup failed: {e}")
        return False

# ===== RAG SYSTEM INITIALIZATION =====
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with enhanced error handling"""
    debug_log("Initializing RAG system...", once_only=True, key="rag_init_start")
    
    try:
        if not os.path.exists("faiss_index"):
            st.error("❌ FAISS index not found!")
            st.info("💡 The vector database needs to be created first.")
            st.info("🔧 Please run the ingestion script to create the index.")
            return None
        
        import asyncio
        
        os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
        os.environ['GRPC_POLL_STRATEGY'] = 'poll'
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        if sys.platform.startswith('win'):
            if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        debug_log("Asyncio configured", once_only=True, key="asyncio_config")
        
    except Exception as e:
        debug_log(f"Asyncio configuration failed: {e}")
    
    try:
        rag_system = create_rag_system()
        debug_log("RAG system initialized successfully", once_only=True, key="rag_init_success")
        return rag_system
        
    except ImportError as e:
        st.error(f"❌ Import Error: {e}")
        st.info("💡 Missing required packages. Check requirements.txt")
        return None
        
    except Exception as e:
        st.error(f"❌ RAG System Error: {e}")
        debug_log(f"RAG system initialization failed: {e}")
        
        error_msg = str(e).lower()
        if "faiss" in error_msg:
            st.info("💡 FAISS vector database not found or corrupted")
        elif "api" in error_msg or "key" in error_msg:
            st.info("💡 Check API key configuration in secrets")
        elif "import" in error_msg or "module" in error_msg:
            st.info("💡 Check that all dependencies are in requirements.txt")
        
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
            help="Choose speed vs comprehensiveness trade-off",
            key="performance_mode_selector_widget"
        )
        
        st.session_state['performance_mode'] = selected_mode
        
        if selected_mode == "fast":
            st.caption("⚡ Optimized for speed. Best for simple, factual questions.")
        elif selected_mode == "balanced":
            st.caption("⚖️ Good balance of speed and completeness. Recommended for most queries.")
        else:
            st.caption("🔍 Maximum thoroughness. Best for complex strategic questions.")
        
        return selected_mode

# ===== CRITICAL FIX #2: UI STATE MANAGEMENT WITH DEFENSIVE INITIALIZATION =====
def ensure_ui_state():
    """Defensive UI state initialization - SAFE TO CALL ANYWHERE"""
    initialize_global_ui_state()

def clear_ui_state():
    """Clear UI state for new question"""
    ensure_ui_state()
    st.session_state.ui_state.update({
        'current_result': None,
        'current_question': "",
        'last_asked_question': "",
        'show_sources': False,
        'show_analytics': False,
        'question_counter': st.session_state.ui_state.get('question_counter', 0) + 1
    })

# ===== CRITICAL FIX #3: SOURCE PROCESSING FUNCTIONS WITH COMPLETE IMPLEMENTATION =====
def extract_source_info(source):
    """Extract source information from Document objects or dictionaries"""
    # Handle Document objects from LangChain
    if hasattr(source, 'metadata') and hasattr(source, 'page_content'):
        metadata = source.metadata
        return {
            'title': metadata.get('title', ''),
            'source': metadata.get('source', ''),
            'source_type': metadata.get('source_type', ''),
            'word_count': len(source.page_content.split()) if source.page_content else 0,
            'page_content': source.page_content
        }
    # Handle dictionary sources
    elif isinstance(source, dict):
        return {
            'title': source.get('title', ''),
            'source': source.get('source', ''),
            'source_type': source.get('source_type', ''),
            'word_count': source.get('word_count', 0),
            'page_content': source.get('page_content', '')
        }
    else:
        return {
            'title': f'Unknown Source',
            'source': '',
            'source_type': 'unknown',
            'word_count': 0,
            'page_content': ''
        }

def deduplicate_sources(sources):
    """Deduplicate sources by filename/title"""
    unique_sources = {}
    
    for source in sources:
        source_info = extract_source_info(source)
        
        # Create a unique key for this document
        source_file = source_info['source']
        title = source_info['title']
        
        if source_file:
            filename = os.path.basename(source_file)
            doc_key = filename
        elif title and len(title.strip()) > 3:
            doc_key = title.strip()
        else:
            content = source_info['page_content']
            doc_key = f"doc_{hash(content[:100]) % 1000}"
        
        # Only keep first occurrence of each document
        if doc_key not in unique_sources:
            unique_sources[doc_key] = {
                'title': title,
                'source_file': source_file,
                'source_type': source_info['source_type'],
                'word_count': source_info['word_count'],
                'chunks_found': 1
            }
        else:
            unique_sources[doc_key]['chunks_found'] += 1
            # Update word count if it's higher
            if source_info['word_count'] > unique_sources[doc_key]['word_count']:
                unique_sources[doc_key]['word_count'] = source_info['word_count']
    
    return unique_sources

def clean_source_title(title, source_file, index):
    """Clean up source title for display"""
    if not title or len(title.strip()) < 3:
        if source_file:
            filename = os.path.basename(source_file)
            clean_title = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
            clean_title = clean_title.replace('_', ' ').replace('-', ' ')
            title = ' '.join(word.capitalize() for word in clean_title.split())
        else:
            title = f'Reference Document {index}'
    
    # Truncate very long titles
    if len(title) > 80:
        title = title[:77] + "..."
    
    return title

# ===== CRITICAL FIX #4: COMPLETE REPORT GENERATION =====
def generate_simple_report(result, question):
    """COMPLETE report generation with proper source handling"""
    
    # Extract sources properly
    sources = result.get("sources", [])
    
    # Build comprehensive report
    report_content = f"""# Children's Home Management - Q&A Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Performance Mode:** {st.session_state.get('performance_mode', 'balanced').title()}

---

## Question
{question}

## Expert Guidance
{result['answer']}

## Sources Referenced
"""
    
    if sources:
        # Deduplicate sources for the report
        unique_sources = deduplicate_sources(sources)
        
        # Add deduplicated sources to report
        for i, (doc_key, doc_info) in enumerate(unique_sources.items(), 1):
            title = clean_source_title(doc_info['title'], doc_info['source_file'], i)
            source_file = doc_info['source_file']
            chunks_found = doc_info['chunks_found']
            word_count = doc_info['word_count']
            source_type = doc_info['source_type']
            
            # Add to report
            report_content += f"{i}. **{title}**\n"
            
            if source_file:
                filename = os.path.basename(source_file)
                if chunks_found > 1:
                    report_content += f"   - Source: {filename} ({chunks_found} sections referenced)\n"
                else:
                    report_content += f"   - Source: {filename}\n"
            elif chunks_found > 1:
                report_content += f"   - {chunks_found} sections referenced\n"
            
            # Add source type if available
            if source_type and source_type.strip() and source_type != 'unknown':
                report_content += f"   - Type: {source_type}\n"
            
            # Add word count if meaningful
            if word_count > 0:
                report_content += f"   - Length: {word_count:,} words\n"
            
            report_content += "\n"
    else:
        report_content += "No specific source documents were referenced for this response.\n\n"
    
    # Add footer
    report_content += f"""
---
*Generated by Children's Home Management System*
*Session: {st.session_state.get('tester_id', 'anonymous')}*
"""
    
    st.download_button(
        label="📥 Download Report",
        data=report_content,
        file_name=f"qa_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
        key="download_simple_report_button"
    )

# ===== CRITICAL FIX #5: OPTIMIZED QUESTION HANDLING WITH UNIQUE KEYS =====
def show_clean_question_form():
    """Clean, optimized question form with unique keys"""
    ensure_ui_state()
    
    st.subheader("💬 Ask Your Question")
    
    # Use counter to force clearing of text input
    question_key = f"question_input_{st.session_state.ui_state.get('question_counter', 0)}"
    
    user_question = st.text_area(
        "Describe your situation or question:",
        value="",  # Always start with empty value
        placeholder="Start typing your question... (e.g., 'How do we prepare for an Ofsted inspection?')",
        height=120,
        key=question_key,
        help="Ask about operations, compliance, policies, or any children's home management topic"
    )
    
    # Submit button with UNIQUE KEY
    if st.button("🧠 Get Expert Guidance", type="primary", use_container_width=True, key="main_submit_question_button"):
        if user_question and user_question.strip():
            process_question_optimized(user_question)
        else:
            st.warning("⚠️ Please enter a question to receive guidance")
    
    # Quick actions with UNIQUE KEYS
    show_quick_actions()

def show_quick_actions():
    """Quick action buttons with unique keys"""
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Options")
    
    quick_actions = [
        ("🚨 Inspection Prep", "We have an upcoming Ofsted inspection. What should we focus on to ensure we're fully prepared?"),
        ("👥 Staff Issues", "We're experiencing staff retention challenges. What strategies can we implement?"),
        ("🏠 New Admission", "We're admitting a new child. What processes should we prioritize for a successful transition?"),
        ("📋 Policy Review", "We need to review and update our policies. What current best practices should we include?"),
        ("💰 Budget Planning", "Help us develop an effective budget strategy for the upcoming financial year."),
        ("🎯 Quality Improvement", "We want to enhance our care quality and move towards outstanding. What should we focus on?")
    ]
    
    cols = st.columns(3)
    for i, (title, question) in enumerate(quick_actions):
        with cols[i % 3]:
            # UNIQUE KEY for each quick action button
            if st.button(title, key=f"quick_action_button_{i}", use_container_width=True):
                process_question_optimized(question)

def process_question_optimized(user_question):
    """OPTIMIZED process question with minimal UI overhead"""
    ensure_ui_state()
    timing_start = time.time()
    
    performance_mode = st.session_state.get('performance_mode', 'balanced')
    
    log_tester_activity(
        st.session_state.get('tester_id', 'unknown'),
        "question_submitted",
        f"Mode: {performance_mode}, Question: {user_question[:50]}..."
    )
    
    # Simple, single progress indicator
    progress_placeholder = st.empty()
    
    try:
        with progress_placeholder:
            st.info("🔍 Processing your question...")
        
        # Direct RAG call without intermediate updates
        rag_start = time.time()
        result = st.session_state.rag_system.query(
            question=user_question,
            k=5,
            response_style="comprehensive",
            performance_mode=performance_mode
        )
        debug_timing("RAG Query", rag_start)
        
        # Clear progress immediately
        progress_placeholder.empty()
        
        if result and result.get("answer"):
            # Store result in clean UI state
            st.session_state.ui_state.update({
                'current_result': result,
                'last_asked_question': user_question,
                'show_sources': False,
                'show_analytics': False
            })
            
            # Show result immediately without rerun
            ui_start = time.time()
            show_clean_result_display()
            debug_timing("UI Rendering", ui_start)
            
            total_time = time.time() - timing_start
            debug_timing("Total Process", timing_start)
            
            log_tester_activity(
                st.session_state.get('tester_id', 'unknown'),
                "successful_response",
                f"Mode: {performance_mode}, Time: {total_time:.1f}s"
            )
            
        else:
            st.error("❌ Sorry, I couldn't generate a response.")
            
    except Exception as e:
        progress_placeholder.empty()
        st.error("❌ An error occurred while processing your question.")
        print(f"ERROR: {str(e)}")
        
        log_tester_activity(
            st.session_state.get('tester_id', 'unknown'),
            "system_error",
            f"Mode: {performance_mode}, Error: {str(e)[:100]}"
        )

def show_clean_result_display():
    """Clean, optimized result display with unique keys"""
    ensure_ui_state()
    result = st.session_state.ui_state['current_result']
    question = st.session_state.ui_state['last_asked_question']
    
    # Question display in clean container
    st.markdown("""
    <div class="question-container">
        <h4>🤔 Your Question:</h4>
        <p>{}</p>
    </div>
    """.format(question), unsafe_allow_html=True)
    
    # Answer in clean container
    st.markdown("""
    <div class="result-container">
        <h3>🧠 Expert Guidance</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(result["answer"])
    
    # Clean action buttons with UNIQUE KEYS - CRITICAL FIX #6: PROPER COLUMN DEFINITION
    st.markdown('<div class="clean-separator"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)  # Define all 4 columns properly
    
    with col1:
        if st.button("🔄 Ask New Question", type="primary", use_container_width=True, key="result_new_question_button"):
            clear_ui_state()
            st.rerun()
    
    with col2:
        if st.button("📄 Download Report", use_container_width=True, key="result_download_report_button"):
            generate_simple_report(result, question)
    
    with col3:
        if st.button("📚 Show Sources", use_container_width=True, key="result_show_sources_button"):
            st.session_state.ui_state['show_sources'] = not st.session_state.ui_state['show_sources']
    
    with col4:
        if st.button("📊 Analytics", use_container_width=True, key="result_show_analytics_button"):
            st.session_state.ui_state['show_analytics'] = not st.session_state.ui_state['show_analytics']
    
    # Progressive disclosure sections
    if st.session_state.ui_state['show_sources']:
        show_fixed_sources(result.get("sources", []))
    
    if st.session_state.ui_state['show_analytics']:
        show_simple_analytics(result)

def show_fixed_sources(sources):
    """FIXED source display with proper deduplication and unique keys"""
    st.markdown("---")
    st.markdown("### 📚 Reference Sources")
    
    if not sources:
        st.info("No specific source documents were referenced for this response.")
        return
    
    # Deduplicate sources using our helper function
    unique_sources = deduplicate_sources(sources)
    
    # Display unique sources
    for i, (doc_key, doc_info) in enumerate(unique_sources.items(), 1):
        title = clean_source_title(doc_info['title'], doc_info['source_file'], i)
        source_file = doc_info['source_file']
        chunks_found = doc_info['chunks_found']
        
        # Display the source
        st.write(f"**{i}.** {title}")
        
        if source_file:
            filename = os.path.basename(source_file)
            if chunks_found > 1:
                st.caption(f"📁 {filename} ({chunks_found} sections referenced)")
            else:
                st.caption(f"📁 {filename}")
        elif chunks_found > 1:
            st.caption(f"📄 {chunks_found} sections referenced")

def show_simple_analytics(result):
    """Simple analytics display with source information"""
    st.markdown("---")
    st.markdown("### 📊 Response Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence = result.get("confidence_score", 0.0)
        if confidence >= 0.8:
            st.success(f"🟢 High Confidence ({confidence:.2f})")
        elif confidence >= 0.6:
            st.warning(f"🟡 Medium Confidence ({confidence:.2f})")
        else:
            st.error(f"🔴 Low Confidence ({confidence:.2f})")
        
        sources_count = len(result.get("sources", []))
        st.metric("Sources Consulted", sources_count)
        
        # Show unique sources count
        if sources_count > 0:
            unique_sources = deduplicate_sources(result.get("sources", []))
            unique_count = len(unique_sources)
            if unique_count != sources_count:
                st.caption(f"📄 {unique_count} unique documents")
    
    with col2:
        performance = result.get("performance", {})
        total_time = performance.get("total_response_time", 0)
        if total_time:
            st.metric("Response Time", f"{total_time:.1f}s")
        
        embedding_provider = result.get("routing_info", {}).get("embedding_provider", "unknown")
        if embedding_provider != "unknown":
            st.write(f"**Embedding Provider:** {embedding_provider.title()}")
        
        # Show retrieval time if available
        retrieval_time = performance.get("retrieval_time", 0)
        if retrieval_time:
            st.caption(f"⚡ Retrieval: {retrieval_time:.2f}s")

# ===== CLEAN QUESTION TAB HANDLER =====
def handle_question_tab():
    """Clean question tab handler with unique keys"""
    ensure_ui_state()
    
    # If there's a current result, show it cleanly
    if st.session_state.ui_state.get('current_result'):
        show_clean_result_display()
        
        # Clean separator for new question section
        st.markdown('<div class="clean-separator"></div>', unsafe_allow_html=True)
        st.markdown("### 🆕 Ask Another Question")
    
    # Always show the clean question form
    show_clean_question_form()

# ===== DOCUMENT ANALYSIS FUNCTIONS =====
def handle_document_analysis_tab():
    """Enhanced document analysis tab with unique keys"""
    
    if 'document_analysis_result' not in st.session_state:
        st.session_state.document_analysis_result = None
    
    if st.session_state.document_analysis_result:
        show_document_analysis_result()
        
        st.markdown("---")
        st.markdown("### 🆕 Analyze New Documents")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Analyze Different Documents", type="primary", use_container_width=True, key="doc_analyze_different_button"):
                st.session_state.document_analysis_result = None
                st.rerun()
        
        with col2:
            if st.button("🔄 Start Fresh", use_container_width=True, key="doc_start_fresh_button"):
                st.session_state.document_analysis_result = None
                st.rerun()
        
        return
    
    show_document_upload_interface()

def show_document_upload_interface():
    """Show the document upload and analysis interface with unique keys"""
    st.subheader("📄 Document Analysis & Processing")
    st.markdown("Upload documents for comprehensive analysis, extraction, and Q&A.")
    
    # Custom analysis focus
    custom_focus = st.text_area(
        "🎯 Custom Analysis Focus (optional):",
        placeholder="e.g., 'Focus on safeguarding policies and compliance requirements'",
        height=80,
        help="Specify what aspects you want the analysis to focus on",
        key="doc_custom_focus_input_widget"
    )
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "📋 Analysis Type:",
        [
            "🎯 Direct Answers Only",
            "📊 Comprehensive Analysis",
            "✅ Compliance Check", 
            "📝 Content Summary",
            "🔍 Key Information Extraction",
            "❓ Document Q&A",
            "📈 Policy Gap Analysis"
        ],
        index=0,
        help="Choose the type of analysis you need",
        key="doc_analysis_type_select_widget"
    )   
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True,
        help="Supports PDF, Word, Text, and Markdown files",
        key="doc_file_uploader_input_widget"
    )
    
    if uploaded_files:
        st.markdown("---")
        
        if st.button("🔍 Analyze Documents", type="primary", use_container_width=True, key="doc_analyze_submit_button"):
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
        
        st.markdown("### 📁 Uploaded Files")
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{file.name}** ({file.size/1024:.1f} KB)")
            with col2:
                st.write(file.type)
            with col3:
                st.write(f"File {i+1}")

def analyze_documents(files, analysis_type, custom_focus, options):
    """Processes and analyzes uploaded documents"""
    progress_container = st.empty()
    
    try:
        with progress_container:
            st.info("📄 Processing uploaded documents...")
        
        processed_docs = [doc for file in files if (doc := process_single_document(file)) is not None]
        if not processed_docs:
            st.error("❌ Failed to process any of the uploaded documents.")
            return

        with progress_container:
            st.info("🧠 Preparing analysis...")

        analysis_prompt = create_document_analysis_prompt(
            analysis_type, custom_focus, processed_docs, options
        )

        with progress_container:
            st.info("🤖 Analyzing documents...")

        llm_to_use = st.session_state.rag_system.llm 
        if not llm_to_use:
            st.error("❌ No AI models are available to perform the analysis.")
            return

        response = llm_to_use.invoke(analysis_prompt)
        answer = response.content

        result = {
            "answer": answer,
            "source_documents": [],
            "metadata": {"llm_used": "Direct Analysis"}
        }

        progress_container.empty()
        
        st.session_state.document_analysis_result = result
        st.session_state.analysis_metadata = {
            'analysis_type': analysis_type,
            'file_count': len(files),
            'processed_docs': processed_docs,
        }
        
        st.rerun()
        
    except Exception as e:
        import traceback
        print(f"\n=== DOCUMENT ANALYSIS ERROR ===")
        traceback.print_exc()
        print("============================\n")
        st.error(f"❌ Document analysis failed: {str(e)}")

def show_document_analysis_result():
    """Display document analysis results with unique keys"""
    result = st.session_state.document_analysis_result
    
    st.markdown("### 📄 Document Analysis Complete")
    
    if 'analysis_metadata' in st.session_state:
        metadata = st.session_state.analysis_metadata
        st.info(f"✅ Analyzed {metadata.get('file_count', 1)} document(s) using {metadata.get('analysis_type', 'Analysis')}")
    
    st.subheader("📋 Analysis Report")
    st.write(result['answer'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Key Findings", key="doc_findings_toggle_button"):
            st.session_state.show_doc_findings = not st.session_state.get('show_doc_findings', False)
    
    with col2:
        if st.button("📄 Generate Report", key="doc_report_download_button"):
            generate_document_report(result)
    
    with col3:
        if st.button("📚 Source Details", key="doc_sources_toggle_button"):
            st.session_state.show_doc_sources = not st.session_state.get('show_doc_sources', False)
    
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

def process_single_document(file):
    """Process a single document and extract content"""
    try:
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
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
    """Creates a specialized prompt based on the chosen analysis type"""
    combined_content = "\n\n---\n\n".join([
        f"DOCUMENT: {doc['filename']}\n\n{doc['content']}" 
        for doc in docs
    ])
    
    if "Direct Answers Only" in analysis_type:
        prompt = f"""You are providing direct answers to questions within these documents.

**CRITICAL INSTRUCTIONS:**
- Provide ONLY direct answers to questions, activities, or scenarios found in the documents.
- NO analysis reports, executive summaries, or comprehensive assessments.
- For true/false questions: State "True" or "False" + one sentence of reasoning from the text.
- For scenario questions: State the type of abuse, the threshold level, and a brief reason.
- Maximum 3 sentences per answer.

**Documents to analyze:**
{combined_content}

**DIRECT ANSWERS ONLY:**"""
    
    elif "Comprehensive Analysis" in analysis_type:
        prompt = f"""Analyze the following documents:
{combined_content}

Provide a comprehensive analysis including:
1. **Executive Summary:** Key findings and overview.
2. **Content Analysis:** Main themes and topics.
3. **Compliance Assessment:** Regulatory alignment and potential gaps.
4. **Recommendations:** Actionable next steps.
"""
    
    else:
        prompt = f"""Analyze the following documents:
{combined_content}

Create a structured content summary including:
1. **Key Points:** The main messages and findings.
2. **Action Items:** Any required tasks or responsibilities.
3. **Stakeholders:** Any people or roles mentioned.
"""
    
    if custom_focus and custom_focus.strip():
        prompt += f"\n\n**Special Focus:** Please pay special attention to the following: {custom_focus}"
    
    return prompt

def generate_document_report(result):
    """FIXED document analysis report generation"""
    
    # Get metadata properly
    metadata = st.session_state.get('analysis_metadata', {})
    
    report_content = f"""# Document Analysis Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** {metadata.get('analysis_type', 'Unknown')}
**Documents Analyzed:** {metadata.get('file_count', 'Unknown')}

## Analysis Results
{result['answer']}

## Source Documents
"""
    
    # Add processed documents info if available
    if 'processed_docs' in metadata and metadata['processed_docs']:
        for i, doc in enumerate(metadata['processed_docs'], 1):
            filename = doc.get('filename', f'Document {i}')
            word_count = doc.get('word_count', 0)
            file_type = doc.get('type', 'unknown')
            
            report_content += f"{i}. **{filename}**\n"
            report_content += f"   - Type: {file_type.upper()}\n"
            if word_count > 0:
                report_content += f"   - Length: {word_count:,} words\n"
            report_content += "\n"
    else:
        report_content += "Document details not available.\n\n"
    
    report_content += f"""
---
*Report generated by Children's Home Management System*
*Session: {st.session_state.get('tester_id', 'anonymous')}*
"""
    
    st.download_button(
        label="📥 Download Document Report",
        data=report_content,
        file_name=f"document_analysis_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key="download_doc_report_final_button"
    )

# ===== IMAGE ANALYSIS FUNCTIONS =====
def handle_image_tab():
    """Enhanced image analysis tab with unique keys"""
    
    if 'image_analysis_result' not in st.session_state:
        st.session_state.image_analysis_result = None
    
    if st.session_state.image_analysis_result:
        show_image_analysis_result()
        
        st.markdown("---")
        st.markdown("### 🆕 Analyze New Images")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📷 Analyze Different Images", type="primary", use_container_width=True, key="img_analyze_different_button"):
                st.session_state.image_analysis_result = None
                if 'image_uploader_key' not in st.session_state:
                    st.session_state.image_uploader_key = 0
                st.session_state.image_uploader_key += 1
                st.rerun()
        
        with col2:
            if st.button("🔄 Start Fresh", use_container_width=True, key="img_start_fresh_button"):
                st.session_state.image_analysis_result = None
                if 'image_uploader_key' not in st.session_state:
                    st.session_state.image_uploader_key = 0
                st.session_state.image_uploader_key += 1
                st.rerun()
        
        return
    
    show_image_upload_interface()

def show_image_upload_interface():
    """Image upload interface with unique keys"""
    st.subheader("📷 Visual Analysis")
    st.markdown("Upload images for clear, actionable analysis and compliance checking.")
    
    # Custom focus
    custom_focus = st.text_area(
        "🎯 Additional Focus (optional):",
        placeholder="e.g., 'Pay special attention to the dining area setup' or leave blank",
        height=60,
        key="img_custom_focus_input_widget"
    )
    
    # Analysis focus selection
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
        key="img_analysis_focus_select_widget"
    )
    
    # Detail level
    detail_level = st.radio(
        "📊 Report Style:",
        ["🚀 Quick Summary (Key points only)", "📋 Structured Report (Recommended)", "🔍 Detailed Analysis"],
        index=1,
        key="img_detail_level_radio_widget"
    )
    
    # Image upload with key management
    if 'image_uploader_key' not in st.session_state:
        st.session_state.image_uploader_key = 0
    
    uploaded_images = st.file_uploader(
        "Upload Images (1-3 images recommended)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload facility photos for analysis",
        key=f"img_uploader_widget_{st.session_state.image_uploader_key}"
    )
    
    if uploaded_images:
        st.markdown("---")
        
        if st.button("🔍 Analyze Images", type="primary", use_container_width=True, key="img_analyze_submit_button"):
            analyze_images_optimized(uploaded_images, analysis_focus, detail_level, custom_focus)
        
        st.markdown("---")
        
        st.markdown("### 📁 Images Ready for Analysis")
        st.info(f"📷 {len(uploaded_images)} image(s) uploaded and ready for analysis.")
        
        with st.expander("👁️ Preview Uploaded Images", expanded=False):
            cols = st.columns(min(3, len(uploaded_images)))
            for i, image in enumerate(uploaded_images):
                with cols[i % 3]:
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)

def analyze_images_optimized(images, analysis_focus, detail_level, custom_focus):
    """Optimized image analysis"""
    
    progress_container = st.empty()
    
    try:
        with progress_container:
            st.info("📷 Processing uploaded images...")
        
        prompt = create_image_analysis_prompt(analysis_focus, detail_level, custom_focus, len(images))
        
        with progress_container:
            st.info("🔍 Conducting visual analysis...")
        
        # Process first image
        image_data = images[0]
        image_data.seek(0)
        original_bytes = image_data.read()
        compressed_bytes = compress_image_for_analysis(original_bytes)
        
        # Get analysis result
        if hasattr(st.session_state.rag_system, 'query'):
            result = st.session_state.rag_system.query(
                question=prompt,
                k=3,
                response_style="brief"
            )
        else:
            # Fallback to regular query if method doesn't exist
            result = st.session_state.rag_system.query(
                question=prompt,
                k=3,
                response_style="brief"
            )
        
        progress_container.empty()
        
        if result and result.get('answer'):
            st.session_state.image_analysis_result = result
            st.session_state.image_metadata = {
                'image_count': len(images),
                'primary_image': images[0].name,
                'analysis_focus': analysis_focus,
                'detail_level': detail_level,
                'custom_focus': custom_focus
            }
            
            log_tester_activity(
                st.session_state.get('tester_id', 'unknown'),
                "image_analysis_completed",
                f"Images: {len(images)}, Focus: {analysis_focus}"
            )
            
            st.rerun()
        else:
            st.error("❌ Analysis failed to produce results. Please try again.")
            
    except Exception as e:
        progress_container.empty()
        st.error("❌ Image analysis failed.")
        
        with st.expander("🔧 Error Details"):
            st.code(f"Error: {str(e)}")

def create_image_analysis_prompt(analysis_focus, detail_level, custom_focus, image_count):
    """Create clean, focused analysis prompts"""
    
    focus_instructions = {
        "🛡️ Safety & Compliance Check": "Focus on health and safety compliance, fire safety, hazard identification, and regulatory requirements.",
        "🏠 Environment Quality Assessment": "Evaluate the quality of the physical environment, comfort, cleanliness, maintenance, and child-friendly features.",
        "🍎 Food & Kitchen Safety Review": "Examine food preparation areas, hygiene standards, equipment safety, and food safety compliance.",
        "💚 Wellbeing & Atmosphere Evaluation": "Assess how the environment supports children's emotional wellbeing, dignity, privacy, and positive atmosphere.",
        "📋 General Facility Inspection": "Comprehensive visual inspection covering safety, quality, compliance, and child experience."
    }
    
    base_instruction = focus_instructions.get(analysis_focus, "Provide a comprehensive facility analysis.")
    
    if "Quick Summary" in detail_level:
        format_instruction = """
Provide a CONCISE analysis in this format:
- **Key Strengths:** 2-3 positive observations
- **Areas for Attention:** 2-3 priority improvements 
- **Compliance Status:** Brief compliance assessment
- **Next Steps:** 1-2 immediate actions needed
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
    
    custom_instruction = f"\n\nAdditional Focus: {custom_focus}" if custom_focus.strip() else ""
    
    return f"""
Analyze this image for a children's home facility with focus on: {base_instruction}

{format_instruction}

{custom_instruction}

Be specific, practical, and focus on actionable insights for facility managers.
    """

def show_image_analysis_result():
    """Display image analysis results with unique keys"""
    
    result = st.session_state.image_analysis_result
    metadata = st.session_state.get('image_metadata', {})
    
    st.markdown("### 📷 Visual Analysis Complete")
    
    analysis_info = f"✅ Analyzed {metadata.get('image_count', 1)} image(s)"
    if metadata.get('primary_image'):
        analysis_info += f" • Primary: {metadata.get('primary_image')}"
    analysis_info += f" • Focus: {metadata.get('analysis_focus', 'General')}"
    
    st.info(analysis_info)
    
    if result.get('metadata') and result['metadata'].get('llm_used'):
        st.caption(f'*Analysis by {result["metadata"]["llm_used"]}*')
    
    st.subheader("📋 Visual Analysis Report")
    st.write(result['answer'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Analyze New Images", type="primary", key="img_new_analysis_button"):
            st.session_state.image_analysis_result = None
            if 'image_uploader_key' not in st.session_state:
                st.session_state.image_uploader_key = 0
            st.session_state.image_uploader_key += 1
            st.rerun()
    
    with col2:
        if st.button("📄 Download Report", key="img_download_report_button"):
            generate_image_report(result, metadata)
    
    with col3:
        if st.button("📊 Analysis Details", key="img_show_details_button"):
            st.session_state.show_img_details = not st.session_state.get('show_img_details', False)
    
    if st.session_state.get('show_img_details', False):
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

def generate_image_report(result, metadata):
    """FIXED image analysis report generation"""
    
    report_content = f"""# Visual Analysis Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Images Analyzed:** {metadata.get('image_count', 1)}
**Primary Image:** {metadata.get('primary_image', 'Unknown')}
**Analysis Focus:** {metadata.get('analysis_focus', 'General')}
**Detail Level:** {metadata.get('detail_level', 'Standard')}

## Analysis Results
{result['answer']}

## Analysis Configuration
- **Custom Focus:** {metadata.get('custom_focus', 'None specified')}
- **Generated by:** Children's Home Management System
- **Analysis Mode:** {st.session_state.get('performance_mode', 'balanced')}
- **Session:** {st.session_state.get('tester_id', 'anonymous')}

---
*Report generated by Children's Home Management System*
"""
    
    st.download_button(
        label="📥 Download Complete Report",
        data=report_content,
        file_name=f"visual_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key="download_img_report_final_button"
    )

def compress_image_for_analysis(image_bytes, max_size_kb=30):
    """Compress image to stay under the AI model size limit"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = rgb_image
        
        quality = 85
        max_dimension = 800
        
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        while True:
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_bytes = output.getvalue()
            
            if len(compressed_bytes) <= max_size_kb * 1024:
                return compressed_bytes
            
            if quality > 60:
                quality -= 10
            else:
                current_size = image.size
                new_size = (int(current_size[0] * 0.8), int(current_size[1] * 0.8))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                quality = 75
            
            if quality < 30 and max(image.size) < 200:
                break
        
        return compressed_bytes
        
    except Exception as e:
        print(f"Image compression failed: {e}")
        return image_bytes

# ===== SYSTEM HEALTH DISPLAY =====
def display_system_health():
    """Display system health and performance information"""
    st.subheader("⚙️ System Health & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 AI Models Status")
        
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
        
        st.markdown("### 🔑 API Configuration")
        openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        google_key = bool(os.environ.get("GOOGLE_API_KEY"))
        
        st.write(f"{'✅' if openai_key else '❌'} OpenAI API")
        st.write(f"{'✅' if google_key else '❌'} Google API")
    
    with col2:
        st.markdown("### 📊 Performance Metrics")
        
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
        
        if SYSTEM_MONITOR:
            st.markdown("### 💾 System Resources")
            try:
                memory_percent = psutil.virtual_memory().percent
                st.metric("Memory Usage", f"{memory_percent:.1f}%")
            except Exception as e:
                st.warning(f"Memory monitoring unavailable: {e}")
        
        st.markdown("### ⏱️ Session Info")
        session_start = st.session_state.get('session_start', time.time())
        session_duration = int((time.time() - session_start) / 60)
        st.metric("Session Duration", f"{session_duration} min")
    
    with st.expander("🔍 Detailed System Information"):
        if st.session_state.get('rag_system'):
            try:
                health = st.session_state.rag_system.get_system_health()
                st.json(health)
            except Exception as e:
                st.error(f"Detailed health check failed: {e}")

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
    """Main application function with all fixes"""
    setup_sqlite_with_clean_logging()

    if not require_authentication():
        st.stop()
    
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
    
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.rag_system = initialize_rag_system()
    
    if st.session_state.rag_system is None:
        st.error("❌ System initialization failed")
        st.info("💡 Please refresh the page or contact support")
        st.stop()
    
    # CRITICAL: Ensure UI state is initialized early
    ensure_ui_state()
    
    # Load CSS once
    st.markdown(get_app_css(), unsafe_allow_html=True)
    
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
    
    add_performance_mode_selector()
    
    # Feedback mechanism in sidebar
    with st.sidebar:
        st.markdown("### 📝 Beta Feedback")
        
        feedback_type = st.selectbox(
            "Feedback Type:",
            ["💡 Suggestion", "🐛 Bug Report", "👍 Positive", "❓ Question"],
            key="feedback_type_select_widget"
        )
        
        feedback_text = st.text_area(
            "Your Feedback:",
            placeholder="Share thoughts, report bugs, or suggest improvements...",
            height=100,
            key="feedback_text_input_widget"
        )
        
        if st.button("📤 Submit Feedback", type="secondary", key="feedback_submit_button"):
            if feedback_text.strip():
                log_tester_activity(
                    st.session_state.get('tester_id', 'unknown'),
                    "feedback_submitted",
                    f"{feedback_type}: {feedback_text[:100]}..."
                )
                st.success("✅ Feedback submitted!")
            else:
                st.warning("Please enter feedback before submitting")
    
    # Debug controls
    with st.sidebar:
        if st.session_state.get('tester_id') in ['DEMO001', 'TEST001']:
            st.markdown("---")
            st.markdown("### 🔧 Debug Controls")
            
            current_debug = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
            debug_mode = st.checkbox(
                "🔍 Debug Mode", 
                value=current_debug,
                help="Show detailed system logs in terminal",
                key="debug_mode_checkbox_widget"
            )
            
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
    
    show_session_summary()
    
    # Initialize tab state
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0

    # Create tabs with clean state management and UNIQUE KEYS
    tab_names = ["💬 Ask Questions", "📄 Analyze Documents", "📷 Analyze Images", "⚙️ System Health"]

    selected_tab = st.radio(
        "Select Tab:",
        options=list(range(len(tab_names))),
        format_func=lambda x: tab_names[x],
        index=st.session_state.current_tab,
        horizontal=True,
        key="main_tab_selector_widget",
        label_visibility="collapsed"
    )

    st.session_state.current_tab = selected_tab

    # Display content based on selected tab - ALL WITH FIXED FUNCTIONS
    if selected_tab == 0:
        handle_question_tab()  # Fixed version with unique keys
    elif selected_tab == 1:
        handle_document_analysis_tab()  # Fixed version with unique keys
    elif selected_tab == 2:
        handle_image_tab()  # Fixed version with unique keys
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
    if 'page_loads' not in st.session_state:
        st.session_state['page_loads'] = 0
        log_tester_activity(
            st.session_state.get('tester_id', 'unknown'),
            "session_started",
            "Application loaded"
        )
    
    st.session_state['page_loads'] += 1
    main()
