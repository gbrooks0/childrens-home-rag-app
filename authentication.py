# authentication.py - Create this as a separate file
import streamlit as st
import hashlib
import time
import datetime
import json

def get_tester_credentials():
    """Get tester credentials from Streamlit secrets or environment"""
    try:
        # Try Streamlit Cloud secrets first
        return st.secrets.get("testers", {})
    except:
        # Fallback for local development
        return {
            "DEMO001": {
                "password": "DemoAccess2024!",
                "name": "Demo Tester",
                "email": "demo@example.com",
                "expires": "2024-12-31"
            },
            "TEST001": {
                "password": "TestAccess456!",
                "name": "Beta Tester 1",
                "email": "tester1@example.com",
                "expires": "2024-12-31"
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
    
    if session_duration > 7200:  # 2 hours in seconds
        st.session_state['authenticated_tester'] = False
        return False
    
    return True

def log_tester_activity(tester_id, action, details=""):
    """Log tester activity for monitoring"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "tester_id": tester_id,
        "action": action,
        "details": details
    }
    
    # In a real deployment, you might want to store this in a database
    # For now, we'll just use session state for simple tracking
    if 'activity_log' not in st.session_state:
        st.session_state['activity_log'] = []
    
    st.session_state['activity_log'].append(log_entry)
    
    # Also print to console for server-side logging
    print(f"TESTER LOG: {timestamp} - {tester_id} - {action} - {details}")

def advanced_authentication():
    """
    Advanced authentication system for testers
    Returns True if authenticated, False otherwise
    """
    
    # Check if already authenticated and session is valid
    if check_session_valid():
        # Show session info in sidebar
        show_session_info()
        return True
    
    # Clear any invalid session data
    st.session_state['authenticated_tester'] = False
    
    # Show authentication interface
    show_login_interface()
    return False

def show_session_info():
    """Show session information in sidebar for authenticated users"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 Session Info")
        
        tester_info = st.session_state.get('tester_info', {})
        st.write(f"**Tester:** {tester_info.get('name', 'Unknown')}")
        st.write(f"**ID:** {st.session_state.get('tester_id', 'N/A')}")
        
        # Session timer
        session_start = st.session_state.get('session_start', time.time())
        elapsed = time.time() - session_start
        remaining = max(0, 7200 - elapsed)  # 2 hours
        
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

def show_login_interface():
    """Show the login interface for testers"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .login-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    .login-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .login-form {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="login-header">
        <h1>🏠 Children's Home Management System</h1>
        <h3>Beta Testing Portal</h3>
        <p>Secure access for authorized testers only</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.markdown("### 🔐 Tester Login")
        
        # Login form
        tester_id = st.text_input(
            "Tester ID:",
            placeholder="e.g., DEMO001",
            help="Unique ID provided by administrator"
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
                st.success("✅ Authentication successful! Initializing system...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please check your Tester ID and Access Code.")
                log_tester_activity(tester_id or "unknown", "failed_login", f"Invalid credentials")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Testing Information")
        
        st.markdown("""
        <div class="info-box">
        <strong>🎯 Testing Objectives:</strong><br>
        • Test all main functionality<br>
        • Evaluate user experience<br>
        • Report bugs or issues<br>
        • Provide improvement suggestions
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **⏱️ Session Details:**
        - Duration: 2 hours per session
        - Auto-logout: After inactivity
        - Multiple sessions: Allowed
        
        **🔒 Security Notes:**
        - Do not share your credentials
        - Report any security concerns
        - Session activity is logged
        
        **📞 Support:**
        - Technical issues: Contact administrator
        - Feedback: Use in-app feedback form
        """)
        
        # New tester registration info
        with st.expander("🆕 Need Testing Access?"):
            st.markdown("""
            **To become a beta tester:**
            
            1. **Contact:** [your-email@organization.com]
            2. **Include:** 
               - Your name and organization
               - Testing experience
               - Areas of interest
            3. **Receive:** Unique Tester ID and Access Code
            
            **Current testing focus:**
            - Core functionality validation
            - User interface feedback
            - Performance assessment
            - Security testing
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with additional info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    🔐 This is a secure testing environment. All activity is monitored and logged.<br>
    By accessing this system, you agree to use it responsibly and provide constructive feedback.
    </div>
    """, unsafe_allow_html=True)

def validate_and_authenticate(tester_id, access_code):
    """Validate credentials and set up authenticated session"""
    
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
    
    # Check if access hasn't expired
    expiry_date = tester_info.get("expires", "2099-12-31")
    try:
        expiry = datetime.datetime.strptime(expiry_date, "%Y-%m-%d").date()
        if datetime.date.today() > expiry:
            return False
    except:
        pass  # If date parsing fails, allow access
    
    # Set up authenticated session
    st.session_state['authenticated_tester'] = True
    st.session_state['tester_id'] = tester_id
    st.session_state['tester_info'] = tester_info
    st.session_state['session_start'] = time.time()
    
    # Log successful authentication
    log_tester_activity(tester_id, "successful_login", f"User: {tester_info.get('name', 'Unknown')}")
    
    return True

def require_authentication():
    """
    Main function to add to your app.py
    Call this at the very beginning of your app
    """
    if not advanced_authentication():
        st.stop()
    
    # Optional: Log page access
    current_page = st.get_option("browser.gatherUsageStats") # or determine page somehow
    log_tester_activity(
        st.session_state.get('tester_id', 'unknown'),
        "page_access",
        f"Accessing main application"
    )