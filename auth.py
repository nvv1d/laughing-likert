# auth.py
import streamlit as st
import hashlib
import hmac
import time
from datetime import datetime, timedelta

class SecurityConfig:
    def __init__(self):
        self.default_username = "Admin"
        self.stored_password_hash = "6315a0a0f67490999b3657c395d229fa5098a3dfd403bc6284e45e34ef5bcb07"
        self.max_login_attempts = 3
        self.lockout_duration_minutes = 15
        self.session_timeout_minutes = 30

    def simple_hash(self, password):
        """Simple hash function"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password):
        """Verify password against stored hash"""
        return self.simple_hash(password) == self.stored_password_hash

# Initialize security config
security = SecurityConfig()

def init_auth_session_state():
    """Initialize all authentication-related session state variables"""
    defaults = {
        'logged_in': False,
        'login_attempts': 0,
        'lockout_until': None,
        'lockout_count': 0,
        'show_password': False,
        'last_activity': datetime.now(),
        'username': ''
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_login_css():
    """Return the original CSS styling for the login page"""
    return """
    <style>
    .login-page {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .login-container {
        max-width: 300px;
        width: 100%;
        text-align: center;
    }

    .stButton > button {
        width: 100%;
        background-color: #4051B5 !important;
        color: white !important;
    }

    .login-footer {
        margin-top: 20px;
        font-size: 12px;
        color: #888;
    }

    .error-message {
        background-color: #FFEBEE;
        color: #D32F2F;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 14px;
    }

    .locked-message {
        background-color: #FFF8E1;
        color: #F57F17;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 14px;
    }
    </style>
    """

def is_locked_out():
    """Check if user is currently locked out"""
    if st.session_state.lockout_until is not None:
        if datetime.now() < st.session_state.lockout_until:
            remaining = st.session_state.lockout_until - datetime.now()
            minutes = remaining.seconds // 60
            seconds = remaining.seconds % 60
            return True, f"{minutes} minutes and {seconds} seconds"
        else:
            st.session_state.lockout_until = None
            st.session_state.login_attempts = 0
            return False, ""
    return False, ""

def check_session_timeout():
    """Check if session has timed out"""
    if st.session_state.logged_in:
        time_since_activity = datetime.now() - st.session_state.last_activity
        if time_since_activity.total_seconds() > security.session_timeout_minutes * 60:
            st.session_state.logged_in = False
            st.session_state.username = ''
            return True
    return False

def process_login(username, password):
    """Handle login process with enhanced security"""
    # Check lockout status
    locked, time_remaining = is_locked_out()
    if locked:
        st.error(f"Account is temporarily locked. Please try again in {time_remaining}.")
        return False
    
    # Simulate processing time to prevent timing attacks
    time.sleep(0.5)
    
    # Verify credentials
    if username == security.default_username and security.verify_password(password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.login_attempts = 0
        st.session_state.lockout_count = 0
        st.session_state.last_activity = datetime.now()
        st.rerun()
        return True
    else:
        # Handle failed login
        st.session_state.login_attempts += 1
        attempts_left = security.max_login_attempts - st.session_state.login_attempts
        
        if attempts_left <= 0:
            # Lock account with exponential backoff
            st.session_state.lockout_count += 1
            lockout_duration = security.lockout_duration_minutes * (2 ** (st.session_state.lockout_count - 1))
            st.session_state.lockout_until = datetime.now() + timedelta(minutes=lockout_duration)
            st.error(f"Too many failed attempts. Your account is locked for {lockout_duration} minutes.")
        else:
            st.error(f"Invalid username or password. {attempts_left} attempts remaining.")
        
        return False

def render_login_page():
    """Render the complete login page using original styling"""
    st.markdown(get_login_css(), unsafe_allow_html=True)
    
    # Check for session timeout
    if check_session_timeout():
        st.warning("Session timed out. Please log in again.")
    
    # Create a simple login container
    st.markdown("""
    <div class="login-page">
        <div class="login-container">
    """, unsafe_allow_html=True)
    
    locked, time_remaining = is_locked_out()
    if locked:
        st.markdown(f"""
        <div class="locked-message">
            Account is temporarily locked. Please try again in {time_remaining}.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create the login form
        with st.form("login_form"):
            username = st.text_input("Username", value="Admin")
            password = st.text_input("Password", type="password")
            
            # Display error message if there were previous attempts
            if st.session_state.login_attempts > 0:
                attempts_left = security.max_login_attempts - st.session_state.login_attempts
                st.markdown(f"""
                <div class="error-message">
                    Invalid username or password. {attempts_left} attempts remaining.
                </div>
                """, unsafe_allow_html=True)
            
            submit = st.form_submit_button("Log In")
            
            if submit:
                process_login(username, password)
    
    # Simple footer
    st.markdown("""
        <div class="login-footer">
            © 2025 Analysis Tool
        </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def update_last_activity():
    """Update the last activity timestamp"""
    if st.session_state.logged_in:
        st.session_state.last_activity = datetime.now()

def logout():
    """Handle user logout"""
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.login_attempts = 0
    st.rerun()

def is_authenticated():
    """Check if user is currently authenticated"""
    return st.session_state.get('logged_in', False)

def get_current_username():
    """Get the current logged-in username"""
    return st.session_state.get('username', '')

def render_logout_button():
    """Render a logout button for the sidebar"""
    if st.button("Logout", key="logout_btn"):
        logout()
