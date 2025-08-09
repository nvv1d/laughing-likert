# auth.py
import streamlit as st
import hashlib
import hmac
import time
from datetime import datetime, timedelta

class SecurityConfig:
    def __init__(self):
        self.default_username = "Admin"
        self.stored_password_hash = "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92"
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
    """Return the CSS styling for the login page"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .login-page {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        margin: -1rem -1rem -1rem -1rem;
    }

    .login-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        max-width: 400px;
        width: 100%;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .login-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0 0 0.5rem 0;
    }

    .login-subtitle {
        text-align: center;
        font-size: 0.95rem;
        color: #6b7280;
        margin: 0 0 2rem 0;
    }

    .stTextInput input {
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        background-color: #f9fafb !important;
    }

    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        background-color: white !important;
    }

    .stButton button {
        width: 100% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        margin-top: 1rem !important;
    }

    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 25px -5px rgba(102, 126, 234, 0.4) !important;
    }

    .error-box {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #dc2626;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #dc2626;
        animation: shake 0.5s ease-in-out;
    }

    .warning-box {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        color: #d97706;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #d97706;
    }

    .success-box {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #059669;
    }

    .login-footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #6b7280;
    }

    .security-info {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1.5rem;
        font-size: 0.8rem;
        color: #1e40af;
    }

    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
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
            return True, f"{minutes}m {seconds}s"
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
        st.error(f"🔒 Account temporarily locked. Try again in {time_remaining}")
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
        st.success("✅ Login successful! Redirecting...")
        time.sleep(1)
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
            st.error(f"🚨 Too many failed attempts! Account locked for {lockout_duration} minutes")
        else:
            st.error(f"❌ Invalid credentials. {attempts_left} attempts remaining")
        
        return False

def render_login_page():
    """Render the complete login page"""
    st.markdown(get_login_css(), unsafe_allow_html=True)
    
    # Check for session timeout
    if check_session_timeout():
        st.warning("⏰ Session timed out. Please log in again.")
    
    st.markdown("""
    <div class="login-page">
        <div class="login-container">
            <h1 class="login-title">📊 Welcome Back</h1>
            <p class="login-subtitle">Likert Scale Pattern Analysis Tool</p>
    """, unsafe_allow_html=True)
    
    # Check lockout status
    locked, time_remaining = is_locked_out()
    
    if locked:
        st.markdown(f"""
        <div class="warning-box">
            🔒 <strong>Account Locked</strong><br>
            Too many failed login attempts. Please wait {time_remaining} before trying again.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", 
                                   value="Admin", 
                                   placeholder="Enter your username",
                                   help="Default username: Admin")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                password = st.text_input("🔐 Password", 
                                       type="password" if not st.session_state.show_password else "text",
                                       placeholder="Enter your password",
                                       help="Default password: 101066")
            with col2:
                if st.form_submit_button("👁️" if not st.session_state.show_password else "🙈", 
                                        help="Toggle password visibility"):
                    st.session_state.show_password = not st.session_state.show_password
                    st.rerun()
            
            # Remember failed attempts
            if st.session_state.login_attempts > 0:
                attempts_left = security.max_login_attempts - st.session_state.login_attempts
                st.markdown(f"""
                <div class="error-box">
                    ❌ <strong>Login Failed</strong><br>
                    Invalid username or password. {attempts_left} attempts remaining.
                </div>
                """, unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Sign In", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.error("⚠️ Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        process_login(username, password)
    
    # Security information
    st.markdown("""
    <div class="security-info">
        🔐 <strong>Security Features:</strong><br>
        • Secure password hashing with SHA-256<br>
        • Account lockout after failed attempts<br>
        • Session timeout protection<br>
        • Brute force attack prevention
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
        <div class="login-footer">
            <p>© 2025 Analysis Tool | Secure Login System</p>
            <small>Last updated: {current_time}</small>
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
    if st.button("🚪 Logout", key="logout_btn"):
        logout()
