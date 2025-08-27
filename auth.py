"""
Simple authentication system for OSME
"""

import streamlit as st
import hashlib
import base64

# Simple credentials - change these!
VALID_USERS = {
    "requisimus": "711b2dd6bcbbea459703a43fbd56cb64",  # symrise
    "user": "ee11cbb19052e40b07aac0ca060c23ee",         # user  
    "osme": "7c6a180b36896a0a8c02787eeafb0e4c"          # osme
}

def hash_password(password):
    """Hash password with MD5"""
    return hashlib.md5(password.encode()).hexdigest()

def check_password(username, password):
    """Check if username/password combination is valid"""
    if username in VALID_USERS:
        return VALID_USERS[username] == hash_password(password)
    return False

def get_base64_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def login_form():
    """Display login form"""
    # White bar with logo (same as main app)
    logo_path = "public/logo.png"
    logo_base64 = get_base64_image(logo_path)
    
    st.markdown("""
    <div style="background-color: white; padding: 10px; margin: 0 0 20px 0; text-align: center;">
        <img src="data:image/png;base64,{}" style="max-height: 60px; width: auto;">
    </div>
    """.format(logo_base64), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username", placeholder="Enter username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
        
        if st.button("Login", use_container_width=True, type="primary"):
            if check_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

def is_authenticated():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def require_auth():
    """Decorator to require authentication"""
    if not is_authenticated():
        login_form()
        st.stop()

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

def get_current_user():
    """Get current authenticated user"""
    return st.session_state.get("username", None)