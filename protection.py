"""
Comprehensive protection system against code inspection
Blocks right-click, dev tools, view source, and more
"""

import streamlit as st

def inject_protection():
    """Inject comprehensive protection JavaScript and CSS"""
    protection_code = """
    <script>
    // === COMPREHENSIVE PROTECTION SYSTEM ===
    
    // 1. DISABLE RIGHT CLICK COMPLETELY
    document.addEventListener('contextmenu', function(e) {
        e.preventDefault();
        e.stopPropagation();
        return false;
    }, true);
    
    // 2. DISABLE ALL DEVELOPER TOOLS SHORTCUTS
    document.addEventListener('keydown', function(e) {
        // F12 - Developer Tools
        if (e.keyCode === 123) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+Shift+I - Developer Tools
        if (e.ctrlKey && e.shiftKey && e.keyCode === 73) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+Shift+J - Console
        if (e.ctrlKey && e.shiftKey && e.keyCode === 74) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+U - View Source
        if (e.ctrlKey && e.keyCode === 85) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+Shift+C - Inspect Element
        if (e.ctrlKey && e.shiftKey && e.keyCode === 67) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+S - Save Page
        if (e.ctrlKey && e.keyCode === 83) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+A - Select All
        if (e.ctrlKey && e.keyCode === 65) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+P - Print
        if (e.ctrlKey && e.keyCode === 80) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+C - Copy
        if (e.ctrlKey && e.keyCode === 67) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+V - Paste  
        if (e.ctrlKey && e.keyCode === 86) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
        
        // Ctrl+X - Cut
        if (e.ctrlKey && e.keyCode === 88) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
    }, true);
    
    // 3. DISABLE TEXT SELECTION
    document.addEventListener('selectstart', function(e) {
        e.preventDefault();
        return false;
    }, true);
    
    document.addEventListener('mousedown', function(e) {
        if (e.detail > 1) { // Prevent multiple clicks
            e.preventDefault();
            return false;
        }
    }, true);
    
    // 4. DISABLE DRAG AND DROP
    document.addEventListener('dragstart', function(e) {
        e.preventDefault();
        return false;
    }, true);
    
    // 5. DISABLE PRINT SCREEN (attempt)
    document.addEventListener('keyup', function(e) {
        if (e.keyCode === 44) {
            e.preventDefault();
            return false;
        }
    }, true);
    
    // 6. DETECT DEVELOPER TOOLS (basic detection)
    let devtools = {open: false, orientation: null};
    const threshold = 160;
    
    setInterval(function() {
        if (window.outerHeight - window.innerHeight > threshold || 
            window.outerWidth - window.innerWidth > threshold) {
            if (!devtools.open) {
                devtools.open = true;
                // Redirect or alert when dev tools detected
                console.clear();
                document.body.innerHTML = '<div style="text-align:center;margin-top:200px;font-size:24px;">⚠️ Access Denied ⚠️</div>';
            }
        } else {
            devtools.open = false;
        }
    }, 500);
    
    // 7. DISABLE MOUSE MIDDLE CLICK
    document.addEventListener('mousedown', function(e) {
        if (e.button === 1) { // Middle click
            e.preventDefault();
            return false;
        }
    }, true);
    
    // 8. OVERRIDE CONSOLE METHODS
    console.log = function() {};
    console.info = function() {};
    console.warn = function() {};
    console.error = function() {};
    console.clear = function() {};
    console.dir = function() {};
    console.dirxml = function() {};
    console.table = function() {};
    console.trace = function() {};
    console.group = function() {};
    console.groupCollapsed = function() {};
    console.groupEnd = function() {};
    console.time = function() {};
    console.timeEnd = function() {};
    console.profile = function() {};
    console.profileEnd = function() {};
    console.count = function() {};
    
    // 9. DISABLE IFRAME ACCESS (if any)
    try {
        if (window.top !== window.self) {
            window.top.location = window.self.location;
        }
    } catch(e) {}
    
    // 10. BLUR ON FOCUS LOSS (prevents screenshots when switching)
    window.addEventListener('blur', function() {
        document.body.style.filter = 'blur(5px)';
    });
    
    window.addEventListener('focus', function() {
        document.body.style.filter = 'none';
    });
    
    </script>
    
    <style>
    /* === CSS PROTECTION === */
    
    /* Disable text selection */
    * {
        -webkit-user-select: none !important;
        -moz-user-select: none !important;
        -ms-user-select: none !important;
        user-select: none !important;
        -webkit-touch-callout: none !important;
        -webkit-tap-highlight-color: transparent !important;
    }
    
    /* Disable drag and drop */
    * {
        -webkit-user-drag: none !important;
        -khtml-user-drag: none !important;
        -moz-user-drag: none !important;
        -o-user-drag: none !important;
        user-drag: none !important;
        draggable: false !important;
    }
    
    /* Hide potential debug info */
    .debug, .console, .inspector {
        display: none !important;
    }
    
    /* Prevent highlighting */
    *::selection {
        background: transparent !important;
    }
    
    *::-moz-selection {
        background: transparent !important;
    }
    
    /* Disable outline */
    * {
        outline: none !important;
    }
    
    /* Prevent zoom */
    html {
        -ms-touch-action: pan-x pan-y !important;
        touch-action: pan-x pan-y !important;
    }
    
    /* Additional protection */
    body {
        pointer-events: auto !important;
    }
    
    /* Disable image dragging */
    img {
        pointer-events: none !important;
        -webkit-user-drag: none !important;
    }
    
    /* Protect buttons and interactive elements */
    button, input, textarea, select {
        -webkit-user-select: none !important;
        user-select: none !important;
    }
    
    </style>
    """
    
    st.markdown(protection_code, unsafe_allow_html=True)

def add_security_headers():
    """Add security-related meta tags and headers"""
    security_headers = """
    <meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https:">
    <meta http-equiv="X-Content-Type-Options" content="nosniff">
    <meta http-equiv="X-Frame-Options" content="DENY">
    <meta http-equiv="X-XSS-Protection" content="1; mode=block">
    <meta name="referrer" content="no-referrer">
    """
    st.markdown(security_headers, unsafe_allow_html=True)

def full_protection():
    """Apply full protection suite"""
    add_security_headers()
    inject_protection()