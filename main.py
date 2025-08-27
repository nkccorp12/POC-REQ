#!/usr/bin/env python3
"""
Main entry point for Vercel deployment
Runs the Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()