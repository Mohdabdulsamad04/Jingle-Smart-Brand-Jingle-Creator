"""
Streamlit Applet Runner for AI Studio
"""
import sys
import subprocess

if __name__ == '__main__':
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=3000",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]
    subprocess.run(cmd)
