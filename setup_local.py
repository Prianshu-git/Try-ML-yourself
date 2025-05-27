#!/usr/bin/env python3
"""
Setup script for running the ML Algorithm Demonstrator locally
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        "streamlit==1.29.0",
        "numpy==1.24.3", 
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "plotly==5.17.0"
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    return True

def create_streamlit_config():
    """Create Streamlit configuration"""
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    config_content = """[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
base = "light"
"""
    
    with open(config_file, "w") as f:
        f.write(config_content)
    print("✓ Created Streamlit configuration")

def main():
    print("🤖 Setting up ML Algorithm Demonstrator for local development")
    print("=" * 60)
    
    # Install packages
    if not install_requirements():
        print("❌ Installation failed. Please check your Python environment.")
        return
    
    # Create config
    create_streamlit_config()
    
    print("\n🎉 Setup complete!")
    print("\nTo run the app locally:")
    print("1. Open terminal in this directory")
    print("2. Run: streamlit run app.py")
    print("3. Open browser to: http://localhost:8501")
    print("\nFor Vercel deployment:")
    print("1. Install Vercel CLI: npm install -g vercel")
    print("2. Run: vercel --platform-version 2")

if __name__ == "__main__":
    main()