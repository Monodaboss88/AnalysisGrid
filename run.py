#!/usr/bin/env python3
"""
MTF Auction Scanner - Web Application Launcher
==============================================

This repo contains multiple historical versions of the UI/server.

This launcher will prefer the *Unified* server/UI (unified_server.py + unified_ui.html).
If those files are missing, it will fall back to the older Trading Terminal (server.py + index.html).

Usage:
    python run.py

Then open http://localhost:8000 in your browser.
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path


def check_dependencies():
    """Check and install required dependencies"""
    required = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'requests']
    # Optional data providers (unified server uses finnhub-python -> import name: finnhub)
    optional = ['finnhub', 'yfinance']
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing required packages: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            *missing, '--quiet'
        ])
    
    # Check optional
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"✅ {pkg} available - live market data enabled")
        except ImportError:
            print(f"⚠️  {pkg} not installed - using demo data")
            print(f"   Install with: pip install {pkg}")


def main():
    print("=" * 60)
    print("MTF AUCTION SCANNER - Launcher")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    print()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Prefer unified stack
    if Path('unified_server.py').exists() and Path('unified_ui.html').exists():
        server_app = 'unified_server:app'
        required_files = ['unified_server.py', 'unified_ui.html', 'chart_input_analyzer.py', 'watchlist_manager.py']
        ui_name = 'Unified UI'
    else:
        server_app = 'server:app'
        required_files = ['server.py', 'index.html', 'mtf_auction_scanner.py', 'scanner_config.py']
        ui_name = 'Trading Terminal (legacy)'

    # Check required files exist
    for f in required_files:
        if not Path(f).exists():
            print(f"❌ Missing required file: {f}")
            sys.exit(1)

    print(f"✅ All required files found ({ui_name})")
    print()
    print("Starting server...")
    print("-" * 60)
    print("🌐 Open http://localhost:8000 in your browser")
    print("📊 Press Ctrl+C to stop the server")
    print("-" * 60)
    print()
    
    # Open browser after short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:8000')
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run uvicorn
    try:
        import uvicorn
        uvicorn.run(server_app, host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


if __name__ == "__main__":
    main()
