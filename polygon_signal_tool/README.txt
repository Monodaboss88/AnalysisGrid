POLYGON SIGNAL TOOL — Setup & Usage
=====================================

REQUIREMENTS:
  Python 3.8+
  pip install requests jinja2

SETUP:
  1. Set your Polygon.io API key:
     export POLYGON_API_KEY="your_key_here"

  2. Or pass it directly:
     python polygon_signal.py SMH --api-key YOUR_KEY

USAGE:
  python polygon_signal.py SMH              # Generate SMH dashboard (1 year lookback)
  python polygon_signal.py AAPL --days 180  # Any ticker, custom lookback
  python polygon_signal.py TSLA --no-open   # Don't auto-open browser

  The tool will:
  - Fetch daily OHLCV from Polygon.io
  - Cache data locally in /cache (only fetches new days on repeat runs)
  - Run full probability analysis (calls, puts, straddle, scalp ranges)
  - Generate an HTML dashboard with signal recommendation
  - Auto-open the dashboard in your browser

FILES:
  polygon_signal.py   - Main CLI script
  signal_analyzer.py  - Core analysis engine
  signal_config.py    - Configuration
  templates/dashboard.html - Dashboard template
  cache/              - Local data cache (auto-populated)

NOTE: SMH data through Feb 19, 2026 is pre-cached in cache/SMH.json
