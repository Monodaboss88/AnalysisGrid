# MTF Auction Scanner - Unified System

Complete multi-timeframe auction analysis system with real-time Finnhub integration.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Finnhub API key (get free at https://finnhub.io)
export FINNHUB_API_KEY=your_key_here

# Windows PowerShell:
#   $env:FINNHUB_API_KEY = "your_key_here"
# Windows cmd.exe:
#   set FINNHUB_API_KEY=your_key_here

# 3. Run the server
python unified_server.py

# 4. Open browser
http://localhost:8000
```

## 📦 What's Included

| File | Purpose |
|------|---------|
| `unified_server.py` | **Main server - RUN THIS** |
| `unified_ui.html` | Web interface |
| `finnhub_scanner.py` | Finnhub data integration |
| `chart_input_analyzer.py` | Manual chart input + alerts + trade tracker |
| `mtf_auction_scanner.py` | Core scanner logic |
| `watchlist_manager.py` | 13 watchlists, 218 symbols |

## 🎯 Features

### Scanner
- **Live Analysis**: Real-time data from Finnhub
- **Manual Input**: Enter your own chart values
- **Multi-Timeframe**: 30min, 1hr, 2hr, 4hr analysis
- **Confluence Scoring**: Combined MTF signal

### Watchlists (Built-in)
- Index ETFs (SPY, QQQ, IWM, DIA)
- Sector ETFs (XLF, XLE, XLK, etc.)
- Mag 7 Tech
- Dow 30
- Nasdaq 100
- S&P 500 Components
- Crypto-Related
- And more...

### Alerts
- Set price levels
- Break up/down triggers
- LONG/SHORT/EXIT actions

### Trade Tracker
- Log setups with R:R
- Track wins/losses
- Performance stats

## 📡 API Endpoints

### Scanner
- `GET /api/analyze/live/{symbol}?timeframe=1HR`
- `GET /api/analyze/live/mtf/{symbol}`
- `POST /api/analyze/manual`

### Watchlists
- `GET /api/watchlists`
- `GET /api/watchlists/{name}/scan`

### Alerts
- `GET /api/alerts`
- `POST /api/alerts`
- `DELETE /api/alerts?symbol=META&level=615`

### Trades
- `GET /api/trades`
- `POST /api/trades`
- `PUT /api/trades` (update status)
- `GET /api/trades/stats`

## 🔧 Usage Examples

### Python
```python
from finnhub_scanner import FinnhubScanner

scanner = FinnhubScanner("YOUR_API_KEY")

# Single analysis
result = scanner.analyze("META", "2HR")
print(f"{result.signal_emoji} {result.signal}")

# MTF analysis
mtf = scanner.analyze_mtf("META")
print(f"Combined: {mtf.signal_emoji} {mtf.dominant_signal}")

# Scan watchlist
results = scanner.scan_symbols(["META", "AAPL", "NVDA"])
```

### Manual Chart Input
```python
from chart_input_analyzer import ChartInputSystem

system = ChartInputSystem()

result = system.analyze(
    symbol="META",
    price=619.28,
    vah=667.72,
    poc=660.40,
    val=647.22,
    vwap=619.63,
    rsi=33.58,
    timeframe="2HR"
)
```

## 📊 Signal Types

| Signal | Meaning | Action |
|--------|---------|--------|
| 🟢 LONG_SETUP | Strong bullish confluence | Look for long entry |
| 🔴 SHORT_SETUP | Strong bearish confluence | Look for short entry |
| 🟡 YELLOW | Mixed signals | Wait for confirmation |
| ⚪ NEUTRAL | No clear setup | Stay out |

---
Built for Rob's Trading Systems
