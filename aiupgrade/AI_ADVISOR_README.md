# AI Trading Advisor - Integration Guide
## Hedge Fund Level Intelligence for MTF Auction Scanner

### Quick Start

1. **Install dependencies:**
```bash
pip install openai anthropic
# OR just one:
pip install openai
```

2. **Set API key:**
```bash
# For OpenAI (recommended for cost/speed)
export OPENAI_API_KEY=sk-your-key-here

# OR for Anthropic (best reasoning)
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

3. **Add to unified_server.py:**
```python
# At top of file, add import:
from ai_advisor_endpoints import ai_router

# After app = FastAPI(...), add:
app.include_router(ai_router, prefix="/api/ai")
```

4. **Run the server:**
```bash
python unified_server.py
```

5. **Test endpoints:**
```bash
# Check status
curl http://localhost:8000/api/ai/status

# Get regime
curl -X POST http://localhost:8000/api/ai/regime \
  -H "Content-Type: application/json" \
  -d '{"spy_price": 595.50, "spy_sma_20": 590.00, "spy_sma_50": 585.00, "spy_sma_200": 550.00, "vix": 16.5}'
```

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR EXISTING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│  unified_server.py    │  MTF Scanner      │  Finnhub Data      │
│  - Manual input       │  - Volume Profile │  - Real-time quotes│
│  - Watchlists         │  - Flow Control   │  - Candles         │
│  - Alerts             │  - RSI Engine     │                    │
│  - Trade tracker      │  - VWAP Engine    │                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEW AI LAYER (ai_trading_advisor.py)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   REGIME     │  │     AI       │  │    TRADE JOURNAL     │  │
│  │  DETECTOR    │  │  COMMENTARY  │  │    + AI REVIEW       │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────┤  │
│  │ • Trending   │  │ • GPT-4/     │  │ • SQLite storage     │  │
│  │ • Range      │  │   Claude     │  │ • Win rate tracking  │  │
│  │ • Vol regime │  │ • Context-   │  │ • R-multiple calc    │  │
│  │ • VIX-aware  │  │   aware      │  │ • Pattern analysis   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              NEWS SENTIMENT ANALYZER                      │  │
│  │  • Headlines → Sentiment score                           │  │
│  │  • Key event extraction                                  │  │
│  │  • Trading timeframe assessment                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

### What Makes This "Hedge Fund Level"

| Feature | Basic GPT Integration | This System |
|---------|----------------------|-------------|
| Context | Single prompt | Full scanner + regime + history |
| Memory | None | SQLite trade journal |
| Calibration | None | Win rate from YOUR trades |
| Regime Awareness | None | VIX + trend + volatility |
| News Integration | None | Sentiment scoring |
| Position Sizing | None | Regime-adjusted notes |
| Trade Review | None | AI post-mortem on exits |

---

### API Endpoints

#### Market Regime
```
POST /api/ai/regime
```
Returns current market regime (TRENDING_BULL, RANGE_BOUND, etc.) with strategy notes.

#### Full Analysis
```
POST /api/ai/analyze
```
Complete analysis combining scanner signal + regime + historical stats + AI commentary.

#### Quick Commentary
```
GET /api/ai/quick-commentary?symbol=META&signal=LONG_SETUP&confidence=72&position=BELOW_VALUE&rsi=33
```
Fast signal interpretation for dashboard integration.

#### Trade Journal
```
POST /api/ai/journal/entry    # Log new trade
POST /api/ai/journal/exit     # Log exit, get R-multiple
GET  /api/ai/journal/trades   # Query history
GET  /api/ai/journal/stats    # Performance summary
POST /api/ai/journal/{id}/review  # AI trade review
```

#### News Analysis
```
POST /api/ai/news/analyze
```
Analyze headlines for sentiment and trading implications.

---

### Example Workflow

```python
import requests

BASE = "http://localhost:8000"

# 1. Check market regime first
regime = requests.post(f"{BASE}/api/ai/regime", json={
    "spy_price": 595.50,
    "spy_sma_20": 590.00,
    "spy_sma_50": 585.00,
    "spy_sma_200": 550.00,
    "vix": 16.5
}).json()

print(f"Regime: {regime['regime']}")
print(f"Strategy: {regime['strategy']}")

# 2. Run your MTF scanner (existing code)
scanner_result = your_scanner.scan(df, "META")

# 3. Get AI-enhanced analysis
analysis = requests.post(f"{BASE}/api/ai/analyze", json={
    "symbol": "META",
    "signal": scanner_result.dominant_signal.value,
    "confidence": scanner_result.confidence,
    # ... other fields from scanner
    "spy_price": 595.50,
    "spy_sma_20": 590.00,
    "spy_sma_50": 585.00,
    "spy_sma_200": 550.00,
    "vix": 16.5
}).json()

print(f"AI Commentary: {analysis['ai_analysis']['commentary']}")
print(f"Historical Win Rate: {analysis['historical_stats']['win_rate']:.0%}")

# 4. If taking the trade, log it
if analysis['scanner_signal'] == 'LONG_SETUP':
    trade = requests.post(f"{BASE}/api/ai/journal/entry", json={
        "symbol": "META",
        "direction": "LONG",
        "entry_price": 620.00,
        "stop_price": 605.00,
        "target_1": 647.00,
        "signal_at_entry": "LONG_SETUP",
        "confidence_at_entry": 72,
        "regime_at_entry": regime['regime']
    }).json()
    
    print(f"Trade logged: {trade['trade_id']}")

# 5. When exiting, log the result
exit_result = requests.post(f"{BASE}/api/ai/journal/exit", json={
    "trade_id": trade['trade_id'],
    "exit_price": 648.50,
    "exit_reason": "TARGET_1"
}).json()

print(f"R-Multiple: {exit_result['r_multiple']:.1f}R")

# 6. Get AI review of the trade
review = requests.post(f"{BASE}/api/ai/journal/{trade['trade_id']}/review").json()
print(f"AI Review: {review['review']}")
```

---

### Cost Optimization

| Model | Cost/1K tokens | Best For |
|-------|---------------|----------|
| GPT-4o-mini | $0.00015 | Quick commentary, reviews |
| GPT-4o | $0.0025 | Full analysis |
| Claude Sonnet | $0.003 | Deep reasoning |

**Recommendation:**
- Use GPT-4o-mini for quick commentary and trade reviews
- Use GPT-4o for full analysis (3-5 per day max)
- Cache regime detection results (recalculate hourly)

**Monthly estimate at 20 trades/day:**
- Quick commentary: ~$3/month
- Full analysis: ~$15/month
- Trade reviews: ~$2/month
- **Total: ~$20/month**

---

### Next Level Enhancements

1. **Real-time news feeds:**
   - Benzinga Pro API (~$99/month)
   - NewsAPI.org (free tier available)

2. **Social sentiment:**
   - Twitter/X API
   - StockTwits API

3. **Options flow:**
   - Unusual Whales API
   - OptionsAI

4. **Earnings calendar:**
   - Finnhub earnings API (already have access)

---

### Files to Add

1. `ai_trading_advisor.py` - Core AI module
2. `ai_advisor_endpoints.py` - FastAPI routes
3. `trade_journal.db` - Created automatically (SQLite)

### Modified Files

1. `unified_server.py` - Add router include (2 lines)
2. `requirements.txt` - Add `openai` or `anthropic`

---

### Troubleshooting

**"AI engine not configured"**
- Check API key: `echo $OPENAI_API_KEY`
- Verify package installed: `pip show openai`

**"No trades in period"**
- The journal starts empty
- Log some trades first via the API

**Slow responses**
- Switch to GPT-4o-mini for quick commentary
- Cache regime detection (changes slowly)

---

Built for Rob's Trading Systems
