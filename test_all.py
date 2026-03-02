"""Full 22-endpoint test with server wake-up."""
import requests, time, json

BASE = 'https://analysisgrid-production.up.railway.app'

# ── Wake server first (cold start can take 30-90s) ──
print("=" * 60)
print("WAKING SERVER (may take up to 2 min)...")
print("=" * 60)
for attempt in range(6):
    try:
        r = requests.get(f'{BASE}/api/health', timeout=120)
        print(f"  Server awake! ({r.status_code})")
        break
    except Exception as e:
        print(f"  Attempt {attempt+1}/6 failed: {type(e).__name__}")
        if attempt < 5:
            time.sleep(10)
else:
    print("  WARNING: Server may still be starting...")

time.sleep(2)
print()

# ── Test all endpoints ──
print("=" * 60)
print("RUNNING ALL 22 ENDPOINT TESTS")
print("=" * 60)

def test(num, name, method, path, body=None, timeout=90):
    url = BASE + path
    t = time.time()
    try:
        if method == 'POST':
            r = requests.post(url, json=body, timeout=timeout)
        else:
            r = requests.get(url, timeout=timeout)
        elapsed = round(time.time() - t, 1)
        snippet = r.text[:120].replace('\n', ' ')
        status = r.status_code
        tag = 'OK' if 200 <= status < 300 else f'FAIL'
        print(f"{num:>2}. [{tag:4s}] {status} {elapsed:>5}s  {name:22s}  {snippet}")
    except requests.exceptions.Timeout:
        elapsed = round(time.time() - t, 1)
        print(f"{num:>2}. [TIME] ---  {elapsed:>5}s  {name:22s}  TIMEOUT after {timeout}s")
    except Exception as e:
        elapsed = round(time.time() - t, 1)
        print(f"{num:>2}. [ERR ] ---  {elapsed:>5}s  {name:22s}  {type(e).__name__}: {e}")

# 1. AI Advisor
test(1, "AI Advisor", "POST", "/api/ai/analyze", {
    "symbol":"AAPL","timeframe":"swing","signal":"bullish","confidence":75,
    "bull_score":70,"bear_score":30,"price":200,"vah":205,"poc":200,"val":195,
    "vwap":200,"position":"above_poc","vwap_zone":"above","rsi":55,"rsi_zone":"neutral",
    "spy_price":500,"spy_sma_20":495,"spy_sma_50":490,"spy_sma_200":480,"vix":15
})

# 2. MTF Auction
test(2, "MTF Auction", "GET", "/api/analyze/live/mtf/AAPL")

# 3. MTF + AI
test(3, "MTF + AI", "POST", "/api/analyze/live/mtf/AAPL/ai", {})

# 4. Analyze Live
test(4, "Analyze Live", "GET", "/api/analyze/live/AAPL")

# 5. Regime Scanner
test(5, "Regime Scanner", "GET", "/api/regime-scan?tickers=AAPL,MSFT")

# 6. Regime Levels
test(6, "Regime Levels", "GET", "/api/regime-levels/AAPL")

# 7. Buffett Scanner
test(7, "Buffett Scanner", "GET", "/api/buffett-scan?tickers=AAPL")

# 8. War Room
test(8, "War Room", "GET", "/api/war-room?tickers=AAPL")

# 9. Signal Probability
test(9, "Signal Probability", "GET", "/api/signal/AAPL")

# 10. Signal Quick
test(10, "Signal Quick", "GET", "/api/signal/AAPL/quick")

# 11. Alpha Scanner
test(11, "Alpha Scanner", "GET", "/api/alpha/scan?tickers=AAPL,MSFT")

# 12. Entry Scanner
test(12, "Entry Scanner", "GET", "/api/entry-scan/scan/AAPL")

# 13. Combo Scanner
test(13, "Combo Scanner", "GET", "/api/combo-scan?tickers=AAPL,MSFT")

# 14. Options Flow
test(14, "Options Flow", "GET", "/api/options-flow?tickers=AAPL")

# 15. Research Builder
test(15, "Research Builder", "POST", "/api/research/build", {
    "config":{"title":"Test","layer2":[{"ticker":"AAPL"},{"ticker":"MSFT"}],"layer3":[]},
    "mode":"full"
})

# 16. Catalyst Scanner
test(16, "Catalyst Scanner", "GET", "/api/scan/live?watchlist=Mega%20Cap%20Tech&limit=2")

# 17. Sustainability Quick
test(17, "Sustainability Quick", "GET", "/api/sustainability/quick?symbols=AAPL")

# 18. Sustainability Analyze
test(18, "Sustain Analyze", "GET", "/api/sustainability/analyze?symbol=AAPL")

# 19. Sustainability Scan
test(19, "Sustain Scan", "POST", "/api/sustainability/scan", {"symbols":["AAPL"]})

# 20. Trading Cards
test(20, "Trading Cards", "GET", "/api/card/AAPL/data")

# 21. Quote
test(21, "Quote", "GET", "/api/quote/AAPL")

# 22. Structure Reversal
test(22, "Structure Reversal", "GET", "/api/structure/reversals/AAPL")

# Post-test health
print()
print("=" * 60)
print("POST-TEST HEALTH CHECK")
print("=" * 60)
try:
    r = requests.get(f'{BASE}/api/health', timeout=15)
    d = r.json()
    print(f"  Status: {d.get('status')}  Version: {d.get('version')}")
    print(f"  Threads: {d.get('executor_threads')}/{d.get('executor_max')}  Pending: {d.get('executor_pending')}")
except Exception as e:
    print(f"  Health check failed: {e}")

print()
print("DONE.")
