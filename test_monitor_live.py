import requests, json

BASE = "https://analysisgrid-production.up.railway.app"

# 1. Monitor status
print("=== 1. MONITOR STATUS ===")
status = requests.get(f"{BASE}/api/monitor/status").json()
print(f"Running: {status['running']}, Market hours: {status['market_hours']}")

# 2. Log a fresh test trade (AAPL LONG, entry lower than current price to test WIN)
print("\n=== 2. LOGGING TEST TRADE ===")
# AAPL is ~230+, so set entry=220, stop=215, target=225 (should be hit since current price > 225)
resp = requests.post(f"{BASE}/api/trades", json={
    "symbol": "AAPL",
    "direction": "LONG",
    "entry": 220.00,
    "stop": 215.00,
    "target": 225.00,
    "timeframe": "swing",
    "signal": "TEST_MONITOR",
    "notes": "Auto-close test — target should be immediately hit"
})
print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2))

# 3. Verify trade exists
print("\n=== 3. VERIFY TRADE ===")
trades = requests.get(f"{BASE}/api/trades").json()
print(f"Trade count: {trades.get('count')}")

# 4. Force a test cycle — should detect AAPL price > 225 and auto-close as WIN
print("\n=== 4. TRIGGERING TEST CYCLE ===")
result = requests.post(f"{BASE}/api/monitor/test").json()
print(json.dumps(result, indent=2))

# 5. Check if trade was closed
print("\n=== 5. TRADES AFTER CYCLE ===")
trades2 = requests.get(f"{BASE}/api/trades").json()
for t in trades2.get("trades", []):
    print(f"  {t.get('symbol')} status={t.get('status')} exit={t.get('exit_price')} pnl={t.get('result_pct')}")
