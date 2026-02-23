import urllib.request, json

print("=== SCAN TEST ===")

# Step 1: Get key from Railway
r = urllib.request.urlopen('https://analysisgrid-production.up.railway.app/api/polygon-key')
key = json.loads(r.read())['key']
print("1. Railway API key endpoint: OK")

# Step 2: Pull AAPL bars from Polygon with that key
url = 'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2026-02-10/2026-02-21?apiKey=' + key
r2 = urllib.request.urlopen(url)
data = json.loads(r2.read())
count = data.get('resultsCount', 0)
print("2. Polygon AAPL daily bars:", count, "results")

# Step 3: Test stats endpoint
req = urllib.request.Request(
    'https://analysisgrid-production.up.railway.app/api/backtest/stats',
    data=json.dumps({"symbols":["AAPL"],"days_back":10,"rules":[{"type":"high_off_open","min_pct":0.2,"max_pct":5.0}]}).encode(),
    headers={"Content-Type":"application/json"}
)
r3 = urllib.request.urlopen(req, timeout=120)
stats = json.loads(r3.read())
s = stats.get('summary', {})
print("3. Stats Scanner AAPL 10d:")
print("   Qualifying:", s.get('qualifying_days','?'), "/", s.get('total_days_scanned','?'))
print("   Avg O->H dollar:", s.get('avg_high_off_open_dollar', '?'))
print("   Avg O->H pct:", s.get('avg_high_off_open', '?'))
print("   Avg O->L dollar:", s.get('avg_low_off_open_dollar', '?'))
print("   Avg O->L pct:", s.get('avg_low_off_open', '?'))
print("   Green pct:", s.get('green_pct', '?'))

print("\n=== ALL TESTS PASSED ===")
