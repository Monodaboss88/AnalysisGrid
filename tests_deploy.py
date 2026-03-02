"""Quick deploy smoke test for the decomposed server."""
import requests, time

BASE = "https://analysisgrid-production.up.railway.app"

def health():
    r = requests.get(f"{BASE}/api/health", timeout=10)
    return r.json()

print("=== Test 1: Regime scan ===")
t0 = time.time()
r = requests.get(f"{BASE}/api/regime-scan?tickers=AAPL&days=30", timeout=60)
print(f"  Status: {r.status_code}, Time: {time.time()-t0:.1f}s")
if r.status_code == 200:
    d = r.json()
    cache = d.get("_cache", {})
    print(f"  Cache: {cache}")

print()
h = health()
print(f"=== Health: {h.get('memory_mb')}MB, threads={h.get('executor_threads')}, busy={h.get('scan_busy')} ===")

print()
print("=== Test 2: War room ===")
t0 = time.time()
r = requests.get(f"{BASE}/api/war-room?tickers=AAPL,MSFT,META", timeout=120)
print(f"  Status: {r.status_code}, Time: {time.time()-t0:.1f}s")

print()
h = health()
print(f"=== Health: {h.get('memory_mb')}MB, threads={h.get('executor_threads')}, busy={h.get('scan_busy')} ===")

print()
print("=== Test 3: Buffett (should be cached) ===")
t0 = time.time()
r = requests.get(f"{BASE}/api/buffett-scan?tickers=AAPL", timeout=45)
print(f"  Status: {r.status_code}, Time: {time.time()-t0:.1f}s")

print()
print("=== Test 4: Pages still work? ===")
for page in ["/regime", "/buffett", "/warroom"]:
    r = requests.get(f"{BASE}{page}", timeout=10)
    print(f"  {page}: {r.status_code}")

print()
h = health()
print(f"=== FINAL Health: {h.get('memory_mb')}MB, threads={h.get('executor_threads')} ===")
print("DONE")
