"""Test regime scanner for buffering."""
import requests, time, json

BASE = 'https://analysisgrid-production.up.railway.app'
tickers = 'NFLX,CRM,UBER,SQ,COIN,SHOP,ABNB,SNAP'

# Wake server
print('Waking server...')
try:
    requests.get(f'{BASE}/api/health', timeout=120)
    print('Server awake.')
except:
    print('Server slow to wake, continuing anyway...')

time.sleep(1)

# Test regime-scan with FRESH tickers (no cache)
print(f'\n--- Regime Scan ({tickers}) ---')
t = time.time()
r = requests.get(f'{BASE}/api/regime-scan?tickers={tickers}', timeout=120)
elapsed = round(time.time() - t, 1)
print(f'Status: {r.status_code}  Time: {elapsed}s')
if r.status_code == 200:
    data = r.json()
    cache_info = data.get('_cache', {})
    print(f'Cache: {cache_info}')
    results = data.get('results', {})
    for sym, val in results.items():
        if isinstance(val, dict):
            days = val.get('days_analyzed', 0)
            rt = val.get('runtime', '?')
            errs = val.get('errors', [])
            print(f'  {sym:6s} days={days} runtime={rt}s errors={errs}')
else:
    print(f'  Body: {r.text[:400]}')

# Test cached (same tickers again)
print(f'\n--- Regime Scan CACHED ---')
t = time.time()
r = requests.get(f'{BASE}/api/regime-scan?tickers={tickers}', timeout=120)
elapsed = round(time.time() - t, 1)
data = r.json()
cache_info = data.get('_cache', {})
print(f'Status: {r.status_code}  Time: {elapsed}s  Cache: {cache_info}')

print('\nDone.')
