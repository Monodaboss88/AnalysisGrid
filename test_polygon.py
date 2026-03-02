"""Test Polygon proxy endpoints used by Catalyst."""
import requests, time

BASE = 'https://analysisgrid-production.up.railway.app'

# Wake up server
for i in range(5):
    try:
        r = requests.get(f'{BASE}/api/health', timeout=120)
        print('Server awake:', r.status_code)
        break
    except:
        print(f'Cold start attempt {i+1}...')
        time.sleep(10)

time.sleep(2)

# Test polygon bars
t = time.time()
r = requests.get(f'{BASE}/api/polygon/bars/NVDA?from_date=2026-01-01&to_date=2026-03-01&adjusted=true&sort=asc&limit=100', timeout=60)
elapsed = round(time.time()-t, 1)
print(f'Polygon bars: {r.status_code} in {elapsed}s')
if r.status_code == 200:
    data = r.json()
    results = data.get('results', [])
    print(f'  Got {len(results)} bars')
else:
    print(f'  Body: {r.text[:200]}')

# Test polygon news
t = time.time()
r = requests.get(f'{BASE}/api/polygon/news/NVDA?from_date=2026-02-01&to_date=2026-03-01&limit=3', timeout=60)
elapsed = round(time.time()-t, 1)
print(f'Polygon news: {r.status_code} in {elapsed}s')
if r.status_code == 200:
    data = r.json()
    results = data.get('results', [])
    print(f'  Got {len(results)} news items')
else:
    print(f'  Body: {r.text[:200]}')

# Test simple scan (used by other scanners)
t = time.time()
r = requests.get(f'{BASE}/api/analyze/live/NVDA', timeout=60)
elapsed = round(time.time()-t, 1)
print(f'Simple scan: {r.status_code} in {elapsed}s')
