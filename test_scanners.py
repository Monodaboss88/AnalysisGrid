"""Quick test of all independent scanner endpoints."""
import requests, time

BASE = 'https://analysisgrid-production.up.railway.app'

tests = [
    ('Health',         '/api/health'),
    ('Simple Scan',    '/api/analyze/live/NVDA'),
    ('MTF Scan',       '/api/analyze/live/mtf/NVDA'),
    ('Signal Quick',   '/api/signal/quick/NVDA'),
    ('War Room',       '/api/war-room?tickers=NVDA'),
    ('Buffett',        '/api/buffett-scan?tickers=NVDA'),
    ('Sustainability', '/api/sustainability/analyze?symbol=NVDA'),
    ('Combo',          '/api/combo-scan?tickers=NVDA'),
    ('Options Flow',   '/api/options-flow?tickers=NVDA'),
    ('Card',           '/api/card/NVDA/data'),
]

for name, path in tests:
    try:
        t = time.time()
        r = requests.get(BASE + path, timeout=90)
        elapsed = round(time.time() - t, 1)
        if r.status_code == 200:
            print(f"  OK    {elapsed:>5}s  {name}")
        else:
            print(f"  FAIL  {elapsed:>5}s  {name} [{r.status_code}] {r.text[:120]}")
    except requests.exceptions.ReadTimeout:
        print(f"  TIMEOUT 90s  {name}")
    except Exception as e:
        print(f"  ERROR        {name}: {type(e).__name__}")
