import requests, concurrent.futures, time

base = 'https://analysisgrid-production.up.railway.app'
tickers = ['AAPL','MSFT','NVDA','TSLA','AMD','META','GOOGL','AMZN','CRM','INTC','ORCL','ADBE','AVGO','NFLX','TXN','SNOW','NOW','IBM']
url = base + '/api/analyze/live/{}?timeframe=2HR&vp_period=position&with_ai=false'

def fetch(t):
    try:
        r = requests.get(url.format(t), timeout=60)
        return (t, r.status_code)
    except Exception as e:
        return (t, str(e)[:40])

for scan_num in range(1, 4):
    print(f'=== SCAN {scan_num} (18 parallel) ===')
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as ex:
        results = list(ex.map(fetch, tickers))
    elapsed = time.time() - t0
    ok = sum(1 for _, s in results if s == 200)
    r429 = sum(1 for _, s in results if s == 429)
    fails = [(t, s) for t, s in results if s != 200 and s != 429]
    print(f'  {ok}/18 OK, {r429} rate-limited (429), {len(fails)} errors in {elapsed:.1f}s')
    for t, s in fails:
        print(f'    FAIL: {t}={s}')
    # Check health
    try:
        h = requests.get(base + '/api/health', timeout=10).json()
        mem = h.get('memory_mb', '?')
        thr = h.get('executor_threads', '?')
        pend = h.get('executor_pending', '?')
        print(f'  Health: mem={mem}MB threads={thr} pending={pend}')
    except Exception as e:
        print(f'  Health: FAILED - {e}')
    if scan_num < 3:
        print(f'  Waiting 10s...')
        time.sleep(10)

print()
print('=== DONE ===')
