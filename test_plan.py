import requests, time, json

base = 'https://analysisgrid-production.up.railway.app'
body = {
    'symbol': 'AAPL',
    'current_price': 264.18,
    'vah': 270,
    'poc': 265,
    'val': 260,
    'vwap': 264,
    'bull_score': 55,
    'bear_score': 45,
    'rsi': 48,
    'confidence': 60,
    'direction': 'long',
    'scan_type': 'entry',
    'timeframe': '1HR'
}

t0 = time.time()
r = requests.post(f'{base}/api/rules/generate', json=body, timeout=45)
elapsed = time.time() - t0

print(f'Status: {r.status_code} Time: {elapsed:.1f}s')
data = r.json()
if r.status_code == 200:
    for k in ['symbol', 'direction', 'confidence', 'entry_price', 'stop_loss', 'target_1', 'options_source']:
        print(f'  {k}: {data.get(k)}')
    explain = data.get('explanation', '')
    print(f'  explanation: {explain[:150]}...' if len(explain) > 150 else f'  explanation: {explain}')
else:
    print(f'Error: {json.dumps(data)[:300]}')
