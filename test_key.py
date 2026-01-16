import finnhub
from datetime import datetime, timedelta

key = "d5l551hr01qgqufkb9vgd5l551hr01qgqufkba00"
c = finnhub.Client(api_key=key)

# Test quote (should work on free)
print("Testing quote...")
try:
    q = c.quote("META")
    print(f"  Quote OK: ${q.get('c')}")
except Exception as e:
    print(f"  Quote FAILED: {e}")

# Test daily candles
print("\nTesting daily candles...")
try:
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=20)).timestamp())
    d = c.stock_candles("META", "D", start, end)
    print(f"  Status: {d.get('s')}, Points: {len(d.get('c', []))}")
except Exception as e:
    print(f"  Candles FAILED: {e}")

# Test hourly candles
print("\nTesting hourly candles...")
try:
    d = c.stock_candles("META", "60", start, end)
    print(f"  Status: {d.get('s')}, Points: {len(d.get('c', []))}")
except Exception as e:
    print(f"  Candles FAILED: {e}")
