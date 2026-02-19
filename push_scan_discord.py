"""
Push a live scan to Discord — one-shot script
"""
import os, time, httpx
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from finnhub_scanner_v2 import FinnhubScanner

ET = ZoneInfo("America/New_York")

api_key = os.environ.get("POLYGON_API_KEY", "")
scanner = FinnhubScanner(api_key)

symbols = [
    "SPY","QQQ","AAPL","MSFT","NVDA","AMD","META","TSLA",
    "AMZN","GOOGL","NFLX","CRM","AVGO","MU","COIN",
    "BA","JPM","GS","XOM","UNH","V","WMT","HD","DIS",
    "SOFI","PLTR","RIVN","HOOD","MARA","DKNG","SMCI","ARM"
]

bullish, bearish, yellow = [], [], []

for sym in symbols:
    try:
        result = scanner.analyze(sym, "2HR")
        if not result:
            continue
        bull = result.bull_score or 0
        bear = result.bear_score or 0
        sig = result.signal or "YELLOW"
        conf = result.confidence or 0
        pos = result.position or "-"
        entry = {"symbol": sym, "signal": sig, "bull": bull, "bear": bear, "conf": conf, "pos": pos}

        if "LONG" in str(sig):
            bullish.append(entry)
        elif "SHORT" in str(sig):
            bearish.append(entry)
        else:
            gap = abs(bull - bear)
            if gap >= 15:
                yellow.append(entry)
        time.sleep(0.3)
    except Exception as e:
        print(f"  skip {sym}: {e}")

bullish.sort(key=lambda x: -x["conf"])
bearish.sort(key=lambda x: -x["conf"])

print(f"\n=== RESULTS ===")
print(f"Bullish: {len(bullish)}")
for b in bullish:
    print(f"  {b}")
print(f"Bearish: {len(bearish)}")
for b in bearish:
    print(f"  {b}")
print(f"Yellow (notable): {len(yellow)}")
for y in yellow:
    print(f"  {y}")

# --- Build Discord embed ---
now_et = datetime.now(ET)
fields = []

if bullish:
    lines = []
    for b in bullish[:8]:
        lines.append(f"**{b['symbol']}** — {b['conf']:.0f}% conf | Bull {b['bull']:.0f} / Bear {b['bear']:.0f} | {b['pos']}")
    fields.append({"name": f"\U0001f7e2 LONG SETUPS ({len(bullish)})", "value": "\n".join(lines), "inline": False})

if bearish:
    lines = []
    for b in bearish[:8]:
        lines.append(f"**{b['symbol']}** — {b['conf']:.0f}% conf | Bull {b['bull']:.0f} / Bear {b['bear']:.0f} | {b['pos']}")
    fields.append({"name": f"\U0001f534 SHORT SETUPS ({len(bearish)})", "value": "\n".join(lines), "inline": False})

if not bullish and not bearish:
    fields.append({"name": "\u26aa No Confirmed Setups", "value": f"Scanned {len(symbols)} symbols — all YELLOW. Market indecisive.", "inline": False})

if yellow:
    y_text = ", ".join([f"{y['symbol']}({'B' if y['bull']>y['bear'] else 'S'})" for y in yellow[:10]])
    fields.append({"name": "\U0001f7e1 Notable Yellows", "value": y_text, "inline": False})

fields.append({"name": "\U0001f4ca Scan Info", "value": f"{len(symbols)} symbols | 2HR timeframe | Score\u226560 + Gap\u226520 filter", "inline": False})

embed = {
    "title": f"\U0001f50d Market Scan — {now_et.strftime('%b %d, %I:%M %p ET')}",
    "color": 0x00D9FF,
    "fields": fields,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "footer": {"text": "SEF Trading Terminal — Tightened Filter v2"}
}

# --- Send to Discord ---
webhook_url = "https://discord.com/api/webhooks/1473815224825155686/LaIO4JSZzLfpRf_dPU64IOsuAKr4obDFKusGgC76MxMvA8dKgOdKiNectI5AxfooZ1Tt"

payload = {"username": "SEF Scanner", "embeds": [embed]}
resp = httpx.post(webhook_url, json=payload, timeout=15)
print(f"\nDiscord status: {resp.status_code}")
if resp.status_code in (200, 204):
    print("\u2705 Scan posted to Discord!")
else:
    print(f"\u274c Failed: {resp.text[:300]}")
