"""
Push a live scan to Discord — mirrors simple.html "All" universe scan
"""
import os, time, httpx
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from finnhub_scanner_v2 import FinnhubScanner
from universe import ALL_SYMBOLS

ET = ZoneInfo("America/New_York")

api_key = os.environ.get("POLYGON_API_KEY", "")
scanner = FinnhubScanner(api_key)

symbols = ALL_SYMBOLS
print(f"Scanning {len(symbols)} symbols (All universe) on 2HR...")

bullish, bearish, yellow = [], [], []

for i, sym in enumerate(symbols):
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
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(symbols)} scanned...")
        time.sleep(0.3)
    except Exception as e:
        print(f"  skip {sym}: {e}")

bullish.sort(key=lambda x: -x["conf"])
bearish.sort(key=lambda x: -x["conf"])

print(f"\n=== RESULTS ===")
print(f"Bullish: {len(bullish)}")
for b in bullish:
    print(f"  {b['symbol']:6s} {b['conf']:.0f}% bull={b['bull']:.0f} bear={b['bear']:.0f} {b['pos']}")
print(f"Bearish: {len(bearish)}")
for b in bearish:
    print(f"  {b['symbol']:6s} {b['conf']:.0f}% bull={b['bull']:.0f} bear={b['bear']:.0f} {b['pos']}")
print(f"Yellow (notable): {len(yellow)}")
for y in yellow:
    print(f"  {y['symbol']:6s} bull={y['bull']:.0f} bear={y['bear']:.0f} {y['pos']}")

# --- Build Discord embeds ---
now_et = datetime.now(ET)
embeds = []

# Main summary embed
summary_fields = []

if bullish:
    lines = []
    for b in bullish[:10]:
        lines.append(f"**{b['symbol']}** \u2014 {b['conf']:.0f}% | B:{b['bull']:.0f} S:{b['bear']:.0f} | {b['pos']}")
    if len(bullish) > 10:
        lines.append(f"*+{len(bullish)-10} more...*")
    summary_fields.append({"name": f"\U0001f7e2 LONG SETUPS ({len(bullish)})", "value": "\n".join(lines), "inline": False})

if bearish:
    lines = []
    for b in bearish[:10]:
        lines.append(f"**{b['symbol']}** \u2014 {b['conf']:.0f}% | B:{b['bull']:.0f} S:{b['bear']:.0f} | {b['pos']}")
    if len(bearish) > 10:
        lines.append(f"*+{len(bearish)-10} more...*")
    summary_fields.append({"name": f"\U0001f534 SHORT SETUPS ({len(bearish)})", "value": "\n".join(lines), "inline": False})

if not bullish and not bearish:
    summary_fields.append({"name": "\u26aa No Confirmed Setups", "value": f"Scanned {len(symbols)} symbols \u2014 all YELLOW.", "inline": False})

if yellow:
    y_text = ", ".join([f"{y['symbol']}({'B' if y['bull']>y['bear'] else 'S'})" for y in yellow[:15]])
    if len(yellow) > 15:
        y_text += f" +{len(yellow)-15} more"
    summary_fields.append({"name": f"\U0001f7e1 Notable Yellows ({len(yellow)})", "value": y_text, "inline": False})

# Scan info
total = len(bullish) + len(bearish)
summary_fields.append({
    "name": "\U0001f4ca Scan Info",
    "value": f"{len(symbols)} symbols | 2HR TF | 60d lookback | Score\u226560 Gap\u226520\n{total} setups | {len(yellow)} yellows | {len(symbols)-total-len(yellow)} no signal",
    "inline": False
})

embeds.append({
    "title": f"\U0001f50d Full Scan \u2014 {now_et.strftime('%b %d, %I:%M %p ET')}",
    "color": 0x00D9FF,
    "fields": summary_fields,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "footer": {"text": "SEF Trading Terminal \u2014 Simple Scanner (All Universe)"}
})

# --- Command reference card ---
quick_cmds = []
if bullish:
    quick_cmds.append(f"`!options {bullish[0]['symbol']}` — Top long flow")
if bearish:
    quick_cmds.append(f"`!options {bearish[0]['symbol']}` — Top short flow")

ref_text = ""
if quick_cmds:
    ref_text += "**⚡ Quick trades from this scan:**\n" + "\n".join(quick_cmds) + "\n\n"
ref_text += (
    "**Dig deeper on any setup:**\n"
    "`!options SYMBOL` — Full options breakdown (IV, flow, OI walls, unusual)\n"
    "`!scan SYMBOL` — Technical scanner (levels, VWAP, volume profile)\n\n"
    "**Run more scans:**\n"
    "`!fullscan` — Full 72-symbol scan + options overlay\n"
    "`!unusual` — Unusual options activity alerts\n"
    "`!unusual AAPL,NVDA,TSLA` — Check specific tickers\n\n"
    "**Other:**\n"
    "`!brief` — Market pulse (SPY, QQQ, IWM, DIA)\n"
    "`!price SYMBOL` — Quick price check\n"
    "`!alerts` — View active alerts"
)

embeds.append({
    "title": "🤖 What's Next? — Command Reference",
    "description": ref_text,
    "color": 0x334155,
    "footer": {"text": "Type any command in this channel • SEF Trading Terminal"}
})

# --- Send to Discord ---
webhook_url = "https://discord.com/api/webhooks/1473815224825155686/LaIO4JSZzLfpRf_dPU64IOsuAKr4obDFKusGgC76MxMvA8dKgOdKiNectI5AxfooZ1Tt"

payload = {"username": "SEF Scanner", "embeds": embeds}
resp = httpx.post(webhook_url, json=payload, timeout=15)
print(f"\nDiscord status: {resp.status_code}")
if resp.status_code in (200, 204):
    print("\u2705 Scan posted to Discord!")
else:
    print(f"\u274c Failed: {resp.text[:300]}")
