"""
Unified Scan + Options → Discord
=================================
Scans the full 72-symbol universe, pulls options data for every confirmed
setup, then pushes a rich combined embed to Discord.

Reusable — called from:
  - python full_scan_discord.py          (manual one-shot)
  - !fullscan  in Discord bot
  - Scheduled cron / background poller
"""

import os, time, json
import httpx
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from finnhub_scanner_v2 import FinnhubScanner
from options_flow_scanner import _scan_single as options_scan_single

ET = ZoneInfo("America/New_York")

WEBHOOK_URL = "https://discord.com/api/webhooks/1473815224825155686/LaIO4JSZzLfpRf_dPU64IOsuAKr4obDFKusGgC76MxMvA8dKgOdKiNectI5AxfooZ1Tt"

# ── Universes (same as simple.html) ──
TECH = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AMD','INTC','CRM',
        'ORCL','ADBE','NFLX','PYPL','SQ','SHOP','SNOW','NET','DDOG','MDB',
        'AVGO','MU','PLTR','APP']
MEGA = ['BRK.B','UNH','JNJ','V','JPM','WMT','PG','MA','HD','DIS','BAC',
        'XOM','PFE','LLY','ABBV','MRK']
ETFS = ['SPY','QQQ','IWM','DIA','XLF','XLK','XLE','GLD','SLV','TLT','SMH',
        'ARKK','SOXX','XBI','VXX','SQQQ','TQQQ','IBIT']
MEME = ['GME','AMC','SOFI','RIVN','LCID','NIO','HOOD','COIN','MARA',
        'RIOT','DKNG','SPCE','TLRY','MSTR']
ALL_SYMBOLS = list(dict.fromkeys(TECH + MEGA + ETFS + MEME))


def run_full_scan(timeframe="2HR", include_options=True, webhook=True, quiet=False):
    """
    Run full universe scan + options overlay.
    Returns dict with bullish, bearish, yellows, and options data.
    """
    api_key = os.environ.get("POLYGON_API_KEY", "")
    scanner = FinnhubScanner(api_key)
    symbols = ALL_SYMBOLS

    if not quiet:
        print(f"[1/3] Scanning {len(symbols)} symbols on {timeframe}...")

    # ── Phase 1: Directional scan ──
    bullish, bearish, yellow = [], [], []

    for i, sym in enumerate(symbols):
        try:
            result = scanner.analyze(sym, timeframe)
            if not result:
                continue
            bull = result.bull_score or 0
            bear = result.bear_score or 0
            sig = result.signal or "YELLOW"
            conf = result.confidence or 0
            pos = result.position or "-"
            entry = {"symbol": sym, "signal": sig, "bull": bull, "bear": bear,
                     "conf": conf, "pos": pos}

            if "LONG" in str(sig):
                bullish.append(entry)
            elif "SHORT" in str(sig):
                bearish.append(entry)
            else:
                gap = abs(bull - bear)
                if gap >= 15:
                    yellow.append(entry)

            if not quiet and (i + 1) % 15 == 0:
                print(f"  {i+1}/{len(symbols)} scanned...")
            time.sleep(0.25)
        except Exception as e:
            if not quiet:
                print(f"  skip {sym}: {e}")

    bullish.sort(key=lambda x: -x["conf"])
    bearish.sort(key=lambda x: -x["conf"])

    if not quiet:
        print(f"  → {len(bullish)} longs, {len(bearish)} shorts, {len(yellow)} yellows")

    # ── Phase 2: Options overlay for confirmed setups ──
    options_data = {}
    if include_options and api_key:
        setup_symbols = [b["symbol"] for b in bullish] + [b["symbol"] for b in bearish]
        if setup_symbols:
            if not quiet:
                print(f"[2/3] Pulling options for {len(setup_symbols)} confirmed setups...")

            with ThreadPoolExecutor(max_workers=3) as executor:
                future_map = {
                    executor.submit(options_scan_single, sym, 30, 0.10): sym
                    for sym in setup_symbols
                }
                for future in as_completed(future_map):
                    sym = future_map[future]
                    try:
                        r = future.result(timeout=45)
                        if not r.get("error"):
                            options_data[sym] = r
                    except Exception:
                        pass

            if not quiet:
                print(f"  → Options data for {len(options_data)}/{len(setup_symbols)} tickers")
        else:
            if not quiet:
                print("[2/3] No confirmed setups — skipping options pull")
    else:
        if not quiet:
            print("[2/3] Options skipped (no API key or disabled)")

    # ── Phase 3: Build + send Discord embeds ──
    if not quiet:
        print("[3/3] Building Discord message...")

    now_et = datetime.now(ET)
    embeds = []

    # ── LONG SETUPS embed ──
    if bullish:
        long_fields = []
        for b in bullish[:12]:
            sym = b["symbol"]
            direction = "LONG"
            line = f"**{sym}** — {b['conf']:.0f}% | B:{b['bull']:.0f} S:{b['bear']:.0f} | {b['pos']}"

            # Append options data if available
            opts = options_data.get(sym)
            if opts:
                sentiment = opts.get("sentiment", "?")
                flow = opts.get("flowScore", 0)
                iv_pct = opts.get("avgIVPct")
                unusual = opts.get("unusualCount", 0)
                mp = opts.get("maxPain")

                # Alignment check
                aligned = "✓" if "BULLISH" in sentiment else ("✗" if "BEARISH" in sentiment else "~")
                iv_str = f"{iv_pct:.0f}%" if iv_pct else "N/A"
                mp_str = f"${mp:.0f}" if mp else "?"

                line += f"\n  └ Flow:{flow} | IV:{iv_str} | Unusual:{unusual} | MP:{mp_str} | {sentiment} {aligned}"

                # Top unusual call contracts
                unusual_calls = [u for u in opts.get("unusualContracts", []) if u["type"] == "call"]
                if unusual_calls:
                    top = unusual_calls[0]
                    line += f"\n  └ 🔥 ${top['strike']} Call {top['expiration']} — Vol:{top['volume']:,} (V/OI:{top['volOiRatio']}x)"

            long_fields.append({"name": f"🟢 {sym}", "value": line, "inline": False})

        embeds.append({
            "title": f"🟢 LONG SETUPS ({len(bullish)})",
            "color": 0x00FF88,
            "fields": long_fields[:10],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── SHORT SETUPS embed ──
    if bearish:
        short_fields = []
        for b in bearish[:12]:
            sym = b["symbol"]
            line = f"**{sym}** — {b['conf']:.0f}% | B:{b['bull']:.0f} S:{b['bear']:.0f} | {b['pos']}"

            opts = options_data.get(sym)
            if opts:
                sentiment = opts.get("sentiment", "?")
                flow = opts.get("flowScore", 0)
                iv_pct = opts.get("avgIVPct")
                unusual = opts.get("unusualCount", 0)
                mp = opts.get("maxPain")

                aligned = "✓" if "BEARISH" in sentiment else ("✗" if "BULLISH" in sentiment else "~")
                iv_str = f"{iv_pct:.0f}%" if iv_pct else "N/A"
                mp_str = f"${mp:.0f}" if mp else "?"

                line += f"\n  └ Flow:{flow} | IV:{iv_str} | Unusual:{unusual} | MP:{mp_str} | {sentiment} {aligned}"

                unusual_puts = [u for u in opts.get("unusualContracts", []) if u["type"] == "put"]
                if unusual_puts:
                    top = unusual_puts[0]
                    line += f"\n  └ 🔥 ${top['strike']} Put {top['expiration']} — Vol:{top['volume']:,} (V/OI:{top['volOiRatio']}x)"

            short_fields.append({"name": f"🔴 {sym}", "value": line, "inline": False})

        embeds.append({
            "title": f"🔴 SHORT SETUPS ({len(bearish)})",
            "color": 0xFF4444,
            "fields": short_fields[:10],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── Summary embed ──
    total_setups = len(bullish) + len(bearish)
    opts_aligned = 0
    opts_conflict = 0
    for b in bullish:
        opts = options_data.get(b["symbol"])
        if opts and "BULLISH" in opts.get("sentiment", ""):
            opts_aligned += 1
        elif opts and "BEARISH" in opts.get("sentiment", ""):
            opts_conflict += 1
    for b in bearish:
        opts = options_data.get(b["symbol"])
        if opts and "BEARISH" in opts.get("sentiment", ""):
            opts_aligned += 1
        elif opts and "BULLISH" in opts.get("sentiment", ""):
            opts_conflict += 1

    # Best setup (highest flow + alignment)
    best_setup = None
    best_score = 0
    for b in bullish + bearish:
        sym = b["symbol"]
        opts = options_data.get(sym)
        if not opts:
            continue
        direction = "LONG" if b in bullish else "SHORT"
        sentiment = opts.get("sentiment", "NEUTRAL")
        flow = opts.get("flowScore", 0)
        base = b["conf"] + flow
        if (direction == "LONG" and "BULLISH" in sentiment) or \
           (direction == "SHORT" and "BEARISH" in sentiment):
            base += 30
        if base > best_score:
            best_score = base
            best_setup = {**b, "direction": direction, "opts": opts}

    summary_lines = [
        f"**{len(symbols)}** symbols scanned | **{timeframe}** timeframe",
        f"**{total_setups}** setups ({len(bullish)}L / {len(bearish)}S) | **{len(yellow)}** yellows",
    ]
    if options_data:
        summary_lines.append(f"Options: **{len(options_data)}** tickers | **{opts_aligned}** aligned | **{opts_conflict}** conflicts")
    if best_setup:
        dir_emoji = "🟢" if best_setup["direction"] == "LONG" else "🔴"
        opts = best_setup["opts"]
        summary_lines.append(
            f"\n🏆 **Best Setup: {dir_emoji} {best_setup['symbol']}**\n"
            f"Conf: {best_setup['conf']:.0f}% | Flow: {opts.get('flowScore',0)} | "
            f"Sentiment: {opts.get('sentiment','?')} ✓ | IV: {opts.get('avgIVPct','?')}%"
        )

    if yellow:
        y_list = ", ".join([f"{y['symbol']}({'B' if y['bull']>y['bear'] else 'S'})" for y in yellow[:12]])
        if len(yellow) > 12:
            y_list += f" +{len(yellow)-12}"
        summary_lines.append(f"\n🟡 **Yellows:** {y_list}")

    embeds.append({
        "title": f"📊 Scan Summary — {now_et.strftime('%b %d, %I:%M %p ET')}",
        "description": "\n".join(summary_lines),
        "color": 0x00D9FF,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "SEF Trading Terminal — Full Scan + Options Flow"}
    })

    result_data = {
        "bullish": bullish,
        "bearish": bearish,
        "yellow": yellow,
        "options": {k: {"flowScore": v.get("flowScore"), "sentiment": v.get("sentiment"),
                        "avgIVPct": v.get("avgIVPct"), "unusualCount": v.get("unusualCount"),
                        "maxPain": v.get("maxPain"), "totalVolume": v.get("totalVolume")}
                    for k, v in options_data.items()},
        "best_setup": best_setup["symbol"] if best_setup else None,
        "timestamp": now_et.isoformat(),
    }

    # ── Command reference card (always last embed) ──
    # Build dynamic quick-trade commands from top setups
    quick_cmds = []
    if bullish:
        quick_cmds.append(f"`!options {bullish[0]['symbol']}` — Top long setup flow")
    if bearish:
        quick_cmds.append(f"`!options {bearish[0]['symbol']}` — Top short setup flow")
    if best_setup:
        quick_cmds.append(f"`!options {best_setup['symbol']}` — 🏆 Best overall setup")

    ref_lines = [
        "**Dig deeper on any setup:**",
        "`!options SYMBOL` — Full options breakdown (IV, flow, OI walls, unusual)",
        "`!scan SYMBOL` — Technical scanner (levels, VWAP, volume profile)",
        "",
        "**Run more scans:**",
        "`!fullscan` — Rerun this full 72-symbol scan + options",
        "`!unusual` — Quick unusual options activity check",
        "`!unusual AAPL,NVDA,TSLA` — Check specific tickers",
        "",
        "**Other:**",
        "`!brief` — Market pulse (SPY, QQQ, IWM, DIA)",
        "`!price SYMBOL` — Quick price check",
        "`!alerts` — View active alerts",
    ]
    if quick_cmds:
        ref_lines.insert(0, "**⚡ Quick trades from this scan:**")
        for qc in quick_cmds:
            ref_lines.insert(1 + quick_cmds.index(qc), qc)
        ref_lines.insert(1 + len(quick_cmds), "")

    embeds.append({
        "title": "🤖 What's Next? — Command Reference",
        "description": "\n".join(ref_lines),
        "color": 0x334155,
        "footer": {"text": "Type any command in this channel • SEF Trading Terminal"}
    })

    # ── Send to Discord (up to 10 embeds per message) ──
    if webhook:
        # Discord max is 10 embeds per message, we have 3-5
        payload = {"username": "SEF Scanner", "embeds": embeds[:10]}
        try:
            resp = httpx.post(WEBHOOK_URL, json=payload, timeout=15)
            if not quiet:
                print(f"Discord: {resp.status_code} {'✅' if resp.status_code in (200,204) else '❌'}")
        except Exception as e:
            if not quiet:
                print(f"Discord error: {e}")

    if not quiet:
        print(f"\nDone — {total_setups} setups, {len(options_data)} with options data")

    return result_data


# ── Unusual Activity Alert (for polling) ──
def check_unusual_activity(symbols=None, min_flow=60, min_unusual=3):
    """
    Quick check for unusual options activity across symbols.
    Returns list of tickers with high flow scores + unusual contracts.
    Designed for frequent polling (lightweight).
    """
    if symbols is None:
        # Default to high-liquidity names
        symbols = ['SPY','QQQ','AAPL','MSFT','NVDA','AMD','META','TSLA',
                    'AMZN','GOOGL','NFLX','COIN','XOM','JPM','MARA']

    alerts = []
    for sym in symbols:
        try:
            r = options_scan_single(sym, dte_max=14, strike_range=0.08)
            if r.get("error"):
                continue

            flow = r.get("flowScore", 0)
            unusual = r.get("unusualCount", 0)
            sentiment = r.get("sentiment", "NEUTRAL")

            if flow >= min_flow or unusual >= min_unusual:
                alerts.append({
                    "symbol": sym,
                    "flowScore": flow,
                    "sentiment": sentiment,
                    "unusualCount": unusual,
                    "avgIVPct": r.get("avgIVPct"),
                    "totalVolume": r.get("totalVolume"),
                    "pcVolRatio": r.get("pcVolumeRatio"),
                    "topContracts": r.get("unusualContracts", [])[:3],
                    "maxPain": r.get("maxPain"),
                    "price": r.get("price"),
                })
            time.sleep(0.2)
        except Exception:
            pass

    alerts.sort(key=lambda x: -x["flowScore"])
    return alerts


def push_unusual_alerts(alerts, quiet=False):
    """Push unusual activity alerts to Discord."""
    if not alerts:
        return

    now_et = datetime.now(ET)
    fields = []
    for a in alerts[:10]:
        sentiment_map = {"BULLISH": "🟢", "LEAN BULLISH": "🟢", "BEARISH": "🔴",
                         "LEAN BEARISH": "🔴", "NEUTRAL": "⚪"}
        emoji = sentiment_map.get(a["sentiment"], "⚪")

        line = (f"Flow: **{a['flowScore']}** | {emoji} {a['sentiment']} | "
                f"Unusual: **{a['unusualCount']}** | Vol: {a['totalVolume']:,}")
        if a.get("avgIVPct"):
            line += f" | IV: {a['avgIVPct']:.0f}%"
        if a.get("topContracts"):
            c = a["topContracts"][0]
            line += f"\n└ 🔥 ${c['strike']} {c['type'].upper()} {c['expiration']} — Vol:{c['volume']:,} (V/OI:{c['volOiRatio']}x)"

        fields.append({"name": f"⚡ {a['symbol']} — ${a['price']:.2f}", "value": line, "inline": False})

    embed = {
        "title": f"⚡ Unusual Options Activity — {now_et.strftime('%I:%M %p ET')}",
        "color": 0xFFAA00,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "SEF Trading Terminal — Options Flow Alert"}
    }

    # Command reference card
    quick_cmds = [f"`!options {a['symbol']}`" for a in alerts[:3]]
    ref_text = (
        "**⚡ Drill into these:**\n" + " • ".join(quick_cmds) + "\n\n"
        "**Commands:**\n"
        "`!options SYMBOL` — Full options breakdown (IV, flow, OI walls, unusual)\n"
        "`!scan SYMBOL` — Technical scanner (levels, VWAP, volume profile)\n"
        "`!fullscan` — Full 72-symbol scan + options overlay\n"
        "`!unusual` — Re-check unusual activity\n"
        "`!brief` — Market pulse • `!price SYMBOL` — Quick price"
    )
    ref_embed = {
        "title": "🤖 What's Next? — Command Reference",
        "description": ref_text,
        "color": 0x334155,
        "footer": {"text": "Type any command in this channel • SEF Trading Terminal"}
    }

    payload = {"username": "SEF Options Alert", "embeds": [embed, ref_embed]}
    try:
        resp = httpx.post(WEBHOOK_URL, json=payload, timeout=15)
        if not quiet:
            print(f"Unusual alerts Discord: {resp.status_code}")
    except Exception as e:
        if not quiet:
            print(f"Discord error: {e}")


# ── CLI Entry Point ──
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "unusual":
        print("Checking unusual options activity...")
        alerts = check_unusual_activity()
        print(f"Found {len(alerts)} unusual tickers")
        for a in alerts:
            print(f"  {a['symbol']:6s} Flow:{a['flowScore']} Unusual:{a['unusualCount']} {a['sentiment']}")
        if alerts:
            push_unusual_alerts(alerts)
    else:
        run_full_scan(timeframe="2HR", include_options=True, webhook=True)
