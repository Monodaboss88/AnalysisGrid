"""
Options Best Setup Finder
=========================
Pulls options data for the top scan results and ranks
the single best high-value trade setup.

Uses: options_flow_scanner → polygon_options → Polygon.io API
"""
import os, json, time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from options_flow_scanner import scan_tickers

ET = ZoneInfo("America/New_York")

# Top scan picks from our latest All-universe 2HR scan
# Longs (high conviction)
LONG_PICKS = ["XLE", "SQQQ", "MRK", "XBI", "PYPL", "MARA", "RIOT", "SHOP", "NET", "UNH", "XOM", "JNJ", "TLT"]
# Shorts (high conviction)
SHORT_PICKS = ["QQQ", "HD", "META", "NFLX", "WMT", "BAC", "TQQQ", "MA", "IBIT", "TSLA", "AVGO", "SPY", "DIA", "AAPL", "AMD"]

ALL_PICKS = LONG_PICKS + SHORT_PICKS
DIRECTION_MAP = {}
for s in LONG_PICKS:
    DIRECTION_MAP[s] = "LONG"
for s in SHORT_PICKS:
    DIRECTION_MAP[s] = "SHORT"

print(f"Pulling options data for {len(ALL_PICKS)} top scan picks...")
print(f"  Longs:  {', '.join(LONG_PICKS)}")
print(f"  Shorts: {', '.join(SHORT_PICKS)}")
print()

# Scan options — 30 DTE window, 15% strike range
data = scan_tickers(ALL_PICKS, dte_max=30, strike_range=0.10, max_workers=3)

results = data["results"]
errors = data["errors"]

print(f"\n{'='*70}")
print(f"OPTIONS DATA: {len(results)} tickers returned, {len(errors)} errors")
print(f"{'='*70}\n")

if errors:
    print(f"Errors: {', '.join(e['ticker'] for e in errors)}\n")

# ───── Score each ticker for "best setup" value ─────
# We want:
#  1. High flow score (unusual activity)
#  2. Sentiment ALIGNED with our scanner direction
#  3. Reasonable IV (not too crushed, not insane)
#  4. Good expected move vs straddle cost
#  5. Unusual contracts in our direction
#  6. Volume to actually trade

ranked = []

for r in results:
    sym = r["ticker"]
    direction = DIRECTION_MAP.get(sym, "?")
    
    # Base: flow score (0-100)
    flow = r.get("flowScore", 0)
    
    # Sentiment alignment bonus
    sentiment = r.get("sentiment", "NEUTRAL")
    alignment = 0
    if direction == "LONG" and "BULLISH" in sentiment:
        alignment = 20
    elif direction == "LONG" and sentiment == "LEAN BULLISH":
        alignment = 10
    elif direction == "SHORT" and "BEARISH" in sentiment:
        alignment = 20
    elif direction == "SHORT" and sentiment == "LEAN BEARISH":
        alignment = 10
    elif direction == "LONG" and "BEARISH" in sentiment:
        alignment = -15  # Options flow disagrees
    elif direction == "SHORT" and "BULLISH" in sentiment:
        alignment = -15
    
    # IV level score
    iv_pct = r.get("avgIVPct")  # e.g. 35.0 = 35%
    iv_score = 0
    if iv_pct:
        if 20 <= iv_pct <= 50:
            iv_score = 10  # Sweet spot
        elif 50 < iv_pct <= 80:
            iv_score = 5   # Elevated but tradeable
        elif iv_pct > 80:
            iv_score = -5  # Too expensive
    
    # Expected move value (bigger move = more opportunity)
    em_pct = r.get("expectedMovePct", 0) or 0
    em_score = min(em_pct * 3, 15)  # Cap at 15
    
    # Unusual activity bonus
    unusual = r.get("unusualCount", 0)
    unusual_score = min(unusual * 3, 15)
    
    # Directional unusual contracts
    unusual_contracts = r.get("unusualContracts", [])
    dir_unusual = 0
    if direction == "LONG":
        dir_unusual = sum(1 for u in unusual_contracts if u["type"] == "call")
    else:
        dir_unusual = sum(1 for u in unusual_contracts if u["type"] == "put")
    dir_unusual_score = min(dir_unusual * 5, 15)
    
    # Total volume (liquidity)
    vol = r.get("totalVolume", 0)
    vol_score = 0
    if vol >= 50000: vol_score = 10
    elif vol >= 20000: vol_score = 8
    elif vol >= 5000: vol_score = 5
    elif vol >= 1000: vol_score = 2
    
    # Composite setup score
    setup_score = flow + alignment + iv_score + em_score + unusual_score + dir_unusual_score + vol_score
    
    ranked.append({
        "symbol": sym,
        "direction": direction,
        "price": r.get("price"),
        "setup_score": round(setup_score, 1),
        "flow_score": flow,
        "sentiment": sentiment,
        "alignment": alignment,
        "iv_pct": iv_pct,
        "iv_level": r.get("ivLevel"),
        "expected_move_pct": em_pct,
        "expected_move_usd": r.get("expectedMoveUSD"),
        "straddle_price": r.get("straddlePrice"),
        "max_pain": r.get("maxPain"),
        "total_volume": vol,
        "total_oi": r.get("totalOI"),
        "pc_vol_ratio": r.get("pcVolumeRatio"),
        "pc_oi_ratio": r.get("pcOIRatio"),
        "unusual_count": unusual,
        "dir_unusual": dir_unusual,
        "nearest_dte": r.get("nearestDTE"),
        "top_calls": r.get("topCalls", [])[:3],
        "top_puts": r.get("topPuts", [])[:3],
        "unusual_contracts": unusual_contracts[:5],
        "oi_walls": r.get("oiWalls", [])[:4],
    })

ranked.sort(key=lambda x: -x["setup_score"])

# ───── Print Rankings ─────
print(f"\n{'='*70}")
print(f"  RANKED OPTIONS SETUPS (Best → Worst)")
print(f"{'='*70}")

for i, r in enumerate(ranked):
    dir_emoji = "🟢" if r["direction"] == "LONG" else "🔴"
    align_tag = "✓ ALIGNED" if r["alignment"] > 0 else ("✗ CONFLICT" if r["alignment"] < 0 else "~ NEUTRAL")
    
    print(f"\n{'─'*60}")
    print(f"  #{i+1}  {dir_emoji} {r['symbol']:6s}  ${r['price']:.2f}  │  Setup Score: {r['setup_score']:.0f}")
    print(f"       Direction: {r['direction']}  │  Sentiment: {r['sentiment']}  │  {align_tag}")
    print(f"       Flow: {r['flow_score']}  │  IV: {r['iv_pct']:.1f}% ({r['iv_level']})  │  ExpMove: {r['expected_move_pct']:.1f}%" if r['iv_pct'] else f"       Flow: {r['flow_score']}  │  IV: N/A  │  ExpMove: N/A")
    print(f"       Vol: {r['total_volume']:,}  │  OI: {r['total_oi']:,}  │  P/C Vol: {r['pc_vol_ratio']}  │  P/C OI: {r['pc_oi_ratio']}")
    print(f"       Unusual: {r['unusual_count']} total ({r['dir_unusual']} in-direction)  │  MaxPain: ${r['max_pain']}" if r['max_pain'] else f"       Unusual: {r['unusual_count']} total ({r['dir_unusual']} in-direction)")
    print(f"       Nearest DTE: {r['nearest_dte']}d  │  Straddle: ${r['straddle_price']}" if r['straddle_price'] else "")

# ───── Deep dive on #1 pick ─────
if ranked:
    best = ranked[0]
    print(f"\n{'='*70}")
    print(f"  🏆 BEST HIGH-VALUE SETUP: {best['symbol']}")
    print(f"{'='*70}")
    print(f"  Direction:      {best['direction']}")
    print(f"  Current Price:  ${best['price']:.2f}")
    print(f"  Setup Score:    {best['setup_score']:.0f} / ~120 max")
    print(f"  Flow Score:     {best['flow_score']} / 100")
    print(f"  Sentiment:      {best['sentiment']} ({'ALIGNED ✓' if best['alignment'] > 0 else 'WATCH ⚠️'})")
    print(f"  IV:             {best['iv_pct']:.1f}% ({best['iv_level']})" if best['iv_pct'] else "  IV:             N/A")
    print(f"  Expected Move:  {best['expected_move_pct']:.2f}% (${best['expected_move_usd']:.2f})" if best['expected_move_usd'] else "  Expected Move:  N/A")
    print(f"  Max Pain:       ${best['max_pain']:.2f}" if best['max_pain'] else "  Max Pain:       N/A")
    print(f"  Options Volume: {best['total_volume']:,}")
    print(f"  Open Interest:  {best['total_oi']:,}")
    print(f"  P/C Vol Ratio:  {best['pc_vol_ratio']}")
    print(f"  P/C OI Ratio:   {best['pc_oi_ratio']}")
    print(f"  Unusual Contracts: {best['unusual_count']} ({best['dir_unusual']} {best['direction'].lower()}-side)")
    
    # Suggested trade
    print(f"\n  ── SUGGESTED TRADE ──")
    if best['direction'] == "LONG":
        # For longs: buy calls or call debit spread
        calls = best.get('top_calls', [])
        if calls:
            c = calls[0]
            mid = c.get('midpoint') or c.get('lastPrice') or 0
            print(f"  Strategy:  BUY CALL")
            print(f"  Contract:  {best['symbol']} ${c['strike']} Call exp {c['expiration']}")
            print(f"  Est Cost:  ~${mid:.2f} per contract (${mid*100:.0f} per lot)")
            print(f"  Delta:     {c['delta']}" if c.get('delta') else "")
            print(f"  IV:        {c['iv']*100:.1f}%" if c.get('iv') else "")
            if best['expected_move_usd'] and mid > 0:
                target = mid + best['expected_move_usd'] * abs(c.get('delta', 0.5))
                pct_return = ((target - mid) / mid) * 100
                print(f"  Target:    ~${target:.2f} ({pct_return:.0f}% return if ExpMove hit)")
        else:
            print(f"  Strategy:  BUY CALLS (ATM, ~{best['nearest_dte']}d DTE)")
    else:
        # For shorts: buy puts or put debit spread
        puts = best.get('top_puts', [])
        if puts:
            p = puts[0]
            mid = p.get('midpoint') or p.get('lastPrice') or 0
            print(f"  Strategy:  BUY PUT")
            print(f"  Contract:  {best['symbol']} ${p['strike']} Put exp {p['expiration']}")
            print(f"  Est Cost:  ~${mid:.2f} per contract (${mid*100:.0f} per lot)")
            print(f"  Delta:     {p['delta']}" if p.get('delta') else "")
            print(f"  IV:        {p['iv']*100:.1f}%" if p.get('iv') else "")
            if best['expected_move_usd'] and mid > 0:
                target = mid + best['expected_move_usd'] * abs(p.get('delta', 0.5))
                pct_return = ((target - mid) / mid) * 100
                print(f"  Target:    ~${target:.2f} ({pct_return:.0f}% return if ExpMove hit)")
        else:
            print(f"  Strategy:  BUY PUTS (ATM, ~{best['nearest_dte']}d DTE)")
    
    # OI walls (support/resistance)
    if best['oi_walls']:
        print(f"\n  ── OI WALLS (S/R) ──")
        for w in best['oi_walls']:
            above_below = "ABOVE ↑" if w['strike'] > best['price'] else "BELOW ↓"
            print(f"  ${w['strike']:>8.2f}  │  Call OI: {w['call_oi']:>6,}  Put OI: {w['put_oi']:>6,}  │  {above_below}")
    
    # Unusual contracts detail
    if best['unusual_contracts']:
        print(f"\n  ── UNUSUAL ACTIVITY ──")
        for u in best['unusual_contracts']:
            print(f"  {u['type'].upper():4s} ${u['strike']} exp {u['expiration']} │ Vol:{u['volume']:,} OI:{u['oi']:,} (V/OI:{u['volOiRatio']}x) │ IV:{u['iv']*100:.0f}%" if u.get('iv') else f"  {u['type'].upper():4s} ${u['strike']} exp {u['expiration']} │ Vol:{u['volume']:,} OI:{u['oi']:,} (V/OI:{u['volOiRatio']}x)")

print(f"\n{'='*70}")
now = datetime.now(ET)
print(f"  Scan completed: {now.strftime('%b %d, %Y %I:%M %p ET')}")
print(f"{'='*70}")
