"""
Options Flow Scanner
====================
Scans multiple tickers for unusual options activity using Polygon.io
Options Snapshot API (Greeks, IV, OI, Volume).

Surfaces:
  - Put/Call volume & OI ratios (sentiment)
  - Unusual activity (volume >> OI)
  - IV levels & skew
  - Expected move (nearest ATM straddle)
  - Max pain strike
  - OI walls (highest open interest strikes)
  - Top volume contracts

Author: Rob's Trading Systems
"""

import asyncio
import math
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from polygon_options import (
    fetch_options_snapshot_filtered,
    parse_contract,
    group_by_expiration,
)


# ── Preset Watchlists (centralized) ──
from universe import OPTIONS_PRESETS as PRESETS


def scan_tickers(
    symbols: List[str],
    dte_max: int = 45,
    strike_range: float = 0.15,
    max_workers: int = 3,
) -> Dict:
    """Scan multiple tickers for options flow in parallel."""
    results = []
    errors = []
    clean = [s.strip().upper() for s in symbols if s.strip()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_scan_single, sym, dte_max, strike_range): sym
            for sym in clean
        }
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                r = future.result(timeout=45)
                if r.get("error"):
                    errors.append({"ticker": sym, "error": r["error"]})
                else:
                    results.append(r)
            except Exception as e:
                errors.append({"ticker": sym, "error": str(e)})

    # Sort by flow score descending
    results.sort(key=lambda r: r.get("flowScore", 0), reverse=True)

    return {
        "results": results,
        "errors": errors,
        "meta": {
            "scanned": len(clean),
            "returned": len(results),
            "failed": len(errors),
            "dteMax": dte_max,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def _scan_single(symbol: str, dte_max: int = 45, strike_range: float = 0.15) -> Dict:
    """Scan a single ticker's options chain."""
    try:
        raw = fetch_options_snapshot_filtered(
            symbol, dte_min=0, dte_max=dte_max, strike_range_pct=strike_range
        )

        if not raw.get("contracts"):
            return {"ticker": symbol, "error": "No options data returned"}

        price = raw.get("underlyingPrice")
        contracts = [parse_contract(c) for c in raw["contracts"]]

        if not price:
            # Try from first contract
            for c in contracts:
                if c.get("underlyingPrice"):
                    price = c["underlyingPrice"]
                    break
        if not price:
            return {"ticker": symbol, "error": "No underlying price"}

        # Separate calls and puts
        calls = [c for c in contracts if c["contractType"] == "call"]
        puts = [c for c in contracts if c["contractType"] == "put"]

        # ── Volume metrics ──
        call_vol = sum(c["dayVolume"] or 0 for c in calls)
        put_vol = sum(c["dayVolume"] or 0 for c in puts)
        total_vol = call_vol + put_vol
        pc_vol_ratio = put_vol / call_vol if call_vol > 0 else 0

        # ── OI metrics ──
        call_oi = sum(c["openInterest"] or 0 for c in calls)
        put_oi = sum(c["openInterest"] or 0 for c in puts)
        total_oi = call_oi + put_oi
        pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 0

        # ── IV metrics (ATM contracts) ──
        atm_range = price * 0.03  # within 3% of current price
        atm_contracts = [c for c in contracts if abs(c["strike"] - price) <= atm_range and c.get("iv")]
        avg_iv = _avg([c["iv"] for c in atm_contracts]) if atm_contracts else None

        atm_calls = [c for c in atm_contracts if c["contractType"] == "call"]
        atm_puts = [c for c in atm_contracts if c["contractType"] == "put"]
        call_iv = _avg([c["iv"] for c in atm_calls]) if atm_calls else None
        put_iv = _avg([c["iv"] for c in atm_puts]) if atm_puts else None
        iv_skew = (put_iv - call_iv) if (put_iv and call_iv) else None

        # ── Expected Move (nearest expiry ATM straddle) ──
        by_exp = group_by_expiration(contracts)
        nearest_exp = list(by_exp.keys())[0] if by_exp else None
        expected_move_pct = None
        expected_move_usd = None
        straddle_price = None
        dte_nearest = None

        if nearest_exp:
            exp_contracts = by_exp[nearest_exp]
            dte_nearest = _calc_dte(nearest_exp)

            # Find nearest ATM call + put in this expiry
            atm_call = _find_nearest_atm(exp_contracts, price, "call")
            atm_put = _find_nearest_atm(exp_contracts, price, "put")

            if atm_call and atm_put:
                c_price = atm_call.get("midpoint") or atm_call.get("lastPrice") or 0
                p_price = atm_put.get("midpoint") or atm_put.get("lastPrice") or 0
                straddle_price = c_price + p_price
                if straddle_price > 0 and price > 0:
                    expected_move_pct = straddle_price / price
                    expected_move_usd = straddle_price

        # ── Max Pain ──
        max_pain = _calc_max_pain(contracts, price)

        # ── Greeks summary (ATM) ──
        avg_delta_call = _avg([c["delta"] for c in atm_calls if c.get("delta")]) if atm_calls else None
        avg_gamma = _avg([c["gamma"] for c in atm_contracts if c.get("gamma")]) if atm_contracts else None
        avg_theta = _avg([c["theta"] for c in atm_contracts if c.get("theta")]) if atm_contracts else None
        avg_vega = _avg([c["vega"] for c in atm_contracts if c.get("vega")]) if atm_contracts else None

        # ── Unusual Activity (volume >> OI) ──
        unusual = []
        for c in contracts:
            vol = c["dayVolume"] or 0
            oi = c["openInterest"] or 0
            if vol > 0 and (oi == 0 or vol / max(oi, 1) >= 1.5) and vol >= 100:
                unusual.append({
                    "strike": c["strike"],
                    "expiration": c["expiration"],
                    "type": c["contractType"],
                    "volume": vol,
                    "oi": oi,
                    "volOiRatio": round(vol / max(oi, 1), 1),
                    "iv": round(c["iv"], 4) if c.get("iv") else None,
                    "delta": round(c["delta"], 3) if c.get("delta") else None,
                    "midpoint": c.get("midpoint"),
                    "lastPrice": c.get("lastPrice"),
                })
        unusual.sort(key=lambda x: x["volume"], reverse=True)

        # ── OI Walls ──
        oi_by_strike = {}
        for c in contracts:
            s = c["strike"]
            oi_by_strike.setdefault(s, {"call_oi": 0, "put_oi": 0, "total_oi": 0})
            oi = c["openInterest"] or 0
            if c["contractType"] == "call":
                oi_by_strike[s]["call_oi"] += oi
            else:
                oi_by_strike[s]["put_oi"] += oi
            oi_by_strike[s]["total_oi"] += oi

        oi_walls = sorted(
            [{"strike": k, **v} for k, v in oi_by_strike.items()],
            key=lambda x: x["total_oi"], reverse=True,
        )[:8]

        # ── Top Volume Contracts ──
        top_calls = sorted(calls, key=lambda c: c["dayVolume"] or 0, reverse=True)[:5]
        top_puts = sorted(puts, key=lambda c: c["dayVolume"] or 0, reverse=True)[:5]

        top_vol_calls = [_summarize_contract(c) for c in top_calls if (c["dayVolume"] or 0) > 0]
        top_vol_puts = [_summarize_contract(c) for c in top_puts if (c["dayVolume"] or 0) > 0]

        # ── Flow Score & Sentiment ──
        flow_score = _calc_flow_score(total_vol, total_oi, len(unusual), pc_vol_ratio, avg_iv)
        sentiment = _calc_sentiment(pc_vol_ratio, pc_oi_ratio, unusual)
        iv_level = _iv_level(avg_iv)

        return {
            "ticker": symbol,
            "price": round(price, 2),
            "error": None,

            # Volume
            "callVolume": call_vol,
            "putVolume": put_vol,
            "totalVolume": total_vol,
            "pcVolumeRatio": round(pc_vol_ratio, 2),

            # OI
            "callOI": call_oi,
            "putOI": put_oi,
            "totalOI": total_oi,
            "pcOIRatio": round(pc_oi_ratio, 2),

            # IV
            "avgIV": round(avg_iv, 4) if avg_iv else None,
            "avgIVPct": round(avg_iv * 100, 1) if avg_iv else None,
            "callIV": round(call_iv, 4) if call_iv else None,
            "putIV": round(put_iv, 4) if put_iv else None,
            "ivSkew": round(iv_skew, 4) if iv_skew else None,
            "ivLevel": iv_level,

            # Expected move
            "expectedMovePct": round(expected_move_pct * 100, 2) if expected_move_pct else None,
            "expectedMoveUSD": round(expected_move_usd, 2) if expected_move_usd else None,
            "straddlePrice": round(straddle_price, 2) if straddle_price else None,
            "nearestExpiry": nearest_exp,
            "nearestDTE": dte_nearest,

            # Max pain
            "maxPain": max_pain,

            # Greeks (ATM avg)
            "avgDeltaCall": round(avg_delta_call, 3) if avg_delta_call else None,
            "avgGamma": round(avg_gamma, 4) if avg_gamma else None,
            "avgTheta": round(avg_theta, 4) if avg_theta else None,
            "avgVega": round(avg_vega, 4) if avg_vega else None,

            # Unusual activity
            "unusualCount": len(unusual),
            "unusualContracts": unusual[:10],

            # OI walls
            "oiWalls": oi_walls,

            # Top volume
            "topCalls": top_vol_calls,
            "topPuts": top_vol_puts,

            # Scores
            "flowScore": flow_score,
            "sentiment": sentiment,

            # Meta
            "contractsAnalyzed": len(contracts),
            "expirations": list(by_exp.keys()),
        }

    except Exception as e:
        return {"ticker": symbol, "error": str(e), "flowScore": 0}


# ── Scoring ──

def _calc_flow_score(total_vol, total_oi, unusual_count, pc_ratio, avg_iv):
    """0-100 score reflecting how much unusual options activity there is."""
    score = 0

    # Volume intensity
    if total_vol >= 50000:
        score += 25
    elif total_vol >= 20000:
        score += 20
    elif total_vol >= 5000:
        score += 15
    elif total_vol >= 1000:
        score += 8

    # Volume/OI ratio (higher = more new positions being opened)
    # If total vol is a large fraction of total OI, that's unusual
    if total_oi > 0:
        vol_oi = total_vol / total_oi
        if vol_oi > 0.50:
            score += 25
        elif vol_oi > 0.30:
            score += 20
        elif vol_oi > 0.15:
            score += 12
        elif vol_oi > 0.05:
            score += 5

    # Unusual contracts
    if unusual_count >= 10:
        score += 25
    elif unusual_count >= 5:
        score += 18
    elif unusual_count >= 3:
        score += 12
    elif unusual_count >= 1:
        score += 6

    # Extreme P/C ratio (either direction signals conviction)
    if pc_ratio > 2.0 or pc_ratio < 0.3:
        score += 15
    elif pc_ratio > 1.5 or pc_ratio < 0.5:
        score += 10
    elif pc_ratio > 1.2 or pc_ratio < 0.7:
        score += 5

    # High IV amplification
    if avg_iv and avg_iv > 0.80:
        score += 10
    elif avg_iv and avg_iv > 0.50:
        score += 5

    return min(score, 100)


def _calc_sentiment(pc_vol_ratio, pc_oi_ratio, unusual):
    """Determine overall sentiment from options flow."""
    signals = []

    # Volume ratio
    if pc_vol_ratio > 1.5:
        signals.append("BEARISH")
    elif pc_vol_ratio < 0.5:
        signals.append("BULLISH")
    else:
        signals.append("NEUTRAL")

    # OI ratio
    if pc_oi_ratio > 1.3:
        signals.append("BEARISH")
    elif pc_oi_ratio < 0.6:
        signals.append("BULLISH")
    else:
        signals.append("NEUTRAL")

    # Unusual flow direction
    if unusual:
        call_unusual = sum(1 for u in unusual if u["type"] == "call")
        put_unusual = sum(1 for u in unusual if u["type"] == "put")
        if call_unusual > put_unusual * 2:
            signals.append("BULLISH")
        elif put_unusual > call_unusual * 2:
            signals.append("BEARISH")

    bullish = signals.count("BULLISH")
    bearish = signals.count("BEARISH")

    if bullish >= 2:
        return "BULLISH"
    if bearish >= 2:
        return "BEARISH"
    if bullish > bearish:
        return "LEAN BULLISH"
    if bearish > bullish:
        return "LEAN BEARISH"
    return "NEUTRAL"


def _iv_level(avg_iv):
    if not avg_iv:
        return "N/A"
    if avg_iv > 1.0:
        return "EXTREME"
    if avg_iv > 0.60:
        return "HIGH"
    if avg_iv > 0.40:
        return "ELEVATED"
    if avg_iv > 0.20:
        return "NORMAL"
    return "LOW"


# ── Max Pain ──

def _calc_max_pain(contracts, price):
    """Find strike where total option holder losses are maximized (max pain)."""
    strikes = sorted(set(c["strike"] for c in contracts))
    if not strikes:
        return None

    min_pain = float("inf")
    max_pain_strike = None

    for test_strike in strikes:
        total_pain = 0
        for c in contracts:
            oi = c["openInterest"] or 0
            if oi == 0:
                continue
            if c["contractType"] == "call":
                # Call holder pain = max(0, strike - test_price) * OI * 100
                pain = max(0, c["strike"] - test_strike) * oi
            else:
                # Put holder pain = max(0, test_price - strike) * OI * 100
                pain = max(0, test_strike - c["strike"]) * oi
            total_pain += pain

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike

    return max_pain_strike


# ── Helpers ──

def _avg(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _find_nearest_atm(contracts, price, contract_type):
    """Find the contract closest to ATM for a given type."""
    filtered = [c for c in contracts if c["contractType"] == contract_type]
    if not filtered:
        return None
    return min(filtered, key=lambda c: abs(c["strike"] - price))


def _calc_dte(exp_str):
    try:
        exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        return (exp - today).days
    except Exception:
        return None


def _summarize_contract(c):
    return {
        "strike": c["strike"],
        "expiration": c["expiration"],
        "type": c["contractType"],
        "volume": c["dayVolume"] or 0,
        "oi": c["openInterest"] or 0,
        "iv": round(c["iv"], 4) if c.get("iv") else None,
        "delta": round(c["delta"], 3) if c.get("delta") else None,
        "midpoint": c.get("midpoint"),
        "lastPrice": c.get("lastPrice"),
        "bid": c.get("bid"),
        "ask": c.get("ask"),
    }


# ── Async Wrapper ──

async def async_scan_tickers(symbols: List[str], **kwargs) -> Dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: scan_tickers(symbols, **kwargs))
