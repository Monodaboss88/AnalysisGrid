"""
Options Flow Scanner + MSP Pathway Engine
==========================================
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
  - GEX (Gamma Exposure) per strike
  - Order Blocks (institutional supply/demand zones)
  - MSP Pathway (OB entry → OI wall barriers → Max Pain target)

MSP = Max Pain System Pathway
  Combines options-driven targets (max pain, OI walls, GEX flip)
  with price-action-driven zones (order blocks) into a single
  trade pathway with confluence scoring.

Author: Rob's Trading Systems
"""

import asyncio
import math
import logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from polygon_options import (
    fetch_options_snapshot_filtered,
    parse_contract,
    group_by_expiration,
)

logger = logging.getLogger(__name__)


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

        # ── GEX (Gamma Exposure) ──
        gex = _calc_gex(contracts, price)

        # ── Order Blocks (institutional supply/demand zones from price action) ──
        order_blocks = _detect_order_blocks(symbol, price)

        # Build base result first (needed for MSP pathway)
        result = {
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

            # GEX
            "gex": gex,

            # Order Blocks
            "orderBlocks": order_blocks,

            # Meta
            "contractsAnalyzed": len(contracts),
            "expirations": list(by_exp.keys()),
        }

        # ── MSP Pathway (combines OB + OI walls + Max Pain + GEX + flow) ──
        msp = _build_msp_pathway(result)
        result["msp"] = msp

        return result

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


# ═══════════════════════════════════════════════════════
#  GEX — Gamma Exposure per Strike
# ═══════════════════════════════════════════════════════

def _calc_gex(contracts, price):
    """
    Calculate net Gamma Exposure (GEX) per strike.

    Dealer gamma: when dealers sell options, they're short gamma.
    - Sold call → dealer is short gamma → must buy dips, sell rips (stabilizing)
    - Sold put  → dealer is long gamma → must sell dips, buy rips (destabilizing)

    GEX per strike = (call_gamma * call_OI - put_gamma * put_OI) * 100 * price

    The GEX flip point is where net GEX crosses from positive (dealer stabilizing)
    to negative (dealer amplifying). Price tends to be attracted to positive GEX zones.

    Returns:
        {
            "gex_by_strike": [{strike, net_gex, call_gex, put_gex}, ...],
            "gex_flip": float or None (strike where net GEX flips sign),
            "total_gex": float,
            "dealer_position": "LONG_GAMMA" | "SHORT_GAMMA" | "NEUTRAL",
        }
    """
    if not contracts or not price:
        return {"gex_by_strike": [], "gex_flip": None, "total_gex": 0, "dealer_position": "NEUTRAL"}

    gex_map = {}  # strike → {call_gex, put_gex, net_gex}
    for c in contracts:
        gamma = c.get("gamma") or 0
        oi = c.get("openInterest") or 0
        strike = c["strike"]
        if gamma == 0 or oi == 0:
            continue

        # GEX = gamma * OI * 100 (shares per contract) * price (dollar-weighted)
        gex_value = gamma * oi * 100 * price

        if strike not in gex_map:
            gex_map[strike] = {"strike": strike, "call_gex": 0, "put_gex": 0, "net_gex": 0}

        if c["contractType"] == "call":
            gex_map[strike]["call_gex"] += gex_value
            gex_map[strike]["net_gex"] += gex_value   # calls = positive GEX (dealer short)
        else:
            gex_map[strike]["put_gex"] += gex_value
            gex_map[strike]["net_gex"] -= gex_value    # puts = negative GEX (dealer long)

    if not gex_map:
        return {"gex_by_strike": [], "gex_flip": None, "total_gex": 0, "dealer_position": "NEUTRAL"}

    gex_list = sorted(gex_map.values(), key=lambda x: x["strike"])

    # Round for readability
    for g in gex_list:
        g["call_gex"] = round(g["call_gex"])
        g["put_gex"] = round(g["put_gex"])
        g["net_gex"] = round(g["net_gex"])

    total_gex = sum(g["net_gex"] for g in gex_list)

    # Find GEX flip point (where net_gex changes sign nearest to price)
    gex_flip = None
    # Filter to strikes near price (within 15%)
    near_strikes = [g for g in gex_list if abs(g["strike"] - price) / price <= 0.15]
    for i in range(len(near_strikes) - 1):
        a, b = near_strikes[i], near_strikes[i + 1]
        if a["net_gex"] * b["net_gex"] < 0:  # sign change
            # Interpolate between the two strikes
            gex_flip = round((a["strike"] + b["strike"]) / 2, 2)
            break

    if total_gex > 0:
        dealer = "LONG_GAMMA"     # dealers stabilize — price tends to mean-revert
    elif total_gex < 0:
        dealer = "SHORT_GAMMA"    # dealers amplify — price tends to trend/break out
    else:
        dealer = "NEUTRAL"

    # Return top 10 by absolute GEX magnitude
    top_gex = sorted(gex_list, key=lambda x: abs(x["net_gex"]), reverse=True)[:10]

    return {
        "gex_by_strike": top_gex,
        "gex_flip": gex_flip,
        "total_gex": total_gex,
        "dealer_position": dealer,
    }


# ═══════════════════════════════════════════════════════
#  ORDER BLOCKS — Institutional Supply/Demand Zones
# ═══════════════════════════════════════════════════════

def _detect_order_blocks(symbol, price, lookback_days=60):
    """
    Detect institutional order blocks from price action.

    Bullish OB = last down candle before a strong rally (unfilled buy orders)
    Bearish OB = last up candle before a strong dump (unfilled sell orders)

    Detection criteria:
      1. Find displacement moves (candles with body > 1.5x ATR)
      2. Look at the candle immediately before the displacement
      3. That candle's range becomes the order block zone
      4. Only keep OBs that haven't been fully mitigated (price revisited and closed through)

    Returns:
        [
            {
                "type": "bullish" | "bearish",
                "zone_low": float,
                "zone_high": float,
                "zone_mid": float,
                "strength": 0-100,
                "age_days": int,
                "tested": bool,
                "date": str,
            },
            ...
        ]
    """
    try:
        from polygon_data import get_bars
    except ImportError:
        logger.debug("polygon_data not available for order block detection")
        return []

    try:
        df = get_bars(symbol, period=f"{lookback_days}d", interval="1d")
        if df is None or df.empty or len(df) < 20:
            return []
    except Exception as e:
        logger.debug(f"Order block detection failed for {symbol}: {e}")
        return []

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(closes)

    # ATR (14-period)
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < 14:
        return []
    atr = sum(trs[-14:]) / 14

    if atr == 0:
        return []

    # Displacement threshold: body > 1.5x ATR
    displacement_mult = 1.5
    order_blocks = []
    dates = df.index

    for i in range(2, n):
        body = abs(closes[i] - opens[i])
        if body < atr * displacement_mult:
            continue

        # This is a displacement candle
        prev_open = opens[i - 1]
        prev_close = closes[i - 1]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]

        is_bullish_displacement = closes[i] > opens[i]  # big green candle
        is_bearish_displacement = closes[i] < opens[i]  # big red candle

        # Previous candle should be opposite or consolidation
        if is_bullish_displacement and prev_close <= prev_open:
            # Bullish OB: previous down candle before rally
            zone_low = prev_low
            zone_high = prev_open  # body high of down candle
            zone_mid = (zone_low + zone_high) / 2

            # Check if OB has been mitigated (price closed below zone_low after OB formed)
            mitigated = False
            tested = False
            for j in range(i + 1, n):
                if lows[j] <= zone_mid:
                    tested = True
                if closes[j] < zone_low:
                    mitigated = True
                    break

            if not mitigated:
                # Strength: size of displacement + recency + untested bonus
                strength = min(100, int(
                    (body / atr) * 20 +          # displacement strength (0-60)
                    max(0, 30 - (n - i)) +        # recency bonus (0-30)
                    (10 if not tested else 0)      # untested bonus
                ))
                age_days = (n - 1 - i)
                try:
                    ob_date = str(dates[i - 1].date()) if hasattr(dates[i - 1], 'date') else str(dates[i - 1])[:10]
                except Exception:
                    ob_date = f"{age_days}d ago"

                order_blocks.append({
                    "type": "bullish",
                    "zone_low": round(zone_low, 2),
                    "zone_high": round(zone_high, 2),
                    "zone_mid": round(zone_mid, 2),
                    "strength": strength,
                    "age_days": age_days,
                    "tested": tested,
                    "date": ob_date,
                })

        elif is_bearish_displacement and prev_close >= prev_open:
            # Bearish OB: previous up candle before dump
            zone_low = prev_open   # body low of up candle
            zone_high = prev_high
            zone_mid = (zone_low + zone_high) / 2

            mitigated = False
            tested = False
            for j in range(i + 1, n):
                if highs[j] >= zone_mid:
                    tested = True
                if closes[j] > zone_high:
                    mitigated = True
                    break

            if not mitigated:
                strength = min(100, int(
                    (body / atr) * 20 +
                    max(0, 30 - (n - i)) +
                    (10 if not tested else 0)
                ))
                age_days = (n - 1 - i)
                try:
                    ob_date = str(dates[i - 1].date()) if hasattr(dates[i - 1], 'date') else str(dates[i - 1])[:10]
                except Exception:
                    ob_date = f"{age_days}d ago"

                order_blocks.append({
                    "type": "bearish",
                    "zone_low": round(zone_low, 2),
                    "zone_high": round(zone_high, 2),
                    "zone_mid": round(zone_mid, 2),
                    "strength": strength,
                    "age_days": age_days,
                    "tested": tested,
                    "date": ob_date,
                })

    # Sort by strength descending, keep top 4
    order_blocks.sort(key=lambda x: x["strength"], reverse=True)
    return order_blocks[:4]


# ═══════════════════════════════════════════════════════
#  MSP PATHWAY — OB Entry → OI Walls → Max Pain Target
# ═══════════════════════════════════════════════════════

def _build_msp_pathway(ticker_data):
    """
    Build the MSP (Max Pain System Pathway) for a scanned ticker.

    Combines:
      - Order blocks (entry zones)
      - OI walls (barriers/magnets)
      - Max pain (destination)
      - GEX flip (dealer positioning)
      - Unusual flow (directional confirmation)

    Returns:
        {
            "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
            "entry_zone": {"low": float, "high": float, "type": str} or None,
            "barriers": [{"strike": float, "type": str, "strength": str}],
            "target": {"price": float, "type": str},
            "confluence_score": 0-100,
            "pathway_text": str,       # One-line readable pathway
            "signals": [str],          # Supporting evidence
        }
    """
    price = ticker_data.get("price", 0)
    max_pain = ticker_data.get("maxPain")
    sentiment = ticker_data.get("sentiment", "NEUTRAL")
    oi_walls = ticker_data.get("oiWalls", [])
    order_blocks = ticker_data.get("orderBlocks", [])
    gex = ticker_data.get("gex", {})
    unusual = ticker_data.get("unusualContracts", [])
    gex_flip = gex.get("gex_flip")
    dealer_pos = gex.get("dealer_position", "NEUTRAL")

    if not price or not max_pain:
        return {"direction": "NEUTRAL", "entry_zone": None, "barriers": [],
                "target": None, "confluence_score": 0, "pathway_text": "Insufficient data",
                "signals": []}

    # ── Determine direction ──
    direction_signals = []
    confluence = 0

    # Max pain direction
    mp_diff_pct = (max_pain - price) / price * 100
    if mp_diff_pct > 1.0:
        direction_signals.append("BULLISH")
        confluence += 15
    elif mp_diff_pct < -1.0:
        direction_signals.append("BEARISH")
        confluence += 15
    else:
        direction_signals.append("NEUTRAL")
        confluence += 5

    # Sentiment from flow
    if sentiment in ("BULLISH", "LEAN BULLISH"):
        direction_signals.append("BULLISH")
        confluence += 15 if sentiment == "BULLISH" else 10
    elif sentiment in ("BEARISH", "LEAN BEARISH"):
        direction_signals.append("BEARISH")
        confluence += 15 if sentiment == "BEARISH" else 10

    # Unusual flow direction
    call_unusual = sum(1 for u in unusual if u.get("type") == "call")
    put_unusual = sum(1 for u in unusual if u.get("type") == "put")
    if call_unusual > put_unusual * 1.5 and call_unusual >= 2:
        direction_signals.append("BULLISH")
        confluence += 10
    elif put_unusual > call_unusual * 1.5 and put_unusual >= 2:
        direction_signals.append("BEARISH")
        confluence += 10

    # GEX flip position
    if gex_flip:
        if price < gex_flip:
            direction_signals.append("BULLISH")  # price below flip → likely to rally toward it
            confluence += 10
        elif price > gex_flip:
            direction_signals.append("BEARISH")
            confluence += 10

    # Tally
    bull_count = direction_signals.count("BULLISH")
    bear_count = direction_signals.count("BEARISH")
    if bull_count > bear_count:
        direction = "BULLISH"
    elif bear_count > bull_count:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # ── Find entry zone (nearest relevant order block) ──
    entry_zone = None
    signals = []

    if order_blocks:
        if direction == "BULLISH":
            # Look for bullish OB below or near price
            bull_obs = [ob for ob in order_blocks if ob["type"] == "bullish" and ob["zone_high"] <= price * 1.02]
            if bull_obs:
                best = max(bull_obs, key=lambda x: x["strength"])
                entry_zone = {"low": best["zone_low"], "high": best["zone_high"],
                              "type": "Bullish OB", "strength": best["strength"],
                              "tested": best["tested"], "date": best["date"]}
                confluence += 15
                signals.append(f"Bullish OB at ${best['zone_low']:.0f}-${best['zone_high']:.0f} ({best['strength']}% str)")
        elif direction == "BEARISH":
            bear_obs = [ob for ob in order_blocks if ob["type"] == "bearish" and ob["zone_low"] >= price * 0.98]
            if bear_obs:
                best = max(bear_obs, key=lambda x: x["strength"])
                entry_zone = {"low": best["zone_low"], "high": best["zone_high"],
                              "type": "Bearish OB", "strength": best["strength"],
                              "tested": best["tested"], "date": best["date"]}
                confluence += 15
                signals.append(f"Bearish OB at ${best['zone_low']:.0f}-${best['zone_high']:.0f} ({best['strength']}% str)")

    # ── Find barriers (OI walls between price and target) ──
    barriers = []
    if oi_walls:
        for wall in oi_walls[:5]:
            strike = wall["strike"]
            if direction == "BULLISH" and price < strike <= max_pain * 1.05:
                wall_type = "Call Wall" if wall["call_oi"] > wall["put_oi"] else "Put Wall"
                barriers.append({"strike": strike, "type": wall_type,
                                 "oi": wall["total_oi"],
                                 "strength": "Heavy" if wall["total_oi"] >= max(w["total_oi"] for w in oi_walls) * 0.7 else "Moderate"})
            elif direction == "BEARISH" and max_pain * 0.95 <= strike < price:
                wall_type = "Put Wall" if wall["put_oi"] > wall["call_oi"] else "Call Wall"
                barriers.append({"strike": strike, "type": wall_type,
                                 "oi": wall["total_oi"],
                                 "strength": "Heavy" if wall["total_oi"] >= max(w["total_oi"] for w in oi_walls) * 0.7 else "Moderate"})

    barriers.sort(key=lambda x: x["strike"], reverse=(direction == "BEARISH"))

    # ── Target ──
    target = {"price": max_pain, "type": "Max Pain"}

    # Check if GEX flip is a better intermediate target
    if gex_flip:
        if direction == "BULLISH" and price < gex_flip < max_pain:
            signals.append(f"GEX flip at ${gex_flip:.0f} (intermediate magnet)")
        elif direction == "BEARISH" and max_pain < gex_flip < price:
            signals.append(f"GEX flip at ${gex_flip:.0f} (intermediate magnet)")

    # ── Additional signals ──
    if max_pain:
        signals.append(f"Max Pain ${max_pain:.0f} ({mp_diff_pct:+.1f}% from price)")
    if dealer_pos == "LONG_GAMMA":
        signals.append("Dealers long gamma → price stabilizes near max pain")
        confluence += 5
    elif dealer_pos == "SHORT_GAMMA":
        signals.append("Dealers short gamma → expect bigger moves")
        confluence += 5
    if sentiment in ("BULLISH", "BEARISH"):
        signals.append(f"Flow sentiment: {sentiment}")

    # ── Confluence: check if OB + Max Pain + OI wall align ──
    if entry_zone and max_pain:
        # Check if any OI wall is within 2% of an order block
        for wall in oi_walls[:5]:
            if entry_zone and abs(wall["strike"] - entry_zone.get("low", 0)) / price < 0.02:
                confluence += 10
                signals.append(f"OI wall at ${wall['strike']:.0f} confluent with OB zone")
                break

    # Max pain near OI wall
    for wall in oi_walls[:3]:
        if max_pain and abs(wall["strike"] - max_pain) / price < 0.01:
            confluence += 10
            signals.append(f"Max Pain + OI wall confluence at ${max_pain:.0f}")
            break

    confluence = min(100, confluence)

    # ── Build pathway text ──
    parts = []
    if entry_zone:
        parts.append(f"📍 ${entry_zone['low']:.0f}-${entry_zone['high']:.0f} {entry_zone['type']}")
    else:
        parts.append(f"📍 ${price:.0f} (current)")

    for b in barriers[:2]:
        parts.append(f"→ ${b['strike']:.0f} {b['type']}")

    parts.append(f"→ 🎯 ${max_pain:.0f} Max Pain")

    pathway_text = " ".join(parts)

    return {
        "direction": direction,
        "entry_zone": entry_zone,
        "barriers": barriers[:3],
        "target": target,
        "confluence_score": confluence,
        "pathway_text": pathway_text,
        "signals": signals[:5],
    }


# ── Async Wrapper ──

async def async_scan_tickers(symbols: List[str], **kwargs) -> Dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: scan_tickers(symbols, **kwargs))
