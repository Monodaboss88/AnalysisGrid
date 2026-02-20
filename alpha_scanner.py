"""
Alpha Scanner — Automated 7-Step Bullish Setup Finder
======================================================
Orchestrates every tool on the desk into a single pipeline:

  Step 1: Market Context (SPY/QQQ breadth check)
  Step 2: Universe Scan (volume profile + directional filter)
  Step 3: Squeeze + Compression Filter
  Step 4: Historical Odds (probability context + predictability map)
  Step 5: War Room Extension DNA (intraday VWAP/exhaustion)
  Step 6: Structure Confirmation (range + reversal detection)
  Step 7: Rank & Output (composite score, sorted)

Each step is a FILTER — candidates get eliminated, not added.
The output is 0-5 highest-conviction bullish setups with full context.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=8)

# ── Universe Presets ──
UNIVERSES = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "CRM", "NFLX",
             "AMD", "QCOM", "ORCL", "ADBE", "INTC", "MU", "NOW", "PANW"],
    "semis": ["NVDA", "AMD", "AVGO", "QCOM", "INTC", "MU", "TSM", "MRVL", "LRCX", "KLAC",
              "AMAT", "ASML", "ON", "NXPI", "TXN"],
    "momentum": ["PLTR", "SMCI", "MSTR", "COIN", "RKLB", "APP", "HOOD", "AFRM", "IONQ", "RDDT",
                  "SOFI", "RIVN", "LCID", "ARM", "CRWD"],
    "etfs": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "SMH", "ARKK", "TQQQ"],
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "all": [],  # filled at runtime by merging others
}
UNIVERSES["all"] = list(set(
    UNIVERSES["tech"] + UNIVERSES["semis"] + UNIVERSES["momentum"] + UNIVERSES["etfs"]
))


# ═══════════════════════════════════════════════════════
#  STEP 1: Market Context
# ═══════════════════════════════════════════════════════

def _check_market_context() -> Dict:
    """Quick SPY/QQQ check — are we in a bullish environment?"""
    try:
        from polygon_data import get_bars
        context = {"bullish": False, "details": {}}

        for sym in ["SPY", "QQQ", "IWM"]:
            try:
                df = get_bars(sym, period="5d", interval="1d")
                if df.empty or len(df) < 2:
                    continue
                today_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                change_pct = (today_close - prev_close) / prev_close * 100
                sma5 = float(df["Close"].tail(5).mean())
                above_sma = today_close > sma5
                context["details"][sym] = {
                    "price": round(today_close, 2),
                    "change_pct": round(change_pct, 2),
                    "above_5d_sma": above_sma,
                    "green": change_pct > 0,
                }
            except Exception:
                continue

        # Bullish if at least 2/3 indices are green or above 5d SMA
        greens = sum(1 for v in context["details"].values() if v.get("green"))
        above = sum(1 for v in context["details"].values() if v.get("above_5d_sma"))
        context["bullish"] = greens >= 2 or above >= 2
        context["green_count"] = greens
        context["above_sma_count"] = above
        context["verdict"] = "BULLISH" if context["bullish"] else "CAUTION"
        return context
    except Exception as e:
        logger.warning(f"Market context failed: {e}")
        return {"bullish": True, "verdict": "UNKNOWN", "details": {}}


# ═══════════════════════════════════════════════════════
#  STEP 2: Universe Scan (Volume Profile + Directional)
# ═══════════════════════════════════════════════════════

def _scan_universe(symbols: List[str]) -> List[Dict]:
    """Quick scan each symbol for bullish structure via Polygon. Returns candidates with scores."""
    from polygon_data import get_bars
    candidates = []

    for sym in symbols:
        try:
            df = get_bars(sym, period="3mo", interval="1d")
            if df.empty or len(df) < 20:
                continue

            close = df["Close"].values
            volume = df["Volume"].values
            high = df["High"].values
            low = df["Low"].values
            current = float(close[-1])

            # SMA 20 / SMA 50 trend check
            sma20 = float(close[-20:].mean())
            sma50 = float(close[-50:].mean()) if len(close) >= 50 else sma20

            # RSI (14)
            deltas = [close[i] - close[i-1] for i in range(1, len(close))]
            gains = [d if d > 0 else 0 for d in deltas[-14:]]
            losses = [-d if d < 0 else 0 for d in deltas[-14:]]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))

            # Relative volume
            avg_vol_20 = float(volume[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
            rvol = float(volume[-1]) / avg_vol_20 if avg_vol_20 > 0 else 1.0

            # Change 1d and 5d
            change_1d = (current - close[-2]) / close[-2] * 100 if len(close) >= 2 else 0
            change_5d = (current - close[-6]) / close[-6] * 100 if len(close) >= 6 else 0

            # Score: higher = more bullish setup
            score = 50  # base
            # Trend: above both MAs
            if current > sma20:
                score += 10
            if current > sma50:
                score += 10
            if sma20 > sma50:
                score += 5  # golden cross zone

            # RSI: moderate is good (40-65 bullish zone, not overbought)
            if 40 <= rsi <= 65:
                score += 10
            elif rsi > 65:
                score += 3  # slightly overbought but still up
            elif rsi < 35:
                score -= 10  # too weak

            # Volume confirmation
            if rvol > 1.2:
                score += 5

            # Recent momentum
            if change_1d > 0:
                score += 3
            if change_5d > 0:
                score += 5

            # Filter: only above 50 (some bullish structure)
            if score < 50:
                continue

            direction = "BULLISH" if current > sma20 and current > sma50 else "NEUTRAL" if current > sma50 else "BEARISH"
            if direction == "BEARISH":
                continue

            candidates.append({
                "symbol": sym,
                "scan_score": min(100, score),
                "direction": direction,
                "price": round(current, 2),
                "rsi": round(rsi, 1),
                "rvol": round(rvol, 2),
                "change_1d": round(change_1d, 2),
                "change_5d": round(change_5d, 2),
            })
        except Exception as e:
            logger.debug(f"Scan failed for {sym}: {e}")
            continue

    # Sort by score descending
    candidates.sort(key=lambda x: x["scan_score"], reverse=True)
    return candidates[:20]  # Top 20 pass to next stage


# ═══════════════════════════════════════════════════════
#  STEP 3: Squeeze + Compression Filter
# ═══════════════════════════════════════════════════════

def _check_squeeze(symbol: str) -> Dict:
    """Check if symbol has active squeeze or compression setup."""
    result = {"has_squeeze": False, "squeeze_score": 0, "squeeze_status": "NONE"}
    try:
        from squeeze_detector_v2 import SqueezeDetectorV2
        detector = SqueezeDetectorV2()
        metrics = detector.analyze(symbol)
        if metrics:
            result["squeeze_score"] = getattr(metrics, "score", 0) or 0
            result["has_squeeze"] = getattr(metrics, "ttm_squeeze", False)
            result["squeeze_duration"] = getattr(metrics, "squeeze_duration", 0) or 0

            release = getattr(metrics, "release", None)
            if release and getattr(release, "is_firing", False):
                result["squeeze_status"] = "FIRING"
                result["squeeze_score"] = max(result["squeeze_score"], 75)
            elif result["has_squeeze"]:
                result["squeeze_status"] = "ACTIVE"
            elif result["squeeze_score"] >= 50:
                result["squeeze_status"] = "FORMING"
    except Exception as e:
        logger.debug(f"Squeeze check failed for {symbol}: {e}")
    return result


# ═══════════════════════════════════════════════════════
#  STEP 4: Historical Odds (Probability + Predictability)
# ═══════════════════════════════════════════════════════

def _check_odds(symbol: str) -> Dict:
    """Get historical probability context for the symbol."""
    result = {
        "call_hit_3d": 0, "call_win_1d": 0, "straddle_rate": 0,
        "regime": "", "zscore": 0, "clv": 0, "gap_fill_rate": 0,
        "vwap_revert_rate": 0,
    }
    try:
        # Add polygon_signal_tool to path
        tool_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polygon_signal_tool")
        if tool_dir not in sys.path:
            sys.path.insert(0, tool_dir)

        from signal_endpoints import _run_analysis, _get_vwap_magnet
        analysis = _run_analysis(symbol, 365)
        if not analysis:
            return result

        sig = analysis["signal"]
        stats = analysis["all_stats"]
        today = sig["today"]

        # Pick the right scenario
        if today["color"] == "RED":
            rs = today["rstreak"]
            call_key = f"call_red{rs}" if rs >= 2 and f"call_red{rs}" in stats else "call_red"
        else:
            gs = today["gstreak"]
            call_key = f"call_green{gs}" if gs >= 2 and f"call_green{gs}" in stats else "call_green"

        cs = stats.get(call_key, stats.get("call_all", {}))
        if cs:
            result["call_hit_3d"] = round(cs.get("rate_3d", 0) * 100, 1)
            result["call_win_1d"] = round(cs.get("close_win_1d", 0) * 100, 1)
            result["call_hit_1d"] = round(cs.get("rate_1d", 0) * 100, 1)
            result["avg_best_3d"] = round(cs.get("avg_best_pct_3d", 0), 2)
            result["call_scenario"] = call_key
            result["sample_size"] = cs.get("count", 0)

        straddle = analysis.get("straddle", {})
        result["straddle_rate"] = round(straddle.get("at_least_one_rate", 0) * 100, 1)

        # Expected moves
        result["expected_up_1d"] = round(sig.get("expected_upside", 0), 2)
        result["expected_dn_1d"] = round(sig.get("expected_downside", 0), 2)

        # Predictability map
        cl = analysis.get("close_location", {})
        ga = analysis.get("gap_analysis", {})
        vr = analysis.get("vol_regime", {})
        ext = analysis.get("extension", {})

        result["clv"] = cl.get("today_clv", 0)
        result["trend_cluster"] = cl.get("trend_cluster", "")
        result["regime"] = vr.get("regime", "")
        result["atr_ratio"] = vr.get("atr_ratio", 0)
        result["zscore"] = ext.get("zscore", 0)
        result["extension_pct"] = ext.get("extension_pct", 0)
        result["gap_up_fill_rate"] = ga.get("gap_ups", {}).get("fill_rate", 0)
        result["gap_dn_fill_rate"] = ga.get("gap_downs", {}).get("fill_rate", 0)

        # VWAP magnet
        vm = _get_vwap_magnet(symbol)
        result["vwap_revert_rate"] = vm.get("vwap_revert_rate", 0)
        result["vwap_crosses"] = vm.get("avg_vwap_crosses", 0)

    except Exception as e:
        logger.debug(f"Odds check failed for {symbol}: {e}")
    return result


# ═══════════════════════════════════════════════════════
#  STEP 5: War Room Extension DNA
# ═══════════════════════════════════════════════════════

def _check_war_room(symbol: str) -> Dict:
    """Get extension DNA and fade signals from War Room."""
    result = {"avg_up_ext": 0, "exhaustion": 0, "fade_conviction": 0, "thin_top_pct": 0}
    try:
        from war_room import get_master_analysis, _compute_signals
        dna = get_master_analysis(symbol, lookback_days=45)
        if not dna:
            return result

        result["avg_up_ext"] = dna.get("avg_up", 0)
        result["avg_dn_ext"] = dna.get("avg_down", 0)
        result["std_up"] = dna.get("std_up", 0)
        result["peak_hour"] = dna.get("peak_hour", 0)
        result["avg_close_pos"] = dna.get("avg_close_pos", 0)
        result["thin_top_pct"] = dna.get("thin_top_pct", 0)
        result["reversal_pct"] = dna.get("reversal_pct", 0)
        result["vwap_revert_rate"] = dna.get("vwap_revert_rate", 0)

        # Compute signals (need spy_dna — skip correlation for speed)
        sig = _compute_signals(dna, None, [])
        result["exhaustion"] = sig.get("exhaustion", 0)
        result["fade_conviction"] = sig.get("fade_conviction", 0)
        result["war_signals"] = sig.get("signals", [])
    except Exception as e:
        logger.debug(f"War Room failed for {symbol}: {e}")
    return result


# ═══════════════════════════════════════════════════════
#  STEP 6: Structure Confirmation
# ═══════════════════════════════════════════════════════

def _check_structure(symbol: str) -> Dict:
    """Check range structure for bullish confirmation."""
    result = {"bullish_structure": False, "structure_score": 0, "pattern": ""}
    try:
        from polygon_data import get_bars
        df = get_bars(symbol, period="6mo", interval="1d")
        if df.empty or len(df) < 30:
            return result
        df.columns = [c.lower() for c in df.columns]

        # Simple HH/HL detection on last 20 bars
        closes = df["close"].tail(20).values
        highs = df["high"].tail(20).values
        lows = df["low"].tail(20).values

        # Find swing highs/lows (3-bar pivots)
        swing_highs = []
        swing_lows = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])

        # Check for higher highs + higher lows
        hh = len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]
        hl = len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]

        if hh and hl:
            result["pattern"] = "UPTREND (HH+HL)"
            result["bullish_structure"] = True
            result["structure_score"] = 80
        elif hl:
            result["pattern"] = "HIGHER LOWS"
            result["bullish_structure"] = True
            result["structure_score"] = 60
        elif hh:
            result["pattern"] = "HIGHER HIGHS"
            result["bullish_structure"] = True
            result["structure_score"] = 50
        else:
            # Check if price is above key MAs
            sma20 = closes[-20:].mean()
            sma50 = df["close"].tail(50).mean() if len(df) >= 50 else sma20
            current = closes[-1]
            if current > sma20 and current > sma50:
                result["pattern"] = "ABOVE KEY MAs"
                result["bullish_structure"] = True
                result["structure_score"] = 40

        # Price relative to 52-week range
        high_52w = df["high"].tail(252).max() if len(df) >= 252 else df["high"].max()
        low_52w = df["low"].tail(252).min() if len(df) >= 252 else df["low"].min()
        r52 = high_52w - low_52w
        if r52 > 0:
            result["range_position_52w"] = round((closes[-1] - low_52w) / r52 * 100, 1)

    except Exception as e:
        logger.debug(f"Structure check failed for {symbol}: {e}")
    return result


# ═══════════════════════════════════════════════════════
#  STEP 7: Composite Scoring + Ranking
# ═══════════════════════════════════════════════════════

def _compute_alpha_score(candidate: Dict) -> float:
    """
    Composite Alpha Score (0-100) for a bullish setup.
    Higher = more conviction.
    """
    score = 0

    # Scanner score (max 20 pts)
    scan = candidate.get("scan_score", 0)
    score += min(20, scan * 0.25)

    # Squeeze bonus (max 20 pts)
    sq = candidate.get("squeeze", {})
    if sq.get("squeeze_status") == "FIRING":
        score += 20
    elif sq.get("squeeze_status") == "ACTIVE":
        score += 14
    elif sq.get("squeeze_status") == "FORMING":
        score += 8
    score += min(6, sq.get("squeeze_score", 0) * 0.08)

    # Historical odds (max 25 pts)
    odds = candidate.get("odds", {})
    call_hit = odds.get("call_hit_3d", 0)
    call_win = odds.get("call_win_1d", 0)
    if call_hit >= 75:
        score += 12
    elif call_hit >= 60:
        score += 8
    elif call_hit >= 50:
        score += 4
    if call_win >= 55:
        score += 6
    elif call_win >= 50:
        score += 3

    # Regime + Z-Score (max 7 pts)
    regime = odds.get("regime", "")
    if regime == "STABLE":
        score += 5
    elif regime == "EXPANDING":
        score += 3
    elif regime == "EXTREME":
        score -= 2  # penalty

    zscore = abs(odds.get("zscore", 0))
    if zscore < 1.0:
        score += 2  # room to run
    elif zscore >= 2.0:
        score -= 3  # already extended

    # Structure (max 15 pts)
    struct = candidate.get("structure", {})
    score += min(15, struct.get("structure_score", 0) * 0.19)

    # War Room: low fade conviction is GOOD for bulls (max 10 pts)
    wr = candidate.get("war_room", {})
    fade = wr.get("fade_conviction", 50)
    if fade <= 20:
        score += 10  # no fade setup = bullish
    elif fade <= 40:
        score += 6
    elif fade >= 60:
        score -= 5  # high fade = bearish risk

    # Bonus: thin_top < 40% is bullish (highs have volume support)
    thin = wr.get("thin_top_pct", 50)
    if thin < 40:
        score += 3
    elif thin > 70:
        score -= 2

    return round(min(100, max(0, score)), 1)


def _assign_duration_tier(candidate: Dict) -> Dict:
    """
    Auto-assign a trade duration tier based on setup characteristics.
    
    Returns dict with:
      - duration_tier: DAY, SWING, POSITION, MACRO
      - setup_type: key for SETUP_TIER_MAP
      - duration_label: readable label
    """
    sq = candidate.get("squeeze", {})
    struct = candidate.get("structure", {})
    wr = candidate.get("war_room", {})
    odds = candidate.get("odds", {})
    
    # Priority 1: Squeeze setups
    sq_status = sq.get("squeeze_status", "NONE")
    if sq_status == "FIRING":
        return {"duration_tier": "SWING", "setup_type": "squeeze_firing", "duration_label": "3-5 Day Swing"}
    if sq_status == "ACTIVE":
        return {"duration_tier": "SWING", "setup_type": "squeeze_active", "duration_label": "3-5 Day Swing"}
    if sq_status == "FORMING":
        return {"duration_tier": "POSITION", "setup_type": "squeeze_forming", "duration_label": "2-Week Hold"}
    
    # Priority 2: Strong structure trend (HH+HL = position trade)
    pattern = struct.get("pattern", "")
    if "HH+HL" in pattern or "UPTREND" in pattern:
        return {"duration_tier": "POSITION", "setup_type": "hh_hl_trend", "duration_label": "2-Week Hold"}
    if "HIGHER LOWS" in pattern:
        return {"duration_tier": "SWING", "setup_type": "higher_lows", "duration_label": "3-5 Day Swing"}
    
    # Priority 3: War room fade/exhaustion signals (day trade)
    fade = wr.get("fade_conviction", 0)
    exhaustion = wr.get("exhaustion", 0)
    if fade >= 60 or exhaustion >= 70:
        return {"duration_tier": "DAY", "setup_type": "extension_fade", "duration_label": "Day Trade"}
    
    # Priority 4: High historical call hit rate + stable regime → position
    call_hit = odds.get("call_hit_3d", 0)
    regime = odds.get("regime", "")
    if call_hit >= 75 and regime == "STABLE":
        return {"duration_tier": "POSITION", "setup_type": "high_prob_stable", "duration_label": "2-Week Hold"}
    
    # Priority 5: Range position (52w) — near lows = macro, mid-range = swing
    range_pos = struct.get("range_position_52w", 50)
    if range_pos < 25:
        return {"duration_tier": "MACRO", "setup_type": "weekly_break", "duration_label": "1-Month Position"}
    
    # Default: SWING
    return {"duration_tier": "SWING", "setup_type": "default", "duration_label": "3-5 Day Swing"}


def _build_verdict(c: Dict) -> str:
    """Generate a human-readable verdict for a candidate."""
    sym = c["symbol"]
    alpha = c.get("alpha_score", 0)
    odds = c.get("odds", {})
    sq = c.get("squeeze", {})
    struct = c.get("structure", {})
    wr = c.get("war_room", {})

    parts = [f"**{sym}** — Alpha Score: **{alpha}/100**"]

    # Duration tier
    tier = c.get("duration_tier", "SWING")
    tier_label = c.get("duration_label", "3-5 Day Swing")
    parts.append(f"⏱️ Trade Type: {tier_label}")

    # Key bullish drivers
    drivers = []
    if sq.get("squeeze_status") in ("FIRING", "ACTIVE"):
        drivers.append(f"Squeeze {sq['squeeze_status']}")
    if odds.get("call_hit_3d", 0) >= 70:
        drivers.append(f"Call hit 3D: {odds['call_hit_3d']}%")
    if struct.get("bullish_structure"):
        drivers.append(struct.get("pattern", "Bullish"))
    if odds.get("regime") == "STABLE":
        drivers.append("Stable regime")
    if wr.get("fade_conviction", 99) <= 20:
        drivers.append("No fade setup")

    if drivers:
        parts.append("Drivers: " + ", ".join(drivers))

    # Risks
    risks = []
    zscore = abs(odds.get("zscore", 0))
    if zscore >= 1.5:
        risks.append(f"Z-Score {odds['zscore']:.1f} (stretched)")
    if wr.get("fade_conviction", 0) >= 40:
        risks.append(f"Fade conviction {wr['fade_conviction']}%")
    if odds.get("regime") == "EXTREME":
        risks.append("Extreme vol regime")
    if wr.get("thin_top_pct", 0) > 70:
        risks.append(f"Thin tops {wr['thin_top_pct']}%")

    if risks:
        parts.append("Risks: " + ", ".join(risks))

    # Expected range
    up1 = odds.get("expected_up_1d", 0)
    dn1 = odds.get("expected_dn_1d", 0)
    if up1 or dn1:
        parts.append(f"Expected 1D: +{up1}% / -{dn1}%")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════

def run_alpha_scan(universe: str = "all", max_results: int = 5) -> Dict:
    """
    Run the full 7-step Alpha Scanner pipeline.
    Returns ranked bullish setups with full context at every stage.
    """
    start_time = datetime.now(timezone.utc)
    pipeline = {"steps": {}, "results": [], "meta": {}}

    # ── Step 1: Market Context ──
    logger.info("Alpha Scanner: Step 1 — Market Context")
    market = _check_market_context()
    pipeline["steps"]["market_context"] = market

    # ── Step 2: Universe Scan ──
    logger.info("Alpha Scanner: Step 2 — Universe Scan")
    symbols = UNIVERSES.get(universe, UNIVERSES["all"])
    if not symbols:
        symbols = UNIVERSES["all"]
    candidates = _scan_universe(symbols)
    pipeline["steps"]["universe_scan"] = {
        "scanned": len(symbols),
        "passed": len(candidates),
        "symbols": [c["symbol"] for c in candidates],
    }

    if not candidates:
        pipeline["meta"]["verdict"] = "NO CANDIDATES — nothing passed the initial scan"
        pipeline["meta"]["duration_sec"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        return pipeline

    # ── Step 3: Squeeze Filter ──
    logger.info("Alpha Scanner: Step 3 — Squeeze Filter")
    for c in candidates:
        c["squeeze"] = _check_squeeze(c["symbol"])

    # Boost candidates with squeeze; don't eliminate others yet
    candidates.sort(key=lambda x: x["squeeze"]["squeeze_score"], reverse=True)
    pipeline["steps"]["squeeze_filter"] = {
        "firing": sum(1 for c in candidates if c["squeeze"]["squeeze_status"] == "FIRING"),
        "active": sum(1 for c in candidates if c["squeeze"]["squeeze_status"] == "ACTIVE"),
        "forming": sum(1 for c in candidates if c["squeeze"]["squeeze_status"] == "FORMING"),
    }

    # Trim to top 10 for deeper analysis
    top = candidates[:10]

    # ── Step 4: Historical Odds ──
    logger.info("Alpha Scanner: Step 4 — Historical Odds")
    for c in top:
        c["odds"] = _check_odds(c["symbol"])

    # Filter: call hit 3D must be >= 50% (history must support)
    top = [c for c in top if c["odds"].get("call_hit_3d", 0) >= 50]
    pipeline["steps"]["odds_filter"] = {
        "passed": len(top),
        "filtered_out": 10 - len(top),
    }

    if not top:
        pipeline["meta"]["verdict"] = "NO CANDIDATES — no symbols had ≥50% call hit rate"
        pipeline["meta"]["duration_sec"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        return pipeline

    # ── Step 5: War Room ──
    logger.info("Alpha Scanner: Step 5 — War Room Extension DNA")
    for c in top:
        c["war_room"] = _check_war_room(c["symbol"])

    pipeline["steps"]["war_room"] = {
        "analyzed": len(top),
        "low_fade": sum(1 for c in top if c["war_room"].get("fade_conviction", 99) <= 20),
    }

    # ── Step 6: Structure ──
    logger.info("Alpha Scanner: Step 6 — Structure Confirmation")
    for c in top:
        c["structure"] = _check_structure(c["symbol"])

    pipeline["steps"]["structure"] = {
        "bullish_structure": sum(1 for c in top if c["structure"].get("bullish_structure")),
    }

    # ── Step 7: Score & Rank ──
    logger.info("Alpha Scanner: Step 7 — Composite Scoring + Duration Tier")
    for c in top:
        c["alpha_score"] = _compute_alpha_score(c)
        # Auto-assign trade duration tier
        tier_info = _assign_duration_tier(c)
        c["duration_tier"] = tier_info["duration_tier"]
        c["setup_type"] = tier_info["setup_type"]
        c["duration_label"] = tier_info["duration_label"]
        c["verdict"] = _build_verdict(c)

    # Sort by alpha score
    top.sort(key=lambda x: x["alpha_score"], reverse=True)

    # Take top N
    results = top[:max_results]

    pipeline["results"] = results
    pipeline["meta"] = {
        "universe": universe,
        "total_scanned": len(symbols),
        "survivors": len(results),
        "market_context": market["verdict"],
        "top_pick": results[0]["symbol"] if results else "NONE",
        "top_score": results[0]["alpha_score"] if results else 0,
        "duration_sec": round((datetime.now(timezone.utc) - start_time).total_seconds(), 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return pipeline


# ── Async wrapper ──

async def async_run_alpha_scan(universe: str = "all", max_results: int = 5) -> Dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pool, lambda: run_alpha_scan(universe, max_results))


# ── CLI ──

if __name__ == "__main__":
    import json
    result = run_alpha_scan("mag7", max_results=3)
    print(json.dumps(result, indent=2, default=str))
