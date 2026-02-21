"""
Alpha Scanner V2 — Bi-Directional Setup Discovery Engine
==========================================================
Finds the highest-conviction long AND short setups across the full universe
using the V2 scanner infrastructure.

Architecture:
  Phase 1: Market Regime  (SPY/QQQ/IWM breadth → multiplier, not a gate)
  Phase 2: Broad Scan     (FinnhubScanner.analyze_enriched per symbol, parallel)
  Phase 3: Deep Enrich    (squeeze, odds, war room, extension — top 15, parallel)
  Phase 4: Classify+Score (setup type → type-specific weighted scoring)
  Phase 5: Output         (ranked results with full context)

Key Differences from V1:
  - Bi-directional (long AND short)
  - Parallel execution via ThreadPoolExecutor (~45s vs ~5min)
  - Reuses canonical V2 infrastructure (no homebrew RSI/VP)
  - Score-first, filter-last (soft scoring, no hard gates)
  - Regime-aware weighting (multiplier, not binary)
  - Setup type classification drives scoring weights
  - Config-aware (accepts SwingTradeConfig)

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=8)

# ── V2 Infrastructure Imports ──
from universe import ALPHA_UNIVERSES as UNIVERSES

try:
    from finnhub_scanner_v2 import FinnhubScanner
    _finnhub_available = True
except ImportError:
    _finnhub_available = False
    logger.warning("finnhub_scanner_v2 not available — alpha scan will fail")

try:
    from scanner_config import SwingTradeConfig, BALANCED
    _config_available = True
except ImportError:
    _config_available = False

from polygon_data import get_bars, get_price_quote


# ═══════════════════════════════════════════════════════
#  FALLBACK SCAN (V1-Style — used when FinnhubScanner unavailable)
# ═══════════════════════════════════════════════════════

def _fallback_scan(symbols: List[str]) -> List[Dict]:
    """
    Lightweight homebrew scan (ported from V1) used when FinnhubScanner
    is unavailable. Computes RSI, SMA, RVOL from raw Polygon bars.
    Returns candidates shaped like _broad_scan output so the rest of
    the pipeline works unchanged.
    """
    candidates = []
    for sym in symbols:
        try:
            df = get_bars(sym, period="3mo", interval="1d")
            if df.empty or len(df) < 20:
                continue

            close = df["Close"].values
            volume = df["Volume"].values
            current = float(close[-1])

            # SMA 20 / SMA 50
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

            # Changes
            change_1d = (current - close[-2]) / close[-2] * 100 if len(close) >= 2 else 0
            change_5d = (current - close[-6]) / close[-6] * 100 if len(close) >= 6 else 0

            # Directional scoring
            bull_score = 50
            if current > sma20: bull_score += 10
            if current > sma50: bull_score += 10
            if sma20 > sma50: bull_score += 5
            if 40 <= rsi <= 65: bull_score += 10
            elif rsi > 65: bull_score += 3
            elif rsi < 35: bull_score -= 10
            if rvol > 1.2: bull_score += 5
            if change_1d > 0: bull_score += 3
            if change_5d > 0: bull_score += 5
            bull_score = max(0, min(100, bull_score))

            # Bear score (inverse logic)
            bear_score = 50
            if current < sma20: bear_score += 10
            if current < sma50: bear_score += 10
            if sma20 < sma50: bear_score += 5
            if rsi < 35: bear_score += 10
            elif rsi > 65: bear_score += 3
            if rvol > 1.2: bear_score += 5
            if change_1d < 0: bear_score += 3
            if change_5d < 0: bear_score += 5
            bear_score = max(0, min(100, bear_score))

            raw_score = max(bull_score, bear_score)

            # Shape output to match _broad_scan format
            candidates.append({
                "symbol": sym,
                "price": round(current, 2),
                "rsi": round(rsi, 1),
                "atr": 0,
                "rvol": round(rvol, 2),
                "squeeze_active": False,
                "squeeze_days": 0,
                "squeeze_momentum": "unknown",
                "weekly_trend": "NEUTRAL",
                "weekly_supports_long": current > sma50,
                "weekly_supports_short": current < sma50,
                "vp_shape": "unknown",
                "vp_position": "unknown",
                "iv_regime": "unknown",
                "bull_score": bull_score,
                "bear_score": bear_score,
                "signal": "LONG" if bull_score > bear_score else "SHORT",
                "confidence": raw_score,
                "raw_long_score": bull_score,
                "raw_short_score": bear_score,
                "raw_score": raw_score,
                "v2ctx": None,
            })
        except Exception as e:
            logger.debug(f"Fallback scan failed for {sym}: {e}")
            continue

    candidates.sort(key=lambda x: x["raw_score"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════
#  SETUP TYPE WEIGHTS
# ═══════════════════════════════════════════════════════

# Each setup type has different scoring weights across 6 dimensions.
# Weights per dimension must sum to 1.0 per setup type.
# Max points per dimension control the ceiling.

DIMENSION_MAX_PTS = {
    "v2_signal": 25,
    "squeeze": 25,
    "odds": 20,
    "structure": 15,
    "war_room": 10,
    "extension": 5,
}

SETUP_WEIGHTS = {
    "squeeze_break": {
        "v2_signal": 0.20, "squeeze": 0.35, "odds": 0.15,
        "structure": 0.10, "war_room": 0.10, "extension": 0.10,
    },
    "trend_continuation": {
        "v2_signal": 0.30, "squeeze": 0.10, "odds": 0.20,
        "structure": 0.25, "war_room": 0.05, "extension": 0.10,
    },
    "mean_reversion": {
        "v2_signal": 0.15, "squeeze": 0.10, "odds": 0.20,
        "structure": 0.15, "war_room": 0.20, "extension": 0.20,
    },
    "capitulation_bounce": {
        "v2_signal": 0.10, "squeeze": 0.10, "odds": 0.15,
        "structure": 0.20, "war_room": 0.15, "extension": 0.30,
    },
    "extension_fade": {
        "v2_signal": 0.15, "squeeze": 0.10, "odds": 0.20,
        "structure": 0.10, "war_room": 0.35, "extension": 0.10,
    },
}


# ═══════════════════════════════════════════════════════
#  PHASE 1: Market Regime
# ═══════════════════════════════════════════════════════

def _compute_market_regime() -> Dict:
    """
    Score the market environment and produce directional multipliers.

    Returns:
        {
            "regime_score": 0-100,
            "regime_label": str,
            "long_multiplier": 0.6-1.4,
            "short_multiplier": 0.6-1.4,
            "details": {...},
        }
    """
    details = {}
    score = 50  # neutral baseline

    # ── Index health (SPY, QQQ, IWM) ──
    for sym in ["SPY", "QQQ", "IWM"]:
        try:
            df = get_bars(sym, period="3mo", interval="1d")
            if df.empty or len(df) < 20:
                continue
            close = df["Close"].values
            current = float(close[-1])
            prev = float(close[-2]) if len(close) >= 2 else current

            sma20 = float(close[-20:].mean())
            sma50 = float(close[-50:].mean()) if len(close) >= 50 else sma20

            change_1d = (current - prev) / prev * 100
            change_5d = (current - float(close[-6])) / float(close[-6]) * 100 if len(close) >= 6 else 0

            above_sma20 = current > sma20
            above_sma50 = current > sma50
            golden = sma20 > sma50

            details[sym] = {
                "price": round(current, 2),
                "change_1d": round(change_1d, 2),
                "change_5d": round(change_5d, 2),
                "above_sma20": above_sma20,
                "above_sma50": above_sma50,
                "golden_cross": golden,
            }

            # Scoring
            if above_sma20:
                score += 4
            if above_sma50:
                score += 4
            if golden:
                score += 2
            if change_1d > 0:
                score += 2
            elif change_1d < -1:
                score -= 3
            if change_5d > 1:
                score += 3
            elif change_5d < -2:
                score -= 4

        except Exception as e:
            logger.debug(f"Regime check failed for {sym}: {e}")

    # ── VIX awareness ──
    try:
        # Try VIXY ETF as proxy (VIX index may not be available on all Polygon tiers)
        vix_quote = get_price_quote("VIXY")
        if vix_quote and vix_quote.get("price"):
            vix_price = vix_quote["price"]
            details["VIXY"] = {"price": round(vix_price, 2)}
            # VIXY tracks VIX — higher = more fear
            vix_change = vix_quote.get("change_pct", 0)
            if vix_change > 5:
                score -= 8  # VIX spiking = bearish
            elif vix_change > 2:
                score -= 4
            elif vix_change < -3:
                score += 4  # VIX falling = bullish
    except Exception:
        pass

    # Clamp
    score = max(0, min(100, score))

    # ── Regime classification ──
    if score >= 75:
        label = "STRONG_BULL"
        long_mult = 1.3
        short_mult = 0.6
    elif score >= 60:
        label = "BULL"
        long_mult = 1.15
        short_mult = 0.8
    elif score >= 40:
        label = "NEUTRAL"
        long_mult = 1.0
        short_mult = 1.0
    elif score >= 25:
        label = "BEAR"
        long_mult = 0.8
        short_mult = 1.15
    else:
        label = "STRONG_BEAR"
        long_mult = 0.6
        short_mult = 1.3

    return {
        "regime_score": score,
        "regime_label": label,
        "long_multiplier": long_mult,
        "short_multiplier": short_mult,
        "details": details,
    }


# ═══════════════════════════════════════════════════════
#  PHASE 2: Broad Scan (Parallel)
# ═══════════════════════════════════════════════════════

def _scan_one_symbol(scanner, symbol: str) -> Optional[Dict]:
    """Scan a single symbol with FinnhubScanner. Returns feature dict or None."""
    try:
        result, ctx = scanner.analyze_enriched(symbol, "2HR", days_back=60,
                                                include_order_flow=False)
        if result is None:
            return None

        bull = getattr(result, 'bull_score', 0) or 0
        bear = getattr(result, 'bear_score', 0) or 0
        signal = getattr(result, 'signal', 'NEUTRAL') or 'NEUTRAL'
        conf = getattr(result, 'confidence', 0) or 0

        # Raw directional score for ranking into Phase 3
        squeeze_bonus = 10 if ctx.squeeze.is_squeezed else 0
        weekly_long_bonus = 10 if ctx.weekly.supports_long else 0
        weekly_short_bonus = 10 if ctx.weekly.supports_short else 0

        raw_long = bull + squeeze_bonus + weekly_long_bonus
        raw_short = bear + squeeze_bonus + weekly_short_bonus
        raw_score = max(raw_long, raw_short)

        return {
            "symbol": symbol,
            "price": ctx.current_price,
            "rsi": ctx.rsi,
            "atr": ctx.atr,
            "rvol": ctx.rvol,
            "squeeze_active": ctx.squeeze.is_squeezed,
            "squeeze_days": ctx.squeeze.squeeze_days,
            "squeeze_momentum": ctx.squeeze.momentum_direction,
            "weekly_trend": ctx.weekly.trend,
            "weekly_supports_long": ctx.weekly.supports_long,
            "weekly_supports_short": ctx.weekly.supports_short,
            "vp_shape": ctx.vp.profile_shape,
            "vp_position": ctx.vp.price_position,
            "iv_regime": ctx.iv.iv_regime,
            "bull_score": bull,
            "bear_score": bear,
            "signal": str(signal),
            "confidence": conf,
            "raw_long_score": raw_long,
            "raw_short_score": raw_short,
            "raw_score": raw_score,
            "v2ctx": ctx,
        }
    except Exception as e:
        logger.debug(f"Broad scan failed for {symbol}: {e}")
        return None


def _broad_scan(symbols: List[str], scanner) -> List[Dict]:
    """Phase 2: Parallel broad scan of all symbols. Returns sorted candidates."""
    candidates = []

    futures = {_pool.submit(_scan_one_symbol, scanner, sym): sym for sym in symbols}

    done_count = 0
    for future in as_completed(futures):
        done_count += 1
        sym = futures[future]
        try:
            result = future.result(timeout=30)
            if result:
                candidates.append(result)
        except Exception as e:
            logger.debug(f"Broad scan timeout/error for {sym}: {e}")

        if done_count % 15 == 0:
            logger.info(f"  Broad scan: {done_count}/{len(symbols)} complete...")

    # Sort by raw score descending — no candidates eliminated
    candidates.sort(key=lambda x: x["raw_score"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════
#  PHASE 3: Deep Enrichment (Parallel)
# ═══════════════════════════════════════════════════════

def _enrich_squeeze(symbol: str) -> Dict:
    """Get detailed squeeze metrics."""
    try:
        from squeeze_detector_v2 import SqueezeDetectorV2
        detector = SqueezeDetectorV2()
        metrics = detector.analyze(symbol)
        if metrics:
            return {
                "score": getattr(metrics, "score", 0) or 0,
                "tier": getattr(metrics, "tier", "NONE"),
                "ttm_squeeze": getattr(metrics, "ttm_squeeze", False),
                "squeeze_duration": getattr(metrics, "squeeze_duration", 0) or 0,
                "direction_bias": getattr(metrics, "direction_bias", "neutral"),
                "quality_grade": getattr(metrics, "quality_grade", "?"),
                "setup_type": getattr(metrics, "setup_type", ""),
                "entry_trigger": getattr(metrics, "entry_trigger", ""),
                "release_firing": bool(getattr(getattr(metrics, "release", None), "is_firing", False)),
            }
    except Exception as e:
        logger.debug(f"Squeeze enrich failed for {symbol}: {e}")
    return {"score": 0, "tier": "NONE", "ttm_squeeze": False, "release_firing": False}


def _enrich_odds(symbol: str) -> Dict:
    """Get historical probability context."""
    result = {
        "call_hit_3d": 0, "call_win_1d": 0, "straddle_rate": 0,
        "regime": "", "zscore": 0, "expected_up_1d": 0, "expected_dn_1d": 0,
    }
    try:
        tool_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polygon_signal_tool")
        if tool_dir not in sys.path:
            sys.path.insert(0, tool_dir)

        from signal_endpoints import _run_analysis
        analysis = _run_analysis(symbol, 365)
        if not analysis:
            return result

        sig = analysis.get("signal", {})
        stats = analysis.get("all_stats", {})
        today = sig.get("today", {})

        # Pick scenario
        if today.get("color") == "RED":
            rs = today.get("rstreak", 1)
            call_key = f"call_red{rs}" if rs >= 2 and f"call_red{rs}" in stats else "call_red"
        else:
            gs = today.get("gstreak", 1)
            call_key = f"call_green{gs}" if gs >= 2 and f"call_green{gs}" in stats else "call_green"

        cs = stats.get(call_key, stats.get("call_all", {}))
        if cs:
            result["call_hit_3d"] = round(cs.get("rate_3d", 0) * 100, 1)
            result["call_win_1d"] = round(cs.get("close_win_1d", 0) * 100, 1)
            result["sample_size"] = cs.get("count", 0)

        straddle = analysis.get("straddle", {})
        result["straddle_rate"] = round(straddle.get("at_least_one_rate", 0) * 100, 1)

        result["expected_up_1d"] = round(sig.get("expected_upside", 0), 2)
        result["expected_dn_1d"] = round(sig.get("expected_downside", 0), 2)

        vr = analysis.get("vol_regime", {})
        ext = analysis.get("extension", {})
        result["regime"] = vr.get("regime", "")
        result["zscore"] = ext.get("zscore", 0)

    except Exception as e:
        logger.debug(f"Odds enrich failed for {symbol}: {e}")
    return result


def _enrich_war_room(symbol: str) -> Dict:
    """Get extension DNA and fade signals."""
    result = {"fade_conviction": 50, "thin_top_pct": 50, "exhaustion": 0}
    try:
        from war_room import get_master_analysis, _compute_signals
        dna = get_master_analysis(symbol, lookback_days=45)
        if not dna:
            return result

        result["avg_up_ext"] = dna.get("avg_up", 0)
        result["avg_dn_ext"] = dna.get("avg_down", 0)
        result["thin_top_pct"] = dna.get("thin_top_pct", 50)
        result["reversal_pct"] = dna.get("reversal_pct", 0)
        result["vwap_revert_rate"] = dna.get("vwap_revert_rate", 0)
        result["peak_hour"] = dna.get("peak_hour", 0)

        sig = _compute_signals(dna, None, [])
        result["exhaustion"] = sig.get("exhaustion", 0)
        result["fade_conviction"] = sig.get("fade_conviction", 50)
        result["war_signals"] = sig.get("signals", [])

    except Exception as e:
        logger.debug(f"War Room enrich failed for {symbol}: {e}")
    return result


def _enrich_extension(symbol: str) -> Dict:
    """Get extension/snap-back analysis."""
    result = {"extension_score": 0, "snap_back_probability": 0, "zone": "unknown", "setup_type": ""}
    try:
        from extension_predictor_v2 import ExtensionPredictorV2
        df = get_bars(symbol, period="1mo", interval="1h")
        if df.empty or len(df) < 20:
            return result
        df.columns = [c.lower() for c in df.columns]

        predictor = ExtensionPredictorV2()
        analysis = predictor.analyze(df, symbol)
        if analysis:
            result["extension_score"] = getattr(analysis, "extension_score", 0) or 0
            result["snap_back_probability"] = getattr(analysis, "snap_back_probability", 0) or 0
            result["zone"] = getattr(analysis, "zone", "unknown")
            result["setup_type"] = getattr(analysis, "setup_type", "")
            result["trigger_level"] = getattr(analysis, "trigger_level", "NONE")
            result["quality_grade"] = getattr(analysis, "quality_grade", "?")

    except Exception as e:
        logger.debug(f"Extension enrich failed for {symbol}: {e}")
    return result


def _deep_enrich(candidates: List[Dict]) -> List[Dict]:
    """Phase 3: Run 4 enrichments per symbol in parallel."""
    enrichment_tasks = []

    for c in candidates:
        sym = c["symbol"]
        enrichment_tasks.append(("squeeze", sym, _enrich_squeeze, sym))
        enrichment_tasks.append(("odds", sym, _enrich_odds, sym))
        enrichment_tasks.append(("war_room", sym, _enrich_war_room, sym))
        enrichment_tasks.append(("extension", sym, _enrich_extension, sym))

    # Submit all tasks
    futures = {}
    for kind, sym, fn, arg in enrichment_tasks:
        future = _pool.submit(fn, arg)
        futures[future] = (kind, sym)

    # Collect results
    results_map = {}  # {symbol: {kind: result}}
    for future in as_completed(futures):
        kind, sym = futures[future]
        if sym not in results_map:
            results_map[sym] = {}
        try:
            results_map[sym][kind] = future.result(timeout=25)
        except Exception as e:
            logger.debug(f"Enrichment {kind} failed for {sym}: {e}")
            results_map[sym][kind] = {}

    # Merge into candidates
    for c in candidates:
        sym = c["symbol"]
        enrichments = results_map.get(sym, {})
        c["squeeze_detail"] = enrichments.get("squeeze", {})
        c["odds"] = enrichments.get("odds", {})
        c["war_room"] = enrichments.get("war_room", {})
        c["extension"] = enrichments.get("extension", {})

    return candidates


# ═══════════════════════════════════════════════════════
#  PHASE 4: Classification + Scoring
# ═══════════════════════════════════════════════════════

def _classify_setup(c: Dict) -> str:
    """Classify a candidate into a setup type based on enrichment data."""
    sq = c.get("squeeze_detail", {})
    ext = c.get("extension", {})
    wr = c.get("war_room", {})
    rsi = c.get("rsi", 50)

    # Priority 1: Squeeze break (release firing or textbook/prime squeeze)
    if sq.get("release_firing") or (sq.get("tier") in ("PRIME", "TEXTBOOK") and sq.get("ttm_squeeze")):
        return "squeeze_break"
    if sq.get("tier") in ("ACTIVE", "PRIME") and sq.get("ttm_squeeze"):
        return "squeeze_break"

    # Priority 2: Capitulation bounce (RSI extreme — rare, high-conviction)
    if rsi < 30 or rsi > 75:
        return "capitulation_bounce"

    # Priority 3: Extension fade (STRONG fade conviction from war room)
    if wr.get("fade_conviction", 0) >= 70:
        return "extension_fade"

    # Priority 4: Mean reversion (high extension score = price stretched from value)
    if ext.get("extension_score", 0) >= 55:
        return "mean_reversion"

    # Priority 5: Trend continuation (most common — trending or neutral)
    return "trend_continuation"


def _score_dimension_v2_signal(c: Dict) -> float:
    """Score the V2 signal dimension (0-100)."""
    direction = "long" if c["bull_score"] > c["bear_score"] else "short"
    return c["bull_score"] if direction == "long" else c["bear_score"]


def _score_dimension_squeeze(c: Dict) -> float:
    """Score the squeeze dimension (0-100)."""
    sq = c.get("squeeze_detail", {})
    score = sq.get("score", 0)

    # Tier bonuses
    tier = sq.get("tier", "NONE")
    if tier == "TEXTBOOK":
        score = min(100, score + 20)
    elif tier == "PRIME":
        score = min(100, score + 15)
    elif tier == "ACTIVE":
        score = min(100, score + 10)

    # Release firing bonus
    if sq.get("release_firing"):
        score = min(100, score + 20)

    return score


def _score_dimension_odds(c: Dict) -> float:
    """Score the historical odds dimension (0-100)."""
    odds = c.get("odds", {})
    call_hit = odds.get("call_hit_3d", 0)
    call_win = odds.get("call_win_1d", 0)
    straddle = odds.get("straddle_rate", 0)

    # Weighted blend, normalized to 0-100
    score = call_hit * 0.4 + call_win * 0.3 + straddle * 0.3

    # Regime bonus/penalty
    regime = odds.get("regime", "")
    if regime == "STABLE":
        score = min(100, score + 10)
    elif regime == "EXTREME":
        score = max(0, score - 10)

    # Z-score penalty (already extended = less room)
    zscore = abs(odds.get("zscore", 0))
    if zscore >= 2.0:
        score = max(0, score - 15)
    elif zscore >= 1.5:
        score = max(0, score - 5)

    return min(100, max(0, score))


def _score_dimension_structure(c: Dict) -> float:
    """Score the weekly structure dimension (0-100)."""
    trend = c.get("weekly_trend", "NEUTRAL")
    direction = "long" if c["bull_score"] > c["bear_score"] else "short"

    # Base score from weekly trend
    trend_scores = {
        "STRONG_UPTREND": 90, "UPTREND": 70,
        "NEUTRAL": 40,
        "DOWNTREND": 30, "STRONG_DOWNTREND": 15,
    }
    base = trend_scores.get(trend, 40)

    # If direction aligns with trend, bonus; if against, penalty
    if direction == "long" and trend in ("STRONG_UPTREND", "UPTREND"):
        base = min(100, base + 10)
    elif direction == "short" and trend in ("STRONG_DOWNTREND", "DOWNTREND"):
        base = min(100, base + 10)
    elif direction == "long" and trend in ("DOWNTREND", "STRONG_DOWNTREND"):
        base = max(0, base - 20)
    elif direction == "short" and trend in ("UPTREND", "STRONG_UPTREND"):
        base = max(0, base - 20)

    # VP position bonus
    vp_pos = c.get("vp_position", "unknown")
    if direction == "long" and vp_pos in ("above_va", "at_poc"):
        base = min(100, base + 10)
    elif direction == "short" and vp_pos in ("below_va", "at_poc"):
        base = min(100, base + 10)

    return base


def _score_dimension_war_room(c: Dict) -> float:
    """Score the war room dimension (0-100)."""
    wr = c.get("war_room", {})
    direction = "long" if c["bull_score"] > c["bear_score"] else "short"

    fade = wr.get("fade_conviction", 50)
    thin_top = wr.get("thin_top_pct", 50)

    if direction == "long":
        # For longs: low fade conviction is GOOD
        score = 100 - fade
        # Thin tops are BAD for longs (no volume support at highs)
        if thin_top > 70:
            score = max(0, score - 15)
        elif thin_top < 35:
            score = min(100, score + 10)
    else:
        # For shorts: high fade conviction is GOOD
        score = fade
        # Thin tops are GOOD for shorts (highs likely to reject)
        if thin_top > 70:
            score = min(100, score + 15)

    return max(0, min(100, score))


def _score_dimension_extension(c: Dict) -> float:
    """Score the extension dimension (0-100)."""
    ext = c.get("extension", {})
    return ext.get("extension_score", 0)


def _compute_alpha_score(c: Dict, regime: Dict) -> Tuple[float, Dict]:
    """
    Compute the final alpha score using type-specific weighted scoring.
    Returns (alpha_score, score_breakdown).
    """
    setup_type = c.get("setup_type", "trend_continuation")
    weights = SETUP_WEIGHTS.get(setup_type, SETUP_WEIGHTS["trend_continuation"])

    # Score each dimension (0-100)
    dimension_scores = {
        "v2_signal": _score_dimension_v2_signal(c),
        "squeeze": _score_dimension_squeeze(c),
        "odds": _score_dimension_odds(c),
        "structure": _score_dimension_structure(c),
        "war_room": _score_dimension_war_room(c),
        "extension": _score_dimension_extension(c),
    }

    # Weighted average: weights sum to 1.0, so result is naturally 0-100
    raw_alpha = 0
    for dim, score in dimension_scores.items():
        raw_alpha += score * weights[dim]

    # Apply regime multiplier
    direction = "long" if c["bull_score"] > c["bear_score"] else "short"
    if direction == "long":
        multiplier = regime.get("long_multiplier", 1.0)
    else:
        multiplier = regime.get("short_multiplier", 1.0)

    alpha_score = min(100, max(0, round(raw_alpha * multiplier, 1)))

    breakdown = {dim: round(score, 1) for dim, score in dimension_scores.items()}
    breakdown["regime_multiplier"] = multiplier
    breakdown["raw_alpha"] = round(raw_alpha, 1)

    return alpha_score, breakdown


# ═══════════════════════════════════════════════════════
#  DURATION TIER + TRADE TIMING RECOMMENDATION
# ═══════════════════════════════════════════════════════

def _assign_duration_tier(c: Dict) -> Dict:
    """
    Data-driven trade duration and timing recommendation.

    Uses enrichment signals to determine:
      - Duration tier (DAY / SWING / POSITION / MACRO)
      - Specific hold period estimate
      - Entry timing guidance
      - Exit strategy suggestion

    Decision matrix (reading real data, not just setup_type):
      1. Extension zone + snap-back prob → how fast the move resolves
      2. Vol regime + z-score → how much room is left
      3. VWAP revert rate → mean-reversion speed
      4. Squeeze state → coiled energy timing
      5. Weekly trend alignment → trend durability
      6. Fade conviction → intraday vs multi-day edge
      7. Call hit 3D rate + regime stability → probability persistence
    """
    setup_type = c.get("setup_type", "default")
    sq = c.get("squeeze_detail", {})
    odds = c.get("odds", {})
    wr = c.get("war_room", {})
    ext = c.get("extension", {})
    direction = "LONG" if c.get("bull_score", 0) > c.get("bear_score", 0) else "SHORT"
    rsi = c.get("rsi", 50)
    weekly = c.get("weekly_trend", "NEUTRAL")

    # ── Collect timing signals ──
    ext_score = ext.get("extension_score", 0)
    snap_prob = ext.get("snap_back_probability", 0)
    zone = ext.get("zone", "unknown")
    fade_conv = wr.get("fade_conviction", 50)
    vwap_revert = wr.get("vwap_revert_rate", 0)
    call_3d = odds.get("call_hit_3d", 0)
    regime = odds.get("regime", "")
    zscore = abs(odds.get("zscore", 0))
    squeeze_tier = sq.get("tier", "NONE")
    squeeze_dur = sq.get("squeeze_duration", 0)

    # ── Score each timeframe (higher = more evidence for that duration) ──
    day_score = 0
    swing_score = 0
    position_score = 0
    entry_notes = []
    exit_notes = []

    # --- Fade conviction: high = intraday edge, low = multi-day hold ---
    if fade_conv >= 70:
        day_score += 30
        entry_notes.append("Intraday fade setup — enter on exhaustion signal")
    elif fade_conv >= 55:
        day_score += 15
        swing_score += 10
    elif fade_conv <= 25:
        position_score += 15
        entry_notes.append("No fade pressure — can hold through pullbacks")

    # --- VWAP revert rate: high = faster mean reversion ---
    if vwap_revert >= 65:
        day_score += 15
        swing_score += 10
        entry_notes.append(f"High VWAP reversion ({vwap_revert:.0f}%) — price tends to snap back quickly")
    elif vwap_revert >= 45:
        swing_score += 15
    else:
        position_score += 10
        entry_notes.append("Low VWAP reversion — moves take time to develop")

    # --- Extension zone: extreme = faster resolution ---
    if zone in ("extreme_above", "extreme_below"):
        if ext_score >= 60:
            swing_score += 20
            entry_notes.append(f"Extreme extension (score {ext_score}) — snap-back likely within 3-5 days")
        else:
            swing_score += 10
    elif zone in ("vah_poc", "poc_val"):
        position_score += 15
        entry_notes.append("Price in value area — breakout needs time")

    # --- Snap-back probability: higher = shorter duration ---
    if snap_prob >= 85:
        swing_score += 15
        day_score += 5
    elif snap_prob >= 70:
        swing_score += 10
    else:
        position_score += 10

    # --- Squeeze: active/forming squeezes need patience ---
    if sq.get("release_firing"):
        swing_score += 25
        entry_notes.append("Squeeze FIRING — ride the release for 3-5 days")
        exit_notes.append("Exit on momentum deceleration or first red day after gap")
    elif squeeze_tier in ("PRIME", "TEXTBOOK"):
        swing_score += 15
        position_score += 10
        entry_notes.append(f"Squeeze {squeeze_tier} — may fire this week")
    elif squeeze_tier == "FORMING" or sq.get("ttm_squeeze"):
        position_score += 20
        entry_notes.append(f"Squeeze building ({squeeze_dur} days) — accumulate, wait for release")
        exit_notes.append("Hold until squeeze fires, then trail stop 3-5 days")

    # --- Vol regime + Z-score: how much room left ---
    if regime == "STABLE" and zscore < 1.0:
        position_score += 15
        entry_notes.append(f"Stable regime, low z-score ({zscore:.1f}) — room to run")
    elif regime == "STABLE" and zscore >= 1.5:
        swing_score += 10
        entry_notes.append(f"Stable but stretched (z={zscore:.1f}) — take profits sooner")
    elif regime == "EXPANDING":
        swing_score += 15
        entry_notes.append("Expanding vol — faster moves, tighter holds")
        exit_notes.append("Trail stop aggressively in expanding vol")
    elif regime == "EXTREME":
        day_score += 15
        entry_notes.append("Extreme vol — day-trade timeframe safest")
        exit_notes.append("Same-day exits preferred in extreme vol")

    # --- Call hit rate: high + stable = longer hold is justified ---
    if call_3d >= 85 and regime == "STABLE":
        position_score += 15
        entry_notes.append(f"Call hit {call_3d:.0f}% + stable — high-prob swing/position")
    elif call_3d >= 75:
        swing_score += 10
    elif call_3d < 55:
        day_score += 10
        entry_notes.append(f"Low call hit ({call_3d:.0f}%) — shorter duration reduces risk")

    # --- Weekly trend alignment ---
    bullish_weekly = weekly in ("UPTREND", "STRONG_UPTREND")
    bearish_weekly = weekly in ("DOWNTREND", "STRONG_DOWNTREND")
    aligned = (direction == "LONG" and bullish_weekly) or (direction == "SHORT" and bearish_weekly)
    counter = (direction == "LONG" and bearish_weekly) or (direction == "SHORT" and bullish_weekly)

    if aligned:
        position_score += 15
        entry_notes.append(f"Weekly trend aligned ({weekly}) — can hold longer")
    elif counter:
        day_score += 10
        swing_score += 5
        entry_notes.append(f"Counter-trend ({weekly}) — shorter holds, tighter stops")
        exit_notes.append("Don't fight the weekly — take quick profits")

    # --- RSI extremes ---
    if rsi > 75:
        day_score += 10
        swing_score += 5
        entry_notes.append(f"RSI overbought ({rsi:.0f}) — reversal could be quick")
    elif rsi < 30:
        day_score += 10
        swing_score += 5
        entry_notes.append(f"RSI oversold ({rsi:.0f}) — bounce could be sharp but brief")

    # ── Determine winner ──
    scores = {"DAY": day_score, "SWING": swing_score, "POSITION": position_score}
    tier = max(scores, key=scores.get)

    # Map to human labels
    tier_labels = {
        "DAY": "Day Trade (0-1 days)",
        "SWING": "Swing Trade (3-5 days)",
        "POSITION": "Position Trade (1-3 weeks)",
    }
    label = tier_labels[tier]

    # Build specific recommendation
    if not exit_notes:
        if tier == "DAY":
            exit_notes.append("Close by end of day or next morning gap")
        elif tier == "SWING":
            exit_notes.append("Trail stop at prior day's low; take profits at +3-5%")
        else:
            exit_notes.append("Hold with weekly stop; scale out at key resistance levels")

    # Confidence in duration (how dominant is the winning tier)
    total = max(1, sum(scores.values()))
    duration_confidence = round(scores[tier] / total * 100)

    return {
        "duration_tier": tier,
        "setup_type": setup_type,
        "duration_label": label,
        "duration_scores": scores,
        "duration_confidence": duration_confidence,
        "entry_timing": entry_notes[:4],
        "exit_strategy": exit_notes[:3],
    }


# ═══════════════════════════════════════════════════════
#  VERDICT BUILDER
# ═══════════════════════════════════════════════════════

def _build_verdict(c: Dict) -> str:
    """Generate a human-readable verdict with trade timing recommendation."""
    sym = c["symbol"]
    alpha = c.get("alpha_score", 0)
    direction = c.get("direction", "LONG")
    setup_type = c.get("setup_type", "unknown")
    tier_label = c.get("duration_label", "3-5 Day Swing")
    dur_conf = c.get("duration_confidence", 0)

    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    parts = [f"{dir_emoji} **{sym}** — Alpha Score: **{alpha}/100** ({direction})"]
    parts.append(f"Setup: {setup_type.replace('_', ' ').title()} | ⏱️ {tier_label} ({dur_conf}% confidence)")

    # Key drivers
    drivers = []
    sq = c.get("squeeze_detail", {})
    odds = c.get("odds", {})
    wr = c.get("war_room", {})

    if sq.get("release_firing"):
        drivers.append("Squeeze FIRING")
    elif sq.get("tier") in ("PRIME", "TEXTBOOK"):
        drivers.append(f"Squeeze {sq['tier']}")
    elif sq.get("ttm_squeeze"):
        drivers.append(f"Squeeze {sq.get('tier', 'ACTIVE')}")

    if odds.get("call_hit_3d", 0) >= 65:
        drivers.append(f"Call hit 3D: {odds['call_hit_3d']}%")
    if c.get("weekly_trend") in ("STRONG_UPTREND", "UPTREND") and direction == "LONG":
        drivers.append(f"Weekly {c['weekly_trend']}")
    elif c.get("weekly_trend") in ("STRONG_DOWNTREND", "DOWNTREND") and direction == "SHORT":
        drivers.append(f"Weekly {c['weekly_trend']}")
    if odds.get("regime") == "STABLE":
        drivers.append("Stable regime")
    if c.get("extension", {}).get("snap_back_probability", 0) >= 60:
        drivers.append(f"Snap-back {c['extension']['snap_back_probability']}%")

    if drivers:
        parts.append("Drivers: " + ", ".join(drivers[:4]))

    # Risks
    risks = []
    zscore = abs(odds.get("zscore", 0))
    if zscore >= 1.5:
        risks.append(f"Z-Score {odds['zscore']:.1f}")
    if direction == "LONG" and wr.get("fade_conviction", 0) >= 45:
        risks.append(f"Fade risk {wr['fade_conviction']}%")
    elif direction == "SHORT" and wr.get("fade_conviction", 0) <= 20:
        risks.append("Low fade conviction")
    if odds.get("regime") == "EXTREME":
        risks.append("Extreme vol regime")
    if wr.get("thin_top_pct", 0) > 70 and direction == "LONG":
        risks.append(f"Thin tops {wr['thin_top_pct']}%")

    if risks:
        parts.append("Risks: " + ", ".join(risks[:3]))

    # Expected range
    up1 = odds.get("expected_up_1d", 0)
    dn1 = odds.get("expected_dn_1d", 0)
    if up1 or dn1:
        parts.append(f"Expected 1D: +{up1}% / -{dn1}%")

    # Trade timing recommendation
    entry_timing = c.get("entry_timing", [])
    exit_strategy = c.get("exit_strategy", [])
    if entry_timing:
        parts.append("📍 Entry: " + " | ".join(entry_timing[:3]))
    if exit_strategy:
        parts.append("🎯 Exit: " + " | ".join(exit_strategy[:2]))

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════

def run_alpha_scan(universe: str = "all", max_results: int = 5,
                   config=None) -> Dict:
    """
    Run the Alpha Scanner V2 pipeline.

    Args:
        universe: Universe preset name (tech, semis, momentum, etfs, mag7, all)
        max_results: Maximum results to return
        config: Optional SwingTradeConfig for tuning scoring thresholds

    Returns:
        Pipeline dict with steps, results, and meta.
    """
    start_time = datetime.now(timezone.utc)
    pipeline = {"steps": {}, "results": [], "meta": {}}

    # ── Phase 1: Market Regime ──
    logger.info("Alpha V2: Phase 1 — Market Regime")
    regime = _compute_market_regime()
    pipeline["steps"]["market_regime"] = regime

    # ── Phase 2: Broad Scan ──
    logger.info("Alpha V2: Phase 2 — Broad Scan")
    symbols = UNIVERSES.get(universe, UNIVERSES.get("all", []))
    if not symbols:
        symbols = UNIVERSES.get("all", [])

    if _finnhub_available:
        api_key = os.environ.get("POLYGON_API_KEY", "")
        scanner = FinnhubScanner(api_key)
        candidates = _broad_scan(symbols, scanner)
    else:
        logger.warning("Alpha V2: FinnhubScanner unavailable — using lightweight fallback scan")
        candidates = _fallback_scan(symbols)
        pipeline["steps"]["fallback_used"] = True

    pipeline["steps"]["broad_scan"] = {
        "scanned": len(symbols),
        "returned": len(candidates),
        "top_raw_scores": [
            {"symbol": c["symbol"], "raw_score": c["raw_score"],
             "bull": c["bull_score"], "bear": c["bear_score"]}
            for c in candidates[:5]
        ],
    }

    if not candidates:
        pipeline["meta"]["verdict"] = "NO CANDIDATES — broad scan returned nothing"
        pipeline["meta"]["duration_sec"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        return pipeline

    # Take top 10 for deep enrichment (balances API call volume vs coverage)
    top = candidates[:10]

    # ── Phase 3: Deep Enrichment ──
    logger.info(f"Alpha V2: Phase 3 — Deep Enrich ({len(top)} symbols)")
    top = _deep_enrich(top)

    pipeline["steps"]["deep_enrich"] = {
        "enriched": len(top),
        "squeeze_found": sum(1 for c in top if c.get("squeeze_detail", {}).get("ttm_squeeze")),
        "high_odds": sum(1 for c in top if c.get("odds", {}).get("call_hit_3d", 0) >= 60),
    }

    # ── Phase 4: Classify + Score ──
    logger.info("Alpha V2: Phase 4 — Classify + Score")
    for c in top:
        # Classify setup type
        c["setup_type"] = _classify_setup(c)

        # Compute alpha score with type-specific weights + regime multiplier
        c["alpha_score"], c["score_breakdown"] = _compute_alpha_score(c, regime)

        # Determine direction
        c["direction"] = "LONG" if c["bull_score"] > c["bear_score"] else "SHORT"

        # Duration tier + trade timing recommendation
        tier_info = _assign_duration_tier(c)
        c["duration_tier"] = tier_info["duration_tier"]
        c["duration_label"] = tier_info["duration_label"]
        c["duration_confidence"] = tier_info["duration_confidence"]
        c["duration_scores"] = tier_info["duration_scores"]
        c["entry_timing"] = tier_info["entry_timing"]
        c["exit_strategy"] = tier_info["exit_strategy"]

        # Verdict
        c["verdict"] = _build_verdict(c)

    # Sort by alpha score descending
    top.sort(key=lambda x: x["alpha_score"], reverse=True)

    # Take top N
    results = top[:max_results]

    # ── Clean up results for output (remove large internal objects) ──
    clean_results = []
    for c in results:
        clean = {k: v for k, v in c.items() if k != "v2ctx"}
        clean_results.append(clean)

    pipeline["results"] = clean_results

    long_count = sum(1 for c in results if c["direction"] == "LONG")
    short_count = sum(1 for c in results if c["direction"] == "SHORT")

    pipeline["steps"]["scoring"] = {
        "setup_types": {st: sum(1 for c in top if c.get("setup_type") == st)
                        for st in SETUP_WEIGHTS},
        "long_candidates": sum(1 for c in top if c["direction"] == "LONG"),
        "short_candidates": sum(1 for c in top if c["direction"] == "SHORT"),
    }

    pipeline["meta"] = {
        "universe": universe,
        "total_scanned": len(symbols),
        "survivors": len(results),
        "market_regime": regime["regime_label"],
        "regime_multiplier_long": regime["long_multiplier"],
        "regime_multiplier_short": regime["short_multiplier"],
        "top_pick": results[0]["symbol"] if results else "NONE",
        "top_score": results[0]["alpha_score"] if results else 0,
        "duration_sec": round((datetime.now(timezone.utc) - start_time).total_seconds(), 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "long_count": long_count,
        "short_count": short_count,
    }

    return pipeline


# ── Async wrapper ──

async def async_run_alpha_scan(universe: str = "all", max_results: int = 5,
                                config=None) -> Dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _pool, lambda: run_alpha_scan(universe, max_results, config)
    )


# ── CLI ──

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    uni = sys.argv[1] if len(sys.argv) > 1 else "mag7"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"\n{'='*60}")
    print(f"  Alpha Scanner V2 — {uni.upper()} universe, top {n}")
    print(f"{'='*60}\n")

    result = run_alpha_scan(uni, max_results=n)

    # Print summary
    meta = result.get("meta", {})
    print(f"Regime: {meta.get('market_regime', '?')} "
          f"(Long ×{meta.get('regime_multiplier_long', 1.0)}, "
          f"Short ×{meta.get('regime_multiplier_short', 1.0)})")
    print(f"Scanned: {meta.get('total_scanned', 0)} → {meta.get('survivors', 0)} results "
          f"({meta.get('long_count', 0)} long, {meta.get('short_count', 0)} short)")
    print(f"Duration: {meta.get('duration_sec', 0)}s\n")

    for i, r in enumerate(result.get("results", []), 1):
        print(f"{'─'*50}")
        print(f"#{i}  {r.get('verdict', r.get('symbol', '?'))}")
        print(f"    Score breakdown: {r.get('score_breakdown', {})}")
        print()

    # Full JSON to file
    with open("alpha_scan_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full results saved to alpha_scan_result.json")
