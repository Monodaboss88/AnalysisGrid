"""
Combo Scanner — Stacked-Edge Setup Detector
============================================
Layers the highest-edge conditions from the signal scanner into tiered setups:

  A-Tier: Rubber-Band Snap  (mean reversion after 2+ red streak + extension)
  B-Tier: Exhaustion Fade   (reversal after 3+ green streak + elevated CLV)
  C-Tier: Compression Break (vol squeeze near breakout + directional lean)

Each ticker gets scored 0-100 based on how many stacked conditions fire.
Only surfaces tickers where a meaningful setup (≥C tier) is present.

Usage:
    from combo_scanner import scan_combos
    results = await scan_combos(["NVDA", "META", "TSLA"])
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Setup type definitions ────────────────────────────────────────────────

SETUP_TYPES = {
    "RUBBER_BAND":       {"label": "Rubber-Band Snap",    "direction": "LONG",  "tier": "A"},
    "EXHAUSTION_FADE":   {"label": "Exhaustion Fade",     "direction": "SHORT", "tier": "B"},
    "COMPRESSION_BREAK": {"label": "Compression Breakout","direction": "LEAN",  "tier": "C"},
}

GRADE_MAP = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
    (0,  "F"),
]


def _grade(score: float) -> str:
    for threshold, letter in GRADE_MAP:
        if score >= threshold:
            return letter
    return "F"


# ── Individual condition checks ───────────────────────────────────────────

@dataclass
class ConditionResult:
    name: str
    fired: bool
    points: float
    detail: str
    icon: str = ""  # ✅ ⚠️ ❌

    def __post_init__(self):
        self.icon = "✅" if self.fired else ("⚠️" if self.points > 0 else "❌")


def _check_streak_red2(signal_data: dict) -> ConditionResult:
    """2+ Red Day Streak — base signal for rubber-band snap."""
    today = signal_data.get("signal", {}).get("today", {})
    rstreak = today.get("rstreak", 0)
    if rstreak >= 3:
        return ConditionResult("Red Streak", True, 30, f"{rstreak} consecutive red days")
    elif rstreak >= 2:
        return ConditionResult("Red Streak", True, 25, f"{rstreak} consecutive red days")
    elif rstreak == 1:
        return ConditionResult("Red Streak", False, 5, "1 red day (need 2+)")
    return ConditionResult("Red Streak", False, 0, "No red streak")


def _check_streak_green3(signal_data: dict) -> ConditionResult:
    """3+ Green Day Streak — base signal for exhaustion fade."""
    today = signal_data.get("signal", {}).get("today", {})
    gstreak = today.get("gstreak", 0)
    if gstreak >= 4:
        return ConditionResult("Green Streak", True, 30, f"{gstreak} consecutive green days")
    elif gstreak >= 3:
        return ConditionResult("Green Streak", True, 25, f"{gstreak} consecutive green days")
    elif gstreak >= 2:
        return ConditionResult("Green Streak", False, 8, f"{gstreak} green days (need 3+)")
    return ConditionResult("Green Streak", False, 0, "No significant green streak")


def _check_extension(signal_data: dict, direction: str) -> ConditionResult:
    """Extension Z-score — oversold (LONG) or overbought (SHORT)."""
    ext = signal_data.get("extension", {})
    zscore = ext.get("zscore", 0)
    status = ext.get("status", "")
    revert = ext.get("revert_after_extreme_rate", 0)

    if direction == "LONG":
        # For longs, we want HIGH z-score (big range = capitulation) or low close
        if zscore >= 2.0:
            return ConditionResult("Extension Z-Score", True, 20,
                                   f"Z={zscore:.2f} EXTREME — {revert:.0f}% revert rate")
        elif zscore >= 1.5:
            return ConditionResult("Extension Z-Score", True, 15,
                                   f"Z={zscore:.2f} STRETCHED — momentum fading")
        elif zscore >= 1.0:
            return ConditionResult("Extension Z-Score", False, 8,
                                   f"Z={zscore:.2f} mildly extended")
        return ConditionResult("Extension Z-Score", False, 0, f"Z={zscore:.2f} normal range")
    else:
        # For shorts (fade), low z-score means compressed, NOT extended yet
        # Actually for exhaustion fades we want the price to be elevated but range normal
        # A negative z-score means today's range was small = calm before the storm
        if zscore < -0.5:
            return ConditionResult("Extension Z-Score", True, 15,
                                   f"Z={zscore:.2f} compressed — calm fade setup")
        elif zscore < 0.5:
            return ConditionResult("Extension Z-Score", True, 10,
                                   f"Z={zscore:.2f} normal range — clean fade")
        elif zscore >= 1.5:
            return ConditionResult("Extension Z-Score", False, 5,
                                   f"Z={zscore:.2f} volatile — fade risky")
        return ConditionResult("Extension Z-Score", False, 3,
                               f"Z={zscore:.2f} slightly elevated")


def _check_close_location(signal_data: dict, direction: str) -> ConditionResult:
    """Close Location Value — where price closed in its range."""
    cl = signal_data.get("close_location", {})
    clv = cl.get("today_clv", 50)  # 0-100 scale
    trend = cl.get("trend_cluster", "")

    if direction == "LONG":
        # Low CLV = closed near lows = sellers exhausted = bullish snap
        if clv <= 20:
            return ConditionResult("Close Location", True, 15,
                                   f"CLV={clv:.0f}% — closed at lows, sellers spent")
        elif clv <= 35:
            return ConditionResult("Close Location", True, 10,
                                   f"CLV={clv:.0f}% — weak close, reversion likely")
        elif clv >= 70:
            return ConditionResult("Close Location", False, 0,
                                   f"CLV={clv:.0f}% — strong close contradicts snap thesis")
        return ConditionResult("Close Location", False, 3, f"CLV={clv:.0f}% — mid-range")
    else:
        # High CLV = closed near highs = last gasp = bearish fade
        if clv >= 80:
            return ConditionResult("Close Location", True, 15,
                                   f"CLV={clv:.0f}% — closed at highs, last gasp")
        elif clv >= 65:
            return ConditionResult("Close Location", True, 10,
                                   f"CLV={clv:.0f}% — elevated close, fade setup")
        elif clv <= 30:
            return ConditionResult("Close Location", False, 0,
                                   f"CLV={clv:.0f}% — already weak, fade played out")
        return ConditionResult("Close Location", False, 3, f"CLV={clv:.0f}% — mid-range")


def _check_straddle(signal_data: dict) -> ConditionResult:
    """Straddle rate — does the stock MOVE enough to trade?"""
    straddle = signal_data.get("straddle", {})
    rate = straddle.get("at_least_one_rate", 0) * 100  # convert to %
    avg_best = straddle.get("avg_daily_best", 0)

    if rate >= 97:
        return ConditionResult("Straddle Rate", True, 10,
                               f"{rate:.0f}% — moves every day, avg ${avg_best:.2f}")
    elif rate >= 92:
        return ConditionResult("Straddle Rate", True, 7,
                               f"{rate:.0f}% — strong mover")
    elif rate >= 85:
        return ConditionResult("Straddle Rate", False, 3,
                               f"{rate:.0f}% — decent movement")
    return ConditionResult("Straddle Rate", False, 0,
                           f"{rate:.0f}% — low movement, avoid")


def _check_vol_regime(signal_data: dict, setup_type: str) -> ConditionResult:
    """Vol regime — does current volatility favor this setup?"""
    vr = signal_data.get("vol_regime", {})
    regime = vr.get("regime", "")
    atr_ratio = vr.get("atr_ratio", 1.0)

    if setup_type == "RUBBER_BAND":
        # Expanding or Extreme vol = bigger snap-back potential
        if regime in ("EXPANDING", "EXTREME"):
            return ConditionResult("Vol Regime", True, 10,
                                   f"{regime} (ATR ratio {atr_ratio:.2f}) — snap-back amplified")
        elif regime == "STABLE":
            return ConditionResult("Vol Regime", False, 5,
                                   f"STABLE (ATR ratio {atr_ratio:.2f}) — normal conditions")
        return ConditionResult("Vol Regime", False, 0,
                               f"SQUEEZE (ATR ratio {atr_ratio:.2f}) — low vol kills snaps")

    elif setup_type == "EXHAUSTION_FADE":
        # Stable or Expanding = healthy for fades
        if regime == "STABLE":
            return ConditionResult("Vol Regime", True, 10,
                                   f"STABLE — clean fade conditions")
        elif regime == "EXPANDING":
            return ConditionResult("Vol Regime", True, 7,
                                   f"EXPANDING — volatile but fadeable")
        return ConditionResult("Vol Regime", False, 3,
                               f"{regime} — suboptimal for fades")

    else:  # COMPRESSION_BREAK
        if regime == "SQUEEZE":
            return ConditionResult("Vol Regime", True, 10,
                                   f"SQUEEZE (ATR ratio {atr_ratio:.2f}) — breakout imminent")
        elif regime == "STABLE" and atr_ratio < 0.90:
            return ConditionResult("Vol Regime", True, 7,
                                   f"Low-end STABLE — compressing")
        return ConditionResult("Vol Regime", False, 0,
                               f"{regime} — not compressed enough")


def _check_historical_odds(signal_data: dict, direction: str, streak: int) -> ConditionResult:
    """Historical hit rate for this exact condition from backtested data."""
    stats = signal_data.get("all_stats", {})

    if direction == "LONG":
        key = f"call_red{streak}" if streak >= 2 else "call_red"
        if key not in stats and streak >= 3:
            key = "call_red2"
        if key not in stats:
            key = "call_red"
    else:
        key = f"put_green{streak}" if streak >= 2 else "put_green"
        if key not in stats and streak >= 3:
            key = "put_green2"
        if key not in stats:
            key = "put_green"

    s = stats.get(key)
    if not s:
        return ConditionResult("Historical Odds", False, 0, "No data for this condition")

    hit_3d = s.get("rate_3d", 0) * 100
    avg_best = s.get("avg_best_pct_3d", 0)
    close_win = s.get("close_win_1d", 0) * 100
    count = s.get("count", 0)

    detail = f"{key}: hit {hit_3d:.0f}% ({count}x), avg best {avg_best:.1f}%, close win {close_win:.0f}%"

    if hit_3d >= 90 and close_win >= 50:
        return ConditionResult("Historical Odds", True, 10, detail)
    elif hit_3d >= 85:
        return ConditionResult("Historical Odds", True, 7, detail)
    elif hit_3d >= 75:
        return ConditionResult("Historical Odds", False, 4, detail)
    return ConditionResult("Historical Odds", False, 0, detail)


def _check_war_room(war_data: dict, direction: str) -> ConditionResult:
    """War Room signals — fade conviction, exhaustion, regime."""
    if not war_data:
        return ConditionResult("War Room", False, 0, "No data")

    signals = war_data.get("signals", [])
    fade_conv = war_data.get("fade_conviction", 0)
    avg_close = war_data.get("avg_close_pos", 50)
    thin_top = war_data.get("thin_top_pct", 0)

    if direction == "LONG":
        # For longs, we want FADING signals (sellers exhausted → reversal)
        if "FADING" in signals or avg_close < 35:
            return ConditionResult("War Room", True, 10,
                                   f"FADING detected, avg close {avg_close:.0f}% — reversal setup")
        elif avg_close < 45:
            return ConditionResult("War Room", False, 5,
                                   f"Avg close {avg_close:.0f}% — slightly weak")
        return ConditionResult("War Room", False, 2,
                               f"Avg close {avg_close:.0f}% — no fade signal")
    else:
        # For shorts, we want thin tops + high fade conviction
        score = 0
        details = []
        if thin_top > 50:
            score += 5
            details.append(f"THIN TOPS {thin_top:.0f}%")
        if fade_conv >= 40:
            score += 5
            details.append(f"Fade conviction {fade_conv}%")
        if "REVERSAL PRONE" in signals:
            score += 3
            details.append("REVERSAL PRONE")
        if "HOD>VWAP" in signals:
            score += 2
            details.append("HOD > VWAP")

        fired = score >= 7
        return ConditionResult("War Room", fired, min(score, 10),
                               "; ".join(details) if details else "No fade signals")


def _check_options_flow(flow_data: dict, direction: str) -> ConditionResult:
    """Options flow alignment — does smart money agree?"""
    if not flow_data:
        return ConditionResult("Options Flow", False, 0, "No data")

    sentiment = str(flow_data.get("flowSentiment", flow_data.get("flow_sentiment", ""))).upper()
    flow_score = flow_data.get("flowScore", flow_data.get("flow_score", 50))
    pc_ratio = flow_data.get("pcRatio", flow_data.get("pc_ratio", 1.0))

    if direction == "LONG":
        if "BULL" in sentiment and flow_score >= 70:
            return ConditionResult("Options Flow", True, 5,
                                   f"{sentiment}, score {flow_score}, P/C {pc_ratio:.2f}")
        elif "BULL" in sentiment:
            return ConditionResult("Options Flow", False, 3,
                                   f"{sentiment} but score {flow_score}")
        elif "BEAR" in sentiment:
            return ConditionResult("Options Flow", False, 0,
                                   f"{sentiment} — smart money opposing")
        return ConditionResult("Options Flow", False, 2, f"Neutral flow")
    else:
        if "BEAR" in sentiment and flow_score <= 40:
            return ConditionResult("Options Flow", True, 5,
                                   f"{sentiment}, score {flow_score}, P/C {pc_ratio:.2f}")
        elif "BEAR" in sentiment:
            return ConditionResult("Options Flow", False, 3,
                                   f"{sentiment} but score {flow_score}")
        elif "BULL" in sentiment:
            return ConditionResult("Options Flow", False, 0,
                                   f"{sentiment} — smart money opposing")
        return ConditionResult("Options Flow", False, 2, f"Neutral flow")


# ── Setup evaluators ─────────────────────────────────────────────────────

@dataclass
class ComboSetup:
    ticker: str
    setup_type: str           # RUBBER_BAND, EXHAUSTION_FADE, COMPRESSION_BREAK
    setup_label: str
    direction: str            # LONG, SHORT, LEAN
    score: float
    grade: str
    conditions: List[dict]    # serialized ConditionResults
    historical_key: str       # e.g. "call_red2"
    hit_3d: float
    avg_best_3d: float
    close_win_1d: float
    expected_move: str        # "$X.XX / Y.Y%"
    streak: int
    clv: float
    zscore: float
    vol_regime: str
    summary: str


def _evaluate_rubber_band(ticker: str, signal_data: dict,
                          war_data: dict, flow_data: dict) -> Optional[ComboSetup]:
    """Evaluate Rubber-Band Snap (mean reversion long after red streak)."""
    today = signal_data.get("signal", {}).get("today", {})
    rstreak = today.get("rstreak", 0)
    if rstreak < 1:
        return None  # At minimum need 1 red day

    conditions = [
        _check_streak_red2(signal_data),
        _check_extension(signal_data, "LONG"),
        _check_close_location(signal_data, "LONG"),
        _check_straddle(signal_data),
        _check_vol_regime(signal_data, "RUBBER_BAND"),
        _check_war_room(war_data, "LONG"),
        _check_options_flow(flow_data, "LONG"),
    ]

    # Add historical odds
    hist = _check_historical_odds(signal_data, "LONG", rstreak)
    conditions.append(hist)

    total_score = sum(c.points for c in conditions)
    total_score = min(100, total_score)

    if total_score < 25:
        return None  # Not enough conditions firing

    # Get historical stats for summary
    stats = signal_data.get("all_stats", {})
    key = f"call_red{rstreak}" if rstreak >= 2 else "call_red"
    if key not in stats and rstreak >= 3:
        key = "call_red2"
    if key not in stats:
        key = "call_all"
    s = stats.get(key, {})

    ext = signal_data.get("extension", {})
    cl = signal_data.get("close_location", {})
    vr = signal_data.get("vol_regime", {})

    fired = sum(1 for c in conditions if c.fired)
    return ComboSetup(
        ticker=ticker,
        setup_type="RUBBER_BAND",
        setup_label="Rubber-Band Snap (LONG)",
        direction="LONG",
        score=total_score,
        grade=_grade(total_score),
        conditions=[asdict(c) for c in conditions],
        historical_key=key,
        hit_3d=round((s.get("rate_3d", 0)) * 100, 1) if s else 0,
        avg_best_3d=round(s.get("avg_best_pct_3d", 0), 2) if s else 0,
        close_win_1d=round((s.get("close_win_1d", 0)) * 100, 1) if s else 0,
        expected_move=f"{s.get('avg_best_pct_3d', 0):.1f}% avg best in 3d" if s else "—",
        streak=rstreak,
        clv=cl.get("today_clv", 50),
        zscore=ext.get("zscore", 0),
        vol_regime=vr.get("regime", ""),
        summary=f"{fired}/{len(conditions)} conditions fired — "
                f"{'STRONG' if fired >= 5 else 'MODERATE' if fired >= 3 else 'WEAK'} setup",
    )


def _evaluate_exhaustion_fade(ticker: str, signal_data: dict,
                               war_data: dict, flow_data: dict) -> Optional[ComboSetup]:
    """Evaluate Exhaustion Fade (short after extended green streak)."""
    today = signal_data.get("signal", {}).get("today", {})
    gstreak = today.get("gstreak", 0)
    if gstreak < 2:
        return None  # Need at least 2 green days

    conditions = [
        _check_streak_green3(signal_data),
        _check_extension(signal_data, "SHORT"),
        _check_close_location(signal_data, "SHORT"),
        _check_straddle(signal_data),
        _check_vol_regime(signal_data, "EXHAUSTION_FADE"),
        _check_war_room(war_data, "SHORT"),
        _check_options_flow(flow_data, "SHORT"),
    ]

    hist = _check_historical_odds(signal_data, "SHORT", gstreak)
    conditions.append(hist)

    total_score = sum(c.points for c in conditions)
    total_score = min(100, total_score)

    if total_score < 25:
        return None

    stats = signal_data.get("all_stats", {})
    key = f"put_green{gstreak}" if gstreak >= 2 else "put_green"
    if key not in stats and gstreak >= 3:
        key = "put_green2"
    if key not in stats:
        key = "put_all"
    s = stats.get(key, {})

    ext = signal_data.get("extension", {})
    cl = signal_data.get("close_location", {})
    vr = signal_data.get("vol_regime", {})

    fired = sum(1 for c in conditions if c.fired)
    return ComboSetup(
        ticker=ticker,
        setup_type="EXHAUSTION_FADE",
        setup_label="Exhaustion Fade (SHORT)",
        direction="SHORT",
        score=total_score,
        grade=_grade(total_score),
        conditions=[asdict(c) for c in conditions],
        historical_key=key,
        hit_3d=round((s.get("rate_3d", 0)) * 100, 1) if s else 0,
        avg_best_3d=round(s.get("avg_best_pct_3d", 0), 2) if s else 0,
        close_win_1d=round((s.get("close_win_1d", 0)) * 100, 1) if s else 0,
        expected_move=f"{s.get('avg_best_pct_3d', 0):.1f}% avg best in 3d" if s else "—",
        streak=gstreak,
        clv=cl.get("today_clv", 50),
        zscore=ext.get("zscore", 0),
        vol_regime=vr.get("regime", ""),
        summary=f"{fired}/{len(conditions)} conditions fired — "
                f"{'STRONG' if fired >= 5 else 'MODERATE' if fired >= 3 else 'WEAK'} setup",
    )


def _evaluate_compression_break(ticker: str, signal_data: dict,
                                 war_data: dict, flow_data: dict,
                                 mtf_data: dict) -> Optional[ComboSetup]:
    """Evaluate Compression Breakout (vol squeeze + directional lean)."""
    vr = signal_data.get("vol_regime", {})
    regime = vr.get("regime", "")
    atr_ratio = vr.get("atr_ratio", 1.0)

    # Must be in squeeze or low-end stable
    if regime not in ("SQUEEZE",) and not (regime == "STABLE" and atr_ratio < 0.90):
        return None

    # Direction from MTF
    dom = str(mtf_data.get("dominant_signal", "")).upper()
    if "LONG" in dom or "BULL" in dom:
        direction = "LONG"
    elif "SHORT" in dom or "BEAR" in dom:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    conditions = [
        _check_vol_regime(signal_data, "COMPRESSION_BREAK"),
        _check_straddle(signal_data),
    ]

    # Extension should be near 0 (flat, no lean)
    ext = signal_data.get("extension", {})
    zscore = ext.get("zscore", 0)
    if abs(zscore) < 0.5:
        conditions.append(ConditionResult("Extension Flat", True, 15,
                                          f"Z={zscore:.2f} — no lean, breakout either way"))
    elif abs(zscore) < 1.0:
        conditions.append(ConditionResult("Extension Flat", False, 5,
                                          f"Z={zscore:.2f} — slight lean"))
    else:
        conditions.append(ConditionResult("Extension Flat", False, 0,
                                          f"Z={zscore:.2f} — already extended"))

    # MTF direction
    if direction != "NEUTRAL":
        conditions.append(ConditionResult("MTF Direction", True, 20,
                                          f"MTF dominant: {dom} → {direction}"))
    else:
        conditions.append(ConditionResult("MTF Direction", False, 5,
                                          f"MTF: {dom} — no clear lean"))

    # Flow
    conditions.append(_check_options_flow(flow_data, direction if direction != "NEUTRAL" else "LONG"))

    total_score = sum(c.points for c in conditions)
    total_score = min(100, total_score)

    if total_score < 25:
        return None

    cl = signal_data.get("close_location", {})
    fired = sum(1 for c in conditions if c.fired)

    return ComboSetup(
        ticker=ticker,
        setup_type="COMPRESSION_BREAK",
        setup_label=f"Compression Breakout ({direction})",
        direction=direction,
        score=total_score,
        grade=_grade(total_score),
        conditions=[asdict(c) for c in conditions],
        historical_key="straddle",
        hit_3d=0,
        avg_best_3d=0,
        close_win_1d=0,
        expected_move="Breakout pending — direction from MTF",
        streak=0,
        clv=cl.get("today_clv", 50),
        zscore=zscore,
        vol_regime=regime,
        summary=f"{fired}/{len(conditions)} conditions fired — "
                f"{'STRONG' if fired >= 4 else 'MODERATE' if fired >= 2 else 'WEAK'} setup",
    )


# ── TTL caches ───────────────────────────────────────────────────────────

_signal_cache: Dict[str, dict] = {}   # ticker → {"data": ..., "ts": float}
_SIGNAL_CACHE_TTL = 120               # 2 minutes

# NOTE: No dedicated _combo_pool — use asyncio.to_thread() which routes to
# the server's default ThreadPoolExecutor(40).  Dedicated pools compete for
# OS threads and Polygon rate-limit tokens, causing cascading stalls.


# ── Data fetchers (all run in thread pool) ────────────────────────────────

def _fetch_signal_sync(ticker: str) -> dict:
    """Fetch signal analysis with TTL cache."""
    ticker = ticker.upper()
    now = time.time()
    if ticker in _signal_cache:
        entry = _signal_cache[ticker]
        if now - entry["ts"] < _SIGNAL_CACHE_TTL:
            logger.debug(f"[Combo] signal cache HIT for {ticker}")
            return entry["data"]
    try:
        from signal_endpoints import _run_analysis
        result = _run_analysis(ticker, 365)
        data = result if result else {}
        _signal_cache[ticker] = {"data": data, "ts": time.time()}
        return data
    except Exception as e:
        logger.debug(f"Combo: signal fetch failed for {ticker}: {e}")
        return {}


def _fetch_war_room_sync(ticker: str) -> dict:
    """Fetch War Room data (uses war_room's built-in 2-min cache)."""
    try:
        from war_room import get_master_analysis, _compute_signals
        dna = get_master_analysis(ticker.upper(), lookback_days=45)
        if not dna:
            return {}
        sig = _compute_signals(dna, None, [])
        regime_dict = dna.get("regime", {})
        return {
            "regime": regime_dict.get("ext_regime", "NORMAL") if isinstance(regime_dict, dict) else str(regime_dict),
            "exhaustion": sig.get("exhaustion", 0),
            "fade_conviction": sig.get("fade_conviction", 0),
            "signals": sig.get("signals", []),
            "avg_close_pos": dna.get("avg_close_pos", 50),
            "thin_top_pct": dna.get("thin_top_pct", 0),
            "reversal_pct": dna.get("reversal_pct", 0),
        }
    except Exception as e:
        logger.debug(f"Combo: war_room failed for {ticker}: {e}")
        return {}


def _fetch_flow_sync(ticker: str) -> dict:
    """Fetch options flow data — calls cached single-scan directly (no pool spawn)."""
    try:
        from options_flow_scanner import _scan_single_cached
        return _scan_single_cached(ticker.upper())
    except Exception as e:
        logger.debug(f"Combo: flow failed for {ticker}: {e}")
        return {}


async def _fetch_mtf(ticker: str) -> dict:
    """Fetch MTF data (already async)."""
    try:
        from unified_server import analyze_live_mtf
        result = await analyze_live_mtf(ticker.upper())
        return result if isinstance(result, dict) else {}
    except Exception as e:
        logger.debug(f"Combo: MTF failed for {ticker}: {e}")
        return {}


async def _fetch_all_for_ticker(ticker: str) -> tuple:
    """Fetch all 4 data sources for a single ticker in parallel.
    Uses asyncio.to_thread() for blocking calls — routes to the server's
    default executor instead of a dedicated pool, preventing thread exhaustion.
    30s timeout prevents any single slow source from stalling the scan.
    """
    signal_fut = asyncio.to_thread(_fetch_signal_sync, ticker)
    war_fut    = asyncio.to_thread(_fetch_war_room_sync, ticker)
    flow_fut   = asyncio.to_thread(_fetch_flow_sync, ticker)
    mtf_fut    = _fetch_mtf(ticker)

    try:
        # Use asyncio.shield so threads finish naturally on timeout
        # (freeing pool slots) instead of becoming permanent zombies
        signal_data, war_data, flow_data, mtf_data = await asyncio.wait_for(
            asyncio.shield(asyncio.gather(signal_fut, war_fut, flow_fut, mtf_fut, return_exceptions=True)),
            timeout=30,
        )
    except asyncio.TimeoutError:
        signal_data, war_data, flow_data, mtf_data = {}, {}, {}, {}

    if isinstance(signal_data, Exception): signal_data = {}
    if isinstance(war_data, Exception):    war_data = {}
    if isinstance(flow_data, Exception):   flow_data = {}
    if isinstance(mtf_data, Exception):    mtf_data = {}

    return signal_data, war_data, flow_data, mtf_data


# ── Core scan functions ───────────────────────────────────────────────────

async def scan_single(ticker: str) -> List[dict]:
    """Scan one ticker for all setup types. Returns list of setups found."""
    t0 = time.time()
    ticker = ticker.upper()

    signal_data, war_data, flow_data, mtf_data = await _fetch_all_for_ticker(ticker)

    elapsed = time.time() - t0
    logger.info(f"[Combo] {ticker} data fetched in {elapsed:.1f}s")

    setups = []
    rb = _evaluate_rubber_band(ticker, signal_data, war_data, flow_data)
    if rb:
        setups.append(asdict(rb))
    ef = _evaluate_exhaustion_fade(ticker, signal_data, war_data, flow_data)
    if ef:
        setups.append(asdict(ef))
    cb = _evaluate_compression_break(ticker, signal_data, war_data, flow_data, mtf_data)
    if cb:
        setups.append(asdict(cb))

    return setups


async def scan_combos(tickers: List[str], min_grade: str = "D") -> dict:
    """
    Scan multiple tickers for combo setups.
    ALL tickers run fully in parallel (unlimited Polygon plan).
    """
    t0 = time.time()
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    min_grade_idx = grade_order.get(min_grade, 3)

    # Run ALL tickers in parallel
    results = await asyncio.gather(
        *[scan_single(t) for t in tickers],
        return_exceptions=True,
    )

    all_setups = []
    for r in results:
        if isinstance(r, list):
            all_setups.extend(r)
        elif isinstance(r, Exception):
            logger.error(f"Combo scan error: {r}")

    # Filter by grade
    filtered = [s for s in all_setups if grade_order.get(s["grade"], 4) <= min_grade_idx]
    filtered.sort(key=lambda s: s["score"], reverse=True)

    elapsed = time.time() - t0
    logger.info(f"[Combo] Full scan: {len(tickers)} tickers in {elapsed:.1f}s — {len(filtered)} setups found")
    return {
        "scan_time": round(elapsed, 1),
        "tickers_scanned": len(tickers),
        "setups_found": len(filtered),
        "results": filtered,
    }


# ── Preset watchlists ────────────────────────────────────────────────────

PRESETS = {
    "mag7": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "CRM", "NFLX", "AVGO", "ORCL"],
    "mega": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "JPM", "V", "UNH", "JNJ", "WMT", "PG", "MA"],
    "etf": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "ARKK", "SOXX"],
    "meme": ["GME", "AMC", "PLTR", "SOFI", "RIVN", "LCID", "NIO", "MARA", "COIN", "HOOD"],
}
