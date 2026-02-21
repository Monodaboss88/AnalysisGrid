"""
card_data_builder.py — Unified data fetcher for Execution + Thesis trading cards.

Calls 7 scanner endpoints internally (no HTTP), normalizes into a flat dict,
and applies reconciliation logic to resolve cross-scanner conflicts.

Usage:
    from card_data_builder import build_card_data
    data = await build_card_data("META", trade_tf="swing")
"""

import asyncio
import traceback
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# CardData — every field both cards need
# ---------------------------------------------------------------------------

@dataclass
class CardData:
    """All data needed for both Execution and Thesis cards."""

    # ── HEADER ──
    symbol: str = ""
    price: float = 0
    timestamp: str = ""

    # ── SIMPLE SCANNER ──
    simple_signal: str = ""
    simple_bull: float = 0
    simple_bear: float = 0
    simple_confidence: float = 0
    simple_high_prob: float = 50
    simple_low_prob: float = 50

    # ── VOLUME PROFILE ──
    vah: float = 0
    poc: float = 0
    val: float = 0
    vwap: float = 0
    rsi: float = 50
    position: str = ""
    vwap_zone: str = ""
    rvol: float = 1.0
    atr: float = 0

    # ── ORDER FLOW ──
    flow_bias: str = ""
    buy_pressure: float = 0
    sell_pressure: float = 0
    flow_momentum: str = ""
    buy_candles: int = 0
    sell_candles: int = 0

    # ── FIB LEVELS ──
    fib_trend: str = ""
    fib_618: float = 0
    fib_786: float = 0
    swing_high: float = 0
    swing_low: float = 0

    # ── EXTENSION ──
    ext_trigger: float = 0
    ext_snap_prob: float = 0
    ext_direction: str = ""

    # ── MTF RAW ──
    mtf_dominant: str = ""
    mtf_confluence: float = 0
    mtf_high_prob: float = 50
    mtf_low_prob: float = 50
    mtf_weighted_bull: float = 0
    mtf_weighted_bear: float = 0
    mtf_30min_signal: str = ""
    mtf_1hr_signal: str = ""
    mtf_2hr_signal: str = ""
    mtf_4hr_signal: str = ""

    # ── MTF AI TRADE PLAN ──
    mtf_preferred: str = ""
    mtf_key_level: float = 0
    mtf_long_grade: str = ""
    mtf_long_conviction: int = 0
    mtf_long_prob: str = ""
    mtf_long_entry_low: float = 0
    mtf_long_entry_high: float = 0
    mtf_long_stop: float = 0
    mtf_long_t1: float = 0
    mtf_long_t2: float = 0
    mtf_long_rr: str = ""
    mtf_long_ev: str = ""
    mtf_long_trigger: str = ""
    mtf_long_invalid: str = ""
    mtf_long_why: str = ""
    mtf_long_size: str = ""
    mtf_long_hold: str = ""
    mtf_short_grade: str = ""
    mtf_short_conviction: int = 0
    mtf_short_prob: str = ""
    mtf_short_entry_low: float = 0
    mtf_short_entry_high: float = 0
    mtf_short_stop: float = 0
    mtf_short_t1: float = 0
    mtf_short_t2: float = 0
    mtf_short_rr: str = ""
    mtf_short_ev: str = ""
    mtf_short_trigger: str = ""
    mtf_short_invalid: str = ""
    mtf_short_why: str = ""
    mtf_short_size: str = ""
    mtf_short_hold: str = ""

    # ── OPTIONS STRATEGY (from AI plan) ──
    opt_call_strike: float = 0
    opt_call_dte: int = 0
    opt_call_expiry: str = ""
    opt_call_alloc: str = ""
    opt_put_strike: float = 0
    opt_put_dte: int = 0
    opt_put_alloc: str = ""
    opt_ratio: str = ""

    # ── QUICK SCAN / HISTORICAL ODDS ──
    condition: str = ""
    call_hit_1d: float = 0
    call_hit_3d: float = 0
    call_avg_best_3d: float = 0
    put_hit_1d: float = 0
    put_hit_3d: float = 0
    close_win_1d: float = 0
    straddle_rate: float = 0
    expected_up_1d: float = 0
    expected_dn_1d: float = 0
    expected_up_3d: float = 0
    expected_dn_3d: float = 0
    vwap_revert_rate: float = 0
    vwap_crosses: float = 0

    # ── OPTIONS FLOW ──
    flow_sentiment: str = ""
    pc_ratio: float = 0
    max_pain: float = 0
    unusual_count: int = 0
    flow_score: int = 0
    iv_pct: float = 0
    iv_level: str = ""
    expected_move_pct: float = 0
    expected_move_usd: float = 0
    call_wall: float = 0
    put_wall: float = 0
    nearest_dte: int = 0

    # ── WAR ROOM ──
    regime: str = ""
    fade_conviction: int = 0
    avg_close_pos: float = 0
    avg_top_vol: float = 0
    avg_up_ext: float = 0
    war_signals: list = field(default_factory=list)
    exhaustion: float = 0

    # ── BUFFETT BLOOD ──
    buffett_grade: str = ""
    buffett_score: int = 0
    drawdown_pct: float = 0
    blood_score: int = 0
    revenue_growth: float = 0
    range_position: float = 0
    buffett_signal: str = ""

    # ── SUSTAINABILITY ──
    rs_score: int = 0
    rs_grade: str = ""
    cycle_phase: str = ""
    recommended_action: str = ""
    insider_sells: int = 0
    insider_buys: int = 0
    insider_signal: str = ""
    gross_margin: float = 0
    gross_margin_trend: str = ""
    rev_trajectory: str = ""

    # ── RECONCILED (computed) ──
    direction: str = ""  # final LONG or SHORT
    scanner_long: int = 0
    scanner_short: int = 0
    scanner_neutral: int = 0
    working_stop: float = 0
    hard_stop: float = 0
    position_size: str = ""
    hold_period: str = ""
    t1_exit_weight: str = ""


# ---------------------------------------------------------------------------
# Internal data fetchers (import scanner internals, no HTTP)
# ---------------------------------------------------------------------------

async def _fetch_analyze(symbol: str) -> dict:
    """Call the analyze/live endpoint function directly."""
    try:
        from unified_server import analyze_live
        result = await analyze_live(symbol, timeframe="1HR", with_ai=False, vp_period="swing")
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[CardBuilder] analyze error: {e}")
        return {}


async def _fetch_mtf_raw(symbol: str) -> dict:
    """Call MTF raw analysis."""
    try:
        from unified_server import analyze_live_mtf
        result = await analyze_live_mtf(symbol)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[CardBuilder] MTF raw error: {e}")
        return {}


async def _fetch_mtf_ai(symbol: str, trade_tf: str) -> dict:
    """Call MTF AI trade plan."""
    try:
        from unified_server import analyze_mtf_with_ai
        result = await analyze_mtf_with_ai(symbol, trade_tf=trade_tf)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[CardBuilder] MTF AI error: {e}")
        return {}


async def _fetch_signal_quick(symbol: str) -> dict:
    """Call signal quick endpoint."""
    try:
        from signal_endpoints import _run_analysis
        result = await asyncio.to_thread(_run_analysis, symbol.upper(), 365)
        if not result:
            return {}
        # Replicate the /quick response shaping
        sig = result.get("signal", {})
        stats = result.get("all_stats", {})
        straddle = result.get("straddle", {})
        today = sig.get("today", {})

        if today.get("color") == "RED":
            rs = today.get("rstreak", 0)
            call_key = f"call_red{rs}" if rs >= 2 and f"call_red{rs}" in stats else "call_red"
            put_key = f"put_red{rs}" if rs >= 2 and f"put_red{rs}" in stats else "put_red"
        else:
            gs = today.get("gstreak", 0)
            call_key = f"call_green{gs}" if gs >= 2 and f"call_green{gs}" in stats else "call_green"
            put_key = f"put_green{gs}" if gs >= 2 and f"put_green{gs}" in stats else "put_green"

        call_stats = stats.get(call_key, stats.get("call_all", {}))
        put_stats = stats.get(put_key, stats.get("put_all", {}))

        def _s(s):
            if not s:
                return {}
            return {
                "hit_1d": round(s.get("rate_1d", 0) * 100, 1),
                "hit_3d": round(s.get("rate_3d", 0) * 100, 1),
                "avg_best_3d": round(s.get("avg_best_pct_3d", 0), 2),
                "close_win_1d": round(s.get("close_win_1d", 0) * 100, 1),
            }

        return {
            "condition": sig.get("range_condition", ""),
            "call_odds": _s(call_stats),
            "put_odds": _s(put_stats),
            "straddle_rate": round(straddle.get("at_least_one_rate", 0) * 100, 1),
            "expected_up_1d": round(sig.get("expected_upside", 0), 2),
            "expected_dn_1d": round(sig.get("expected_downside", 0), 2),
            "expected_up_3d": round(sig.get("expected_upside_3d", 0), 2),
            "expected_dn_3d": round(sig.get("expected_downside_3d", 0), 2),
        }
    except Exception as e:
        print(f"[CardBuilder] Signal quick error: {e}")
        traceback.print_exc()
        return {}


async def _fetch_options_flow(symbol: str) -> dict:
    """Call options flow scanner."""
    try:
        from options_flow_scanner import scan_tickers
        result = await asyncio.to_thread(scan_tickers, [symbol.upper()])
        results = result.get("results", [])
        return results[0] if results else {}
    except Exception as e:
        print(f"[CardBuilder] Options flow error: {e}")
        return {}


async def _fetch_war_room(symbol: str) -> dict:
    """Call war room analysis."""
    try:
        from war_room import get_master_analysis
        result = await asyncio.to_thread(get_master_analysis, symbol.upper())
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[CardBuilder] War room error: {e}")
        return {}


async def _fetch_buffett(symbol: str) -> dict:
    """Call buffett blood scan."""
    try:
        from buffett_scanner import scan_tickers as buffett_scan
        result = await asyncio.to_thread(buffett_scan, [symbol.upper()])
        results = result.get("results", [])
        return results[0] if results else {}
    except Exception as e:
        print(f"[CardBuilder] Buffett error: {e}")
        return {}


async def _fetch_sustainability(symbol: str) -> dict:
    """Call sustainability analyzer."""
    try:
        from run_sustainability_analyzer import RunSustainabilityAnalyzer
        analyzer = RunSustainabilityAnalyzer()
        result = await asyncio.to_thread(analyzer.analyze, symbol.upper())
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[CardBuilder] Sustainability error: {e}")
        return {}


# ---------------------------------------------------------------------------
# GPT Commentary parser — extract structured fields from AI text
# ---------------------------------------------------------------------------

import re

def _parse_ai_commentary(text: str) -> dict:
    """
    Parse the GPT-generated AI trade plan text into structured fields.
    Extracts grades, entries, stops, targets, R:R, triggers, etc. for both
    LONG and SHORT scenarios, plus verdict.
    """
    d = {}
    if not text:
        return d

    # Split into long / short / verdict sections
    long_start = re.search(r'(🟢\s*(?:SCENARIO\s*1[:\s]*)?LONG\s*SETUP)', text, re.IGNORECASE)
    short_start = re.search(r'(🔴\s*(?:SCENARIO\s*2[:\s]*)?SHORT\s*SETUP)', text, re.IGNORECASE)
    verdict_start = re.search(r'(⚖️\s*VERDICT)', text, re.IGNORECASE)

    long_sec = short_sec = verdict_sec = ""
    if long_start and short_start and verdict_start:
        li, si, vi = long_start.start(), short_start.start(), verdict_start.start()
        if li < si:
            long_sec = text[li:si]
            short_sec = text[si:vi]
        else:
            short_sec = text[si:li]
            long_sec = text[li:vi]
        verdict_sec = text[vi:]

    def _extract_section(sec, prefix):
        """Extract fields from one scenario section."""
        m = re.search(r'GRADE:\s*([A-F][+\-]?)', sec, re.I)
        d[f'{prefix}_grade'] = m.group(1) if m else ""

        m = re.search(r'CONVICTION:\s*(\d+)/10', sec, re.I)
        d[f'{prefix}_conviction'] = int(m.group(1)) if m else 0

        m = re.search(r'PROBABILITY:?\s*([\d]+(?:-[\d]+)?%?\s*\[?\w*\]?)', sec, re.I)
        d[f'{prefix}_prob'] = m.group(1).strip() if m else ""

        m = re.search(r'ENTRY\s*ZONE:\s*\$?([\d,.]+)\s*[-–]\s*\$?([\d,.]+)', sec, re.I)
        if m:
            d[f'{prefix}_entry_low'] = float(m.group(1).replace(',', ''))
            d[f'{prefix}_entry_high'] = float(m.group(2).replace(',', ''))

        m = re.search(r'ENTRY\s*\(midpoint\):\s*\$?([\d,.]+)', sec, re.I)
        if m:
            d[f'{prefix}_entry_mid'] = float(m.group(1).replace(',', ''))

        m = re.search(r'STOP:\s*\$?([\d,.]+)', sec, re.I)
        d[f'{prefix}_stop'] = float(m.group(1).replace(',', '')) if m else 0

        m = re.search(r'T1:\s*\$?([\d,.]+)', sec, re.I)
        d[f'{prefix}_t1'] = float(m.group(1).replace(',', '')) if m else 0

        m = re.search(r'T2:\s*\$?([\d,.]+)', sec, re.I)
        d[f'{prefix}_t2'] = float(m.group(1).replace(',', '')) if m else 0

        m = re.search(r'R:R\s*=\s*([\d.]+:[\d.]+)', sec, re.I)
        d[f'{prefix}_rr'] = m.group(1) if m else ""

        m = re.search(r'EV:\s*\$?([\d,.]+)\s*per\s*\$100.*?→\s*(\w+)', sec, re.I)
        d[f'{prefix}_ev'] = f"${m.group(1)} → {m.group(2)}" if m else ""

        m = re.search(r'SIZE:\s*([\d.]+R)', sec, re.I)
        d[f'{prefix}_size'] = m.group(1) if m else ""

        m = re.search(r'HOLD:\s*(.+?)(?:\n|$)', sec, re.I)
        d[f'{prefix}_hold'] = m.group(1).strip() if m else ""

        m = re.search(r'TRIGGER:\s*(.+?)(?:\n|$)', sec, re.I)
        d[f'{prefix}_trigger'] = m.group(1).strip() if m else ""

        m = re.search(r'INVALID\s*IF:\s*(.+?)(?:\n|$)', sec, re.I)
        d[f'{prefix}_invalid'] = m.group(1).strip() if m else ""

        m = re.search(r'WHY\s*(?:LONG|SHORT):\s*(.+?)(?:\n\n|\n[🔴⚖️]|$)', sec, re.I | re.S)
        d[f'{prefix}_why'] = m.group(1).strip()[:200] if m else ""

    _extract_section(long_sec, 'long')
    _extract_section(short_sec, 'short')

    # Verdict
    m = re.search(r'PREFERRED:\s*(LONG|SHORT)', verdict_sec, re.I)
    d['preferred'] = m.group(1).upper() if m else ""

    m = re.search(r'KEY\s*LEVEL:\s*\$?([\d,.]+)', verdict_sec, re.I)
    d['key_level'] = float(m.group(1).replace(',', '')) if m else 0

    return d


# ---------------------------------------------------------------------------
# Reconciliation — resolve cross-scanner conflicts
# ---------------------------------------------------------------------------

def _reconcile(cd: CardData) -> CardData:
    """
    Apply the reconciliation rules:
    1. Scanner consensus (count long/short/neutral across scanners)
    2. Position size (average Simple + MTF sizing)
    3. Working stop (Fib 618 confirmed by VAL/breakdown)
    4. T1 exit weight (if MTF POC near T1, increase exit %)
    5. Hold period (from MTF AI or default)
    6. Final direction (majority vote)
    """
    # ── Scanner consensus ──
    long_votes = 0
    short_votes = 0
    neutral = 0

    # Simple scanner
    if "LONG" in cd.simple_signal.upper():
        long_votes += 1
    elif "SHORT" in cd.simple_signal.upper():
        short_votes += 1
    else:
        neutral += 1

    # MTF preferred
    if cd.mtf_preferred == "LONG":
        long_votes += 1
    elif cd.mtf_preferred == "SHORT":
        short_votes += 1
    else:
        neutral += 1

    # Options flow sentiment
    if cd.flow_sentiment == "BULLISH":
        long_votes += 1
    elif cd.flow_sentiment == "BEARISH":
        short_votes += 1
    else:
        neutral += 1

    # War Room fade conviction (high fade = contrarian)
    if cd.fade_conviction >= 60:
        short_votes += 1  # high fade conviction = likely reversal
    elif cd.fade_conviction <= 20:
        long_votes += 1  # low fade = trend continuation
    else:
        neutral += 1

    # Buffett signal
    if cd.buffett_signal in ("VALUE", "BLOOD"):
        long_votes += 1
    elif cd.buffett_signal in ("OVERVALUED",):
        short_votes += 1
    else:
        neutral += 1

    # Sustainability cycle
    if cd.cycle_phase in ("EARLY", "MID"):
        long_votes += 1
    elif cd.cycle_phase in ("LATE", "EXTENDED"):
        short_votes += 1
    else:
        neutral += 1

    # Order flow
    if cd.flow_bias == "BULLISH":
        long_votes += 1
    elif cd.flow_bias == "BEARISH":
        short_votes += 1
    else:
        neutral += 1

    cd.scanner_long = long_votes
    cd.scanner_short = short_votes
    cd.scanner_neutral = neutral
    cd.direction = "LONG" if long_votes >= short_votes else "SHORT"

    # ── Position sizing ──
    # Parse MTF size
    mtf_size_str = cd.mtf_long_size if cd.direction == "LONG" else cd.mtf_short_size
    mtf_size = 0.75  # default
    m = re.search(r'([\d.]+)R', mtf_size_str)
    if m:
        mtf_size = float(m.group(1))
    simple_size = 1.0
    avg_size = round((simple_size + mtf_size) / 2, 2)
    # Cap at 0.5R if extended
    if cd.ext_snap_prob >= 70:
        avg_size = min(avg_size, 0.5)
    cd.position_size = f"{avg_size}R"

    # ── Working stop ──
    if cd.direction == "LONG":
        fib_stop = cd.fib_618
        level_stop = cd.val
    else:
        fib_stop = cd.fib_618
        level_stop = cd.vah

    if fib_stop > 0 and level_stop > 0 and abs(fib_stop - level_stop) / max(fib_stop, 1) < 0.02:
        cd.working_stop = round((fib_stop + level_stop) / 2, 2)
    elif fib_stop > 0:
        cd.working_stop = round(fib_stop, 2)
    else:
        cd.working_stop = round(level_stop, 2)

    # ── Hard stop ──
    if cd.direction == "LONG":
        cd.hard_stop = cd.mtf_long_stop if cd.mtf_long_stop > 0 else cd.working_stop
    else:
        cd.hard_stop = cd.mtf_short_stop if cd.mtf_short_stop > 0 else cd.working_stop

    # ── T1 exit weight ──
    if cd.direction == "LONG":
        t1 = cd.mtf_long_t1
    else:
        t1 = cd.mtf_short_t1
    # If MTF POC is near T1, increase exit weight (ceiling confirmed)
    if cd.poc > 0 and t1 > 0 and abs(cd.poc - t1) / max(cd.poc, 1) < 0.02:
        cd.t1_exit_weight = "70-80%"
    else:
        cd.t1_exit_weight = "30-40%"

    # ── Hold period ──
    if cd.direction == "LONG":
        cd.hold_period = cd.mtf_long_hold or "3-5 days"
    else:
        cd.hold_period = cd.mtf_short_hold or "3-5 days"

    return cd


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

async def build_card_data(symbol: str, trade_tf: str = "swing") -> dict:
    """
    Fetch all scanner data, normalize into CardData, reconcile, return as dict.
    Parallelizes independent fetches for speed.
    """
    sym = symbol.upper()
    print(f"[CardBuilder] Building card data for {sym} ({trade_tf})...")

    # Phase 1: Parallel independent fetches
    analyze, mtf_raw, signal, flow, war, buffett, sustain = await asyncio.gather(
        _fetch_analyze(sym),
        _fetch_mtf_raw(sym),
        _fetch_signal_quick(sym),
        _fetch_options_flow(sym),
        _fetch_war_room(sym),
        _fetch_buffett(sym),
        _fetch_sustainability(sym),
        return_exceptions=True,
    )

    # Safely unwrap (exceptions become empty dicts)
    if isinstance(analyze, Exception):
        print(f"[CardBuilder] analyze exception: {analyze}")
        analyze = {}
    if isinstance(mtf_raw, Exception):
        print(f"[CardBuilder] mtf_raw exception: {mtf_raw}")
        mtf_raw = {}
    if isinstance(signal, Exception):
        print(f"[CardBuilder] signal exception: {signal}")
        signal = {}
    if isinstance(flow, Exception):
        print(f"[CardBuilder] flow exception: {flow}")
        flow = {}
    if isinstance(war, Exception):
        print(f"[CardBuilder] war exception: {war}")
        war = {}
    if isinstance(buffett, Exception):
        print(f"[CardBuilder] buffett exception: {buffett}")
        buffett = {}
    if isinstance(sustain, Exception):
        print(f"[CardBuilder] sustain exception: {sustain}")
        sustain = {}

    # Phase 2: MTF AI (depends on nothing but takes longest — run after phase 1 starts)
    mtf_ai = await _fetch_mtf_ai(sym, trade_tf)

    # Parse AI commentary
    ai_parsed = _parse_ai_commentary(mtf_ai.get("ai_commentary", ""))

    # ── Build CardData ──
    cd = CardData()
    cd.symbol = sym
    cd.price = float(analyze.get("current_price", 0))
    cd.timestamp = datetime.now().isoformat()

    # Simple scanner
    cd.simple_signal = str(analyze.get("signal", ""))
    cd.simple_bull = float(analyze.get("bull_score", 0))
    cd.simple_bear = float(analyze.get("bear_score", 0))
    cd.simple_confidence = float(analyze.get("confidence", 0))
    cd.simple_high_prob = float(analyze.get("high_prob", 50))
    cd.simple_low_prob = float(analyze.get("low_prob", 50))

    # Volume profile
    cd.vah = float(analyze.get("vah", 0))
    cd.poc = float(analyze.get("poc", 0))
    cd.val = float(analyze.get("val", 0))
    cd.vwap = float(analyze.get("vwap", 0))
    cd.rsi = float(analyze.get("rsi", 50))
    cd.position = str(analyze.get("position", ""))
    cd.vwap_zone = str(analyze.get("vwap_zone", ""))
    cd.rvol = float(analyze.get("rvol", 1.0))
    cd.atr = float(analyze.get("atr", 0))

    # Order flow
    of = analyze.get("order_flow") or {}
    cd.flow_bias = str(of.get("flow_bias", ""))
    cd.buy_pressure = float(of.get("buy_pressure", 0))
    cd.sell_pressure = float(of.get("sell_pressure", 0))
    cd.flow_momentum = str(of.get("momentum", ""))
    cd.buy_candles = int(of.get("buy_candles", 0))
    cd.sell_candles = int(of.get("sell_candles", 0))

    # Fib levels
    fib = analyze.get("fib_levels") or {}
    cd.fib_trend = str(fib.get("trend", ""))
    cd.fib_618 = float(fib.get("bear_fib_618", 0))
    cd.fib_786 = float(fib.get("bear_fib_786", 0))
    cd.swing_high = float(fib.get("swing_high", 0))
    cd.swing_low = float(fib.get("swing_low", 0))

    # Extension
    ext = analyze.get("extension") or {}
    hottest = ext.get("hottest_setup") or {}
    cd.ext_trigger = float(hottest.get("trigger", 0))
    cd.ext_snap_prob = float(hottest.get("snap_back_prob", 0))
    cd.ext_direction = str(hottest.get("direction", ""))

    # MTF raw
    cd.mtf_dominant = str(mtf_raw.get("dominant_signal", ""))
    cd.mtf_confluence = float(mtf_raw.get("confluence_pct", 0))
    cd.mtf_high_prob = float(mtf_raw.get("high_prob", 50))
    cd.mtf_low_prob = float(mtf_raw.get("low_prob", 50))
    cd.mtf_weighted_bull = float(mtf_raw.get("weighted_bull", 0))
    cd.mtf_weighted_bear = float(mtf_raw.get("weighted_bear", 0))
    tfs = mtf_raw.get("timeframes") or {}
    cd.mtf_30min_signal = str((tfs.get("30MIN") or {}).get("signal", ""))
    cd.mtf_1hr_signal = str((tfs.get("1HR") or {}).get("signal", ""))
    cd.mtf_2hr_signal = str((tfs.get("2HR") or {}).get("signal", ""))
    cd.mtf_4hr_signal = str((tfs.get("4HR") or {}).get("signal", ""))

    # MTF AI parsed
    cd.mtf_preferred = ai_parsed.get("preferred", str(mtf_ai.get("leading_direction", "")))
    cd.mtf_key_level = ai_parsed.get("key_level", float(mtf_ai.get("poc", 0)))

    cd.mtf_long_grade = ai_parsed.get("long_grade", "")
    cd.mtf_long_conviction = ai_parsed.get("long_conviction", 0)
    cd.mtf_long_prob = ai_parsed.get("long_prob", "")
    cd.mtf_long_entry_low = ai_parsed.get("long_entry_low", 0)
    cd.mtf_long_entry_high = ai_parsed.get("long_entry_high", 0)
    cd.mtf_long_stop = ai_parsed.get("long_stop", 0)
    cd.mtf_long_t1 = ai_parsed.get("long_t1", 0)
    cd.mtf_long_t2 = ai_parsed.get("long_t2", 0)
    cd.mtf_long_rr = ai_parsed.get("long_rr", "")
    cd.mtf_long_ev = ai_parsed.get("long_ev", "")
    cd.mtf_long_trigger = ai_parsed.get("long_trigger", "")
    cd.mtf_long_invalid = ai_parsed.get("long_invalid", "")
    cd.mtf_long_why = ai_parsed.get("long_why", "")
    cd.mtf_long_size = ai_parsed.get("long_size", "")
    cd.mtf_long_hold = ai_parsed.get("long_hold", "")

    cd.mtf_short_grade = ai_parsed.get("short_grade", "")
    cd.mtf_short_conviction = ai_parsed.get("short_conviction", 0)
    cd.mtf_short_prob = ai_parsed.get("short_prob", "")
    cd.mtf_short_entry_low = ai_parsed.get("short_entry_low", 0)
    cd.mtf_short_entry_high = ai_parsed.get("short_entry_high", 0)
    cd.mtf_short_stop = ai_parsed.get("short_stop", 0)
    cd.mtf_short_t1 = ai_parsed.get("short_t1", 0)
    cd.mtf_short_t2 = ai_parsed.get("short_t2", 0)
    cd.mtf_short_rr = ai_parsed.get("short_rr", "")
    cd.mtf_short_ev = ai_parsed.get("short_ev", "")
    cd.mtf_short_trigger = ai_parsed.get("short_trigger", "")
    cd.mtf_short_invalid = ai_parsed.get("short_invalid", "")
    cd.mtf_short_why = ai_parsed.get("short_why", "")
    cd.mtf_short_size = ai_parsed.get("short_size", "")
    cd.mtf_short_hold = ai_parsed.get("short_hold", "")

    # Options strategy from AI
    cd.opt_call_strike = float(mtf_ai.get("call_strike", 0) or 0)
    cd.opt_put_strike = float(mtf_ai.get("put_strike", 0) or 0)

    # Quick scan / historical odds
    cd.condition = str(signal.get("condition", ""))
    call_o = signal.get("call_odds") or {}
    put_o = signal.get("put_odds") or {}
    cd.call_hit_1d = float(call_o.get("hit_1d", 0))
    cd.call_hit_3d = float(call_o.get("hit_3d", 0))
    cd.call_avg_best_3d = float(call_o.get("avg_best_3d", 0))
    cd.put_hit_1d = float(put_o.get("hit_1d", 0))
    cd.put_hit_3d = float(put_o.get("hit_3d", 0))
    cd.close_win_1d = float(call_o.get("close_win_1d", 0))
    cd.straddle_rate = float(signal.get("straddle_rate", 0))
    cd.expected_up_1d = float(signal.get("expected_up_1d", 0))
    cd.expected_dn_1d = float(signal.get("expected_dn_1d", 0))
    cd.expected_up_3d = float(signal.get("expected_up_3d", 0))
    cd.expected_dn_3d = float(signal.get("expected_dn_3d", 0))

    # VWAP magnet (fetched inside signal_endpoints via war_room)
    try:
        from signal_endpoints import _get_vwap_magnet
        vwap_mag = _get_vwap_magnet(sym)
        cd.vwap_revert_rate = round(float(vwap_mag.get("vwap_revert_rate", 0)) * 100, 1)
        cd.vwap_crosses = float(vwap_mag.get("avg_vwap_crosses", 0))
    except:
        pass

    # Options flow
    cd.flow_sentiment = str(flow.get("sentiment", ""))
    cd.pc_ratio = float(flow.get("pcVolumeRatio", 0))
    cd.max_pain = float(flow.get("maxPain", 0))
    cd.unusual_count = int(flow.get("unusualCount", 0))
    cd.flow_score = int(flow.get("flowScore", 0))
    cd.iv_pct = float(flow.get("avgIVPct", 0))
    cd.iv_level = str(flow.get("ivLevel", ""))
    cd.expected_move_pct = float(flow.get("expectedMovePct", 0))
    cd.expected_move_usd = float(flow.get("expectedMoveUSD", 0))
    cd.nearest_dte = int(flow.get("nearestDTE", 0))

    # OI walls
    oi_walls = flow.get("oiWalls") or []
    best_call_wall = 0
    best_put_wall = 0
    for w in oi_walls:
        if w.get("call_oi", 0) > w.get("put_oi", 0):
            if w["strike"] > cd.price and (best_call_wall == 0 or w["total_oi"] > best_call_wall):
                cd.call_wall = w["strike"]
                best_call_wall = w["total_oi"]
        else:
            if w["strike"] < cd.price and (best_put_wall == 0 or w["total_oi"] > best_put_wall):
                cd.put_wall = w["strike"]
                best_put_wall = w["total_oi"]

    # War room
    regime_data = war.get("regime") or {}
    cd.regime = str(regime_data.get("ext_regime", ""))
    cd.fade_conviction = int(war.get("fade_conviction", 0))
    cd.avg_close_pos = round(float(war.get("avg_close_pos", 0)) * 100, 1)
    cd.avg_top_vol = round(float(war.get("avg_top_vol", 0)) * 100, 1)
    cd.avg_up_ext = round(float(war.get("avg_up", 0)) * 100, 2)
    cd.war_signals = war.get("signals") or []
    cd.exhaustion = float(war.get("exhaustion", 0))

    # Buffett
    cd.buffett_grade = str(buffett.get("grade", ""))
    cd.buffett_score = int(buffett.get("compositeScore", 0))
    cd.drawdown_pct = round(float(buffett.get("drawdownPct", 0)) * 100, 1)
    cd.blood_score = int(buffett.get("bloodScore", 0))
    cd.revenue_growth = round(float(buffett.get("revenueGrowth", 0)) * 100, 1)
    cd.range_position = round(float(buffett.get("rangePosition", 0)) * 100)
    cd.buffett_signal = str(buffett.get("signal", ""))

    # Sustainability
    cd.rs_score = int(sustain.get("overall_score", 0))
    cd.rs_grade = str(sustain.get("overall_grade", ""))
    cycle_pos = sustain.get("cycle_position") or {}
    cd.cycle_phase = str(cycle_pos.get("estimated_cycle_phase", ""))
    cd.recommended_action = str(sustain.get("recommended_action", ""))
    smart = sustain.get("smart_money") or {}
    cd.insider_sells = int(smart.get("insider_sells_6mo", 0))
    cd.insider_buys = int(smart.get("insider_buys_6mo", 0))
    cd.insider_signal = str(smart.get("insider_net_signal", ""))
    rev_health = sustain.get("revenue_health") or {}
    cd.gross_margin = round(float(rev_health.get("gross_margin_current", 0)) * 100, 1)
    cd.gross_margin_trend = str(rev_health.get("gross_margin_trend", ""))
    cd.rev_trajectory = str(cycle_pos.get("revenue_growth_trajectory", ""))

    # ── Reconcile ──
    cd = _reconcile(cd)

    return asdict(cd)
