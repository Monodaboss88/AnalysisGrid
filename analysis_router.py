"""
Analysis Router — Chart analysis, MTF analysis, and AI commentary endpoints.
=============================================================================
Extracted from unified_server.py (~760 lines).
Each endpoint runs heavy sync work in a thread pool via safe_timeout.
"""

import os
import json
import time
import math
import asyncio
import logging
from datetime import datetime
from typing import Optional
from collections import OrderedDict
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ── TTL Cache (duplicate-free — import from shared module if desired) ──
class TTLCache:
    def __init__(self, ttl_seconds=60, max_size=500):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, key):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['ts'] < self.ttl:
                self._cache.move_to_end(key)
                return entry['data']
            else:
                del self._cache[key]
        return None

    def set(self, key, data):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = {'data': data, 'ts': time.time()}
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    @property
    def size(self):
        return len(self._cache)


analysis_cache = TTLCache(ttl_seconds=45, max_size=300)
_ai_response_cache: dict = {}   # key -> (timestamp, response_dict) — 5 min TTL

# ── Concurrency limiter for analyze endpoints ──
# Prevents bulk scans (18+ parallel requests) from overwhelming the thread pool
_analyze_semaphore = asyncio.Semaphore(6)  # max 6 concurrent analyses


# ── safe_timeout helper ──
async def _safe_timeout(coro, *, timeout: float, label: str = "task"):
    shielded = asyncio.shield(coro)
    try:
        return await asyncio.wait_for(shielded, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[%s] timed out after %.0fs", label, timeout)
        raise


# ── Lazy singletons ──
_finnhub_scanner = None
_chart_system = None
_anthropic_client = None


def _get_finnhub_scanner():
    """Lazy-load FinnhubScanner and keep one instance."""
    global _finnhub_scanner
    if _finnhub_scanner is None:
        from finnhub_scanner_v2 import FinnhubScanner
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        polygon_key = os.environ.get("POLYGON_API_KEY", "")
        if not api_key and not polygon_key:
            raise HTTPException(status_code=400, detail="No API keys set.")
        if not api_key:
            api_key = "dummy_key_polygon_only"
        _finnhub_scanner = FinnhubScanner(api_key)
        if polygon_key and hasattr(_finnhub_scanner, 'set_polygon_key'):
            _finnhub_scanner.set_polygon_key(polygon_key)
    return _finnhub_scanner


def _get_chart_system():
    global _chart_system
    if _chart_system is None:
        try:
            from chart_input_analyzer import ChartInputSystem
            _chart_system = ChartInputSystem(data_dir="./scanner_data")
        except Exception:
            pass
    return _chart_system


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if key:
                _anthropic_client = anthropic.Anthropic(api_key=key, timeout=25.0)
        except Exception:
            pass
    return _anthropic_client


def _safe_dict(obj):
    """Convert a dataclass (or dict) to JSON-safe dict, handling numpy types."""
    import numpy as np
    if obj is None:
        return None
    d = asdict(obj) if hasattr(obj, '__dataclass_fields__') else dict(obj)

    def _convert(v):
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_convert(item) for item in v]
        return v
    return {k: _convert(v) for k, v in d.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ANALYSIS (Polygon-powered)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/analyze/live/{symbol}")
async def analyze_live(
    symbol: str,
    timeframe: str = Query("1HR", description="30MIN, 1HR, 2HR, 4HR, DAILY"),
    with_ai: bool = Query(True, description="Include AI commentary"),
    use_rules: bool = Query(False, description="Use rule-based analysis instead of AI"),
    entry_signal: str = Query(None, description="Entry signal e.g. 'failed_breakout:short'"),
    vp_period: str = Query("swing", description="VP lookback: 'day','swing','position','investment'")
):
    """Analyze symbol with live Polygon data"""
    try:
        symbol = symbol.upper().strip()
        cache_key = f"{symbol}:{timeframe.upper()}:{vp_period}"
        cached = analysis_cache.get(cache_key)
        if cached and not with_ai:
            cached['_cached'] = True
            return cached

        if _analyze_semaphore.locked():
            # All 6 slots busy — wait instead of rejecting
            pass
        async with _analyze_semaphore:
            response = await _safe_timeout(
                asyncio.to_thread(_analyze_live_sync, symbol, timeframe, with_ai, use_rules, entry_signal, vp_period),
                timeout=45, label="analyze-live"
            )

        if not with_ai:
            analysis_cache.set(cache_key, response)

        return _safe_dict(response)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error("analyze_live error for %s: %s", symbol, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


def _analyze_live_sync(symbol, timeframe, with_ai, use_rules, entry_signal, vp_period):
    """All sync work for analyze_live — runs in thread pool, NOT on event loop."""
    from polygon_data import get_bars
    from finnhub_scanner_v2 import TechnicalCalculator

    scanner = None
    use_yfinance = False
    try:
        scanner = _get_finnhub_scanner()
    except Exception:
        use_yfinance = True

    chart_system = _get_chart_system()
    anthropic_client = _get_anthropic_client()

    # Resolution mapping
    resolution_map = {
        "5MIN": "5", "15MIN": "15", "30MIN": "30",
        "1HR": "60", "2HR": "60", "4HR": "60", "DAILY": "D"
    }
    resolution = resolution_map.get(timeframe.upper(), "60")

    # VP_BARS based on vp_period
    swing_bars = {"5MIN": 200, "15MIN": 80, "30MIN": 50, "1HR": 35, "2HR": 20, "4HR": 12, "DAILY": 20}
    period_multipliers = {"day": 0.2, "swing": 1.0, "position": 4.0, "investment": 12.0}
    base_bars = swing_bars.get(timeframe.upper(), 35)
    multiplier = period_multipliers.get(vp_period.lower(), 1.0)
    VP_BARS = int(base_bars * multiplier)
    if vp_period.lower() == "investment" and timeframe.upper() == "DAILY":
        VP_BARS = 200

    # Days of data to fetch
    days_base = {"5MIN": 3, "15MIN": 5, "30MIN": 7, "1HR": 10, "2HR": 20, "4HR": 40, "DAILY": 60}
    days_multiplier = {"day": 1, "swing": 1, "position": 3, "investment": 5}
    days_back = days_base.get(timeframe.upper(), 10) * days_multiplier.get(vp_period.lower(), 1)
    days_back = min(days_back, 365)

    # Get candle data
    df = None
    if scanner:
        df = scanner._get_candles(symbol, resolution, days_back)

    if df is None or len(df) < 10:
        use_yfinance = True
        if get_bars:
            yf_interval_map = {"5MIN": "5m", "15MIN": "15m", "30MIN": "30m", "1HR": "1h", "2HR": "1h", "4HR": "1h", "DAILY": "1d"}
            yf_period_map = {"5MIN": "5d", "15MIN": "1mo", "30MIN": "1mo", "1HR": "1mo", "2HR": "3mo", "4HR": "3mo", "DAILY": "1y"}
            yf_interval = yf_interval_map.get(timeframe.upper(), "1h")
            yf_period = yf_period_map.get(timeframe.upper(), "1mo")
            if vp_period.lower() in ("position", "investment"):
                if timeframe.upper() == "DAILY":
                    yf_period = "1y"
                elif timeframe.upper() in ("1HR", "2HR", "4HR"):
                    yf_period = "3mo"
            df = get_bars(symbol, period=yf_period, interval=yf_interval)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                if df.index.tz is None:
                    df.index = df.index.tz_localize('US/Eastern')
            else:
                df = None
            print(f"Polygon fallback for {symbol}: {len(df) if df is not None else 0} bars ({yf_interval})")

    # Resample for 2HR/4HR
    resample_map = {"2HR": "2h", "4HR": "4h"}
    if df is not None and timeframe.upper() in resample_map:
        if scanner and not use_yfinance:
            df = scanner._resample_to_timeframe(df, timeframe)
        else:
            df = df.resample(resample_map[timeframe.upper()]).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()

    # Trim to VP_BARS
    if df is not None and len(df) > VP_BARS:
        df = df.tail(VP_BARS)

    print(f"VP Calculation using {len(df) if df is not None else 0} bars for {symbol} {timeframe}")

    # Calculate VP levels
    _atr = 0
    if df is not None and len(df) >= 10:
        calc = scanner.calc if scanner else (TechnicalCalculator() if TechnicalCalculator else None)
        if calc:
            poc, vah, val = calc.calculate_volume_profile(df)
            vwap = calc.calculate_vwap(df)
            rsi = calc.calculate_rsi(df)
            _atr = calc.calculate_atr(df)
        else:
            poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50
    else:
        poc, vah, val, vwap, rsi = 0, 0, 0, 0, 50

    # Get real-time quote
    quote = None
    day_open = day_high = day_low = prev_close = 0
    current_price = 0
    quote_source = 'none'
    if scanner:
        quote = scanner.get_quote(symbol)
    if quote and quote.get('current'):
        current_price = float(quote['current'])
        quote_source = quote.get('source', 'unknown')
        day_open = float(quote.get('open', 0) or 0)
        day_high = float(quote.get('high', 0) or 0)
        day_low = float(quote.get('low', 0) or 0)
        prev_close = float(quote.get('prev_close', 0) or 0)
    elif df is not None and len(df) > 0:
        current_price = float(df['close'].iloc[-1])
        quote_source = 'yfinance' if use_yfinance else 'candle_fallback'
        day_open = float(df['open'].iloc[-1]) if 'open' in df.columns else 0
        day_high = float(df['high'].iloc[-1]) if 'high' in df.columns else 0
        day_low = float(df['low'].iloc[-1]) if 'low' in df.columns else 0
        prev_close = float(df['close'].iloc[-2]) if len(df) > 1 else 0

    # Run analysis
    result = None
    if scanner:
        result = scanner.analyze(symbol, timeframe)
    if result is None and df is not None and len(df) >= 10 and chart_system:
        calc = scanner.calc if scanner else (TechnicalCalculator() if TechnicalCalculator else None)
        if calc:
            _rvol = calc.calculate_relative_volume(df)
            _vol_trend = calc.calculate_volume_trend(df)
            _vol_div = calc.detect_volume_divergence(df)
            _has_rejection = False
            if current_price and current_price < val and val > 0:
                _has_rejection = calc.is_rejection_candle(df, "bullish")
            elif current_price and current_price > vah and vah > 0:
                _has_rejection = calc.is_rejection_candle(df, "bearish")
            result = chart_system.analyze(
                symbol=symbol, price=current_price,
                vah=vah, poc=poc, val=val, vwap=vwap, rsi=rsi,
                timeframe=timeframe, rvol=_rvol,
                volume_trend=_vol_trend, volume_divergence=_vol_div,
                atr=_atr, has_rejection=_has_rejection
            )

    if not result:
        raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")

    # Get order flow
    order_flow = None
    try:
        if scanner:
            of_result = scanner.get_order_flow_analysis(symbol, timeframe, vp_period)
            if of_result and hasattr(of_result, 'to_dict'):
                order_flow = of_result.to_dict()
            elif of_result and isinstance(of_result, dict):
                order_flow = of_result
    except Exception as e:
        print(f"Order flow error: {e}")

    response = {
        "symbol": symbol,
        "timeframe": str(result.timeframe),
        "signal": str(result.signal),
        "signal_emoji": str(result.signal_emoji),
        "bull_score": float(result.bull_score),
        "bear_score": float(result.bear_score),
        "confidence": float(result.confidence),
        "high_prob": float(result.high_prob),
        "low_prob": float(result.low_prob),
        "position": str(result.position),
        "vwap_zone": str(result.vwap_zone),
        "rsi_zone": str(result.rsi_zone),
        "notes": [str(n) for n in result.notes] if result.notes else [],
        "timestamp": datetime.now().isoformat(),
        "current_price": float(current_price) if current_price else 0,
        "day_open": float(day_open),
        "day_high": float(day_high),
        "day_low": float(day_low),
        "prev_close": float(prev_close),
        "quote_source": quote_source,
        "vah": float(vah) if vah else 0,
        "poc": float(poc) if poc else 0,
        "val": float(val) if val else 0,
        "vwap": float(vwap) if vwap else 0,
        "rsi": float(rsi) if rsi else 50,
        "rvol": float(getattr(result, 'rvol', 1.0)),
        "volume_trend": str(getattr(result, 'volume_trend', 'neutral')),
        "volume_divergence": bool(getattr(result, 'volume_divergence', False)),
        "signal_type": str(getattr(result, 'signal_type', 'none')),
        "signal_strength": str(getattr(result, 'signal_strength', 'moderate')),
        "atr": float(getattr(result, 'atr', 0)),
        "extension_atr": float(getattr(result, 'extension_atr', 0)),
        "has_rejection": bool(getattr(result, 'has_rejection', False)),
        "order_flow": order_flow
    }

    # Fibonacci retracement levels
    try:
        fib_period_days = {"day": 5, "swing": 20, "position": 60, "investment": 120}
        fib_days = fib_period_days.get(vp_period.lower(), 20)
        df_fib = None
        if scanner:
            df_fib = scanner._get_candles(symbol, "D", fib_days)
        if df_fib is None or len(df_fib) < 5:
            if get_bars:
                fib_period_yf = "6mo" if fib_days <= 120 else "1y"
                df_fib = get_bars(symbol, period=fib_period_yf, interval="1d")
                if df_fib is not None and not df_fib.empty:
                    df_fib.columns = [c.lower() for c in df_fib.columns]
                else:
                    df_fib = None
        if df_fib is not None and len(df_fib) >= 5:
            df_fib = df_fib.tail(fib_days)
            cp = float(df_fib['close'].iloc[-1])
            valid_lows = df_fib['low'][df_fib['low'] > cp * 0.5]
            valid_highs = df_fib['high'][df_fib['high'] < cp * 2.0]
            swing_high = float(valid_highs.max()) if len(valid_highs) > 0 else float(df_fib['high'].iloc[-5:].max())
            swing_low = float(valid_lows.min()) if len(valid_lows) > 0 else float(df_fib['low'].iloc[-5:].min())
            fib_range = swing_high - swing_low

            if fib_range > swing_high * 0.20:
                swing_high = float(df_fib['high'].quantile(0.95))
                swing_low = float(df_fib['low'].quantile(0.05))
                fib_range = swing_high - swing_low

            high_idx = df_fib['high'].idxmax()
            low_idx = df_fib['low'].idxmin()
            high_pos = list(df_fib.index).index(high_idx) if high_idx in df_fib.index else len(df_fib) - 1
            low_pos = list(df_fib.index).index(low_idx) if low_idx in df_fib.index else 0
            uptrend = high_pos > low_pos

            if fib_range < swing_high * 0.01:
                fib_range = swing_high * 0.05

            bull_fib_236 = swing_low + (fib_range * 0.236)
            bull_fib_382 = swing_low + (fib_range * 0.382)
            bull_fib_500 = swing_low + (fib_range * 0.500)
            bull_fib_618 = swing_low + (fib_range * 0.618)
            bull_fib_786 = swing_low + (fib_range * 0.786)
            bear_fib_236 = swing_high - (fib_range * 0.236)
            bear_fib_382 = swing_high - (fib_range * 0.382)
            bear_fib_500 = swing_high - (fib_range * 0.500)
            bear_fib_618 = swing_high - (fib_range * 0.618)
            bear_fib_786 = swing_high - (fib_range * 0.786)

            response["fib_levels"] = {
                "swing_high": swing_high, "swing_low": swing_low,
                "lookback_days": fib_days,
                "trend": "UPTREND" if uptrend else "DOWNTREND",
                "bull_fib_236": bull_fib_236, "bull_fib_382": bull_fib_382,
                "bull_fib_500": bull_fib_500, "bull_fib_618": bull_fib_618,
                "bull_fib_786": bull_fib_786,
                "bear_fib_236": bear_fib_236, "bear_fib_382": bear_fib_382,
                "bear_fib_500": bear_fib_500, "bear_fib_618": bear_fib_618,
                "bear_fib_786": bear_fib_786,
                "fib_236": bear_fib_236, "fib_382": bear_fib_382,
                "fib_500": bear_fib_500, "fib_618": bear_fib_618,
                "fib_786": bear_fib_786
            }

            # Fib position
            if uptrend:
                if cp >= swing_high: response["fib_position"] = "Above swing high (extended)"
                elif cp >= bull_fib_786: response["fib_position"] = "Above Bull 78.6% (strong uptrend)"
                elif cp >= bull_fib_618: response["fib_position"] = "Above Bull 61.8% (healthy uptrend)"
                elif cp >= bull_fib_500: response["fib_position"] = "Bull 50%-61.8% GOLDEN ZONE (best long entry)"
                elif cp >= bull_fib_382: response["fib_position"] = "Bull 38.2%-50% (pullback entry zone)"
                elif cp >= bull_fib_236: response["fib_position"] = "Bull 23.6%-38.2% (shallow pullback)"
                else: response["fib_position"] = "Below Bull 23.6% (trend may be broken)"
            else:
                if cp <= swing_low: response["fib_position"] = "Below swing low (extended)"
                elif cp <= bear_fib_786: response["fib_position"] = "Below Bear 78.6% (strong downtrend)"
                elif cp <= bear_fib_618: response["fib_position"] = "Below Bear 61.8% (healthy downtrend)"
                elif cp <= bear_fib_500: response["fib_position"] = "Bear 50%-61.8% GOLDEN ZONE (best short entry)"
                elif cp <= bear_fib_382: response["fib_position"] = "Bear 38.2%-50% (bounce entry zone)"
                elif cp <= bear_fib_236: response["fib_position"] = "Bear 23.6%-38.2% (shallow bounce)"
                else: response["fib_position"] = "Above Bear 23.6% (trend may be reversing)"

            # Fib score adjustment
            fib_pos = response.get("fib_position", "")
            fib_bull_adj = fib_bear_adj = 0
            signal_str = str(response.get("signal", "")).upper()
            is_long_signal = "LONG" in signal_str
            is_short_signal = "SHORT" in signal_str
            is_counter_trend = (not uptrend and is_long_signal) or (uptrend and is_short_signal)

            adj_map = {
                "GOLDEN ZONE": 18 if is_counter_trend else 15,
                "38.2%-50%": 10,
                "23.6%-38.2%": 5,
                "78.6%": 3 if is_counter_trend else 5,
            }
            for label, adj in adj_map.items():
                if label in fib_pos and "GOLDEN" not in fib_pos.replace(label, ""):
                    if is_counter_trend:
                        if is_long_signal: fib_bull_adj = adj
                        else: fib_bear_adj = adj
                    elif uptrend: fib_bull_adj = adj
                    else: fib_bear_adj = adj
                    break
            if "61.8%" in fib_pos and "GOLDEN" not in fib_pos:
                adj = 8
                if is_counter_trend:
                    if is_long_signal: fib_bull_adj = adj
                    else: fib_bear_adj = adj
                elif uptrend: fib_bull_adj = adj
                else: fib_bear_adj = adj

            if fib_bull_adj or fib_bear_adj:
                response["bull_score"] = max(0, min(100, response["bull_score"] + fib_bull_adj))
                response["bear_score"] = max(0, min(100, response["bear_score"] + fib_bear_adj))
                response["fib_score_adj"] = {"bull": fib_bull_adj, "bear": fib_bear_adj}
                new_bull, new_bear = response["bull_score"], response["bear_score"]
                gap = abs(new_bull - new_bear)
                winning = max(new_bull, new_bear)
                response["confidence"] = min(95, 40 + (winning * 0.5) + (gap * 0.1))
                total = new_bull + new_bear
                if total > 0:
                    response["high_prob"] = round(max(new_bull, new_bear) / total * 100, 1)
                    response["low_prob"] = round(min(new_bull, new_bear) / total * 100, 1)
                response["notes"].append(f"Fib adj: bull {fib_bull_adj:+d}, bear {fib_bear_adj:+d} ({fib_pos})")

            # VP + Fib confluence
            active_fibs = {
                "23.6%": bull_fib_236 if uptrend else bear_fib_236,
                "38.2%": bull_fib_382 if uptrend else bear_fib_382,
                "50%": bull_fib_500 if uptrend else bear_fib_500,
                "61.8%": bull_fib_618 if uptrend else bear_fib_618,
                "78.6%": bull_fib_786 if uptrend else bear_fib_786,
            }
            confluences = []
            for fn, fv in active_fibs.items():
                if vah > 0 and abs(vah - fv) / vah < 0.015:
                    confluences.append(f"VAH ~ Fib {fn} at ${vah:.2f}")
                if poc > 0 and abs(poc - fv) / poc < 0.015:
                    confluences.append(f"POC ~ Fib {fn} at ${poc:.2f}")
                if val > 0 and abs(val - fv) / val < 0.015:
                    confluences.append(f"VAL ~ Fib {fn} at ${val:.2f}")
            if confluences:
                response["fib_confluence"] = confluences
                response["notes"].append(f"Fib Confluence: {'; '.join(confluences)}")

            # Trade scenarios
            if current_price > 0 and vah > 0 and val > 0 and poc > 0:
                vp_range = vah - val if vah > val else 1
                scen_atr = _atr if _atr > 0 else vp_range * 0.3
                long_entry_low, long_entry_high = val, poc
                long_mid = (long_entry_low + long_entry_high) / 2
                long_stop = long_mid - (scen_atr * 0.5)
                long_target1 = max(vah, long_mid + scen_atr)
                long_target2 = max(bull_fib_786 if bull_fib_786 > long_target1 else swing_high, long_mid + scen_atr * 2)
                long_risk = long_mid - long_stop
                long_reward = long_target1 - long_mid
                long_rr = long_reward / long_risk if long_risk > 0 else 0

                short_entry_low, short_entry_high = poc, vah
                short_mid = (short_entry_low + short_entry_high) / 2
                short_stop = short_mid + (scen_atr * 0.5)
                short_target1 = min(val, short_mid - scen_atr)
                short_target2 = min(bear_fib_618 if bear_fib_618 < short_target1 else swing_low, short_mid - scen_atr * 2)
                short_risk = short_stop - short_mid
                short_reward = short_mid - short_target1
                short_rr = short_reward / short_risk if short_risk > 0 else 0

                agg_long_stop = current_price - (scen_atr * 0.5)
                agg_short_stop = current_price + (scen_atr * 0.5)
                long_agg_valid = current_price > long_stop
                short_agg_valid = current_price < short_stop
                agg_long_rr = (long_target1 - current_price) / (current_price - agg_long_stop) if current_price > agg_long_stop else 0
                agg_short_rr = (current_price - short_target1) / (agg_short_stop - current_price) if agg_short_stop > current_price else 0
                agg_risk_pct = (scen_atr * 0.5 / current_price) * 100 if current_price > 0 else 0

                response["trade_scenarios"] = {
                    "long": {
                        "entry_zone": [f"{long_entry_low:.2f}", f"{long_entry_high:.2f}"],
                        "stop_loss": long_stop, "target": long_target1, "target2": long_target2,
                        "r_r_ratio": f"{long_rr:.1f}:1",
                        "aggressive_entry": current_price if long_agg_valid else None,
                        "aggressive_stop": agg_long_stop, "aggressive_valid": long_agg_valid,
                        "aggressive_rr": f"{agg_long_rr:.1f}:1" if long_agg_valid else None,
                        "aggressive_risk_pct": round(agg_risk_pct, 1) if long_agg_valid else None
                    },
                    "short": {
                        "entry_zone": [f"{short_entry_low:.2f}", f"{short_entry_high:.2f}"],
                        "stop_loss": short_stop, "target": short_target1, "target2": short_target2,
                        "r_r_ratio": f"{short_rr:.1f}:1",
                        "aggressive_entry": current_price if short_agg_valid else None,
                        "aggressive_stop": agg_short_stop, "aggressive_valid": short_agg_valid,
                        "aggressive_rr": f"{agg_short_rr:.1f}:1" if short_agg_valid else None,
                        "aggressive_risk_pct": round(agg_risk_pct, 1) if short_agg_valid else None
                    },
                    "decision_point": {
                        "bull_trigger": vah + (vp_range * 0.02),
                        "bear_trigger": val - (vp_range * 0.02),
                        "current_price": current_price
                    }
                }
    except Exception as e:
        print(f"Fib/Trade scenarios error: {e}")

    if entry_signal:
        response["entry_signal"] = entry_signal

    # AI commentary
    if with_ai and anthropic_client:
        try:
            response["ai_commentary"] = _get_ai_commentary(response, symbol, anthropic_client)
            response["analysis_source"] = "claude"
        except Exception as e:
            print(f"AI commentary error: {e}")
            response["analysis_source"] = "none"

    return response


def _get_ai_commentary(analysis: dict, symbol: str, anthropic_client) -> str:
    """Generate AI commentary using Claude"""
    if not anthropic_client:
        return ""
    try:
        prompt = f"""Analyze this stock data for {symbol} and give a concise 2-3 sentence trading outlook:
Signal: {analysis.get('signal')} | Bull: {analysis.get('bull_score')}/100 | Bear: {analysis.get('bear_score')}/100
Price: ${analysis.get('current_price', 0):.2f} | VAH: ${analysis.get('vah', 0):.2f} | POC: ${analysis.get('poc', 0):.2f} | VAL: ${analysis.get('val', 0):.2f}
RSI: {analysis.get('rsi', 50):.1f} | VWAP Zone: {analysis.get('vwap_zone')} | Position: {analysis.get('position')}
Fib Position: {analysis.get('fib_position', 'N/A')}
Notes: {'; '.join(analysis.get('notes', [])[:5])}
Be specific about price levels and actionable."""
        msg = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        print(f"Claude AI error: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# MTF (Multi-Timeframe) ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/analyze/live/mtf/{symbol}")
async def analyze_live_mtf(symbol: str):
    """Multi-timeframe analysis with live data"""
    try:
        def _mtf_sync():
            scanner = _get_finnhub_scanner()
            result = scanner.analyze_mtf(symbol.upper())
            if not result:
                return None

            current_price = None
            try:
                quote = scanner.get_quote(symbol.upper())
                if quote and quote.get('current'):
                    current_price = float(quote['current'])
            except Exception:
                pass

            tf_results = {}
            for tf, r in result.timeframe_results.items():
                tf_results[tf] = {
                    "signal": r.signal,
                    "signal_emoji": r.signal_emoji,
                    "bull_score": r.bull_score,
                    "bear_score": r.bear_score,
                    "confidence": r.confidence,
                    "position": r.position,
                    "rsi_zone": r.rsi_zone
                }

            return {
                "symbol": result.symbol,
                "current_price": current_price,
                "timestamp": str(result.timestamp) if result.timestamp else None,
                "dominant_signal": result.dominant_signal,
                "signal_emoji": result.signal_emoji,
                "confluence_pct": float(result.confluence_pct) if result.confluence_pct else 0,
                "weighted_bull": float(result.weighted_bull) if result.weighted_bull else 0,
                "weighted_bear": float(result.weighted_bear) if result.weighted_bear else 0,
                "high_prob": float(result.high_prob) if result.high_prob else 0,
                "low_prob": float(result.low_prob) if result.low_prob else 0,
                "timeframes": tf_results,
                "notes": list(result.notes) if result.notes else []
            }

        data = await _safe_timeout(asyncio.to_thread(_mtf_sync), timeout=30, label="mtf")
        if data is None:
            raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")
        return data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"MTF analysis timed out for {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error("MTF analysis error for %s: %s", symbol, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"MTF analysis error: {str(e)}")


@router.post("/api/analyze/live/mtf/{symbol}/ai")
async def analyze_mtf_with_ai(
    symbol: str,
    trade_tf: str = Query("swing", description="Trade timeframe: intraday, swing, position, longterm, investment"),
    entry_signal: str = Query(None, description="Entry signal e.g. 'failed_breakout:short'")
):
    """Generate AI trade plan using full MTF context"""
    anthropic_client = _get_anthropic_client()
    if not anthropic_client:
        raise HTTPException(status_code=400, detail="AI API key not set")

    # ── AI response cache (5 min TTL) ──
    _cache_key = f"{symbol.upper()}:{trade_tf}:{entry_signal or ''}"
    _cached = _ai_response_cache.get(_cache_key)
    if _cached and (time.time() - _cached[0]) < 300:
        return _cached[1]

    def _gather_mtf_data():
        scanner = _get_finnhub_scanner()
        result = scanner.analyze_mtf(symbol.upper())
        if not result:
            return None, None, None

        tf_config = {
            "scalp":      {"days": 1, "label": "SCALP (15 min – 2 hrs)",  "stop_mult": 0.15, "target_mult": 0.3,  "hold": "15 min – 2 hours", "candle_res": "5",  "candle_bars": 50},
            "intraday":   {"days": 1, "label": "SAME DAY (Intraday)",     "stop_mult": 0.3,  "target_mult": 0.5,  "hold": "1-4 hours",        "candle_res": "15", "candle_bars": 40},
            "daytrade":   {"days": 2, "label": "1-2 DAY TRADE",           "stop_mult": 0.4,  "target_mult": 0.7,  "hold": "1-2 days",         "candle_res": "30", "candle_bars": 30},
            "swing":      {"days": 5, "label": "3-5 DAY SWING",           "stop_mult": 0.5,  "target_mult": 1.0,  "hold": "3-5 days",         "candle_res": "60", "candle_bars": 20},
            "position":   {"days": 21, "label": "2-4 WEEK POSITION",      "stop_mult": 1.0,  "target_mult": 2.0,  "hold": "2-4 weeks",        "candle_res": "D",  "candle_bars": 30},
            "longterm":   {"days": 60, "label": "1-3 MONTH SETUP",        "stop_mult": 2.0,  "target_mult": 4.0,  "hold": "1-3 months",       "candle_res": "D",  "candle_bars": 60},
            "investment": {"days": 180, "label": "6+ MONTH INVESTMENT",    "stop_mult": 5.0,  "target_mult": 10.0, "hold": "6+ months",        "candle_res": "D",  "candle_bars": 120}
        }
        config = tf_config.get(trade_tf, tf_config["swing"])

        candle_res = config.get("candle_res", "60")
        candle_bars = config.get("candle_bars", 20)
        df = scanner._get_candles(symbol.upper(), candle_res, candle_bars)
        poc, vah, val, vwap, rsi, rvol, volume_trend, current_price = 0, 0, 0, 0, 50, 1.0, "neutral", 0
        if df is not None and len(df) >= 5:
            poc, vah, val = scanner.calc.calculate_volume_profile(df)
            vwap = scanner.calc.calculate_vwap(df)
            rsi = scanner.calc.calculate_rsi(df)
            rvol = scanner.calc.calculate_relative_volume(df)
            volume_trend = scanner.calc.calculate_volume_trend(df)
            current_price = float(df['close'].iloc[-1])
            quote = scanner.get_quote(symbol.upper())
            if quote and quote.get('current'):
                current_price = float(quote['current'])

        fib_text = ""
        fib_days = min(config["days"] * 3, 60) if config["days"] <= 5 else 15
        fib_days = max(fib_days, 5)
        try:
            df_daily = scanner._get_candles(symbol.upper(), "D", fib_days)
            if df_daily is not None and len(df_daily) >= 5:
                swing_high = float(df_daily['high'].max())
                swing_low = float(df_daily['low'].min())
                fib_range = swing_high - swing_low
                fib_236 = swing_high - (fib_range * 0.236)
                fib_382 = swing_high - (fib_range * 0.382)
                fib_500 = swing_high - (fib_range * 0.500)
                fib_618 = swing_high - (fib_range * 0.618)
                fib_786 = swing_high - (fib_range * 0.786)
                fib_text = (
                    f"Fib ({fib_days}d): High ${swing_high:.2f} Low ${swing_low:.2f} | "
                    f"23.6% ${fib_236:.2f} | 38.2% ${fib_382:.2f} | 50% ${fib_500:.2f} | "
                    f"61.8% ${fib_618:.2f} | 78.6% ${fib_786:.2f}"
                )
        except Exception:
            pass

        atr_daily = 0
        try:
            if df is not None and len(df) >= 14:
                _high = df['high']
                _low = df['low']
                _prev = df['close'].shift(1)
                import pandas as _pd
                _tr = _pd.concat([_high - _low, (_high - _prev).abs(), (_low - _prev).abs()], axis=1).max(axis=1)
                atr_daily = float(_tr.rolling(14).mean().iloc[-1])
        except Exception:
            pass
        if atr_daily <= 0:
            atr_daily = (vah - val) * 0.3 if vah > val else current_price * 0.015 if current_price > 0 else 1

        return result, config, {
            "poc": poc, "vah": vah, "val": val, "vwap": vwap, "rsi": rsi,
            "rvol": rvol, "volume_trend": volume_trend, "current_price": current_price,
            "fib_text": fib_text, "atr_daily": atr_daily,
            "candle_res": candle_res, "candle_bars": candle_bars,
        }

    result, config, ctx = await _safe_timeout(asyncio.to_thread(_gather_mtf_data), timeout=30, label="mtf-ai")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Could not analyze {symbol}")

    try:
        poc, vah, val = ctx["poc"], ctx["vah"], ctx["val"]
        vwap, rsi, rvol = ctx["vwap"], ctx["rsi"], ctx["rvol"]
        volume_trend = ctx["volume_trend"]
        current_price = ctx["current_price"]
        fib_text = ctx["fib_text"]
        atr_daily = ctx["atr_daily"]
        candle_res = ctx["candle_res"]
        candle_bars = ctx["candle_bars"]
        stop_distance = atr_daily * config["stop_mult"]
        target_distance = atr_daily * config["target_mult"]

        bull_total = result.weighted_bull or 0
        bear_total = result.weighted_bear or 0
        score_diff = bull_total - bear_total
        if entry_signal:
            parts = entry_signal.split(':')
            leading_direction = parts[1].upper() if len(parts) > 1 else ("LONG" if score_diff >= 0 else "SHORT")
            leading_reason = f"VP Entry Signal: {parts[0].replace('_', ' ').title()}" if parts else "Entry signal"
        elif score_diff > 10:
            leading_direction = "LONG"
            leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f}"
        elif score_diff < -10:
            leading_direction = "SHORT"
            leading_reason = f"Bull/Bear Score: {bull_total:.0f} vs {bear_total:.0f}"
        elif current_price > poc and poc > 0:
            leading_direction = "SHORT"
            leading_reason = f"Price above POC (${poc:.2f})"
        elif poc > 0:
            leading_direction = "LONG"
            leading_reason = f"Price below POC (${poc:.2f})"
        else:
            leading_direction = "LONG" if "LONG" in str(result.dominant_signal) else "SHORT"
            leading_reason = f"MTF Dominant: {result.dominant_signal}"

        tf_summary = []
        for tf, r in result.timeframe_results.items():
            tf_summary.append(f"{tf}: {r.signal} (Bull:{r.bull_score}, Bear:{r.bear_score})")

        prompt = f"""ANALYZE MTF: {symbol.upper()} @ ${current_price:.2f} | {config["label"]}
VP RESOLUTION: {candle_res} candles ({candle_bars} bars) — levels reflect THIS timeframe

LEADING DIRECTION: {leading_direction} ({leading_reason})
MTF CONFLUENCE: {result.confluence_pct}% | Dominant: {result.dominant_signal}
HIGH PROB: {result.high_prob:.0f}% | LOW PROB: {result.low_prob:.0f}%
Bull: {result.weighted_bull:.0f} | Bear: {result.weighted_bear:.0f}

VOLUME: RVOL {rvol:.1f}x | Trend: {volume_trend}
TIMEFRAMES: {' | '.join(tf_summary)}

LEVELS: VAH ${vah:.2f} | POC ${poc:.2f} | VAL ${val:.2f} | VWAP ${vwap:.2f} | RSI {rsi:.0f}
{fib_text}

ATR: ${atr_daily:.2f} | Stop: ${stop_distance:.2f} ({config['stop_mult']}x ATR) | Target: ${target_distance:.2f} ({config['target_mult']}x ATR)
Hold: {config['hold']}
IMPORTANT: Size entries, stops, and targets for a {config['label']} trade — NOT a swing/position trade.

NOTES: {'; '.join(result.notes[:3]) if result.notes else 'None'}

Give BOTH long and short setups with entry zones, stops, targets, R:R math.
Lead with {leading_direction}. End with a VERDICT picking the preferred direction."""

        system_prompt = f"""You are an expert MTF trading analyst planning a {config['label']} trade.
Output FULL SETUPS for BOTH directions using this EXACT format (including emojis):

🟢 LONG SETUP
GRADE: [A+ to F]
CONVICTION: [1-10]/10
PROBABILITY: [X]% [LABEL]
ENTRY ZONE: $[low] – $[high]
STOP: $[price]
T1: $[price]
T2: $[price]
R:R = [X:X]
EV: $[X] per $100 risked → [POSITIVE/NEGATIVE]
TRIGGER: [what confirms entry]
INVALID IF: [what kills the trade]
WHY LONG: [1-2 sentence reason]
SIZE: [X]R
HOLD: [duration]

🔴 SHORT SETUP
GRADE: [A+ to F]
CONVICTION: [1-10]/10
PROBABILITY: [X]% [LABEL]
ENTRY ZONE: $[low] – $[high]
STOP: $[price]
T1: $[price]
T2: $[price]
R:R = [X:X]
EV: $[X] per $100 risked → [POSITIVE/NEGATIVE]
TRIGGER: [what confirms entry]
INVALID IF: [what kills the trade]
WHY SHORT: [1-2 sentence reason]
SIZE: [X]R
HOLD: [duration]

⚖️ VERDICT
PREFERRED: [LONG or SHORT]
KEY LEVEL: $[price]
[1-2 sentence summary]

Use VP levels and Fib levels for entries/stops/targets.
CRITICAL: All entries, stops, and targets MUST be sized for a {config['hold']} hold period.
For scalps/intraday: tighter entries, closer stops, smaller targets.
For position/longterm: wider entries, wider stops, larger targets.
Follow the format EXACTLY — no extra sections, no missing fields."""

        msg = await _safe_timeout(
            asyncio.to_thread(
                anthropic_client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=30, label="AI-MTF"
        )
        _response = {
            "symbol": symbol.upper(),
            "ai_commentary": msg.content[0].text,
            "high_prob": result.high_prob,
            "low_prob": result.low_prob,
            "confluence": result.confluence_pct,
            "dominant_signal": result.dominant_signal,
            "trade_timeframe": config["label"],
            "leading_direction": leading_direction,
            "leading_reason": leading_reason,
            "bull_score": result.weighted_bull,
            "bear_score": result.weighted_bear,
            "rvol": rvol,
            "volume_trend": volume_trend,
            "vah": vah, "poc": poc, "val": val, "vwap": vwap, "rsi": rsi,
            "current_price": current_price
        }
        _ai_response_cache[_cache_key] = (time.time(), _response)
        if len(_ai_response_cache) > 50:
            oldest_key = min(_ai_response_cache, key=lambda k: _ai_response_cache[k][0])
            _ai_response_cache.pop(oldest_key, None)
        return _response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI analysis timed out (30s). Try again.")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
