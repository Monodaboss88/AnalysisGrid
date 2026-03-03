"""
Scanner Router — All heavy scanner endpoints extracted from unified_server.py.
==============================================================================
Every scanner module is LAZY-LOADED on first use (not at boot).
This keeps the main server lightweight and prevents one scanner crash
from taking down the entire API.

Endpoints:
  GET /api/buffett-scan      — Buffett Blood Scanner
  GET /api/war-room           — War Room (intraday extension DNA)
  GET /api/regime-scan        — Regime Scanner (cross-gate strategy)
  GET /api/regime-levels/{s}  — ATR-scaled strategy levels
  GET /api/scan/live          — Live watchlist scan
  GET /api/options-flow       — Options flow unusual activity
  POST /api/options-flow/stream/start
  POST /api/options-flow/stream/stop
  GET  /api/options-flow/stream/status
  GET  /api/options-flow/stream/events
  GET  /api/structure/reversals/{symbol}
"""

import os
import gc
import json
import time
import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Scan concurrency lock — only 1 heavy scan at a time ──
_scan_semaphore = asyncio.Semaphore(1)


# ── Bounded cache helper ──
def _evict_cache(cache: dict, max_size: int):
    """Evict oldest entries from a {key: (timestamp, data)} cache."""
    while len(cache) > max_size:
        oldest_key = min(cache, key=lambda k: cache[k][0])
        del cache[oldest_key]


# ── safe_timeout (no asyncio.shield — let timeouts cancel properly) ──
async def _safe_timeout(coro, *, timeout: float, label: str = "task"):
    """Run a coroutine with a timeout. No shield = no zombie threads."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[%s] timed out after %.0fs", label, timeout)
        raise


# ── Lazy-loaded singletons ──
_regime_scanner = None
_regime_lock = asyncio.Lock()

async def _get_regime_scanner():
    """Lazy-load RegimeScanner on first use."""
    global _regime_scanner
    if _regime_scanner is None:
        async with _regime_lock:
            if _regime_scanner is None:
                from regime_scanner import RegimeScanner
                _regime_scanner = RegimeScanner()
                logger.info("RegimeScanner loaded (lazy)")
    return _regime_scanner


# ═══════════════════════════════════════════════════════════════════════════════
# BUFFETT BLOOD SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

_buffett_cache: dict = {}
_BUFFETT_TTL = 300
_BUFFETT_CACHE_MAX = 20
_buffett_hits = 0


@router.get("/api/buffett-scan")
async def buffett_scan(tickers: str = "", preset: str = ""):
    """Buffett Blood Scanner — scan tickers for value + crisis metrics"""
    global _buffett_hits
    try:
        from buffett_scanner import async_scan_tickers, PRESETS

        if preset and preset in PRESETS:
            symbols = PRESETS[preset]
        elif tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            raise HTTPException(status_code=400, detail="Provide tickers or preset param")

        if len(symbols) > 30:
            raise HTTPException(status_code=400, detail="Max 30 tickers per scan")

        cache_key = ",".join(sorted(symbols))
        now = time.time()
        entry = _buffett_cache.get(cache_key)
        if entry and (now - entry[0]) < _BUFFETT_TTL:
            _buffett_hits += 1
            data = entry[1]
            data["meta"]["cached"] = True
            data["meta"]["cache_age"] = round(now - entry[0], 1)
            data["meta"]["total_hits"] = _buffett_hits
            return data

        if _scan_semaphore.locked():
            raise HTTPException(status_code=429, detail="Another scan is running. Please wait a few seconds.")
        async with _scan_semaphore:
            t0 = time.time()
            data = await _safe_timeout(async_scan_tickers(symbols), timeout=30, label="buffett")
            elapsed = time.time() - t0

        _buffett_cache[cache_key] = (now, data)
        _evict_cache(_buffett_cache, _BUFFETT_CACHE_MAX)
        gc.collect()

        data["meta"]["cached"] = False
        data["meta"]["scan_time"] = round(elapsed, 2)
        data["meta"]["total_hits"] = _buffett_hits
        return data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Blood Scanner timed out (30s). Try fewer tickers.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Buffett scan error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS FLOW SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/options-flow")
async def options_flow_scan(tickers: str = "", preset: str = ""):
    """Options Flow Scanner — unusual options activity via Polygon"""
    try:
        from options_flow_scanner import async_scan_tickers as async_options_scan, PRESETS as OPT_PRESETS

        if preset and preset in OPT_PRESETS:
            symbols = OPT_PRESETS[preset]
        elif tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            raise HTTPException(status_code=400, detail="Provide tickers or preset param")

        if len(symbols) > 12:
            raise HTTPException(status_code=400, detail="Max 12 tickers per scan")

        if _scan_semaphore.locked():
            raise HTTPException(status_code=429, detail="Another scan is running. Please wait a few seconds.")
        async with _scan_semaphore:
            data = await asyncio.wait_for(
                async_options_scan(symbols),
                timeout=60,
            )
        gc.collect()
        return data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Options flow scan timed out (60s). Try fewer tickers.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Options flow error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Options Flow Stream (SSE) ──

_flow_stream = None

def _get_flow_stream():
    global _flow_stream
    if _flow_stream is None:
        try:
            from options_flow_stream import get_flow_stream
            _flow_stream = get_flow_stream()
        except Exception:
            pass
    return _flow_stream


@router.post("/api/options-flow/stream/start")
async def start_options_flow_stream(tickers: str = "", preset: str = ""):
    fs = _get_flow_stream()
    if not fs:
        raise HTTPException(status_code=400, detail="Options flow stream module not available")
    from options_flow_scanner import PRESETS as OPT_PRESETS
    if preset and preset in OPT_PRESETS:
        symbols = OPT_PRESETS[preset]
    elif tickers:
        symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        raise HTTPException(status_code=400, detail="Provide tickers or preset param")
    if len(symbols) > 8:
        raise HTTPException(status_code=400, detail="Max 8 tickers for live stream")
    fs.set_tickers(symbols)
    if not fs.is_running:
        fs.start_background()
    return {"status": "ok", "message": f"Flow stream started for {len(symbols)} tickers",
            "tickers": symbols, "stream": fs.get_status()}


@router.post("/api/options-flow/stream/stop")
async def stop_options_flow_stream():
    fs = _get_flow_stream()
    if not fs:
        raise HTTPException(status_code=400, detail="Flow stream not available")
    fs.stop()
    return {"status": "ok", "message": "Flow stream stopped"}


@router.get("/api/options-flow/stream/status")
async def options_flow_stream_status():
    fs = _get_flow_stream()
    if not fs:
        return {"available": False, "running": False}
    status = fs.get_status()
    status["available"] = True
    return status


@router.get("/api/options-flow/stream/events")
async def options_flow_sse(request: Request):
    fs = _get_flow_stream()
    if not fs:
        raise HTTPException(status_code=400, detail="Flow stream not available")
    from starlette.responses import StreamingResponse
    q = fs.subscribe()

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'status': fs.get_status()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            fs.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════════════
# WAR ROOM
# ═══════════════════════════════════════════════════════════════════════════════

_warroom_cache: dict = {}
_WARROOM_TTL = 120
_WARROOM_CACHE_MAX = 20


@router.get("/api/war-room")
async def war_room_scan(tickers: str = "", preset: str = ""):
    """War Room — Pre-market extension DNA analysis via Polygon intraday bars"""
    try:
        from war_room import async_run_war_room, PRESETS as WR_PRESETS

        if preset and preset in WR_PRESETS:
            symbols = WR_PRESETS[preset]
        elif tickers:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            raise HTTPException(status_code=400, detail="Provide tickers or preset param")

        if len(symbols) > 15:
            raise HTTPException(status_code=400, detail="Max 15 tickers per scan")

        cache_key = ",".join(sorted(symbols))
        now = time.time()
        entry = _warroom_cache.get(cache_key)
        if entry and (now - entry[0]) < _WARROOM_TTL:
            return entry[1]

        if _scan_semaphore.locked():
            raise HTTPException(status_code=429, detail="Another scan is running. Please wait a few seconds.")
        async with _scan_semaphore:
            t0 = time.time()
            data = await _safe_timeout(async_run_war_room(symbols), timeout=90, label="war-room")
            elapsed = time.time() - t0

        _warroom_cache[cache_key] = (now, data)
        _evict_cache(_warroom_cache, _WARROOM_CACHE_MAX)
        gc.collect()
        return data
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="War Room scan timed out (90s). Try fewer tickers.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("War Room error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

_regime_cache: dict = {}
_REGIME_TTL = 300
_REGIME_CACHE_MAX = 20


def _safe_json_response(data):
    """Serialize with NaN/Infinity handling."""
    import math
    import numpy as np

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o) if not (math.isnan(o) or math.isinf(o)) else None
            if isinstance(o, np.ndarray):
                return o.tolist()
            if hasattr(o, 'isoformat'):
                return o.isoformat()
            return super().default(o)

    body = json.dumps(data, cls=_Enc)
    return JSONResponse(content=json.loads(body))


@router.get("/api/regime-scan")
async def regime_scan(tickers: str = "", days: int = 30):
    """Regime Scanner — cross-gate strategy analysis for one or more symbols"""
    try:
        if not tickers:
            raise HTTPException(status_code=400, detail="Provide tickers param (comma-separated)")
        symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if len(symbols) > 15:
            raise HTTPException(status_code=400, detail="Max 15 tickers per scan")
        days = min(max(days, 5), 90)

        cache_key = f"{','.join(sorted(symbols))}:{days}"
        now = time.time()
        entry = _regime_cache.get(cache_key)
        if entry and (now - entry[0]) < _REGIME_TTL:
            data = entry[1]
            data["_cache"] = {"hit": True, "age": round(now - entry[0], 1)}
            return _safe_json_response(data)

        if _scan_semaphore.locked():
            raise HTTPException(status_code=429, detail="Another scan is running. Please wait a few seconds.")

        scanner = await _get_regime_scanner()

        async with _scan_semaphore:
            t0 = time.time()
            if len(symbols) == 1:
                result = await _safe_timeout(asyncio.to_thread(scanner.scan, symbols[0], days), timeout=45, label="regime-single")
                resp = result.to_dict()
            else:
                results = await _safe_timeout(asyncio.to_thread(scanner.scan_watchlist, symbols, days), timeout=60, label="regime-watchlist")
                comparison = scanner.compare_watchlist(results)
                result_dicts = {}
                for sym, r in results.items():
                    try:
                        result_dicts[sym] = r.to_dict()
                    except Exception as e2:
                        logger.error("to_dict failed for %s: %s", sym, e2)
                        result_dicts[sym] = {"symbol": sym, "error": str(e2), "days_analyzed": 0}
                resp = {"comparison": comparison, "results": result_dicts}
            elapsed = time.time() - t0

        resp["_cache"] = {"hit": False, "scan_time": round(elapsed, 2)}
        _regime_cache[cache_key] = (now, resp)
        _evict_cache(_regime_cache, _REGIME_CACHE_MAX)
        gc.collect()
        return _safe_json_response(resp)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Regime scan timed out. Try fewer tickers or shorter lookback.")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error("Regime scan error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/regime-levels/{symbol}")
async def regime_levels(symbol: str):
    """Get ATR-scaled strategy levels for a symbol"""
    try:
        scanner = await _get_regime_scanner()
        data = await _safe_timeout(asyncio.to_thread(scanner.get_strategy_levels, symbol.upper()), timeout=20, label="regime-levels")
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE WATCHLIST SCAN
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/scan/live")
async def scan_live(watchlist: str = "Mega Cap Tech", limit: int = 20):
    try:
        from finnhub_scanner_v2 import FinnhubScanner
        from watchlist_manager import WatchlistManager

        def _scan_sync():
            # Lazy load
            key = os.environ.get("FINNHUB_API_KEY", "")
            scanner = FinnhubScanner(api_key=key)
            mgr = WatchlistManager()
            wl = mgr.get_watchlist(watchlist)
            if not wl:
                return None
            symbols = [s.symbol for s in wl.symbols][:limit]
            results = []
            for sym in symbols:
                try:
                    analysis = scanner.analyze(sym)
                    if analysis:
                        # safe dict conversion
                        d = {}
                        for k, v in (analysis if isinstance(analysis, dict) else analysis.__dict__).items():
                            try:
                                json.dumps(v)
                                d[k] = v
                            except (TypeError, ValueError):
                                d[k] = str(v)
                        results.append(d)
                except Exception:
                    continue
            return {"watchlist": watchlist, "count": len(results), "results": results}

        if _scan_semaphore.locked():
            raise HTTPException(status_code=429, detail="Another scan is running. Please wait a few seconds.")
        async with _scan_semaphore:
            result = await _safe_timeout(asyncio.to_thread(_scan_sync), timeout=60, label="scan-live")
        if result is None:
            raise HTTPException(status_code=404, detail=f"Watchlist '{watchlist}' not found")
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Live scan timed out (60s). Try fewer symbols.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE REVERSAL DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/api/structure/reversals/{symbol}")
async def structure_reversals(symbol: str, min_confidence: float = 40.0):
    """Structure-based reversal detection using macro structure analysis."""
    try:
        from polygon_data import get_bars
        from structure_reversal_detector import StructureReversalDetector, StructureContext
        from rangewatcher.range_watcher import RangeWatcher

        detector = StructureReversalDetector(min_confidence=min_confidence)
        range_analyzer = RangeWatcher()

        def _structure_sync():
            sym = symbol.upper()
            df_daily = get_bars(sym, period="3mo", interval="1d")
            df_weekly = get_bars(sym, period="1y", interval="1wk")

            if df_daily is None or df_daily.empty or len(df_daily) < 30:
                return {"error": f"Insufficient daily data for {sym}"}
            if df_weekly is None or df_weekly.empty or len(df_weekly) < 8:
                return {"error": f"Insufficient weekly data for {sym}"}

            df_daily.columns = [c.lower() for c in df_daily.columns]
            df_weekly.columns = [c.lower() for c in df_weekly.columns]

            range_result = range_analyzer.analyze(df_daily, symbol=sym)

            from chart_input_analyzer import RangeContext
            from finnhub_scanner_v2 import TechnicalCalculator

            range_context = TechnicalCalculator.calculate_range_structure(
                df_weekly=df_weekly, df_daily=df_daily,
                current_price=df_daily['close'].iloc[-1]
            )

            period_3d = range_result.periods.get(3)
            period_6d = range_result.periods.get(6)
            period_30d = range_result.periods.get(30)

            structure_ctx = StructureContext(
                weekly_trend=range_context.trend,
                weekly_hh=range_context.hh_count,
                weekly_hl=range_context.hl_count,
                weekly_lh=range_context.lh_count,
                weekly_ll=range_context.ll_count,
                weekly_close_position=range_context.weekly_close_position,
                period_3d_hh=period_3d.higher_highs if period_3d else False,
                period_3d_hl=period_3d.higher_lows if period_3d else False,
                period_3d_lh=period_3d.lower_highs if period_3d else False,
                period_3d_ll=period_3d.lower_lows if period_3d else False,
                period_6d_hh=period_6d.higher_highs if period_6d else False,
                period_6d_hl=period_6d.higher_lows if period_6d else False,
                period_6d_lh=period_6d.lower_highs if period_6d else False,
                period_6d_ll=period_6d.lower_lows if period_6d else False,
                period_30d_hh=period_30d.higher_highs if period_30d else False,
                period_30d_hl=period_30d.higher_lows if period_30d else False,
                period_30d_lh=period_30d.lower_highs if period_30d else False,
                period_30d_ll=period_30d.lower_lows if period_30d else False,
                current_price=range_result.current_price,
                position_in_3d_range=period_3d.position_in_range if period_3d else 0.5,
                position_in_30d_range=period_30d.position_in_range if period_30d else 0.5,
                compression_ratio=range_context.compression_ratio,
                nearest_resistance=period_30d.nearest_resistance if period_30d else range_result.current_price * 1.05,
                nearest_support=period_30d.nearest_support if period_30d else range_result.current_price * 0.95,
            )

            vp_data = None
            try:
                from volume_profile import calculate_volume_profile
                vp = calculate_volume_profile(df_daily.tail(20))
                vp_data = {'val': vp.get('val'), 'vah': vp.get('vah'), 'poc': vp.get('poc')}
            except Exception:
                pass

            detector.min_confidence = min_confidence
            alerts = detector.analyze(
                df=df_daily, structure_context=structure_ctx,
                symbol=sym, vp_data=vp_data
            )

            alerts_dict = []
            for alert in alerts:
                alerts_dict.append({
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "confidence": round(alert.confidence, 1),
                    "current_price": round(alert.current_price, 2),
                    "trigger_level": round(alert.trigger_level, 2) if alert.trigger_level else None,
                    "target_level": round(alert.target_level, 2) if alert.target_level else None,
                    "stop_level": round(alert.stop_level, 2) if alert.stop_level else None,
                    "description": alert.description,
                    "timeframe": alert.timeframe,
                    "signals": alert.signals,
                    "structure_score": round(alert.structure_score, 1),
                    "volume_score": round(alert.volume_score, 1),
                    "vp_confluence": round(alert.vp_confluence, 1),
                    "momentum_score": round(alert.momentum_score, 1),
                    "range_position": round(alert.range_position, 1),
                    "divergence_score": round(alert.divergence_score, 1),
                })

            return {
                "symbol": sym,
                "alert_count": len(alerts),
                "alerts": alerts_dict,
                "structure_context": {
                    "weekly_trend": structure_ctx.weekly_trend,
                    "weekly_hh": structure_ctx.weekly_hh,
                    "weekly_hl": structure_ctx.weekly_hl,
                    "weekly_lh": structure_ctx.weekly_lh,
                    "weekly_ll": structure_ctx.weekly_ll,
                    "compression_ratio": round(structure_ctx.compression_ratio, 3),
                    "position_in_30d_range": round(structure_ctx.position_in_30d_range * 100, 1),
                },
                "timestamp": datetime.now().isoformat()
            }

        data = await _safe_timeout(asyncio.to_thread(_structure_sync), timeout=30, label="structure")
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Structure reversal analysis timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Structure reversal error: %s", e)
        raise HTTPException(status_code=500, detail=f"Structure reversal analysis failed: {str(e)}")
