"""
Convergence Edge — FastAPI Endpoint
====================================
FastAPI router that mirrors the pattern used by other routers in unified_server.py.

Register in unified_server.py:
    from convergence_endpoint import convergence_router
    app.include_router(convergence_router, prefix="/api/convergence")

Routes
------
GET  /scan/{ticker}          → single-ticker convergence scan
POST /watchlist               → batch scan a list
GET  /profiles                → list available weight profiles
"""

import asyncio
import logging
import traceback
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from convergence_scanner import (
    score_convergence,
    WEIGHT_PROFILES,
)

logger = logging.getLogger(__name__)
convergence_router = APIRouter(tags=["convergence"])


# ── Pydantic models ────────────────────────────────────────────────

class WatchlistRequest(BaseModel):
    tickers: List[str]
    profile: str = "equal"
    sort_by: str = "convergence"        # "convergence" | "direction" | "conviction"
    min_convergence: float = 0
    alert_filter: Optional[str] = None  # "CONVERGENCE" | "DIVERGENCE" | None


# ── Fetch all 8 scanners for one ticker (async, parallel) ──────────

async def _fetch_all_scanners(ticker: str) -> dict:
    """
    Call each of the 8 scanners in parallel and collect raw outputs.
    Uses the same real scanner functions as card_data_builder / alpha_scanner.
    """
    ticker = ticker.upper()

    async def _fetch_simple(sym: str) -> dict:
        try:
            from alpha_scanner import _scan_universe
            rows = await asyncio.to_thread(_scan_universe, [sym])
            return rows[0] if rows else {}
        except Exception as e:
            logger.debug(f"Convergence simple scan failed for {sym}: {e}")
            return {}

    async def _fetch_mtf_raw(sym: str) -> dict:
        try:
            from unified_server import analyze_live_mtf
            r = await analyze_live_mtf(sym)
            return r if isinstance(r, dict) else {}
        except Exception as e:
            logger.debug(f"Convergence MTF raw failed for {sym}: {e}")
            return {}

    async def _fetch_mtf_ai(sym: str) -> dict:
        try:
            from unified_server import analyze_mtf_with_ai
            r = await analyze_mtf_with_ai(sym, trade_tf="swing")
            return r if isinstance(r, dict) else {}
        except Exception as e:
            logger.debug(f"Convergence MTF AI failed for {sym}: {e}")
            return {}

    async def _fetch_signal_quick(sym: str) -> dict:
        try:
            from signal_endpoints import _run_analysis
            analysis = await asyncio.to_thread(_run_analysis, sym, 365)
            if not analysis:
                return {}
            sig = analysis.get("signal", {})
            stats = analysis.get("all_stats", {})
            straddle = analysis.get("straddle", {})
            today = sig.get("today", {})

            if today.get("color") == "RED":
                rs = today.get("rstreak", 0)
                call_key = f"call_red{rs}" if rs >= 2 and f"call_red{rs}" in stats else "call_red"
                put_key = f"put_red{rs}" if rs >= 2 and f"put_red{rs}" in stats else "put_red"
            else:
                gs = today.get("gstreak", 0)
                call_key = f"call_green{gs}" if gs >= 2 and f"call_green{gs}" in stats else "call_green"
                put_key = f"put_green{gs}" if gs >= 2 and f"put_green{gs}" in stats else "put_green"

            cs = stats.get(call_key, stats.get("call_all", {}))
            ps = stats.get(put_key, stats.get("put_all", {}))

            return {
                "call_hit_1d": round(cs.get("rate_1d", 0) * 100, 1),
                "call_hit_3d": round(cs.get("rate_3d", 0) * 100, 1),
                "put_hit_1d": round(ps.get("rate_1d", 0) * 100, 1),
                "put_hit_3d": round(ps.get("rate_3d", 0) * 100, 1),
                "straddle_rate": round(straddle.get("at_least_one_rate", 0) * 100, 1),
            }
        except Exception as e:
            logger.debug(f"Convergence signal_quick failed for {sym}: {e}")
            return {}

    async def _fetch_options_flow(sym: str) -> dict:
        try:
            from options_flow_scanner import scan_tickers
            result = await asyncio.to_thread(scan_tickers, [sym])
            rows = result.get("results", [])
            return rows[0] if rows else {}
        except Exception as e:
            logger.debug(f"Convergence options_flow failed for {sym}: {e}")
            return {}

    async def _fetch_war_room(sym: str) -> dict:
        try:
            from war_room import get_master_analysis, _compute_signals
            dna = await asyncio.to_thread(get_master_analysis, sym, lookback_days=45)
            if not dna:
                return {}
            sig = _compute_signals(dna, None, [])
            return {
                "regime": dna.get("regime", ""),
                "exhaustion": sig.get("exhaustion", 0),
                "fade_conviction": sig.get("fade_conviction", 0),
                "avg_up_ext": dna.get("avg_up", 0),
                "avg_dn_ext": dna.get("avg_down", 0),
            }
        except Exception as e:
            logger.debug(f"Convergence war_room failed for {sym}: {e}")
            return {}

    async def _fetch_buffett(sym: str) -> dict:
        try:
            from buffett_scanner import scan_tickers as buffett_scan
            result = await asyncio.to_thread(buffett_scan, [sym])
            rows = result.get("results", [])
            return rows[0] if rows else {}
        except Exception as e:
            logger.debug(f"Convergence buffett failed for {sym}: {e}")
            return {}

    async def _fetch_sustainability(sym: str) -> dict:
        try:
            from run_sustainability_analyzer import RunSustainabilityAnalyzer
            analyzer = RunSustainabilityAnalyzer()
            result = await asyncio.to_thread(analyzer.analyze, sym)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.debug(f"Convergence sustainability failed for {sym}: {e}")
            return {}

    # Run all 8 in parallel
    (simple, mtf_raw, mtf_ai, signal_quick,
     options_flow, war_room, buffett, sustainability) = await asyncio.gather(
        _fetch_simple(ticker),
        _fetch_mtf_raw(ticker),
        _fetch_mtf_ai(ticker),
        _fetch_signal_quick(ticker),
        _fetch_options_flow(ticker),
        _fetch_war_room(ticker),
        _fetch_buffett(ticker),
        _fetch_sustainability(ticker),
        return_exceptions=True,
    )

    # Convert exceptions to empty dicts
    def _safe(v):
        return v if isinstance(v, dict) else {}

    return {
        "simple":         _safe(simple),
        "mtf_raw":        _safe(mtf_raw),
        "mtf_ai":         _safe(mtf_ai),
        "signal_quick":   _safe(signal_quick),
        "options_flow":   _safe(options_flow),
        "war_room":       _safe(war_room),
        "buffett":        _safe(buffett),
        "sustainability": _safe(sustainability),
    }


# ── Single ticker ──────────────────────────────────────────────────

@convergence_router.get("/scan/{ticker}")
async def convergence_single(
    ticker: str,
    profile: str = Query("equal", description="Weight profile: daytrade, swing, equal"),
):
    """
    Scan one ticker across all 8 scanners and return the convergence result.

    Example: GET /api/convergence/scan/AAPL?profile=daytrade
    """
    ticker = ticker.upper()
    try:
        scanner_data = await _fetch_all_scanners(ticker)
        result = score_convergence(ticker, scanner_data, profile=profile)
        return {
            "status": "success",
            "scanner": "convergence_edge",
            "data": result.to_dict(),
        }
    except Exception as e:
        logger.error(f"Convergence scan failed for {ticker}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Watchlist batch ────────────────────────────────────────────────

@convergence_router.post("/watchlist")
async def convergence_watchlist(body: WatchlistRequest):
    """
    Batch-scan a watchlist.

    JSON body:
    {
        "tickers":          ["AAPL","MSFT","TSLA"],
        "profile":          "swing",
        "sort_by":          "convergence",
        "min_convergence":  50,
        "alert_filter":     "CONVERGENCE"
    }
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    tickers = [t.upper() for t in body.tickers]

    # Run all tickers in parallel (cap at 5 concurrent to avoid rate limits)
    semaphore = asyncio.Semaphore(5)

    async def _scan_one(t: str):
        async with semaphore:
            data = await _fetch_all_scanners(t)
            return score_convergence(t, data, profile=body.profile)

    raw_results = await asyncio.gather(
        *[_scan_one(t) for t in tickers],
        return_exceptions=True,
    )

    results = []
    for r in raw_results:
        if isinstance(r, Exception):
            continue
        if r.convergence_score >= body.min_convergence:
            if body.alert_filter is None or r.alert_type == body.alert_filter:
                results.append(r)

    # Sort
    if body.sort_by == "direction":
        results.sort(key=lambda r: abs(r.direction_score), reverse=True)
    elif body.sort_by == "conviction":
        grade_order = {"A+": 0, "A": 1, "B+": 2, "B": 3, "C+": 4, "C": 5, "D": 6, "F": 7}
        results.sort(key=lambda r: grade_order.get(r.conviction, 99))
    else:
        results.sort(key=lambda r: r.convergence_score, reverse=True)

    return {
        "status": "success",
        "scanner": "convergence_edge",
        "count": len(results),
        "profile": body.profile,
        "data": [r.to_dict() for r in results],
    }


# ── Profiles list ──────────────────────────────────────────────────

@convergence_router.get("/profiles")
async def convergence_profiles():
    """Return available weight profiles so the front-end can offer a picker."""
    return {
        "status": "success",
        "profiles": {
            name: {k: round(v, 4) for k, v in weights.items()}
            for name, weights in WEIGHT_PROFILES.items()
        },
    }
