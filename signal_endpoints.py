"""
Signal Probability Endpoints
==============================
Exposes the polygon_signal_tool analysis as an API.
Returns historical probability context (not a signal — context for trade decisions).

Endpoints:
  GET /api/signal/{ticker}         — Full probability analysis
  GET /api/signal/{ticker}/quick   — Compact probability card data

Author: Rob's Trading Systems
"""

import os
import sys
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

# Add polygon_signal_tool to path
TOOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polygon_signal_tool")
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

from signal_analyzer import classify_days, run_full_analysis
from signal_config import POLYGON_BASE_URL

import requests
import json
from datetime import datetime, timedelta

signal_router = APIRouter(tags=["signal"])

# In-memory cache: ticker -> { data, timestamp }
_cache = {}
_vwap_cache = {}
CACHE_TTL_SECONDS = 900  # 15 min

# ── VWAP magnet helper (uses War Room intraday analysis) ──

def _get_vwap_magnet(ticker: str) -> dict:
    """Get VWAP reversion stats. Returns dict or empty if War Room unavailable."""
    now = datetime.now().timestamp()
    cache_key = ticker.upper()
    if cache_key in _vwap_cache:
        entry = _vwap_cache[cache_key]
        if now - entry["ts"] < CACHE_TTL_SECONDS:
            return entry["data"]
    try:
        from war_room import get_master_analysis
        dna = get_master_analysis(ticker.upper(), lookback_days=45)
        if not dna:
            return {}
        result = {
            "avg_max_vwap_dist": dna.get("avg_max_vwap_dist", 0),
            "vwap_revert_rate": dna.get("vwap_revert_rate", 0),
            "avg_vwap_crosses": dna.get("avg_vwap_crosses", 0),
            "avg_min_dist_after": dna.get("avg_min_dist_after", 0),
        }
        _vwap_cache[cache_key] = {"data": result, "ts": now}
        return result
    except Exception:
        return {}


def _fetch_data(ticker: str, days: int, api_key: str):
    """Fetch OHLCV from Polygon.io (with file cache for delta-sync)."""
    cache_dir = os.path.join(TOOL_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker.upper()}.json")

    cached_data = []
    last_cached_date = None

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        if cached_data:
            last_cached_date = cached_data[-1]["date"]

    end_date = datetime.now()
    if last_cached_date:
        start_date = datetime.strptime(last_cached_date, "%Y-%m-%d") + timedelta(days=1)
        if start_date.date() > end_date.date():
            cutoff = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
            return [d for d in cached_data if d["date"] >= cutoff]
    else:
        start_date = end_date - timedelta(days=days + 30)

    from_str = start_date.strftime("%Y-%m-%d")
    to_str = end_date.strftime("%Y-%m-%d")

    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{from_str}/{to_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            if cached_data:
                cutoff = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
                return [d for d in cached_data if d["date"] >= cutoff]
            return None

        body = resp.json()
        results = body.get("results", [])
        if not results and cached_data:
            cutoff = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
            return [d for d in cached_data if d["date"] >= cutoff]

        new_data = []
        for bar in results:
            ts = bar["t"] / 1000
            dt = datetime.utcfromtimestamp(ts)
            new_data.append({
                "date": dt.strftime("%Y-%m-%d"),
                "open": bar["o"], "high": bar["h"],
                "low": bar["l"], "close": bar["c"],
                "volume": bar.get("v", 0),
            })

        if cached_data:
            existing = {d["date"] for d in cached_data}
            for nd in new_data:
                if nd["date"] not in existing:
                    cached_data.append(nd)
            cached_data.sort(key=lambda x: x["date"])
            merged = cached_data
        else:
            merged = new_data

        with open(cache_file, "w") as f:
            json.dump(merged, f)

        cutoff = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
        return [d for d in merged if d["date"] >= cutoff]
    except Exception:
        if cached_data:
            cutoff = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
            return [d for d in cached_data if d["date"] >= cutoff]
        return None


def _run_analysis(ticker: str, days: int = 365):
    """Run full probability analysis, with in-memory TTL cache."""
    cache_key = f"{ticker.upper()}_{days}"
    now = datetime.now().timestamp()

    if cache_key in _cache:
        entry = _cache[cache_key]
        if now - entry["ts"] < CACHE_TTL_SECONDS:
            return entry["data"]

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        return None

    raw = _fetch_data(ticker, days, api_key)
    if not raw or len(raw) < 10:
        return None

    classified = classify_days(raw)
    analysis = run_full_analysis(classified)
    _cache[cache_key] = {"data": analysis, "ts": now}
    return analysis


@signal_router.get("/api/signal/{ticker}")
async def get_signal(ticker: str, days: int = Query(365, description="Lookback")):
    """Full probability analysis for a ticker."""
    analysis = _run_analysis(ticker.upper(), days)
    if not analysis:
        return JSONResponse(content={"error": "No data available"}, status_code=404)

    sig = analysis["signal"]
    stats = analysis["all_stats"]
    straddle = analysis["straddle"]
    rg = analysis["range_groups"]

    # Build scenario summary
    scenarios = {}
    for key in ["call_all", "call_green", "call_red", "call_green2", "call_red2",
                 "call_green3", "call_red3", "put_all", "put_green", "put_red",
                 "put_green2", "put_red2", "put_green3", "put_red3"]:
        s = stats.get(key)
        if s:
            scenarios[key] = {
                "count": s["count"],
                "hit_1d": round(s["rate_1d"] * 100, 1),
                "hit_3d": round(s["rate_3d"] * 100, 1),
                "hit_5d": round(s["rate_5d"] * 100, 1),
                "avg_best_3d": round(s.get("avg_best_pct_3d", 0), 2),
                "avg_best_3d_dollars": round(s["avg_best_3d"], 2),
                "avg_worst_3d": round(s.get("avg_worst_pct_3d", 0), 2),
                "avg_worst_3d_dollars": round(s["avg_worst_3d"], 2),
                "close_win_1d": round(s["close_win_1d"] * 100, 1),
            }

    # Range summary
    ranges = {}
    for label, data in rg.items():
        if data:
            ranges[label] = {
                "count": data["count"],
                "avg_up_1d": round(data["avg_up_1d"], 2),
                "avg_dn_1d": round(data["avg_dn_1d"], 2),
                "avg_up_pct_1d": round(data["avg_up_pct_1d"] * 100, 2),
                "avg_dn_pct_1d": round(data["avg_dn_pct_1d"] * 100, 2),
                "up_wins_1d": round(data["up_bigger_rate_1d"] * 100, 1),
                "avg_up_3d": round(data["avg_up_3d"], 2),
                "avg_dn_3d": round(data["avg_dn_3d"], 2),
                "avg_up_pct_3d": round(data["avg_up_pct_3d"] * 100, 2),
                "avg_dn_pct_3d": round(data["avg_dn_pct_3d"] * 100, 2),
            }

    return JSONResponse(content={
        "ticker": ticker.upper(),
        "days_analyzed": analysis["n"],
        "green_days": analysis["green_count"],
        "red_days": analysis["red_count"],
        "today": sig["today"],
        "current_condition": sig["range_condition"],
        "scenarios": scenarios,
        "ranges": ranges,
        "straddle": {
            "both_rate": round(straddle["both_rate"] * 100, 1),
            "at_least_one_rate": round(straddle["at_least_one_rate"] * 100, 1),
            "avg_daily_best": round(straddle["avg_daily_best"], 2),
            "avg_call_scalp": round(straddle["avg_call_scalp"], 2),
            "avg_put_scalp": round(straddle["avg_put_scalp"], 2),
        },
        "expected": {
            "upside_1d": round(sig["expected_upside"], 2),
            "downside_1d": round(sig["expected_downside"], 2),
            "upside_3d": round(sig.get("expected_upside_3d", 0), 2),
            "downside_3d": round(sig.get("expected_downside_3d", 0), 2),
        },
        "close_location": analysis.get("close_location", {}),
        "gap_analysis": analysis.get("gap_analysis", {}),
        "vol_regime": analysis.get("vol_regime", {}),
        "extension": analysis.get("extension", {}),
        "opex": analysis.get("opex", {}),
        "vwap_magnet": _get_vwap_magnet(ticker),
    })


@signal_router.get("/api/signal/{ticker}/quick")
async def get_signal_quick(ticker: str, days: int = Query(365)):
    """
    Compact probability context card data.
    Returns just the key numbers for embedding in a trade plan UI.
    """
    analysis = _run_analysis(ticker.upper(), days)
    if not analysis:
        return JSONResponse(content={"error": "No data available"}, status_code=404)

    sig = analysis["signal"]
    stats = analysis["all_stats"]
    straddle = analysis["straddle"]
    today = sig["today"]

    # Pick the right scenario based on today's condition
    if today["color"] == "RED":
        rs = today["rstreak"]
        call_key = f"call_red{rs}" if rs >= 2 and f"call_red{rs}" in stats else "call_red"
        put_key = f"put_red{rs}" if rs >= 2 and f"put_red{rs}" in stats else "put_red"
    else:
        gs = today["gstreak"]
        call_key = f"call_green{gs}" if gs >= 2 and f"call_green{gs}" in stats else "call_green"
        put_key = f"put_green{gs}" if gs >= 2 and f"put_green{gs}" in stats else "put_green"

    call_stats = stats.get(call_key, stats.get("call_all"))
    put_stats = stats.get(put_key, stats.get("put_all"))

    def _s(s):
        if not s:
            return {}
        return {
            "sample": s["count"],
            "hit_1d": round(s["rate_1d"] * 100, 1),
            "hit_3d": round(s["rate_3d"] * 100, 1),
            "hit_5d": round(s["rate_5d"] * 100, 1),
            "avg_best_3d": round(s.get("avg_best_pct_3d", 0), 2),
            "avg_best_3d_dollars": round(s["avg_best_3d"], 2),
            "close_win_1d": round(s["close_win_1d"] * 100, 1),
        }

    # ── New predictability context ──
    cl = analysis.get("close_location", {})
    ga = analysis.get("gap_analysis", {})
    vr = analysis.get("vol_regime", {})
    ext = analysis.get("extension", {})
    opx = analysis.get("opex", {})

    return JSONResponse(content={
        "ticker": ticker.upper(),
        "days_analyzed": analysis["n"],
        "condition": sig["range_condition"],
        "today_color": today["color"],
        "streak": today["rstreak"] if today["color"] == "RED" else today["gstreak"],
        "call_odds": _s(call_stats),
        "put_odds": _s(put_stats),
        "straddle_rate": round(straddle["at_least_one_rate"] * 100, 1),
        "expected_up_1d": round(sig["expected_upside"], 2),
        "expected_dn_1d": round(sig["expected_downside"], 2),
        "expected_up_3d": round(sig.get("expected_upside_3d", 0), 2),
        "expected_dn_3d": round(sig.get("expected_downside_3d", 0), 2),
        "call_scenario": call_key,
        "put_scenario": put_key,
        # ── Predictability Map ──
        "close_location": {
            "today_clv": cl.get("today_clv", 0),
            "trend_cluster": cl.get("trend_cluster", ""),
            "strong_higher_open": cl.get("strong_closers", {}).get("higher_open_rate", 0),
            "weak_lower_open": cl.get("weak_closers", {}).get("lower_open_rate", 0),
        },
        "gap": {
            "today_gap_pct": ga.get("today_gap_pct", 0),
            "today_gap_dir": ga.get("today_gap_direction", ""),
            "today_filled": ga.get("today_gap_filled", False),
            "gap_up_fill_rate": ga.get("gap_ups", {}).get("fill_rate", 0),
            "gap_dn_fill_rate": ga.get("gap_downs", {}).get("fill_rate", 0),
        },
        "regime": {
            "current": vr.get("regime", ""),
            "action": vr.get("action", ""),
            "atr_ratio": vr.get("atr_ratio", 0),
        },
        "extension": {
            "zscore": ext.get("zscore", 0),
            "status": ext.get("status", ""),
            "extension_pct": ext.get("extension_pct", 0),
            "revert_rate": ext.get("revert_after_extreme_rate", 0),
        },
        "opex": {
            "today_is_opex": opx.get("today_is_opex", False),
            "opex_pin_rate": opx.get("opex", {}).get("pin_rate", 0),
            "opex_avg_range": opx.get("opex", {}).get("avg_range_pct", 0),
            "normal_avg_range": opx.get("non_opex", {}).get("avg_range_pct", 0),
        },
        "vwap_magnet": _get_vwap_magnet(ticker),
    })
