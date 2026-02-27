"""
Combo Scanner Endpoints
========================
FastAPI router for the stacked-edge combo scanner.

GET /api/combo-scan         — scan tickers for combo setups
GET /api/combo-scan/{ticker} — scan single ticker (detail view)
"""

from __future__ import annotations
import logging
from typing import Optional

from fastapi import APIRouter, Query as QueryParam
from combo_scanner import scan_combos, scan_single, PRESETS

logger = logging.getLogger(__name__)
combo_router = APIRouter(prefix="/api", tags=["combo"])


@combo_router.get("/combo-scan")
async def combo_scan(
    tickers: Optional[str] = QueryParam(None, description="Comma-separated tickers"),
    preset: Optional[str] = QueryParam(None, description="Preset watchlist: mag7, tech, mega, etf, meme"),
    min_grade: str = QueryParam("D", description="Minimum grade to include: A, B, C, D, F"),
):
    """
    Scan tickers for high-conviction combo setups.
    Pass ?tickers=NVDA,META  or  ?preset=mag7
    """
    ticker_list = []

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif preset:
        preset_key = preset.lower()
        if preset_key in PRESETS:
            ticker_list = PRESETS[preset_key]
        else:
            return {
                "error": f"Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}",
                "results": [],
            }
    else:
        return {
            "error": "Provide ?tickers=NVDA,META or ?preset=mag7",
            "results": [],
        }

    if len(ticker_list) > 30:
        ticker_list = ticker_list[:30]

    result = await scan_combos(ticker_list, min_grade=min_grade.upper())
    return result


@combo_router.get("/combo-scan/{ticker}")
async def combo_scan_single(ticker: str):
    """
    Detailed combo scan for a single ticker.
    Returns all setups found, including those below D grade.
    """
    ticker = ticker.upper()
    setups = await scan_single(ticker)

    # Sort by score
    setups.sort(key=lambda s: s["score"], reverse=True)

    best = setups[0] if setups else None
    return {
        "ticker": ticker,
        "setups_found": len(setups),
        "best_setup": best.get("setup_label") if best else "None",
        "best_score": best.get("score") if best else 0,
        "best_grade": best.get("grade") if best else "F",
        "all_setups": setups,
    }
