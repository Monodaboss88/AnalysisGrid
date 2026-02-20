"""
Alpha Scanner Endpoints — FastAPI router for the 7-step automated bullish finder.
"""
import logging
from typing import Optional
from fastapi import APIRouter, Query
from alpha_scanner import async_run_alpha_scan, UNIVERSES

logger = logging.getLogger(__name__)

alpha_router = APIRouter(tags=["alpha"])

VALID_TIERS = {"DAY", "SWING", "POSITION", "MACRO"}


@alpha_router.get("/api/alpha/scan")
async def alpha_scan(
    universe: str = Query("all", description="Universe preset: all, tech, semis, momentum, etfs, mag7"),
    max_results: int = Query(5, ge=1, le=10, description="Max results to return"),
    duration_tier: Optional[str] = Query(None, description="Filter by trade duration: DAY, SWING, POSITION, MACRO"),
):
    """Run the full 7-step Alpha Scanner pipeline. Optionally filter by trade duration tier."""
    if universe not in UNIVERSES:
        return {"error": f"Unknown universe '{universe}'. Options: {list(UNIVERSES.keys())}"}

    # Run full scan (unfiltered)
    result = await async_run_alpha_scan(universe, max_results=max_results if not duration_tier else 10)
    
    # Apply duration tier filter if requested
    if duration_tier and duration_tier.upper() in VALID_TIERS:
        tier = duration_tier.upper()
        filtered = [c for c in result.get("results", []) if c.get("duration_tier") == tier]
        result["results"] = filtered[:max_results]
        result["meta"]["duration_filter"] = tier
        result["meta"]["survivors"] = len(filtered[:max_results])
    
    return result


@alpha_router.get("/api/alpha/universes")
async def alpha_universes():
    """List available universe presets."""
    return {k: len(v) for k, v in UNIVERSES.items()}
