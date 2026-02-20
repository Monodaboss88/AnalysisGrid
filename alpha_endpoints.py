"""
Alpha Scanner Endpoints — FastAPI router for the 7-step automated bullish finder.
"""
import logging
from fastapi import APIRouter, Query
from alpha_scanner import async_run_alpha_scan, UNIVERSES

logger = logging.getLogger(__name__)

alpha_router = APIRouter(tags=["alpha"])


@alpha_router.get("/api/alpha/scan")
async def alpha_scan(
    universe: str = Query("all", description="Universe preset: all, tech, semis, momentum, etfs, mag7"),
    max_results: int = Query(5, ge=1, le=10, description="Max results to return"),
):
    """Run the full 7-step Alpha Scanner pipeline."""
    if universe not in UNIVERSES:
        return {"error": f"Unknown universe '{universe}'. Options: {list(UNIVERSES.keys())}"}

    result = await async_run_alpha_scan(universe, max_results)
    return result


@alpha_router.get("/api/alpha/universes")
async def alpha_universes():
    """List available universe presets."""
    return {k: len(v) for k, v in UNIVERSES.items()}
