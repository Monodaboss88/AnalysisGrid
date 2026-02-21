"""
Alpha Scanner Endpoints — FastAPI router for V2 bi-directional setup discovery.
"""
import logging
from typing import Optional
from fastapi import APIRouter, Query

# Try V2 scanner_refactor first, fall back to V1
try:
    import sys, os
    _refactor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner_refactor")
    if _refactor_dir not in sys.path:
        sys.path.insert(0, _refactor_dir)
    from alpha_scanner import async_run_alpha_scan, UNIVERSES
    _scanner_version = "V2"
except ImportError:
    from alpha_scanner import async_run_alpha_scan, UNIVERSES
    _scanner_version = "V1"

logger = logging.getLogger(__name__)
logger.info(f"Alpha Scanner endpoints using {_scanner_version}")

alpha_router = APIRouter(tags=["alpha"])

VALID_TIERS = {"DAY", "SWING", "POSITION", "MACRO"}


@alpha_router.get("/api/alpha/scan")
async def alpha_scan(
    universe: str = Query("all", description="Universe preset: all, tech, semis, momentum, etfs, mag7"),
    max_results: int = Query(5, ge=1, le=10, description="Max results to return"),
    duration_tier: Optional[str] = Query(None, description="Filter by trade duration: DAY, SWING, POSITION"),
):
    """Run the Alpha Scanner V2 pipeline. Optionally filter by trade duration tier."""
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

    result["meta"]["scanner_version"] = _scanner_version
    return result


@alpha_router.get("/api/alpha/universes")
async def alpha_universes():
    """List available universe presets."""
    return {k: len(v) for k, v in UNIVERSES.items()}
