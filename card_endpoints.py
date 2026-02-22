"""
card_endpoints.py — FastAPI router for trading card generation.

Endpoints:
    GET /api/card/{symbol}/data       — Raw JSON card data
    GET /api/card/{symbol}/execution  — Rendered HTML execution card
    GET /api/card/{symbol}/thesis     — Rendered HTML thesis card
    GET /api/card/{symbol}/both       — Both cards side-by-side
"""

import time
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse
from card_data_builder import build_card_data
from card_renderer import render_capital_ladder, render_thesis_card

card_router = APIRouter(tags=["Trading Cards"])

# ── Short-lived scan cache (avoids re-scanning for render endpoints) ──
_card_cache: dict = {}   # key = "SYM:tf" → {"data": dict, "ts": float}
_CACHE_TTL = 120         # seconds — render calls reuse scan data


def _cache_key(symbol: str, tf: str) -> str:
    return f"{symbol.upper()}:{tf}"


def _get_cached(symbol: str, tf: str) -> dict | None:
    key = _cache_key(symbol, tf)
    entry = _card_cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _set_cache(symbol: str, tf: str, data: dict):
    key = _cache_key(symbol, tf)
    _card_cache[key] = {"data": data, "ts": time.time()}
    # Evict old entries (keep max 20)
    if len(_card_cache) > 20:
        oldest = min(_card_cache, key=lambda k: _card_cache[k]["ts"])
        _card_cache.pop(oldest, None)


async def _get_card_data(symbol: str, tf: str) -> dict:
    """Get card data from cache or build fresh."""
    cached = _get_cached(symbol, tf)
    if cached:
        return cached
    data = await build_card_data(symbol, trade_tf=tf)
    _set_cache(symbol, tf, data)
    return data


@card_router.get("/api/card/{symbol}/data")
async def get_card_data(symbol: str, trade_tf: str = Query("swing")):
    """Return structured JSON with all scanner data for both cards."""
    data = await _get_card_data(symbol, trade_tf)
    return JSONResponse(content=data)


@card_router.get("/api/card/{symbol}/execution")
async def get_execution_card(symbol: str, trade_tf: str = Query("swing")):
    """Return rendered HTML Capital Ladder card (replaces old execution card)."""
    data = await _get_card_data(symbol, trade_tf)
    html = render_capital_ladder(data)
    return HTMLResponse(content=html)


@card_router.get("/api/card/{symbol}/thesis")
async def get_thesis_card(symbol: str, trade_tf: str = Query("swing")):
    """Return rendered HTML thesis card."""
    data = await _get_card_data(symbol, trade_tf)
    html = render_thesis_card(data)
    return HTMLResponse(content=html)


@card_router.get("/api/card/{symbol}/both")
async def get_both_cards(symbol: str, trade_tf: str = Query("swing")):
    """Return both cards side-by-side in a single HTML page."""
    data = await _get_card_data(symbol, trade_tf)
    exec_html = render_capital_ladder(data)
    thesis_html = render_thesis_card(data)

    page = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data.get('symbol','')} Trading Cards</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0a0e17;
            font-family: 'Inter', -apple-system, sans-serif;
            display: flex; justify-content: center; align-items: flex-start;
            gap: 24px; padding: 24px; min-height: 100vh; flex-wrap: wrap;
        }}
        @media (max-width: 768px) {{
            body {{ padding: 8px; gap: 12px; flex-direction: column; align-items: center; }}
        }}
        @media print {{
            body {{ background: white; gap: 12px; padding: 8px; }}
        }}
    </style>
</head>
<body>
{thesis_html}
{exec_html}
</body>
</html>"""
    return HTMLResponse(content=page)
