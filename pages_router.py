"""
Pages Router — All HTML page-serving routes.
============================================
Zero heavy imports — just FileResponse for static .html files.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

router = APIRouter()

_no_cache = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@router.get("/")
@router.get("/index")
@router.get("/index.html")
async def serve_root():
    for f in ("public/index.html", "index.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    return HTMLResponse("<h1>AnalysisGrid API</h1><p>Server is running.</p>")


@router.get("/v2")
async def serve_v2():
    for f in ("public/stock-options-scanner.html", "stock-options-scanner.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    return FileResponse("stock-options-scanner.html", headers=_no_cache)


@router.get("/desk")
async def serve_desk():
    for f in ("public/desk.html", "public/trade-desk.html", "desk.html", "trade-desk.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    raise HTTPException(status_code=404, detail="desk.html not found")


@router.get("/catalyst.html")
@router.get("/catalyst")
async def serve_catalyst():
    for f in ("public/catalyst.html", "catalyst.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    raise HTTPException(status_code=404, detail="catalyst.html not found")


@router.get("/regime")
@router.get("/regime.html")
async def serve_regime():
    for f in ("public/regime.html", "regime.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    raise HTTPException(status_code=404, detail="regime.html not found")


@router.get("/login.html")
@router.get("/login")
async def serve_login():
    for f in ("public/login.html", "login.html"):
        if os.path.exists(f):
            return FileResponse(f, headers=_no_cache)
    raise HTTPException(status_code=404, detail="login.html not found")


# ── Convenience routes for every public/*.html page ──
_page_routes = {
    "/cards": "cards.html",
    "/options": "options.html",
    "/charts": "charts.html",
    "/journal": "journal.html",
    "/sustainability": "sustainability.html",
    "/research": "research.html",
    "/growth": "growth.html",
    "/warroom": "warroom.html",
    "/buffett": "buffett.html",
    "/backtest": "backtest.html",
    "/convergence": "convergence.html",
    "/combo": "combo.html",
    "/simple": "simple.html",
    "/admin": "admin.html",
    "/upgrade": "upgrade.html",
    "/claude-options": "claude-options.html",
    "/desk-view": "desk-view.html",
}

for _route, _filename in _page_routes.items():
    def _make_handler(fname=_filename):
        async def handler():
            for f in (f"public/{fname}", fname):
                if os.path.exists(f):
                    return FileResponse(f, headers=_no_cache)
            raise HTTPException(status_code=404, detail=f"{fname} not found")
        return handler
    router.get(_route)(_make_handler())
    router.get(f"/{_filename}")(_make_handler())
