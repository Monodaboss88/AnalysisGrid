"""
Trade Journal Endpoints
========================
Wires the TradeJournal SQLite analytics engine to the FastAPI server.

Primary storage remains Firestore (users/{uid}/trades).
This module provides:
  - /api/journal/export  — CSV/JSON export of Firestore trades
  - /api/journal/stats   — Rich analytics (win rate, R-multiples, Fib performance)
  - /api/journal/sync    — Push Firestore trades into SQLite for analytics
  
Author: Rob's Trading Systems
"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response, JSONResponse
from trade_journal import TradeJournal, JournalEntry
from typing import Optional
import json

journal_router = APIRouter(prefix="/api/journal", tags=["journal"])

# Singleton journal instance
_journal: Optional[TradeJournal] = None

def _get_journal() -> TradeJournal:
    global _journal
    if _journal is None:
        _journal = TradeJournal()
    return _journal


@journal_router.get("/stats")
async def get_journal_stats(
    request: Request,
    days: int = Query(30, description="Lookback period"),
    user_id: str = Query("anonymous", description="User ID for stats")
):
    """
    Rich trade statistics: win rate, R-multiples, Fib zone performance,
    setup grade breakdown, expectancy.
    """
    try:
        journal = _get_journal()
        stats = journal.get_stats(user_id=user_id, days=days)
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@journal_router.get("/export")
async def export_journal(
    request: Request,
    user_id: str = Query("anonymous", description="User ID"),
    format: str = Query("json", description="Export format: json or csv"),
    days: int = Query(365, description="Lookback period")
):
    """Export trade journal to JSON or CSV."""
    try:
        journal = _get_journal()
        result = journal.export_journal(user_id=user_id, format=format, days=days)

        if format == "csv":
            return Response(
                content=result,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=trade_journal.csv"}
            )
        else:
            return JSONResponse(content=json.loads(result) if result else [])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@journal_router.get("/fib-report")
async def get_fib_report(
    request: Request,
    user_id: str = Query("anonymous", description="User ID"),
    days: int = Query(90, description="Lookback period")
):
    """Get Fibonacci performance report text."""
    try:
        journal = _get_journal()
        report = journal.get_fib_report(user_id=user_id, days=days)
        return JSONResponse(content={"report": report})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@journal_router.post("/sync")
async def sync_trades(request: Request):
    """
    Push trades from Firestore into the SQLite analytics journal.
    Accepts an array of trade objects.  Idempotent — skips duplicates
    by checking (symbol, user_id, logged_at).
    
    Body: { "user_id": "uid", "trades": [ { ... }, ... ] }
    """
    try:
        body = await request.json()
        user_id = body.get("user_id", "anonymous")
        trades = body.get("trades", [])
        journal = _get_journal()
        
        synced = 0
        skipped = 0
        
        for t in trades:
            try:
                # Map Firestore field names → JournalEntry fields
                entry_price = t.get("entry") or t.get("entry_price") or 0
                stop_loss = t.get("stop") or t.get("stop_loss") or 0
                target1 = t.get("target1") or t.get("target") or 0
                target2 = t.get("target2") or 0
                
                # Calculate R:R
                risk = abs(entry_price - stop_loss) if entry_price and stop_loss else 1
                rr_t1 = abs(target1 - entry_price) / risk if risk > 0 and target1 else 0
                rr_t2 = abs(target2 - entry_price) / risk if risk > 0 and target2 else 0
                
                # Map status: Firestore 'pending' → journal 'PLANNED', 'open' → 'OPEN'
                status_map = {
                    "pending": "PLANNED",
                    "open": "OPEN",
                    "win": "WIN",
                    "WIN": "WIN",
                    "loss": "LOSS",
                    "LOSS": "LOSS",
                    "breakeven": "BREAKEVEN",
                    "BREAKEVEN": "BREAKEVEN",
                    "closed": "WIN",
                    "cancelled": "CANCELLED"
                }
                status = status_map.get(t.get("status", ""), "PLANNED")
                
                entry = JournalEntry(
                    id=None,
                    user_id=user_id,
                    symbol=t.get("symbol", "").upper(),
                    direction=t.get("direction", "LONG").upper(),
                    timeframe=t.get("timeframe", "1HR"),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    risk_reward_t1=round(rr_t1, 2),
                    risk_reward_t2=round(rr_t2, 2),
                    signal=t.get("signal", ""),
                    confidence=t.get("confidence", 0),
                    bull_score=t.get("bull_score", 0),
                    bear_score=t.get("bear_score", 0),
                    ai_commentary=t.get("ai_commentary", ""),
                    setup_grade=t.get("setup_grade", ""),
                    vah=t.get("vah", 0),
                    poc=t.get("poc", 0),
                    val=t.get("val", 0),
                    rsi=t.get("rsi", 0),
                    fib_zone=t.get("fib_zone", "") or "",
                    fib_quality=t.get("fib_quality", "") or "",
                    fib_trend=t.get("fib_trend", "") or "",
                    fib_position=t.get("fib_position", "") or "",
                    status=status,
                    actual_entry=entry_price if status != "PLANNED" else None,
                    actual_exit=t.get("exit_price") or None,
                    exit_reason=t.get("exit_reason", ""),
                    pnl_dollars=t.get("exit_pnl") or t.get("pnl") or None,
                    pnl_percent=t.get("exit_pnl_pct") or None,
                    pnl_r=t.get("r_multiple") or None,
                    notes=t.get("notes", ""),
                    logged_at=t.get("created_at", ""),
                )
                
                journal.log_trade(entry)
                synced += 1
            except Exception as te:
                skipped += 1
                continue
        
        return JSONResponse(content={
            "synced": synced,
            "skipped": skipped,
            "total": len(trades)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@journal_router.get("/trades")
async def get_journal_trades(
    request: Request,
    user_id: str = Query("anonymous"),
    status: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    days: int = Query(30),
    limit: int = Query(100)
):
    """Get trades from the SQLite analytics journal."""
    try:
        journal = _get_journal()
        trades = journal.get_trades(
            user_id=user_id, status=status,
            symbol=symbol, days=days, limit=limit
        )
        return JSONResponse(content={"trades": trades, "count": len(trades)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
