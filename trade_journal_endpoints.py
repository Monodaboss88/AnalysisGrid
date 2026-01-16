"""
Trade Journal API Endpoints
============================
Log trades with one click, track performance
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from trade_journal import TradeJournal, JournalEntry


# Create router
journal_router = APIRouter(prefix="/api/journal", tags=["Trade Journal"])

# Initialize journal
journal = TradeJournal()


class LogTradeRequest(BaseModel):
    """Request to log a trade from scan results"""
    symbol: str
    direction: str = "LONG"  # LONG or SHORT
    timeframe: str = "4hr"
    
    # Entry levels
    entry_price: float
    stop_loss: float
    target1: float
    target2: float = 0
    
    # From scan
    signal: str = ""
    confidence: float = 0
    bull_score: int = 0
    bear_score: int = 0
    ai_commentary: str = ""
    setup_grade: str = ""
    
    # Price levels
    vah: float = 0
    poc: float = 0
    val: float = 0
    vwap: float = 0
    rsi: float = 0
    
    # Optional
    notes: str = ""
    tags: str = ""
    position_size: Optional[float] = None
    risk_amount: Optional[float] = None


class UpdateTradeRequest(BaseModel):
    """Request to update a trade"""
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    status: Optional[str] = None
    exit_reason: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[str] = None
    position_size: Optional[float] = None


@journal_router.post("/log")
async def log_trade(request: LogTradeRequest):
    """
    📝 Log a trade from scan results with ONE CLICK
    
    This captures the scan data and creates a journal entry.
    """
    # Calculate R:R
    if request.direction == "LONG":
        risk = request.entry_price - request.stop_loss
        reward1 = request.target1 - request.entry_price
        reward2 = request.target2 - request.entry_price if request.target2 else 0
    else:
        risk = request.stop_loss - request.entry_price
        reward1 = request.entry_price - request.target1
        reward2 = request.entry_price - request.target2 if request.target2 else 0
    
    rr_t1 = reward1 / risk if risk > 0 else 0
    rr_t2 = reward2 / risk if risk > 0 and request.target2 else 0
    
    entry = JournalEntry(
        id=None,
        symbol=request.symbol.upper(),
        direction=request.direction.upper(),
        timeframe=request.timeframe,
        entry_price=request.entry_price,
        stop_loss=request.stop_loss,
        target1=request.target1,
        target2=request.target2,
        risk_reward_t1=round(rr_t1, 2),
        risk_reward_t2=round(rr_t2, 2),
        position_size=request.position_size,
        risk_amount=request.risk_amount,
        signal=request.signal,
        confidence=request.confidence,
        bull_score=request.bull_score,
        bear_score=request.bear_score,
        ai_commentary=request.ai_commentary,
        setup_grade=request.setup_grade,
        vah=request.vah,
        poc=request.poc,
        val=request.val,
        vwap=request.vwap,
        rsi=request.rsi,
        notes=request.notes,
        tags=request.tags
    )
    
    entry_id = journal.log_trade(entry)
    
    return {
        "status": "logged",
        "id": entry_id,
        "symbol": request.symbol.upper(),
        "direction": request.direction.upper(),
        "entry": request.entry_price,
        "stop": request.stop_loss,
        "target1": request.target1,
        "target2": request.target2,
        "risk_reward_t1": round(rr_t1, 2),
        "risk_reward_t2": round(rr_t2, 2),
        "setup_grade": request.setup_grade,
        "message": f"Trade logged! ID: {entry_id}"
    }


@journal_router.post("/{entry_id}/open")
async def open_trade(
    entry_id: int,
    actual_entry: float = Query(..., description="Actual entry price")
):
    """Mark a planned trade as OPEN with actual entry price"""
    success = journal.open_trade(entry_id, actual_entry)
    
    if not success:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    return {
        "status": "opened",
        "id": entry_id,
        "actual_entry": actual_entry,
        "message": f"Trade #{entry_id} marked as OPEN"
    }


@journal_router.post("/{entry_id}/close")
async def close_trade(
    entry_id: int,
    actual_exit: float = Query(..., description="Actual exit price"),
    exit_reason: str = Query("manual", description="Reason: hit_target1, hit_target2, hit_stop, manual, time")
):
    """Close a trade and calculate P&L"""
    success = journal.close_trade(entry_id, actual_exit, exit_reason)
    
    if not success:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    # Get updated trade to show P&L
    trade = journal.get_trade(entry_id)
    
    return {
        "status": trade["status"],
        "id": entry_id,
        "actual_exit": actual_exit,
        "pnl_percent": trade.get("pnl_percent"),
        "pnl_r": trade.get("pnl_r"),
        "exit_reason": exit_reason,
        "message": f"Trade closed: {trade['pnl_r']}R ({trade['pnl_percent']}%)"
    }


@journal_router.post("/{entry_id}/cancel")
async def cancel_trade(
    entry_id: int,
    reason: str = Query("", description="Why cancelled")
):
    """Cancel a planned trade"""
    success = journal.cancel_trade(entry_id, reason)
    
    if not success:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    return {
        "status": "cancelled",
        "id": entry_id,
        "message": f"Trade #{entry_id} cancelled"
    }


@journal_router.get("/{entry_id}")
async def get_trade(entry_id: int):
    """Get a single trade entry"""
    trade = journal.get_trade(entry_id)
    
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    return trade


@journal_router.put("/{entry_id}")
async def update_trade(entry_id: int, request: UpdateTradeRequest):
    """Update trade details"""
    updates = {k: v for k, v in request.dict().items() if v is not None}
    
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    success = journal.update_trade(entry_id, updates)
    
    if not success:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    return {
        "status": "updated",
        "id": entry_id,
        "updates": updates
    }


@journal_router.get("/")
async def get_trades(
    status: Optional[str] = Query(None, description="Filter by status: PLANNED, OPEN, WIN, LOSS"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    days: int = Query(30, description="Days of history")
):
    """Get trade journal entries"""
    trades = journal.get_trades(status=status, symbol=symbol, days=days)
    
    return {
        "count": len(trades),
        "filters": {
            "status": status,
            "symbol": symbol,
            "days": days
        },
        "trades": trades
    }


@journal_router.get("/stats")
async def get_stats(days: int = Query(30, description="Days to analyze")):
    """
    📊 Get trading performance statistics
    
    Returns win rate, average R, expectancy, and breakdown by setup grade.
    """
    stats = journal.get_stats(days)
    
    return stats


@journal_router.get("/export")
async def export_journal(
    format: str = Query("json", description="Format: json or csv"),
    days: int = Query(365, description="Days of history")
):
    """Export journal for external analysis (Excel, Python, etc.)"""
    data = journal.export_journal(format, days)
    
    return {
        "format": format,
        "days": days,
        "data": data if format == "json" else None,
        "csv": data if format == "csv" else None
    }


@journal_router.get("/open")
async def get_open_trades():
    """Get all currently open trades"""
    trades = journal.get_trades(status="OPEN", days=365)
    
    return {
        "count": len(trades),
        "trades": trades
    }


@journal_router.get("/planned")
async def get_planned_trades():
    """Get all planned (not yet entered) trades"""
    trades = journal.get_trades(status="PLANNED", days=30)
    
    return {
        "count": len(trades),
        "trades": trades
    }
