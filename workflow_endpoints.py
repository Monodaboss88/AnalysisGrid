"""
Trading Workflow & Discipline System
=====================================
Endpoints for:
- Mental state tracking
- Pre-trade gate checks
- Daily trade limits
- Risk management rules

Author: Rob's Trading Systems
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import json
import os

# Firestore for persistence
try:
    from firestore_store import get_firestore
    firestore_available = True
except ImportError:
    firestore_available = False

workflow_router = APIRouter(prefix="/api/workflow", tags=["Workflow"])

# =============================================================================
# DATA MODELS
# =============================================================================

class MentalStateUpdate(BaseModel):
    state: str  # GREEN, YELLOW, RED
    answers: Dict[str, Any] = {}
    notes: Optional[str] = None

class PreTradeCheckRequest(BaseModel):
    symbol: str
    signal: str  # LONG, SHORT
    confidence: float = 50.0
    risk_r: float = 1.0

class GateResult(BaseModel):
    gate: str
    passed: bool
    message: str
    severity: str = "warning"  # info, warning, blocker

# =============================================================================
# MENTAL STATE QUESTIONS
# =============================================================================

MENTAL_CHECK_QUESTIONS = [
    {
        "id": "sleep",
        "question": "Did you get 7+ hours of quality sleep?",
        "red_if": False,
        "weight": 2
    },
    {
        "id": "revenge",
        "question": "Are you trying to recover losses from yesterday?",
        "red_if": True,
        "weight": 3
    },
    {
        "id": "fomo",
        "question": "Are you feeling FOMO about missing moves?",
        "red_if": True,
        "weight": 2
    },
    {
        "id": "distracted",
        "question": "Are you distracted or multitasking?",
        "red_if": True,
        "weight": 1
    },
    {
        "id": "plan",
        "question": "Do you have a clear plan with defined R:R?",
        "red_if": False,
        "weight": 2
    },
    {
        "id": "stressed",
        "question": "Are you stressed about money/life situations?",
        "red_if": True,
        "weight": 2
    },
    {
        "id": "forcing",
        "question": "Do you feel like you MUST take a trade today?",
        "red_if": True,
        "weight": 3
    },
    {
        "id": "prepared",
        "question": "Have you done your pre-market prep?",
        "red_if": False,
        "weight": 1
    }
]

# =============================================================================
# PRE-TRADE GATES
# =============================================================================

def check_daily_trade_limit(user_trades: List[Dict], max_trades: int = 4) -> GateResult:
    """Check if user has exceeded daily trade limit"""
    today = date.today().isoformat()
    today_trades = [t for t in user_trades if t.get('created_at', '').startswith(today)]
    count = len(today_trades)
    
    if count >= max_trades:
        return GateResult(
            gate="Daily Trade Limit",
            passed=False,
            message=f"Already taken {count}/{max_trades} trades today. STOP.",
            severity="blocker"
        )
    elif count >= max_trades - 1:
        return GateResult(
            gate="Daily Trade Limit",
            passed=True,
            message=f"This would be trade {count + 1}/{max_trades}. Last trade of the day!",
            severity="warning"
        )
    else:
        return GateResult(
            gate="Daily Trade Limit",
            passed=True,
            message=f"Trade {count + 1}/{max_trades} for today",
            severity="info"
        )

def check_consecutive_losses(user_trades: List[Dict], max_consecutive: int = 2) -> GateResult:
    """Check for consecutive losses - stop after 2 in a row"""
    today = date.today().isoformat()
    today_trades = [t for t in user_trades if t.get('created_at', '').startswith(today)]
    today_trades.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    consecutive_losses = 0
    for t in today_trades:
        pnl = t.get('pnl', 0) or t.get('r_multiple', 0)
        if pnl < 0:
            consecutive_losses += 1
        else:
            break
    
    if consecutive_losses >= max_consecutive:
        return GateResult(
            gate="Loss Streak Protection",
            passed=False,
            message=f"{consecutive_losses} consecutive losses. Take a break!",
            severity="blocker"
        )
    elif consecutive_losses == 1:
        return GateResult(
            gate="Loss Streak Protection",
            passed=True,
            message="1 loss in a row. Stay focused, next one counts.",
            severity="warning"
        )
    else:
        return GateResult(
            gate="Loss Streak Protection",
            passed=True,
            message="No consecutive losses",
            severity="info"
        )

def check_daily_drawdown(user_trades: List[Dict], max_daily_r: float = -3.0) -> GateResult:
    """Check if daily drawdown limit hit"""
    today = date.today().isoformat()
    today_trades = [t for t in user_trades if t.get('created_at', '').startswith(today)]
    
    total_r = sum(t.get('r_multiple', t.get('pnl', 0)) or 0 for t in today_trades)
    
    if total_r <= max_daily_r:
        return GateResult(
            gate="Daily Drawdown Limit",
            passed=False,
            message=f"Down {total_r:.1f}R today. Max loss reached. STOP TRADING.",
            severity="blocker"
        )
    elif total_r <= max_daily_r / 2:
        return GateResult(
            gate="Daily Drawdown Limit",
            passed=True,
            message=f"Down {total_r:.1f}R today. Approaching limit ({max_daily_r}R).",
            severity="warning"
        )
    else:
        pnl_str = f"+{total_r:.1f}" if total_r >= 0 else f"{total_r:.1f}"
        return GateResult(
            gate="Daily Drawdown Limit",
            passed=True,
            message=f"Daily P&L: {pnl_str}R",
            severity="info"
        )

def check_mental_state(mental_state: str) -> GateResult:
    """Check if mental state allows trading"""
    if mental_state == "RED":
        return GateResult(
            gate="Mental State",
            passed=False,
            message="Mental state is RED. Do not trade!",
            severity="blocker"
        )
    elif mental_state == "YELLOW":
        return GateResult(
            gate="Mental State",
            passed=True,
            message="Mental state is YELLOW. Trade with reduced size.",
            severity="warning"
        )
    else:
        return GateResult(
            gate="Mental State",
            passed=True,
            message="Mental state is GREEN. Clear to trade.",
            severity="info"
        )

def check_confidence(confidence: float, min_confidence: float = 60.0) -> GateResult:
    """Check if trade has sufficient confidence"""
    if confidence < min_confidence:
        return GateResult(
            gate="Setup Confidence",
            passed=False,
            message=f"Confidence {confidence:.0f}% is below threshold ({min_confidence:.0f}%). Find a better setup.",
            severity="blocker"
        )
    elif confidence < 70:
        return GateResult(
            gate="Setup Confidence",
            passed=True,
            message=f"Confidence {confidence:.0f}% - marginal. Consider reduced size.",
            severity="warning"
        )
    else:
        return GateResult(
            gate="Setup Confidence",
            passed=True,
            message=f"Confidence {confidence:.0f}% - good setup",
            severity="info"
        )

def check_time_of_day() -> GateResult:
    """Check if it's a good time to trade"""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    
    # Market hours: 9:30 AM - 4:00 PM ET
    # Best times: First hour (9:30-10:30) and last hour (3:00-4:00)
    # Choppy: 11:30 AM - 2:00 PM
    
    if hour < 9 or (hour == 9 and minute < 30):
        return GateResult(
            gate="Time of Day",
            passed=False,
            message="Pre-market. Wait for the open.",
            severity="warning"
        )
    elif hour >= 16:
        return GateResult(
            gate="Time of Day",
            passed=False,
            message="After hours. Market is closed.",
            severity="blocker"
        )
    elif 11 <= hour < 14:
        return GateResult(
            gate="Time of Day",
            passed=True,
            message="Lunch chop zone (11-2). Be extra selective.",
            severity="warning"
        )
    else:
        return GateResult(
            gate="Time of Day",
            passed=True,
            message="Good trading hours",
            severity="info"
        )

# =============================================================================
# ENDPOINTS
# =============================================================================

@workflow_router.get("/status")
async def get_workflow_status(uid: Optional[str] = Header(None)):
    """Get current workflow status for the day"""
    user_trades = []
    mental_state = "GREEN"
    mental_state_desc = "Clear, patient, process-focused"
    
    # Load user's trades if authenticated
    if uid and firestore_available:
        try:
            db = get_firestore()
            if db:
                # Get today's trades
                today = date.today().isoformat()
                trades_ref = db.collection('users').document(uid).collection('trades')
                docs = trades_ref.stream()
                for doc in docs:
                    trade = doc.to_dict()
                    if trade.get('created_at', '').startswith(today):
                        user_trades.append(trade)
                
                # Get mental state
                state_ref = db.collection('users').document(uid).collection('workflow').document('mental_state')
                state_doc = state_ref.get()
                if state_doc.exists:
                    state_data = state_doc.to_dict()
                    if state_data.get('date') == today:
                        mental_state = state_data.get('state', 'GREEN')
                        mental_state_desc = state_data.get('description', '')
        except Exception as e:
            print(f"Workflow status error: {e}")
    
    # Calculate stats
    total_r = sum(t.get('r_multiple', t.get('pnl', 0)) or 0 for t in user_trades)
    
    # Count consecutive losses
    sorted_trades = sorted(user_trades, key=lambda x: x.get('created_at', ''), reverse=True)
    consecutive_losses = 0
    for t in sorted_trades:
        pnl = t.get('pnl', 0) or t.get('r_multiple', 0)
        if pnl < 0:
            consecutive_losses += 1
        else:
            break
    
    return {
        "trades": len(user_trades),
        "max_trades": 4,
        "total_r": total_r,
        "consecutive_losses": consecutive_losses,
        "mental_state": mental_state,
        "mental_state_description": mental_state_desc,
        "date": date.today().isoformat()
    }

@workflow_router.get("/mental-check/questions")
async def get_mental_check_questions():
    """Get the mental state check questions"""
    return {
        "questions": MENTAL_CHECK_QUESTIONS
    }

@workflow_router.post("/mental-check")
async def submit_mental_check(data: MentalStateUpdate, uid: Optional[str] = Header(None)):
    """Submit mental state check answers"""
    today = date.today().isoformat()
    
    # Calculate mental state from answers if not explicitly provided
    state = data.state
    description = ""
    
    if data.answers:
        score = 0
        max_score = sum(q['weight'] for q in MENTAL_CHECK_QUESTIONS)
        
        for q in MENTAL_CHECK_QUESTIONS:
            answer = data.answers.get(q['id'])
            if answer is not None:
                # If answer matches red_if condition, subtract points
                if answer == q['red_if']:
                    score -= q['weight']
                else:
                    score += q['weight']
        
        # Convert score to state
        pct = (score + max_score) / (2 * max_score) * 100
        if pct >= 70:
            state = "GREEN"
            description = "Clear, patient, process-focused"
        elif pct >= 40:
            state = "YELLOW"
            description = "Caution - some concerns. Reduce size."
        else:
            state = "RED"
            description = "Do not trade today. Step away."
    
    # Save to Firestore if authenticated
    if uid and firestore_available:
        try:
            db = get_firestore()
            if db:
                state_ref = db.collection('users').document(uid).collection('workflow').document('mental_state')
                state_ref.set({
                    'date': today,
                    'state': state,
                    'description': description,
                    'answers': data.answers,
                    'notes': data.notes,
                    'updated_at': datetime.now().isoformat()
                })
        except Exception as e:
            print(f"Error saving mental state: {e}")
    
    return {
        "state": state,
        "description": description,
        "date": today
    }

@workflow_router.get("/pre-trade-check")
async def pre_trade_check(
    signal: str = "LONG",
    confidence: float = 50.0,
    uid: Optional[str] = Header(None)
):
    """Run all pre-trade gate checks"""
    user_trades = []
    mental_state = "GREEN"
    
    # Load user data if authenticated
    if uid and firestore_available:
        try:
            db = get_firestore()
            if db:
                today = date.today().isoformat()
                
                # Get trades
                trades_ref = db.collection('users').document(uid).collection('trades')
                docs = trades_ref.stream()
                for doc in docs:
                    user_trades.append(doc.to_dict())
                
                # Get mental state
                state_ref = db.collection('users').document(uid).collection('workflow').document('mental_state')
                state_doc = state_ref.get()
                if state_doc.exists:
                    state_data = state_doc.to_dict()
                    if state_data.get('date') == today:
                        mental_state = state_data.get('state', 'GREEN')
        except Exception as e:
            print(f"Pre-trade check error: {e}")
    
    # Run all gate checks
    results = [
        check_mental_state(mental_state),
        check_daily_trade_limit(user_trades),
        check_consecutive_losses(user_trades),
        check_daily_drawdown(user_trades),
        check_confidence(confidence),
        check_time_of_day()
    ]
    
    # Determine if trade is approved
    blockers = [r for r in results if not r.passed and r.severity == "blocker"]
    warnings = [r for r in results if r.severity == "warning"]
    
    approved = len(blockers) == 0
    
    # Position size multiplier (reduce on warnings)
    position_size = 1.0
    if mental_state == "YELLOW":
        position_size *= 0.5
    if len(warnings) >= 2:
        position_size *= 0.75
    
    return {
        "approved": approved,
        "position_size_multiplier": position_size,
        "blockers": len(blockers),
        "warnings": len(warnings),
        "results": [r.dict() for r in results],
        "signal": signal,
        "confidence": confidence
    }

@workflow_router.post("/reset-daily")
async def reset_daily_state(uid: Optional[str] = Header(None)):
    """Reset the daily workflow state (for testing or new day)"""
    if uid and firestore_available:
        try:
            db = get_firestore()
            if db:
                state_ref = db.collection('users').document(uid).collection('workflow').document('mental_state')
                state_ref.delete()
                return {"status": "reset", "message": "Daily workflow state cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    return {"status": "ok", "message": "No state to reset (not authenticated)"}
