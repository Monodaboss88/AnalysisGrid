"""
Trade Rule Engine API Endpoints
================================
API endpoints for the deterministic trade rule engine.
Generates trade plans from scanner data and tracks outcomes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json

from trade_rule_engine import (
    RuleEngine, 
    TradingRules, 
    TradePlan,
    AIExplainer,
    LearningDatabase,
    ReportKnowledgeBase,
    generate_plan,
    record_trade_outcome,
    get_learning_stats
)


# Router
rule_router = APIRouter(tags=["Rule Engine"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ScannerData(BaseModel):
    """Scanner result data"""
    symbol: str
    current_price: Optional[float] = None
    price: Optional[float] = None
    vah: float
    poc: float
    val: float
    vwap: Optional[float] = None
    bull_score: float = 0
    bear_score: float = 0
    rsi: float = 50
    rvol: float = 1.0
    confidence: float = 50
    direction: Optional[str] = None


class TradePlanResponse(BaseModel):
    """Trade plan response"""
    plan_id: int
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_per_share: float
    risk_reward_t1: float
    risk_reward_t2: float
    position_size_pct: float
    risk_pct: float
    entry_reasons: List[str]
    caution_flags: List[str]
    invalidation: str
    explanation: str
    formatted_plan: str


class OutcomeRequest(BaseModel):
    """Record trade outcome"""
    plan_id: int
    outcome: str  # 'WIN_T1', 'WIN_T2', 'WIN_T3', 'LOSS', 'SCRATCH', 'SKIPPED'
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    actual_r: Optional[float] = None
    exit_reason: Optional[str] = None
    notes: Optional[str] = None


class RulesUpdate(BaseModel):
    """Update trading rules"""
    MIN_SCORE_FULL_SIZE: Optional[int] = None
    MIN_SCORE_HALF_SIZE: Optional[int] = None
    MIN_SCORE_NO_TRADE: Optional[int] = None
    LONG_STOP_BELOW_VAL_PCT: Optional[float] = None
    SHORT_STOP_ABOVE_VAH_PCT: Optional[float] = None
    MAX_STOP_DISTANCE_PCT: Optional[float] = None
    T2_R_MULTIPLE: Optional[float] = None
    T3_R_MULTIPLE: Optional[float] = None
    BASE_RISK_PCT: Optional[float] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@rule_router.post("/generate", response_model=TradePlanResponse)
async def generate_trade_plan(data: ScannerData, explain: bool = True, save: bool = True):
    """
    Generate a trade plan from scanner data.
    
    The rule engine applies YOUR deterministic rules to generate exact levels.
    AI only explains the reasoning - it cannot change the levels.
    """
    try:
        # Convert to dict
        scanner_dict = data.dict()
        
        # Generate plan
        plan, explanation, plan_id = generate_plan(scanner_dict, explain=explain, save=save)
        
        # Format text version
        engine = RuleEngine()
        formatted = engine.format_plan_text(plan)
        
        return TradePlanResponse(
            plan_id=plan_id,
            symbol=plan.symbol,
            direction=plan.direction,
            confidence=plan.confidence,
            entry_price=plan.entry_price,
            entry_zone_low=plan.entry_zone_low,
            entry_zone_high=plan.entry_zone_high,
            stop_loss=plan.stop_loss,
            target_1=plan.target_1,
            target_2=plan.target_2,
            target_3=plan.target_3,
            risk_per_share=plan.risk_per_share,
            risk_reward_t1=plan.risk_reward_t1,
            risk_reward_t2=plan.risk_reward_t2,
            position_size_pct=plan.position_size_pct,
            risk_pct=plan.risk_pct,
            entry_reasons=plan.entry_reasons,
            caution_flags=plan.caution_flags,
            invalidation=plan.invalidation,
            explanation=explanation or "Rule-based plan generated.",
            formatted_plan=formatted
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.post("/outcome")
async def log_trade_outcome(outcome: OutcomeRequest):
    """
    Record the outcome of a trade.
    This feeds back into the learning system.
    """
    try:
        record_trade_outcome(
            plan_id=outcome.plan_id,
            outcome=outcome.outcome,
            actual_r=outcome.actual_r,
            notes=outcome.notes
        )
        
        return {"status": "success", "message": f"Outcome recorded for plan #{outcome.plan_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/stats")
async def get_stats():
    """
    Get learning statistics.
    Shows which patterns are working and overall performance.
    """
    try:
        stats = get_learning_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/rules")
async def get_current_rules():
    """
    Get current trading rules.
    """
    rules = TradingRules()
    return {
        "entry": {
            "MIN_SCORE_FULL_SIZE": rules.MIN_SCORE_FULL_SIZE,
            "MIN_SCORE_HALF_SIZE": rules.MIN_SCORE_HALF_SIZE,
            "MIN_SCORE_NO_TRADE": rules.MIN_SCORE_NO_TRADE,
            "LONG_REQUIRES_ABOVE_POC": rules.LONG_REQUIRES_ABOVE_POC,
            "SHORT_REQUIRES_BELOW_POC": rules.SHORT_REQUIRES_BELOW_POC,
            "MIN_RVOL_FOR_ENTRY": rules.MIN_RVOL_FOR_ENTRY,
        },
        "stops": {
            "LONG_STOP_BELOW_VAL_PCT": rules.LONG_STOP_BELOW_VAL_PCT,
            "SHORT_STOP_ABOVE_VAH_PCT": rules.SHORT_STOP_ABOVE_VAH_PCT,
            "MAX_STOP_DISTANCE_PCT": rules.MAX_STOP_DISTANCE_PCT,
        },
        "targets": {
            "T1_AT_OPPOSITE_VA": rules.T1_AT_OPPOSITE_VA,
            "T2_R_MULTIPLE": rules.T2_R_MULTIPLE,
            "T3_R_MULTIPLE": rules.T3_R_MULTIPLE,
        },
        "sizing": {
            "BASE_RISK_PCT": rules.BASE_RISK_PCT,
            "HIGH_SCORE_RISK_MULT": rules.HIGH_SCORE_RISK_MULT,
            "MED_SCORE_RISK_MULT": rules.MED_SCORE_RISK_MULT,
        },
        "filters": {
            "RSI_OVERBOUGHT": rules.RSI_OVERBOUGHT,
            "RSI_OVERSOLD": rules.RSI_OVERSOLD,
            "AVOID_FIRST_15_MIN": rules.AVOID_FIRST_15_MIN,
            "AVOID_LAST_30_MIN": rules.AVOID_LAST_30_MIN,
        }
    }


@rule_router.get("/recent")
async def get_recent_plans(limit: int = 20):
    """
    Get recent trade plans with outcomes.
    """
    try:
        db = LearningDatabase()
        plans = db.get_recent_plans(limit)
        return {"plans": plans}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/patterns")
async def get_pattern_performance():
    """
    Get performance stats by pattern.
    Shows which setups are working best.
    """
    try:
        db = LearningDatabase()
        patterns = db.get_pattern_stats()
        return {"patterns": patterns}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/knowledge")
async def get_knowledge_base():
    """
    Get what the AI has learned from your reports.
    Shows loaded reports and extracted patterns.
    """
    try:
        kb = ReportKnowledgeBase()
        
        return {
            "reports_loaded": len(kb.reports),
            "symbols_analyzed": [r['symbol'] for r in kb.reports],
            "reports": [
                {
                    "filename": r['filename'],
                    "symbol": r['symbol'],
                    "date": r['date'],
                    "signal": r['signal'],
                    "key_levels": r['key_levels'],
                    "setups": r['setups'],
                    "risks": r['risks'][:3]
                }
                for r in kb.reports
            ],
            "analysis_style": kb.get_analysis_style_prompt()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/knowledge/{symbol}")
async def get_symbol_knowledge(symbol: str):
    """
    Get knowledge about a specific symbol from reports.
    """
    try:
        kb = ReportKnowledgeBase()
        report = kb.get_context_for_symbol(symbol)
        
        if not report:
            return {"found": False, "message": f"No report found for {symbol}"}
        
        return {
            "found": True,
            "symbol": report['symbol'],
            "date": report['date'],
            "signal": report['signal'],
            "key_levels": report['key_levels'],
            "setups": report['setups'],
            "risks": report['risks'],
            "summary": kb.get_report_summary(symbol)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
