"""
AI Advisor Server Integration
=============================
Add these endpoints to your unified_server.py to enable hedge-fund level AI analysis.

Usage:
1. Copy this file to your project directory
2. Import and register the router in unified_server.py:
   
   from ai_advisor_endpoints import ai_router
   app.include_router(ai_router, prefix="/api/ai")

3. Set environment variables:
   export OPENAI_API_KEY=your_key_here
   # OR
   export ANTHROPIC_API_KEY=your_key_here

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Import our AI advisor
from ai_trading_advisor import (
    HedgeFundAdvisor, TradeRecord, MarketRegime, 
    RegimeDetector, AICommentaryEngine
)

# Try to import existing modules
try:
    from finnhub_scanner_v2 import FinnhubScanner
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RegimeRequest(BaseModel):
    """Market regime analysis request"""
    spy_price: float
    spy_sma_20: float
    spy_sma_50: float
    spy_sma_200: float
    spy_atr_20: float = 8.0
    spy_high_20: float = None
    spy_low_20: float = None
    vix: float


class FullAnalysisRequest(BaseModel):
    """Complete analysis request"""
    symbol: str
    # Scanner data (from your MTF scanner)
    signal: str
    confidence: float
    confluence: float = 0
    bull_score: float
    bear_score: float
    price: float
    vah: float
    poc: float
    val: float
    vwap: float
    position: str
    vwap_zone: str
    rsi: float
    rsi_zone: str
    flow_imbalance: float = 0
    flow_state: str = "BALANCED"
    notes: List[str] = []
    # Market context
    spy_price: float
    spy_sma_20: float
    spy_sma_50: float
    spy_sma_200: float
    vix: float
    # Optional
    news_headlines: List[str] = []
    earnings_days: int = -1


class TradeEntryRequest(BaseModel):
    """Trade journal entry request"""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    stop_price: float
    target_1: float
    target_2: Optional[float] = None
    signal_at_entry: str
    confidence_at_entry: float
    regime_at_entry: str
    scanner_notes: List[str] = []
    ai_commentary: str = ""


class TradeExitRequest(BaseModel):
    """Trade exit logging request"""
    trade_id: str
    exit_price: float
    exit_reason: str  # TARGET_1, TARGET_2, STOP, TIME, MANUAL


# =============================================================================
# ROUTER
# =============================================================================

ai_router = APIRouter(tags=["AI Advisor"])

# Global advisor instance (initialized on first use)
_advisor: Optional[HedgeFundAdvisor] = None


def get_advisor() -> HedgeFundAdvisor:
    """Get or create the AI advisor instance"""
    global _advisor
    if _advisor is None:
        # Determine provider based on available API keys
        if os.environ.get('OPENAI_API_KEY'):
            provider = 'openai'
        elif os.environ.get('ANTHROPIC_API_KEY'):
            provider = 'anthropic'
        else:
            raise HTTPException(
                status_code=400,
                detail="No AI API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )
        _advisor = HedgeFundAdvisor(provider=provider)
    return _advisor


# =============================================================================
# ENDPOINTS
# =============================================================================

@ai_router.get("/status")
async def ai_status():
    """Check AI advisor status"""
    has_openai = bool(os.environ.get('OPENAI_API_KEY'))
    has_anthropic = bool(os.environ.get('ANTHROPIC_API_KEY'))
    
    return {
        "openai_available": has_openai,
        "anthropic_available": has_anthropic,
        "active_provider": "openai" if has_openai else ("anthropic" if has_anthropic else None),
        "features": {
            "regime_detection": True,
            "ai_commentary": has_openai or has_anthropic,
            "trade_journaling": True,
            "news_analysis": has_openai or has_anthropic,
            "performance_tracking": True
        }
    }


@ai_router.post("/regime")
async def analyze_regime(request: RegimeRequest):
    """
    Analyze current market regime.
    
    Call this before individual stock analysis to get market context.
    """
    detector = RegimeDetector()
    
    spy_data = {
        'price': request.spy_price,
        'sma_20': request.spy_sma_20,
        'sma_50': request.spy_sma_50,
        'sma_200': request.spy_sma_200,
        'atr_20': request.spy_atr_20,
        'high_20': request.spy_high_20 or request.spy_price * 1.03,
        'low_20': request.spy_low_20 or request.spy_price * 0.97
    }
    
    result = detector.analyze(spy_data, request.vix)
    
    return {
        "regime": result.regime.value,
        "strategy": result.regime.strategy_notes,
        "confidence": result.confidence,
        "vix_level": result.vix_level,
        "trend_strength": result.trend_strength,
        "range_bound_score": result.range_bound_score,
        "notes": result.notes
    }


@ai_router.post("/analyze")
async def full_analysis(request: FullAnalysisRequest):
    """
    Get complete hedge-fund level analysis of a setup.
    
    Combines:
    - Scanner signal interpretation
    - Market regime context
    - Historical win rate on similar setups
    - AI commentary
    - News sentiment (if provided)
    """
    advisor = get_advisor()
    
    scanner_result = {
        'signal': request.signal,
        'confidence': request.confidence,
        'confluence': request.confluence,
        'bull_score': request.bull_score,
        'bear_score': request.bear_score,
        'price': request.price,
        'vah': request.vah,
        'poc': request.poc,
        'val': request.val,
        'vwap': request.vwap,
        'position': request.position,
        'vwap_zone': request.vwap_zone,
        'rsi': request.rsi,
        'rsi_zone': request.rsi_zone,
        'flow_imbalance': request.flow_imbalance,
        'flow_state': request.flow_state,
        'notes': request.notes
    }
    
    spy_data = {
        'price': request.spy_price,
        'sma_20': request.spy_sma_20,
        'sma_50': request.spy_sma_50,
        'sma_200': request.spy_sma_200,
        'atr_20': 8.0,  # Default
        'high_20': request.spy_price * 1.03,
        'low_20': request.spy_price * 0.97
    }
    
    result = advisor.analyze_setup(
        symbol=request.symbol,
        scanner_result=scanner_result,
        spy_data=spy_data,
        vix=request.vix,
        news_headlines=request.news_headlines,
        earnings_days=request.earnings_days
    )
    
    return result


@ai_router.post("/quick-commentary")
async def quick_commentary(
    symbol: str,
    signal: str,
    confidence: float,
    position: str,
    rsi: float,
    vix: float = Query(16, description="Current VIX level")
):
    """
    Get quick AI commentary without full analysis.
    
    Use this for rapid signal interpretation.
    """
    advisor = get_advisor()
    
    # Minimal scanner result
    scanner_result = {
        'signal': signal,
        'confidence': confidence,
        'position': position,
        'rsi': rsi,
        'rsi_zone': 'OVERSOLD' if rsi < 30 else ('OVERBOUGHT' if rsi > 70 else 'NEUTRAL'),
        'notes': []
    }
    
    # Get regime for context
    regime = advisor.regime_detector.analyze(
        {'price': 590, 'sma_20': 585, 'sma_50': 580, 'sma_200': 550},
        vix
    )
    
    from ai_trading_advisor import TradeContext
    
    context = TradeContext(
        symbol=symbol,
        scanner_result=scanner_result,
        regime=regime,
        recent_trades=[],
        win_rate=0.55,
        avg_r_multiple=1.5,
        news_headlines=[],
        earnings_days=-1
    )
    
    result = advisor.ai_engine.analyze(context)
    
    return {
        'symbol': symbol,
        'signal': signal,
        'regime': regime.regime.value,
        'commentary': result.get('commentary', ''),
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# TRADE JOURNAL ENDPOINTS
# =============================================================================

@ai_router.post("/journal/entry")
async def log_trade_entry(request: TradeEntryRequest):
    """Log a new trade entry"""
    import uuid
    advisor = get_advisor()
    
    trade = TradeRecord(
        id=str(uuid.uuid4())[:8],
        symbol=request.symbol,
        direction=request.direction,
        entry_date=datetime.now(),
        entry_price=request.entry_price,
        stop_price=request.stop_price,
        target_1=request.target_1,
        target_2=request.target_2,
        signal_at_entry=request.signal_at_entry,
        confidence_at_entry=request.confidence_at_entry,
        regime_at_entry=request.regime_at_entry,
        scanner_notes=request.scanner_notes,
        ai_commentary=request.ai_commentary
    )
    
    trade_id = advisor.journal.log_entry(trade)
    
    return {
        'trade_id': trade_id,
        'symbol': request.symbol,
        'direction': request.direction,
        'entry_price': request.entry_price,
        'status': 'OPEN',
        'timestamp': datetime.now().isoformat()
    }


@ai_router.post("/journal/exit")
async def log_trade_exit(request: TradeExitRequest):
    """Log a trade exit and get R-multiple"""
    advisor = get_advisor()
    
    result = advisor.journal.log_exit(
        trade_id=request.trade_id,
        exit_price=request.exit_price,
        exit_reason=request.exit_reason
    )
    
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return result


@ai_router.get("/journal/trades")
async def get_trades(
    symbol: str = None,
    signal: str = None,
    regime: str = None,
    limit: int = Query(20, le=100)
):
    """Get historical trades with optional filters"""
    advisor = get_advisor()
    trades = advisor.journal.get_similar_trades(
        symbol=symbol,
        signal=signal,
        regime=regime,
        limit=limit
    )
    return {'trades': trades, 'count': len(trades)}


@ai_router.get("/journal/stats")
async def get_performance_stats(days: int = Query(30, le=365)):
    """Get performance statistics"""
    advisor = get_advisor()
    return advisor.journal.get_performance_summary(days)


@ai_router.get("/journal/win-rate")
async def get_win_rate(
    signal: str = None,
    regime: str = None
):
    """Get win rate for specific signal/regime combination"""
    advisor = get_advisor()
    return advisor.journal.get_win_rate(signal=signal, regime=regime)


@ai_router.post("/journal/{trade_id}/review")
async def generate_trade_review(trade_id: str):
    """Generate AI review of a completed trade"""
    advisor = get_advisor()
    review = advisor.journal.generate_ai_review(trade_id)
    return {'trade_id': trade_id, 'review': review}


# =============================================================================
# NEWS ANALYSIS
# =============================================================================

@ai_router.post("/news/analyze")
async def analyze_news(
    symbol: str,
    headlines: List[str]
):
    """
    Analyze news headlines for trading impact.
    
    Provide recent headlines and get sentiment + trading implications.
    """
    advisor = get_advisor()
    result = advisor.news_analyzer.analyze_headlines(symbol, headlines)
    return result


# =============================================================================
# INTEGRATION EXAMPLES
# =============================================================================

@ai_router.get("/examples")
async def get_integration_examples():
    """Get example requests for testing the AI advisor"""
    return {
        "regime_example": {
            "endpoint": "POST /api/ai/regime",
            "body": {
                "spy_price": 595.50,
                "spy_sma_20": 590.00,
                "spy_sma_50": 585.00,
                "spy_sma_200": 550.00,
                "vix": 16.5
            }
        },
        "full_analysis_example": {
            "endpoint": "POST /api/ai/analyze",
            "body": {
                "symbol": "META",
                "signal": "LONG_SETUP",
                "confidence": 72,
                "confluence": 75,
                "bull_score": 68,
                "bear_score": 32,
                "price": 619.28,
                "vah": 667.72,
                "poc": 660.40,
                "val": 647.22,
                "vwap": 619.63,
                "position": "BELOW_VALUE",
                "vwap_zone": "AT_VWAP",
                "rsi": 33.58,
                "rsi_zone": "NEAR_OVERSOLD",
                "flow_imbalance": 0.15,
                "flow_state": "MILD_BUY_FLOW",
                "notes": ["Price below VAL", "RSI oversold"],
                "spy_price": 595.50,
                "spy_sma_20": 590.00,
                "spy_sma_50": 585.00,
                "spy_sma_200": 550.00,
                "vix": 16.5,
                "earnings_days": -5
            }
        },
        "quick_commentary_example": {
            "endpoint": "GET /api/ai/quick-commentary",
            "params": {
                "symbol": "NVDA",
                "signal": "YELLOW",
                "confidence": 55,
                "position": "IN_VALUE",
                "rsi": 48,
                "vix": 18
            }
        },
        "trade_entry_example": {
            "endpoint": "POST /api/ai/journal/entry",
            "body": {
                "symbol": "META",
                "direction": "LONG",
                "entry_price": 620.00,
                "stop_price": 605.00,
                "target_1": 647.00,
                "target_2": 668.00,
                "signal_at_entry": "LONG_SETUP",
                "confidence_at_entry": 72,
                "regime_at_entry": "TRENDING_BULL"
            }
        }
    }
