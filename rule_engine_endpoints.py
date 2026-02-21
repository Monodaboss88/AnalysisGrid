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
import httpx
import os

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

from auto_report_generator import (
    AutoReportGenerator,
    auto_generate_report,
    generate_scan_summary
)

from firestore_store import get_firestore
from earnings_calendar import EarningsCalendar
from polygon_options import fetch_options_snapshot_filtered, parse_contract, group_by_expiration

# Global earnings calendar instance
_earnings_calendar = None

def get_earnings_calendar():
    global _earnings_calendar
    if _earnings_calendar is None:
        _earnings_calendar = EarningsCalendar()
    return _earnings_calendar


async def fetch_options_for_plan(symbol: str, scan_type: str = None, timeframe: str = None, confidence: float = 50, avg_iv_hint: float = None) -> Optional[Dict]:
    """Fetch options data for rule engine integration via Polygon.io (unlimited).
    Smart DTE selection based on scan type, timeframe, confidence, and IV.
    
    Returns data in the format the rule engine expects:
    { 'data': [{ 'pc_ratio', 'max_call_oi_strike', 'max_put_oi_strike', 
                  'avg_call_iv', 'avg_put_iv', 'expiration' }],
      'flat': { 'pc_ratio', 'call_wall', 'put_wall', 'avg_iv', 'dte', ... } }
    """
    # === DETERMINE IDEAL DTE RANGE based on trade context ===
    dte_ranges = {
        'squeeze':      (7, 21),
        'capitulation': (14, 30),
        'extension':    (14, 30),
        'entry':        (14, 35),
        'bullish':      (21, 45),
        'bearish':      (21, 45),
        'highVolume':   (14, 30),
        'atLevels':     (21, 45),
        'manual':       (21, 45),
    }
    min_dte, max_dte = dte_ranges.get(scan_type or '', (14, 35))
    
    tf_adjustments = {
        '5MIN':  (-10, -15),
        '15MIN': (-7, -10),
        '30MIN': (-3, -5),
        '1HR':   (0, 0),
        '2HR':   (5, 10),
        '4HR':   (10, 15),
    }
    tf_adj = tf_adjustments.get(timeframe or '1HR', (0, 0))
    min_dte = max(3, min_dte + tf_adj[0])
    max_dte = max(min_dte + 7, max_dte + tf_adj[1])
    
    if confidence >= 75:
        min_dte = max(3, min_dte - 5)
        max_dte = max(min_dte + 5, max_dte - 7)
    elif confidence < 40:
        min_dte += 7
        max_dte += 14
    
    print(f"Options DTE target for {symbol}: {min_dte}-{max_dte}d (scan={scan_type}, tf={timeframe}, conf={confidence:.0f})")
    
    # Build reason string for frontend
    dte_reason_parts = []
    if scan_type:
        type_labels = {
            'squeeze': 'Squeeze (fast move expected)',
            'capitulation': 'Capitulation (mean reversion)',
            'extension': 'Extension (fade/revert)',
            'entry': 'Entry signal',
            'bullish': 'Bullish trend',
            'bearish': 'Bearish trend',
            'highVolume': 'High volume momentum',
            'atLevels': 'At key level',
            'manual': 'Manual analysis'
        }
        dte_reason_parts.append(type_labels.get(scan_type, scan_type))
    if timeframe and timeframe != '1HR':
        tf_labels = {'5MIN': '5m scalp', '15MIN': '15m swing', '30MIN': '30m swing', '2HR': '2h position', '4HR': '4h position'}
        if timeframe in tf_labels:
            dte_reason_parts.append(tf_labels[timeframe])
    if confidence >= 75:
        dte_reason_parts.append('high confidence -> shorter DTE')
    elif confidence < 40:
        dte_reason_parts.append('low confidence -> longer DTE for time')
    dte_reason = ' | '.join(dte_reason_parts) if dte_reason_parts else None
    
    try:
        import asyncio
        # Fetch from Polygon (unlimited API calls)
        loop = asyncio.get_running_loop()
        snapshot = await loop.run_in_executor(
            None, 
            lambda: fetch_options_snapshot_filtered(symbol, dte_min=0, dte_max=max(max_dte + 14, 60), strike_range_pct=0.15)
        )
        
        raw_contracts = snapshot.get("contracts", [])
        if not raw_contracts:
            print(f"No Polygon options data for {symbol}")
            return None
        
        # Parse all contracts
        parsed = [parse_contract(c) for c in raw_contracts]
        
        # Group by expiration
        by_exp = group_by_expiration(parsed)
        
        from datetime import datetime as dt
        today = dt.now()
        best_exp = None
        best_dte = 0
        all_exp_data = []
        
        for exp_date_str, contracts in sorted(by_exp.items()):
            try:
                exp_dt = dt.strptime(exp_date_str, '%Y-%m-%d')
                dte = max(1, (exp_dt - today).days)
            except:
                dte = 7
            
            calls = [c for c in contracts if c.get("contractType") == "call"]
            puts = [c for c in contracts if c.get("contractType") == "put"]
            
            if not calls and not puts:
                continue
            
            # P/C ratio by volume
            total_call_vol = sum(c.get("dayVolume", 0) or 0 for c in calls)
            total_put_vol = sum(p.get("dayVolume", 0) or 0 for p in puts)
            pc_ratio = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 1.0
            
            # Call/Put walls (max OI strikes) + OI magnitude at wall
            call_wall_contract = max(calls, key=lambda x: x.get("openInterest", 0) or 0) if calls else {}
            put_wall_contract = max(puts, key=lambda x: x.get("openInterest", 0) or 0) if puts else {}
            call_wall = call_wall_contract.get("strike") if call_wall_contract else None
            put_wall = put_wall_contract.get("strike") if put_wall_contract else None
            call_wall_oi = call_wall_contract.get("openInterest", 0) or 0
            put_wall_oi = put_wall_contract.get("openInterest", 0) or 0
            
            # Average IV from Polygon (already in decimal 0-1)
            call_ivs = [c.get("iv") for c in calls if c.get("iv") and c.get("iv") > 0]
            put_ivs = [p.get("iv") for p in puts if p.get("iv") and p.get("iv") > 0]
            avg_call_iv_raw = sum(call_ivs) / max(len(call_ivs), 1) if call_ivs else 0
            avg_put_iv_raw = sum(put_ivs) / max(len(put_ivs), 1) if put_ivs else 0
            avg_iv_pct = round((avg_call_iv_raw + avg_put_iv_raw) / 2 * 100, 1)
            
            # Total OI
            total_call_oi = sum(c.get("openInterest", 0) or 0 for c in calls)
            total_put_oi = sum(p.get("openInterest", 0) or 0 for p in puts)
            
            # Unusual activity (vol/OI > 2x)
            unusual_calls = [c for c in calls if (c.get("dayVolume") or 0) > 0 and (c.get("openInterest") or 1) > 0 and (c.get("dayVolume") or 0) / (c.get("openInterest") or 1) > 2.0]
            unusual_puts = [p for p in puts if (p.get("dayVolume") or 0) > 0 and (p.get("openInterest") or 1) > 0 and (p.get("dayVolume") or 0) / (p.get("openInterest") or 1) > 2.0]
            
            exp_entry = {
                "expiration": exp_date_str,
                "dte": dte,
                "pc_ratio": pc_ratio,
                "max_call_oi_strike": call_wall,
                "max_put_oi_strike": put_wall,
                "avg_call_iv": avg_call_iv_raw,
                "avg_put_iv": avg_put_iv_raw,
                "avg_iv_pct": avg_iv_pct,
                "total_call_volume": total_call_vol,
                "total_put_volume": total_put_vol,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "call_wall_oi": call_wall_oi,
                "put_wall_oi": put_wall_oi,
                "unusual_call_count": len(unusual_calls),
                "unusual_put_count": len(unusual_puts),
            }
            all_exp_data.append(exp_entry)
            
            # Pick best: prefer within computed DTE range with decent volume
            if min_dte <= dte <= max_dte and (total_call_vol + total_put_vol) > 0:
                if best_dte == 0 or abs(dte - (min_dte + max_dte) / 2) < abs(best_dte - (min_dte + max_dte) / 2):
                    best_exp = exp_date_str
                    best_dte = dte
        
        if not all_exp_data:
            return None
        
        if best_dte == 0:
            best_exp = all_exp_data[0]["expiration"]
            best_dte = all_exp_data[0]["dte"]
        
        best_entry = next((e for e in all_exp_data if e["expiration"] == best_exp), all_exp_data[0])
        
        # Compute flow score from volume ratios
        total_call_vol_all = sum(e.get("total_call_volume", 0) for e in all_exp_data)
        total_put_vol_all = sum(e.get("total_put_volume", 0) for e in all_exp_data)
        flow_score = 0
        if total_call_vol_all + total_put_vol_all > 0:
            call_pct = total_call_vol_all / (total_call_vol_all + total_put_vol_all)
            flow_score = round((call_pct - 0.5) * 200)  # -100 to +100 scale
        
        total_unusual = sum(e.get("unusual_call_count", 0) + e.get("unusual_put_count", 0) for e in all_exp_data)
        
        print(f"Polygon options for {symbol}: {len(all_exp_data)} exps, best={best_exp} ({best_dte}d), IV={best_entry['avg_iv_pct']}%, P/C={best_entry['pc_ratio']}, flow={flow_score}, unusual={total_unusual}")
        
        return {
            "data": all_exp_data,
            "flat": {
                "pc_ratio": best_entry["pc_ratio"],
                "call_wall": best_entry["max_call_oi_strike"],
                "put_wall": best_entry["max_put_oi_strike"],
                "avg_iv": best_entry["avg_iv_pct"],
                "total_call_volume": best_entry["total_call_volume"],
                "total_put_volume": best_entry["total_put_volume"],
                "expiration": best_exp,
                "dte": best_dte,
                "expirations_available": [e["expiration"] for e in all_exp_data],
                "total_call_oi": best_entry.get("total_call_oi", 0),
                "total_put_oi": best_entry.get("total_put_oi", 0),
                "call_wall_oi": best_entry.get("call_wall_oi", 0),
                "put_wall_oi": best_entry.get("put_wall_oi", 0),
                "dte_reason": dte_reason,
                "dte_range": f"{min_dte}-{max_dte}d",
                "scan_type": scan_type,
                "flow_score": flow_score,
                "unusual_activity_count": total_unusual,
                "source": "polygon",
            }
        }
            
    except Exception as e:
        print(f"Options fetch error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


# Get scanner for weekly structure
_scanner = None

def set_scanner_for_rules(scanner):
    """Set the scanner instance for weekly structure lookups"""
    global _scanner
    _scanner = scanner

def get_rules_scanner():
    return _scanner


async def fetch_weekly_structure_for_plan(symbol: str) -> Optional[Dict]:
    """Fetch weekly structure data for rule engine integration"""
    scanner = get_rules_scanner()
    if scanner is None:
        return None
    
    try:
        # Get weekly candles
        weekly_df = scanner._get_candles(symbol, "W", 52)
        if weekly_df is None or len(weekly_df) < 6:
            return None
        
        # Get daily candles
        daily_df = scanner._get_candles(symbol, "D", 60)
        if daily_df is None or len(daily_df) < 5:
            return None
        
        current_price = daily_df['close'].iloc[-1]
        
        # Use scanner's calculate_range_structure
        if hasattr(scanner, 'calc') and hasattr(scanner.calc, 'calculate_range_structure'):
            ctx = scanner.calc.calculate_range_structure(weekly_df, daily_df, current_price)
            return {
                "trend": ctx.trend,
                "range_state": ctx.range_state,
                "compression_ratio": float(ctx.compression_ratio) if ctx.compression_ratio else None,
                "hh_count": int(ctx.hh_count) if ctx.hh_count else 0,
                "hl_count": int(ctx.hl_count) if ctx.hl_count else 0,
                "lh_count": int(ctx.lh_count) if ctx.lh_count else 0,
                "ll_count": int(ctx.ll_count) if ctx.ll_count else 0,
                "near_support": bool(ctx.near_support) if ctx.near_support is not None else False,
                "near_resistance": bool(ctx.near_resistance) if ctx.near_resistance is not None else False,
                "weekly_close_position": ctx.weekly_close_position,
                "weekly_close_signal": ctx.weekly_close_signal,
                "last_week_structure": ctx.last_week_structure
            }
    except Exception as e:
        print(f"Weekly structure error for {symbol}: {e}")
    
    return None


# Router
rule_router = APIRouter(tags=["Rule Engine"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PastReport(BaseModel):
    """Past report from Firestore"""
    symbol: str
    date: Optional[str] = None
    direction: Optional[str] = None
    bull_score: float = 0
    bear_score: float = 0
    price: float = 0
    notes: Optional[List[str]] = []
    created_at: Optional[str] = None


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
    past_reports: Optional[List[PastReport]] = None  # Reports from Firestore
    # Squeeze data (optional - only present for squeeze scans)
    squeeze_score: Optional[int] = None
    squeeze_tier: Optional[str] = None  # FORMING, ACTIVE, EXTREME
    ttm_squeeze: Optional[bool] = None
    squeeze_duration: Optional[int] = None
    atr_compression: Optional[float] = None
    adx: Optional[float] = None
    direction_bias: Optional[str] = None  # long, short, neutral
    bias_score: Optional[int] = None
    price_drift: Optional[str] = None  # up, down, flat
    volume_bias: Optional[str] = None  # accumulation, distribution, neutral
    scan_type: Optional[str] = None  # squeeze, capitulation, entry, etc.
    timeframe: Optional[str] = None   # 5MIN, 15MIN, 30MIN, 1HR, 2HR, 4HR
    # Earnings data (optional - will be fetched if not provided)
    earnings_days: Optional[int] = None
    earnings_date: Optional[str] = None
    # Fibonacci data (optional - from /api/analyze/live/{symbol})
    fib_zone: Optional[str] = None          # golden_zone, pullback_zone, extended, broken, etc.
    fib_quality: Optional[str] = None       # A+, A, B, C
    fib_trend: Optional[str] = None         # UPTREND, DOWNTREND
    fib_position: Optional[str] = None      # Human-readable position text
    fib_confluence: Optional[List[str]] = None  # VP+Fib confluence points
    fib_levels: Optional[Dict] = None       # All numeric fib levels
    # Structure reversal alerts (optional - from /api/structure/reversals/{symbol})
    structure_reversals: Optional[List[Dict]] = None  # Reversal alerts from structure analysis
    # Trade duration tier (optional — auto-assigned if not provided)
    duration_tier: Optional[str] = None       # DAY, SWING, POSITION, MACRO, CUSTOM
    setup_type: Optional[str] = None          # For auto-assign: squeeze_firing, vp_rejection, etc.
    custom_hold_days: Optional[int] = None    # For CUSTOM tier: exact days


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
    # Level data for NO_TRADE visualization
    current_price: Optional[float] = None
    vah: Optional[float] = None
    poc: Optional[float] = None
    val: Optional[float] = None
    vwap: Optional[float] = None
    # Options integration data
    options_sentiment: Optional[str] = None  # BULLISH, BEARISH, NEUTRAL
    pc_ratio: Optional[float] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    expected_move: Optional[float] = None
    avg_iv: Optional[float] = None
    nearest_expiration: Optional[str] = None  # Best expiration date (YYYY-MM-DD)
    dte: Optional[int] = None                 # Days to expiration for recommended exp
    expirations_available: Optional[List[str]] = None  # All available expirations
    total_call_oi: Optional[int] = None
    total_put_oi: Optional[int] = None
    dte_reason: Optional[str] = None      # Why this DTE was chosen
    scan_type_used: Optional[str] = None  # What scan type drove the selection
    # Polygon flow data
    flow_score: Optional[int] = None          # -100 to +100 (calls vs puts)
    unusual_activity_count: Optional[int] = None  # Number of unusual contracts
    options_source: Optional[str] = None      # "polygon"
    # OI scoring data
    call_wall_oi: Optional[int] = None        # OI at the call wall strike
    put_wall_oi: Optional[int] = None         # OI at the put wall strike
    oi_skew: Optional[float] = None           # Call OI / Put OI ratio
    oi_skew_sentiment: Optional[str] = None   # BULLISH, BEARISH, NEUTRAL
    unusual_call_count: Optional[int] = None  # Contracts with Vol/OI > 2x
    unusual_put_count: Optional[int] = None
    unusual_activity_sentiment: Optional[str] = None  # Direction of unusual flow
    # Earnings data
    earnings_days: Optional[int] = None
    earnings_date: Optional[str] = None
    # Dual-direction probabilities
    long_prob: Optional[float] = None
    short_prob: Optional[float] = None
    # Fibonacci data
    fib_zone: Optional[str] = None
    fib_quality: Optional[str] = None
    fib_trend: Optional[str] = None
    fib_position: Optional[str] = None
    fib_confluence: Optional[List[str]] = None
    fib_used_for_stop: Optional[bool] = None
    fib_used_for_target: Optional[bool] = None
    # Trade duration tier
    duration_tier: Optional[str] = None          # DAY, SWING, POSITION, MACRO, CUSTOM
    duration_label: Optional[str] = None         # Human-readable label
    expected_hold_days: Optional[str] = None     # "3-5 days", "Intraday", etc.
    recommended_dte: Optional[int] = None        # Recommended option DTE for this tier


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
async def generate_trade_plan(data: ScannerData, explain: bool = True, save: bool = True, include_options: bool = True):
    """
    Generate a trade plan from scanner data.
    
    The rule engine applies YOUR deterministic rules to generate exact levels.
    AI only explains the reasoning - it cannot change the levels.
    Past reports from Firestore are passed in to give AI learning context.
    Options data is fetched and used to adjust confidence.
    Weekly structure data is fetched to provide macro context.
    """
    # Kill switch: force explain=False to skip AI calls
    try:
        import unified_server
        if getattr(unified_server, 'AI_KILL_SWITCH', False):
            explain = False
    except Exception:
        pass
    
    try:
        # Convert to dict
        scanner_dict = data.dict()
        
        # Extract past reports for AI context
        past_reports = scanner_dict.pop('past_reports', None) or []
        
        # Fetch options data for rule engine integration
        options_data = None
        if include_options:
            options_data = await fetch_options_for_plan(
                scanner_dict['symbol'],
                scan_type=scanner_dict.get('scan_type'),
                timeframe=scanner_dict.get('timeframe'),
                confidence=scanner_dict.get('confidence', 50)
            )
        
        # Fetch weekly structure for macro context
        weekly_structure = await fetch_weekly_structure_for_plan(scanner_dict['symbol'])
        if weekly_structure:
            scanner_dict['weekly_structure'] = weekly_structure
        
        # Fetch earnings data if not provided
        earnings_days = scanner_dict.get('earnings_days')
        earnings_date = scanner_dict.get('earnings_date')
        if earnings_days is None:
            try:
                cal = get_earnings_calendar()
                earnings_info = cal.get_earnings_info(scanner_dict['symbol'])
                if earnings_info:
                    earnings_days = earnings_info.days_until
                    earnings_date = earnings_info.date
            except Exception as e:
                print(f"Earnings fetch error: {e}")
        
        # Generate plan with past report context and options data
        plan, explanation, plan_id = generate_plan(
            scanner_dict, 
            explain=explain, 
            save=save,
            past_reports=past_reports,
            options_data=options_data
        )
        
        # Persist AI explanation to Firestore
        if explanation and explain:
            try:
                from unified_server import _save_ai_suggestion_bg
                _save_ai_suggestion_bg(scanner_dict.get('symbol', ''), "rule_explanation",
                    explanation, {
                        "direction": plan.direction,
                        "confidence": plan.confidence,
                        "entry": plan.entry_price,
                        "stop": plan.stop_loss,
                        "target_1": plan.target_1
                    })
            except Exception:
                pass
        
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
            formatted_plan=formatted,
            # Include levels for NO_TRADE visualization
            current_price=scanner_dict.get('current_price') or scanner_dict.get('price'),
            vah=scanner_dict.get('vah'),
            poc=scanner_dict.get('poc'),
            val=scanner_dict.get('val'),
            vwap=scanner_dict.get('vwap'),
            # Options integration data
            options_sentiment=plan.options_sentiment,
            pc_ratio=plan.pc_ratio,
            call_wall=plan.call_wall,
            put_wall=plan.put_wall,
            expected_move=plan.expected_move,
            avg_iv=plan.avg_iv,
            nearest_expiration=options_data.get('flat', {}).get('expiration') if options_data else None,
            dte=options_data.get('flat', {}).get('dte') if options_data else None,
            expirations_available=options_data.get('flat', {}).get('expirations_available') if options_data else None,
            total_call_oi=options_data.get('flat', {}).get('total_call_oi') if options_data else None,
            total_put_oi=options_data.get('flat', {}).get('total_put_oi') if options_data else None,
            dte_reason=options_data.get('flat', {}).get('dte_reason') if options_data else None,
            scan_type_used=scanner_dict.get('scan_type'),
            # Polygon flow data
            flow_score=options_data.get('flat', {}).get('flow_score') if options_data else None,
            unusual_activity_count=options_data.get('flat', {}).get('unusual_activity_count') if options_data else None,
            options_source=options_data.get('flat', {}).get('source', 'polygon') if options_data else None,
            # OI scoring data
            call_wall_oi=plan.call_wall_oi,
            put_wall_oi=plan.put_wall_oi,
            oi_skew=plan.oi_skew,
            oi_skew_sentiment=plan.oi_skew_sentiment,
            unusual_call_count=plan.unusual_call_count,
            unusual_put_count=plan.unusual_put_count,
            unusual_activity_sentiment=plan.unusual_activity_sentiment,
            # Earnings data
            earnings_days=earnings_days,
            earnings_date=earnings_date,
            # Dual-direction probabilities from bull/bear scores
            long_prob=min(95, max(15, scanner_dict.get('bull_score', 50))),
            short_prob=min(95, max(15, scanner_dict.get('bear_score', 50))),
            # Fibonacci data
            fib_zone=plan.fib_zone,
            fib_quality=plan.fib_quality,
            fib_trend=plan.fib_trend,
            fib_position=plan.fib_position,
            fib_confluence=plan.fib_confluence,
            fib_used_for_stop=plan.fib_used_for_stop,
            fib_used_for_target=plan.fib_used_for_target,
            # Trade duration tier
            duration_tier=plan.duration_tier,
            duration_label=plan.duration_label,
            expected_hold_days=plan.expected_hold_days,
            recommended_dte=plan.recommended_dte,
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


# =============================================================================
# AUTO-REPORT GENERATION ENDPOINTS
# =============================================================================

class ReportRequest(BaseModel):
    """Request to generate a report"""
    scanner_data: Dict
    trade_plan: Optional[Dict] = None
    

class BatchReportRequest(BaseModel):
    """Request to generate batch summary"""
    results: List[Dict]
    generate_individual: bool = False  # Also generate individual reports?


@rule_router.post("/report/generate")
async def generate_report(request: ReportRequest):
    """
    Generate an analysis report from scanner data.
    Report is saved to reports/ folder for AI learning.
    """
    try:
        filepath = auto_generate_report(request.scanner_data, request.trade_plan)
        
        return {
            "status": "success",
            "filepath": filepath,
            "symbol": request.scanner_data.get('symbol', 'UNKNOWN'),
            "message": "Report generated and saved for AI learning"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.post("/report/batch")
async def generate_batch_report(request: BatchReportRequest):
    """
    Generate a summary report from batch scan results.
    Optionally generate individual reports for top symbols.
    """
    try:
        generator = AutoReportGenerator()
        
        # Generate summary
        summary_path = generator.generate_batch_summary(request.results)
        
        individual_reports = []
        if request.generate_individual:
            # Generate individual reports for top scores
            sorted_results = sorted(
                request.results,
                key=lambda x: max(x.get('bull_score', 0), x.get('bear_score', 0)),
                reverse=True
            )
            
            # Top 5 only to avoid too many reports
            for result in sorted_results[:5]:
                if max(result.get('bull_score', 0), result.get('bear_score', 0)) >= 60:
                    path = generator.generate_report(result)
                    individual_reports.append({
                        'symbol': result.get('symbol'),
                        'filepath': path
                    })
        
        return {
            "status": "success",
            "summary_path": summary_path,
            "symbols_scanned": len(request.results),
            "individual_reports": individual_reports,
            "message": f"Generated summary + {len(individual_reports)} individual reports"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/reports/list")
async def list_reports(symbol: str = None, limit: int = 50):
    """
    List all generated reports from Firestore.
    """
    try:
        fs = get_firestore()
        reports = fs.get_reports(symbol=symbol, limit=limit)
        
        # Format for response
        formatted = []
        for r in reports:
            formatted.append({
                "id": r.get('id'),
                "symbol": r.get('symbol'),
                "date": r.get('date'),
                "type": r.get('type'),
                "created_at": r.get('created_at')
            })
        
        return {
            "reports": formatted,
            "count": len(formatted),
            "source": "firestore" if fs.is_available() else "local"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/reports/{doc_id}")
async def get_report(doc_id: str):
    """
    Get a specific report's content by ID.
    """
    try:
        fs = get_firestore()
        content = fs.get_report_content(doc_id)
        
        if not content:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "id": doc_id,
            "content": content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rule_router.get("/debug/firestore")
async def debug_firestore():
    """Debug endpoint to check Firestore connection status"""
    import os
    fs = get_firestore()
    has_env = bool(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
    
    return {
        "has_env_var": has_env,
        "env_var_length": len(os.getenv('FIREBASE_SERVICE_ACCOUNT', '')),
        "db_available": fs.is_available(),
        "db_object": str(type(fs.db)) if fs.db else None
    }
