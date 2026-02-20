"""
Trade Rule Engine - Deterministic Trading Rules + Learning Layer
================================================================
Generates consistent trade plans based on YOUR defined rules.
AI only explains the reasoning - never overrides the levels.

Learns from outcomes to refine rules over time.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sqlite3


# =============================================================================
# CORE TRADING RULES - EDIT THESE TO MATCH YOUR STRATEGY
# =============================================================================

class TradingRules:
    """
    YOUR trading rules - deterministic, no AI interpretation.
    Edit these values to match your personal strategy.
    """
    
    # =========================
    # ENTRY RULES
    # =========================
    
    # Minimum scores to take a trade
    MIN_SCORE_FULL_SIZE = 70      # Score needed for full position
    MIN_SCORE_HALF_SIZE = 50      # Score needed for half position
    MIN_SCORE_NO_TRADE = 50       # Below this = no trade
    
    # Position requirements
    LONG_REQUIRES_ABOVE_POC = True    # Price must be above POC for longs
    SHORT_REQUIRES_BELOW_POC = True   # Price must be below POC for shorts
    
    # VWAP confirmation
    LONG_PREFER_ABOVE_VWAP = True     # Prefer longs above VWAP
    SHORT_PREFER_BELOW_VWAP = True    # Prefer shorts below VWAP
    
    # Volume requirements
    MIN_RVOL_FOR_ENTRY = 0.8          # Minimum relative volume
    HIGH_RVOL_BONUS = 1.5             # RVOL above this = higher confidence
    
    # =========================
    # STOP LOSS RULES
    # =========================
    
    # Stop placement
    LONG_STOP_BELOW_VAL_PCT = 0.25    # Long stops X% below VAL
    SHORT_STOP_ABOVE_VAH_PCT = 0.25   # Short stops X% above VAH
    
    # Alternative: ATR-based stops (if ATR available)
    USE_ATR_STOPS = False
    ATR_STOP_MULTIPLIER = 1.5
    
    # Maximum stop distance (% of price) - risk control
    MAX_STOP_DISTANCE_PCT = 3.0       # Never risk more than 3%
    
    # =========================
    # TARGET RULES
    # =========================
    
    # Target 1: Opposite value area level
    T1_AT_OPPOSITE_VA = True          # First target at VAH (longs) or VAL (shorts)
    T1_TAKE_PARTIAL = 0.5             # Take 50% off at T1
    
    # Target 2: R-multiple based
    T2_R_MULTIPLE = 2.0               # Second target at 2R
    T2_TAKE_PARTIAL = 0.3             # Take 30% off at T2
    
    # Target 3: Runner
    T3_R_MULTIPLE = 3.0               # Let 20% run to 3R
    T3_TRAIL_STOP = True              # Trail stop on runner
    
    # =========================
    # POSITION SIZING RULES
    # =========================
    
    # Base risk per trade (% of account)
    BASE_RISK_PCT = 1.0               # Risk 1% per trade normally
    
    # Adjustments based on conditions
    HIGH_SCORE_RISK_MULT = 1.0        # Score 70+: full size (1x)
    MED_SCORE_RISK_MULT = 0.5         # Score 50-69: half size (0.5x)
    HIGH_VOL_RISK_MULT = 0.5          # VIX > 25: reduce to 0.5x
    
    # =========================
    # FILTER RULES
    # =========================
    
    # RSI filters
    RSI_OVERBOUGHT = 75               # Avoid new longs above this
    RSI_OVERSOLD = 25                 # Avoid new shorts below this
    
    # Time filters
    AVOID_FIRST_15_MIN = True         # Skip signals in first 15 min
    AVOID_LAST_30_MIN = True          # Skip signals in last 30 min
    AVOID_LUNCH = False               # 11:30-1:30 ET
    
    # Earnings filter
    AVOID_EARNINGS_DAYS = 3           # Skip if earnings within X days
    
    # =========================
    # OPTIONS INTEGRATION RULES
    # =========================
    
    # Options sentiment adjustments
    USE_OPTIONS_DATA = True           # Enable options integration
    PC_RATIO_BULLISH = 0.7            # Below this = bullish (call heavy)
    PC_RATIO_BEARISH = 1.3            # Above this = bearish (put heavy)
    
    # Confidence adjustments from options
    OPTIONS_CONFIRM_BONUS = 10        # Add to confidence if options confirm direction
    OPTIONS_CONFLICT_PENALTY = 10     # Subtract if options conflict with direction
    
    # Wall levels
    USE_CALL_WALL_AS_RESISTANCE = True  # Call wall can cap upside
    USE_PUT_WALL_AS_SUPPORT = True      # Put wall can provide floor
    
    # Expected move validation
    WARN_IF_TARGET_BEYOND_EXPECTED = True  # Flag if T1 > expected move
    
    # IV-based adjustments
    HIGH_IV_THRESHOLD = 50            # IV% considered high (earnings, events)
    HIGH_IV_REDUCE_SIZE = True        # Reduce position size in high IV
    
    # =========================
    # OI SCORING RULES
    # =========================
    
    # Wall OI magnitude (is the wall strong or weak?)
    USE_OI_SCORING = True             # Enable OI-based score adjustments
    STRONG_WALL_OI_THRESHOLD = 5000   # OI at wall > this = strong wall
    STRONG_WALL_BONUS = 5             # Pts bonus when wall confirms direction
    WEAK_WALL_OI_THRESHOLD = 500      # OI at wall < this = ignore wall
    
    # Unusual activity (volume >> OI = new money opening)
    UNUSUAL_VOL_OI_RATIO = 2.0        # Vol/OI > this = unusual activity
    UNUSUAL_ACTIVITY_BONUS = 7        # Pts when unusual activity confirms direction
    UNUSUAL_ACTIVITY_PENALTY = 5      # Pts when unusual activity conflicts
    MIN_UNUSUAL_CONTRACTS = 3         # Need at least N unusual contracts to score
    
    # OI skew (total call OI vs total put OI across chain)
    OI_SKEW_BULLISH = 1.5             # Call OI / Put OI > this = bullish positioning
    OI_SKEW_BEARISH = 0.67            # Call OI / Put OI < this = bearish positioning
    OI_SKEW_BONUS = 5                 # Pts when OI skew confirms direction


@dataclass
class TradePlan:
    """Deterministic trade plan output"""
    symbol: str
    direction: str                    # 'LONG', 'SHORT', 'NO_TRADE'
    confidence: float                 # 0-100
    
    # Levels
    entry_price: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    
    # Risk metrics
    risk_per_share: float             # $ risk per share
    risk_reward_t1: float             # R:R to target 1
    risk_reward_t2: float             # R:R to target 2
    
    # Position sizing
    position_size_pct: float          # % of normal size (0.5 = half)
    risk_pct: float                   # Actual % of account risked
    
    # Reasoning (for AI to expand on)
    entry_reasons: List[str]
    caution_flags: List[str]
    invalidation: str
    
    # Options data (from Polygon)
    options_data: Optional[Dict] = None
    options_sentiment: Optional[str] = None    # 'BULLISH', 'BEARISH', 'NEUTRAL'
    pc_ratio: Optional[float] = None
    max_pain: Optional[float] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    expected_move: Optional[float] = None
    avg_iv: Optional[float] = None
    
    # OI scoring fields
    call_wall_oi: Optional[int] = None         # OI at the call wall strike
    put_wall_oi: Optional[int] = None          # OI at the put wall strike
    oi_skew: Optional[float] = None            # Call OI / Put OI ratio
    oi_skew_sentiment: Optional[str] = None    # 'BULLISH', 'BEARISH', 'NEUTRAL'
    unusual_call_count: Optional[int] = None   # Contracts with Vol/OI > 2x
    unusual_put_count: Optional[int] = None
    unusual_activity_sentiment: Optional[str] = None  # Direction of unusual flow
    
    # Full report markdown (for learning)
    full_report: Optional[str] = None
    
    # Fibonacci data
    fib_zone: Optional[str] = None              # golden_zone, pullback_zone, extended, broken, etc.
    fib_quality: Optional[str] = None           # A+, A, B, C
    fib_trend: Optional[str] = None             # UPTREND, DOWNTREND
    fib_position: Optional[str] = None          # Human-readable position text
    fib_confluence: Optional[List[str]] = None  # VP+Fib confluence points
    fib_levels: Optional[Dict] = None           # All numeric fib levels
    fib_used_for_stop: bool = False             # Whether fib improved the stop
    fib_used_for_target: bool = False           # Whether fib improved a target
    
    # Metadata
    timestamp: str = ""
    scanner_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RuleEngine:
    """
    Generates trade plans using deterministic rules.
    No AI interpretation - just math and your rules.
    """
    
    def __init__(self, rules: TradingRules = None):
        self.rules = rules or TradingRules()
        self.learning_db = LearningDatabase()
    
    def generate_plan(self, scanner_result: Dict, options_data: Dict = None) -> TradePlan:
        """
        Generate a complete trade plan from scanner data.
        
        Args:
            scanner_result: Dict from your scanner with keys:
                - symbol, price, vah, poc, val, vwap
                - bull_score, bear_score, confidence
                - rsi, rvol, direction, signal
            options_data: Optional dict from Tradier with keys:
                - pc_ratio, max_call_oi_strike, max_put_oi_strike
                - avg_call_iv, avg_put_iv, expected_move
                
        Returns:
            TradePlan with all levels calculated
        """
        r = self.rules
        s = scanner_result
        
        symbol = s.get('symbol', 'UNKNOWN')
        price = s.get('current_price') or s.get('price', 0)
        vah = s.get('vah') or price * 1.01
        poc = s.get('poc') or price
        val = s.get('val') or price * 0.99
        vwap = s.get('vwap') or price
        
        bull_score = s.get('bull_score') or 0
        bear_score = s.get('bear_score') or 0
        rsi = s.get('rsi') or 50
        rvol = s.get('rvol') or 1.0
        
        # EXPLICIT direction from scan (extension, capitulation, etc.)
        # This overrides bull/bear score direction when present
        scan_direction = s.get('scan_direction')
        
        # Squeeze data (optional)
        squeeze_score = s.get('squeeze_score')
        squeeze_tier = s.get('squeeze_tier')
        ttm_squeeze = s.get('ttm_squeeze')
        squeeze_duration = s.get('squeeze_duration') or 0
        direction_bias = s.get('direction_bias')
        bias_score = s.get('bias_score') or 0
        price_drift = s.get('price_drift')
        volume_bias = s.get('volume_bias')
        scan_type = s.get('scan_type')
        timeframe = s.get('timeframe', '1HR')  # Extract timeframe (5MIN, 15MIN, 30MIN, 1HR, 2HR, 4HR)
        
        # Handle edge case: rvol of 0 or near-0 means no data (weekend/after hours)
        # Treat as "unknown" (1.0) rather than "zero volume"
        if rvol < 0.01:
            rvol = 1.0  # Default to average, not zero
        
        entry_reasons = []
        caution_flags = []
        
        # =========================
        # PROCESS SQUEEZE DATA
        # =========================
        
        if scan_type == 'squeeze' and squeeze_score:
            # Add squeeze context to entry reasons
            if ttm_squeeze:
                entry_reasons.append(f"🎰 TTM Squeeze ({squeeze_duration}d compression)")
            
            if squeeze_tier == 'EXTREME':
                entry_reasons.append(f"💥 EXTREME squeeze ({squeeze_score}pt)")
            elif squeeze_tier == 'ACTIVE':
                entry_reasons.append(f"🎯 ACTIVE squeeze ({squeeze_score}pt)")
            else:
                entry_reasons.append(f"🎰 FORMING squeeze ({squeeze_score}pt)")
            
            # Direction bias from squeeze analysis
            if direction_bias and bias_score >= 30:
                if direction_bias == 'long':
                    entry_reasons.append(f"Bias: LONG ({bias_score}% conf) - {price_drift} drift, {volume_bias}")
                    bull_score += 10  # Boost bull score
                elif direction_bias == 'short':
                    entry_reasons.append(f"Bias: SHORT ({bias_score}% conf) - {price_drift} drift, {volume_bias}")
                    bear_score += 10  # Boost bear score
            
            # Longer squeezes = bigger potential moves
            if squeeze_duration >= 5:
                entry_reasons.append(f"⏱️ {squeeze_duration}d compression = bigger release potential")
        
        # =========================
        # PROCESS WEEKLY STRUCTURE
        # =========================
        
        weekly_structure = s.get('weekly_structure')
        if weekly_structure:
            weekly_trend = weekly_structure.get('trend', 'NEUTRAL')
            weekly_close_signal = weekly_structure.get('weekly_close_signal', '')
            weekly_close_position = weekly_structure.get('weekly_close_position', 0.5)
            near_support = weekly_structure.get('near_support', False)
            near_resistance = weekly_structure.get('near_resistance', False)
            last_week_struct = weekly_structure.get('last_week_structure', '')
            
            # REVERSAL SIGNALS - high impact (15 pts)
            if weekly_close_signal == 'BULLISH_REVERSAL':
                bull_score += 15
                entry_reasons.append(f"📊 Weekly BULLISH REVERSAL (LL but closed {weekly_close_position*100:.0f}% up)")
            elif weekly_close_signal == 'BEARISH_REVERSAL':
                bear_score += 15
                entry_reasons.append(f"📊 Weekly BEARISH REVERSAL (HH but closed {weekly_close_position*100:.0f}% down)")
            
            # STRONG CONTINUATION - medium impact (10 pts)
            elif weekly_close_signal == 'STRONG_BULL_CLOSE':
                bull_score += 10
                entry_reasons.append(f"📊 Weekly strong bull close ({weekly_close_position*100:.0f}%)")
            elif weekly_close_signal == 'STRONG_BEAR_CLOSE':
                bear_score += 10
                entry_reasons.append(f"📊 Weekly strong bear close ({weekly_close_position*100:.0f}%)")
            
            # CLOSE POSITION - lower impact (5 pts)
            elif weekly_close_signal == 'STRONG_CLOSE':
                bull_score += 5
            elif weekly_close_signal == 'WEAK_CLOSE':
                bear_score += 5
            
            # TREND ALIGNMENT - boost scores when close aligns with trend (5 pts)
            if 'UPTREND' in weekly_trend:
                if weekly_close_position > 0.6:
                    bull_score += 5
                    entry_reasons.append(f"Weekly uptrend + strong close confirms bias")
                elif weekly_close_position < 0.3:
                    caution_flags.append(f"⚠️ Weekly uptrend but weak close - watch for reversal")
            elif 'DOWNTREND' in weekly_trend:
                if weekly_close_position < 0.4:
                    bear_score += 5
                    entry_reasons.append(f"Weekly downtrend + weak close confirms bias")
                elif weekly_close_position > 0.7:
                    caution_flags.append(f"⚠️ Weekly downtrend but strong close - watch for reversal")
            
            # PROXIMITY + CLOSE SIGNALS - key for entries
            if near_support and weekly_close_position > 0.5:
                bull_score += 8
                entry_reasons.append(f"📍 Near weekly support with buyers (close {weekly_close_position*100:.0f}%)")
            elif near_resistance and weekly_close_position < 0.5:
                bear_score += 8
                entry_reasons.append(f"📍 Near weekly resistance with sellers (close {weekly_close_position*100:.0f}%)")
            
            # Warnings for bad positioning
            if near_resistance and weekly_close_position > 0.7:
                caution_flags.append("⚠️ At weekly resistance - late long entry")
            if near_support and weekly_close_position < 0.3:
                caution_flags.append("⚠️ At weekly support - late short entry")
        
        # =========================
        # PROCESS STRUCTURE REVERSAL ALERTS
        # =========================
        
        structure_reversals = s.get('structure_reversals') or []
        if structure_reversals:
            for alert in structure_reversals:
                alert_type = alert.get('alert_type', '')
                confidence = alert.get('confidence', 0)
                severity = alert.get('severity', 'LOW')
                description = alert.get('description', '')
                
                # CRITICAL/HIGH severity alerts have major impact
                if severity in ('CRITICAL', 'HIGH'):
                    impact_points = 15 if severity == 'CRITICAL' else 10
                    
                    # STRUCTURE BREAK alerts
                    if 'STRUCTURE_BREAK_LONG' in alert_type:
                        # LL in uptrend = bearish warning
                        bear_score += impact_points
                        caution_flags.insert(0, f"⚠️ REVERSAL ALERT ({confidence:.0f}%): {description}")
                    elif 'STRUCTURE_BREAK_SHORT' in alert_type:
                        # HH in downtrend = bullish warning
                        bull_score += impact_points
                        caution_flags.insert(0, f"⚠️ REVERSAL ALERT ({confidence:.0f}%): {description}")
                    
                    # MOMENTUM EXHAUSTION alerts
                    elif 'MOMENTUM_EXHAUSTION_LONG' in alert_type:
                        # Uptrend losing steam = reduce bullish conviction
                        bull_score -= impact_points
                        caution_flags.insert(0, f"⏸️ EXHAUSTION ({confidence:.0f}%): {description}")
                    elif 'MOMENTUM_EXHAUSTION_SHORT' in alert_type:
                        # Downtrend losing steam = reduce bearish conviction
                        bear_score -= impact_points
                        caution_flags.insert(0, f"⏸️ EXHAUSTION ({confidence:.0f}%): {description}")
                    
                    # RANGE EXTREME + DIVERGENCE alerts (potential reversals)
                    elif 'RANGE_EXTREME_LONG' in alert_type or 'STRUCTURE_DIVERGENCE_LONG' in alert_type:
                        # Bullish reversal setup
                        bull_score += impact_points
                        entry_reasons.append(f"📊 Reversal Setup ({confidence:.0f}%): {description[:60]}")
                    elif 'RANGE_EXTREME_SHORT' in alert_type or 'STRUCTURE_DIVERGENCE_SHORT' in alert_type:
                        # Bearish reversal setup
                        bear_score += impact_points
                        entry_reasons.append(f"📊 Reversal Setup ({confidence:.0f}%): {description[:60]}")
                    
                    # COMPRESSION BREAKOUT alerts
                    elif 'COMPRESSION_BREAKOUT_LONG' in alert_type:
                        bull_score += impact_points  
                        entry_reasons.append(f"💥 Compression Setup ({confidence:.0f}%): {description[:60]}")
                    elif 'COMPRESSION_BREAKOUT_SHORT' in alert_type:
                        bear_score += impact_points
                        entry_reasons.append(f"💥 Compression Setup ({confidence:.0f}%): {description[:60]}")
                
                # MEDIUM severity - smaller impact
                elif severity == 'MEDIUM':
                    impact_points = 5
                    
                    if 'EXHAUSTION' in alert_type:
                        if 'LONG' in alert_type:
                            bull_score -= impact_points
                        else:
                            bear_score -= impact_points
                        caution_flags.append(f"⏸️ {description[:80]}")
                    elif 'REVERSAL' in alert_type or 'EXTREME' in alert_type:
                        if 'LONG' in alert_type:
                            bull_score += impact_points
                        else:
                            bear_score += impact_points
                        entry_reasons.append(f"📊 {description[:80]}")
        
        # =========================
        # PROCESS ABSORPTION WALLS
        # =========================
        
        absorption_zones = s.get('absorption_zones') or []
        if absorption_zones:
            primary_zone = None
            # Find primary zone (highest score)
            for zone in absorption_zones:
                if not primary_zone or zone.get('score', 0) > primary_zone.get('score', 0):
                    primary_zone = zone
            
            if primary_zone:
                zone_price = primary_zone.get('center_price', 0)
                zone_type = primary_zone.get('absorption_type', '')
                strength = primary_zone.get('strength', '')
                status = primary_zone.get('status', '')
                touches = primary_zone.get('total_touches', 0)
                zone_rvol = primary_zone.get('rvol_ratio', 0)
                score = primary_zone.get('score', 0)
                
                # Impact points based on strength
                impact_map = {
                    'INSTITUTIONAL': 15,
                    'STRONG': 10,
                    'MODERATE': 5,
                    'WEAK': 2
                }
                impact_points = impact_map.get(strength, 0)
                
                # Price distance to zone (as % of current price)
                price_dist_pct = abs(zone_price - price) / price * 100 if price else 999
                near_price = price_dist_pct < 2.0  # Within 2%
                
                # CEILING absorption (passive sellers)
                if zone_type == 'CEILING':
                    # If ceiling is above current price and DEFENDED/HOLDING
                    if zone_price > price and status in ('DEFENDED', 'HOLDING'):
                        if near_price:
                            # Strong resistance overhead - reduce bullish conviction
                            bull_score -= impact_points
                            caution_flags.insert(0, 
                                f"🧱 CEILING at ${zone_price:.2f} ({strength}, {touches} touches, {zone_rvol:.1f}x RVOL) - Resistance overhead")
                        else:
                            # Distant ceiling - just note it
                            caution_flags.append(
                                f"🧱 CEILING at ${zone_price:.2f} ({strength}) - Upside resistance")
                    
                    # If ceiling is WEAKENING/BROKEN - potential breakout
                    elif status in ('WEAKENING', 'BROKEN'):
                        if near_price:
                            bear_score -= impact_points // 2  # Counter-trend fading less attractive
                            entry_reasons.append(
                                f"💥 CEILING BREAKING at ${zone_price:.2f} ({status}) - Breakout potential")
                
                # FLOOR absorption (passive buyers)
                elif zone_type == 'FLOOR':
                    # If floor is below current price and DEFENDED/HOLDING
                    if zone_price < price and status in ('DEFENDED', 'HOLDING'):
                        if near_price:
                            # Strong support below - reduce bearish conviction
                            bear_score -= impact_points
                            caution_flags.insert(0,
                                f"🛡️ FLOOR at ${zone_price:.2f} ({strength}, {touches} touches, {zone_rvol:.1f}x RVOL) - Support below")
                        else:
                            # Distant floor - just note it
                            caution_flags.append(
                                f"🛡️ FLOOR at ${zone_price:.2f} ({strength}) - Downside support")
                    
                    # If floor is WEAKENING/BROKEN - potential breakdown
                    elif status in ('WEAKENING', 'BROKEN'):
                        if near_price:
                            bull_score -= impact_points // 2  # Counter-trend fading less attractive
                            entry_reasons.append(
                                f"💥 FLOOR BREAKING at ${zone_price:.2f} ({status}) - Breakdown potential")
                
                # PINNING absorption (trapped in range)
                elif zone_type == 'PINNING':
                    if near_price and status == 'HOLDING':
                        # Both sides absorbing - range-bound, reduce conviction for directional trades
                        bull_score -= impact_points // 2
                        bear_score -= impact_points // 2
                        caution_flags.insert(0,
                            f"⚖️ PINNING at ${zone_price:.2f} ({strength}) - Range-bound, directional risk")
        
        # =========================
        # PROCESS OPTIONS DATA
        # =========================
        
        options_sentiment = None
        pc_ratio = None
        call_wall = None
        put_wall = None
        max_pain = None
        expected_move = None
        avg_iv = None
        # OI scoring variables
        call_wall_oi_val = None
        put_wall_oi_val = None
        oi_skew = None
        oi_skew_sentiment = None
        unusual_calls = None
        unusual_puts = None
        unusual_activity_sentiment = None
        
        if options_data and r.USE_OPTIONS_DATA:
            # Extract options metrics from the data array
            # Use the best expiration (from flat.expiration) if available, otherwise nearest
            # BUT skip ultra-short expirations based on timeframe
            
            # Determine minimum DTE based on timeframe (shorter TF = can use shorter DTE)
            MIN_DTE_BY_TIMEFRAME = {
                '5MIN':  2,   # Day trading: can use 2-7 DTE (0DTE if desperate)
                '15MIN': 3,   # Intraday swing: 3-10 DTE
                '30MIN': 5,   # Short swing: 5-14 DTE
                '1HR':   7,   # Standard swing: 7-21 DTE
                '2HR':   10,  # Longer swing: 10-30 DTE
                '4HR':   14,  # Position trade: 14-45 DTE
                'Daily': 21,  # Position trade: 21-60 DTE
            }
            MIN_DTE_FOR_TRADE = MIN_DTE_BY_TIMEFRAME.get(timeframe, 7)  # Default to 7 for 1HR
            
            if options_data.get('data') and len(options_data['data']) > 0:
                # Filter expirations to only those with sufficient DTE for this timeframe
                from datetime import datetime as dt
                valid_expirations = []
                for exp_data in options_data['data']:
                    dte_val = exp_data.get('dte', 0)
                    if not dte_val and exp_data.get('expiration'):
                        try:
                            dte_val = max(0, (dt.strptime(exp_data['expiration'], '%Y-%m-%d') - dt.now()).days)
                        except:
                            dte_val = 0
                    
                    if dte_val >= MIN_DTE_FOR_TRADE:
                        valid_expirations.append(exp_data)
                
                # If no valid expirations at preferred DTE, fall back to any available
                if not valid_expirations:
                    valid_expirations = options_data['data']
                
                # Now pick the best from valid expirations
                best_exp = options_data.get('flat', {}).get('expiration', '') if options_data.get('flat') else ''
                nearest = next((e for e in valid_expirations if e.get('expiration') == best_exp), valid_expirations[0])
                
                pc_ratio = nearest.get('pc_ratio', 1.0)
                call_wall = nearest.get('max_call_oi_strike', 0)
                put_wall = nearest.get('max_put_oi_strike', 0)
                
                # IV: avg_call_iv/avg_put_iv are raw (0-1), avg_iv_pct is already in %
                avg_iv_pct = nearest.get('avg_iv_pct', 0)
                if not avg_iv_pct:
                    # Fallback: compute from raw IVs
                    avg_call_iv = nearest.get('avg_call_iv', 0)
                    avg_put_iv = nearest.get('avg_put_iv', 0)
                    avg_iv_pct = round((avg_call_iv + avg_put_iv) / 2 * 100, 1)
                avg_iv = avg_iv_pct  # Store as percentage for display/thresholds
                
                # Calculate max pain (midpoint of walls)
                if call_wall and put_wall:
                    max_pain = (call_wall + put_wall) / 2
                
                # Calculate expected move using IV%
                if avg_iv and price:
                    # Get days to expiration
                    dte = nearest.get('dte', 0)
                    exp_date = nearest.get('expiration', '')
                    if not dte and exp_date:
                        try:
                            dte = max(1, (dt.strptime(exp_date, '%Y-%m-%d') - dt.now()).days)
                        except:
                            dte = 7
                    dte = max(MIN_DTE_FOR_TRADE, dte)  # Enforce timeframe-appropriate minimum
                    expected_move = price * (avg_iv / 100) * (dte / 365) ** 0.5
                
                # Determine options sentiment
                if pc_ratio < r.PC_RATIO_BULLISH:
                    options_sentiment = 'BULLISH'
                    entry_reasons.append(f"Options bullish (P/C {pc_ratio:.2f} < {r.PC_RATIO_BULLISH})")
                elif pc_ratio > r.PC_RATIO_BEARISH:
                    options_sentiment = 'BEARISH'
                    entry_reasons.append(f"Options bearish (P/C {pc_ratio:.2f} > {r.PC_RATIO_BEARISH})")
                else:
                    options_sentiment = 'NEUTRAL'
                
                # Add wall levels to reasons
                if call_wall and call_wall > price:
                    entry_reasons.append(f"Call wall at ${call_wall:.2f} (resistance)")
                if put_wall and put_wall < price:
                    entry_reasons.append(f"Put wall at ${put_wall:.2f} (support)")
                
                # High IV warning
                if avg_iv and avg_iv > r.HIGH_IV_THRESHOLD:
                    caution_flags.append(f"High IV ({avg_iv:.0f}%) - possible event/earnings")
                
                # =========================
                # OI SCORING (new)
                # =========================
                
                if r.USE_OI_SCORING:
                    # --- 1. Wall OI Magnitude ---
                    call_wall_oi_val = nearest.get('call_wall_oi', 0) or 0
                    put_wall_oi_val = nearest.get('put_wall_oi', 0) or 0
                    
                    # Strong call wall above price = resistance (confirms short, cautions long)
                    if call_wall and call_wall > price and call_wall_oi_val >= r.STRONG_WALL_OI_THRESHOLD:
                        if direction == 'SHORT':
                            bear_score += r.STRONG_WALL_BONUS
                            entry_reasons.append(f"Strong call wall ${call_wall:.0f} (OI: {call_wall_oi_val:,}) caps upside")
                        elif direction == 'LONG':
                            caution_flags.append(f"Strong call wall ${call_wall:.0f} (OI: {call_wall_oi_val:,}) may cap upside")
                    elif call_wall and call_wall > price and call_wall_oi_val < r.WEAK_WALL_OI_THRESHOLD:
                        caution_flags.append(f"Weak call wall ${call_wall:.0f} (OI: {call_wall_oi_val:,}) - unreliable resistance")
                    
                    # Strong put wall below price = support (confirms long, cautions short)
                    if put_wall and put_wall < price and put_wall_oi_val >= r.STRONG_WALL_OI_THRESHOLD:
                        if direction == 'LONG':
                            bull_score += r.STRONG_WALL_BONUS
                            entry_reasons.append(f"Strong put wall ${put_wall:.0f} (OI: {put_wall_oi_val:,}) provides floor")
                        elif direction == 'SHORT':
                            caution_flags.append(f"Strong put wall ${put_wall:.0f} (OI: {put_wall_oi_val:,}) may provide floor")
                    elif put_wall and put_wall < price and put_wall_oi_val < r.WEAK_WALL_OI_THRESHOLD:
                        caution_flags.append(f"Weak put wall ${put_wall:.0f} (OI: {put_wall_oi_val:,}) - unreliable support")
                    
                    # --- 2. Unusual Activity (Vol >> OI = new positions) ---
                    unusual_calls = nearest.get('unusual_call_count', 0) or 0
                    unusual_puts = nearest.get('unusual_put_count', 0) or 0
                    
                    if unusual_calls >= r.MIN_UNUSUAL_CONTRACTS and unusual_calls > unusual_puts * 2:
                        # Heavy unusual call activity = bullish conviction
                        unusual_activity_sentiment = 'BULLISH'
                        if direction == 'LONG':
                            bull_score += r.UNUSUAL_ACTIVITY_BONUS
                            entry_reasons.append(f"Unusual call activity ({unusual_calls} contracts with Vol/OI > 2x)")
                        elif direction == 'SHORT':
                            bear_score -= r.UNUSUAL_ACTIVITY_PENALTY
                            caution_flags.append(f"Unusual call activity ({unusual_calls} contracts) conflicts with short")
                    elif unusual_puts >= r.MIN_UNUSUAL_CONTRACTS and unusual_puts > unusual_calls * 2:
                        # Heavy unusual put activity = bearish conviction
                        unusual_activity_sentiment = 'BEARISH'
                        if direction == 'SHORT':
                            bear_score += r.UNUSUAL_ACTIVITY_BONUS
                            entry_reasons.append(f"Unusual put activity ({unusual_puts} contracts with Vol/OI > 2x)")
                        elif direction == 'LONG':
                            bull_score -= r.UNUSUAL_ACTIVITY_PENALTY
                            caution_flags.append(f"Unusual put activity ({unusual_puts} contracts) conflicts with long")
                    else:
                        unusual_activity_sentiment = 'NEUTRAL'
                    
                    # --- 3. OI Skew (total call OI vs put OI across chain) ---
                    total_call_oi = nearest.get('total_call_oi', 0) or 0
                    total_put_oi = nearest.get('total_put_oi', 0) or 0
                    oi_skew = total_call_oi / max(total_put_oi, 1)
                    
                    if oi_skew > r.OI_SKEW_BULLISH:
                        oi_skew_sentiment = 'BULLISH'
                        if direction == 'LONG':
                            bull_score += r.OI_SKEW_BONUS
                            entry_reasons.append(f"OI skew bullish (Call/Put OI: {oi_skew:.2f})")
                        elif direction == 'SHORT':
                            caution_flags.append(f"OI skew favors bulls (Call/Put OI: {oi_skew:.2f})")
                    elif oi_skew < r.OI_SKEW_BEARISH:
                        oi_skew_sentiment = 'BEARISH'
                        if direction == 'SHORT':
                            bear_score += r.OI_SKEW_BONUS
                            entry_reasons.append(f"OI skew bearish (Call/Put OI: {oi_skew:.2f})")
                        elif direction == 'LONG':
                            caution_flags.append(f"OI skew favors bears (Call/Put OI: {oi_skew:.2f})")
                    else:
                        oi_skew_sentiment = 'NEUTRAL'
        
        # =========================
        # PROCESS FIB DATA
        # =========================
        
        fib_zone = s.get('fib_zone')
        fib_quality = s.get('fib_quality')
        fib_trend = s.get('fib_trend')
        fib_position = s.get('fib_position')
        fib_confluence_list = s.get('fib_confluence') or []
        fib_levels_data = s.get('fib_levels') or {}
        fib_used_for_stop = False
        fib_used_for_target = False
        
        if fib_zone:
            # CONFIDENCE ADJUSTMENTS based on fib zone
            if fib_zone in ('golden_zone',):
                bull_score += 8 if fib_trend == 'UPTREND' else 0
                bear_score += 8 if fib_trend == 'DOWNTREND' else 0
                entry_reasons.append(f"📐 Fib GOLDEN ZONE ({fib_position})")
            elif fib_zone in ('pullback_zone',):
                bull_score += 5 if fib_trend == 'UPTREND' else 0
                bear_score += 5 if fib_trend == 'DOWNTREND' else 0
                entry_reasons.append(f"📐 Fib pullback zone ({fib_position})")
            elif fib_zone in ('shallow_pullback',):
                entry_reasons.append(f"📐 Shallow fib pullback ({fib_position})")
            elif fib_zone in ('strong_trend',):
                entry_reasons.append(f"📐 Strong trend position ({fib_position})")
            elif fib_zone in ('extended',):
                caution_flags.append(f"📐 Extended beyond fib range ({fib_position})")
            elif fib_zone in ('broken',):
                caution_flags.append(f"📐 Fib trend may be broken ({fib_position})")
                # Reduce confidence for trend-following trades
                if fib_trend == 'UPTREND':
                    bull_score -= 5
                elif fib_trend == 'DOWNTREND':
                    bear_score -= 5
            
            # QUALITY-based adjustment
            if fib_quality == 'A+':
                entry_reasons.append("📐 Fib quality A+ (textbook setup)")
                bull_score += 3 if fib_trend == 'UPTREND' else 0
                bear_score += 3 if fib_trend == 'DOWNTREND' else 0
            elif fib_quality == 'C':
                caution_flags.append("📐 Fib quality C (weak swing structure)")
        
        # VP+FIB CONFLUENCE bonus - boost only the trend-aligned side
        if fib_confluence_list:
            confluence_count = len(fib_confluence_list)
            if confluence_count >= 2:
                entry_reasons.append(f"🎯 {confluence_count}x VP+Fib confluence: {'; '.join(fib_confluence_list[:2])}")
                if fib_trend == 'UPTREND':
                    bull_score += 5
                elif fib_trend == 'DOWNTREND':
                    bear_score += 5
                else:
                    bull_score += 3
                    bear_score += 3
            elif confluence_count == 1:
                entry_reasons.append(f"🎯 VP+Fib confluence: {fib_confluence_list[0]}")
                if fib_trend == 'UPTREND':
                    bull_score += 3
                elif fib_trend == 'DOWNTREND':
                    bear_score += 3
                else:
                    bull_score += 2
                    bear_score += 2
        
        # =========================
        # DETERMINE DIRECTION
        # =========================
        
        direction = 'NO_TRADE'
        max_score = max(bull_score, bear_score)
        
        # If scan provides explicit direction (extension, capitulation, etc.), use it
        # These scans have specific logic that may differ from bull/bear scores
        if scan_direction:
            if scan_direction.lower() == 'long':
                direction = 'LONG'
                entry_reasons.append(f"Scan direction: LONG (mean reversion / bounce expected)")
                if bear_score > bull_score:
                    caution_flags.append(f"⚠️ Bear score {bear_score:.0f} > Bull {bull_score:.0f} - counter-trend trade")
                if max_score < r.MIN_SCORE_NO_TRADE:
                    caution_flags.append(f"⚠️ Low conviction ({max_score:.0f} < {r.MIN_SCORE_NO_TRADE}) - reduce size")
            elif scan_direction.lower() == 'short':
                direction = 'SHORT'
                entry_reasons.append(f"Scan direction: SHORT (mean reversion / fade expected)")
                if bull_score > bear_score:
                    caution_flags.append(f"⚠️ Bull score {bull_score:.0f} > Bear {bear_score:.0f} - counter-trend trade")
                if max_score < r.MIN_SCORE_NO_TRADE:
                    caution_flags.append(f"⚠️ Low conviction ({max_score:.0f} < {r.MIN_SCORE_NO_TRADE}) - reduce size")
        # Otherwise use bull/bear scores
        elif max_score < r.MIN_SCORE_NO_TRADE:
            caution_flags.append(f"Score too low ({max_score:.0f} < {r.MIN_SCORE_NO_TRADE})")
        else:
            if bull_score > bear_score:
                # Potential LONG
                direction = 'LONG'
                entry_reasons.append(f"Bull score {bull_score:.0f} > Bear {bear_score:.0f}")
                
                # Check POC requirement
                if r.LONG_REQUIRES_ABOVE_POC and price < poc:
                    caution_flags.append(f"Price ${price:.2f} below POC ${poc:.2f}")
                    if price < val:
                        direction = 'NO_TRADE'
                else:
                    entry_reasons.append(f"Price above POC")
                
                # Check VWAP
                if r.LONG_PREFER_ABOVE_VWAP:
                    if price > vwap:
                        entry_reasons.append("Above VWAP ✓")
                    else:
                        caution_flags.append("Below VWAP - counter-trend")
                
                # RSI filter
                if rsi > r.RSI_OVERBOUGHT:
                    caution_flags.append(f"RSI overbought ({rsi:.0f})")
                    
            elif bear_score > bull_score:
                # Potential SHORT
                direction = 'SHORT'
                entry_reasons.append(f"Bear score {bear_score:.0f} > Bull {bull_score:.0f}")
                
                # Check POC requirement
                if r.SHORT_REQUIRES_BELOW_POC and price > poc:
                    caution_flags.append(f"Price ${price:.2f} above POC ${poc:.2f}")
                    if price > vah:
                        direction = 'NO_TRADE'
                else:
                    entry_reasons.append(f"Price below POC")
                
                # Check VWAP
                if r.SHORT_PREFER_BELOW_VWAP:
                    if price < vwap:
                        entry_reasons.append("Below VWAP ✓")
                    else:
                        caution_flags.append("Above VWAP - counter-trend")
                
                # RSI filter
                if rsi < r.RSI_OVERSOLD:
                    caution_flags.append(f"RSI oversold ({rsi:.0f})")
        
        # Volume check
        if rvol < r.MIN_RVOL_FOR_ENTRY:
            caution_flags.append(f"Low volume ({rvol:.1f}x)")
        elif rvol >= r.HIGH_RVOL_BONUS:
            entry_reasons.append(f"Strong volume ({rvol:.1f}x) ✓")
        
        # =========================
        # CALCULATE LEVELS
        # =========================
        
        if direction == 'LONG':
            # Entry zone: current price to slightly above
            entry_price = price
            entry_zone_low = max(poc, vwap) if price > poc else price * 0.998
            entry_zone_low = min(entry_zone_low, price)  # Prevent inverted zone
            entry_zone_high = price * 1.005
            
            # Stop below VAL with buffer
            stop_loss = val * (1 - r.LONG_STOP_BELOW_VAL_PCT / 100)
            
            # Enforce max stop distance
            max_stop = price * (1 - r.MAX_STOP_DISTANCE_PCT / 100)
            stop_loss = max(stop_loss, max_stop)
            
            # Calculate VP-based risk BEFORE fib tightening (used for targets)
            vp_risk = price - stop_loss
            
            # FIB-ENHANCED STOP: If fib_786 (bullish) is tighter than VAL-stop and still gives room
            if fib_levels_data and fib_trend == 'UPTREND':
                fib_786 = fib_levels_data.get('bull_fib_786', 0)
                fib_618 = fib_levels_data.get('bull_fib_618', 0)
                if fib_786 > 0 and fib_786 < price:
                    # Use fib_786 as stop if it's above the VAL-stop (tighter = more confident)
                    # but still at least 0.3% below price for breathing room
                    fib_stop = fib_786 * (1 - 0.003)  # 0.3% below fib level
                    if fib_stop > stop_loss and fib_stop < price * 0.997:
                        stop_loss = fib_stop
                        fib_used_for_stop = True
                        entry_reasons.append(f"📐 Stop tightened to Fib 78.6% (${fib_786:.2f})")
            
            # Risk per share (actual stop distance for R:R display)
            risk = price - stop_loss
            
            # Targets use VP-based risk to preserve R:R even when stop is fib-tightened
            target_risk = vp_risk if fib_used_for_stop else risk
            
            # Targets - T1 must be ABOVE entry price
            if r.T1_AT_OPPOSITE_VA and vah > price * 1.005:
                # Use VAH only if it's above entry
                target_1 = vah
            else:
                # Default to 1R above entry
                target_1 = price + target_risk
            
            target_2 = price + target_risk * r.T2_R_MULTIPLE
            target_3 = price + target_risk * r.T3_R_MULTIPLE
            
            # FIB-ENHANCED TARGETS: Use fib levels as intermediate targets
            if fib_levels_data and fib_trend == 'UPTREND':
                swing_high = fib_levels_data.get('swing_high', 0)
                fib_786 = fib_levels_data.get('bull_fib_786', 0)
                # If swing high is a better T2 than R-multiple
                if swing_high > target_1 and swing_high < target_2:
                    target_2 = swing_high
                    fib_used_for_target = True
                    entry_reasons.append(f"📐 T2 at swing high (${swing_high:.2f})")
                # If fib_786 provides a good T1 alternative
                elif fib_786 > price * 1.005 and fib_786 > target_1 * 0.99 and fib_786 < target_1 * 1.02:
                    # Close to existing T1 — validates the level
                    entry_reasons.append(f"📐 T1 confirmed by Fib 78.6%")
            
            # Invalidation
            invalidation = f"Close below ${val:.2f} (VAL) invalidates the long thesis"
            
        elif direction == 'SHORT':
            # Entry zone - tighter for shorts (enter on pops)
            entry_price = price
            entry_zone_low = price * 0.998  # Only 0.2% below (tighter)
            entry_zone_high = min(poc, vwap) if price < poc else price * 1.002
            
            # Stop above VAH with buffer
            stop_loss = vah * (1 + r.SHORT_STOP_ABOVE_VAH_PCT / 100)
            
            # Enforce max stop distance
            max_stop = price * (1 + r.MAX_STOP_DISTANCE_PCT / 100)
            stop_loss = min(stop_loss, max_stop)
            
            # Calculate VP-based risk BEFORE fib tightening (used for targets)
            vp_risk = stop_loss - price
            
            # FIB-ENHANCED STOP: If bear fib_786 is tighter than VAH-stop
            if fib_levels_data and fib_trend == 'DOWNTREND':
                bear_fib_786 = fib_levels_data.get('bear_fib_786', 0)
                if bear_fib_786 > 0 and bear_fib_786 > price:
                    # Use bear fib_786 as stop if tighter than VAH-stop
                    fib_stop = bear_fib_786 * (1 + 0.003)  # 0.3% above fib level
                    if fib_stop < stop_loss and fib_stop > price * 1.003:
                        stop_loss = fib_stop
                        fib_used_for_stop = True
                        entry_reasons.append(f"📐 Stop tightened to Bear Fib 78.6% (${bear_fib_786:.2f})")
            
            # Risk per share (actual stop distance for R:R display)
            risk = stop_loss - price
            
            # Targets use VP-based risk to preserve R:R even when stop is fib-tightened
            target_risk = vp_risk if fib_used_for_stop else risk
            
            # Targets - T1 must be BELOW entry price
            if r.T1_AT_OPPOSITE_VA and val < price * 0.995:
                # Use VAL only if it's below entry
                target_1 = val
            else:
                # Default to 1R below entry
                target_1 = price - target_risk
            
            target_2 = price - target_risk * r.T2_R_MULTIPLE
            target_3 = price - target_risk * r.T3_R_MULTIPLE
            
            # FIB-ENHANCED TARGETS: Use fib levels as intermediate targets
            if fib_levels_data and fib_trend == 'DOWNTREND':
                swing_low = fib_levels_data.get('swing_low', 0)
                bear_fib_786_val = fib_levels_data.get('bear_fib_786', 0)
                # If swing low is a better T2 than R-multiple
                if swing_low > 0 and swing_low < target_1 and swing_low > target_2:
                    target_2 = swing_low
                    fib_used_for_target = True
                    entry_reasons.append(f"📐 T2 at swing low (${swing_low:.2f})")
                # If bear fib_786 validates T1
                elif bear_fib_786_val > 0 and bear_fib_786_val < price * 0.995:
                    if abs(bear_fib_786_val - target_1) / target_1 < 0.02:
                        entry_reasons.append(f"📐 T1 confirmed by Bear Fib 78.6%")
            
            # Invalidation
            invalidation = f"Close above ${vah:.2f} (VAH) invalidates the short thesis"
            
        else:
            # NO_TRADE - set neutral levels
            entry_price = price
            entry_zone_low = price
            entry_zone_high = price
            stop_loss = price
            target_1 = price
            target_2 = price
            target_3 = price
            invalidation = "No valid setup - waiting for better conditions"
        
        # =========================
        # CALCULATE RISK METRICS
        # =========================
        
        risk_per_share = abs(entry_price - stop_loss)
        reward_t1 = abs(target_1 - entry_price)
        reward_t2 = abs(target_2 - entry_price)
        
        risk_reward_t1 = reward_t1 / risk_per_share if risk_per_share > 0 else 0
        risk_reward_t2 = reward_t2 / risk_per_share if risk_per_share > 0 else 0
        
        # =========================
        # OPTIONS ADJUSTMENTS (before sizing so confidence is factored in)
        # =========================
        
        adjusted_confidence = max_score
        
        if options_sentiment and direction != 'NO_TRADE':
            # Options confirmation/conflict
            if direction == 'LONG' and options_sentiment == 'BULLISH':
                adjusted_confidence += r.OPTIONS_CONFIRM_BONUS
                entry_reasons.append(f"Options confirm LONG (+{r.OPTIONS_CONFIRM_BONUS}% confidence)")
            elif direction == 'SHORT' and options_sentiment == 'BEARISH':
                adjusted_confidence += r.OPTIONS_CONFIRM_BONUS
                entry_reasons.append(f"Options confirm SHORT (+{r.OPTIONS_CONFIRM_BONUS}% confidence)")
            elif direction == 'LONG' and options_sentiment == 'BEARISH':
                adjusted_confidence -= r.OPTIONS_CONFLICT_PENALTY
                caution_flags.append(f"Options conflict with LONG (-{r.OPTIONS_CONFLICT_PENALTY}% confidence)")
            elif direction == 'SHORT' and options_sentiment == 'BULLISH':
                adjusted_confidence -= r.OPTIONS_CONFLICT_PENALTY
                caution_flags.append(f"Options conflict with SHORT (-{r.OPTIONS_CONFLICT_PENALTY}% confidence)")
            
            # Warn if target beyond expected move
            if expected_move and r.WARN_IF_TARGET_BEYOND_EXPECTED:
                if direction == 'LONG' and target_1 > price + expected_move:
                    caution_flags.append(f"T1 beyond expected move (±${expected_move:.2f})")
                elif direction == 'SHORT' and target_1 < price - expected_move:
                    caution_flags.append(f"T1 beyond expected move (±${expected_move:.2f})")
            
            # Adjust targets based on walls
            if direction == 'LONG' and call_wall and r.USE_CALL_WALL_AS_RESISTANCE:
                if target_1 > call_wall and call_wall > price:
                    caution_flags.append(f"T1 above call wall ${call_wall:.2f}")
                    # Optionally cap T1 at call wall
                    # target_1 = min(target_1, call_wall)
            
            if direction == 'SHORT' and put_wall and r.USE_PUT_WALL_AS_SUPPORT:
                if target_1 < put_wall and put_wall < price:
                    caution_flags.append(f"T1 below put wall ${put_wall:.2f}")
        
        # =========================
        # POSITION SIZING (uses adjusted_confidence from options)
        # =========================
        
        sizing_score = adjusted_confidence  # Use post-options score for sizing
        
        if direction == 'NO_TRADE':
            position_size_pct = 0
            risk_pct = 0
        elif sizing_score >= r.MIN_SCORE_FULL_SIZE:
            position_size_pct = r.HIGH_SCORE_RISK_MULT
            risk_pct = r.BASE_RISK_PCT * position_size_pct
            entry_reasons.append(f"Full size - score {sizing_score:.0f} ≥ {r.MIN_SCORE_FULL_SIZE}")
        else:
            position_size_pct = r.MED_SCORE_RISK_MULT
            risk_pct = r.BASE_RISK_PCT * position_size_pct
            caution_flags.append(f"Half size - score {sizing_score:.0f} < {r.MIN_SCORE_FULL_SIZE}")
        
        # High IV reduces size (options-dependent)
        if options_sentiment and direction != 'NO_TRADE':
            if avg_iv and avg_iv > r.HIGH_IV_THRESHOLD and r.HIGH_IV_REDUCE_SIZE:
                position_size_pct *= 0.5
                risk_pct *= 0.5
                caution_flags.append(f"Half size due to high IV ({avg_iv:.0f}%)")
        
        # =========================
        # LEVEL VALIDATION
        # =========================
        # Ensure targets are at minimum 1R away from entry
        
        if direction == 'LONG':
            # For LONG: T1 must be at least 1R ABOVE entry midpoint
            entry_mid = (entry_zone_low + entry_zone_high) / 2
            min_target_1 = entry_mid + risk_per_share
            
            if target_1 <= min_target_1:
                # T1 too close - push it up to at least 1R above entry
                old_t1 = target_1
                target_1 = min_target_1
                caution_flags.append(f"T1 adjusted to 1R above entry (was ${old_t1:.2f})")
            
            if target_2 <= target_1:
                target_2 = target_1 + risk_per_share
            if target_3 <= target_2:
                target_3 = target_2 + risk_per_share
            # Entry zone low should not go below stop
            if entry_zone_low <= stop_loss:
                entry_zone_low = stop_loss + 0.01
                
        elif direction == 'SHORT':
            # For SHORT: T1 must be at least 1R BELOW entry midpoint
            entry_mid = (entry_zone_low + entry_zone_high) / 2
            min_target_1 = entry_mid - risk_per_share
            
            if target_1 >= min_target_1:
                # T1 too close - push it down to at least 1R below entry
                old_t1 = target_1
                target_1 = min_target_1
                caution_flags.append(f"T1 adjusted to 1R below entry (was ${old_t1:.2f})")
            
            if target_2 >= target_1:
                target_2 = target_1 - risk_per_share
            if target_3 >= target_2:
                target_3 = target_2 - risk_per_share
            # Entry zone high should not go above stop
            if entry_zone_high >= stop_loss:
                entry_zone_high = stop_loss - 0.01
        
        # =========================
        # BUILD PLAN
        # =========================
        
        plan = TradePlan(
            symbol=symbol,
            direction=direction,
            confidence=min(100, adjusted_confidence),  # Cap at 100
            
            entry_price=round(entry_price, 2),
            entry_zone_low=round(entry_zone_low, 2),
            entry_zone_high=round(entry_zone_high, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            target_3=round(target_3, 2),
            
            risk_per_share=round(risk_per_share, 2),
            risk_reward_t1=round(risk_reward_t1, 2),
            risk_reward_t2=round(risk_reward_t2, 2),
            
            position_size_pct=position_size_pct,
            risk_pct=risk_pct,
            
            entry_reasons=entry_reasons,
            caution_flags=caution_flags,
            invalidation=invalidation,
            
            # Options data
            options_data=options_data,
            options_sentiment=options_sentiment,
            pc_ratio=pc_ratio,
            max_pain=max_pain,
            call_wall=call_wall,
            put_wall=put_wall,
            expected_move=round(expected_move, 2) if expected_move else None,
            avg_iv=round(avg_iv, 1) if avg_iv else None,
            
            # OI scoring data
            call_wall_oi=call_wall_oi_val,
            put_wall_oi=put_wall_oi_val,
            oi_skew=round(oi_skew, 2) if oi_skew else None,
            oi_skew_sentiment=oi_skew_sentiment,
            unusual_call_count=unusual_calls,
            unusual_put_count=unusual_puts,
            unusual_activity_sentiment=unusual_activity_sentiment,
            
            timestamp=datetime.now().isoformat(),
            scanner_data=scanner_result,
            
            # Fibonacci data
            fib_zone=fib_zone,
            fib_quality=fib_quality,
            fib_trend=fib_trend,
            fib_position=fib_position,
            fib_confluence=fib_confluence_list if fib_confluence_list else None,
            fib_levels=fib_levels_data if fib_levels_data else None,
            fib_used_for_stop=fib_used_for_stop,
            fib_used_for_target=fib_used_for_target
        )
        
        return plan
    
    def format_plan_text(self, plan: TradePlan) -> str:
        """Format plan as readable text"""
        
        if plan.direction == 'NO_TRADE':
            return f"""❌ NO TRADE - {plan.symbol}

Reasons:
{chr(10).join('• ' + c for c in plan.caution_flags)}

{plan.invalidation}
"""
        
        emoji = '🟢' if plan.direction == 'LONG' else '🔴'
        
        return f"""{emoji} {plan.direction} {plan.symbol} @ ${plan.entry_price:.2f}

📊 CONFIDENCE: {plan.confidence:.0f}%
📏 SIZE: {plan.position_size_pct * 100:.0f}% (risking {plan.risk_pct:.1f}% of account)

📍 LEVELS:
• Entry Zone: ${plan.entry_zone_low:.2f} - ${plan.entry_zone_high:.2f}
• Stop Loss: ${plan.stop_loss:.2f} (${plan.risk_per_share:.2f} risk/share)
• Target 1: ${plan.target_1:.2f} ({plan.risk_reward_t1:.1f}R)
• Target 2: ${plan.target_2:.2f} ({plan.risk_reward_t2:.1f}R)
• Target 3: ${plan.target_3:.2f}

📐 Fibonacci:
• Zone: {plan.fib_zone or 'N/A'} | Quality: {plan.fib_quality or 'N/A'} | Trend: {plan.fib_trend or 'N/A'}
{('• Fib used for stop ✔' if plan.fib_used_for_stop else '')}{('• Fib used for target ✔' if plan.fib_used_for_target else '')}
{('• Confluence: ' + '; '.join(plan.fib_confluence)) if plan.fib_confluence else ''}

📊 OI ANALYSIS:
• Call Wall: ${f'{plan.call_wall:.0f}' if plan.call_wall else 'N/A'} (OI: {f'{plan.call_wall_oi:,}' if plan.call_wall_oi else 'N/A'})
• Put Wall: ${f'{plan.put_wall:.0f}' if plan.put_wall else 'N/A'} (OI: {f'{plan.put_wall_oi:,}' if plan.put_wall_oi else 'N/A'})
• OI Skew: {f'{plan.oi_skew:.2f}' if plan.oi_skew else 'N/A'} ({plan.oi_skew_sentiment or 'N/A'})
• Unusual Activity: {f'Calls={plan.unusual_call_count} Puts={plan.unusual_put_count}' if plan.unusual_call_count is not None else 'N/A'} ({plan.unusual_activity_sentiment or 'N/A'})

✅ ENTRY REASONS:
{chr(10).join('• ' + r for r in plan.entry_reasons)}

⚠️ WATCH FOR:
{chr(10).join('• ' + c for c in plan.caution_flags) if plan.caution_flags else '• No major concerns'}

🚫 INVALIDATION:
{plan.invalidation}
"""


# =============================================================================
# LEARNING DATABASE - TRAIN AS YOU GO
# =============================================================================

class LearningDatabase:
    """
    Stores trade plans and outcomes to learn from results.
    Tracks what rules work best in different conditions.
    """
    
    def __init__(self, db_path: str = "trade_data/learning.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Trade plans table
        c.execute('''
            CREATE TABLE IF NOT EXISTS trade_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                target_1 REAL,
                target_2 REAL,
                target_3 REAL,
                risk_reward_t1 REAL,
                position_size_pct REAL,
                scanner_data TEXT,
                entry_reasons TEXT,
                caution_flags TEXT,
                
                -- Options data
                options_sentiment TEXT,
                pc_ratio REAL,
                max_pain REAL,
                call_wall REAL,
                put_wall REAL,
                expected_move REAL,
                avg_iv REAL,
                options_data TEXT,
                
                -- Full report markdown
                full_report TEXT,
                
                -- Outcome tracking (updated later)
                outcome TEXT,           -- 'WIN_T1', 'WIN_T2', 'WIN_T3', 'LOSS', 'SCRATCH', 'SKIPPED'
                actual_entry REAL,
                actual_exit REAL,
                actual_r_multiple REAL,
                exit_reason TEXT,
                notes TEXT,
                outcome_timestamp TEXT
            )
        ''')
        
        # Rule performance tracking
        c.execute('''
            CREATE TABLE IF NOT EXISTS rule_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT UNIQUE,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_r REAL DEFAULT 0,
                avg_winner_r REAL DEFAULT 0,
                avg_loser_r REAL DEFAULT 0,
                last_updated TEXT
            )
        ''')
        
        # Pattern learning
        c.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_signature TEXT UNIQUE,
                description TEXT,
                occurrences INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                avg_r REAL DEFAULT 0,
                best_conditions TEXT,
                worst_conditions TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_plan(self, plan: TradePlan, full_report: str = None) -> int:
        """Save a trade plan, returns plan ID for later outcome tracking"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO trade_plans (
                timestamp, symbol, direction, confidence,
                entry_price, stop_loss, target_1, target_2, target_3,
                risk_reward_t1, position_size_pct,
                scanner_data, entry_reasons, caution_flags,
                options_sentiment, pc_ratio, max_pain, call_wall, put_wall,
                expected_move, avg_iv, options_data, full_report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            plan.timestamp, plan.symbol, plan.direction, plan.confidence,
            plan.entry_price, plan.stop_loss, plan.target_1, plan.target_2, plan.target_3,
            plan.risk_reward_t1, plan.position_size_pct,
            json.dumps(plan.scanner_data),
            json.dumps(plan.entry_reasons),
            json.dumps(plan.caution_flags),
            plan.options_sentiment,
            plan.pc_ratio,
            plan.max_pain,
            plan.call_wall,
            plan.put_wall,
            plan.expected_move,
            plan.avg_iv,
            json.dumps(plan.options_data) if plan.options_data else None,
            full_report or plan.full_report
        ))
        
        plan_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return plan_id
    
    def record_outcome(self, plan_id: int, outcome: str, actual_entry: float = None,
                       actual_exit: float = None, actual_r: float = None,
                       exit_reason: str = None, notes: str = None):
        """
        Record the outcome of a trade.
        
        Args:
            plan_id: ID returned from save_plan()
            outcome: 'WIN_T1', 'WIN_T2', 'WIN_T3', 'LOSS', 'SCRATCH', 'SKIPPED'
            actual_entry: Actual entry price
            actual_exit: Actual exit price
            actual_r: Actual R-multiple achieved
            exit_reason: Why you exited
            notes: Any additional notes
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE trade_plans SET
                outcome = ?,
                actual_entry = ?,
                actual_exit = ?,
                actual_r_multiple = ?,
                exit_reason = ?,
                notes = ?,
                outcome_timestamp = ?
            WHERE id = ?
        ''', (outcome, actual_entry, actual_exit, actual_r, exit_reason, notes,
              datetime.now().isoformat(), plan_id))
        
        conn.commit()
        conn.close()
        
        # Update pattern stats
        self._update_pattern_stats(plan_id, outcome, actual_r)
    
    def _update_pattern_stats(self, plan_id: int, outcome: str, actual_r: float):
        """Update learning stats based on outcome - includes options correlation"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get the plan details including options data
        c.execute('''SELECT scanner_data, direction, confidence, 
                     options_sentiment, pc_ratio, avg_iv 
                     FROM trade_plans WHERE id = ?''', (plan_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return
        
        scanner_data = json.loads(row[0])
        direction = row[1]
        confidence = row[2]
        options_sentiment = row[3]  # BULLISH, BEARISH, NEUTRAL
        pc_ratio = row[4]
        avg_iv = row[5]
        
        # Create pattern signature
        # e.g., "LONG_HIGH_SCORE_ABOVE_VWAP_BULLISH_OPTIONS_HIGH_IV"
        parts = [direction]
        if confidence >= 70:
            parts.append("HIGH_SCORE")
        elif confidence >= 50:
            parts.append("MED_SCORE")
        
        price = scanner_data.get('current_price') or scanner_data.get('price', 0)
        vwap = scanner_data.get('vwap', price)
        if price > vwap:
            parts.append("ABOVE_VWAP")
        else:
            parts.append("BELOW_VWAP")
        
        rvol = scanner_data.get('rvol', 1.0)
        if rvol >= 1.5:
            parts.append("HIGH_RVOL")
        elif rvol <= 0.7:
            parts.append("LOW_RVOL")
        
        # Add fib zone to pattern
        fib_zone = scanner_data.get('fib_zone')
        if fib_zone:
            parts.append(f"FIB_{fib_zone.upper()}")
        
        # Add options conditions to pattern
        if options_sentiment:
            parts.append(f"{options_sentiment}_OPTIONS")
            
            # Add IV condition
            if avg_iv and avg_iv > 50:
                parts.append("HIGH_IV")
            elif avg_iv and avg_iv > 30:
                parts.append("MED_IV")
            elif avg_iv:
                parts.append("LOW_IV")
            
            # Add P/C ratio extremes
            if pc_ratio and pc_ratio < 0.5:
                parts.append("EXTREME_CALLS")
            elif pc_ratio and pc_ratio > 2.0:
                parts.append("EXTREME_PUTS")
        
        pattern = "_".join(parts)
        is_win = outcome.startswith('WIN')
        r = actual_r or 0
        
        # Update or insert pattern
        c.execute('''
            INSERT INTO patterns (pattern_signature, occurrences, wins, avg_r)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(pattern_signature) DO UPDATE SET
                occurrences = occurrences + 1,
                wins = wins + ?,
                avg_r = (avg_r * (occurrences - 1) + ?) / occurrences
        ''', (pattern, 1 if is_win else 0, r, 1 if is_win else 0, r))
        
        conn.commit()
        conn.close()
    
    def get_pattern_stats(self) -> List[Dict]:
        """Get performance stats for each pattern"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT pattern_signature, occurrences, wins, avg_r,
                   CAST(wins AS FLOAT) / NULLIF(occurrences, 0) as win_rate
            FROM patterns
            WHERE occurrences >= 3
            ORDER BY avg_r DESC
        ''')
        
        results = []
        for row in c.fetchall():
            results.append({
                'pattern': row[0],
                'trades': row[1],
                'wins': row[2],
                'avg_r': row[3],
                'win_rate': row[4] or 0
            })
        
        conn.close()
        return results
    
    def get_recent_plans(self, limit: int = 20) -> List[Dict]:
        """Get recent trade plans with outcomes"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT id, timestamp, symbol, direction, confidence,
                   entry_price, target_1, stop_loss, outcome, actual_r_multiple
            FROM trade_plans
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in c.fetchall():
            results.append({
                'id': row[0],
                'timestamp': row[1],
                'symbol': row[2],
                'direction': row[3],
                'confidence': row[4],
                'entry': row[5],
                'target': row[6],
                'stop': row[7],
                'outcome': row[8],
                'r_multiple': row[9]
            })
        
        conn.close()
        return results
    
    def get_stats_summary(self) -> Dict:
        """Get overall performance stats"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN outcome LIKE 'WIN%' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN outcome IS NOT NULL THEN actual_r_multiple END) as avg_r,
                SUM(actual_r_multiple) as total_r
            FROM trade_plans
            WHERE outcome IS NOT NULL AND outcome != 'SKIPPED'
        ''')
        
        row = c.fetchone()
        conn.close()
        
        total = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total if total > 0 else 0,
            'avg_r': row[3] or 0,
            'total_r': row[4] or 0
        }


# =============================================================================
# REPORT KNOWLEDGE BASE - Learn from your analysis reports
# =============================================================================

class ReportKnowledgeBase:
    """
    Reads and indexes your analysis reports to enhance AI context.
    The AI learns your analysis style and terminology.
    """
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports = []
        self.analysis_patterns = {}
        self._load_reports()
    
    def _load_reports(self):
        """Load all markdown reports from the reports directory"""
        if not self.reports_dir.exists():
            return
        
        for report_file in self.reports_dir.glob("*.md"):
            try:
                content = report_file.read_text(encoding='utf-8')
                symbol = self._extract_symbol(report_file.name, content)
                self.reports.append({
                    'filename': report_file.name,
                    'symbol': symbol,
                    'content': content,
                    'date': self._extract_date(content),
                    'signal': self._extract_signal(content),
                    'key_levels': self._extract_levels(content),
                    'setups': self._extract_setups(content),
                    'risks': self._extract_risks(content)
                })
            except Exception as e:
                print(f"Warning: Could not load report {report_file}: {e}")
    
    def _extract_symbol(self, filename: str, content: str) -> str:
        """Extract ticker symbol from filename or content"""
        # Try filename first (e.g., META_Analysis_2026-01-25.md)
        parts = filename.split('_')
        if parts:
            return parts[0].upper()
        # Fallback to content header
        if content.startswith('# '):
            first_line = content.split('\n')[0]
            words = first_line.replace('#', '').strip().split()
            if words:
                return words[0].upper()
        return 'UNKNOWN'
    
    def _extract_date(self, content: str) -> str:
        """Extract report date"""
        import re
        date_match = re.search(r'\*\*Generated:\*\*\s*([^|]+)', content)
        if date_match:
            return date_match.group(1).strip()
        return ''
    
    def _extract_signal(self, content: str) -> str:
        """Extract the main signal from report"""
        import re
        signal_match = re.search(r'\*\*Signal:\*\*\s*(.+)', content)
        if signal_match:
            return signal_match.group(1).strip()
        return 'UNKNOWN'
    
    def _extract_levels(self, content: str) -> Dict:
        """Extract key price levels from report"""
        import re
        levels = {}
        
        # Find VAH, POC, VAL patterns
        vah_match = re.search(r'VAH\s*\|\s*\$?([\d,.]+)', content)
        poc_match = re.search(r'POC\s*\|\s*\$?([\d,.]+)', content)
        val_match = re.search(r'VAL\s*\|\s*\$?([\d,.]+)', content)
        
        if vah_match:
            levels['vah'] = float(vah_match.group(1).replace(',', ''))
        if poc_match:
            levels['poc'] = float(poc_match.group(1).replace(',', ''))
        if val_match:
            levels['val'] = float(val_match.group(1).replace(',', ''))
        
        return levels
    
    def _extract_setups(self, content: str) -> List[str]:
        """Extract trade setups mentioned in report"""
        setups = []
        
        # Look for Long Setup and Short Setup sections
        if '### Long Setup' in content:
            setups.append('LONG')
        if '### Short Setup' in content:
            setups.append('SHORT')
        
        return setups
    
    def _extract_risks(self, content: str) -> List[str]:
        """Extract risk factors from report"""
        risks = []
        
        # Find ## Risk Factors section
        if '## Risk Factors' in content:
            risk_section = content.split('## Risk Factors')[1].split('##')[0]
            # Extract numbered items
            import re
            risk_items = re.findall(r'\d+\.\s*\*\*([^*]+)\*\*', risk_section)
            risks.extend(risk_items[:5])  # Top 5 risks
        
        return risks
    
    def get_context_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get report context for a specific symbol"""
        for report in self.reports:
            if report['symbol'] == symbol.upper():
                return report
        return None
    
    def get_analysis_style_prompt(self) -> str:
        """
        Generate a prompt section that teaches the AI your analysis style
        based on your reports.
        """
        if not self.reports:
            return ""
        
        # Analyze common patterns across reports
        sample_report = self.reports[0]
        
        style_prompt = """
YOUR ANALYSIS STYLE (learned from reports):
- Use professional trader terminology
- Structure analysis with clear sections: Technical, Volume Profile, Scenarios
- Always include specific price levels for entry, stop, targets
- Consider multiple timeframes (1HR, Daily)
- Include risk/reward calculations
- Mention key catalysts (earnings, etc.)
- Use tables for data when appropriate
- End with clear "Recommended Action" section
- Key levels format: $XXX.XX with significance note
"""
        
        # Add symbol-specific context if available
        symbols_analyzed = [r['symbol'] for r in self.reports]
        if symbols_analyzed:
            style_prompt += f"\nSymbols you've analyzed: {', '.join(symbols_analyzed)}"
        
        return style_prompt
    
    def get_report_summary(self, symbol: str) -> str:
        """Get a summary of the report for AI context"""
        report = self.get_context_for_symbol(symbol)
        if not report:
            return ""
        
        summary = f"""
EXISTING ANALYSIS FOR {symbol}:
- Signal: {report['signal']}
- Key Levels: VAH ${report['key_levels'].get('vah', 'N/A')}, POC ${report['key_levels'].get('poc', 'N/A')}, VAL ${report['key_levels'].get('val', 'N/A')}
- Setups Identified: {', '.join(report['setups']) if report['setups'] else 'None'}
- Top Risks: {', '.join(report['risks'][:3]) if report['risks'] else 'None identified'}
"""
        return summary


# =============================================================================
# AI EXPLANATION LAYER - Adds context without changing levels
# =============================================================================

class AIExplainer:
    """
    Uses AI to explain the trade plan in natural language.
    Does NOT change any levels - just adds context and reasoning.
    Now enhanced with knowledge from your analysis reports.
    """
    
    def __init__(self):
        self.anthropic_client = None
        self.openai_client = None
        self.knowledge_base = ReportKnowledgeBase()
        
        # Try to init API clients
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            except:
                pass
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
            except:
                pass
    
    def explain(self, plan: TradePlan, pattern_stats: List[Dict] = None, past_reports: List[Dict] = None) -> str:
        """
        Generate AI explanation for the trade plan.
        
        The AI explains WHY, it never changes the WHAT.
        Now includes context from Firestore reports passed from frontend.
        """
        if not self.anthropic_client and not self.openai_client:
            return self._simple_explanation(plan)
        
        # Build context for AI
        pattern_context = ""
        if pattern_stats:
            relevant = [p for p in pattern_stats if plan.direction in p['pattern']][:3]
            if relevant:
                pattern_context = "\n\nHISTORICAL PATTERNS FROM YOUR TRADES:\n"
                for p in relevant:
                    pattern_context += f"- {p['pattern']}: {p['trades']} trades, {p['win_rate']*100:.0f}% win rate, {p['avg_r']:.1f}R avg\n"
        
        # Build past reports context from Firestore data
        report_context = ""
        if past_reports and len(past_reports) > 0:
            report_context = f"\n\nPAST ANALYSIS FOR {plan.symbol} (from your saved reports):\n"
            for r in past_reports[:5]:  # Max 5 reports
                direction = r.get('direction', '?')
                bull = r.get('bull_score', 0)
                bear = r.get('bear_score', 0)
                price = r.get('price', 0)
                date = r.get('date', 'unknown')
                notes = r.get('notes', [])
                report_context += f"- {date}: {direction} bias (Bull:{bull:.0f}/Bear:{bear:.0f}) @ ${price:.2f}"
                if notes:
                    report_context += f" - {', '.join(notes[:2])}"
                report_context += "\n"
            report_context += f"\nTotal past scans for {plan.symbol}: {len(past_reports)}\n"
        else:
            # Fallback to local report knowledge base
            report_context = self.knowledge_base.get_report_summary(plan.symbol)
        
        # Get your analysis style
        style_context = self.knowledge_base.get_analysis_style_prompt()
        
        prompt = f"""You are explaining a trade plan to a trader. The levels are FIXED - do not suggest different levels.
Your job is to explain the reasoning and add market context.

{style_context}

TRADE PLAN (LEVELS ARE FIXED):
{self._format_plan_for_ai(plan)}

{report_context}

{pattern_context}

In 2-3 sentences:
1. Explain why this setup makes sense (or doesn't if NO_TRADE)
2. What market conditions would make this work best
3. Reference any relevant info from the existing analysis if available

Be concise and direct. Trader language only. Match the analysis style from reports."""

        try:
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                return response.choices[0].message.content
        except Exception as e:
            return self._simple_explanation(plan)
    
    def _format_plan_for_ai(self, plan: TradePlan) -> str:
        """Format plan for AI prompt"""
        return f"""Symbol: {plan.symbol}
Direction: {plan.direction}
Confidence: {plan.confidence:.0f}%
Entry: ${plan.entry_price:.2f}
Stop: ${plan.stop_loss:.2f}
Target 1: ${plan.target_1:.2f} ({plan.risk_reward_t1:.1f}R)
Entry Reasons: {', '.join(plan.entry_reasons)}
Cautions: {', '.join(plan.caution_flags) if plan.caution_flags else 'None'}"""
    
    def _simple_explanation(self, plan: TradePlan) -> str:
        """Fallback explanation without AI"""
        if plan.direction == 'NO_TRADE':
            return "Setup doesn't meet criteria. Wait for better conditions."
        
        reasons = ' '.join(plan.entry_reasons[:2])
        caution = plan.caution_flags[0] if plan.caution_flags else "No major concerns"
        
        return f"{reasons}. Watch for: {caution}. Invalidated if stop is hit."


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_plan(scanner_result: Dict, explain: bool = True, save: bool = True, past_reports: List[Dict] = None, options_data: Dict = None) -> Tuple[TradePlan, str, int]:
    """
    Main entry point - generate a trade plan from scanner data.
    
    Args:
        scanner_result: Dict from scanner
        explain: Whether to generate AI explanation
        save: Whether to save to learning database
        past_reports: List of past reports from Firestore for AI context
        options_data: Options chain data from Tradier for confidence adjustment
        
    Returns:
        (TradePlan, explanation_text, plan_id)
    """
    engine = RuleEngine()
    plan = engine.generate_plan(scanner_result, options_data=options_data)
    
    # Save to DB
    plan_id = 0
    if save:
        plan_id = engine.learning_db.save_plan(plan)
    
    # Generate explanation with past report context
    explanation = ""
    if explain:
        explainer = AIExplainer()
        pattern_stats = engine.learning_db.get_pattern_stats()
        explanation = explainer.explain(plan, pattern_stats, past_reports=past_reports or [])
    
    return plan, explanation, plan_id


def record_trade_outcome(plan_id: int, outcome: str, actual_r: float = None, notes: str = None):
    """
    Record the outcome of a trade.
    
    Args:
        plan_id: ID from generate_plan()
        outcome: 'WIN_T1', 'WIN_T2', 'WIN_T3', 'LOSS', 'SCRATCH', 'SKIPPED'
        actual_r: Actual R-multiple achieved
        notes: Any notes
    """
    db = LearningDatabase()
    db.record_outcome(plan_id, outcome, actual_r=actual_r, notes=notes)


def get_learning_stats() -> Dict:
    """Get learning statistics"""
    db = LearningDatabase()
    return {
        'summary': db.get_stats_summary(),
        'patterns': db.get_pattern_stats(),
        'recent': db.get_recent_plans()
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with sample scanner data
    sample_scan = {
        'symbol': 'AAPL',
        'current_price': 185.50,
        'vah': 188.00,
        'poc': 186.00,
        'val': 183.50,
        'vwap': 185.00,
        'bull_score': 72,
        'bear_score': 35,
        'rsi': 58,
        'rvol': 1.8,
        'confidence': 72,
        'direction': 'long'
    }
    
    plan, explanation, plan_id = generate_plan(sample_scan, explain=False)
    
    engine = RuleEngine()
    print(engine.format_plan_text(plan))
    print(f"\n📝 Saved as Plan #{plan_id}")
    print(f"\n🤖 AI: {explanation if explanation else 'No AI available'}")
