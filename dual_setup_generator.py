"""
Dual Setup Generator - Zero API Cost Alternative to GPT

Outputs the same format as GPT (LONG + SHORT setups, grades, probabilities)
using deterministic rules. No API calls needed.

Rules are based on:
- Volume Profile levels (VAH/POC/VAL)
- VWAP position
- RSI zones
- Extension/compression states
- Order flow data
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class Setup:
    """Single direction setup"""
    direction: str  # LONG or SHORT
    grade: str  # A+, A, B, C, F
    conviction: int  # 1-10
    probability_low: int
    probability_high: int
    probability_label: str  # High, Medium, Low
    entry_low: float
    entry_high: float
    aggressive_entry: float
    stop: float
    target_1: float
    target_2: float
    risk_reward: float
    ev: float
    trigger: str
    invalidation: str
    why: str


@dataclass  
class DualSetupResult:
    """Complete dual setup output"""
    symbol: str
    current_price: float
    
    # Setups
    long_setup: Setup
    short_setup: Setup
    
    # Verdict
    preferred_direction: str
    verdict_reason: str
    key_level: float
    key_level_desc: str
    
    # Bookmap checklist
    bookmap_long: str
    bookmap_short: str
    
    # Options strategy (hedged play)
    options_strategy: Optional[Dict] = None


@dataclass
class OptionsStrategy:
    """Hedged options play for a setup"""
    direction: str  # LONG or SHORT bias
    
    # Primary leg (directional)
    primary_type: str  # CALL or PUT
    primary_strike: float
    primary_expiry: str  # "4 weeks"
    primary_delta: str  # "0.50-0.60 ATM"
    
    # Hedge leg (protection)
    hedge_type: str  # PUT or CALL (opposite of primary)
    hedge_strike: float
    hedge_expiry: str  # "2 weeks"
    hedge_delta: str  # "0.30-0.40 OTM"
    
    # Leverage & Risk
    leverage_ratio: str  # "2:1 favor primary"
    max_loss: str
    breakeven: str
    
    # Management rules
    profit_target: str
    cut_loss: str
    reposition_signal: str


class DualSetupGenerator:
    """
    Generates LONG and SHORT setups using deterministic rules.
    Same output format as GPT but with zero API cost.
    """
    
    def __init__(self):
        # Probability base rates (from your prompt)
        self.BASE_VAH_REJECTION = 65  # First test from below
        self.BASE_VAL_BOUNCE = 65     # First test from above
        self.BASE_POC_RETURN = 70     # Price returns to POC
        self.RETEST_DECAY = -15       # After 2+ tests
        
        # Adjustments
        self.MTF_ALIGNED_BONUS = 15
        self.VWAP_TREND_BONUS = 5
        self.EXTENDED_PENALTY = -10
        self.REJECTION_CANDLE_BONUS = 10
        
        # Grading thresholds
        self.GRADE_THRESHOLDS = {
            'A+': {'min_prob': 75, 'min_rr': 3.0},
            'A': {'min_prob': 70, 'min_rr': 2.5},
            'B': {'min_prob': 60, 'min_rr': 2.0},
            'C': {'min_prob': 50, 'min_rr': 1.5},
            'F': {'min_prob': 0, 'min_rr': 0}
        }
    
    def generate(self, data: Dict) -> DualSetupResult:
        """
        Generate dual setups from scanner data.
        
        Args:
            data: Dict with symbol, current_price, vah, poc, val, vwap, 
                  bull_score, bear_score, rsi, rvol, atr, order_flow, etc.
        
        Returns:
            DualSetupResult with both setups
        """
        symbol = data.get('symbol', 'UNKNOWN')
        price = float(data.get('current_price') or data.get('price') or 0)
        vah = float(data.get('vah') or 0)
        poc = float(data.get('poc') or 0)
        val = float(data.get('val') or 0)
        vwap = float(data.get('vwap') or poc)
        
        bull_score = float(data.get('bull_score') or 0)
        bear_score = float(data.get('bear_score') or 0)
        rsi = float(data.get('rsi') or 50)
        rvol = float(data.get('rvol') or 1.0)
        atr = float(data.get('atr') or (vah - val) * 0.3)  # Estimate ATR if not provided
        
        order_flow = data.get('order_flow') or {}
        
        # Calculate position in value area
        if vah > val:
            va_range = vah - val
            if va_range > 0:
                position_pct = (price - val) / va_range * 100
            else:
                position_pct = 50
        else:
            position_pct = 50
        
        # Determine price zone
        above_vah = price > vah
        below_val = price < val
        above_vwap = price > vwap
        above_poc = price > poc
        in_value = val <= price <= vah
        
        # Generate both setups
        long_setup = self._generate_long_setup(
            price, vah, poc, val, vwap, atr,
            bull_score, bear_score, rsi, rvol,
            position_pct, above_vwap, order_flow
        )
        
        short_setup = self._generate_short_setup(
            price, vah, poc, val, vwap, atr,
            bull_score, bear_score, rsi, rvol,
            position_pct, above_vwap, order_flow
        )
        
        # Determine verdict
        preferred, verdict_reason = self._determine_verdict(
            long_setup, short_setup, 
            price, vah, poc, val, vwap,
            bull_score, bear_score, order_flow
        )
        
        # Key decision level
        key_level = self._calculate_key_level(price, vah, poc, val, vwap)
        
        # Bookmap checklist
        bookmap_long = self._generate_bookmap_checklist('LONG', long_setup, vah, poc, val)
        bookmap_short = self._generate_bookmap_checklist('SHORT', short_setup, vah, poc, val)
        
        # Generate hedged options strategy for the preferred direction
        options_strategy = self._generate_options_strategy(
            preferred, price, atr, 
            long_setup if preferred == 'LONG' else short_setup,
            short_setup if preferred == 'LONG' else long_setup
        )
        
        return DualSetupResult(
            symbol=symbol,
            current_price=price,
            long_setup=long_setup,
            short_setup=short_setup,
            preferred_direction=preferred,
            verdict_reason=verdict_reason,
            key_level=key_level,
            key_level_desc=f"${key_level:.2f} - Above = Long, Below = Short",
            bookmap_long=bookmap_long,
            bookmap_short=bookmap_short,
            options_strategy=options_strategy
        )
    
    def _generate_long_setup(self, price, vah, poc, val, vwap, atr,
                              bull_score, bear_score, rsi, rvol,
                              position_pct, above_vwap, order_flow) -> Setup:
        """Generate LONG setup"""
        
        # Calculate probability
        prob = self._calculate_long_probability(
            price, vah, poc, val, vwap,
            bull_score, bear_score, rsi, rvol,
            position_pct, above_vwap, order_flow
        )
        
        # Entry levels (conservative = pullback to support)
        if price > vah:
            # Extended above - entry on pullback to VAH
            entry_low = vah
            entry_high = vah + atr * 0.2
            aggressive = price
        elif price > poc:
            # Above POC - entry on pullback to POC
            entry_low = poc
            entry_high = poc + atr * 0.3
            aggressive = price
        else:
            # Near VAL - entry at current or VAL
            entry_low = val
            entry_high = min(price, poc)
            aggressive = price
        
        # Stop loss (below key support)
        if price > poc:
            stop = poc - atr * 0.5
        else:
            stop = val - atr * 0.5
        
        # Ensure stop is reasonable
        stop = max(stop, price * 0.95)  # Max 5% stop
        
        # Targets
        target_1 = vah if price < vah else vah + atr
        target_2 = vah + atr * 1.5 if price < vah else vah + atr * 2
        
        # Ensure targets above entry
        if target_1 <= entry_high:
            target_1 = entry_high + atr * 0.5
        if target_2 <= target_1:
            target_2 = target_1 + atr
        
        # Risk/Reward
        risk = entry_high - stop if entry_high > stop else atr * 0.5
        reward = target_1 - entry_high
        rr = reward / risk if risk > 0 else 0
        
        # Expected Value
        ev = (prob / 100 * reward) - ((100 - prob) / 100 * risk)
        
        # Grade
        grade, conviction = self._calculate_grade(prob, rr, bull_score, bear_score, 'LONG')
        
        # Probability label
        if prob >= 65:
            prob_label = "High"
        elif prob >= 55:
            prob_label = "Medium"
        else:
            prob_label = "Low"
        
        # Generate trigger and invalidation
        trigger = self._generate_long_trigger(price, vah, poc, val, vwap, rvol)
        invalidation = self._generate_long_invalidation(price, vah, poc, val, stop)
        why = self._generate_long_why(price, vah, poc, val, vwap, rsi, bull_score)
        
        return Setup(
            direction='LONG',
            grade=grade,
            conviction=conviction,
            probability_low=int(prob - 5),
            probability_high=int(prob + 5),
            probability_label=prob_label,
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            aggressive_entry=round(aggressive, 2),
            stop=round(stop, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            risk_reward=round(rr, 1),
            ev=round(ev, 2),
            trigger=trigger,
            invalidation=invalidation,
            why=why
        )
    
    def _generate_short_setup(self, price, vah, poc, val, vwap, atr,
                               bull_score, bear_score, rsi, rvol,
                               position_pct, above_vwap, order_flow) -> Setup:
        """Generate SHORT setup"""
        
        # Calculate probability
        prob = self._calculate_short_probability(
            price, vah, poc, val, vwap,
            bull_score, bear_score, rsi, rvol,
            position_pct, above_vwap, order_flow
        )
        
        # Entry levels (conservative = rally to resistance)
        if price < val:
            # Extended below - entry on rally to VAL
            entry_low = val - atr * 0.2
            entry_high = val
            aggressive = price
        elif price < poc:
            # Below POC - entry on rally to POC
            entry_low = poc - atr * 0.3
            entry_high = poc
            aggressive = price
        else:
            # Near VAH - entry at current or VAH
            entry_low = max(price, poc)
            entry_high = vah
            aggressive = price
        
        # Stop loss (above key resistance)
        if price < poc:
            stop = poc + atr * 0.5
        else:
            stop = vah + atr * 0.5
        
        # Ensure stop is reasonable  
        stop = min(stop, price * 1.05)  # Max 5% stop
        
        # Targets
        target_1 = val if price > val else val - atr
        target_2 = val - atr * 1.5 if price > val else val - atr * 2
        
        # Ensure targets below entry
        if target_1 >= entry_low:
            target_1 = entry_low - atr * 0.5
        if target_2 >= target_1:
            target_2 = target_1 - atr
        
        # Risk/Reward
        risk = stop - entry_low if stop > entry_low else atr * 0.5
        reward = entry_low - target_1
        rr = reward / risk if risk > 0 else 0
        
        # Expected Value
        ev = (prob / 100 * reward) - ((100 - prob) / 100 * risk)
        
        # Grade
        grade, conviction = self._calculate_grade(prob, rr, bull_score, bear_score, 'SHORT')
        
        # Probability label
        if prob >= 65:
            prob_label = "High"
        elif prob >= 55:
            prob_label = "Medium"
        else:
            prob_label = "Low"
        
        # Generate trigger and invalidation
        trigger = self._generate_short_trigger(price, vah, poc, val, vwap, rvol)
        invalidation = self._generate_short_invalidation(price, vah, poc, val, stop)
        why = self._generate_short_why(price, vah, poc, val, vwap, rsi, bear_score)
        
        return Setup(
            direction='SHORT',
            grade=grade,
            conviction=conviction,
            probability_low=int(prob - 5),
            probability_high=int(prob + 5),
            probability_label=prob_label,
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            aggressive_entry=round(aggressive, 2),
            stop=round(stop, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            risk_reward=round(rr, 1),
            ev=round(ev, 2),
            trigger=trigger,
            invalidation=invalidation,
            why=why
        )
    
    def _calculate_long_probability(self, price, vah, poc, val, vwap,
                                     bull_score, bear_score, rsi, rvol,
                                     position_pct, above_vwap, order_flow) -> float:
        """Calculate LONG probability using rules"""
        
        prob = 50  # Base
        
        # Position-based probability
        if price <= val:
            # At VAL - high bounce probability
            prob = self.BASE_VAL_BOUNCE
        elif price <= poc:
            # Between VAL and POC
            prob = 55 + (position_pct / 100 * 10)
        elif price <= vah:
            # Between POC and VAH
            prob = 50 + (50 - position_pct) / 100 * 5
        else:
            # Extended above VAH - lower probability
            prob = 45
        
        # VWAP adjustment
        if above_vwap:
            prob += self.VWAP_TREND_BONUS
        
        # Score adjustment
        score_diff = bull_score - bear_score
        if score_diff > 30:
            prob += 10
        elif score_diff > 15:
            prob += 5
        elif score_diff < -15:
            prob -= 10
        
        # RSI adjustment
        if rsi < 30:
            prob += 10  # Oversold bounce
        elif rsi > 70:
            prob -= 5   # Overbought caution
        
        # Volume adjustment
        if rvol > 1.5:
            if score_diff > 0:
                prob += 5  # High volume confirms direction
            else:
                prob -= 5  # High volume against
        
        # Order flow adjustment
        if order_flow:
            buy_pressure = order_flow.get('buy_pressure', 50)
            if buy_pressure > 60:
                prob += 5
            elif buy_pressure < 40:
                prob -= 5
        
        return max(20, min(85, prob))
    
    def _calculate_short_probability(self, price, vah, poc, val, vwap,
                                      bull_score, bear_score, rsi, rvol,
                                      position_pct, above_vwap, order_flow) -> float:
        """Calculate SHORT probability using rules"""
        
        prob = 50  # Base
        
        # Position-based probability
        if price >= vah:
            # At VAH - high rejection probability
            prob = self.BASE_VAH_REJECTION
        elif price >= poc:
            # Between POC and VAH
            prob = 55 + ((100 - position_pct) / 100 * 10)
        elif price >= val:
            # Between VAL and POC
            prob = 50 + (position_pct / 100 * 5)
        else:
            # Extended below VAL - lower probability
            prob = 45
        
        # VWAP adjustment
        if not above_vwap:
            prob += self.VWAP_TREND_BONUS
        
        # Score adjustment
        score_diff = bear_score - bull_score
        if score_diff > 30:
            prob += 10
        elif score_diff > 15:
            prob += 5
        elif score_diff < -15:
            prob -= 10
        
        # RSI adjustment
        if rsi > 70:
            prob += 10  # Overbought reversal
        elif rsi < 30:
            prob -= 5   # Oversold caution
        
        # Volume adjustment
        if rvol > 1.5:
            if score_diff > 0:
                prob += 5
            else:
                prob -= 5
        
        # Order flow adjustment
        if order_flow:
            sell_pressure = order_flow.get('sell_pressure', 50)
            if sell_pressure > 60:
                prob += 5
            elif sell_pressure < 40:
                prob -= 5
        
        return max(20, min(85, prob))
    
    def _calculate_grade(self, prob, rr, bull_score, bear_score, direction) -> Tuple[str, int]:
        """Calculate grade and conviction"""
        
        # Score alignment check
        if direction == 'LONG':
            score_aligned = bull_score > bear_score
            score_diff = bull_score - bear_score
        else:
            score_aligned = bear_score > bull_score
            score_diff = bear_score - bull_score
        
        # Grade based on probability and R:R
        if prob >= 75 and rr >= 3.0 and score_aligned:
            grade = 'A+'
            base_conviction = 10
        elif prob >= 70 and rr >= 2.5 and score_aligned:
            grade = 'A'
            base_conviction = 9
        elif prob >= 60 and rr >= 2.0:
            grade = 'B'
            base_conviction = 7
        elif prob >= 50 and rr >= 1.5:
            grade = 'C'
            base_conviction = 5
        else:
            grade = 'F'
            base_conviction = 3
        
        # Adjust conviction for score alignment
        if not score_aligned:
            base_conviction -= 2
        
        # Adjust for extreme score difference
        if score_diff > 50:
            base_conviction += 1
        elif score_diff < -20:
            base_conviction -= 1
        
        conviction = max(1, min(10, base_conviction))
        
        return grade, conviction
    
    def _determine_verdict(self, long_setup, short_setup, 
                           price, vah, poc, val, vwap,
                           bull_score, bear_score, order_flow) -> Tuple[str, str]:
        """Determine which direction is preferred and why"""
        
        reasons = []
        long_score = 0
        short_score = 0
        
        # Grade comparison
        grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'F': 1}
        long_grade_score = grade_order.get(long_setup.grade, 0)
        short_grade_score = grade_order.get(short_setup.grade, 0)
        
        if long_grade_score > short_grade_score:
            long_score += 2
        elif short_grade_score > long_grade_score:
            short_score += 2
        
        # Probability comparison
        if long_setup.probability_high > short_setup.probability_high:
            long_score += 1
        elif short_setup.probability_high > long_setup.probability_high:
            short_score += 1
        
        # Bull/Bear score
        if bull_score > bear_score:
            long_score += 2
            reasons.append("bullish bias")
        elif bear_score > bull_score:
            short_score += 2
            reasons.append("bearish bias")
        
        # VWAP position
        if price > vwap:
            long_score += 1
            reasons.append("VWAP support")
        else:
            short_score += 1
            reasons.append("below VWAP")
        
        # Price position
        if price < poc:
            if price > val:
                long_score += 1
                reasons.append("near value support")
        else:
            if price < vah:
                short_score += 1
                reasons.append("near value resistance")
        
        # Order flow
        if order_flow:
            flow_bias = order_flow.get('flow_bias', 'NEUTRAL')
            if flow_bias == 'BULLISH':
                long_score += 1
                reasons.append("bullish order flow")
            elif flow_bias == 'BEARISH':
                short_score += 1
                reasons.append("bearish order flow")
        
        # Determine winner
        if long_score > short_score:
            preferred = 'LONG'
            reason = f"the {' and '.join(reasons[:2])} suggest a higher probability of a successful long trade"
        elif short_score > long_score:
            preferred = 'SHORT'
            reason = f"the {' and '.join(reasons[:2])} suggest a higher probability of a successful short trade"
        else:
            preferred = 'NEUTRAL'
            reason = "conflicting signals - wait for clearer direction"
        
        return preferred, reason
    
    def _calculate_key_level(self, price, vah, poc, val, vwap) -> float:
        """Calculate the key decision level"""
        
        # Key level is typically POC or VWAP - whichever is closer to price
        poc_dist = abs(price - poc)
        vwap_dist = abs(price - vwap)
        
        if poc_dist < vwap_dist:
            return poc
        else:
            return vwap
    
    def _generate_bookmap_checklist(self, direction, setup, vah, poc, val) -> str:
        """Generate Bookmap confirmation checklist"""
        
        if direction == 'LONG':
            level = setup.entry_low
            return f"absorption at ${level:.2f} (buyers absorbing sellers), delta flip positive, iceberg bids"
        else:
            level = setup.entry_high
            return f"absorption at ${level:.2f} (sellers absorbing buyers), delta flip negative, iceberg offers"
    
    def _generate_options_strategy(self, direction: str, price: float, atr: float,
                                    primary_setup: 'Setup', hedge_setup: 'Setup') -> Dict:
        """
        Generate hedged options play based on the preferred direction.
        
        Strategy: 
        - LONG bias: Buy 4-week call (primary) + Buy 2-week put (hedge)
        - SHORT bias: Buy 4-week put (primary) + Buy 2-week call (hedge)
        
        Leverage balanced so:
        - If thesis correct: primary gains > hedge losses
        - If thesis wrong: hedge gains offset primary losses
        - Can cut loser early or hold as indicator for repositioning
        """
        if direction == 'NEUTRAL':
            return {
                'strategy': 'WAIT',
                'reason': 'No clear direction - wait for better setup'
            }
        
        # Round strikes to nearest $5 for most stocks, $1 for lower priced
        def round_strike(p, increment=5):
            if p < 50:
                increment = 1
            elif p < 100:
                increment = 2.5
            return round(p / increment) * increment
        
        if direction == 'LONG':
            # LONG BIAS: Call (primary) + Put (hedge)
            
            # Primary: ATM or slightly ITM call, 4 weeks out
            # Strike at or just below current price for higher delta
            call_strike = round_strike(price * 0.98)  # Slightly ITM for better delta
            
            # Hedge: OTM put, 2 weeks out (cheaper, shorter duration)
            # Strike below entry zone (protect the downside)
            put_strike = round_strike(primary_setup.stop)  # At stop level
            
            # Calculate theoretical position sizes (2:1 leverage favoring calls)
            # If $1000 budget: $650-700 on calls, $300-350 on puts
            leverage_ratio = "2:1 favoring CALL"
            
            # Max loss scenarios
            max_loss_if_flat = "Premium paid on both legs (theta decay)"
            max_loss_if_crash = f"Call loses most value, but PUT profits offset - net loss capped"
            
            # Breakeven (simplified)
            breakeven_up = f"${call_strike + (atr * 0.3):.2f} (call strike + premium)"
            breakeven_down = f"${put_strike - (atr * 0.2):.2f} (put strike - premium)"
            
            return {
                'strategy': 'HEDGED LONG (Call + Put Hedge)',
                'bias': 'BULLISH',
                
                # Primary Leg - The directional bet
                'primary': {
                    'type': 'CALL',
                    'strike': call_strike,
                    'expiry': '4 weeks (28-35 DTE)',
                    'delta': '0.55-0.65 (ATM/slightly ITM)',
                    'allocation': '65-70% of position',
                    'entry_timing': 'On pullback to entry zone or breakout confirmation'
                },
                
                # Hedge Leg - The protection
                'hedge': {
                    'type': 'PUT',
                    'strike': put_strike,
                    'expiry': '2 weeks (14-21 DTE)',
                    'delta': '0.30-0.40 (OTM)',
                    'allocation': '30-35% of position',
                    'entry_timing': 'Same time as call OR on first sign of weakness'
                },
                
                'leverage_ratio': leverage_ratio,
                'max_risk': 'Total premium paid (defined risk)',
                
                # Profit scenarios
                'if_bullish_correct': f"Call gains 2-3x, put expires worthless. Target: ${primary_setup.target_1:.2f}",
                'if_bearish_wrong': f"Put gains offset call losses. Can roll call or close for small net loss",
                'if_chop': "Both decay - cut early if no movement in 3-5 days",
                
                # Management Rules
                'management': {
                    'take_profit_call': f"50-75% gain on call OR price hits T1 ${primary_setup.target_1:.2f}",
                    'take_profit_put': f"If price drops to ${put_strike:.2f}, close put for profit, reassess call",
                    'cut_loss': f"If call down 50% with no bounce at entry zone - close both",
                    'roll_strategy': "If bullish thesis intact but call losing, roll out 2 weeks",
                    'reposition_signal': f"Put profitable = bearish momentum. Close call, add to put OR flip short"
                },
                
                # The Edge
                'edge': "Put acts as insurance AND indicator. Profitable put = thesis wrong, time to flip."
            }
        
        else:  # SHORT
            # SHORT BIAS: Put (primary) + Call (hedge)
            
            # Primary: ATM or slightly ITM put, 4 weeks out
            put_strike = round_strike(price * 1.02)  # Slightly ITM for better delta
            
            # Hedge: OTM call, 2 weeks out
            call_strike = round_strike(hedge_setup.stop)  # At stop level (above resistance)
            
            leverage_ratio = "2:1 favoring PUT"
            
            return {
                'strategy': 'HEDGED SHORT (Put + Call Hedge)',
                'bias': 'BEARISH',
                
                # Primary Leg
                'primary': {
                    'type': 'PUT',
                    'strike': put_strike,
                    'expiry': '4 weeks (28-35 DTE)',
                    'delta': '0.55-0.65 (ATM/slightly ITM)',
                    'allocation': '65-70% of position',
                    'entry_timing': 'On rally to resistance or breakdown confirmation'
                },
                
                # Hedge Leg
                'hedge': {
                    'type': 'CALL',
                    'strike': call_strike,
                    'expiry': '2 weeks (14-21 DTE)',
                    'delta': '0.30-0.40 (OTM)',
                    'allocation': '30-35% of position',
                    'entry_timing': 'Same time as put OR on first sign of strength'
                },
                
                'leverage_ratio': leverage_ratio,
                'max_risk': 'Total premium paid (defined risk)',
                
                # Profit scenarios
                'if_bearish_correct': f"Put gains 2-3x, call expires worthless. Target: ${primary_setup.target_1:.2f}",
                'if_bullish_wrong': f"Call gains offset put losses. Can roll put or close for small net loss",
                'if_chop': "Both decay - cut early if no movement in 3-5 days",
                
                # Management Rules
                'management': {
                    'take_profit_put': f"50-75% gain on put OR price hits T1 ${primary_setup.target_1:.2f}",
                    'take_profit_call': f"If price rallies to ${call_strike:.2f}, close call for profit, reassess put",
                    'cut_loss': f"If put down 50% with no rejection at resistance - close both",
                    'roll_strategy': "If bearish thesis intact but put losing, roll out 2 weeks",
                    'reposition_signal': f"Call profitable = bullish momentum. Close put, add to call OR flip long"
                },
                
                'edge': "Call acts as insurance AND indicator. Profitable call = thesis wrong, time to flip."
            }
    
    def _generate_long_trigger(self, price, vah, poc, val, vwap, rvol) -> str:
        """Generate LONG trigger condition"""
        
        if price < val:
            return f"Bounce from VAL ${val:.2f} with volume > 1.5x average"
        elif price < poc:
            return f"Hold above VAL ${val:.2f} with bullish candle"
        elif price < vah:
            return f"Break above POC ${poc:.2f} with increasing volume"
        else:
            return f"Pullback to VAH ${vah:.2f} holds as support"
    
    def _generate_short_trigger(self, price, vah, poc, val, vwap, rvol) -> str:
        """Generate SHORT trigger condition"""
        
        if price > vah:
            return f"Rejection at VAH ${vah:.2f} with volume > 1.5x average"
        elif price > poc:
            return f"Fail below VAH ${vah:.2f} with bearish candle"
        elif price > val:
            return f"Break below POC ${poc:.2f} with increasing volume"
        else:
            return f"Rally to VAL ${val:.2f} rejected as resistance"
    
    def _generate_long_invalidation(self, price, vah, poc, val, stop) -> str:
        """Generate LONG invalidation condition"""
        
        if price > poc:
            return f"Break below ${stop:.2f}"
        else:
            return f"Close below ${val:.2f} (VAL)"
    
    def _generate_short_invalidation(self, price, vah, poc, val, stop) -> str:
        """Generate SHORT invalidation condition"""
        
        if price < poc:
            return f"Break above ${stop:.2f}"
        else:
            return f"Close above ${vah:.2f} (VAH)"
    
    def _generate_long_why(self, price, vah, poc, val, vwap, rsi, bull_score) -> str:
        """Generate LONG reasoning"""
        
        reasons = []
        
        if price > vwap:
            reasons.append("price holding above VWAP")
        if price > poc:
            reasons.append("above POC")
        elif price > val:
            reasons.append("near VAL support")
        if bull_score > 60:
            reasons.append("strong bullish score")
        if rsi < 40:
            reasons.append("oversold RSI")
        
        if reasons:
            return f"{', '.join(reasons[:2])} indicates continuation potential"
        return "Price structure favors upside continuation"
    
    def _generate_short_why(self, price, vah, poc, val, vwap, rsi, bear_score) -> str:
        """Generate SHORT reasoning"""
        
        reasons = []
        
        if price < vwap:
            reasons.append("price below VWAP")
        if price < poc:
            reasons.append("below POC")
        elif price < vah:
            reasons.append("near VAH resistance")
        if bear_score > 60:
            reasons.append("strong bearish score")
        if rsi > 60:
            reasons.append("overbought RSI")
        
        if reasons:
            return f"{', '.join(reasons[:2])} indicates reversal potential"
        return "Price structure favors downside reversal"
    
    def format_as_ai_text(self, result: DualSetupResult) -> str:
        """
        Format the result as AI commentary text (same format as GPT output).
        This can be parsed by the existing parseAIResponse() function.
        """
        
        long = result.long_setup
        short = result.short_setup
        opts = result.options_strategy
        
        text = f"""🟢 LONG SETUP
⭐ GRADE: {long.grade} | 🎯 CONVICTION: {long.conviction}/10
📈 PROBABILITY: {long.probability_low}-{long.probability_high}% [{long.probability_label}]
📍 ENTRY: ${long.entry_low:.2f} - ${long.entry_high:.2f} (conservative, wait for pullback)
⚡ AGGRESSIVE: ${long.aggressive_entry:.2f} NOW
🛑 STOP: ${long.stop:.2f}
💰 T1: ${long.target_1:.2f} | 🚀 T2: ${long.target_2:.2f}
📐 R:R: {long.risk_reward:.1f}:1 | 💹 EV: ${long.ev:.2f}
✅ TRIGGER: {long.trigger}
❌ INVALID: {long.invalidation}
💡 WHY: {long.why}

🔴 SHORT SETUP
⭐ GRADE: {short.grade} | 🎯 CONVICTION: {short.conviction}/10
📈 PROBABILITY: {short.probability_low}-{short.probability_high}% [{short.probability_label}]
📍 ENTRY: ${short.entry_low:.2f} - ${short.entry_high:.2f} (conservative, wait for rally)
⚡ AGGRESSIVE: ${short.aggressive_entry:.2f} NOW
🛑 STOP: ${short.stop:.2f}
💰 T1: ${short.target_1:.2f} | 🚀 T2: ${short.target_2:.2f}
📐 R:R: {short.risk_reward:.1f}:1 | 💹 EV: ${short.ev:.2f}
✅ TRIGGER: {short.trigger}
❌ INVALID: {short.invalidation}
💡 WHY: {short.why}

⚖️ VERDICT: {result.preferred_direction} preferred because {result.verdict_reason}
⚠️ KEY LEVEL: {result.key_level_desc}

📊 BOOKMAP ORDER FLOW CHECKLIST (confirm before entry):
🔍 LONG: Look for {result.bookmap_long}
🔍 SHORT: Look for {result.bookmap_short}"""

        # Add options strategy section
        if opts and opts.get('strategy') != 'WAIT':
            primary = opts.get('primary', {})
            hedge = opts.get('hedge', {})
            mgmt = opts.get('management', {})
            
            text += f"""

🎰 OPTIONS STRATEGY: {opts.get('strategy', 'N/A')}
📊 Bias: {opts.get('bias', 'N/A')} | Leverage: {opts.get('leverage_ratio', '2:1')}

💚 PRIMARY LEG ({primary.get('type', 'N/A')}):
   Strike: ${primary.get('strike', 0):.2f} | Expiry: {primary.get('expiry', '4 weeks')}
   Delta: {primary.get('delta', '0.55-0.65')} | Size: {primary.get('allocation', '65-70%')}
   ⏰ Entry: {primary.get('entry_timing', 'On confirmation')}

🛡️ HEDGE LEG ({hedge.get('type', 'N/A')}):
   Strike: ${hedge.get('strike', 0):.2f} | Expiry: {hedge.get('expiry', '2 weeks')}   
   Delta: {hedge.get('delta', '0.30-0.40')} | Size: {hedge.get('allocation', '30-35%')}
   ⏰ Entry: {hedge.get('entry_timing', 'Same time or on weakness')}

📈 SCENARIOS:
   ✅ If {opts.get('bias', 'thesis').lower()} correct: {opts.get('if_bullish_correct', opts.get('if_bearish_correct', 'Primary gains, hedge expires'))}
   ❌ If wrong: {opts.get('if_bullish_wrong', opts.get('if_bearish_wrong', 'Hedge gains offset losses'))}
   ↔️ If chop: {opts.get('if_chop', 'Both decay - cut early')}

🎯 MANAGEMENT:
   💰 Take Profit: {mgmt.get('take_profit_call', mgmt.get('take_profit_put', '50-75% gain'))}
   ✂️ Cut Loss: {mgmt.get('cut_loss', 'If primary down 50%')}
   🔄 Roll: {mgmt.get('roll_strategy', 'Roll out 2 weeks if thesis intact')}
   ⚠️ FLIP SIGNAL: {mgmt.get('reposition_signal', 'Hedge profitable = thesis wrong')}

💡 EDGE: {opts.get('edge', 'Hedge acts as insurance AND indicator')}"""
        
        return text


# Convenience function
def generate_dual_setup(data: Dict) -> str:
    """
    Generate dual setup text from scanner data.
    Returns formatted text that can be parsed by parseAIResponse().
    """
    generator = DualSetupGenerator()
    result = generator.generate(data)
    return generator.format_as_ai_text(result)


if __name__ == "__main__":
    # Test
    test_data = {
        'symbol': 'TSLA',
        'current_price': 411.78,
        'vah': 414.00,
        'poc': 408.00,
        'val': 402.00,
        'vwap': 409.50,
        'bull_score': 67,
        'bear_score': 35,
        'rsi': 55,
        'rvol': 1.2,
        'atr': 8.0,
        'order_flow': {
            'buy_pressure': 44.7,
            'sell_pressure': 41.3,
            'flow_bias': 'NEUTRAL'
        }
    }
    
    result = generate_dual_setup(test_data)
    print(result)
