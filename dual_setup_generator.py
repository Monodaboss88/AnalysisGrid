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
            bookmap_short=bookmap_short
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
