"""
Integrated Trading Scanner
==========================
Combines MTF Auction Scanner with Overnight/Gap Prediction
for complete pre-market and intraday analysis.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from mtf_auction_scanner import (
    MTFAuctionScanner, ScanResult, SignalState, Timeframe,
    VolumeProfileEngine
)
from overnight_model import (
    OvernightPredictionEngine, OvernightPrediction, OvernightBias,
    GapType, GapFillProbability
)


@dataclass
class IntegratedAnalysis:
    """Complete analysis combining intraday and overnight context"""
    symbol: str
    analysis_time: datetime
    
    # Components
    mtf_scan: ScanResult
    overnight: Optional[OvernightPrediction]
    
    # Combined signal
    combined_bias: str           # "STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"
    combined_confidence: float
    
    # Scenarios with probabilities
    high_scenario_prob: float
    low_scenario_prob: float
    chop_scenario_prob: float
    
    # Key levels for the day
    key_levels: Dict[str, float]
    
    # Trade plan
    trade_plan: Optional[Dict[str, Any]]
    
    # Notes
    notes: List[str] = field(default_factory=list)


class IntegratedScanner:
    """
    Combines MTF auction analysis with overnight/gap prediction
    
    For Brokers:
    -----------
    This gives you the COMPLETE picture:
    1. What happened overnight? (Gap, direction, key levels)
    2. Where are we in the auction? (Value area, flow, momentum)
    3. What's the combined bias? (Synthesis of both)
    4. What's the trade plan? (Entries, stops, targets)
    
    For Programmers:
    ---------------
    Orchestrates both scanners and combines their outputs
    with weighted scoring for final signal.
    """
    
    def __init__(self):
        self.mtf_scanner = MTFAuctionScanner()
        self.overnight_engine = OvernightPredictionEngine()
        self.vp_engine = VolumeProfileEngine()
    
    def analyze(self, 
                df: pd.DataFrame, 
                symbol: str = "UNKNOWN") -> IntegratedAnalysis:
        """
        Run complete integrated analysis
        
        Args:
            df: OHLCV dataframe with datetime index
            symbol: Ticker symbol
        
        Returns:
            IntegratedAnalysis object
        """
        # Run MTF scan
        mtf_result = self.mtf_scanner.scan(df, symbol=symbol)
        
        # Get prior day value area for overnight analysis
        daily = df.resample('D').agg({
            'open': 'first', 'high': 'max', 
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        if len(daily) >= 2:
            prior_day_df = df[df.index.date == daily.index[-2].date()]
            if len(prior_day_df) > 10:
                vp = self.vp_engine.calculate(prior_day_df)
                prior_poc = vp.poc
                prior_vah = vp.vah
                prior_val = vp.val
            else:
                prior_poc = prior_vah = prior_val = None
        else:
            prior_poc = prior_vah = prior_val = None
        
        # Run overnight prediction
        overnight = self.overnight_engine.predict(
            df, symbol=symbol,
            prior_poc=prior_poc, prior_vah=prior_vah, prior_val=prior_val
        )
        
        # Combine signals
        combined_bias, combined_conf = self._combine_signals(mtf_result, overnight)
        
        # Calculate scenario probabilities
        high_prob, low_prob, chop_prob = self._calculate_scenarios(
            mtf_result, overnight, combined_bias
        )
        
        # Compile key levels
        key_levels = self._compile_key_levels(mtf_result, overnight)
        
        # Generate trade plan if actionable
        trade_plan = self._generate_trade_plan(
            mtf_result, overnight, combined_bias, combined_conf, key_levels
        )
        
        # Compile notes
        notes = self._compile_notes(mtf_result, overnight, combined_bias)
        
        return IntegratedAnalysis(
            symbol=symbol,
            analysis_time=datetime.now(),
            mtf_scan=mtf_result,
            overnight=overnight,
            combined_bias=combined_bias,
            combined_confidence=combined_conf,
            high_scenario_prob=high_prob,
            low_scenario_prob=low_prob,
            chop_scenario_prob=chop_prob,
            key_levels=key_levels,
            trade_plan=trade_plan,
            notes=notes
        )
    
    def _combine_signals(self, 
                         mtf: ScanResult, 
                         overnight: Optional[OvernightPrediction]) -> tuple:
        """Combine MTF and overnight signals into unified bias"""
        
        # MTF contribution (60% weight)
        mtf_bull_score = 0
        if mtf.dominant_signal == SignalState.LONG_SETUP:
            mtf_bull_score = 80 + (mtf.confluence_score / 5)
        elif mtf.dominant_signal == SignalState.YELLOW:
            mtf_bull_score = 50 + (mtf.high_scenario_prob - 0.5) * 30
        elif mtf.dominant_signal == SignalState.SHORT_SETUP:
            mtf_bull_score = 20 - (mtf.confluence_score / 5)
        else:
            mtf_bull_score = 50
        
        # Overnight contribution (40% weight)
        overnight_bull_score = 50
        if overnight:
            if overnight.bias == OvernightBias.STRONG_BULLISH:
                overnight_bull_score = 90
            elif overnight.bias == OvernightBias.BULLISH:
                overnight_bull_score = 70
            elif overnight.bias == OvernightBias.BEARISH:
                overnight_bull_score = 30
            elif overnight.bias == OvernightBias.STRONG_BEARISH:
                overnight_bull_score = 10
            else:
                overnight_bull_score = 50
            
            # Gap type adjustment
            if overnight.gap.gap_type in [GapType.GAP_UP_LARGE, GapType.GAP_UP_SMALL]:
                if overnight.gap.gap_fill_probability == GapFillProbability.LOW:
                    overnight_bull_score += 10
            elif overnight.gap.gap_type in [GapType.GAP_DOWN_LARGE, GapType.GAP_DOWN_SMALL]:
                if overnight.gap.gap_fill_probability == GapFillProbability.LOW:
                    overnight_bull_score -= 10
        
        # Weighted combination
        combined_score = (mtf_bull_score * 0.6) + (overnight_bull_score * 0.4)
        
        # Determine bias
        if combined_score >= 75:
            bias = "STRONG_BULL"
        elif combined_score >= 60:
            bias = "BULL"
        elif combined_score <= 25:
            bias = "STRONG_BEAR"
        elif combined_score <= 40:
            bias = "BEAR"
        else:
            bias = "NEUTRAL"
        
        # Confidence based on agreement
        agreement_bonus = 0
        if overnight:
            mtf_bullish = mtf.dominant_signal == SignalState.LONG_SETUP
            overnight_bullish = overnight.bias in [OvernightBias.STRONG_BULLISH, OvernightBias.BULLISH]
            mtf_bearish = mtf.dominant_signal == SignalState.SHORT_SETUP
            overnight_bearish = overnight.bias in [OvernightBias.STRONG_BEARISH, OvernightBias.BEARISH]
            
            if (mtf_bullish and overnight_bullish) or (mtf_bearish and overnight_bearish):
                agreement_bonus = 15  # Signals agree
            elif (mtf_bullish and overnight_bearish) or (mtf_bearish and overnight_bullish):
                agreement_bonus = -15  # Signals conflict
        
        confidence = min(95, max(20, abs(combined_score - 50) * 2 + agreement_bonus))
        
        return bias, confidence
    
    def _calculate_scenarios(self,
                             mtf: ScanResult,
                             overnight: Optional[OvernightPrediction],
                             bias: str) -> tuple:
        """Calculate scenario probabilities"""
        
        # Start with MTF probabilities
        high_prob = mtf.high_scenario_prob
        low_prob = mtf.low_scenario_prob
        
        # Adjust based on overnight
        if overnight:
            if overnight.bias in [OvernightBias.STRONG_BULLISH, OvernightBias.BULLISH]:
                high_prob = high_prob * 1.2
                low_prob = low_prob * 0.8
            elif overnight.bias in [OvernightBias.STRONG_BEARISH, OvernightBias.BEARISH]:
                high_prob = high_prob * 0.8
                low_prob = low_prob * 1.2
            
            # Gap adjustment
            if overnight.gap.gap_fill_probability == GapFillProbability.HIGH:
                # Gap likely to fill - adds chop probability
                chop_prob = 0.25
            else:
                chop_prob = 0.10
        else:
            chop_prob = mtf.neutral_prob
        
        # Normalize
        total = high_prob + low_prob + chop_prob
        if total > 0:
            high_prob /= total
            low_prob /= total
            chop_prob /= total
        
        return high_prob, low_prob, chop_prob
    
    def _compile_key_levels(self,
                            mtf: ScanResult,
                            overnight: Optional[OvernightPrediction]) -> Dict[str, float]:
        """Compile all key levels"""
        levels = {}
        
        # From MTF analysis (use 4hr or highest available)
        for tf in [Timeframe.H4, Timeframe.H2, Timeframe.H1, Timeframe.M30]:
            if tf in mtf.timeframe_analyses:
                analysis = mtf.timeframe_analyses[tf]
                levels['session_vah'] = analysis.volume_profile.vah
                levels['session_poc'] = analysis.volume_profile.poc
                levels['session_val'] = analysis.volume_profile.val
                if analysis.vwap:
                    levels['vwap'] = analysis.vwap.vwap
                    levels['vwap_upper_1sd'] = analysis.vwap.upper_band_1
                    levels['vwap_lower_1sd'] = analysis.vwap.lower_band_1
                break
        
        # From overnight analysis
        if overnight:
            levels.update({
                'overnight_high': overnight.overnight.overnight_high,
                'overnight_low': overnight.overnight.overnight_low,
                'overnight_mid': overnight.overnight.overnight_midpoint,
                'prior_close': overnight.prior_day.prior_close,
                'prior_high': overnight.prior_day.prior_high,
                'prior_low': overnight.prior_day.prior_low,
                'prior_poc': overnight.prior_day.prior_poc,
                'prior_vah': overnight.prior_day.prior_vah,
                'prior_val': overnight.prior_day.prior_val,
                'gap_fill': overnight.gap.gap_fill_level
            })
        
        return levels
    
    def _generate_trade_plan(self,
                             mtf: ScanResult,
                             overnight: Optional[OvernightPrediction],
                             bias: str,
                             confidence: float,
                             levels: Dict[str, float]) -> Optional[Dict]:
        """Generate trade plan if conditions warrant"""
        
        if confidence < 50:
            return None  # Not confident enough
        
        if bias == "NEUTRAL":
            return None  # No clear direction
        
        current_price = mtf.timeframe_analyses[list(mtf.timeframe_analyses.keys())[0]].current_price
        
        if bias in ["STRONG_BULL", "BULL"]:
            # Long setup
            entry_zone_low = levels.get('session_val', current_price * 0.99)
            entry_zone_high = current_price
            stop = levels.get('overnight_low', levels.get('prior_low', current_price * 0.97))
            target1 = levels.get('session_vah', current_price * 1.02)
            target2 = levels.get('prior_high', current_price * 1.04)
            direction = "LONG"
            
        else:  # BEAR or STRONG_BEAR
            entry_zone_low = current_price
            entry_zone_high = levels.get('session_vah', current_price * 1.01)
            stop = levels.get('overnight_high', levels.get('prior_high', current_price * 1.03))
            target1 = levels.get('session_val', current_price * 0.98)
            target2 = levels.get('prior_low', current_price * 0.96)
            direction = "SHORT"
        
        risk = abs(current_price - stop)
        reward = abs(target1 - current_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'direction': direction,
            'bias_strength': bias,
            'confidence': confidence,
            'entry_zone': f"${entry_zone_low:.2f} - ${entry_zone_high:.2f}",
            'stop_loss': stop,
            'target_1': target1,
            'target_2': target2,
            'risk_per_share': risk,
            'reward_risk_ratio': rr_ratio,
            'position_notes': [
                f"Entry on pullback to {entry_zone_low:.2f}" if direction == "LONG" else f"Entry on rally to {entry_zone_high:.2f}",
                f"Stop {'below' if direction == 'LONG' else 'above'} overnight {'low' if direction == 'LONG' else 'high'}",
                f"Scale out 50% at Target 1"
            ]
        }
    
    def _compile_notes(self,
                       mtf: ScanResult,
                       overnight: Optional[OvernightPrediction],
                       bias: str) -> List[str]:
        """Compile analysis notes"""
        notes = []
        
        # MTF signal
        notes.append(f"MTF Signal: {mtf.dominant_signal.emoji} {mtf.dominant_signal.value} ({mtf.confluence_score:.0f}% confluence)")
        
        # Overnight
        if overnight:
            notes.append(f"Overnight: {overnight.bias.emoji} {overnight.bias.value}")
            notes.append(f"Gap: {overnight.gap.gap_type.emoji} {overnight.gap.gap_pct:+.2f}%")
            
            if overnight.gap.gap_fill_probability == GapFillProbability.HIGH:
                notes.append("⚠️ Gap likely to fill - watch for fade")
            elif overnight.gap.gap_fill_probability == GapFillProbability.LOW:
                notes.append("✅ Gap likely to hold - continuation expected")
        
        # Agreement/conflict
        if overnight:
            mtf_bull = mtf.dominant_signal == SignalState.LONG_SETUP
            mtf_bear = mtf.dominant_signal == SignalState.SHORT_SETUP
            on_bull = overnight.bias in [OvernightBias.STRONG_BULLISH, OvernightBias.BULLISH]
            on_bear = overnight.bias in [OvernightBias.STRONG_BEARISH, OvernightBias.BEARISH]
            
            if (mtf_bull and on_bull) or (mtf_bear and on_bear):
                notes.append("✅ MTF and Overnight AGREE - higher confidence")
            elif (mtf_bull and on_bear) or (mtf_bear and on_bull):
                notes.append("⚠️ MTF and Overnight CONFLICT - reduced confidence")
        
        # Combined bias
        notes.append(f"Combined Bias: {bias}")
        
        return notes
    
    def print_analysis(self, analysis: IntegratedAnalysis) -> str:
        """Format analysis for display"""
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"INTEGRATED ANALYSIS: {analysis.symbol}")
        lines.append(f"Time: {analysis.analysis_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 80)
        
        # Combined signal
        bias_emoji = {
            "STRONG_BULL": "🟢🟢", "BULL": "🟢",
            "NEUTRAL": "🟡",
            "BEAR": "🔴", "STRONG_BEAR": "🔴🔴"
        }.get(analysis.combined_bias, "⚪")
        
        lines.append(f"\n{bias_emoji} COMBINED BIAS: {analysis.combined_bias}")
        lines.append(f"   Confidence: {analysis.combined_confidence:.0f}%")
        
        # Scenario probabilities
        lines.append(f"\n📊 SCENARIO PROBABILITIES:")
        lines.append(f"   HIGH (Bullish): {analysis.high_scenario_prob:.0%}")
        lines.append(f"   LOW (Bearish):  {analysis.low_scenario_prob:.0%}")
        lines.append(f"   CHOP (Range):   {analysis.chop_scenario_prob:.0%}")
        
        # Component summaries
        lines.append(f"\n📈 MTF ANALYSIS:")
        lines.append(f"   Signal: {analysis.mtf_scan.dominant_signal.emoji} {analysis.mtf_scan.dominant_signal.value}")
        lines.append(f"   Confluence: {analysis.mtf_scan.confluence_score:.0f}%")
        
        if analysis.overnight:
            lines.append(f"\n🌙 OVERNIGHT ANALYSIS:")
            lines.append(f"   Bias: {analysis.overnight.bias.emoji} {analysis.overnight.bias.value}")
            lines.append(f"   Gap: {analysis.overnight.gap.gap_type.emoji} {analysis.overnight.gap.gap_pct:+.2f}%")
            lines.append(f"   Fill Prob: {analysis.overnight.gap.gap_fill_probability.value}")
        
        # Key levels
        lines.append(f"\n📍 KEY LEVELS:")
        sorted_levels = sorted(analysis.key_levels.items(), key=lambda x: x[1], reverse=True)
        for name, level in sorted_levels[:12]:  # Top 12 levels
            lines.append(f"   {name:<18}: ${level:.2f}")
        
        # Trade plan
        if analysis.trade_plan:
            tp = analysis.trade_plan
            lines.append(f"\n{'='*80}")
            lines.append(f"📋 TRADE PLAN: {tp['direction']}")
            lines.append(f"{'='*80}")
            lines.append(f"   Bias: {tp['bias_strength']} ({tp['confidence']:.0f}% conf)")
            lines.append(f"   Entry Zone: {tp['entry_zone']}")
            lines.append(f"   Stop Loss:  ${tp['stop_loss']:.2f}")
            lines.append(f"   Target 1:   ${tp['target_1']:.2f}")
            lines.append(f"   Target 2:   ${tp['target_2']:.2f}")
            lines.append(f"   Risk/Share: ${tp['risk_per_share']:.2f}")
            lines.append(f"   R:R Ratio:  {tp['reward_risk_ratio']:.2f}")
            lines.append(f"\n   Notes:")
            for note in tp['position_notes']:
                lines.append(f"     • {note}")
        else:
            lines.append(f"\n⏸️ NO TRADE PLAN - Wait for higher confidence setup")
        
        # Analysis notes
        lines.append(f"\n📝 NOTES:")
        for note in analysis.notes:
            lines.append(f"   • {note}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from overnight_model import generate_overnight_demo_data
    
    print("Generating demo data...")
    df = generate_overnight_demo_data()
    
    print(f"Data shape: {df.shape}")
    
    # Run integrated analysis
    scanner = IntegratedScanner()
    analysis = scanner.analyze(df, symbol="DEMO")
    
    # Print report
    report = scanner.print_analysis(analysis)
    print(report)
