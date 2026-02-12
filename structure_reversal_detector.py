"""
Structure Reversal Detector - Macro Structure-Based Reversal Alerts
===================================================================
Detects high-probability reversal setups using macro structure analysis:

1. STRUCTURE_BREAK - HH/LL breaks counter to established trend
2. MOMENTUM_EXHAUSTION - Failing to make new highs/lows
3. RANGE_EXTREME_REVERSAL - At 90/10% of range with structure weakening
4. COMPRESSION_BREAKOUT - Tight range at key resistance/support
5. STRUCTURE_DIVERGENCE - Multi-timeframe structure conflicts

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class ReversalType(Enum):
    """Types of structure-based reversals"""
    STRUCTURE_BREAK_LONG = "STRUCTURE_BREAK_LONG"      # LL in uptrend → reversal down
    STRUCTURE_BREAK_SHORT = "STRUCTURE_BREAK_SHORT"    # HH in downtrend → reversal up
    MOMENTUM_EXHAUSTION_LONG = "MOMENTUM_EXHAUSTION_LONG"    # No HH → losing steam
    MOMENTUM_EXHAUSTION_SHORT = "MOMENTUM_EXHAUSTION_SHORT"  # No LL → losing steam
    RANGE_EXTREME_LONG = "RANGE_EXTREME_LONG"          # At bottom + HL forming
    RANGE_EXTREME_SHORT = "RANGE_EXTREME_SHORT"        # At top + LH forming
    COMPRESSION_BREAKOUT_LONG = "COMPRESSION_BREAKOUT_LONG"    # Tight range at support
    COMPRESSION_BREAKOUT_SHORT = "COMPRESSION_BREAKOUT_SHORT"  # Tight range at resistance
    STRUCTURE_DIVERGENCE_LONG = "STRUCTURE_DIVERGENCE_LONG"    # Weekly bull, daily bear
    STRUCTURE_DIVERGENCE_SHORT = "STRUCTURE_DIVERGENCE_SHORT"  # Weekly bear, daily bull


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "CRITICAL"  # 80-100 confidence
    HIGH = "HIGH"          # 60-79 confidence
    MEDIUM = "MEDIUM"      # 40-59 confidence
    LOW = "LOW"            # 20-39 confidence


@dataclass
class ReversalAlert:
    """A structure-based reversal alert"""
    alert_type: ReversalType
    severity: AlertSeverity
    confidence: float  # 0-100
    
    # Context
    symbol: str
    current_price: float
    trigger_level: Optional[float] = None  # Price level that triggered alert
    target_level: Optional[float] = None   # Potential target
    stop_level: Optional[float] = None     # Suggested stop
    
    # Factors
    structure_score: float = 0.0  # 0-30 points
    volume_score: float = 0.0     # 0-20 points
    vp_confluence: float = 0.0    # 0-15 points
    momentum_score: float = 0.0   # 0-15 points
    range_position: float = 0.0   # 0-10 points
    divergence_score: float = 0.0 # 0-10 points
    
    # Details
    description: str = ""
    timeframe: str = ""  # "daily", "weekly", "multi-period"
    signals: List[str] = field(default_factory=list)
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class StructureContext:
    """Condensed structure info for reversal detection"""
    # Weekly structure
    weekly_trend: str
    weekly_hh: int
    weekly_hl: int
    weekly_lh: int
    weekly_ll: int
    weekly_close_position: float  # 0-1
    
    # Multi-period structure
    period_3d_hh: bool
    period_3d_hl: bool
    period_3d_lh: bool
    period_3d_ll: bool
    
    period_6d_hh: bool
    period_6d_hl: bool
    period_6d_lh: bool
    period_6d_ll: bool
    
    period_30d_hh: bool
    period_30d_hl: bool
    period_30d_lh: bool
    period_30d_ll: bool
    
    # Range metrics
    current_price: float
    position_in_3d_range: float   # 0-1
    position_in_30d_range: float  # 0-1
    compression_ratio: float      # Current range vs avg
    
    # Support/Resistance
    nearest_resistance: float
    nearest_support: float
    

# =============================================================================
# STRUCTURE REVERSAL DETECTOR
# =============================================================================

class StructureReversalDetector:
    """
    Detects reversal patterns using macro structure analysis
    """
    
    def __init__(self,
                 min_confidence: float = 40.0,
                 structure_weight: float = 30.0,
                 volume_weight: float = 20.0,
                 vp_weight: float = 15.0,
                 momentum_weight: float = 15.0,
                 range_weight: float = 10.0,
                 divergence_weight: float = 10.0):
        """
        Initialize detector
        
        Args:
            min_confidence: Minimum confidence to generate alert (default 40)
            *_weight: Maximum points for each scoring factor
        """
        self.min_confidence = min_confidence
        self.structure_weight = structure_weight
        self.volume_weight = volume_weight
        self.vp_weight = vp_weight
        self.momentum_weight = momentum_weight
        self.range_weight = range_weight
        self.divergence_weight = divergence_weight
    
    def analyze(self, 
                df: pd.DataFrame, 
                structure_context: StructureContext,
                symbol: str = "UNKNOWN",
                vp_data: Optional[Dict] = None) -> List[ReversalAlert]:
        """
        Analyze for structure-based reversal patterns
        
        Args:
            df: OHLCV DataFrame
            structure_context: Condensed structure metrics
            symbol: Stock symbol
            vp_data: Optional Volume Profile data for confluence
        
        Returns:
            List of ReversalAlert objects sorted by confidence
        """
        if len(df) < 30:
            return []
        
        alerts = []
        
        # Detect each reversal type
        alerts.extend(self._detect_structure_breaks(df, structure_context, symbol, vp_data))
        alerts.extend(self._detect_momentum_exhaustion(df, structure_context, symbol, vp_data))
        alerts.extend(self._detect_range_extremes(df, structure_context, symbol, vp_data))
        alerts.extend(self._detect_compression_breakouts(df, structure_context, symbol, vp_data))
        alerts.extend(self._detect_divergences(df, structure_context, symbol, vp_data))
        
        # Filter by confidence and sort
        alerts = [a for a in alerts if a.confidence >= self.min_confidence]
        alerts.sort(key=lambda x: x.confidence, reverse=True)
        
        return alerts
    
    # =========================================================================
    # PATTERN DETECTORS
    # =========================================================================
    
    def _detect_structure_breaks(self, df, ctx: StructureContext, symbol: str, vp_data) -> List[ReversalAlert]:
        """
        Detect when price makes HH/LL counter to established trend
        
        LONG Alert: Lower Low (LL) in established uptrend (HH+HL pattern)
        SHORT Alert: Higher High (HH) in established downtrend (LH+LL pattern)
        """
        alerts = []
        
        # LONG: LL in uptrend
        if self._is_uptrend(ctx) and (ctx.period_3d_ll or ctx.period_6d_ll):
            score, signals = self._score_structure_break_long(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.STRUCTURE_BREAK_LONG,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    trigger_level=ctx.nearest_support,
                    target_level=ctx.nearest_support * 0.97,  # 3% below support
                    stop_level=ctx.current_price * 1.01,
                    description=f"Lower Low detected in uptrend - potential reversal to downside. {ctx.weekly_trend} macro trend showing weakness.",
                    timeframe="daily+weekly",
                    signals=signals
                )
                alert.structure_score = min(self.structure_weight, score * 0.3)
                alerts.append(alert)
        
        # SHORT: HH in downtrend  
        if self._is_downtrend(ctx) and (ctx.period_3d_hh or ctx.period_6d_hh):
            score, signals = self._score_structure_break_short(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.STRUCTURE_BREAK_SHORT,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    trigger_level=ctx.nearest_resistance,
                    target_level=ctx.nearest_resistance * 1.03,  # 3% above resistance
                    stop_level=ctx.current_price * 0.99,
                    description=f"Higher High detected in downtrend - potential reversal to upside. {ctx.weekly_trend} macro trend showing strength.",
                    timeframe="daily+weekly",
                    signals=signals
                )
                alert.structure_score = min(self.structure_weight, score * 0.3)
                alerts.append(alert)
        
        return alerts
    
    def _detect_momentum_exhaustion(self, df, ctx: StructureContext, symbol: str, vp_data) -> List[ReversalAlert]:
        """
        Detect when price fails to make new highs/lows - momentum fading
        
        LONG: Uptrend but no HH in recent periods
        SHORT: Downtrend but no LL in recent periods
        """
        alerts = []
        
        # LONG: No HH recently in uptrend
        if self._is_uptrend(ctx):
            no_recent_hh = not ctx.period_3d_hh and not ctx.period_6d_hh
            has_older_hh = ctx.period_30d_hh
            
            if no_recent_hh and has_older_hh:
                score, signals = self._score_exhaustion_long(df, ctx, vp_data)
                
                if score >= self.min_confidence:
                    alert = ReversalAlert(
                        alert_type=ReversalType.MOMENTUM_EXHAUSTION_LONG,
                        severity=self._get_severity(score),
                        confidence=score,
                        symbol=symbol,
                        current_price=ctx.current_price,
                        description=f"Uptrend momentum exhaustion - no new highs in 6+ days. Buyers weakening.",
                        timeframe="daily",
                        signals=signals
                    )
                    alert.momentum_score = min(self.momentum_weight, score * 0.15)
                    alerts.append(alert)
        
        # SHORT: No LL recently in downtrend
        if self._is_downtrend(ctx):
            no_recent_ll = not ctx.period_3d_ll and not ctx.period_6d_ll
            has_older_ll = ctx.period_30d_ll
            
            if no_recent_ll and has_older_ll:
                score, signals = self._score_exhaustion_short(df, ctx, vp_data)
                
                if score >= self.min_confidence:
                    alert = ReversalAlert(
                        alert_type=ReversalType.MOMENTUM_EXHAUSTION_SHORT,
                        severity=self._get_severity(score),
                        confidence=score,
                        symbol=symbol,
                        current_price=ctx.current_price,
                        description=f"Downtrend momentum exhaustion - no new lows in 6+ days. Sellers weakening.",
                        timeframe="daily",
                        signals=signals
                    )
                    alert.momentum_score = min(self.momentum_weight, score * 0.15)
                    alerts.append(alert)
        
        return alerts
    
    def _detect_range_extremes(self, df, ctx: StructureContext, symbol: str, vp_data) -> List[ReversalAlert]:
        """
        Detect when price at extreme of range with structure forming
        
        LONG: At bottom 10% + HL forming
        SHORT: At top 90% + LH forming
        """
        alerts = []
        
        # LONG: Bottom extreme + HL
        at_bottom = ctx.position_in_30d_range < 0.15
        hl_forming = ctx.period_3d_hl or ctx.period_6d_hl
        
        if at_bottom and hl_forming:
            score, signals = self._score_range_extreme_long(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.RANGE_EXTREME_LONG,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    trigger_level=ctx.nearest_support,
                    target_level=ctx.current_price * 1.05,
                    stop_level=ctx.nearest_support * 0.98,
                    description=f"At bottom of range with Higher Low forming - potential bounce. Position: {ctx.position_in_30d_range*100:.1f}%",
                    timeframe="30-day range",
                    signals=signals
                )
                alert.range_position = min(self.range_weight, score * 0.1)
                alerts.append(alert)
        
        # SHORT: Top extreme + LH
        at_top = ctx.position_in_30d_range > 0.85
        lh_forming = ctx.period_3d_lh or ctx.period_6d_lh
        
        if at_top and lh_forming:
            score, signals = self._score_range_extreme_short(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.RANGE_EXTREME_SHORT,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    trigger_level=ctx.nearest_resistance,
                    target_level=ctx.current_price * 0.95,
                    stop_level=ctx.nearest_resistance * 1.02,
                    description=f"At top of range with Lower High forming - potential fade. Position: {ctx.position_in_30d_range*100:.1f}%",
                    timeframe="30-day range",
                    signals=signals
                )
                alert.range_position = min(self.range_weight, score * 0.1)
                alerts.append(alert)
        
        return alerts
    
    def _detect_compression_breakouts(self, df, ctx: StructureContext, symbol: str, vp_data) -> List[ReversalAlert]:
        """
        Detect tight range at key support/resistance
        
        LONG: Compression at support
        SHORT: Compression at resistance
        """
        alerts = []
        
        compressed = ctx.compression_ratio < 0.6  # Range < 60% of average
        
        if not compressed:
            return alerts
        
        # LONG: Compression at support
        near_support = ctx.position_in_30d_range < 0.25
        
        if near_support:
            score, signals = self._score_compression_long(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.COMPRESSION_BREAKOUT_LONG,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    trigger_level=ctx.nearest_support * 1.01,
                    target_level=ctx.current_price * 1.06,
                    stop_level=ctx.nearest_support * 0.985,
                    description=f"Range compression at support - breakout potential. Compression: {ctx.compression_ratio:.2f}",
                    timeframe="daily",
                    signals=signals
                )
                alerts.append(alert)
        
        # SHORT: Compression at resistance
        near_resistance = ctx.position_in_30d_range > 0.75
        
        if near_resistance:
            score, signals = self._score_compression_short(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.COMPRESSION_BREAKOUT_SHORT,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    trigger_level=ctx.nearest_resistance * 0.99,
                    target_level=ctx.current_price * 0.94,
                    stop_level=ctx.nearest_resistance * 1.015,
                    description=f"Range compression at resistance - breakdown potential. Compression: {ctx.compression_ratio:.2f}",
                    timeframe="daily",
                    signals=signals
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_divergences(self, df, ctx: StructureContext, symbol: str, vp_data) -> List[ReversalAlert]:
        """
        Detect multi-timeframe structure divergence
        
        LONG: Weekly bearish but daily showing bullish structure
        SHORT: Weekly bullish but daily showing bearish structure
        """
        alerts = []
        
        weekly_bullish = ctx.weekly_hh > ctx.weekly_ll and ctx.weekly_hl > ctx.weekly_lh
        weekly_bearish = ctx.weekly_ll > ctx.weekly_hh and ctx.weekly_lh > ctx.weekly_hl
        
        daily_bullish = (ctx.period_3d_hh or ctx.period_6d_hh) and (ctx.period_3d_hl or ctx.period_6d_hl)
        daily_bearish = (ctx.period_3d_lh or ctx.period_6d_lh) and (ctx.period_3d_ll or ctx.period_6d_ll)
        
        # LONG: Weekly bearish BUT daily bullish
        if weekly_bearish and daily_bullish:
            score, signals = self._score_divergence_long(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.STRUCTURE_DIVERGENCE_LONG,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    description=f"Structure divergence: Daily showing strength vs Weekly weakness - potential reversal up",
                    timeframe="weekly vs daily",
                    signals=signals
                )
                alert.divergence_score = min(self.divergence_weight, score * 0.1)
                alerts.append(alert)
        
        # SHORT: Weekly bullish BUT daily bearish
        if weekly_bullish and daily_bearish:
            score, signals = self._score_divergence_short(df, ctx, vp_data)
            
            if score >= self.min_confidence:
                alert = ReversalAlert(
                    alert_type=ReversalType.STRUCTURE_DIVERGENCE_SHORT,
                    severity=self._get_severity(score),
                    confidence=score,
                    symbol=symbol,
                    current_price=ctx.current_price,
                    description=f"Structure divergence: Daily showing weakness vs Weekly strength - potential reversal down",
                    timeframe="weekly vs daily",
                    signals=signals
                )
                alert.divergence_score = min(self.divergence_weight, score * 0.1)
                alerts.append(alert)
        
        return alerts
    
    # =========================================================================
    # SCORING HELPERS
    # =========================================================================
    
    def _score_structure_break_long(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score LONG structure break (LL in uptrend)"""
        score = 0.0
        signals = []
        
        # Base: Structure break occurred
        score += 25
        signals.append("Lower Low in uptrend")
        
        # Volume confirmation
        vol_score, vol_signals = self._score_volume(df, bearish=True)
        score += vol_score
        signals.extend(vol_signals)
        
        # VP confluence (at support)
        if vp_data:
            vp_score, vp_signals = self._score_vp_confluence(ctx.current_price, vp_data, looking_for_support=True)
            score += vp_score
            signals.extend(vp_signals)
        
        # Recent HL (fighting back)
        if ctx.period_3d_hl:
            score += 8
            signals.append("Higher Low also forming (indecision)")
        
        # Weekly close position
        if ctx.weekly_close_position < 0.3:
            score += 7
            signals.append("Weekly closed near lows")
        
        return score, signals
    
    def _score_structure_break_short(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score SHORT structure break (HH in downtrend)"""
        score = 0.0
        signals = []
        
        # Base: Structure break occurred
        score += 25
        signals.append("Higher High in downtrend")
        
        # Volume confirmation
        vol_score, vol_signals = self._score_volume(df, bearish=False)
        score += vol_score
        signals.extend(vol_signals)
        
        # VP confluence (at resistance)
        if vp_data:
            vp_score, vp_signals = self._score_vp_confluence(ctx.current_price, vp_data, looking_for_support=False)
            score += vp_score
            signals.extend(vp_signals)
        
        # Recent LH (fighting back)
        if ctx.period_3d_lh:
            score += 8
            signals.append("Lower High also forming (indecision)")
        
        # Weekly close position
        if ctx.weekly_close_position > 0.7:
            score += 7
            signals.append("Weekly closed near highs")
        
        return score, signals
    
    def _score_exhaustion_long(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score momentum exhaustion (uptrend losing steam)"""
        score = 0.0
        signals = []
        
        # Base: No HH in 6+ days
        score += 20
        signals.append("No new highs for 6+ days")
        
        # Declining volume
        vol_score, vol_signals = self._score_volume(df, bearish=True, check_declining=True)
        score += vol_score
        signals.extend(vol_signals)
        
        # At resistance
        near_res = ctx.position_in_30d_range > 0.7
        if near_res:
            score += 10
            signals.append("At top of range")
        
        # LH forming
        if ctx.period_3d_lh:
            score += 12
            signals.append("Lower High forming")
        
        return score, signals
    
    def _score_exhaustion_short(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score momentum exhaustion (downtrend losing steam)"""
        score = 0.0
        signals = []
        
        # Base: No LL in 6+ days
        score += 20
        signals.append("No new lows for 6+ days")
        
        # Declining volume
        vol_score, vol_signals = self._score_volume(df, bearish=False, check_declining=True)
        score += vol_score
        signals.extend(vol_signals)
        
        # At support
        near_sup = ctx.position_in_30d_range < 0.3
        if near_sup:
            score += 10
            signals.append("At bottom of range")
        
        # HL forming
        if ctx.period_3d_hl:
            score += 12
            signals.append("Higher Low forming")
        
        return score, signals
    
    def _score_range_extreme_long(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score range extreme reversal (bottom + HL)"""
        score = 0.0
        signals = []
        
        # Base: At bottom
        bottom_pct = (1.0 - ctx.position_in_30d_range) * 100
        score += min(20, bottom_pct * 0.5)
        signals.append(f"At {100 - bottom_pct:.1f}% of range")
        
        # HL forming
        score += 15
        signals.append("Higher Low forming")
        
        # Volume
        vol_score, vol_signals = self._score_volume(df, bearish=False)
        score += vol_score
        signals.extend(vol_signals)
        
        # VP at support
        if vp_data:
            vp_score, vp_signals = self._score_vp_confluence(ctx.current_price, vp_data, looking_for_support=True)
            score += vp_score
            signals.extend(vp_signals)
        
        return score, signals
    
    def _score_range_extreme_short(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score range extreme reversal (top + LH)"""
        score = 0.0
        signals = []
        
        # Base: At top
        top_pct = ctx.position_in_30d_range * 100
        score += min(20, top_pct * 0.5)
        signals.append(f"At {top_pct:.1f}% of range")
        
        # LH forming
        score += 15
        signals.append("Lower High forming")
        
        # Volume
        vol_score, vol_signals = self._score_volume(df, bearish=True)
        score += vol_score
        signals.extend(vol_signals)
        
        # VP at resistance
        if vp_data:
            vp_score, vp_signals = self._score_vp_confluence(ctx.current_price, vp_data, looking_for_support=False)
            score += vp_score
            signals.extend(vp_signals)
        
        return score, signals
    
    def _score_compression_long(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score compression breakout setup (bullish)"""
        score = 0.0
        signals = []
        
        # Base: Compression
        compression_score = (1.0 - ctx.compression_ratio) * 30
        score += compression_score
        signals.append(f"Range compression {ctx.compression_ratio:.2f}")
        
        # At support
        score += 15
        signals.append("At support zone")
        
        # Volume building
        vol_score, vol_signals = self._score_volume(df, bearish=False)
        score += vol_score
        signals.extend(vol_signals)
        
        # VP confluence
        if vp_data:
            vp_score, vp_signals = self._score_vp_confluence(ctx.current_price, vp_data, looking_for_support=True)
            score += vp_score
            signals.extend(vp_signals)
        
        return score, signals
    
    def _score_compression_short(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score compression breakdown setup (bearish)"""
        score = 0.0
        signals = []
        
        # Base: Compression
        compression_score = (1.0 - ctx.compression_ratio) * 30
        score += compression_score
        signals.append(f"Range compression {ctx.compression_ratio:.2f}")
        
        # At resistance
        score += 15
        signals.append("At resistance zone")
        
        # Volume building
        vol_score, vol_signals = self._score_volume(df, bearish=True)
        score += vol_score
        signals.extend(vol_signals)
        
        # VP confluence
        if vp_data:
            vp_score, vp_signals = self._score_vp_confluence(ctx.current_price, vp_data, looking_for_support=False)
            score += vp_score
            signals.extend(vp_signals)
        
        return score, signals
    
    def _score_divergence_long(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score MTF divergence (bearish weekly, bullish daily)"""
        score = 0.0
        signals = []
        
        # Base: Divergence
        score += 20
        signals.append("Daily structure turning bullish")
        
        # How bearish is weekly
        weekly_bear_strength = (ctx.weekly_ll + ctx.weekly_lh) / max(1, ctx.weekly_ll + ctx.weekly_lh + ctx.weekly_hh + ctx.weekly_hl)
        score += weekly_bear_strength * 15
        signals.append(f"Weekly bearish ({ctx.weekly_ll} LL, {ctx.weekly_lh} LH)")
        
        # Volume
        vol_score, vol_signals = self._score_volume(df, bearish=False)
        score += vol_score
        signals.extend(vol_signals)
        
        return score, signals
    
    def _score_divergence_short(self, df, ctx: StructureContext, vp_data) -> Tuple[float, List[str]]:
        """Score MTF divergence (bullish weekly, bearish daily)"""
        score = 0.0
        signals = []
        
        # Base: Divergence
        score += 20
        signals.append("Daily structure turning bearish")
        
        # How bullish is weekly
        weekly_bull_strength = (ctx.weekly_hh + ctx.weekly_hl) / max(1, ctx.weekly_ll + ctx.weekly_lh + ctx.weekly_hh + ctx.weekly_hl)
        score += weekly_bull_strength * 15
        signals.append(f"Weekly bullish ({ctx.weekly_hh} HH, {ctx.weekly_hl} HL)")
        
        # Volume
        vol_score, vol_signals = self._score_volume(df, bearish=True)
        score += vol_score
        signals.extend(vol_signals)
        
        return score, signals
    
    def _score_volume(self, df, bearish: bool, check_declining: bool = False) -> Tuple[float, List[str]]:
        """Score volume confirmation"""
        if len(df) < 20:
            return 0.0, []
        
        score = 0.0
        signals = []
        
        recent_vol = df['volume'].iloc[-5:].mean()
        avg_vol = df['volume'].iloc[-20:].mean()
        rvol = recent_vol / avg_vol if avg_vol > 0 else 1.0
        
        if check_declining:
            # Check if volume declining
            older_vol = df['volume'].iloc[-20:-5].mean()
            if recent_vol < older_vol * 0.85:
                score += 12
                signals.append("Volume declining")
        else:
            # Check if volume increasing
            if rvol > 1.3:
                score += 15
                signals.append(f"Volume {rvol:.1f}x average")
            elif rvol > 1.1:
                score += 8
                signals.append(f"Volume {rvol:.1f}x average")
        
        return score, signals
    
    def _score_vp_confluence(self, price: float, vp_data: Dict, looking_for_support: bool) -> Tuple[float, List[str]]:
        """Score Volume Profile confluence"""
        if not vp_data:
            return 0.0, []
        
        score = 0.0
        signals = []
        
        val = vp_data.get('val', 0)
        vah = vp_data.get('vah', 0)
        poc = vp_data.get('poc', 0)
        
        if looking_for_support:
            # Check if near VAL or POC support
            if val > 0 and abs(price - val) / val < 0.02:
                score += 12
                signals.append("At VAL support")
            elif poc > 0 and abs(price - poc) / poc < 0.015:
                score += 10
                signals.append("At POC")
        else:
            # Check if near VAH or POC resistance
            if vah > 0 and abs(price - vah) / vah < 0.02:
                score += 12
                signals.append("At VAH resistance")
            elif poc > 0 and abs(price - poc) / poc < 0.015:
                score += 10
                signals.append("At POC")
        
        return score, signals
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _is_uptrend(self, ctx: StructureContext) -> bool:
        """Check if in uptrend"""
        return "UPTREND" in ctx.weekly_trend.upper()
    
    def _is_downtrend(self, ctx: StructureContext) -> bool:
        """Check if in downtrend"""
        return "DOWNTREND" in ctx.weekly_trend.upper()
    
    def _get_severity(self, confidence: float) -> AlertSeverity:
        """Convert confidence to severity level"""
        if confidence >= 80:
            return AlertSeverity.CRITICAL
        elif confidence >= 60:
            return AlertSeverity.HIGH
        elif confidence >= 40:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
