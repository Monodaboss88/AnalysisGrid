"""
Range Watcher - Multi-Period Higher High / Lower Low Structure Analysis
=======================================================================
Tracks swing structure across 3, 6, 9, 12, 15, and 30 day periods to identify:
- Trend structure (HH/HL = bullish, LH/LL = bearish)
- Range compression (narrowing ranges = breakout coming)
- Range expansion (volatility increasing)
- Key levels to watch (recent swing highs/lows)

Integrates with MTF Auction Scanner to provide macro context.

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

class TrendStructure(Enum):
    """Trend structure based on swing highs/lows"""
    STRONG_UPTREND = "STRONG_UPTREND"      # HH + HL consistently
    UPTREND = "UPTREND"                     # Mostly HH/HL
    WEAK_UPTREND = "WEAK_UPTREND"          # HH but HL breaking
    RANGE = "RANGE"                         # No clear direction
    WEAK_DOWNTREND = "WEAK_DOWNTREND"      # LH but LL holding
    DOWNTREND = "DOWNTREND"                 # Mostly LH/LL
    STRONG_DOWNTREND = "STRONG_DOWNTREND"  # LH + LL consistently
    
    @property
    def emoji(self) -> str:
        return {
            "STRONG_UPTREND": "🟢🟢",
            "UPTREND": "🟢",
            "WEAK_UPTREND": "🟢⚠️",
            "RANGE": "🟡",
            "WEAK_DOWNTREND": "🔴⚠️",
            "DOWNTREND": "🔴",
            "STRONG_DOWNTREND": "🔴🔴"
        }[self.value]
    
    @property
    def bias(self) -> str:
        if "UPTREND" in self.value:
            return "BULLISH"
        elif "DOWNTREND" in self.value:
            return "BEARISH"
        return "NEUTRAL"


class RangeState(Enum):
    """Range compression/expansion state"""
    COMPRESSING = "COMPRESSING"    # Range narrowing - breakout coming
    STABLE = "STABLE"              # Normal range
    EXPANDING = "EXPANDING"        # Range widening - volatility increasing
    EXTREME_COMPRESSION = "EXTREME_COMPRESSION"  # Very tight - imminent move
    EXTREME_EXPANSION = "EXTREME_EXPANSION"      # Blow-off / capitulation


@dataclass
class SwingPoint:
    """A swing high or low point"""
    date: datetime
    price: float
    type: str  # "HIGH" or "LOW"
    period_days: int  # How many days this was a swing point for
    
    @property
    def age_days(self) -> int:
        return (datetime.now() - self.date).days


@dataclass
class PeriodAnalysis:
    """Analysis for a single lookback period"""
    period_days: int
    
    # Range metrics
    high: float
    low: float
    range_size: float
    range_pct: float  # Range as % of price
    
    # Current position
    current_price: float
    position_in_range: float  # 0 = at low, 1 = at high
    
    # Swing structure
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    
    # Structure analysis
    higher_highs: bool  # Most recent high > previous high
    higher_lows: bool   # Most recent low > previous low
    lower_highs: bool   # Most recent high < previous high
    lower_lows: bool    # Most recent low < previous low
    
    # Key levels
    nearest_resistance: float
    nearest_support: float
    distance_to_resistance_pct: float
    distance_to_support_pct: float


@dataclass
class RangeWatcherResult:
    """Complete range analysis across all periods"""
    symbol: str
    current_price: float
    analysis_time: datetime
    
    # Period analyses
    periods: Dict[int, PeriodAnalysis]  # key = days (3, 6, 9, 12, 15, 30)
    
    # Aggregate assessment
    trend_structure: TrendStructure
    range_state: RangeState
    trend_strength: float  # -100 to +100
    
    # Key levels across all periods
    major_resistance_levels: List[Tuple[float, str]]  # (price, description)
    major_support_levels: List[Tuple[float, str]]
    
    # Actionable insights
    breakout_watch: Optional[float]  # Price level that confirms breakout
    breakdown_watch: Optional[float]  # Price level that confirms breakdown
    
    # Notes
    notes: List[str] = field(default_factory=list)


# =============================================================================
# SWING DETECTION ENGINE
# =============================================================================

class SwingDetector:
    """
    Detects swing highs and lows in price data.
    
    A swing high is a high that is higher than N bars on each side.
    A swing low is a low that is lower than N bars on each side.
    """
    
    def __init__(self, swing_strength: int = 3):
        """
        Args:
            swing_strength: Number of bars on each side to confirm swing (default 3)
        """
        self.swing_strength = swing_strength
    
    def find_swing_highs(self, df: pd.DataFrame, lookback_days: int = 30) -> List[SwingPoint]:
        """Find swing highs in the data"""
        swings = []
        n = self.swing_strength
        
        # Filter to lookback period - use data's last date, not datetime.now()
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            last_date = df['date'].max()
            cutoff = last_date - timedelta(days=lookback_days)
            df = df[df['date'] >= cutoff]
        elif isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index.max()
            cutoff = last_date - timedelta(days=lookback_days)
            df = df[df.index >= cutoff]
        
        highs = df['high'].values
        
        for i in range(n, len(highs) - n):
            # Check if this is a swing high
            is_swing = True
            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                if isinstance(df.index, pd.DatetimeIndex):
                    date = df.index[i]
                else:
                    date = df.iloc[i].get('date', datetime.now())
                
                swings.append(SwingPoint(
                    date=date if isinstance(date, datetime) else pd.to_datetime(date),
                    price=highs[i],
                    type="HIGH",
                    period_days=lookback_days
                ))
        
        return swings
    
    def find_swing_lows(self, df: pd.DataFrame, lookback_days: int = 30) -> List[SwingPoint]:
        """Find swing lows in the data"""
        swings = []
        n = self.swing_strength
        
        # Filter to lookback period - use data's last date, not datetime.now()
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            last_date = df['date'].max()
            cutoff = last_date - timedelta(days=lookback_days)
            df = df[df['date'] >= cutoff]
        elif isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index.max()
            cutoff = last_date - timedelta(days=lookback_days)
            df = df[df.index >= cutoff]
        
        lows = df['low'].values
        
        for i in range(n, len(lows) - n):
            # Check if this is a swing low
            is_swing = True
            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                if isinstance(df.index, pd.DatetimeIndex):
                    date = df.index[i]
                else:
                    date = df.iloc[i].get('date', datetime.now())
                
                swings.append(SwingPoint(
                    date=date if isinstance(date, datetime) else pd.to_datetime(date),
                    price=lows[i],
                    type="LOW",
                    period_days=lookback_days
                ))
        
        return swings


# =============================================================================
# RANGE WATCHER ENGINE
# =============================================================================

class RangeWatcher:
    """
    Multi-period range and structure analysis.
    
    Analyzes price action across 3, 6, 9, 12, 15, and 30 day periods
    to identify trend structure and key levels.
    """
    
    DEFAULT_PERIODS = [3, 6, 9, 12, 15, 30]
    
    def __init__(self, periods: List[int] = None, swing_strength: int = 2):
        """
        Args:
            periods: List of lookback periods in days (default: [3, 6, 9, 12, 15, 30])
            swing_strength: Bars on each side to confirm swing point
        """
        self.periods = periods or self.DEFAULT_PERIODS
        self.swing_detector = SwingDetector(swing_strength=swing_strength)
    
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> RangeWatcherResult:
        """
        Run complete range analysis.
        
        Args:
            df: DataFrame with OHLCV data (needs 'high', 'low', 'close' columns)
                Index should be datetime or have a 'date' column
            symbol: Stock symbol for labeling
        
        Returns:
            RangeWatcherResult with complete analysis
        """
        current_price = df['close'].iloc[-1]
        period_analyses = {}
        all_notes = []
        
        # Analyze each period
        for period in self.periods:
            analysis = self._analyze_period(df, period, current_price)
            period_analyses[period] = analysis
        
        # Determine overall trend structure
        trend_structure, trend_strength = self._determine_trend_structure(period_analyses)
        
        # Determine range state
        range_state = self._determine_range_state(period_analyses)
        
        # Find major support/resistance levels
        resistance_levels = self._find_major_resistance(period_analyses, current_price)
        support_levels = self._find_major_support(period_analyses, current_price)
        
        # Determine breakout/breakdown watch levels
        breakout_watch = None
        breakdown_watch = None
        
        if resistance_levels:
            # Breakout = clear the nearest major resistance
            breakout_watch = resistance_levels[0][0] * 1.002  # Slightly above
        
        if support_levels:
            # Breakdown = lose the nearest major support
            breakdown_watch = support_levels[0][0] * 0.998  # Slightly below
        
        # Generate notes
        all_notes = self._generate_notes(
            trend_structure, range_state, period_analyses, 
            current_price, resistance_levels, support_levels
        )
        
        return RangeWatcherResult(
            symbol=symbol,
            current_price=current_price,
            analysis_time=datetime.now(),
            periods=period_analyses,
            trend_structure=trend_structure,
            range_state=range_state,
            trend_strength=trend_strength,
            major_resistance_levels=resistance_levels,
            major_support_levels=support_levels,
            breakout_watch=breakout_watch,
            breakdown_watch=breakdown_watch,
            notes=all_notes
        )
    
    def _analyze_period(self, df: pd.DataFrame, period_days: int, current_price: float) -> PeriodAnalysis:
        """Analyze a single lookback period"""
        
        # Filter to period
        if isinstance(df.index, pd.DatetimeIndex):
            cutoff = df.index[-1] - timedelta(days=period_days)
            period_df = df[df.index >= cutoff]
        else:
            period_df = df.tail(period_days * 8)  # Rough estimate for intraday data
        
        # For short periods (3-6 days), accept less data points
        min_required = 2 if period_days <= 6 else 5
        if len(period_df) < min_required:
            # Not enough data
            return self._empty_period_analysis(period_days, current_price)
        
        # Basic range metrics
        high = period_df['high'].max()
        low = period_df['low'].min()
        range_size = high - low
        range_pct = (range_size / low) * 100 if low > 0 else 0
        
        # Position in range
        if range_size > 0:
            position_in_range = (current_price - low) / range_size
        else:
            position_in_range = 0.5
        
        # Find swing points
        swing_highs = self.swing_detector.find_swing_highs(period_df, period_days)
        swing_lows = self.swing_detector.find_swing_lows(period_df, period_days)
        
        # Determine HH/HL/LH/LL - pass period high/low for single-swing fallback
        higher_highs, lower_highs = self._analyze_swing_highs(swing_highs, period_high=high)
        higher_lows, lower_lows = self._analyze_swing_lows(swing_lows, period_low=low)
        
        # Find nearest support/resistance from swings
        nearest_resistance = high
        nearest_support = low
        
        # Look for swing highs above current price
        highs_above = [s.price for s in swing_highs if s.price > current_price]
        if highs_above:
            nearest_resistance = min(highs_above)
        
        # Look for swing lows below current price
        lows_below = [s.price for s in swing_lows if s.price < current_price]
        if lows_below:
            nearest_support = max(lows_below)
        
        # Distances
        dist_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
        dist_to_support = ((current_price - nearest_support) / current_price) * 100
        
        return PeriodAnalysis(
            period_days=period_days,
            high=high,
            low=low,
            range_size=range_size,
            range_pct=range_pct,
            current_price=current_price,
            position_in_range=position_in_range,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
            nearest_resistance=nearest_resistance,
            nearest_support=nearest_support,
            distance_to_resistance_pct=dist_to_resistance,
            distance_to_support_pct=dist_to_support
        )
    
    def _analyze_swing_highs(self, swings: List[SwingPoint], period_high: float = None) -> Tuple[bool, bool]:
        """Determine if making higher highs or lower highs"""
        if len(swings) < 2:
            # Fallback: if only 1 swing, compare to period high
            if len(swings) == 1 and period_high is not None:
                recent = swings[0].price
                # If the swing high IS the period high, assume HH
                higher_highs = abs(recent - period_high) / period_high < 0.005
                return higher_highs, False
            return False, False
        
        # Sort by date
        sorted_swings = sorted(swings, key=lambda x: x.date)
        
        # Compare last two swing highs
        recent = sorted_swings[-1].price
        previous = sorted_swings[-2].price
        
        higher_highs = recent > previous
        lower_highs = recent < previous
        
        return higher_highs, lower_highs
    
    def _analyze_swing_lows(self, swings: List[SwingPoint], period_low: float = None) -> Tuple[bool, bool]:
        """Determine if making higher lows or lower lows"""
        if len(swings) < 2:
            # Fallback: if only 1 swing, compare to period low
            if len(swings) == 1 and period_low is not None:
                recent = swings[0].price
                # If the swing low IS the period low, assume LL
                lower_lows = abs(recent - period_low) / period_low < 0.005
                return False, lower_lows
            return False, False
        
        # Sort by date
        sorted_swings = sorted(swings, key=lambda x: x.date)
        
        # Compare last two swing lows
        recent = sorted_swings[-1].price
        previous = sorted_swings[-2].price
        
        higher_lows = recent > previous
        lower_lows = recent < previous
        
        return higher_lows, lower_lows
    
    def _determine_trend_structure(self, period_analyses: Dict[int, PeriodAnalysis]) -> Tuple[TrendStructure, float]:
        """Determine overall trend structure from all periods"""
        
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Weight longer periods more heavily
        weights = {3: 1, 6: 1.5, 9: 2, 12: 2.5, 15: 3, 30: 4}
        
        for period, analysis in period_analyses.items():
            weight = weights.get(period, 1)
            
            # HH + HL = bullish
            if analysis.higher_highs and analysis.higher_lows:
                bullish_signals += 2 * weight
            elif analysis.higher_highs:
                bullish_signals += 1 * weight
            elif analysis.higher_lows:
                bullish_signals += 1 * weight
            
            # LH + LL = bearish
            if analysis.lower_highs and analysis.lower_lows:
                bearish_signals += 2 * weight
            elif analysis.lower_highs:
                bearish_signals += 1 * weight
            elif analysis.lower_lows:
                bearish_signals += 1 * weight
            
            total_signals += 2 * weight
        
        # Calculate trend strength (-100 to +100)
        if total_signals > 0:
            trend_strength = ((bullish_signals - bearish_signals) / total_signals) * 100
        else:
            trend_strength = 0
        
        # Determine structure
        if trend_strength >= 70:
            structure = TrendStructure.STRONG_UPTREND
        elif trend_strength >= 40:
            structure = TrendStructure.UPTREND
        elif trend_strength >= 15:
            structure = TrendStructure.WEAK_UPTREND
        elif trend_strength <= -70:
            structure = TrendStructure.STRONG_DOWNTREND
        elif trend_strength <= -40:
            structure = TrendStructure.DOWNTREND
        elif trend_strength <= -15:
            structure = TrendStructure.WEAK_DOWNTREND
        else:
            structure = TrendStructure.RANGE
        
        return structure, trend_strength
    
    def _determine_range_state(self, period_analyses: Dict[int, PeriodAnalysis]) -> RangeState:
        """Determine if ranges are compressing or expanding"""
        
        # Compare shorter period range % to longer period range %
        short_range = period_analyses.get(6, period_analyses.get(3))
        long_range = period_analyses.get(30, period_analyses.get(15))
        
        if short_range is None or long_range is None:
            return RangeState.STABLE
        
        # Ratio of short-term range to long-term range
        if long_range.range_pct > 0:
            compression_ratio = short_range.range_pct / long_range.range_pct
        else:
            compression_ratio = 1.0
        
        if compression_ratio < 0.3:
            return RangeState.EXTREME_COMPRESSION
        elif compression_ratio < 0.5:
            return RangeState.COMPRESSING
        elif compression_ratio > 1.5:
            return RangeState.EXTREME_EXPANSION
        elif compression_ratio > 1.2:
            return RangeState.EXPANDING
        else:
            return RangeState.STABLE
    
    def _find_major_resistance(self, period_analyses: Dict[int, PeriodAnalysis], 
                               current_price: float) -> List[Tuple[float, str]]:
        """Find major resistance levels above current price"""
        levels = []
        
        for period, analysis in period_analyses.items():
            # Period high
            if analysis.high > current_price:
                levels.append((analysis.high, f"{period}D High"))
            
            # Swing highs
            for swing in analysis.swing_highs:
                if swing.price > current_price:
                    levels.append((swing.price, f"{period}D Swing High"))
        
        # Deduplicate similar levels (within 0.5%)
        unique_levels = []
        for price, desc in sorted(levels, key=lambda x: x[0]):
            if not unique_levels or (price - unique_levels[-1][0]) / unique_levels[-1][0] > 0.005:
                unique_levels.append((price, desc))
        
        return unique_levels[:5]  # Top 5 nearest
    
    def _find_major_support(self, period_analyses: Dict[int, PeriodAnalysis],
                            current_price: float) -> List[Tuple[float, str]]:
        """Find major support levels below current price"""
        levels = []
        
        for period, analysis in period_analyses.items():
            # Period low
            if analysis.low < current_price:
                levels.append((analysis.low, f"{period}D Low"))
            
            # Swing lows
            for swing in analysis.swing_lows:
                if swing.price < current_price:
                    levels.append((swing.price, f"{period}D Swing Low"))
        
        # Deduplicate similar levels (within 0.5%)
        unique_levels = []
        for price, desc in sorted(levels, key=lambda x: x[0], reverse=True):
            if not unique_levels or (unique_levels[-1][0] - price) / price > 0.005:
                unique_levels.append((price, desc))
        
        return unique_levels[:5]  # Top 5 nearest
    
    def _generate_notes(self, trend: TrendStructure, range_state: RangeState,
                        period_analyses: Dict[int, PeriodAnalysis],
                        current_price: float,
                        resistance: List, support: List) -> List[str]:
        """Generate actionable notes"""
        notes = []
        
        # Trend note
        notes.append(f"Trend Structure: {trend.emoji} {trend.value} (Bias: {trend.bias})")
        
        # Range state
        if range_state == RangeState.EXTREME_COMPRESSION:
            notes.append("⚡ EXTREME COMPRESSION - Major move imminent")
        elif range_state == RangeState.COMPRESSING:
            notes.append("📐 Range compressing - watch for breakout")
        elif range_state == RangeState.EXPANDING:
            notes.append("📈 Range expanding - volatility increasing")
        elif range_state == RangeState.EXTREME_EXPANSION:
            notes.append("🌋 EXTREME EXPANSION - Possible blow-off/capitulation")
        
        # Position in range
        p30 = period_analyses.get(30)
        if p30:
            if p30.position_in_range > 0.9:
                notes.append(f"⚠️ At top of 30D range ({p30.position_in_range:.0%})")
            elif p30.position_in_range < 0.1:
                notes.append(f"⚠️ At bottom of 30D range ({p30.position_in_range:.0%})")
        
        # Key level proximity
        if resistance and resistance[0][0]:
            dist = ((resistance[0][0] - current_price) / current_price) * 100
            if dist < 1:
                notes.append(f"🎯 Near resistance: ${resistance[0][0]:.2f} ({dist:.1f}% away)")
        
        if support and support[0][0]:
            dist = ((current_price - support[0][0]) / current_price) * 100
            if dist < 1:
                notes.append(f"🎯 Near support: ${support[0][0]:.2f} ({dist:.1f}% away)")
        
        # Divergence between timeframes
        short_trend = self._get_period_trend(period_analyses.get(3))
        long_trend = self._get_period_trend(period_analyses.get(30))
        
        if short_trend == "BULLISH" and long_trend == "BEARISH":
            notes.append("⚠️ Short-term bullish in longer-term downtrend - potential bear flag")
        elif short_trend == "BEARISH" and long_trend == "BULLISH":
            notes.append("⚠️ Short-term bearish in longer-term uptrend - potential bull flag")
        
        return notes
    
    def _get_period_trend(self, analysis: Optional[PeriodAnalysis]) -> str:
        """Get simple trend for a single period"""
        if analysis is None:
            return "NEUTRAL"
        
        if analysis.higher_highs and analysis.higher_lows:
            return "BULLISH"
        elif analysis.lower_highs and analysis.lower_lows:
            return "BEARISH"
        return "NEUTRAL"
    
    def _empty_period_analysis(self, period_days: int, current_price: float) -> PeriodAnalysis:
        """Return empty analysis for insufficient data"""
        return PeriodAnalysis(
            period_days=period_days,
            high=current_price,
            low=current_price,
            range_size=0,
            range_pct=0,
            current_price=current_price,
            position_in_range=0.5,
            swing_highs=[],
            swing_lows=[],
            higher_highs=False,
            higher_lows=False,
            lower_highs=False,
            lower_lows=False,
            nearest_resistance=current_price,
            nearest_support=current_price,
            distance_to_resistance_pct=0,
            distance_to_support_pct=0
        )
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_report(self, result: RangeWatcherResult) -> str:
        """Generate formatted report"""
        lines = []
        
        lines.append("=" * 70)
        lines.append(f"📊 RANGE WATCHER: {result.symbol}")
        lines.append(f"   Price: ${result.current_price:.2f}")
        lines.append(f"   Time: {result.analysis_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        
        # Trend Structure
        lines.append(f"\n{result.trend_structure.emoji} TREND: {result.trend_structure.value}")
        lines.append(f"   Strength: {result.trend_strength:+.0f} (Bias: {result.trend_structure.bias})")
        lines.append(f"   Range State: {result.range_state.value}")
        
        # Period Summary Table
        lines.append("\n" + "-" * 70)
        lines.append("PERIOD ANALYSIS:")
        lines.append("-" * 70)
        lines.append(f"{'Period':>8} | {'High':>10} | {'Low':>10} | {'Range%':>7} | {'Position':>8} | Structure")
        lines.append("-" * 70)
        
        for period in sorted(result.periods.keys()):
            p = result.periods[period]
            
            # Structure indicator
            if p.higher_highs and p.higher_lows:
                struct = "🟢 HH+HL"
            elif p.lower_highs and p.lower_lows:
                struct = "🔴 LH+LL"
            elif p.higher_highs:
                struct = "🟢 HH"
            elif p.higher_lows:
                struct = "🟢 HL"
            elif p.lower_highs:
                struct = "🔴 LH"
            elif p.lower_lows:
                struct = "🔴 LL"
            else:
                struct = "🟡 ---"
            
            lines.append(f"{period:>6}D | ${p.high:>9.2f} | ${p.low:>9.2f} | {p.range_pct:>6.1f}% | {p.position_in_range:>7.0%} | {struct}")
        
        # Key Levels
        lines.append("\n" + "-" * 70)
        lines.append("KEY LEVELS:")
        lines.append("-" * 70)
        
        lines.append("\n📈 RESISTANCE (above):")
        for price, desc in result.major_resistance_levels[:3]:
            dist = ((price - result.current_price) / result.current_price) * 100
            lines.append(f"   ${price:.2f} ({desc}) - {dist:+.1f}%")
        
        lines.append("\n📉 SUPPORT (below):")
        for price, desc in result.major_support_levels[:3]:
            dist = ((result.current_price - price) / result.current_price) * 100
            lines.append(f"   ${price:.2f} ({desc}) - {dist:.1f}% cushion")
        
        # Watch Levels
        lines.append("\n" + "-" * 70)
        lines.append("⚡ WATCH LEVELS:")
        if result.breakout_watch:
            lines.append(f"   Breakout above: ${result.breakout_watch:.2f}")
        if result.breakdown_watch:
            lines.append(f"   Breakdown below: ${result.breakdown_watch:.2f}")
        
        # Notes
        if result.notes:
            lines.append("\n" + "-" * 70)
            lines.append("📝 NOTES:")
            for note in result.notes:
                lines.append(f"   • {note}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# DATA FETCHING (Optional - uses yfinance if available)
# =============================================================================

def fetch_data(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a symbol.
    
    Tries yfinance first, falls back to demo data.
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d")
        
        if len(df) > 0:
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            return df
    except ImportError:
        pass
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    
    return None


def generate_demo_data(days: int = 60, start_price: float = 450.0) -> pd.DataFrame:
    """Generate realistic demo data for testing"""
    import numpy as np
    
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate price series with trend and mean reversion
    price = start_price
    data = []
    
    for i, date in enumerate(dates):
        # Add some trend
        trend = 0.001 * np.sin(i / 20)  # Slow oscillation
        noise = np.random.randn() * 0.015
        
        daily_return = trend + noise
        
        open_price = price
        close_price = price * (1 + daily_return)
        
        # Generate realistic high/low
        intraday_vol = abs(np.random.randn()) * 0.01
        high_price = max(open_price, close_price) * (1 + intraday_vol)
        low_price = min(open_price, close_price) * (1 - intraday_vol)
        
        volume = int(np.random.exponential(5000000))
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
    
    df = pd.DataFrame(data, index=dates)
    return df


# =============================================================================
# CLI DEMO
# =============================================================================

def main():
    """Demo the range watcher"""
    import sys
    
    print("=" * 70)
    print("RANGE WATCHER - Multi-Period Structure Analysis")
    print("=" * 70)
    
    # Get symbol from command line or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    print(f"\nAnalyzing: {symbol}")
    
    # Try to fetch real data
    df = fetch_data(symbol, days=60)
    
    if df is None or len(df) < 30:
        print("Using demo data...")
        df = generate_demo_data(days=60)
        symbol = "DEMO"
    else:
        print(f"Fetched {len(df)} days of data")
    
    # Run analysis
    watcher = RangeWatcher()
    result = watcher.analyze(df, symbol=symbol)
    
    # Print report
    print(watcher.print_report(result))


if __name__ == "__main__":
    main()
