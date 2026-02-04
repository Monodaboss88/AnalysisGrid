"""
Volume Profile Entry Detection Module
=====================================
Detects the 10 entry scenarios from the volume profile playbook.

Requirements:
- VAH, POC, VAL levels (input or calculated)
- OHLCV price data
- Volume data for confirmation

Integration: Call detect_entries() with your data to get active signals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
import numpy as np


class ProfileType(Enum):
    NORMAL = "normal"       # Fat middle (high volume at POC)
    INVERTED = "inverted"   # Thin middle (low volume at POC, high at extremes)
    NEUTRAL = "neutral"     # Unclear shape


class EntryType(Enum):
    # Normal Profile - Longs (Mean Reversion)
    VAL_TOUCH_REJECTION = "val_touch_rejection"
    POC_RECLAIM = "poc_reclaim"
    FAILED_BREAKDOWN = "failed_breakdown"
    
    # Normal Profile - Shorts (Mean Reversion)
    VAH_TOUCH_REJECTION = "vah_touch_rejection"
    POC_REJECTION = "poc_rejection"
    FAILED_BREAKOUT = "failed_breakout"
    
    # Inverted Profile - Breakouts
    BREAKOUT_RETEST_LONG = "breakout_retest_long"
    BREAKOUT_RETEST_SHORT = "breakout_retest_short"
    LVN_REJECTION_LONG = "lvn_rejection_long"
    LVN_REJECTION_SHORT = "lvn_rejection_short"
    VOLUME_BREAK_LONG = "volume_break_long"
    VOLUME_BREAK_SHORT = "volume_break_short"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class VolumeProfileLevels:
    """Volume profile key levels"""
    vah: float              # Value Area High
    poc: float              # Point of Control
    val: float              # Value Area Low
    profile_type: ProfileType = ProfileType.NEUTRAL
    
    @property
    def va_height(self) -> float:
        """Value area height for extension calculations"""
        return self.vah - self.val
    
    @property
    def upper_ext_1x(self) -> float:
        """1x extension above VAH"""
        return self.vah + self.va_height
    
    @property
    def lower_ext_1x(self) -> float:
        """1x extension below VAL"""
        return self.val - self.va_height


@dataclass
class PriceBar:
    """Single price bar/candle"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Optional[str] = None
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class EntrySignal:
    """Detected entry signal with enhanced context"""
    entry_type: EntryType
    direction: Direction
    entry_price: float
    stop_price: float
    target_1: float
    target_2: Optional[float] = None
    confidence: float = 0.0  # 0-100
    notes: str = ""
    
    # NEW: Enhanced context factors
    trend_context: str = "neutral"      # "with_trend", "counter_trend", "neutral"
    volume_confirmation: str = "normal" # "surge", "normal", "drying"
    rvol: float = 1.0                    # Relative volume on signal bar
    rsi_confluence: bool = False         # RSI also at extreme
    rsi_value: float = 50.0              # Current RSI
    prior_level_tests: int = 0           # Times level has held before
    session_context: str = "unknown"     # "open", "mid", "close"
    consecutive_bars: int = 0            # Bars in direction before signal
    candle_quality: str = "normal"       # "strong", "normal", "weak"
    
    @property
    def risk(self) -> float:
        if self.direction == Direction.LONG:
            return self.entry_price - self.stop_price
        return self.stop_price - self.entry_price
    
    @property
    def reward_1(self) -> float:
        if self.direction == Direction.LONG:
            return self.target_1 - self.entry_price
        return self.entry_price - self.target_1
    
    @property
    def rr_ratio(self) -> float:
        if self.risk > 0:
            return self.reward_1 / self.risk
        return 0
    
    def to_dict(self) -> dict:
        """Serialize to dict for API response"""
        return {
            'entry_type': self.entry_type.value,
            'direction': self.direction.value,
            'entry_price': round(self.entry_price, 2),
            'stop_price': round(self.stop_price, 2),
            'target_1': round(self.target_1, 2),
            'target_2': round(self.target_2, 2) if self.target_2 else None,
            'risk': round(self.risk, 2),
            'reward_1': round(self.reward_1, 2),
            'rr_ratio': round(self.rr_ratio, 2),
            'confidence': round(self.confidence, 1),
            'notes': self.notes,
            # Context factors
            'trend_context': self.trend_context,
            'volume_confirmation': self.volume_confirmation,
            'rvol': round(self.rvol, 2),
            'rsi_confluence': self.rsi_confluence,
            'rsi_value': round(self.rsi_value, 1),
            'prior_level_tests': self.prior_level_tests,
            'session_context': self.session_context,
            'consecutive_bars': self.consecutive_bars,
            'candle_quality': self.candle_quality
        }


class VolumeProfileEntryDetector:
    """
    Detects volume profile entry setups.
    
    Usage:
        detector = VolumeProfileEntryDetector()
        levels = VolumeProfileLevels(vah=100, poc=98, val=96)
        bars = [PriceBar(...), ...]  # Recent price bars
        
        signals = detector.detect_entries(levels, bars, avg_volume=1000000)
    """
    
    def __init__(self, 
                 touch_tolerance_pct: float = 0.1,
                 rejection_wick_ratio: float = 0.5,
                 reclaim_buffer_pct: float = 0.05,
                 volume_surge_multiplier: float = 1.5):
        """
        Args:
            touch_tolerance_pct: How close price must be to level (% of VA height)
            rejection_wick_ratio: Min wick/body ratio for rejection candle
            reclaim_buffer_pct: Buffer above/below level for reclaim confirmation
            volume_surge_multiplier: Volume multiple for "surge" confirmation
        """
        self.touch_tolerance_pct = touch_tolerance_pct
        self.rejection_wick_ratio = rejection_wick_ratio
        self.reclaim_buffer_pct = reclaim_buffer_pct
        self.volume_surge_multiplier = volume_surge_multiplier
    
    def detect_entries(self, 
                       levels: VolumeProfileLevels, 
                       bars: List[PriceBar],
                       avg_volume: float = None,
                       rsi: float = None,
                       timestamp = None) -> List[EntrySignal]:
        """
        Detect all valid entry signals with enhanced context.
        
        Args:
            levels: Volume profile levels (VAH, POC, VAL)
            bars: Recent price bars (most recent last)
            avg_volume: Average volume for comparison (optional)
            rsi: Current RSI value (optional, for confluence)
            timestamp: Current bar timestamp (optional, for session context)
        
        Returns:
            List of detected entry signals with context factors
        """
        if len(bars) < 3:
            return []
        
        signals = []
        
        # Get current and recent bars
        current = bars[-1]
        prev = bars[-2]
        prev2 = bars[-3] if len(bars) >= 3 else None
        
        tolerance = levels.va_height * self.touch_tolerance_pct
        
        # Calculate context factors once for all signals
        context = self._calculate_context(bars, levels, avg_volume, rsi, timestamp)
        
        # Detect based on profile type
        if levels.profile_type == ProfileType.NORMAL:
            signals.extend(self._detect_normal_entries(levels, bars, tolerance, avg_volume, context))
        elif levels.profile_type == ProfileType.INVERTED:
            signals.extend(self._detect_inverted_entries(levels, bars, tolerance, avg_volume, context))
        else:
            # Check both if profile type unknown
            signals.extend(self._detect_normal_entries(levels, bars, tolerance, avg_volume, context))
            signals.extend(self._detect_inverted_entries(levels, bars, tolerance, avg_volume, context))
        
        return signals
    
    def _calculate_context(self, 
                           bars: List[PriceBar], 
                           levels: VolumeProfileLevels,
                           avg_volume: float,
                           rsi: float,
                           timestamp) -> dict:
        """Calculate all context factors for signal enhancement"""
        current = bars[-1]
        
        # 1. Trend context (simple: compare to 20 bars ago)
        trend_context = "neutral"
        if len(bars) >= 20:
            old_price = bars[-20].close
            current_price = current.close
            change_pct = (current_price - old_price) / old_price
            if change_pct > 0.02:
                trend_context = "uptrend"
            elif change_pct < -0.02:
                trend_context = "downtrend"
        
        # 2. Volume confirmation
        rvol = 1.0
        volume_confirmation = "normal"
        if avg_volume and avg_volume > 0:
            rvol = current.volume / avg_volume
            if rvol >= self.volume_surge_multiplier:
                volume_confirmation = "surge"
            elif rvol < 0.6:
                volume_confirmation = "drying"
        
        # 3. RSI confluence
        rsi_value = rsi if rsi else 50.0
        rsi_confluence = False
        if rsi:
            rsi_confluence = rsi <= 35 or rsi >= 65  # Near extremes
        
        # 4. Session context
        session_context = "unknown"
        if timestamp and hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            if 9 <= hour < 11:
                session_context = "open"
            elif 11 <= hour < 14:
                session_context = "mid"
            elif 14 <= hour <= 16:
                session_context = "close"
        
        # 5. Consecutive bars in same direction
        consecutive_up = 0
        consecutive_down = 0
        for bar in reversed(bars[:-1]):  # Exclude current bar
            if bar.is_bullish:
                if consecutive_down > 0:
                    break
                consecutive_up += 1
            elif bar.is_bearish:
                if consecutive_up > 0:
                    break
                consecutive_down += 1
            else:
                break
        
        # 6. Candle quality
        candle_quality = self._assess_candle_quality(current)
        
        # 7. Prior level tests (count touches at VAL and VAH)
        prior_val_tests = self._count_level_tests(bars[:-1], levels.val, levels.va_height * 0.1)
        prior_vah_tests = self._count_level_tests(bars[:-1], levels.vah, levels.va_height * 0.1)
        
        return {
            'trend_context': trend_context,
            'volume_confirmation': volume_confirmation,
            'rvol': rvol,
            'rsi_confluence': rsi_confluence,
            'rsi_value': rsi_value,
            'session_context': session_context,
            'consecutive_up': consecutive_up,
            'consecutive_down': consecutive_down,
            'candle_quality': candle_quality,
            'prior_val_tests': prior_val_tests,
            'prior_vah_tests': prior_vah_tests
        }
    
    def _assess_candle_quality(self, bar: PriceBar) -> str:
        """Assess quality of the signal candle"""
        if bar.range == 0:
            return "weak"
        
        body_ratio = bar.body_size / bar.range
        
        # Strong: big body, small wicks
        if body_ratio > 0.7:
            return "strong"
        # Weak: small body, big wicks (indecision)
        elif body_ratio < 0.3:
            return "weak"
        return "normal"
    
    def _count_level_tests(self, bars: List[PriceBar], level: float, tolerance: float) -> int:
        """Count how many times price tested a level and held"""
        tests = 0
        for i, bar in enumerate(bars[-20:]):  # Last 20 bars
            # Check if bar touched level
            if bar.low <= level + tolerance and bar.high >= level - tolerance:
                # Check if it bounced (next bar if exists)
                if i < len(bars) - 1:
                    next_bar = bars[i + 1] if i + 1 < len(bars) else None
                    if next_bar and next_bar.close > level:
                        tests += 1
        return min(tests, 5)  # Cap at 5
    
    def _detect_normal_entries(self, 
                                levels: VolumeProfileLevels,
                                bars: List[PriceBar],
                                tolerance: float,
                                avg_volume: float,
                                context: dict = None) -> List[EntrySignal]:
        """Detect mean reversion entries for normal profiles"""
        signals = []
        current = bars[-1]
        prev = bars[-2]
        ctx = context or {}
        
        # Determine trend alignment for LONG entries
        long_trend = "with_trend" if ctx.get('trend_context') == "uptrend" else (
            "counter_trend" if ctx.get('trend_context') == "downtrend" else "neutral")
        short_trend = "with_trend" if ctx.get('trend_context') == "downtrend" else (
            "counter_trend" if ctx.get('trend_context') == "uptrend" else "neutral")
        
        # === LONG ENTRIES ===
        
        # 1. VAL Touch + Rejection
        if self._is_touching_level(current, levels.val, tolerance):
            if self._is_bullish_rejection(current):
                signals.append(EntrySignal(
                    entry_type=EntryType.VAL_TOUCH_REJECTION,
                    direction=Direction.LONG,
                    entry_price=current.close,
                    stop_price=levels.val - tolerance,
                    target_1=levels.poc,
                    target_2=levels.vah,
                    confidence=self._calculate_confidence_enhanced(current, levels, "val_touch", ctx, Direction.LONG),
                    notes="VAL touch with bullish rejection candle",
                    trend_context=long_trend,
                    volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                    rvol=ctx.get('rvol', 1.0),
                    rsi_confluence=ctx.get('rsi_confluence', False),
                    rsi_value=ctx.get('rsi_value', 50.0),
                    prior_level_tests=ctx.get('prior_val_tests', 0),
                    session_context=ctx.get('session_context', 'unknown'),
                    consecutive_bars=ctx.get('consecutive_down', 0),
                    candle_quality=ctx.get('candle_quality', 'normal')
                ))
        
        # 2. POC Reclaim
        if self._is_poc_reclaim(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.POC_RECLAIM,
                direction=Direction.LONG,
                entry_price=current.close,
                stop_price=levels.val - tolerance,
                target_1=levels.vah,
                confidence=self._calculate_confidence_enhanced(current, levels, "poc_reclaim", ctx, Direction.LONG),
                notes="Price reclaimed POC from below",
                trend_context=long_trend,
                volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                rvol=ctx.get('rvol', 1.0),
                rsi_confluence=ctx.get('rsi_confluence', False),
                rsi_value=ctx.get('rsi_value', 50.0),
                prior_level_tests=0,
                session_context=ctx.get('session_context', 'unknown'),
                consecutive_bars=ctx.get('consecutive_down', 0),
                candle_quality=ctx.get('candle_quality', 'normal')
            ))
        
        # 3. Failed Breakdown (Bear Trap)
        if self._is_failed_breakdown(bars, levels, tolerance):
            swing_low = min(b.low for b in bars[-5:])
            signals.append(EntrySignal(
                entry_type=EntryType.FAILED_BREAKDOWN,
                direction=Direction.LONG,
                entry_price=current.close,
                stop_price=swing_low - tolerance,
                target_1=levels.vah,
                confidence=self._calculate_confidence_enhanced(current, levels, "failed_breakdown", ctx, Direction.LONG),
                notes="Failed breakdown - bear trap detected",
                trend_context=long_trend,
                volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                rvol=ctx.get('rvol', 1.0),
                rsi_confluence=ctx.get('rsi_confluence', False),
                rsi_value=ctx.get('rsi_value', 50.0),
                prior_level_tests=ctx.get('prior_val_tests', 0),
                session_context=ctx.get('session_context', 'unknown'),
                consecutive_bars=ctx.get('consecutive_down', 0),
                candle_quality=ctx.get('candle_quality', 'normal')
            ))
        
        # === SHORT ENTRIES ===
        
        # 4. VAH Touch + Rejection
        if self._is_touching_level(current, levels.vah, tolerance):
            if self._is_bearish_rejection(current):
                signals.append(EntrySignal(
                    entry_type=EntryType.VAH_TOUCH_REJECTION,
                    direction=Direction.SHORT,
                    entry_price=current.close,
                    stop_price=levels.vah + tolerance,
                    target_1=levels.poc,
                    target_2=levels.val,
                    confidence=self._calculate_confidence_enhanced(current, levels, "vah_touch", ctx, Direction.SHORT),
                    notes="VAH touch with bearish rejection candle",
                    trend_context=short_trend,
                    volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                    rvol=ctx.get('rvol', 1.0),
                    rsi_confluence=ctx.get('rsi_confluence', False),
                    rsi_value=ctx.get('rsi_value', 50.0),
                    prior_level_tests=ctx.get('prior_vah_tests', 0),
                    session_context=ctx.get('session_context', 'unknown'),
                    consecutive_bars=ctx.get('consecutive_up', 0),
                    candle_quality=ctx.get('candle_quality', 'normal')
                ))
        
        # 5. POC Rejection
        if self._is_poc_rejection(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.POC_REJECTION,
                direction=Direction.SHORT,
                entry_price=current.close,
                stop_price=levels.vah + tolerance,
                target_1=levels.val,
                confidence=self._calculate_confidence_enhanced(current, levels, "poc_rejection", ctx, Direction.SHORT),
                notes="Price rejected at POC from below",
                trend_context=short_trend,
                volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                rvol=ctx.get('rvol', 1.0),
                rsi_confluence=ctx.get('rsi_confluence', False),
                rsi_value=ctx.get('rsi_value', 50.0),
                prior_level_tests=0,
                session_context=ctx.get('session_context', 'unknown'),
                consecutive_bars=ctx.get('consecutive_up', 0),
                candle_quality=ctx.get('candle_quality', 'normal')
            ))
        
        # 6. Failed Breakout (Bull Trap)
        if self._is_failed_breakout(bars, levels, tolerance):
            swing_high = max(b.high for b in bars[-5:])
            signals.append(EntrySignal(
                entry_type=EntryType.FAILED_BREAKOUT,
                direction=Direction.SHORT,
                entry_price=current.close,
                stop_price=swing_high + tolerance,
                target_1=levels.val,
                confidence=self._calculate_confidence_enhanced(current, levels, "failed_breakout", ctx, Direction.SHORT),
                notes="Failed breakout - bull trap detected",
                trend_context=short_trend,
                volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                rvol=ctx.get('rvol', 1.0),
                rsi_confluence=ctx.get('rsi_confluence', False),
                rsi_value=ctx.get('rsi_value', 50.0),
                prior_level_tests=ctx.get('prior_vah_tests', 0),
                session_context=ctx.get('session_context', 'unknown'),
                consecutive_bars=ctx.get('consecutive_up', 0),
                candle_quality=ctx.get('candle_quality', 'normal')
            ))
        
        return signals
    
    def _detect_inverted_entries(self,
                                  levels: VolumeProfileLevels,
                                  bars: List[PriceBar],
                                  tolerance: float,
                                  avg_volume: float,
                                  context: dict = None) -> List[EntrySignal]:
        """Detect breakout entries for inverted profiles"""
        signals = []
        current = bars[-1]
        prev = bars[-2]
        ctx = context or {}
        
        # Determine trend alignment
        long_trend = "with_trend" if ctx.get('trend_context') == "uptrend" else (
            "counter_trend" if ctx.get('trend_context') == "downtrend" else "neutral")
        short_trend = "with_trend" if ctx.get('trend_context') == "downtrend" else (
            "counter_trend" if ctx.get('trend_context') == "uptrend" else "neutral")
        
        # === BREAKOUT + RETEST ===
        
        # 7. Long: Broke above VAH, retesting
        if self._is_breakout_retest_long(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.BREAKOUT_RETEST_LONG,
                direction=Direction.LONG,
                entry_price=current.close,
                stop_price=levels.poc,  # Stop at POC (inside VA)
                target_1=levels.upper_ext_1x,
                confidence=self._calculate_confidence_enhanced(current, levels, "breakout_retest_long", ctx, Direction.LONG),
                notes="Breakout above VAH, retesting as support",
                trend_context=long_trend,
                volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                rvol=ctx.get('rvol', 1.0),
                rsi_confluence=ctx.get('rsi_confluence', False),
                rsi_value=ctx.get('rsi_value', 50.0),
                prior_level_tests=ctx.get('prior_vah_tests', 0),
                session_context=ctx.get('session_context', 'unknown'),
                consecutive_bars=ctx.get('consecutive_up', 0),
                candle_quality=ctx.get('candle_quality', 'normal')
            ))
        
        # 8. Short: Broke below VAL, retesting
        if self._is_breakout_retest_short(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.BREAKOUT_RETEST_SHORT,
                direction=Direction.SHORT,
                entry_price=current.close,
                stop_price=levels.poc,  # Stop at POC (inside VA)
                target_1=levels.lower_ext_1x,
                confidence=self._calculate_confidence_enhanced(current, levels, "breakout_retest_short", ctx, Direction.SHORT),
                notes="Breakout below VAL, retesting as resistance",
                trend_context=short_trend,
                volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                rvol=ctx.get('rvol', 1.0),
                rsi_confluence=ctx.get('rsi_confluence', False),
                rsi_value=ctx.get('rsi_value', 50.0),
                prior_level_tests=ctx.get('prior_val_tests', 0),
                session_context=ctx.get('session_context', 'unknown'),
                consecutive_bars=ctx.get('consecutive_down', 0),
                candle_quality=ctx.get('candle_quality', 'normal')
            ))
        
        # === LVN SPEED ZONE ===
        
        # 9/10. LVN Rejection (price enters thin middle, gets rejected)
        if self._is_in_lvn_zone(current, levels):
            if self._is_bullish_rejection(current):
                signals.append(EntrySignal(
                    entry_type=EntryType.LVN_REJECTION_LONG,
                    direction=Direction.LONG,
                    entry_price=current.close,
                    stop_price=levels.val - tolerance,
                    target_1=levels.vah,
                    confidence=self._calculate_confidence_enhanced(current, levels, "lvn_rejection", ctx, Direction.LONG),
                    notes="Bullish rejection from LVN zone",
                    trend_context=long_trend,
                    volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                    rvol=ctx.get('rvol', 1.0),
                    rsi_confluence=ctx.get('rsi_confluence', False),
                    rsi_value=ctx.get('rsi_value', 50.0),
                    prior_level_tests=0,
                    session_context=ctx.get('session_context', 'unknown'),
                    consecutive_bars=ctx.get('consecutive_down', 0),
                    candle_quality=ctx.get('candle_quality', 'normal')
                ))
            elif self._is_bearish_rejection(current):
                signals.append(EntrySignal(
                    entry_type=EntryType.LVN_REJECTION_SHORT,
                    direction=Direction.SHORT,
                    entry_price=current.close,
                    stop_price=levels.vah + tolerance,
                    target_1=levels.val,
                    confidence=self._calculate_confidence_enhanced(current, levels, "lvn_rejection", ctx, Direction.SHORT),
                    notes="Bearish rejection from LVN zone",
                    trend_context=short_trend,
                    volume_confirmation=ctx.get('volume_confirmation', 'normal'),
                    rvol=ctx.get('rvol', 1.0),
                    rsi_confluence=ctx.get('rsi_confluence', False),
                    rsi_value=ctx.get('rsi_value', 50.0),
                    prior_level_tests=0,
                    session_context=ctx.get('session_context', 'unknown'),
                    consecutive_bars=ctx.get('consecutive_up', 0),
                    candle_quality=ctx.get('candle_quality', 'normal')
                ))
        
        # === VOLUME CONFIRMATION BREAK ===
        
        # 11/12. Breakout with volume surge
        if avg_volume and current.volume >= avg_volume * self.volume_surge_multiplier:
            rvol = current.volume / avg_volume
            # Long break
            if current.close > levels.vah and current.is_bullish:
                signals.append(EntrySignal(
                    entry_type=EntryType.VOLUME_BREAK_LONG,
                    direction=Direction.LONG,
                    entry_price=current.close,
                    stop_price=levels.vah - tolerance,
                    target_1=levels.upper_ext_1x,
                    confidence=self._calculate_confidence_enhanced(current, levels, "volume_break", ctx, Direction.LONG),
                    notes=f"Volume breakout: {rvol:.1f}x avg",
                    trend_context=long_trend,
                    volume_confirmation="surge",
                    rvol=rvol,
                    rsi_confluence=ctx.get('rsi_confluence', False),
                    rsi_value=ctx.get('rsi_value', 50.0),
                    prior_level_tests=ctx.get('prior_vah_tests', 0),
                    session_context=ctx.get('session_context', 'unknown'),
                    consecutive_bars=ctx.get('consecutive_up', 0),
                    candle_quality=ctx.get('candle_quality', 'normal')
                ))
            # Short break
            elif current.close < levels.val and current.is_bearish:
                signals.append(EntrySignal(
                    entry_type=EntryType.VOLUME_BREAK_SHORT,
                    direction=Direction.SHORT,
                    entry_price=current.close,
                    stop_price=levels.val + tolerance,
                    target_1=levels.lower_ext_1x,
                    confidence=self._calculate_confidence_enhanced(current, levels, "volume_break", ctx, Direction.SHORT),
                    notes=f"Volume breakdown: {rvol:.1f}x avg",
                    trend_context=short_trend,
                    volume_confirmation="surge",
                    rvol=rvol,
                    rsi_confluence=ctx.get('rsi_confluence', False),
                    rsi_value=ctx.get('rsi_value', 50.0),
                    prior_level_tests=ctx.get('prior_val_tests', 0),
                    session_context=ctx.get('session_context', 'unknown'),
                    consecutive_bars=ctx.get('consecutive_down', 0),
                    candle_quality=ctx.get('candle_quality', 'normal')
                ))
        
        return signals
    
    # === HELPER METHODS ===
    
    def _is_touching_level(self, bar: PriceBar, level: float, tolerance: float) -> bool:
        """Check if price bar is touching a level"""
        return bar.low <= level + tolerance and bar.high >= level - tolerance
    
    def _is_bullish_rejection(self, bar: PriceBar) -> bool:
        """Check for bullish rejection candle (long lower wick)"""
        if bar.range == 0:
            return False
        wick_ratio = bar.lower_wick / bar.range
        return wick_ratio >= self.rejection_wick_ratio and bar.is_bullish
    
    def _is_bearish_rejection(self, bar: PriceBar) -> bool:
        """Check for bearish rejection candle (long upper wick)"""
        if bar.range == 0:
            return False
        wick_ratio = bar.upper_wick / bar.range
        return wick_ratio >= self.rejection_wick_ratio and bar.is_bearish
    
    def _is_poc_reclaim(self, bars: List[PriceBar], levels: VolumeProfileLevels, tolerance: float) -> bool:
        """Check if price dipped below POC and reclaimed it"""
        if len(bars) < 3:
            return False
        current = bars[-1]
        prev = bars[-2]
        
        # Previous bar closed below POC, current bar closed above
        dipped_below = prev.close < levels.poc
        reclaimed = current.close > levels.poc + (levels.va_height * self.reclaim_buffer_pct)
        
        return dipped_below and reclaimed and current.is_bullish
    
    def _is_poc_rejection(self, bars: List[PriceBar], levels: VolumeProfileLevels, tolerance: float) -> bool:
        """Check if price rallied to POC and got rejected"""
        if len(bars) < 3:
            return False
        current = bars[-1]
        prev = bars[-2]
        
        # Approached POC from below, failed to hold above
        approached = prev.high >= levels.poc - tolerance
        rejected = current.close < levels.poc and current.is_bearish
        came_from_below = bars[-3].close < levels.poc if len(bars) >= 3 else True
        
        return approached and rejected and came_from_below
    
    def _is_failed_breakdown(self, bars: List[PriceBar], levels: VolumeProfileLevels, tolerance: float) -> bool:
        """Check for failed breakdown (bear trap)"""
        if len(bars) < 4:
            return False
        current = bars[-1]
        
        # Look for break below VAL followed by reclaim
        broke_val = any(b.low < levels.val - tolerance for b in bars[-4:-1])
        reclaimed = current.close > levels.val + (levels.va_height * self.reclaim_buffer_pct)
        
        return broke_val and reclaimed and current.is_bullish
    
    def _is_failed_breakout(self, bars: List[PriceBar], levels: VolumeProfileLevels, tolerance: float) -> bool:
        """Check for failed breakout (bull trap)"""
        if len(bars) < 4:
            return False
        current = bars[-1]
        
        # Look for break above VAH followed by reversal back inside
        broke_vah = any(b.high > levels.vah + tolerance for b in bars[-4:-1])
        reversed = current.close < levels.vah - (levels.va_height * self.reclaim_buffer_pct)
        
        return broke_vah and reversed and current.is_bearish
    
    def _is_breakout_retest_long(self, bars: List[PriceBar], levels: VolumeProfileLevels, tolerance: float) -> bool:
        """Check for breakout above VAH followed by retest"""
        if len(bars) < 5:
            return False
        current = bars[-1]
        
        # Need: Prior close above VAH, pullback to VAH, holding
        prior_break = any(b.close > levels.vah + tolerance for b in bars[-5:-2])
        retesting = current.low <= levels.vah + tolerance and current.low >= levels.vah - tolerance
        holding = current.close > levels.vah
        
        return prior_break and retesting and holding
    
    def _is_breakout_retest_short(self, bars: List[PriceBar], levels: VolumeProfileLevels, tolerance: float) -> bool:
        """Check for breakout below VAL followed by retest"""
        if len(bars) < 5:
            return False
        current = bars[-1]
        
        # Need: Prior close below VAL, rally to VAL, rejecting
        prior_break = any(b.close < levels.val - tolerance for b in bars[-5:-2])
        retesting = current.high >= levels.val - tolerance and current.high <= levels.val + tolerance
        rejecting = current.close < levels.val
        
        return prior_break and retesting and rejecting
    
    def _is_in_lvn_zone(self, bar: PriceBar, levels: VolumeProfileLevels) -> bool:
        """Check if price is in the LVN (low volume node) zone around POC"""
        lvn_upper = levels.poc + (levels.va_height * 0.15)
        lvn_lower = levels.poc - (levels.va_height * 0.15)
        return bar.close >= lvn_lower and bar.close <= lvn_upper
    
    def _calculate_confidence(self, bar: PriceBar, levels: VolumeProfileLevels, setup_type: str) -> float:
        """
        Calculate confidence score for the setup (0-100).
        Override this method to add your own confidence factors.
        """
        base_confidence = 50.0
        
        # Add confidence for strong rejection candles
        if bar.range > 0:
            wick_strength = max(bar.lower_wick, bar.upper_wick) / bar.range
            base_confidence += wick_strength * 20
        
        # Add confidence for closes near entry level
        # (customize based on setup type)
        
        return min(100, max(0, base_confidence))
    
    def _calculate_confidence_enhanced(self, 
                                        bar: PriceBar, 
                                        levels: VolumeProfileLevels, 
                                        setup_type: str,
                                        context: dict,
                                        direction: Direction) -> float:
        """
        Enhanced confidence calculation with context factors.
        
        Scoring breakdown:
        - Base: 40 points
        - Candle quality: up to 15 points
        - Trend alignment: up to 15 points
        - Volume confirmation: up to 10 points
        - RSI confluence: up to 10 points
        - Prior level tests: up to 10 points
        - Session timing: up to 5 points
        - Consecutive momentum: up to 5 points
        """
        score = 40.0  # Base score
        
        # 1. Candle quality (0-15 points)
        candle_quality = context.get('candle_quality', 'normal')
        if candle_quality == 'strong':
            score += 15
        elif candle_quality == 'normal':
            score += 8
        # weak = 0 points
        
        # Also check rejection wick strength
        if bar.range > 0:
            if direction == Direction.LONG:
                wick_ratio = bar.lower_wick / bar.range
            else:
                wick_ratio = bar.upper_wick / bar.range
            score += wick_ratio * 5  # Up to 5 bonus for strong rejection
        
        # 2. Trend alignment (0-15 points)
        trend = context.get('trend_context', 'neutral')
        if direction == Direction.LONG:
            if trend == 'uptrend':
                score += 15  # With trend
            elif trend == 'neutral':
                score += 8
            # downtrend = 0 (counter-trend)
        else:  # SHORT
            if trend == 'downtrend':
                score += 15  # With trend
            elif trend == 'neutral':
                score += 8
            # uptrend = 0 (counter-trend)
        
        # 3. Volume confirmation (0-10 points)
        volume_conf = context.get('volume_confirmation', 'normal')
        if setup_type in ['volume_break', 'failed_breakdown', 'failed_breakout']:
            # For breakout/trap setups, we want volume surge
            if volume_conf == 'surge':
                score += 10
            elif volume_conf == 'normal':
                score += 5
        else:
            # For mean reversion, drying volume on approach is good
            if volume_conf == 'drying':
                score += 10
            elif volume_conf == 'normal':
                score += 5
        
        # 4. RSI confluence (0-10 points)
        if context.get('rsi_confluence', False):
            rsi = context.get('rsi_value', 50)
            if direction == Direction.LONG and rsi <= 35:
                score += 10  # Oversold + long = great confluence
            elif direction == Direction.SHORT and rsi >= 65:
                score += 10  # Overbought + short = great confluence
            else:
                score += 5  # Partial confluence
        
        # 5. Prior level tests (0-10 points)
        prior_tests = context.get('prior_val_tests', 0) if direction == Direction.LONG else context.get('prior_vah_tests', 0)
        score += min(10, prior_tests * 3)  # 3 points per test, max 10
        
        # 6. Session timing (0-5 points)
        session = context.get('session_context', 'unknown')
        if session == 'close':
            score += 5  # End of day entries more reliable
        elif session == 'mid':
            score += 3
        # open = 0 (volatile)
        
        # 7. Consecutive momentum into signal (0-5 points)
        if direction == Direction.LONG:
            consec = context.get('consecutive_down', 0)
        else:
            consec = context.get('consecutive_up', 0)
        
        if consec >= 3:
            score += 5  # Good exhaustion signal
        elif consec >= 2:
            score += 3
        
        return min(100, max(0, score))


def classify_profile(volume_at_vah: float, 
                     volume_at_poc: float, 
                     volume_at_val: float) -> ProfileType:
    """
    Classify volume profile as normal or inverted.
    
    Args:
        volume_at_vah: Volume at/near VAH
        volume_at_poc: Volume at/near POC
        volume_at_val: Volume at/near VAL
    
    Returns:
        ProfileType enum
    """
    edge_volume = (volume_at_vah + volume_at_val) / 2
    
    if volume_at_poc > edge_volume * 1.3:
        return ProfileType.NORMAL  # Fat middle
    elif volume_at_poc < edge_volume * 0.7:
        return ProfileType.INVERTED  # Thin middle
    else:
        return ProfileType.NEUTRAL


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    # Example: Detect entries for a stock
    
    # 1. Define your VP levels (from your chart or calculated)
    levels = VolumeProfileLevels(
        vah=152.50,
        poc=150.00,
        val=147.50,
        profile_type=ProfileType.NORMAL
    )
    
    # 2. Create recent price bars (most recent last)
    bars = [
        PriceBar(open=149.00, high=150.50, low=148.50, close=150.00, volume=1000000),
        PriceBar(open=150.00, high=150.25, low=147.25, close=147.50, volume=1200000),
        PriceBar(open=147.50, high=148.75, low=147.00, close=148.50, volume=1500000),  # Rejection at VAL
    ]
    
    # 3. Run detection
    detector = VolumeProfileEntryDetector()
    signals = detector.detect_entries(levels, bars, avg_volume=1000000)
    
    # 4. Print results
    for signal in signals:
        print(f"\n{'='*50}")
        print(f"SIGNAL: {signal.entry_type.value}")
        print(f"Direction: {signal.direction.value.upper()}")
        print(f"Entry: ${signal.entry_price:.2f}")
        print(f"Stop: ${signal.stop_price:.2f}")
        print(f"Target 1: ${signal.target_1:.2f}")
        if signal.target_2:
            print(f"Target 2: ${signal.target_2:.2f}")
        print(f"R:R Ratio: {signal.rr_ratio:.2f}")
        print(f"Confidence: {signal.confidence:.0f}%")
        print(f"Notes: {signal.notes}")
