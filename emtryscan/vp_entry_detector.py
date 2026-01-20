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
    """Detected entry signal"""
    entry_type: EntryType
    direction: Direction
    entry_price: float
    stop_price: float
    target_1: float
    target_2: Optional[float] = None
    confidence: float = 0.0  # 0-100
    notes: str = ""
    
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
                       avg_volume: float = None) -> List[EntrySignal]:
        """
        Detect all valid entry signals.
        
        Args:
            levels: Volume profile levels (VAH, POC, VAL)
            bars: Recent price bars (most recent last)
            avg_volume: Average volume for comparison (optional)
        
        Returns:
            List of detected entry signals
        """
        if len(bars) < 3:
            return []
        
        signals = []
        
        # Get current and recent bars
        current = bars[-1]
        prev = bars[-2]
        prev2 = bars[-3] if len(bars) >= 3 else None
        
        tolerance = levels.va_height * self.touch_tolerance_pct
        
        # Detect based on profile type
        if levels.profile_type == ProfileType.NORMAL:
            signals.extend(self._detect_normal_entries(levels, bars, tolerance, avg_volume))
        elif levels.profile_type == ProfileType.INVERTED:
            signals.extend(self._detect_inverted_entries(levels, bars, tolerance, avg_volume))
        else:
            # Check both if profile type unknown
            signals.extend(self._detect_normal_entries(levels, bars, tolerance, avg_volume))
            signals.extend(self._detect_inverted_entries(levels, bars, tolerance, avg_volume))
        
        return signals
    
    def _detect_normal_entries(self, 
                                levels: VolumeProfileLevels,
                                bars: List[PriceBar],
                                tolerance: float,
                                avg_volume: float) -> List[EntrySignal]:
        """Detect mean reversion entries for normal profiles"""
        signals = []
        current = bars[-1]
        prev = bars[-2]
        
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
                    confidence=self._calculate_confidence(current, levels, "val_touch"),
                    notes="VAL touch with bullish rejection candle"
                ))
        
        # 2. POC Reclaim
        if self._is_poc_reclaim(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.POC_RECLAIM,
                direction=Direction.LONG,
                entry_price=current.close,
                stop_price=levels.val - tolerance,
                target_1=levels.vah,
                confidence=self._calculate_confidence(current, levels, "poc_reclaim"),
                notes="Price reclaimed POC from below"
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
                confidence=self._calculate_confidence(current, levels, "failed_breakdown"),
                notes="Failed breakdown - bear trap detected"
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
                    confidence=self._calculate_confidence(current, levels, "vah_touch"),
                    notes="VAH touch with bearish rejection candle"
                ))
        
        # 5. POC Rejection
        if self._is_poc_rejection(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.POC_REJECTION,
                direction=Direction.SHORT,
                entry_price=current.close,
                stop_price=levels.vah + tolerance,
                target_1=levels.val,
                confidence=self._calculate_confidence(current, levels, "poc_rejection"),
                notes="Price rejected at POC from below"
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
                confidence=self._calculate_confidence(current, levels, "failed_breakout"),
                notes="Failed breakout - bull trap detected"
            ))
        
        return signals
    
    def _detect_inverted_entries(self,
                                  levels: VolumeProfileLevels,
                                  bars: List[PriceBar],
                                  tolerance: float,
                                  avg_volume: float) -> List[EntrySignal]:
        """Detect breakout entries for inverted profiles"""
        signals = []
        current = bars[-1]
        prev = bars[-2]
        
        # === BREAKOUT + RETEST ===
        
        # 7. Long: Broke above VAH, retesting
        if self._is_breakout_retest_long(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.BREAKOUT_RETEST_LONG,
                direction=Direction.LONG,
                entry_price=current.close,
                stop_price=levels.poc,  # Stop at POC (inside VA)
                target_1=levels.upper_ext_1x,
                confidence=self._calculate_confidence(current, levels, "breakout_retest_long"),
                notes="Breakout above VAH, retesting as support"
            ))
        
        # 8. Short: Broke below VAL, retesting
        if self._is_breakout_retest_short(bars, levels, tolerance):
            signals.append(EntrySignal(
                entry_type=EntryType.BREAKOUT_RETEST_SHORT,
                direction=Direction.SHORT,
                entry_price=current.close,
                stop_price=levels.poc,  # Stop at POC (inside VA)
                target_1=levels.lower_ext_1x,
                confidence=self._calculate_confidence(current, levels, "breakout_retest_short"),
                notes="Breakout below VAL, retesting as resistance"
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
                    confidence=self._calculate_confidence(current, levels, "lvn_rejection"),
                    notes="Bullish rejection from LVN zone"
                ))
            elif self._is_bearish_rejection(current):
                signals.append(EntrySignal(
                    entry_type=EntryType.LVN_REJECTION_SHORT,
                    direction=Direction.SHORT,
                    entry_price=current.close,
                    stop_price=levels.vah + tolerance,
                    target_1=levels.val,
                    confidence=self._calculate_confidence(current, levels, "lvn_rejection"),
                    notes="Bearish rejection from LVN zone"
                ))
        
        # === VOLUME CONFIRMATION BREAK ===
        
        # 11/12. Breakout with volume surge
        if avg_volume and current.volume >= avg_volume * self.volume_surge_multiplier:
            # Long break
            if current.close > levels.vah and current.is_bullish:
                signals.append(EntrySignal(
                    entry_type=EntryType.VOLUME_BREAK_LONG,
                    direction=Direction.LONG,
                    entry_price=current.close,
                    stop_price=levels.vah - tolerance,
                    target_1=levels.upper_ext_1x,
                    confidence=self._calculate_confidence(current, levels, "volume_break"),
                    notes=f"Volume breakout: {current.volume/avg_volume:.1f}x avg"
                ))
            # Short break
            elif current.close < levels.val and current.is_bearish:
                signals.append(EntrySignal(
                    entry_type=EntryType.VOLUME_BREAK_SHORT,
                    direction=Direction.SHORT,
                    entry_price=current.close,
                    stop_price=levels.val + tolerance,
                    target_1=levels.lower_ext_1x,
                    confidence=self._calculate_confidence(current, levels, "volume_break"),
                    notes=f"Volume breakdown: {current.volume/avg_volume:.1f}x avg"
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
