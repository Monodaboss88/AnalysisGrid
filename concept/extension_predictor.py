"""
SEF Trading System - Extension Duration Predictor
Tracks consecutive candles in extension and predicts resolution

Simple Logic:
- 1 candle extended = noise, ignore
- 2 candles extended (4 hours) = attention, something brewing
- 3 candles extended (6 hours) = high probability of resolution soon
- 4+ candles extended (8+ hours) = extreme tension, snap-back imminent

Works for:
- Price vs VWAP (daily, weekly, monthly)
- Price vs POC (daily, weekly)
- Price vs VAH/VAL (outside value area)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ExtensionZone(Enum):
    """Where price is relative to fair value"""
    EXTREME_ABOVE = "extreme_above"    # Way above (> 2 ATR from VWAP)
    ABOVE_VALUE = "above_value"        # Above VAH or > 1 ATR above VWAP
    IN_VALUE = "in_value"              # Between VAL and VAH, near VWAP
    BELOW_VALUE = "below_value"        # Below VAL or > 1 ATR below VWAP
    EXTREME_BELOW = "extreme_below"    # Way below (> 2 ATR from VWAP)


class TriggerLevel(Enum):
    """Alert/trigger levels based on extension duration"""
    NONE = 0           # No significant extension
    WATCHING = 1       # 1 candle - just watching
    ALERT = 2          # 2 candles (4 hours) - pay attention
    HIGH_PROB = 3      # 3 candles (6 hours) - high probability setup
    EXTREME = 4        # 4+ candles (8+ hours) - snap-back imminent


class ResolutionBias(Enum):
    """Expected resolution direction"""
    SNAP_BACK = "snap_back"          # Return to value
    CONTINUATION = "continuation"     # Keep going
    CONSOLIDATE = "consolidate"       # Chop sideways
    UNDETERMINED = "undetermined"


@dataclass
class CandleInExtension:
    """Single candle's extension data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Extension metrics
    vwap: float
    poc: float
    vah: float
    val: float
    atr: float
    
    # Calculated
    distance_from_vwap: float = 0.0
    distance_from_poc: float = 0.0
    zone: ExtensionZone = ExtensionZone.IN_VALUE
    
    # Candle characteristics
    is_rejection: bool = False  # Long wick in direction of value
    is_continuation: bool = False  # Strong close away from value
    body_vs_range: float = 0.0  # Body / total range
    
    def __post_init__(self):
        # Calculate distances
        mid_price = (self.high + self.low + self.close) / 3
        self.distance_from_vwap = mid_price - self.vwap
        self.distance_from_poc = mid_price - self.poc
        
        # Determine zone
        self.zone = self._determine_zone(mid_price)
        
        # Candle characteristics
        self._analyze_candle()
    
    def _determine_zone(self, price: float) -> ExtensionZone:
        """Determine which zone price is in"""
        vwap_distance_atr = abs(price - self.vwap) / self.atr if self.atr > 0 else 0
        
        if price > self.vwap:
            if vwap_distance_atr > 2.0 or price > self.vah + self.atr:
                return ExtensionZone.EXTREME_ABOVE
            elif price > self.vah or vwap_distance_atr > 1.0:
                return ExtensionZone.ABOVE_VALUE
        else:
            if vwap_distance_atr > 2.0 or price < self.val - self.atr:
                return ExtensionZone.EXTREME_BELOW
            elif price < self.val or vwap_distance_atr > 1.0:
                return ExtensionZone.BELOW_VALUE
        
        return ExtensionZone.IN_VALUE
    
    def _analyze_candle(self):
        """Analyze candle structure"""
        range_ = self.high - self.low
        body = abs(self.close - self.open)
        
        if range_ > 0:
            self.body_vs_range = body / range_
            
            # Rejection candle: long wick pointing toward value
            upper_wick = self.high - max(self.open, self.close)
            lower_wick = min(self.open, self.close) - self.low
            
            if self.zone in [ExtensionZone.ABOVE_VALUE, ExtensionZone.EXTREME_ABOVE]:
                # Above value - rejection = long upper wick (sellers rejecting highs)
                self.is_rejection = (upper_wick / range_) > 0.5
                self.is_continuation = self.close > self.open and (lower_wick / range_) < 0.2
            
            elif self.zone in [ExtensionZone.BELOW_VALUE, ExtensionZone.EXTREME_BELOW]:
                # Below value - rejection = long lower wick (buyers rejecting lows)
                self.is_rejection = (lower_wick / range_) > 0.5
                self.is_continuation = self.close < self.open and (upper_wick / range_) < 0.2


@dataclass
class ExtensionStreak:
    """Tracks consecutive candles in extension"""
    level_name: str  # "vwap_daily", "vah_daily", etc.
    direction: str  # "above" or "below"
    
    # Streak data
    candles: List[CandleInExtension] = field(default_factory=list)
    streak_count: int = 0
    
    # Timing
    start_time: Optional[datetime] = None
    duration_minutes: int = 0
    duration_hours: float = 0.0
    
    # Extension metrics
    avg_extension: float = 0.0  # Average distance during streak
    max_extension: float = 0.0  # Peak extension
    total_extension_time: float = 0.0  # Σ(extension × time)
    
    # Trigger level
    trigger: TriggerLevel = TriggerLevel.NONE
    
    # Resolution prediction
    resolution_bias: ResolutionBias = ResolutionBias.UNDETERMINED
    snap_back_probability: float = 0.5
    
    # Targets
    snap_back_target: float = 0.0  # Where to target on mean reversion
    continuation_target: float = 0.0  # Where to target if trend continues
    stop_level: float = 0.0  # Invalidation
    
    @property
    def is_actionable(self) -> bool:
        """Ready for a trade setup?"""
        return self.trigger.value >= TriggerLevel.ALERT.value
    
    @property
    def hours_extended(self) -> float:
        return self.duration_minutes / 60


@dataclass
class ExtensionAlert:
    """Alert when extension reaches actionable threshold"""
    timestamp: datetime
    symbol: str
    
    level_name: str  # What level price is extended from
    direction: str  # "above" or "below"
    trigger_level: TriggerLevel
    
    # Streak info
    candle_count: int
    hours_extended: float
    
    # Current state
    current_price: float
    reference_level: float
    extension_distance: float
    extension_atr: float
    
    # Prediction
    resolution_bias: ResolutionBias
    snap_back_probability: float
    
    # Trade setup
    snap_back_target: float
    stop_loss: float
    risk_reward: float
    
    # Score
    quality_score: float  # 0-100
    
    def __str__(self) -> str:
        return (
            f"EXTENSION ALERT: {self.symbol}\n"
            f"  Level: {self.level_name} ({self.direction})\n"
            f"  Duration: {self.candle_count} candles ({self.hours_extended:.1f} hours)\n"
            f"  Trigger: {self.trigger_level.name}\n"
            f"  Snap-back probability: {self.snap_back_probability:.0%}\n"
            f"  Target: {self.snap_back_target:.2f}\n"
            f"  Score: {self.quality_score:.0f}"
        )


class ExtensionDurationPredictor:
    """
    Main predictor class
    Tracks extension duration and generates predictions
    """
    
    # Thresholds for 2H candles
    CANDLE_THRESHOLDS = {
        1: TriggerLevel.WATCHING,
        2: TriggerLevel.ALERT,       # 4 hours
        3: TriggerLevel.HIGH_PROB,   # 6 hours
        4: TriggerLevel.EXTREME,     # 8+ hours
    }
    
    # Historical resolution probabilities by candle count
    # Based on typical market behavior
    SNAP_BACK_PROBABILITIES = {
        1: 0.45,  # Coin flip
        2: 0.55,  # Slight edge to snap-back
        3: 0.65,  # Good probability
        4: 0.75,  # High probability
        5: 0.80,  # Very high
        6: 0.85,  # Extreme
    }
    
    def __init__(self, candle_minutes: int = 120):
        """
        Args:
            candle_minutes: Minutes per candle (default 120 = 2 hours)
        """
        self.candle_minutes = candle_minutes
        
        # Active streaks by level
        self._streaks: Dict[str, ExtensionStreak] = {}
        
        # Historical data for pattern matching
        self._resolution_history: List[Dict] = []
        
        # Current state
        self._last_candle: Optional[CandleInExtension] = None
    
    def update(
        self,
        candle: CandleInExtension
    ) -> List[ExtensionAlert]:
        """
        Update predictor with new candle data
        
        Args:
            candle: New 2H candle with extension data
            
        Returns:
            List of alerts if any thresholds crossed
        """
        alerts = []
        
        # Check each reference level
        levels_to_check = [
            ("vwap_daily", candle.vwap, candle.distance_from_vwap),
            ("poc_daily", candle.poc, candle.distance_from_poc),
            ("vah", candle.vah, candle.close - candle.vah if candle.close > candle.vah else 0),
            ("val", candle.val, candle.val - candle.close if candle.close < candle.val else 0),
        ]
        
        for level_name, ref_level, distance in levels_to_check:
            alert = self._check_level(candle, level_name, ref_level, distance)
            if alert:
                alerts.append(alert)
        
        self._last_candle = candle
        return alerts
    
    def _check_level(
        self,
        candle: CandleInExtension,
        level_name: str,
        ref_level: float,
        distance: float
    ) -> Optional[ExtensionAlert]:
        """Check extension status for a specific level"""
        
        # Determine if extended
        is_extended = abs(distance) > (candle.atr * 0.5)  # Extended if > 0.5 ATR
        direction = "above" if distance > 0 else "below"
        
        streak_key = f"{level_name}_{direction}"
        
        if is_extended:
            # Continue or start streak
            if streak_key not in self._streaks:
                # Start new streak
                self._streaks[streak_key] = ExtensionStreak(
                    level_name=level_name,
                    direction=direction,
                    start_time=candle.timestamp
                )
            
            streak = self._streaks[streak_key]
            streak.candles.append(candle)
            streak.streak_count += 1
            streak.duration_minutes += self.candle_minutes
            streak.duration_hours = streak.duration_minutes / 60
            
            # Update metrics
            self._update_streak_metrics(streak, candle, ref_level, distance)
            
            # Check if threshold crossed
            old_trigger = streak.trigger
            streak.trigger = self._get_trigger_level(streak.streak_count)
            
            # Generate alert if threshold just crossed
            if streak.trigger.value > old_trigger.value and streak.trigger.value >= TriggerLevel.ALERT.value:
                return self._create_alert(streak, candle, ref_level)
        
        else:
            # Extension ended - record resolution
            if streak_key in self._streaks:
                self._record_resolution(self._streaks[streak_key], candle)
                del self._streaks[streak_key]
            
            # Also check opposite direction
            opposite_key = f"{level_name}_{'below' if direction == 'above' else 'above'}"
            if opposite_key in self._streaks:
                self._record_resolution(self._streaks[opposite_key], candle)
                del self._streaks[opposite_key]
        
        return None
    
    def _update_streak_metrics(
        self,
        streak: ExtensionStreak,
        candle: CandleInExtension,
        ref_level: float,
        distance: float
    ):
        """Update streak metrics with new candle"""
        
        # Extension metrics
        abs_distance = abs(distance)
        streak.max_extension = max(streak.max_extension, abs_distance)
        
        # Running average
        streak.avg_extension = (
            (streak.avg_extension * (streak.streak_count - 1) + abs_distance) 
            / streak.streak_count
        )
        
        # Total extension-time (distance × minutes)
        streak.total_extension_time += abs_distance * self.candle_minutes
        
        # Resolution prediction
        streak.snap_back_probability = self._calculate_snap_back_prob(streak, candle)
        streak.resolution_bias = self._determine_resolution_bias(streak, candle)
        
        # Calculate targets
        streak.snap_back_target = ref_level  # Simple: target the reference level
        
        if streak.direction == "above":
            streak.continuation_target = candle.high + candle.atr
            streak.stop_level = ref_level - (candle.atr * 0.5)
        else:
            streak.continuation_target = candle.low - candle.atr
            streak.stop_level = ref_level + (candle.atr * 0.5)
    
    def _get_trigger_level(self, candle_count: int) -> TriggerLevel:
        """Get trigger level based on candle count"""
        if candle_count >= 4:
            return TriggerLevel.EXTREME
        return self.CANDLE_THRESHOLDS.get(candle_count, TriggerLevel.NONE)
    
    def _calculate_snap_back_prob(
        self,
        streak: ExtensionStreak,
        candle: CandleInExtension
    ) -> float:
        """Calculate probability of snap-back"""
        
        # Base probability by candle count
        base_prob = self.SNAP_BACK_PROBABILITIES.get(
            min(streak.streak_count, 6), 
            0.85
        )
        
        # Adjust based on candle characteristics
        adjustments = 0.0
        
        # Rejection candle increases snap-back probability
        if candle.is_rejection:
            adjustments += 0.10
        
        # Continuation candle decreases it
        if candle.is_continuation:
            adjustments -= 0.10
        
        # Extreme extension increases probability
        if candle.zone in [ExtensionZone.EXTREME_ABOVE, ExtensionZone.EXTREME_BELOW]:
            adjustments += 0.05
        
        # Volume analysis (if volume declining in extension, higher snap-back prob)
        if len(streak.candles) >= 2:
            recent_vol = streak.candles[-1].volume
            prev_vol = streak.candles[-2].volume
            if recent_vol < prev_vol * 0.8:  # Volume declining
                adjustments += 0.05
        
        return min(0.95, max(0.30, base_prob + adjustments))
    
    def _determine_resolution_bias(
        self,
        streak: ExtensionStreak,
        candle: CandleInExtension
    ) -> ResolutionBias:
        """Determine most likely resolution type"""
        
        if streak.snap_back_probability > 0.65:
            return ResolutionBias.SNAP_BACK
        elif streak.snap_back_probability < 0.45:
            return ResolutionBias.CONTINUATION
        else:
            return ResolutionBias.UNDETERMINED
    
    def _create_alert(
        self,
        streak: ExtensionStreak,
        candle: CandleInExtension,
        ref_level: float
    ) -> ExtensionAlert:
        """Create alert for actionable extension"""
        
        current_price = candle.close
        extension_distance = abs(current_price - ref_level)
        extension_atr = extension_distance / candle.atr if candle.atr > 0 else 0
        
        # Calculate risk/reward
        if streak.direction == "above":
            risk = abs(current_price - streak.stop_level)
            reward = abs(current_price - streak.snap_back_target)
        else:
            risk = abs(streak.stop_level - current_price)
            reward = abs(streak.snap_back_target - current_price)
        
        risk_reward = reward / risk if risk > 0 else 0
        
        # Quality score
        quality_score = self._calculate_quality_score(streak, candle, risk_reward)
        
        return ExtensionAlert(
            timestamp=candle.timestamp,
            symbol="",  # Set by caller
            level_name=streak.level_name,
            direction=streak.direction,
            trigger_level=streak.trigger,
            candle_count=streak.streak_count,
            hours_extended=streak.hours_extended,
            current_price=current_price,
            reference_level=ref_level,
            extension_distance=extension_distance,
            extension_atr=extension_atr,
            resolution_bias=streak.resolution_bias,
            snap_back_probability=streak.snap_back_probability,
            snap_back_target=streak.snap_back_target,
            stop_loss=streak.stop_level,
            risk_reward=risk_reward,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(
        self,
        streak: ExtensionStreak,
        candle: CandleInExtension,
        risk_reward: float
    ) -> float:
        """Calculate overall quality score for the setup"""
        score = 0.0
        
        # Duration score (up to 30 points)
        duration_score = min(30, streak.streak_count * 8)
        score += duration_score
        
        # Probability score (up to 30 points)
        prob_score = streak.snap_back_probability * 30
        score += prob_score
        
        # Risk/reward score (up to 20 points)
        rr_score = min(20, risk_reward * 10)
        score += rr_score
        
        # Candle confirmation score (up to 20 points)
        if candle.is_rejection:
            score += 15
        if candle.body_vs_range < 0.3:  # Indecision candle
            score += 5
        
        return min(100, score)
    
    def _record_resolution(self, streak: ExtensionStreak, candle: CandleInExtension):
        """Record how a streak resolved for pattern learning"""
        if streak.streak_count < 2:
            return  # Don't record very short streaks
        
        resolution = {
            "level_name": streak.level_name,
            "direction": streak.direction,
            "candle_count": streak.streak_count,
            "duration_hours": streak.hours_extended,
            "avg_extension": streak.avg_extension,
            "max_extension": streak.max_extension,
            "predicted_prob": streak.snap_back_probability,
            "timestamp": candle.timestamp
        }
        
        self._resolution_history.append(resolution)
        
        # Keep only last 100 resolutions
        if len(self._resolution_history) > 100:
            self._resolution_history = self._resolution_history[-100:]
    
    def get_active_streaks(self) -> Dict[str, ExtensionStreak]:
        """Get all currently active extension streaks"""
        return self._streaks.copy()
    
    def get_actionable_setups(self) -> List[ExtensionStreak]:
        """Get streaks that have reached actionable threshold"""
        return [
            streak for streak in self._streaks.values()
            if streak.is_actionable
        ]
    
    def get_streak_summary(self) -> str:
        """Get human-readable summary of active streaks"""
        if not self._streaks:
            return "No active extension streaks"
        
        lines = ["Active Extension Streaks:"]
        for key, streak in self._streaks.items():
            lines.append(
                f"  {streak.level_name} {streak.direction}: "
                f"{streak.streak_count} candles ({streak.hours_extended:.1f}h) "
                f"- {streak.trigger.name} "
                f"[{streak.snap_back_probability:.0%} snap-back]"
            )
        
        return "\n".join(lines)


class MultiTimeframeExtensionPredictor:
    """
    Tracks extensions across multiple timeframes simultaneously
    """
    
    def __init__(self):
        # Predictors for different candle sizes
        self.predictors = {
            "2h": ExtensionDurationPredictor(candle_minutes=120),
            "1h": ExtensionDurationPredictor(candle_minutes=60),
            "4h": ExtensionDurationPredictor(candle_minutes=240),
        }
        
        # Cross-timeframe alignment tracking
        self._alignment_score: float = 0.0
    
    def update(
        self,
        timeframe: str,
        candle: CandleInExtension
    ) -> List[ExtensionAlert]:
        """Update specific timeframe predictor"""
        if timeframe not in self.predictors:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return []
        
        return self.predictors[timeframe].update(candle)
    
    def get_hottest_timeframe(self) -> Tuple[str, Optional[ExtensionStreak]]:
        """Find timeframe with highest energy/most actionable setup"""
        best_tf = None
        best_streak = None
        best_score = 0
        
        for tf, predictor in self.predictors.items():
            for streak in predictor.get_active_streaks().values():
                if streak.is_actionable:
                    # Score based on duration + probability
                    score = streak.streak_count * streak.snap_back_probability * 100
                    if score > best_score:
                        best_score = score
                        best_tf = tf
                        best_streak = streak
        
        return (best_tf or "none", best_streak)
    
    def get_all_actionable(self) -> Dict[str, List[ExtensionStreak]]:
        """Get all actionable setups by timeframe"""
        result = {}
        for tf, predictor in self.predictors.items():
            actionable = predictor.get_actionable_setups()
            if actionable:
                result[tf] = actionable
        return result
    
    def get_cross_tf_alignment(self) -> Dict:
        """Check if multiple timeframes are extended in same direction"""
        above_count = 0
        below_count = 0
        
        for predictor in self.predictors.values():
            for streak in predictor.get_active_streaks().values():
                if streak.direction == "above":
                    above_count += 1
                else:
                    below_count += 1
        
        total = above_count + below_count
        if total == 0:
            return {"alignment": "none", "direction": "neutral", "score": 0}
        
        if above_count == total:
            return {"alignment": "full", "direction": "above", "score": 100}
        elif below_count == total:
            return {"alignment": "full", "direction": "below", "score": 100}
        else:
            dominant = "above" if above_count > below_count else "below"
            score = max(above_count, below_count) / total * 100
            return {"alignment": "partial", "direction": dominant, "score": score}


# ============ Integration Helper ============

def create_candle_from_data(
    timestamp: datetime,
    ohlcv: Dict,
    vwap: float,
    poc: float,
    vah: float,
    val: float,
    atr: float
) -> CandleInExtension:
    """
    Helper to create CandleInExtension from raw data
    
    Args:
        timestamp: Candle timestamp
        ohlcv: Dict with open, high, low, close, volume
        vwap: Current VWAP
        poc: Current POC
        vah: Value Area High
        val: Value Area Low
        atr: Current ATR
    """
    return CandleInExtension(
        timestamp=timestamp,
        open=ohlcv["open"],
        high=ohlcv["high"],
        low=ohlcv["low"],
        close=ohlcv["close"],
        volume=ohlcv.get("volume", 0),
        vwap=vwap,
        poc=poc,
        vah=vah,
        val=val,
        atr=atr
    )


# ============ Example Usage ============

def example_usage():
    """Example showing how to use the predictor"""
    
    # Initialize predictor for 2H candles
    predictor = ExtensionDurationPredictor(candle_minutes=120)
    
    # Simulate some candles
    # In real use, these would come from your data feed
    
    base_time = datetime(2024, 1, 15, 9, 30)
    vwap = 585.00
    poc = 584.50
    vah = 586.00
    val = 583.00
    atr = 2.50
    
    # Candle 1: Slightly above VWAP
    candle1 = CandleInExtension(
        timestamp=base_time,
        open=585.20, high=586.80, low=585.00, close=586.50,
        volume=1000000,
        vwap=vwap, poc=poc, vah=vah, val=val, atr=atr
    )
    alerts = predictor.update(candle1)
    print(f"After candle 1: {predictor.get_streak_summary()}")
    
    # Candle 2: Continues above - should trigger ALERT
    candle2 = CandleInExtension(
        timestamp=base_time + timedelta(hours=2),
        open=586.50, high=587.50, low=586.00, close=587.20,
        volume=900000,
        vwap=vwap, poc=poc, vah=vah, val=val, atr=atr
    )
    alerts = predictor.update(candle2)
    print(f"After candle 2: {predictor.get_streak_summary()}")
    for alert in alerts:
        print(f"\n{alert}")
    
    # Candle 3: Still extended - HIGH_PROB
    candle3 = CandleInExtension(
        timestamp=base_time + timedelta(hours=4),
        open=587.20, high=588.00, low=586.50, close=586.80,  # Rejection candle
        volume=800000,
        vwap=vwap, poc=poc, vah=vah, val=val, atr=atr
    )
    alerts = predictor.update(candle3)
    print(f"After candle 3: {predictor.get_streak_summary()}")
    for alert in alerts:
        print(f"\n{alert}")


if __name__ == "__main__":
    example_usage()
