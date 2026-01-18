"""
S.E.F. Extension Duration Predictor
====================================
THE KEY EDGE: It's not just WHERE price is, but HOW LONG it's been there.

Logic:
- 1 candle (2h) extended = 45% snap-back → WATCHING
- 2 candles (4h) extended = 55% snap-back → ALERT
- 3 candles (6h) extended = 65% snap-back → HIGH_PROB 🔥
- 4+ candles (8h+) extended = 75%+ snap-back → EXTREME 💥

Think of it like a rubber band - the longer it's stretched, the harder it snaps back.

Author: Rob's Trading Systems
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np


class TriggerLevel(Enum):
    """Alert levels based on extension duration"""
    NONE = 0
    WATCHING = 1      # 1 candle - just observing
    ALERT = 2         # 2 candles (4 hours) - prepare
    HIGH_PROB = 3     # 3 candles (6 hours) - look for entry
    EXTREME = 4       # 4+ candles (8+ hours) - high conviction


class ExtensionZone(Enum):
    """Where price is relative to fair value"""
    EXTREME_ABOVE = "extreme_above"    # > 2 ATR above VWAP
    ABOVE_VALUE = "above_value"        # Above VAH or > 1 ATR above VWAP
    IN_VALUE = "in_value"              # Between VAL and VAH
    BELOW_VALUE = "below_value"        # Below VAL or > 1 ATR below VWAP
    EXTREME_BELOW = "extreme_below"    # > 2 ATR below VWAP


@dataclass
class CandleData:
    """Single candle with extension metrics"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Reference levels
    vwap: float = 0.0
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    atr: float = 0.0
    
    # Calculated
    zone: ExtensionZone = ExtensionZone.IN_VALUE
    distance_from_vwap_atr: float = 0.0
    is_rejection: bool = False
    is_continuation: bool = False
    
    def analyze(self):
        """Analyze candle characteristics"""
        mid = (self.high + self.low + self.close) / 3
        
        # Distance from VWAP in ATR units
        if self.atr > 0:
            self.distance_from_vwap_atr = (mid - self.vwap) / self.atr
        
        # Determine zone
        self.zone = self._get_zone(mid)
        
        # Candle structure
        self._analyze_structure()
    
    def _get_zone(self, price: float) -> ExtensionZone:
        """Determine which zone price is in"""
        atr_dist = abs(self.distance_from_vwap_atr)
        
        if price > self.vwap:
            if atr_dist > 2.0 or price > self.vah + self.atr:
                return ExtensionZone.EXTREME_ABOVE
            elif price > self.vah or atr_dist > 1.0:
                return ExtensionZone.ABOVE_VALUE
        else:
            if atr_dist > 2.0 or price < self.val - self.atr:
                return ExtensionZone.EXTREME_BELOW
            elif price < self.val or atr_dist > 1.0:
                return ExtensionZone.BELOW_VALUE
        
        return ExtensionZone.IN_VALUE
    
    def _analyze_structure(self):
        """Analyze if candle shows rejection or continuation"""
        range_ = self.high - self.low
        if range_ == 0:
            return
        
        upper_wick = self.high - max(self.open, self.close)
        lower_wick = min(self.open, self.close) - self.low
        
        # Rejection = long wick pointing toward value
        if self.zone in [ExtensionZone.ABOVE_VALUE, ExtensionZone.EXTREME_ABOVE]:
            self.is_rejection = (upper_wick / range_) > 0.5
            self.is_continuation = self.close > self.open and (lower_wick / range_) < 0.2
        elif self.zone in [ExtensionZone.BELOW_VALUE, ExtensionZone.EXTREME_BELOW]:
            self.is_rejection = (lower_wick / range_) > 0.5
            self.is_continuation = self.close < self.open and (upper_wick / range_) < 0.2


@dataclass
class ExtensionStreak:
    """Tracks consecutive candles in extension"""
    level_name: str           # "vwap", "vah", "val", "poc"
    direction: str            # "above" or "below"
    candles: List[CandleData] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.candles)
    
    @property
    def hours(self) -> float:
        return self.count * 2  # Assuming 2-hour candles
    
    @property
    def trigger(self) -> TriggerLevel:
        if self.count >= 4:
            return TriggerLevel.EXTREME
        elif self.count == 3:
            return TriggerLevel.HIGH_PROB
        elif self.count == 2:
            return TriggerLevel.ALERT
        elif self.count == 1:
            return TriggerLevel.WATCHING
        return TriggerLevel.NONE
    
    @property
    def avg_extension_atr(self) -> float:
        if not self.candles:
            return 0
        return sum(abs(c.distance_from_vwap_atr) for c in self.candles) / len(self.candles)
    
    @property
    def has_rejection(self) -> bool:
        return any(c.is_rejection for c in self.candles[-2:]) if self.candles else False
    
    @property
    def declining_volume(self) -> bool:
        if len(self.candles) < 2:
            return False
        vols = [c.volume for c in self.candles]
        return vols[-1] < vols[0] * 0.8  # Volume declined 20%+


@dataclass
class ExtensionAlert:
    """Alert when extension reaches actionable threshold"""
    symbol: str
    timestamp: datetime
    
    level_name: str
    direction: str
    trigger: TriggerLevel
    
    candle_count: int
    hours_extended: float
    extension_atr: float
    
    current_price: float
    snap_back_target: float
    stop_loss: float
    
    snap_back_probability: float
    quality_score: float
    risk_reward: float
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'level_name': self.level_name,
            'direction': self.direction,
            'trigger': self.trigger.name,
            'trigger_emoji': self._trigger_emoji(),
            'candle_count': self.candle_count,
            'hours_extended': self.hours_extended,
            'extension_atr': round(self.extension_atr, 2),
            'current_price': round(self.current_price, 2),
            'snap_back_target': round(self.snap_back_target, 2),
            'stop_loss': round(self.stop_loss, 2),
            'snap_back_probability': round(self.snap_back_probability * 100, 1),
            'quality_score': round(self.quality_score, 1),
            'risk_reward': round(self.risk_reward, 2),
            'trade_direction': 'SHORT' if self.direction == 'above' else 'LONG'
        }
    
    def _trigger_emoji(self) -> str:
        return {
            TriggerLevel.NONE: "⚪",
            TriggerLevel.WATCHING: "👀",
            TriggerLevel.ALERT: "⚠️",
            TriggerLevel.HIGH_PROB: "🔥",
            TriggerLevel.EXTREME: "💥"
        }.get(self.trigger, "")


class ExtensionPredictor:
    """
    Main predictor class - THE EDGE
    
    Tracks how long price has been extended from key levels
    and predicts snap-back probability.
    """
    
    # Snap-back probabilities by candle count
    BASE_PROBABILITIES = {
        0: 0.40,
        1: 0.45,
        2: 0.55,
        3: 0.65,
        4: 0.75,
        5: 0.80,
        6: 0.85,
    }
    
    # Probability adjustments
    REJECTION_BONUS = 0.10
    DECLINING_VOLUME_BONUS = 0.05
    EXTREME_EXTENSION_BONUS = 0.05
    
    def __init__(self, candle_minutes: int = 120):
        self.candle_minutes = candle_minutes
        
        # Active streaks per symbol per level
        # Key: symbol -> {level_direction: ExtensionStreak}
        self._streaks: Dict[str, Dict[str, ExtensionStreak]] = {}
        
        # Historical data per symbol
        self._history: Dict[str, List[CandleData]] = {}
    
    def update(self, 
               symbol: str,
               candle: CandleData) -> List[ExtensionAlert]:
        """
        Update with new candle and return any alerts
        
        Args:
            symbol: Stock symbol
            candle: New 2H candle with VP/VWAP levels
            
        Returns:
            List of ExtensionAlert if actionable thresholds crossed
        """
        candle.analyze()
        
        # Initialize if needed
        if symbol not in self._streaks:
            self._streaks[symbol] = {}
        if symbol not in self._history:
            self._history[symbol] = []
        
        self._history[symbol].append(candle)
        
        # Keep last 50 candles
        if len(self._history[symbol]) > 50:
            self._history[symbol] = self._history[symbol][-50:]
        
        alerts = []
        
        # Track extension from each level
        levels = [
            ("vwap", candle.vwap),
            ("poc", candle.poc),
            ("vah", candle.vah),
            ("val", candle.val),
        ]
        
        price = candle.close
        
        for level_name, level_price in levels:
            if level_price <= 0:
                continue
            
            # Determine direction
            if price > level_price + (candle.atr * 0.5):
                direction = "above"
            elif price < level_price - (candle.atr * 0.5):
                direction = "below"
            else:
                direction = None
            
            key = f"{level_name}_{direction}" if direction else None
            
            # Update or reset streak
            if direction:
                if key not in self._streaks[symbol]:
                    self._streaks[symbol][key] = ExtensionStreak(
                        level_name=level_name,
                        direction=direction
                    )
                
                streak = self._streaks[symbol][key]
                streak.candles.append(candle)
                
                # Check for alert
                if streak.trigger.value >= TriggerLevel.ALERT.value:
                    alert = self._create_alert(symbol, streak, candle, level_price)
                    alerts.append(alert)
            else:
                # Price returned to value - clear streaks for this level
                for d in ["above", "below"]:
                    clear_key = f"{level_name}_{d}"
                    if clear_key in self._streaks[symbol]:
                        del self._streaks[symbol][clear_key]
        
        return alerts
    
    def _create_alert(self, 
                      symbol: str, 
                      streak: ExtensionStreak,
                      candle: CandleData,
                      level_price: float) -> ExtensionAlert:
        """Create an extension alert"""
        
        # Calculate snap-back probability
        base_prob = self.BASE_PROBABILITIES.get(
            min(streak.count, 6), 
            0.85
        )
        
        # Apply modifiers
        prob = base_prob
        if streak.has_rejection:
            prob += self.REJECTION_BONUS
        if streak.declining_volume:
            prob += self.DECLINING_VOLUME_BONUS
        if streak.avg_extension_atr > 2.0:
            prob += self.EXTREME_EXTENSION_BONUS
        
        prob = min(0.95, prob)
        
        # Calculate targets
        if streak.direction == "above":
            # SHORT setup - target snap back to level
            snap_back_target = level_price
            stop_loss = candle.high + candle.atr * 0.5
            risk = stop_loss - candle.close
            reward = candle.close - snap_back_target
        else:
            # LONG setup - target snap back to level
            snap_back_target = level_price
            stop_loss = candle.low - candle.atr * 0.5
            risk = candle.close - stop_loss
            reward = snap_back_target - candle.close
        
        risk_reward = reward / risk if risk > 0 else 0
        
        # Quality score (0-100)
        quality = 0
        quality += min(40, streak.count * 10)  # Up to 40 for duration
        quality += prob * 30  # Up to 30 for probability
        quality += min(20, risk_reward * 10)  # Up to 20 for R:R
        if streak.has_rejection:
            quality += 10
        
        return ExtensionAlert(
            symbol=symbol,
            timestamp=candle.timestamp,
            level_name=streak.level_name.upper(),
            direction=streak.direction,
            trigger=streak.trigger,
            candle_count=streak.count,
            hours_extended=streak.hours,
            extension_atr=streak.avg_extension_atr,
            current_price=candle.close,
            snap_back_target=snap_back_target,
            stop_loss=stop_loss,
            snap_back_probability=prob,
            quality_score=quality,
            risk_reward=risk_reward
        )
    
    def get_active_streaks(self, symbol: str) -> Dict[str, dict]:
        """Get all active extension streaks for a symbol"""
        if symbol not in self._streaks:
            return {}
        
        result = {}
        for key, streak in self._streaks[symbol].items():
            result[key] = {
                'level': streak.level_name,
                'direction': streak.direction,
                'candles': streak.count,
                'hours': streak.hours,
                'trigger': streak.trigger.name,
                'trigger_emoji': {
                    TriggerLevel.NONE: "⚪",
                    TriggerLevel.WATCHING: "👀",
                    TriggerLevel.ALERT: "⚠️",
                    TriggerLevel.HIGH_PROB: "🔥",
                    TriggerLevel.EXTREME: "💥"
                }.get(streak.trigger, ""),
                'avg_extension_atr': round(streak.avg_extension_atr, 2),
                'has_rejection': streak.has_rejection,
                'snap_back_prob': round(
                    self.BASE_PROBABILITIES.get(min(streak.count, 6), 0.85) * 100, 1
                )
            }
        
        return result
    
    def get_hottest_setup(self, symbol: str) -> Optional[dict]:
        """Get the highest probability setup for a symbol"""
        streaks = self.get_active_streaks(symbol)
        if not streaks:
            return None
        
        # Find highest candle count
        hottest = max(streaks.values(), key=lambda s: s['candles'])
        if hottest['candles'] >= 2:
            return hottest
        return None
    
    def get_all_alerts(self) -> List[dict]:
        """Get actionable alerts across all symbols"""
        alerts = []
        
        for symbol in self._streaks:
            for key, streak in self._streaks[symbol].items():
                if streak.trigger.value >= TriggerLevel.ALERT.value:
                    hottest = self.get_active_streaks(symbol).get(key)
                    if hottest:
                        hottest['symbol'] = symbol
                        alerts.append(hottest)
        
        # Sort by candle count (most extended first)
        alerts.sort(key=lambda a: a['candles'], reverse=True)
        return alerts
    
    def analyze_from_dataframe(self,
                               symbol: str,
                               df: pd.DataFrame,
                               vwap: float,
                               poc: float,
                               vah: float,
                               val: float) -> List[ExtensionAlert]:
        """
        Analyze a DataFrame of 2H candles and return alerts
        
        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV columns
            vwap, poc, vah, val: Current reference levels
        """
        if df.empty:
            return []
        
        # Calculate ATR
        df = df.copy()
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = df['tr'].rolling(14).mean().iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            atr = (df['high'] - df['low']).mean()
        
        # Clear existing streaks for this symbol
        self._streaks[symbol] = {}
        self._history[symbol] = []
        
        all_alerts = []
        
        # Process last 10 candles
        for idx in range(-10, 0):
            if abs(idx) > len(df):
                continue
            
            row = df.iloc[idx]
            
            candle = CandleData(
                timestamp=row.name if hasattr(row.name, 'isoformat') else datetime.now(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
                vwap=vwap,
                poc=poc,
                vah=vah,
                val=val,
                atr=atr
            )
            
            alerts = self.update(symbol, candle)
            all_alerts.extend(alerts)
        
        # Return only the latest alerts (not duplicates)
        return all_alerts[-5:] if all_alerts else []


# =============================================================================
# INTEGRATION WITH SCANNER
# =============================================================================

def add_extension_to_analysis(analysis_result: dict, 
                               extension_data: dict) -> dict:
    """
    Add extension data to an analysis result
    
    Args:
        analysis_result: Existing analysis dict from scanner
        extension_data: Extension streaks and alerts
        
    Returns:
        Enhanced analysis dict
    """
    analysis_result['extension'] = extension_data
    
    # Boost confidence if extension is HIGH_PROB or EXTREME
    for key, streak in extension_data.get('streaks', {}).items():
        if streak.get('trigger') in ['HIGH_PROB', 'EXTREME']:
            # Add extension bonus to signal
            if 'extension_bonus' not in analysis_result:
                analysis_result['extension_bonus'] = 0
            
            if streak.get('candles', 0) >= 3:
                analysis_result['extension_bonus'] += 15
            if streak.get('candles', 0) >= 4:
                analysis_result['extension_bonus'] += 10
            if streak.get('has_rejection'):
                analysis_result['extension_bonus'] += 5
    
    return analysis_result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test the predictor
    predictor = ExtensionPredictor()
    
    # Simulate candles
    import random
    
    base_price = 600
    for i in range(6):
        # Simulate price extending above VWAP
        price = base_price + (i * 5) + random.uniform(-2, 2)
        
        candle = CandleData(
            timestamp=datetime.now() - timedelta(hours=(6-i)*2),
            open=price - 1,
            high=price + 2,
            low=price - 3,
            close=price,
            volume=1000000 - (i * 100000),
            vwap=base_price,
            poc=base_price + 2,
            vah=base_price + 10,
            val=base_price - 10,
            atr=5.0
        )
        
        alerts = predictor.update("TEST", candle)
        
        if alerts:
            for alert in alerts:
                print(f"\n{'='*50}")
                print(f"ALERT: {alert.trigger.name}")
                print(f"  Candles: {alert.candle_count}")
                print(f"  Probability: {alert.snap_back_probability:.0%}")
                print(f"  Quality: {alert.quality_score:.0f}")
    
    print("\n\nActive Streaks:")
    for key, streak in predictor.get_active_streaks("TEST").items():
        print(f"  {key}: {streak}")
