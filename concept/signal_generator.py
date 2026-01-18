"""
SEF Trading System - Signal Generator
Hybrid Mean Reversion + Trend Following Signal Engine
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from enum import Enum
import logging

from config import SignalConfig, RiskConfig
from volume_profile import VolumeProfileResult, MultiTimeframeVP
from vwap_engine import VWAPResult, MultiTimeframeVWAP
from data_client import OHLCV

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal classification"""
    NONE = "none"
    LONG_MR = "long_mean_reversion"
    SHORT_MR = "short_mean_reversion"
    LONG_TREND = "long_trend"
    SHORT_TREND = "short_trend"


class SignalStrength(Enum):
    """Signal conviction level"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class SignalContext:
    """Contextual data for signal generation"""
    # Current price data
    current_price: float
    current_candle: Optional[OHLCV] = None
    
    # Volume Profile context
    vp: Optional[MultiTimeframeVP] = None
    
    # VWAP context
    vwap: Optional[MultiTimeframeVWAP] = None
    
    # Volatility
    atr: float = 0.0
    atr_percent: float = 0.0  # ATR as % of price
    
    # Bias
    poc_bias: str = "mixed"  # bullish, bearish, mixed
    vwap_bias: str = "mixed"
    
    # Session info
    time_of_day: str = "mid"  # open, mid, close
    minutes_into_session: int = 0


@dataclass
class Signal:
    """Generated trading signal"""
    signal_type: SignalType
    strength: SignalStrength
    timestamp: datetime
    
    # Entry
    entry_price: float
    suggested_entry: str = "market"  # market, limit, stop
    
    # Targets
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    target_3: float = 0.0
    
    # Context
    trigger_level: str = ""  # What level triggered the signal
    trigger_reason: str = ""  # Why this signal
    bias: str = "neutral"
    
    # Scoring
    score: float = 0.0  # 0-100 signal quality score
    factors: Dict[str, float] = field(default_factory=dict)
    
    # Risk
    risk_reward_1: float = 0.0
    risk_reward_2: float = 0.0
    position_size_suggestion: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.signal_type in [SignalType.LONG_MR, SignalType.LONG_TREND]
    
    @property
    def is_short(self) -> bool:
        return self.signal_type in [SignalType.SHORT_MR, SignalType.SHORT_TREND]
    
    @property
    def is_mean_reversion(self) -> bool:
        return self.signal_type in [SignalType.LONG_MR, SignalType.SHORT_MR]


class SignalGenerator:
    """
    Core signal generation engine
    Combines Volume Profile and VWAP for hybrid signals
    """
    
    def __init__(self, signal_config: SignalConfig, risk_config: RiskConfig):
        self.config = signal_config
        self.risk = risk_config
        
    def generate(self, context: SignalContext) -> Optional[Signal]:
        """
        Generate trading signal based on context
        
        Args:
            context: SignalContext with all market data
            
        Returns:
            Signal if conditions met, None otherwise
        """
        if context.vp is None or context.vwap is None:
            logger.warning("Incomplete context - VP or VWAP missing")
            return None
        
        # Update bias from context
        context.poc_bias = context.vp.get_poc_alignment()
        context.vwap_bias = context.vwap.get_alignment(context.current_price)
        
        # Check for mean reversion signals first
        mr_signal = None
        if self.config.mr_enabled:
            mr_signal = self._check_mean_reversion(context)
        
        # Check for trend signals
        trend_signal = None
        if self.config.trend_enabled:
            trend_signal = self._check_trend(context)
        
        # Select best signal based on hybrid rules
        signal = self._select_signal(mr_signal, trend_signal, context)
        
        if signal:
            # Add risk management
            signal = self._add_risk_management(signal, context)
            # Score the signal
            signal = self._score_signal(signal, context)
        
        return signal
    
    def _check_mean_reversion(self, context: SignalContext) -> Optional[Signal]:
        """
        Check for mean reversion setups
        
        Conditions:
        - Price extended beyond VAL/VAH
        - Rejection candle pattern
        - Target: POC of triggering timeframe
        """
        price = context.current_price
        candle = context.current_candle
        vp = context.vp
        atr = context.atr
        
        # Check each timeframe for extension
        for tf_name in ["daily", "weekly"]:
            tf_vp = getattr(vp, tf_name)
            if tf_vp is None:
                continue
            
            # Calculate extension
            extension_from_val = (tf_vp.val - price) / atr if atr > 0 else 0
            extension_from_vah = (price - tf_vp.vah) / atr if atr > 0 else 0
            
            # LONG: Price below VAL with rejection
            if extension_from_val >= self.config.mr_min_extension_atr:
                if candle and self._is_rejection_candle(candle, "bullish"):
                    return Signal(
                        signal_type=SignalType.LONG_MR,
                        strength=self._get_mr_strength(extension_from_val),
                        timestamp=datetime.now(),
                        entry_price=price,
                        trigger_level=f"{tf_name}_val",
                        trigger_reason=f"Price {extension_from_val:.1f} ATR below {tf_name} VAL with bullish rejection",
                        bias=context.poc_bias
                    )
            
            # SHORT: Price above VAH with rejection
            if extension_from_vah >= self.config.mr_min_extension_atr:
                if candle and self._is_rejection_candle(candle, "bearish"):
                    return Signal(
                        signal_type=SignalType.SHORT_MR,
                        strength=self._get_mr_strength(extension_from_vah),
                        timestamp=datetime.now(),
                        entry_price=price,
                        trigger_level=f"{tf_name}_vah",
                        trigger_reason=f"Price {extension_from_vah:.1f} ATR above {tf_name} VAH with bearish rejection",
                        bias=context.poc_bias
                    )
        
        return None
    
    def _check_trend(self, context: SignalContext) -> Optional[Signal]:
        """
        Check for trend continuation setups
        
        Conditions:
        - Price holding above VAH (bullish) or below VAL (bearish)
        - VWAP slope confirms direction
        - POC alignment if required
        """
        price = context.current_price
        vp = context.vp
        vwap = context.vwap
        
        if vp.daily is None:
            return None
        
        daily_vp = vp.daily
        daily_vwap = vwap.daily if vwap.daily else None
        weekly_vp = vp.weekly
        
        # Check VWAP slope
        vwap_slope = daily_vwap.slope if daily_vwap and daily_vwap.slope else 0
        
        # Check POC alignment if required
        if self.config.trend_poc_alignment_required:
            if context.poc_bias == "mixed":
                return None
        
        # LONG TREND: Above daily VAH, VWAP rising, above weekly POC
        if price > daily_vp.vah:
            if vwap_slope >= self.config.trend_min_slope:
                if weekly_vp is None or price > weekly_vp.poc:
                    if context.poc_bias in ["bullish", "mixed"]:
                        return Signal(
                            signal_type=SignalType.LONG_TREND,
                            strength=self._get_trend_strength(context),
                            timestamp=datetime.now(),
                            entry_price=price,
                            trigger_level="daily_vah_breakout",
                            trigger_reason=f"Price above daily VAH, VWAP slope {vwap_slope:.4f}",
                            bias=context.poc_bias
                        )
        
        # SHORT TREND: Below daily VAL, VWAP falling, below weekly POC
        if price < daily_vp.val:
            if vwap_slope <= -self.config.trend_min_slope:
                if weekly_vp is None or price < weekly_vp.poc:
                    if context.poc_bias in ["bearish", "mixed"]:
                        return Signal(
                            signal_type=SignalType.SHORT_TREND,
                            strength=self._get_trend_strength(context),
                            timestamp=datetime.now(),
                            entry_price=price,
                            trigger_level="daily_val_breakdown",
                            trigger_reason=f"Price below daily VAL, VWAP slope {vwap_slope:.4f}",
                            bias=context.poc_bias
                        )
        
        return None
    
    def _is_rejection_candle(self, candle: OHLCV, direction: str) -> bool:
        """
        Check if candle shows rejection pattern
        
        Args:
            candle: OHLCV bar to check
            direction: "bullish" (rejection of lows) or "bearish" (rejection of highs)
        """
        if candle.range == 0:
            return False
        
        wick_ratio = self.config.mr_rejection_wick_ratio
        
        if direction == "bullish":
            # Lower wick should be significant portion of range
            lower_wick_ratio = candle.lower_wick / candle.range
            return lower_wick_ratio >= wick_ratio
        
        else:  # bearish
            # Upper wick should be significant portion of range
            upper_wick_ratio = candle.upper_wick / candle.range
            return upper_wick_ratio >= wick_ratio
    
    def _get_mr_strength(self, extension: float) -> SignalStrength:
        """Determine mean reversion signal strength based on extension"""
        if extension >= 3.0:
            return SignalStrength.VERY_STRONG
        elif extension >= 2.5:
            return SignalStrength.STRONG
        elif extension >= 2.0:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _get_trend_strength(self, context: SignalContext) -> SignalStrength:
        """Determine trend signal strength based on alignment"""
        score = 0
        
        # POC alignment
        if context.poc_bias in ["bullish", "bearish"]:
            score += 2
        
        # VWAP alignment
        if context.vwap_bias == context.poc_bias:
            score += 1
        
        # Volume confirmation (would need additional data)
        
        if score >= 3:
            return SignalStrength.STRONG
        elif score >= 2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _select_signal(
        self,
        mr_signal: Optional[Signal],
        trend_signal: Optional[Signal],
        context: SignalContext
    ) -> Optional[Signal]:
        """
        Select best signal using hybrid rules
        
        Rules:
        - If bias is mixed and mr_only configured, only take MR signals
        - Prefer stronger signals
        - Prefer signals aligned with bias
        """
        signals = [s for s in [mr_signal, trend_signal] if s is not None]
        
        if not signals:
            return None
        
        if len(signals) == 1:
            signal = signals[0]
            
            # Check mixed bias rule
            if self.config.mixed_bias_mr_only and context.poc_bias == "mixed":
                if not signal.is_mean_reversion:
                    return None
            
            return signal
        
        # Both signals present - select best
        mr = mr_signal
        trend = trend_signal
        
        # If mixed bias, prefer MR
        if self.config.mixed_bias_mr_only and context.poc_bias == "mixed":
            return mr
        
        # If bias confirms trend direction, prefer trend
        if context.poc_bias == "bullish" and trend.is_long:
            return trend
        if context.poc_bias == "bearish" and trend.is_short:
            return trend
        
        # Otherwise prefer stronger signal
        if mr.strength.value > trend.strength.value:
            return mr
        
        return trend
    
    def _add_risk_management(self, signal: Signal, context: SignalContext) -> Signal:
        """Add stop loss and targets to signal"""
        atr = context.atr
        vp = context.vp
        
        if signal.is_mean_reversion:
            # MR: Stop beyond the extreme, target to POC
            if signal.is_long:
                # Stop below recent low
                signal.stop_loss = signal.entry_price - (atr * self.risk.atr_stop_multiplier)
                
                # Targets: POC, then VAL, then VWAP
                if vp.daily:
                    signal.target_1 = vp.daily.poc
                    signal.target_2 = vp.daily.vah
                if vp.weekly:
                    signal.target_3 = vp.weekly.poc
                    
            else:  # Short MR
                signal.stop_loss = signal.entry_price + (atr * self.risk.atr_stop_multiplier)
                
                if vp.daily:
                    signal.target_1 = vp.daily.poc
                    signal.target_2 = vp.daily.val
                if vp.weekly:
                    signal.target_3 = vp.weekly.poc
        
        else:  # Trend
            if signal.is_long:
                # Stop below structure (VAH becomes support)
                if vp.daily:
                    signal.stop_loss = min(
                        vp.daily.vah - (atr * 0.5),
                        signal.entry_price - (atr * self.risk.atr_stop_multiplier)
                    )
                else:
                    signal.stop_loss = signal.entry_price - (atr * self.risk.atr_stop_multiplier)
                
                # Targets: Extensions
                signal.target_1 = signal.entry_price + (atr * 2)
                signal.target_2 = signal.entry_price + (atr * 3)
                if vp.weekly:
                    signal.target_3 = vp.weekly.vah
                    
            else:  # Short trend
                if vp.daily:
                    signal.stop_loss = max(
                        vp.daily.val + (atr * 0.5),
                        signal.entry_price + (atr * self.risk.atr_stop_multiplier)
                    )
                else:
                    signal.stop_loss = signal.entry_price + (atr * self.risk.atr_stop_multiplier)
                
                signal.target_1 = signal.entry_price - (atr * 2)
                signal.target_2 = signal.entry_price - (atr * 3)
                if vp.weekly:
                    signal.target_3 = vp.weekly.val
        
        # Calculate risk/reward
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk > 0:
            signal.risk_reward_1 = abs(signal.target_1 - signal.entry_price) / risk if signal.target_1 else 0
            signal.risk_reward_2 = abs(signal.target_2 - signal.entry_price) / risk if signal.target_2 else 0
        
        return signal
    
    def _score_signal(self, signal: Signal, context: SignalContext) -> Signal:
        """
        Score signal quality from 0-100
        
        Factors:
        - Strength (25 points)
        - Bias alignment (20 points)
        - Risk/Reward (25 points)
        - Time of day (10 points)
        - Multi-TF confluence (20 points)
        """
        score = 0.0
        factors = {}
        
        # Strength score (0-25)
        strength_scores = {
            SignalStrength.WEAK: 10,
            SignalStrength.MODERATE: 15,
            SignalStrength.STRONG: 20,
            SignalStrength.VERY_STRONG: 25
        }
        strength_score = strength_scores.get(signal.strength, 10)
        factors["strength"] = strength_score
        score += strength_score
        
        # Bias alignment (0-20)
        bias_score = 0
        if signal.is_long and context.poc_bias == "bullish":
            bias_score = 20
        elif signal.is_short and context.poc_bias == "bearish":
            bias_score = 20
        elif context.poc_bias == "mixed":
            bias_score = 10 if signal.is_mean_reversion else 5
        else:
            bias_score = 5  # Against bias
        factors["bias_alignment"] = bias_score
        score += bias_score
        
        # Risk/Reward (0-25)
        rr = signal.risk_reward_1
        if rr >= 3:
            rr_score = 25
        elif rr >= 2:
            rr_score = 20
        elif rr >= 1.5:
            rr_score = 15
        elif rr >= 1:
            rr_score = 10
        else:
            rr_score = 5
        factors["risk_reward"] = rr_score
        score += rr_score
        
        # Time of day (0-10)
        # Generally better signals in first 2 hours and last hour
        tod_score = 7  # Default mid-day
        if context.time_of_day == "open":
            tod_score = 10
        elif context.time_of_day == "close":
            tod_score = 8
        factors["time_of_day"] = tod_score
        score += tod_score
        
        # Multi-TF confluence (0-20)
        confluence_score = 0
        vp = context.vp
        
        # Check if multiple timeframes confirm
        if vp:
            confirmations = 0
            price = context.current_price
            
            for tf_name in ["daily", "weekly", "monthly"]:
                tf_vp = getattr(vp, tf_name)
                if tf_vp:
                    if signal.is_long and price < tf_vp.poc:
                        confirmations += 1  # Below POC for long
                    elif signal.is_short and price > tf_vp.poc:
                        confirmations += 1  # Above POC for short
            
            confluence_score = min(confirmations * 7, 20)
        
        factors["confluence"] = confluence_score
        score += confluence_score
        
        signal.score = score
        signal.factors = factors
        
        return signal


class SignalFilter:
    """
    Pre and post filters for signal validation
    """
    
    @staticmethod
    def time_filter(signal: Signal, context: SignalContext) -> bool:
        """Filter signals based on time of day"""
        # Avoid first 15 minutes (volatility)
        if context.minutes_into_session < 15:
            return False
        
        # Avoid last 15 minutes for new entries
        if context.minutes_into_session > 375:  # 6.25 hours into session
            return False
        
        return True
    
    @staticmethod
    def quality_filter(signal: Signal, min_score: float = 50) -> bool:
        """Filter based on signal quality score"""
        return signal.score >= min_score
    
    @staticmethod
    def risk_reward_filter(signal: Signal, min_rr: float = 1.5) -> bool:
        """Filter based on minimum risk/reward"""
        return signal.risk_reward_1 >= min_rr
