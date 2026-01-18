"""
SEF Trading System - VWAP Engine
Multi-timeframe VWAP calculation with bands
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from enum import Enum
import logging

from config import VWAPConfig

logger = logging.getLogger(__name__)


@dataclass
class VWAPResult:
    """VWAP calculation result for a single anchor"""
    anchor: str  # monthly, weekly, daily, rolling
    timestamp: datetime  # Anchor start time
    vwap: float  # Current VWAP value
    
    # Optional bands (std dev based)
    upper_band_1: Optional[float] = None
    lower_band_1: Optional[float] = None
    upper_band_2: Optional[float] = None
    lower_band_2: Optional[float] = None
    
    # Metrics
    cumulative_volume: int = 0
    cumulative_pv: float = 0.0  # price * volume
    
    # Slope for trend detection
    slope: Optional[float] = None
    slope_periods: int = 5
    
    @property
    def band_1_width(self) -> Optional[float]:
        if self.upper_band_1 and self.lower_band_1:
            return self.upper_band_1 - self.lower_band_1
        return None
    
    def price_vs_vwap(self, price: float) -> float:
        """Return price position relative to VWAP (positive = above)"""
        return price - self.vwap
    
    def price_in_bands(self, price: float) -> str:
        """Determine which band zone price is in"""
        if self.upper_band_2 and price > self.upper_band_2:
            return "above_2sd"
        elif self.upper_band_1 and price > self.upper_band_1:
            return "above_1sd"
        elif self.lower_band_1 and price < self.lower_band_1:
            return "below_1sd"
        elif self.lower_band_2 and price < self.lower_band_2:
            return "below_2sd"
        else:
            return "within_value"


@dataclass
class MultiTimeframeVWAP:
    """Container for multiple timeframe VWAPs"""
    monthly: Optional[VWAPResult] = None
    weekly: Optional[VWAPResult] = None
    daily: Optional[VWAPResult] = None
    rolling: Optional[VWAPResult] = None
    
    def get_alignment(self, price: float) -> str:
        """
        Determine price alignment relative to VWAPs
        Returns: 'bullish', 'bearish', or 'mixed'
        """
        above_count = 0
        below_count = 0
        total = 0
        
        for vwap in [self.monthly, self.weekly, self.daily]:
            if vwap is not None:
                total += 1
                if price > vwap.vwap:
                    above_count += 1
                else:
                    below_count += 1
        
        if total == 0:
            return "mixed"
        
        if above_count == total:
            return "bullish"
        elif below_count == total:
            return "bearish"
        return "mixed"
    
    def get_vwap_stack(self) -> List[tuple]:
        """Return VWAPs sorted by value (lowest to highest)"""
        vwaps = []
        for name in ["monthly", "weekly", "daily", "rolling"]:
            v = getattr(self, name)
            if v is not None:
                vwaps.append((name, v.vwap))
        
        return sorted(vwaps, key=lambda x: x[1])


class VWAPEngine:
    """
    Core engine for VWAP calculation
    Supports multiple anchor points and standard deviation bands
    """
    
    def __init__(self, config: VWAPConfig):
        self.config = config
        self.include_bands = config.include_bands
        self.band_multipliers = config.band_multipliers
        self.rolling_periods = config.rolling_periods
    
    def calculate_vwap(
        self,
        df: pd.DataFrame,
        anchor_start: Optional[datetime] = None
    ) -> VWAPResult:
        """
        Calculate VWAP from anchor point
        
        Args:
            df: DataFrame with OHLCV data
            anchor_start: Optional start time for VWAP calculation
            
        Returns:
            VWAPResult with VWAP and optional bands
        """
        if df.empty:
            raise ValueError("Cannot calculate VWAP on empty DataFrame")
        
        # Filter to anchor period if specified
        if anchor_start:
            df = df[df.index >= anchor_start]
        
        if df.empty:
            raise ValueError("No data after anchor filter")
        
        # Calculate typical price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        
        # Cumulative price * volume
        cum_pv = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        
        # VWAP series
        vwap_series = cum_pv / cum_vol
        vwap_series = vwap_series.replace([np.inf, -np.inf], np.nan).ffill()
        
        current_vwap = vwap_series.iloc[-1]
        
        result = VWAPResult(
            anchor="calculated",
            timestamp=df.index[0],
            vwap=current_vwap,
            cumulative_volume=int(cum_vol.iloc[-1]),
            cumulative_pv=float(cum_pv.iloc[-1])
        )
        
        # Calculate bands if configured
        if self.include_bands and len(df) > 1:
            result = self._add_bands(result, df, vwap_series, typical_price)
        
        # Calculate slope
        result = self._add_slope(result, vwap_series)
        
        return result
    
    def _add_bands(
        self,
        result: VWAPResult,
        df: pd.DataFrame,
        vwap_series: pd.Series,
        typical_price: pd.Series
    ) -> VWAPResult:
        """Add standard deviation bands to VWAP result"""
        # Calculate squared deviations from VWAP
        squared_dev = (typical_price - vwap_series) ** 2
        
        # Cumulative sum of squared deviations weighted by volume
        cum_vol = df["volume"].cumsum()
        cum_sq_dev = (squared_dev * df["volume"]).cumsum()
        
        # Standard deviation
        variance = cum_sq_dev / cum_vol
        std_dev = np.sqrt(variance).iloc[-1]
        
        # Apply band multipliers
        if len(self.band_multipliers) >= 1:
            result.upper_band_1 = result.vwap + std_dev * self.band_multipliers[0]
            result.lower_band_1 = result.vwap - std_dev * self.band_multipliers[0]
        
        if len(self.band_multipliers) >= 2:
            result.upper_band_2 = result.vwap + std_dev * self.band_multipliers[1]
            result.lower_band_2 = result.vwap - std_dev * self.band_multipliers[1]
        
        return result
    
    def _add_slope(
        self,
        result: VWAPResult,
        vwap_series: pd.Series,
        periods: int = 5
    ) -> VWAPResult:
        """Calculate VWAP slope for trend detection"""
        if len(vwap_series) < periods:
            result.slope = 0.0
            return result
        
        # Use last N periods to calculate slope
        recent = vwap_series.tail(periods)
        
        # Simple linear regression slope
        x = np.arange(len(recent))
        y = recent.values
        
        # Normalized slope (as percentage of VWAP)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            result.slope = slope / result.vwap if result.vwap != 0 else 0.0
        else:
            result.slope = 0.0
        
        result.slope_periods = periods
        return result


class MultiTimeframeVWAPEngine:
    """
    Manages VWAP calculation across multiple anchor points
    """
    
    def __init__(self, config: VWAPConfig):
        self.config = config
        self.engine = VWAPEngine(config)
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        current_time: Optional[datetime] = None
    ) -> MultiTimeframeVWAP:
        """
        Calculate VWAPs for all configured anchors
        
        Args:
            df: Intraday DataFrame with OHLCV data
            current_time: Reference time for anchor calculations
            
        Returns:
            MultiTimeframeVWAP with all calculated VWAPs
        """
        current_time = current_time or datetime.now()
        result = MultiTimeframeVWAP()
        
        # Monthly VWAP
        if "monthly" in self.config.anchors:
            try:
                result.monthly = self._calculate_monthly(df, current_time)
            except Exception as e:
                logger.warning(f"Monthly VWAP failed: {e}")
        
        # Weekly VWAP
        if "weekly" in self.config.anchors:
            try:
                result.weekly = self._calculate_weekly(df, current_time)
            except Exception as e:
                logger.warning(f"Weekly VWAP failed: {e}")
        
        # Daily VWAP
        if "daily" in self.config.anchors:
            try:
                result.daily = self._calculate_daily(df, current_time)
            except Exception as e:
                logger.warning(f"Daily VWAP failed: {e}")
        
        # Rolling VWAP
        if "rolling" in self.config.anchors:
            try:
                result.rolling = self._calculate_rolling(df)
            except Exception as e:
                logger.warning(f"Rolling VWAP failed: {e}")
        
        return result
    
    def _calculate_monthly(
        self,
        df: pd.DataFrame,
        current_time: datetime
    ) -> Optional[VWAPResult]:
        """Calculate VWAP anchored to month start"""
        if df.empty:
            return None
        
        month_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_data = df[df.index >= month_start]
        
        if month_data.empty:
            return None
        
        result = self.engine.calculate_vwap(month_data, month_start)
        result.anchor = "monthly"
        return result
    
    def _calculate_weekly(
        self,
        df: pd.DataFrame,
        current_time: datetime
    ) -> Optional[VWAPResult]:
        """Calculate VWAP anchored to week start (Monday)"""
        if df.empty:
            return None
        
        days_since_monday = current_time.weekday()
        week_start = (current_time - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        week_data = df[df.index >= week_start]
        
        if week_data.empty:
            return None
        
        result = self.engine.calculate_vwap(week_data, week_start)
        result.anchor = "weekly"
        return result
    
    def _calculate_daily(
        self,
        df: pd.DataFrame,
        current_time: datetime
    ) -> Optional[VWAPResult]:
        """Calculate VWAP anchored to today's open"""
        if df.empty:
            return None
        
        day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        day_data = df[df.index >= day_start]
        
        if day_data.empty:
            # Use latest day's data
            latest_date = df.index[-1].date()
            day_data = df[df.index.date == latest_date]
        
        if day_data.empty:
            return None
        
        result = self.engine.calculate_vwap(day_data)
        result.anchor = "daily"
        return result
    
    def _calculate_rolling(self, df: pd.DataFrame) -> Optional[VWAPResult]:
        """Calculate rolling VWAP over configured period"""
        if df.empty:
            return None
        
        # Use last N bars
        rolling_data = df.tail(self.config.rolling_periods * 12)  # Approximate for intraday
        
        if rolling_data.empty:
            return None
        
        result = self.engine.calculate_vwap(rolling_data)
        result.anchor = "rolling"
        return result


class VWAPTracker:
    """
    Real-time VWAP tracker for live updates
    Maintains running calculations without full recalculation
    """
    
    def __init__(self):
        self._cumulative_pv: float = 0.0
        self._cumulative_vol: int = 0
        self._cumulative_sq_dev: float = 0.0
        self._current_vwap: float = 0.0
        self._anchor_time: Optional[datetime] = None
    
    def reset(self, anchor_time: datetime):
        """Reset tracker for new anchor period"""
        self._cumulative_pv = 0.0
        self._cumulative_vol = 0
        self._cumulative_sq_dev = 0.0
        self._current_vwap = 0.0
        self._anchor_time = anchor_time
    
    def update(self, high: float, low: float, close: float, volume: int) -> float:
        """
        Update VWAP with new bar data
        
        Returns:
            Current VWAP value
        """
        typical_price = (high + low + close) / 3
        
        self._cumulative_pv += typical_price * volume
        self._cumulative_vol += volume
        
        if self._cumulative_vol > 0:
            self._current_vwap = self._cumulative_pv / self._cumulative_vol
        
        # Update squared deviation for bands
        if self._current_vwap > 0:
            sq_dev = (typical_price - self._current_vwap) ** 2
            self._cumulative_sq_dev += sq_dev * volume
        
        return self._current_vwap
    
    @property
    def vwap(self) -> float:
        return self._current_vwap
    
    @property
    def std_dev(self) -> float:
        if self._cumulative_vol > 0:
            variance = self._cumulative_sq_dev / self._cumulative_vol
            return np.sqrt(variance)
        return 0.0
    
    def get_bands(self, multipliers: List[float] = [1.0, 2.0]) -> Dict[str, float]:
        """Get current VWAP bands"""
        std = self.std_dev
        bands = {"vwap": self._current_vwap}
        
        for i, mult in enumerate(multipliers, 1):
            bands[f"upper_{i}"] = self._current_vwap + std * mult
            bands[f"lower_{i}"] = self._current_vwap - std * mult
        
        return bands
