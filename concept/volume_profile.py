"""
SEF Trading System - Volume Profile Engine
Calculates VAH, VAL, POC, and VP distribution across multiple timeframes
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from config import VolumeProfileConfig

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes for VP calculation"""
    HOURLY = "hourly"
    SESSION = "session"  # Daily RTH session
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class VolumeProfileResult:
    """Volume Profile calculation result"""
    timeframe: Timeframe
    timestamp: datetime  # Period start
    
    # Core levels
    poc: float  # Point of Control - highest volume price
    vah: float  # Value Area High
    val: float  # Value Area Low
    
    # Distribution data
    price_levels: np.ndarray  # Price bins
    volume_at_price: np.ndarray  # Volume per bin
    
    # Additional metrics
    total_volume: int
    value_area_volume: int
    high: float
    low: float
    
    # Profile shape analysis
    profile_type: str = "normal"  # normal, p-shape, b-shape, d-shape
    
    @property
    def value_area_range(self) -> float:
        return self.vah - self.val
    
    @property
    def poc_position(self) -> float:
        """POC position within range (0 = low, 1 = high)"""
        if self.high == self.low:
            return 0.5
        return (self.poc - self.low) / (self.high - self.low)
    
    def price_in_value_area(self, price: float) -> bool:
        """Check if price is within value area"""
        return self.val <= price <= self.vah
    
    def price_relative_to_poc(self, price: float) -> float:
        """Return price distance from POC in ATR-like units"""
        return price - self.poc


@dataclass
class MultiTimeframeVP:
    """Container for multiple timeframe Volume Profiles"""
    monthly: Optional[VolumeProfileResult] = None
    weekly: Optional[VolumeProfileResult] = None
    daily: Optional[VolumeProfileResult] = None
    session: Optional[VolumeProfileResult] = None
    hourly: Optional[VolumeProfileResult] = None
    
    def get_poc_alignment(self) -> str:
        """
        Determine POC alignment across timeframes
        Returns: 'bullish', 'bearish', or 'mixed'
        """
        pocs = []
        
        for tf in [self.monthly, self.weekly, self.daily]:
            if tf is not None:
                pocs.append(tf.poc)
        
        if len(pocs) < 2:
            return "mixed"
        
        # Check if POCs are aligned (each shorter TF POC above/below longer TF)
        all_ascending = all(pocs[i] <= pocs[i+1] for i in range(len(pocs)-1))
        all_descending = all(pocs[i] >= pocs[i+1] for i in range(len(pocs)-1))
        
        if all_ascending:
            return "bullish"
        elif all_descending:
            return "bearish"
        return "mixed"
    
    def get_nearest_support(self, price: float) -> Tuple[float, str]:
        """Find nearest support level below price"""
        supports = []
        
        for name, vp in [("monthly", self.monthly), ("weekly", self.weekly), 
                         ("daily", self.daily), ("session", self.session)]:
            if vp is not None:
                if vp.val < price:
                    supports.append((vp.val, f"{name}_val"))
                if vp.poc < price:
                    supports.append((vp.poc, f"{name}_poc"))
        
        if not supports:
            return (0, "none")
        
        # Return highest support below price
        supports.sort(key=lambda x: x[0], reverse=True)
        return supports[0]
    
    def get_nearest_resistance(self, price: float) -> Tuple[float, str]:
        """Find nearest resistance level above price"""
        resistances = []
        
        for name, vp in [("monthly", self.monthly), ("weekly", self.weekly),
                         ("daily", self.daily), ("session", self.session)]:
            if vp is not None:
                if vp.vah > price:
                    resistances.append((vp.vah, f"{name}_vah"))
                if vp.poc > price:
                    resistances.append((vp.poc, f"{name}_poc"))
        
        if not resistances:
            return (float('inf'), "none")
        
        # Return lowest resistance above price
        resistances.sort(key=lambda x: x[0])
        return resistances[0]


class VolumeProfileEngine:
    """
    Core engine for calculating Volume Profile metrics
    Uses configurable binning and value area calculation
    """
    
    def __init__(self, config: VolumeProfileConfig):
        self.config = config
        self.num_bins = config.num_bins
        self.value_area_pct = config.value_area_pct
        self.poc_smoothing = config.poc_smoothing
        self.smoothing_window = config.poc_smoothing_window
    
    def calculate(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        period_start: Optional[datetime] = None
    ) -> VolumeProfileResult:
        """
        Calculate Volume Profile for given data
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe label for the result
            period_start: Optional period start timestamp
            
        Returns:
            VolumeProfileResult with all VP metrics
        """
        if df.empty:
            raise ValueError("Cannot calculate VP on empty DataFrame")
        
        # Get price range
        high = df["high"].max()
        low = df["low"].min()
        
        if high == low:
            # No range - return flat profile
            return self._create_flat_profile(df, timeframe, period_start)
        
        # Create price bins
        price_levels = np.linspace(low, high, self.num_bins + 1)
        bin_centers = (price_levels[:-1] + price_levels[1:]) / 2
        
        # Calculate volume at each price level
        volume_at_price = self._distribute_volume(df, price_levels)
        
        # Apply smoothing if configured
        if self.poc_smoothing:
            volume_at_price = self._smooth_profile(volume_at_price)
        
        # Find POC (highest volume bin)
        poc_idx = np.argmax(volume_at_price)
        poc = bin_centers[poc_idx]
        
        # Calculate Value Area (VAH/VAL)
        vah, val, va_volume = self._calculate_value_area(
            bin_centers, volume_at_price, poc_idx
        )
        
        # Determine profile type
        profile_type = self._classify_profile(volume_at_price, poc_idx)
        
        return VolumeProfileResult(
            timeframe=timeframe,
            timestamp=period_start or df.index[0],
            poc=poc,
            vah=vah,
            val=val,
            price_levels=bin_centers,
            volume_at_price=volume_at_price,
            total_volume=int(df["volume"].sum()),
            value_area_volume=int(va_volume),
            high=high,
            low=low,
            profile_type=profile_type
        )
    
    def _distribute_volume(
        self,
        df: pd.DataFrame,
        price_levels: np.ndarray
    ) -> np.ndarray:
        """
        Distribute volume across price bins using typical price
        More sophisticated than simple close-based binning
        """
        volume_at_price = np.zeros(len(price_levels) - 1)
        
        for _, row in df.iterrows():
            # Use typical price for bin assignment
            typical = (row["high"] + row["low"] + row["close"]) / 3
            volume = row["volume"]
            
            # Also distribute some volume across the bar's range
            bar_high = row["high"]
            bar_low = row["low"]
            bar_range = bar_high - bar_low
            
            for i in range(len(price_levels) - 1):
                bin_low = price_levels[i]
                bin_high = price_levels[i + 1]
                bin_center = (bin_low + bin_high) / 2
                
                # Check if bar's range overlaps this bin
                if bar_low <= bin_high and bar_high >= bin_low:
                    # Calculate overlap
                    overlap_low = max(bar_low, bin_low)
                    overlap_high = min(bar_high, bin_high)
                    
                    if bar_range > 0:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                    else:
                        overlap_pct = 1.0 if bin_low <= typical <= bin_high else 0.0
                    
                    volume_at_price[i] += volume * overlap_pct
        
        return volume_at_price
    
    def _smooth_profile(self, volume_at_price: np.ndarray) -> np.ndarray:
        """Apply smoothing to prevent spiky POC detection"""
        if len(volume_at_price) < self.smoothing_window:
            return volume_at_price
        
        # Simple moving average smoothing
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed = np.convolve(volume_at_price, kernel, mode='same')
        
        return smoothed
    
    def _calculate_value_area(
        self,
        bin_centers: np.ndarray,
        volume_at_price: np.ndarray,
        poc_idx: int
    ) -> Tuple[float, float, float]:
        """
        Calculate Value Area (VAH/VAL) containing specified percentage of volume
        Expands outward from POC
        """
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.value_area_pct
        
        # Start at POC and expand outward
        va_volume = volume_at_price[poc_idx]
        lower_idx = poc_idx
        upper_idx = poc_idx
        
        while va_volume < target_volume:
            # Calculate volume above and below current VA
            vol_above = (
                volume_at_price[upper_idx + 1] 
                if upper_idx + 1 < len(volume_at_price) 
                else 0
            )
            vol_below = (
                volume_at_price[lower_idx - 1] 
                if lower_idx > 0 
                else 0
            )
            
            # Expand in direction of higher volume
            if vol_above >= vol_below and upper_idx + 1 < len(volume_at_price):
                upper_idx += 1
                va_volume += vol_above
            elif lower_idx > 0:
                lower_idx -= 1
                va_volume += vol_below
            else:
                break
        
        vah = bin_centers[upper_idx]
        val = bin_centers[lower_idx]
        
        return vah, val, va_volume
    
    def _classify_profile(
        self,
        volume_at_price: np.ndarray,
        poc_idx: int
    ) -> str:
        """
        Classify profile shape for context
        - P-shape: High volume at top (short covering / weak longs)
        - b-shape: High volume at bottom (accumulation)
        - D-shape: Balanced / normal distribution
        """
        n = len(volume_at_price)
        third = n // 3
        
        upper_vol = volume_at_price[2*third:].sum()
        middle_vol = volume_at_price[third:2*third].sum()
        lower_vol = volume_at_price[:third].sum()
        
        total = upper_vol + middle_vol + lower_vol
        if total == 0:
            return "normal"
        
        upper_pct = upper_vol / total
        lower_pct = lower_vol / total
        
        if upper_pct > 0.45:
            return "p-shape"
        elif lower_pct > 0.45:
            return "b-shape"
        else:
            return "d-shape"
    
    def _create_flat_profile(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        period_start: Optional[datetime]
    ) -> VolumeProfileResult:
        """Create a flat profile when there's no price range"""
        price = df["close"].iloc[-1]
        total_vol = int(df["volume"].sum())
        
        return VolumeProfileResult(
            timeframe=timeframe,
            timestamp=period_start or df.index[0],
            poc=price,
            vah=price,
            val=price,
            price_levels=np.array([price]),
            volume_at_price=np.array([total_vol]),
            total_volume=total_vol,
            value_area_volume=total_vol,
            high=price,
            low=price,
            profile_type="flat"
        )


class MultiTimeframeVPEngine:
    """
    Manages Volume Profile calculation across multiple timeframes
    """
    
    def __init__(self, config: VolumeProfileConfig):
        self.engine = VolumeProfileEngine(config)
        self._cache: Dict[str, VolumeProfileResult] = {}
    
    def calculate_all(
        self,
        intraday_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        current_time: Optional[datetime] = None
    ) -> MultiTimeframeVP:
        """
        Calculate VP for all timeframes
        
        Args:
            intraday_df: 5-minute intraday data (for hourly, session, daily)
            daily_df: Daily data (for weekly, monthly)
            current_time: Reference time for period boundaries
            
        Returns:
            MultiTimeframeVP with all calculated profiles
        """
        current_time = current_time or datetime.now()
        
        result = MultiTimeframeVP()
        
        # Calculate each timeframe
        try:
            result.monthly = self._calculate_monthly(daily_df, current_time)
        except Exception as e:
            logger.warning(f"Monthly VP calculation failed: {e}")
        
        try:
            result.weekly = self._calculate_weekly(daily_df, current_time)
        except Exception as e:
            logger.warning(f"Weekly VP calculation failed: {e}")
        
        try:
            result.daily = self._calculate_daily(intraday_df, current_time)
        except Exception as e:
            logger.warning(f"Daily VP calculation failed: {e}")
        
        try:
            result.session = self._calculate_session(intraday_df, current_time)
        except Exception as e:
            logger.warning(f"Session VP calculation failed: {e}")
        
        return result
    
    def _calculate_monthly(
        self,
        df: pd.DataFrame,
        current_time: datetime
    ) -> Optional[VolumeProfileResult]:
        """Calculate current month's VP from daily data"""
        if df.empty:
            return None
        
        month_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_data = df[df.index >= month_start]
        
        if month_data.empty or len(month_data) < 2:
            # Use last N days if current month has insufficient data
            month_data = df.tail(20)
        
        if month_data.empty:
            return None
        
        return self.engine.calculate(month_data, Timeframe.MONTHLY, month_start)
    
    def _calculate_weekly(
        self,
        df: pd.DataFrame,
        current_time: datetime
    ) -> Optional[VolumeProfileResult]:
        """Calculate current week's VP from daily data"""
        if df.empty:
            return None
        
        # Get Monday of current week
        days_since_monday = current_time.weekday()
        week_start = (current_time - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        week_data = df[df.index >= week_start]
        
        if week_data.empty or len(week_data) < 1:
            # Use last 5 trading days
            week_data = df.tail(5)
        
        if week_data.empty:
            return None
        
        return self.engine.calculate(week_data, Timeframe.WEEKLY, week_start)
    
    def _calculate_daily(
        self,
        df: pd.DataFrame,
        current_time: datetime
    ) -> Optional[VolumeProfileResult]:
        """Calculate today's VP from intraday data"""
        if df.empty:
            return None
        
        day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        day_data = df[df.index >= day_start]
        
        if day_data.empty:
            # Use latest available day
            latest_date = df.index[-1].date()
            day_data = df[df.index.date == latest_date]
        
        if day_data.empty:
            return None
        
        return self.engine.calculate(day_data, Timeframe.DAILY, day_start)
    
    def _calculate_session(
        self,
        df: pd.DataFrame,
        current_time: datetime
    ) -> Optional[VolumeProfileResult]:
        """Calculate current RTH session VP"""
        # Session is essentially same as daily for RTH data
        return self._calculate_daily(df, current_time)
    
    def get_composite_levels(self, mtf_vp: MultiTimeframeVP) -> Dict[str, List[float]]:
        """
        Extract key levels from multi-timeframe VP
        Useful for quick reference
        """
        levels = {
            "poc": [],
            "vah": [],
            "val": []
        }
        
        for name in ["monthly", "weekly", "daily", "session"]:
            vp = getattr(mtf_vp, name)
            if vp is not None:
                levels["poc"].append((name, vp.poc))
                levels["vah"].append((name, vp.vah))
                levels["val"].append((name, vp.val))
        
        return levels
