"""
MTF Auction Scanner - Multi-Timeframe Non-Bias Setup Detection
==============================================================
A directionally neutral scanner that identifies HIGH and LOW scenario setups
across multiple timeframes (30min to 4hr) using Volume Profile, Flow Control,
and RSI with explicit YELLOW/NEUTRAL period detection.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class SignalState(Enum):
    """Signal states with explicit neutral handling"""
    LONG_SETUP = "LONG_SETUP"
    SHORT_SETUP = "SHORT_SETUP"
    YELLOW = "YELLOW"          # Mixed signals - wait
    NEUTRAL = "NEUTRAL"        # No setup detected
    NO_DATA = "NO_DATA"        # Insufficient data
    
    @property
    def emoji(self) -> str:
        return {
            "LONG_SETUP": "🟢",
            "SHORT_SETUP": "🔴", 
            "YELLOW": "🟡",
            "NEUTRAL": "⚪",
            "NO_DATA": "⬜"
        }[self.value]
    
    @property
    def action(self) -> str:
        return {
            "LONG_SETUP": "PREPARE LONG",
            "SHORT_SETUP": "PREPARE SHORT",
            "YELLOW": "WAIT - MIXED",
            "NEUTRAL": "NO SETUP",
            "NO_DATA": "SKIP"
        }[self.value]


class Timeframe(Enum):
    """Supported scanning timeframes"""
    M30 = ("30min", 30)
    H1 = ("1hour", 60)
    H2 = ("2hour", 120)
    H4 = ("4hour", 240)
    
    @property
    def label(self) -> str:
        return self.value[0]
    
    @property
    def minutes(self) -> int:
        return self.value[1]


@dataclass
class VolumeProfile:
    """Volume Profile metrics for a given period"""
    poc: float              # Point of Control - highest volume price
    vah: float              # Value Area High
    val: float              # Value Area Low
    value_area_pct: float   # % of volume in value area (typically 70%)
    total_volume: float
    price_levels: Dict[float, float] = field(default_factory=dict)  # price -> volume
    
    @property
    def value_width(self) -> float:
        """Width of value area in price terms"""
        return self.vah - self.val
    
    @property
    def value_center(self) -> float:
        """Center of value area"""
        return (self.vah + self.val) / 2


@dataclass
class FlowMetrics:
    """Order flow and delta metrics"""
    cumulative_delta: float      # Net buying - selling pressure
    delta_momentum: float        # Rate of change of delta
    buy_volume_pct: float        # % of volume on upticks
    sell_volume_pct: float       # % of volume on downticks
    flow_imbalance: float        # Normalized flow bias (-1 to +1)
    
    @property
    def flow_state(self) -> str:
        if self.flow_imbalance > 0.3:
            return "STRONG_BUY_FLOW"
        elif self.flow_imbalance > 0.1:
            return "MILD_BUY_FLOW"
        elif self.flow_imbalance < -0.3:
            return "STRONG_SELL_FLOW"
        elif self.flow_imbalance < -0.1:
            return "MILD_SELL_FLOW"
        else:
            return "BALANCED"


@dataclass
class RSIMetrics:
    """RSI with zone classification"""
    value: float
    slope: float                 # Direction of RSI movement
    divergence: Optional[str]    # None, "BULLISH_DIV", "BEARISH_DIV"
    
    @property
    def zone(self) -> str:
        if self.value >= 75:
            return "OVERBOUGHT"
        elif self.value >= 65:
            return "NEAR_OVERBOUGHT"  # Potential reversal zone
        elif self.value >= 55:
            return "BULLISH"
        elif self.value >= 45:
            return "NEUTRAL"
        elif self.value >= 35:
            return "BEARISH"
        elif self.value >= 30:
            return "NEAR_OVERSOLD"    # Potential bounce zone
        else:
            return "OVERSOLD"
    
    @property
    def momentum_aligned(self) -> Optional[bool]:
        """True if RSI zone aligns with momentum direction"""
        if self.zone in ["BULLISH", "OVERBOUGHT", "NEAR_OVERBOUGHT"] and self.slope > 0:
            return True
        elif self.zone in ["BEARISH", "OVERSOLD", "NEAR_OVERSOLD"] and self.slope < 0:
            return True
        elif self.zone == "NEUTRAL":
            return None
        return False


@dataclass
class VWAPMetrics:
    """VWAP and deviation bands"""
    vwap: float                  # Volume Weighted Average Price
    upper_band_1: float          # +1 standard deviation
    lower_band_1: float          # -1 standard deviation
    upper_band_2: float          # +2 standard deviation
    lower_band_2: float          # -2 standard deviation
    price_vs_vwap: float         # Current price distance from VWAP
    deviation_pct: float         # % deviation from VWAP
    
    @property
    def zone(self) -> str:
        """Where is price relative to VWAP bands"""
        if self.price_vs_vwap > self.upper_band_2:
            return "EXTREME_ABOVE"
        elif self.price_vs_vwap > self.upper_band_1:
            return "ABOVE_1SD"
        elif self.price_vs_vwap > 0:
            return "ABOVE_VWAP"
        elif self.price_vs_vwap > -abs(self.lower_band_1 - self.vwap):
            return "BELOW_VWAP"
        elif self.price_vs_vwap > -abs(self.lower_band_2 - self.vwap):
            return "BELOW_1SD"
        else:
            return "EXTREME_BELOW"


@dataclass
class TimeframeAnalysis:
    """Complete analysis for a single timeframe"""
    timeframe: Timeframe
    timestamp: datetime
    current_price: float
    volume_profile: VolumeProfile
    flow: FlowMetrics
    rsi: RSIMetrics
    
    # Position relative to value
    price_vs_poc: float          # Distance from POC (normalized)
    price_vs_vah: float          # Distance from VAH
    price_vs_val: float          # Distance from VAL
    position_in_value: str       # "ABOVE_VALUE", "IN_VALUE", "BELOW_VALUE"
    
    # Scores
    bull_score: float            # 0-100 bullish evidence
    bear_score: float            # 0-100 bearish evidence
    
    # Final signal
    signal: SignalState
    confidence: float            # 0-100
    
    # Optional fields with defaults at the end
    vwap: Optional[VWAPMetrics] = None  # VWAP analysis
    notes: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """Complete scan result across all timeframes"""
    symbol: str
    scan_time: datetime
    timeframe_analyses: Dict[Timeframe, TimeframeAnalysis]
    
    # Aggregated assessment
    dominant_signal: SignalState
    confluence_score: float      # How aligned are timeframes
    actionable: bool             # Is there a tradeable setup
    
    # Scenario assessment
    high_scenario_prob: float    # Probability of upside resolution
    low_scenario_prob: float     # Probability of downside resolution
    neutral_prob: float          # Probability of chop/range
    
    summary: str


# =============================================================================
# VOLUME PROFILE ENGINE
# =============================================================================

class VolumeProfileEngine:
    """
    Calculates Volume Profile metrics including POC, VAH, VAL
    
    For Brokers:
    -----------
    Volume Profile shows WHERE the most trading occurred, not just how much.
    - POC (Point of Control): The "fair price" where most volume traded
    - Value Area (VAH-VAL): The range containing 70% of volume - institutional acceptance zone
    - Price above VAH: Market exploring higher prices, potential breakout or rejection
    - Price below VAL: Market exploring lower prices, potential breakdown or rejection
    
    For Programmers:
    ---------------
    Uses histogram approach: discretize price into bins, aggregate volume per bin,
    then find POC as max-volume bin and expand outward to capture 70% of total volume.
    """
    
    def __init__(self, value_area_pct: float = 0.70, num_bins: int = 50):
        self.value_area_pct = value_area_pct
        self.num_bins = num_bins
    
    def calculate(self, df: pd.DataFrame) -> VolumeProfile:
        """
        Calculate volume profile from OHLCV data
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            VolumeProfile object with POC, VAH, VAL
        """
        if len(df) < 5:
            # Return neutral profile if insufficient data
            mid = df['close'].iloc[-1] if len(df) > 0 else 0
            return VolumeProfile(
                poc=mid, vah=mid, val=mid,
                value_area_pct=0, total_volume=0
            )
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max == price_min:
            price_max = price_min * 1.001  # Avoid division by zero
        
        bin_size = (price_max - price_min) / self.num_bins
        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Distribute volume across price levels using typical price
        volume_at_price = np.zeros(self.num_bins)
        
        for _, row in df.iterrows():
            # Distribute bar's volume across its range
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']
            
            # Find bins this bar touches
            low_bin = max(0, int((bar_low - price_min) / bin_size))
            high_bin = min(self.num_bins - 1, int((bar_high - price_min) / bin_size))
            
            # Distribute volume (weighted toward typical price)
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            typical_bin = min(self.num_bins - 1, max(0, int((typical_price - price_min) / bin_size)))
            
            for b in range(low_bin, high_bin + 1):
                # More volume near typical price
                distance = abs(b - typical_bin)
                weight = 1 / (1 + distance * 0.5)
                volume_at_price[b] += bar_volume * weight
        
        # Normalize
        total_vol = volume_at_price.sum()
        if total_vol > 0:
            volume_at_price = volume_at_price / total_vol * df['volume'].sum()
        
        # Find POC (highest volume bin)
        poc_bin = np.argmax(volume_at_price)
        poc = bin_centers[poc_bin]
        
        # Calculate Value Area (expand from POC until 70% captured)
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.value_area_pct
        
        captured_volume = volume_at_price[poc_bin]
        val_bin = poc_bin
        vah_bin = poc_bin
        
        while captured_volume < target_volume and (val_bin > 0 or vah_bin < self.num_bins - 1):
            # Look at bins above and below, add the one with more volume
            vol_below = volume_at_price[val_bin - 1] if val_bin > 0 else 0
            vol_above = volume_at_price[vah_bin + 1] if vah_bin < self.num_bins - 1 else 0
            
            if vol_above >= vol_below and vah_bin < self.num_bins - 1:
                vah_bin += 1
                captured_volume += vol_above
            elif val_bin > 0:
                val_bin -= 1
                captured_volume += vol_below
            else:
                vah_bin += 1
                captured_volume += vol_above
        
        val = bin_centers[val_bin]
        vah = bin_centers[vah_bin]
        
        # Build price level dictionary
        price_levels = {bin_centers[i]: volume_at_price[i] for i in range(self.num_bins)}
        
        return VolumeProfile(
            poc=poc,
            vah=vah,
            val=val,
            value_area_pct=captured_volume / total_volume if total_volume > 0 else 0,
            total_volume=total_volume,
            price_levels=price_levels
        )


# =============================================================================
# FLOW CONTROL ENGINE
# =============================================================================

class FlowControlEngine:
    """
    Analyzes order flow and volume delta
    
    For Brokers:
    -----------
    Flow analysis reveals WHO is in control - buyers or sellers.
    - Positive delta: More aggressive buying (hitting the ask)
    - Negative delta: More aggressive selling (hitting the bid)
    - Flow imbalance: Normalized measure of buyer/seller dominance
    - Momentum: Is the flow accelerating or decelerating?
    
    For Programmers:
    ---------------
    Without true tick data, we estimate delta using price movement:
    - Up bars (close > open): Volume attributed to buyers
    - Down bars (close < open): Volume attributed to sellers
    - Doji bars: Split 50/50
    Delta momentum uses rolling rate of change.
    """
    
    def __init__(self, momentum_period: int = 5):
        self.momentum_period = momentum_period
    
    def calculate(self, df: pd.DataFrame) -> FlowMetrics:
        """
        Calculate flow metrics from OHLCV data
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            FlowMetrics object
        """
        if len(df) < 3:
            return FlowMetrics(
                cumulative_delta=0, delta_momentum=0,
                buy_volume_pct=0.5, sell_volume_pct=0.5,
                flow_imbalance=0
            )
        
        # Estimate delta per bar
        df = df.copy()
        df['delta'] = df.apply(self._estimate_bar_delta, axis=1)
        
        # Cumulative delta
        df['cum_delta'] = df['delta'].cumsum()
        cumulative_delta = df['cum_delta'].iloc[-1]
        
        # Delta momentum (rate of change)
        if len(df) >= self.momentum_period:
            recent_delta = df['cum_delta'].iloc[-1]
            past_delta = df['cum_delta'].iloc[-self.momentum_period]
            delta_momentum = (recent_delta - past_delta) / self.momentum_period
        else:
            delta_momentum = 0
        
        # Buy/Sell volume percentages
        total_volume = df['volume'].sum()
        if total_volume > 0:
            buy_volume = df[df['delta'] > 0]['volume'].sum()
            sell_volume = df[df['delta'] < 0]['volume'].sum()
            neutral_volume = df[df['delta'] == 0]['volume'].sum()
            
            buy_volume_pct = (buy_volume + neutral_volume * 0.5) / total_volume
            sell_volume_pct = (sell_volume + neutral_volume * 0.5) / total_volume
        else:
            buy_volume_pct = 0.5
            sell_volume_pct = 0.5
        
        # Flow imbalance (-1 to +1)
        flow_imbalance = buy_volume_pct - sell_volume_pct
        
        return FlowMetrics(
            cumulative_delta=cumulative_delta,
            delta_momentum=delta_momentum,
            buy_volume_pct=buy_volume_pct,
            sell_volume_pct=sell_volume_pct,
            flow_imbalance=flow_imbalance
        )
    
    def _estimate_bar_delta(self, row: pd.Series) -> float:
        """Estimate delta for a single bar"""
        bar_range = row['high'] - row['low']
        if bar_range == 0:
            return 0
        
        # Close location value (0 = low, 1 = high)
        clv = (row['close'] - row['low']) / bar_range
        
        # Delta estimate: volume * (2 * CLV - 1)
        # CLV > 0.5: buying pressure, CLV < 0.5: selling pressure
        delta = row['volume'] * (2 * clv - 1)
        
        return delta


# =============================================================================
# RSI ENGINE
# =============================================================================

class RSIEngine:
    """
    RSI calculation with slope and divergence detection
    
    For Brokers:
    -----------
    RSI measures momentum strength on a 0-100 scale.
    - Above 70: Overbought - momentum stretched, potential reversal
    - Below 30: Oversold - momentum stretched, potential reversal
    - 40-60: Neutral zone - no clear momentum bias
    - Slope: Direction RSI is moving (accelerating/decelerating)
    - Divergence: Price makes new high/low but RSI doesn't confirm
    
    For Programmers:
    ---------------
    Standard Wilder's RSI with added slope calculation and divergence detection.
    Slope is simply the rate of change of RSI over recent bars.
    Divergence compares price highs/lows with RSI highs/lows.
    """
    
    def __init__(self, period: int = 14, slope_period: int = 3):
        self.period = period
        self.slope_period = slope_period
    
    def calculate(self, df: pd.DataFrame) -> RSIMetrics:
        """
        Calculate RSI metrics
        
        Args:
            df: DataFrame with 'close' column
        
        Returns:
            RSIMetrics object
        """
        if len(df) < self.period + 5:
            return RSIMetrics(value=50, slope=0, divergence=None)
        
        # Calculate RSI
        close = df['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # RSI slope
        if len(rsi) >= self.slope_period:
            slope = (rsi.iloc[-1] - rsi.iloc[-self.slope_period]) / self.slope_period
        else:
            slope = 0
        
        # Divergence detection
        divergence = self._detect_divergence(df, rsi)
        
        return RSIMetrics(
            value=current_rsi,
            slope=slope,
            divergence=divergence
        )
    
    def _detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 10) -> Optional[str]:
        """Detect bullish or bearish divergence"""
        if len(df) < lookback:
            return None
        
        recent_prices = df['close'].iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]
        
        # Find swing points
        price_highs = recent_prices.iloc[recent_prices.values.argmax()]
        price_lows = recent_prices.iloc[recent_prices.values.argmin()]
        
        rsi_at_price_high = recent_rsi.iloc[recent_prices.values.argmax()]
        rsi_at_price_low = recent_rsi.iloc[recent_prices.values.argmin()]
        
        current_price = df['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Bearish divergence: price higher high, RSI lower high
        if current_price >= price_highs * 0.998 and current_rsi < rsi_at_price_high - 5:
            return "BEARISH_DIV"
        
        # Bullish divergence: price lower low, RSI higher low
        if current_price <= price_lows * 1.002 and current_rsi > rsi_at_price_low + 5:
            return "BULLISH_DIV"
        
        return None


# =============================================================================
# VWAP ENGINE
# =============================================================================

class VWAPEngine:
    """
    VWAP calculation with standard deviation bands
    
    For Brokers:
    -----------
    VWAP (Volume Weighted Average Price) shows the average price weighted by volume.
    - Institutional traders use VWAP as a benchmark
    - Price above VWAP: Buyers in control for the session
    - Price below VWAP: Sellers in control for the session
    - Bands show standard deviations - extreme moves often revert
    
    For Programmers:
    ---------------
    VWAP = cumsum(price * volume) / cumsum(volume)
    Standard deviation bands calculated from typical price variance.
    """
    
    def __init__(self, band_multiplier_1: float = 1.0, band_multiplier_2: float = 2.0):
        self.band_mult_1 = band_multiplier_1
        self.band_mult_2 = band_multiplier_2
    
    def calculate(self, df: pd.DataFrame) -> VWAPMetrics:
        """
        Calculate VWAP and deviation bands
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            VWAPMetrics object
        """
        if len(df) < 5:
            price = df['close'].iloc[-1] if len(df) > 0 else 0
            return VWAPMetrics(
                vwap=price, upper_band_1=price, lower_band_1=price,
                upper_band_2=price, lower_band_2=price,
                price_vs_vwap=0, deviation_pct=0
            )
        
        df = df.copy()
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Cumulative calculations
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['cum_tp_volume'] = df['tp_volume'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        
        # VWAP
        df['vwap'] = df['cum_tp_volume'] / df['cum_volume']
        
        # Standard deviation for bands
        df['squared_diff'] = (df['typical_price'] - df['vwap']) ** 2
        df['cum_squared_diff'] = (df['squared_diff'] * df['volume']).cumsum()
        df['variance'] = df['cum_squared_diff'] / df['cum_volume']
        df['std_dev'] = np.sqrt(df['variance'])
        
        # Current values
        current_vwap = df['vwap'].iloc[-1]
        current_std = df['std_dev'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Bands
        upper_1 = current_vwap + (current_std * self.band_mult_1)
        lower_1 = current_vwap - (current_std * self.band_mult_1)
        upper_2 = current_vwap + (current_std * self.band_mult_2)
        lower_2 = current_vwap - (current_std * self.band_mult_2)
        
        # Price vs VWAP
        price_vs_vwap = current_price - current_vwap
        deviation_pct = (price_vs_vwap / current_vwap * 100) if current_vwap != 0 else 0
        
        return VWAPMetrics(
            vwap=current_vwap,
            upper_band_1=upper_1,
            lower_band_1=lower_1,
            upper_band_2=upper_2,
            lower_band_2=lower_2,
            price_vs_vwap=price_vs_vwap,
            deviation_pct=deviation_pct
        )


# =============================================================================
# SIGNAL SCORER
# =============================================================================

class SignalScorer:
    """
    Non-biased scoring system that evaluates bullish and bearish evidence separately
    
    For Brokers:
    -----------
    This scorer doesn't try to predict direction - it measures the STRENGTH of 
    evidence for each scenario independently. A strong bullish score with weak
    bearish score = long setup. Both scores moderate = YELLOW (wait).
    
    For Programmers:
    ---------------
    Separate scoring functions accumulate points for bullish/bearish evidence.
    Final signal determination uses thresholds and the gap between scores.
    Yellow state triggers when scores are close or both are moderate.
    """
    
    # Thresholds
    STRONG_THRESHOLD = 65       # Score above this = strong signal
    MODERATE_THRESHOLD = 45     # Score above this = moderate signal
    MIN_SCORE_GAP = 20          # Minimum gap between bull/bear for directional signal
    
    def score(self, 
              current_price: float,
              vp: VolumeProfile, 
              flow: FlowMetrics, 
              rsi: RSIMetrics,
              vwap: Optional[VWAPMetrics] = None) -> Tuple[float, float, SignalState, float, List[str]]:
        """
        Score bullish and bearish evidence
        
        Returns:
            Tuple of (bull_score, bear_score, signal_state, confidence, notes)
        """
        bull_score = 0.0
        bear_score = 0.0
        notes = []
        
        # =====================================================================
        # PRICE VS VALUE AREA (25 points max each direction)
        # =====================================================================
        
        if current_price > vp.vah:
            # Above value - potential strength or overextension
            distance_pct = (current_price - vp.vah) / vp.value_width if vp.value_width > 0 else 0
            if distance_pct < 0.5:
                bull_score += 20  # Breaking out, not overextended
                notes.append("Price above VAH - bullish breakout zone")
            else:
                bull_score += 8  # Extended, less bullish
                bear_score += 12  # Potential rejection
                notes.append("Price extended above VAH - watch for rejection")
                
        elif current_price < vp.val:
            # Below value - potential weakness or oversold
            distance_pct = (vp.val - current_price) / vp.value_width if vp.value_width > 0 else 0
            if distance_pct < 0.5:
                bear_score += 20  # Breaking down, not overextended
                notes.append("Price below VAL - bearish breakdown zone")
            else:
                bear_score += 8  # Extended, less bearish
                bull_score += 12  # Potential rejection/bounce
                notes.append("Price extended below VAL - watch for bounce")
                
        else:
            # Inside value - neutral, look at position relative to POC
            if current_price > vp.poc:
                bull_score += 8
                notes.append("Inside value, above POC")
            elif current_price < vp.poc:
                bear_score += 8
                notes.append("Inside value, below POC")
            else:
                notes.append("At POC - balanced")
        
        # =====================================================================
        # VWAP ANALYSIS (20 points max each direction) - KEY FOR ROB'S TECHNIQUE
        # =====================================================================
        
        if vwap is not None:
            if vwap.zone == "EXTREME_ABOVE":
                bear_score += 15  # Mean reversion likely
                notes.append(f"Price extreme above VWAP (+{vwap.deviation_pct:.1f}%) - reversion risk")
            elif vwap.zone == "ABOVE_1SD":
                bull_score += 10  # Strong but extended
                notes.append(f"Price above VWAP +1SD ({vwap.deviation_pct:.1f}%)")
            elif vwap.zone == "ABOVE_VWAP":
                bull_score += 15  # Healthy above VWAP
                notes.append(f"Price above VWAP (+{vwap.deviation_pct:.1f}%) - buyers in control")
            elif vwap.zone == "BELOW_VWAP":
                bear_score += 15  # Healthy below VWAP
                notes.append(f"Price below VWAP ({vwap.deviation_pct:.1f}%) - sellers in control")
            elif vwap.zone == "BELOW_1SD":
                bear_score += 10  # Weak but extended
                notes.append(f"Price below VWAP -1SD ({vwap.deviation_pct:.1f}%)")
            elif vwap.zone == "EXTREME_BELOW":
                bull_score += 15  # Mean reversion likely
                notes.append(f"Price extreme below VWAP ({vwap.deviation_pct:.1f}%) - bounce likely")
            
            # VWAP as support/resistance (price near VWAP)
            if abs(vwap.deviation_pct) < 0.3:
                notes.append("⚡ Price at VWAP - key decision point")
        
        # =====================================================================
        # FLOW CONTROL (30 points max each direction)
        # =====================================================================
        
        if flow.flow_imbalance > 0.3:
            bull_score += 30
            notes.append(f"Strong buy flow ({flow.flow_imbalance:.2f})")
        elif flow.flow_imbalance > 0.15:
            bull_score += 20
            notes.append(f"Moderate buy flow ({flow.flow_imbalance:.2f})")
        elif flow.flow_imbalance > 0.05:
            bull_score += 12
            notes.append(f"Mild buy flow ({flow.flow_imbalance:.2f})")
        elif flow.flow_imbalance < -0.3:
            bear_score += 30
            notes.append(f"Strong sell flow ({flow.flow_imbalance:.2f})")
        elif flow.flow_imbalance < -0.15:
            bear_score += 20
            notes.append(f"Moderate sell flow ({flow.flow_imbalance:.2f})")
        elif flow.flow_imbalance < -0.05:
            bear_score += 12
            notes.append(f"Mild sell flow ({flow.flow_imbalance:.2f})")
        else:
            notes.append("Flow balanced")
        
        # Delta momentum bonus
        if flow.delta_momentum > 0 and flow.flow_imbalance > 0:
            bull_score += 8
            notes.append("Accelerating buy pressure")
        elif flow.delta_momentum < 0 and flow.flow_imbalance < 0:
            bear_score += 8
            notes.append("Accelerating sell pressure")
        
        # =====================================================================
        # RSI (25 points max each direction)
        # =====================================================================
        
        if rsi.zone == "BULLISH":
            bull_score += 20
            notes.append(f"RSI bullish zone ({rsi.value:.1f})")
        elif rsi.zone == "OVERBOUGHT":
            if rsi.slope > 0:
                bull_score += 10  # Still climbing
                notes.append(f"RSI overbought but climbing ({rsi.value:.1f})")
            else:
                bear_score += 15  # Potential reversal
                notes.append(f"RSI overbought and rolling ({rsi.value:.1f})")
        elif rsi.zone == "NEAR_OVERBOUGHT":
            # 65-75 range - stretched but not extreme
            if rsi.slope > 0:
                bull_score += 12
                notes.append(f"RSI near overbought, climbing ({rsi.value:.1f})")
            else:
                bear_score += 10  # Potential reversal starting
                bull_score += 5   # But still has momentum
                notes.append(f"RSI near overbought, rolling ({rsi.value:.1f}) ⚠️")
        elif rsi.zone == "BEARISH":
            bear_score += 15
            notes.append(f"RSI bearish zone ({rsi.value:.1f})")
        elif rsi.zone == "NEAR_OVERSOLD":
            # 30-35 range - bounce potential building
            if rsi.slope < 0:
                bear_score += 8   # Still weak
                bull_score += 8   # But bounce potential
                notes.append(f"RSI near oversold ({rsi.value:.1f}) - bounce potential")
            else:
                bull_score += 15  # Turning up from lows
                notes.append(f"RSI near oversold and turning ({rsi.value:.1f}) ✓")
        elif rsi.zone == "OVERSOLD":
            if rsi.slope < 0:
                bear_score += 10  # Still falling
                notes.append(f"RSI oversold but falling ({rsi.value:.1f})")
            else:
                bull_score += 18  # Strong reversal potential
                notes.append(f"RSI oversold and turning ({rsi.value:.1f}) ✓")
        else:
            notes.append(f"RSI neutral ({rsi.value:.1f})")
        
        # Divergence signals
        if rsi.divergence == "BULLISH_DIV":
            bull_score += 12
            bear_score -= 8
            notes.append("⚠️ Bullish RSI divergence detected")
        elif rsi.divergence == "BEARISH_DIV":
            bear_score += 12
            bull_score -= 8
            notes.append("⚠️ Bearish RSI divergence detected")
        
        # =====================================================================
        # DETERMINE SIGNAL STATE
        # =====================================================================
        
        # Clamp scores to 0-100
        bull_score = max(0, min(100, bull_score))
        bear_score = max(0, min(100, bear_score))
        
        score_gap = abs(bull_score - bear_score)
        max_score = max(bull_score, bear_score)
        min_score = min(bull_score, bear_score)
        
        # Decision logic
        # NEUTRAL = truly no evidence (both sides weak)
        # YELLOW = conflicting evidence (both sides have something)
        
        if max_score < 25 and min_score < 15:
            # Very little evidence either way = true NEUTRAL
            signal = SignalState.NEUTRAL
            confidence = 100 - max_score
            notes.append("Insufficient directional evidence")
            
        elif max_score < self.MODERATE_THRESHOLD and min_score >= 15:
            # Both sides have evidence but neither dominant = YELLOW
            signal = SignalState.YELLOW
            if bull_score > bear_score:
                confidence = 45 + (score_gap / 2)
                notes.append(f"Mixed signals, slight bullish lean - YELLOW")
            elif bear_score > bull_score:
                confidence = 45 + (score_gap / 2)
                notes.append(f"Mixed signals, slight bearish lean - YELLOW")
            else:
                confidence = 40
                notes.append("Mixed signals, no lean - YELLOW")
            
        elif score_gap < self.MIN_SCORE_GAP:
            signal = SignalState.YELLOW
            confidence = 50 - (score_gap / self.MIN_SCORE_GAP * 25)
            notes.append(f"Mixed signals - gap only {score_gap:.1f} points")
            
        elif bull_score > bear_score and bull_score >= self.STRONG_THRESHOLD:
            signal = SignalState.LONG_SETUP
            confidence = min(95, bull_score - bear_score + 40)
            notes.append("✓ Long setup confirmed")
            
        elif bear_score > bull_score and bear_score >= self.STRONG_THRESHOLD:
            signal = SignalState.SHORT_SETUP
            confidence = min(95, bear_score - bull_score + 40)
            notes.append("✓ Short setup confirmed")
            
        elif bull_score > bear_score:
            signal = SignalState.YELLOW  # Leaning long but not confirmed
            confidence = 40 + (score_gap / 2)
            notes.append("Leaning bullish but not confirmed - YELLOW")
            
        elif bear_score > bull_score:
            signal = SignalState.YELLOW  # Leaning short but not confirmed
            confidence = 40 + (score_gap / 2)
            notes.append("Leaning bearish but not confirmed - YELLOW")
            
        else:
            signal = SignalState.NEUTRAL
            confidence = 50
        
        return bull_score, bear_score, signal, confidence, notes


# =============================================================================
# MULTI-TIMEFRAME SCANNER
# =============================================================================

class MTFAuctionScanner:
    """
    Multi-Timeframe Non-Bias Auction Scanner
    
    Scans across 30min, 1hr, 2hr, and 4hr timeframes to identify
    directionally-neutral setups with explicit YELLOW/NEUTRAL states.
    
    For Brokers:
    -----------
    This scanner looks at the market through multiple lenses simultaneously.
    - Each timeframe gives its own "vote" on direction
    - When timeframes agree = higher confidence setup
    - When timeframes disagree = YELLOW state (wait for clarity)
    - No bias toward long or short - purely evidence-based
    
    For Programmers:
    ---------------
    Resamples input data to each target timeframe, runs independent analysis,
    then aggregates results with confluence scoring.
    """
    
    def __init__(self):
        self.vp_engine = VolumeProfileEngine()
        self.flow_engine = FlowControlEngine()
        self.rsi_engine = RSIEngine()
        self.vwap_engine = VWAPEngine()
        self.scorer = SignalScorer()
        
        self.timeframes = [Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]
    
    def scan(self, 
             df: pd.DataFrame, 
             symbol: str = "UNKNOWN",
             timeframes: Optional[List[Timeframe]] = None) -> ScanResult:
        """
        Run complete multi-timeframe scan
        
        Args:
            df: DataFrame with OHLCV data (assumes smallest available timeframe)
                Must have columns: [open, high, low, close, volume]
                Index should be DatetimeIndex
            symbol: Ticker symbol
            timeframes: List of timeframes to scan (default: all)
        
        Returns:
            ScanResult with analysis for each timeframe and aggregate assessment
        """
        if timeframes is None:
            timeframes = self.timeframes
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Current price
        current_price = df['close'].iloc[-1]
        scan_time = df.index[-1]
        
        # Analyze each timeframe
        analyses = {}
        for tf in timeframes:
            tf_df = self._resample_to_timeframe(df, tf)
            if len(tf_df) >= 20:  # Minimum bars needed
                analysis = self._analyze_timeframe(tf_df, tf, current_price)
                analyses[tf] = analysis
            else:
                # Create NO_DATA analysis
                analyses[tf] = self._create_no_data_analysis(tf, current_price, scan_time)
        
        # Aggregate across timeframes
        result = self._aggregate_analyses(symbol, scan_time, analyses)
        
        return result
    
    def _resample_to_timeframe(self, df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        """Resample data to target timeframe"""
        rule = f"{tf.minutes}min" if tf.minutes < 60 else f"{tf.minutes // 60}h"
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def _analyze_timeframe(self, 
                           df: pd.DataFrame, 
                           tf: Timeframe,
                           current_price: float) -> TimeframeAnalysis:
        """Analyze a single timeframe"""
        
        # Calculate components
        vp = self.vp_engine.calculate(df)
        flow = self.flow_engine.calculate(df)
        rsi = self.rsi_engine.calculate(df)
        vwap = self.vwap_engine.calculate(df)
        
        # Price position relative to value
        price_vs_poc = (current_price - vp.poc) / vp.value_width if vp.value_width > 0 else 0
        price_vs_vah = current_price - vp.vah
        price_vs_val = current_price - vp.val
        
        if current_price > vp.vah:
            position = "ABOVE_VALUE"
        elif current_price < vp.val:
            position = "BELOW_VALUE"
        else:
            position = "IN_VALUE"
        
        # Score (now includes VWAP)
        bull_score, bear_score, signal, confidence, notes = self.scorer.score(
            current_price, vp, flow, rsi, vwap
        )
        
        return TimeframeAnalysis(
            timeframe=tf,
            timestamp=df.index[-1],
            current_price=current_price,
            volume_profile=vp,
            flow=flow,
            rsi=rsi,
            price_vs_poc=price_vs_poc,
            price_vs_vah=price_vs_vah,
            price_vs_val=price_vs_val,
            position_in_value=position,
            bull_score=bull_score,
            bear_score=bear_score,
            signal=signal,
            confidence=confidence,
            vwap=vwap,
            notes=notes
        )
    
    def _create_no_data_analysis(self, 
                                  tf: Timeframe, 
                                  current_price: float,
                                  timestamp: datetime) -> TimeframeAnalysis:
        """Create placeholder for insufficient data"""
        return TimeframeAnalysis(
            timeframe=tf,
            timestamp=timestamp,
            current_price=current_price,
            volume_profile=VolumeProfile(current_price, current_price, current_price, 0, 0),
            flow=FlowMetrics(0, 0, 0.5, 0.5, 0),
            rsi=RSIMetrics(50, 0, None),
            price_vs_poc=0,
            price_vs_vah=0,
            price_vs_val=0,
            position_in_value="NO_DATA",
            bull_score=0,
            bear_score=0,
            signal=SignalState.NO_DATA,
            confidence=0,
            vwap=VWAPMetrics(current_price, current_price, current_price, current_price, current_price, 0, 0),
            notes=["Insufficient data for analysis"]
        )
    
    def _aggregate_analyses(self,
                            symbol: str,
                            scan_time: datetime,
                            analyses: Dict[Timeframe, TimeframeAnalysis]) -> ScanResult:
        """Aggregate timeframe analyses into final result"""
        
        # Count signals
        long_count = sum(1 for a in analyses.values() if a.signal == SignalState.LONG_SETUP)
        short_count = sum(1 for a in analyses.values() if a.signal == SignalState.SHORT_SETUP)
        yellow_count = sum(1 for a in analyses.values() if a.signal == SignalState.YELLOW)
        neutral_count = sum(1 for a in analyses.values() if a.signal in [SignalState.NEUTRAL, SignalState.NO_DATA])
        
        valid_analyses = [a for a in analyses.values() if a.signal != SignalState.NO_DATA]
        
        if not valid_analyses:
            return ScanResult(
                symbol=symbol,
                scan_time=scan_time,
                timeframe_analyses=analyses,
                dominant_signal=SignalState.NO_DATA,
                confluence_score=0,
                actionable=False,
                high_scenario_prob=0.33,
                low_scenario_prob=0.33,
                neutral_prob=0.34,
                summary="Insufficient data across all timeframes"
            )
        
        # Calculate aggregate scores
        avg_bull = np.mean([a.bull_score for a in valid_analyses])
        avg_bear = np.mean([a.bear_score for a in valid_analyses])
        
        # Confluence scoring
        total_valid = len(valid_analyses)
        if long_count > short_count and long_count >= total_valid / 2:
            dominant = SignalState.LONG_SETUP
            confluence = (long_count / total_valid) * 100
        elif short_count > long_count and short_count >= total_valid / 2:
            dominant = SignalState.SHORT_SETUP
            confluence = (short_count / total_valid) * 100
        elif yellow_count >= total_valid / 2:
            dominant = SignalState.YELLOW
            confluence = (yellow_count / total_valid) * 100
        else:
            dominant = SignalState.NEUTRAL
            confluence = 50
        
        # Scenario probabilities
        score_total = avg_bull + avg_bear + 1  # +1 to avoid division by zero
        high_prob = avg_bull / score_total
        low_prob = avg_bear / score_total
        neutral_prob = 1 - high_prob - low_prob
        neutral_prob = max(0, neutral_prob)  # Ensure non-negative
        
        # Normalize probabilities
        prob_total = high_prob + low_prob + neutral_prob
        if prob_total > 0:
            high_prob /= prob_total
            low_prob /= prob_total
            neutral_prob /= prob_total
        
        # Actionable determination
        actionable = (
            dominant in [SignalState.LONG_SETUP, SignalState.SHORT_SETUP] and
            confluence >= 50 and
            yellow_count < total_valid / 2
        )
        
        # Generate summary
        summary = self._generate_summary(
            dominant, confluence, long_count, short_count, yellow_count,
            avg_bull, avg_bear, high_prob, low_prob, actionable, valid_analyses
        )
        
        return ScanResult(
            symbol=symbol,
            scan_time=scan_time,
            timeframe_analyses=analyses,
            dominant_signal=dominant,
            confluence_score=confluence,
            actionable=actionable,
            high_scenario_prob=high_prob,
            low_scenario_prob=low_prob,
            neutral_prob=neutral_prob,
            summary=summary
        )
    
    def _generate_summary(self, dominant, confluence, long_count, short_count,
                          yellow_count, avg_bull, avg_bear, high_prob, low_prob,
                          actionable, valid_analyses) -> str:
        """Generate human-readable summary"""
        lines = []
        
        lines.append(f"Signal: {dominant.emoji} {dominant.value} ({confluence:.0f}% confluence)")
        lines.append(f"Timeframe votes: {long_count}L / {short_count}S / {yellow_count}Y")
        lines.append(f"Avg Scores: Bull {avg_bull:.1f} | Bear {avg_bear:.1f}")
        lines.append(f"Scenario Odds: HIGH {high_prob:.0%} | LOW {low_prob:.0%}")
        
        if actionable:
            lines.append("✅ ACTIONABLE SETUP")
        elif dominant == SignalState.YELLOW:
            lines.append("⚠️ WAIT FOR CLARITY")
        else:
            lines.append("⏸️ NO SETUP - STAND ASIDE")
        
        return "\n".join(lines)
    
    def print_report(self, result: ScanResult) -> str:
        """Generate formatted report string"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"MTF AUCTION SCAN: {result.symbol}")
        lines.append(f"Time: {result.scan_time}")
        lines.append("=" * 70)
        lines.append("")
        
        # Timeframe breakdown
        lines.append("TIMEFRAME BREAKDOWN:")
        lines.append("-" * 70)
        
        for tf in self.timeframes:
            if tf in result.timeframe_analyses:
                a = result.timeframe_analyses[tf]
                lines.append(f"\n{tf.label.upper():>8} | {a.signal.emoji} {a.signal.value:12} | "
                           f"Bull: {a.bull_score:5.1f} | Bear: {a.bear_score:5.1f} | "
                           f"Conf: {a.confidence:5.1f}%")
                lines.append(f"         | Position: {a.position_in_value:12} | "
                           f"RSI: {a.rsi.value:5.1f} ({a.rsi.zone}) | "
                           f"Flow: {a.flow.flow_imbalance:+.2f}")
                
                # Value area info
                vp = a.volume_profile
                lines.append(f"         | VAH: {vp.vah:.2f} | POC: {vp.poc:.2f} | VAL: {vp.val:.2f}")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("AGGREGATE ASSESSMENT:")
        lines.append("-" * 70)
        lines.append(result.summary)
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# DEMO / TEST
# =============================================================================

def generate_demo_data(days: int = 5, interval_minutes: int = 5) -> pd.DataFrame:
    """Generate realistic demo OHLCV data for testing"""
    np.random.seed(42)
    
    periods = days * 24 * 60 // interval_minutes
    
    # Start price
    price = 450.0
    
    data = []
    timestamp = datetime.now() - timedelta(days=days)
    
    for i in range(periods):
        # Add some trend and mean reversion
        trend = 0.0001 * np.sin(i / 100)  # Slow oscillation
        noise = np.random.randn() * 0.002
        
        # Volume varies by "time of day"
        hour = (i * interval_minutes // 60) % 24
        volume_mult = 1.5 if 9 <= hour <= 16 else 0.5  # Higher during market hours
        
        returns = trend + noise
        
        open_price = price
        close_price = price * (1 + returns)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.001)
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.001)
        volume = int(np.random.exponential(100000) * volume_mult)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
        timestamp += timedelta(minutes=interval_minutes)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


if __name__ == "__main__":
    # Demo
    print("Generating demo data...")
    df = generate_demo_data(days=10, interval_minutes=5)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print()
    
    # Run scanner
    scanner = MTFAuctionScanner()
    result = scanner.scan(df, symbol="DEMO")
    
    # Print report
    report = scanner.print_report(result)
    print(report)
    
    print("\n\nScenario Probabilities:")
    print(f"  HIGH scenario: {result.high_scenario_prob:.1%}")
    print(f"  LOW scenario:  {result.low_scenario_prob:.1%}")
    print(f"  NEUTRAL:       {result.neutral_prob:.1%}")
