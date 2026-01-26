"""
Compression Reversal Scanner
============================
Scans for compression reversal setups ideal for options trading.

Setup Criteria:
- Normal Volume Profile (football shape)
- Extreme compression (tight range, low ATR)
- Price at or approaching VAL
- RSI approaching 37 (oversold zone)
- Reversal candle confirmation

Trade Parameters:
- Entry: 0.65 delta calls, 3+ weeks expiration
- Stop: -12.5% on contract value
- Target: +1.5% price move, RSI > 72
- Lock: Sell 50% or buy ATM weekly put

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class ProfileShape(Enum):
    """Volume profile shape classification"""
    NORMAL = "NORMAL"           # Football shape - good for mean reversion
    INVERTED = "INVERTED"       # P or b shape - breakout play
    DOUBLE_DIST = "DOUBLE_DIST" # Two POCs - range bound
    FLAT = "FLAT"               # Low volume, unclear
    
    @property
    def emoji(self) -> str:
        return {
            "NORMAL": "🏈",
            "INVERTED": "📊",
            "DOUBLE_DIST": "📈📉",
            "FLAT": "➖"
        }[self.value]


class CompressionLevel(Enum):
    """Compression intensity"""
    EXTREME = "EXTREME"         # < 30% of normal ATR - imminent move
    HIGH = "HIGH"               # 30-50% of normal ATR
    MODERATE = "MODERATE"       # 50-70% of normal ATR  
    NORMAL = "NORMAL"           # 70-100% of normal ATR
    EXPANDED = "EXPANDED"       # > 100% of normal ATR
    
    @property
    def score(self) -> int:
        """Score contribution for setup quality"""
        return {
            "EXTREME": 25,
            "HIGH": 20,
            "MODERATE": 10,
            "NORMAL": 0,
            "EXPANDED": -10
        }[self.value]


class SetupQuality(Enum):
    """Overall setup quality rating"""
    A_PLUS = "A+"       # 90+ score - textbook setup
    A = "A"             # 80-89 - strong setup
    B = "B"             # 70-79 - good setup
    C = "C"             # 60-69 - marginal
    NO_SETUP = "NO"     # < 60 - criteria not met
    
    @property
    def tradeable(self) -> bool:
        return self in [SetupQuality.A_PLUS, SetupQuality.A, SetupQuality.B]


@dataclass
class CompressionMetrics:
    """Compression analysis results"""
    atr_14: float
    atr_5: float
    avg_atr_30: float           # 30-day average ATR for comparison
    compression_ratio: float    # Current ATR / Avg ATR (lower = more compressed)
    compression_level: CompressionLevel
    range_5d_pct: float         # 5-day range as % of price
    range_10d_pct: float        # 10-day range as % of price
    bollinger_width: float      # BB width (lower = squeezed)
    keltner_squeeze: bool       # Inside Keltner channel = squeeze


@dataclass
class ProfileAnalysis:
    """Volume profile analysis results"""
    poc: float
    vah: float
    val: float
    shape: ProfileShape
    value_area_width_pct: float  # VAH-VAL as % of price
    volume_distribution: str     # "balanced", "top_heavy", "bottom_heavy"
    poc_volume_pct: float        # % of volume at POC level


@dataclass
class RSIAnalysis:
    """RSI analysis for reversal detection"""
    current_rsi: float
    rsi_slope: float            # Rate of change
    in_reversal_zone: bool      # RSI 35-40
    oversold: bool              # RSI < 30
    distance_to_37: float       # How far from ideal entry
    rsi_divergence: bool        # Price lower low, RSI higher low


@dataclass
class ReversalCandle:
    """Reversal candle pattern analysis"""
    detected: bool
    pattern_type: str           # "hammer", "bullish_engulfing", "doji", "none"
    lower_wick_ratio: float     # Lower wick / total range
    body_ratio: float           # Body / total range
    confirmation_strength: str  # "strong", "moderate", "weak"


@dataclass
class OptionsParams:
    """Recommended options parameters"""
    direction: str              # "CALL" or "PUT"
    delta: float                # Recommended delta
    min_dte: int                # Minimum days to expiration
    stop_loss_pct: float        # Stop loss on contract value
    target_price_move_pct: float  # Target price move
    target_rsi: float           # RSI target for exit
    
    # Example calculations
    example_entry: float        # Example contract price
    example_stop: float         # Stop price
    example_max_loss: float     # Max loss in dollars


@dataclass
class CompressionReversalSetup:
    """Complete compression reversal setup analysis"""
    symbol: str
    analysis_time: datetime
    current_price: float
    
    # Component analyses
    profile: ProfileAnalysis
    compression: CompressionMetrics
    rsi: RSIAnalysis
    reversal_candle: ReversalCandle
    
    # Position relative to levels
    distance_to_val_pct: float
    distance_to_poc_pct: float
    at_val: bool                # Within 0.5% of VAL
    
    # Scoring
    setup_score: int            # 0-100
    setup_quality: SetupQuality
    criteria_met: Dict[str, bool]
    
    # Trade parameters
    options_params: OptionsParams
    
    # Key levels
    entry_zone: Tuple[float, float]  # (low, high) for entry
    stop_level: float           # Price-based stop reference
    target_1: float             # +1.5% target
    target_2: float             # Extended target (POC)
    
    # Notes
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        # Convert criteria_met to native Python bools
        criteria_native = {k: bool(v) for k, v in self.criteria_met.items()}
        
        return {
            'symbol': self.symbol,
            'analysis_time': self.analysis_time.isoformat(),
            'current_price': float(self.current_price),
            'setup_score': int(self.setup_score),
            'setup_quality': self.setup_quality.value,
            'tradeable': bool(self.setup_quality.tradeable),
            'criteria_met': criteria_native,
            'profile_shape': self.profile.shape.value,
            'compression_level': self.compression.compression_level.value,
            'compression_ratio': float(self.compression.compression_ratio),
            'rsi': float(self.rsi.current_rsi),
            'rsi_in_zone': bool(self.rsi.in_reversal_zone),
            'at_val': bool(self.at_val),
            'distance_to_val_pct': float(self.distance_to_val_pct),
            'reversal_candle': bool(self.reversal_candle.detected),
            'reversal_pattern': self.reversal_candle.pattern_type,
            'entry_zone': [float(x) for x in self.entry_zone],
            'stop_level': float(self.stop_level),
            'target_1': float(self.target_1),
            'target_2': float(self.target_2),
            'levels': {
                'val': float(self.profile.val),
                'poc': float(self.profile.poc),
                'vah': float(self.profile.vah)
            },
            'options': {
                'direction': self.options_params.direction,
                'delta': float(self.options_params.delta),
                'min_dte': int(self.options_params.min_dte),
                'stop_loss_pct': float(self.options_params.stop_loss_pct),
                'target_move_pct': float(self.options_params.target_price_move_pct)
            },
            'notes': self.notes
        }


# =============================================================================
# TECHNICAL ANALYSIS ENGINE
# =============================================================================

class CompressionAnalyzer:
    """
    Analyzes price compression and volatility squeeze conditions
    """
    
    def __init__(self, 
                 atr_period: int = 14,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 keltner_period: int = 20,
                 keltner_mult: float = 1.5):
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.keltner_period = keltner_period
        self.keltner_mult = keltner_mult
    
    def analyze(self, df: pd.DataFrame) -> CompressionMetrics:
        """
        Analyze compression state
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            CompressionMetrics
        """
        if len(df) < 30:
            return self._default_metrics()
        
        # Calculate ATR
        atr_14 = self._calculate_atr(df, 14)
        atr_5 = self._calculate_atr(df, 5)
        
        # Calculate 30-day average ATR for comparison
        atr_series = self._calculate_atr_series(df, 14)
        avg_atr_30 = atr_series.tail(30).mean() if len(atr_series) >= 30 else atr_series.mean()
        
        # Compression ratio (lower = more compressed)
        compression_ratio = atr_5 / avg_atr_30 if avg_atr_30 > 0 else 1.0
        
        # Determine compression level
        if compression_ratio < 0.30:
            compression_level = CompressionLevel.EXTREME
        elif compression_ratio < 0.50:
            compression_level = CompressionLevel.HIGH
        elif compression_ratio < 0.70:
            compression_level = CompressionLevel.MODERATE
        elif compression_ratio <= 1.0:
            compression_level = CompressionLevel.NORMAL
        else:
            compression_level = CompressionLevel.EXPANDED
        
        # Range calculations
        current_price = df['close'].iloc[-1]
        range_5d = df['high'].tail(5).max() - df['low'].tail(5).min()
        range_10d = df['high'].tail(10).max() - df['low'].tail(10).min()
        range_5d_pct = (range_5d / current_price) * 100
        range_10d_pct = (range_10d / current_price) * 100
        
        # Bollinger Band width
        bb_width = self._calculate_bb_width(df)
        
        # Keltner squeeze detection
        keltner_squeeze = self._detect_keltner_squeeze(df)
        
        return CompressionMetrics(
            atr_14=round(atr_14, 4),
            atr_5=round(atr_5, 4),
            avg_atr_30=round(avg_atr_30, 4),
            compression_ratio=round(compression_ratio, 3),
            compression_level=compression_level,
            range_5d_pct=round(range_5d_pct, 2),
            range_10d_pct=round(range_10d_pct, 2),
            bollinger_width=round(bb_width, 4),
            keltner_squeeze=keltner_squeeze
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate ATR"""
        if len(df) < period + 1:
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _calculate_atr_series(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR series"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_bb_width(self, df: pd.DataFrame) -> float:
        """Calculate Bollinger Band width (normalized)"""
        if len(df) < self.bb_period:
            return 0.0
        
        close = df['close']
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        
        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)
        
        width = (upper - lower) / sma  # Normalized width
        return width.iloc[-1] if not pd.isna(width.iloc[-1]) else 0.0
    
    def _detect_keltner_squeeze(self, df: pd.DataFrame) -> bool:
        """Detect if BB is inside Keltner Channel (squeeze)"""
        if len(df) < self.keltner_period:
            return False
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Bollinger Bands
        bb_sma = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_sma + (bb_std * self.bb_std)
        bb_lower = bb_sma - (bb_std * self.bb_std)
        
        # Keltner Channel (using ATR)
        atr = self._calculate_atr(df, self.keltner_period)
        kc_mid = close.rolling(window=self.keltner_period).mean()
        kc_upper = kc_mid.iloc[-1] + (atr * self.keltner_mult)
        kc_lower = kc_mid.iloc[-1] - (atr * self.keltner_mult)
        
        # Squeeze = BB inside KC
        return bb_lower.iloc[-1] > kc_lower and bb_upper.iloc[-1] < kc_upper
    
    def _default_metrics(self) -> CompressionMetrics:
        """Return default metrics when insufficient data"""
        return CompressionMetrics(
            atr_14=0.0,
            atr_5=0.0,
            avg_atr_30=0.0,
            compression_ratio=1.0,
            compression_level=CompressionLevel.NORMAL,
            range_5d_pct=0.0,
            range_10d_pct=0.0,
            bollinger_width=0.0,
            keltner_squeeze=False
        )


class VolumeProfileAnalyzer:
    """
    Analyzes volume profile shape and characteristics
    """
    
    def __init__(self, 
                 value_area_pct: float = 0.70,
                 num_bins: int = 50):
        self.value_area_pct = value_area_pct
        self.num_bins = num_bins
    
    def analyze(self, df: pd.DataFrame) -> ProfileAnalysis:
        """
        Analyze volume profile
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            ProfileAnalysis
        """
        if len(df) < 20:
            return self._default_profile(df)
        
        # Calculate VP levels
        poc, vah, val, volume_profile = self._calculate_volume_profile(df)
        
        # Determine shape
        shape = self._classify_shape(volume_profile, poc, vah, val, df)
        
        # Value area width
        current_price = df['close'].iloc[-1]
        va_width_pct = ((vah - val) / current_price) * 100
        
        # Volume distribution
        vol_dist = self._analyze_distribution(volume_profile, poc)
        
        # POC volume percentage
        poc_vol_pct = self._calculate_poc_volume_pct(volume_profile, poc)
        
        return ProfileAnalysis(
            poc=round(poc, 2),
            vah=round(vah, 2),
            val=round(val, 2),
            shape=shape,
            value_area_width_pct=round(va_width_pct, 2),
            volume_distribution=vol_dist,
            poc_volume_pct=round(poc_vol_pct, 2)
        )
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Tuple[float, float, float, np.ndarray]:
        """Calculate POC, VAH, VAL and return volume profile array"""
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max == price_min:
            return price_max, price_max, price_min, np.array([1.0])
        
        bin_size = (price_max - price_min) / self.num_bins
        bins = np.arange(price_min, price_max + bin_size, bin_size)
        
        volume_profile = np.zeros(len(bins) - 1)
        
        for _, row in df.iterrows():
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']
            
            for i in range(len(bins) - 1):
                bin_low = bins[i]
                bin_high = bins[i + 1]
                
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)
                
                if overlap_high > overlap_low:
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        volume_profile[i] += bar_volume * overlap_pct
        
        # POC
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area
        total_volume = volume_profile.sum()
        target_volume = total_volume * self.value_area_pct
        
        va_volume = volume_profile[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while va_volume < target_volume:
            expand_low = va_low_idx > 0
            expand_high = va_high_idx < len(volume_profile) - 1
            
            if not expand_low and not expand_high:
                break
            
            low_vol = volume_profile[va_low_idx - 1] if expand_low else 0
            high_vol = volume_profile[va_high_idx + 1] if expand_high else 0
            
            if low_vol >= high_vol and expand_low:
                va_low_idx -= 1
                va_volume += low_vol
            elif expand_high:
                va_high_idx += 1
                va_volume += high_vol
            elif expand_low:
                va_low_idx -= 1
                va_volume += low_vol
        
        val = bins[va_low_idx]
        vah = bins[va_high_idx + 1]
        
        return poc, vah, val, volume_profile
    
    def _classify_shape(self, volume_profile: np.ndarray, poc: float, 
                        vah: float, val: float, df: pd.DataFrame) -> ProfileShape:
        """
        Classify volume profile shape
        
        Normal (Football): POC near center, bell curve distribution
        Inverted (P/b): POC at extreme, skewed distribution
        Double Distribution: Two distinct peaks
        Flat: Low variance, no clear POC
        """
        if len(volume_profile) < 3:
            return ProfileShape.FLAT
        
        total_vol = volume_profile.sum()
        if total_vol == 0:
            return ProfileShape.FLAT
        
        # Normalize
        norm_profile = volume_profile / total_vol
        
        # Find peaks
        peaks = []
        for i in range(1, len(norm_profile) - 1):
            if norm_profile[i] > norm_profile[i-1] and norm_profile[i] > norm_profile[i+1]:
                peaks.append((i, norm_profile[i]))
        
        # Check for double distribution
        significant_peaks = [p for p in peaks if p[1] > 0.05]  # > 5% of volume
        if len(significant_peaks) >= 2:
            # Check if peaks are separated
            peak_indices = [p[0] for p in significant_peaks]
            if max(peak_indices) - min(peak_indices) > len(norm_profile) * 0.3:
                return ProfileShape.DOUBLE_DIST
        
        # Check POC position for inverted
        poc_position = np.argmax(volume_profile) / len(volume_profile)
        
        if poc_position < 0.25 or poc_position > 0.75:
            # POC at extreme = inverted
            return ProfileShape.INVERTED
        
        # Check for bell curve (normal)
        # Calculate skewness
        mid = len(norm_profile) // 2
        upper_vol = norm_profile[mid:].sum()
        lower_vol = norm_profile[:mid].sum()
        
        skew_ratio = upper_vol / lower_vol if lower_vol > 0 else 1.0
        
        if 0.7 < skew_ratio < 1.3:
            # Relatively balanced = normal
            return ProfileShape.NORMAL
        
        # Check variance
        variance = np.var(norm_profile)
        if variance < 0.001:
            return ProfileShape.FLAT
        
        return ProfileShape.NORMAL  # Default to normal
    
    def _analyze_distribution(self, volume_profile: np.ndarray, poc: float) -> str:
        """Analyze volume distribution around POC"""
        if len(volume_profile) < 3:
            return "balanced"
        
        poc_idx = np.argmax(volume_profile)
        upper_vol = volume_profile[poc_idx:].sum()
        lower_vol = volume_profile[:poc_idx+1].sum()
        total = upper_vol + lower_vol
        
        if total == 0:
            return "balanced"
        
        upper_pct = upper_vol / total
        
        if upper_pct > 0.6:
            return "top_heavy"
        elif upper_pct < 0.4:
            return "bottom_heavy"
        return "balanced"
    
    def _calculate_poc_volume_pct(self, volume_profile: np.ndarray, poc: float) -> float:
        """Calculate % of volume concentrated at POC level"""
        if len(volume_profile) == 0:
            return 0.0
        
        total_vol = volume_profile.sum()
        if total_vol == 0:
            return 0.0
        
        poc_idx = np.argmax(volume_profile)
        
        # Include adjacent bins
        poc_vol = volume_profile[poc_idx]
        if poc_idx > 0:
            poc_vol += volume_profile[poc_idx - 1] * 0.5
        if poc_idx < len(volume_profile) - 1:
            poc_vol += volume_profile[poc_idx + 1] * 0.5
        
        return (poc_vol / total_vol) * 100
    
    def _default_profile(self, df: pd.DataFrame) -> ProfileAnalysis:
        """Return default profile when insufficient data"""
        if len(df) > 0:
            mid = df['close'].mean()
            high = df['high'].max()
            low = df['low'].min()
        else:
            mid = high = low = 0.0
        
        return ProfileAnalysis(
            poc=mid,
            vah=high,
            val=low,
            shape=ProfileShape.FLAT,
            value_area_width_pct=0.0,
            volume_distribution="balanced",
            poc_volume_pct=0.0
        )


class RSIAnalyzer:
    """
    RSI analysis for reversal detection
    """
    
    def __init__(self, 
                 period: int = 14,
                 reversal_zone_low: float = 35,
                 reversal_zone_high: float = 40,
                 ideal_entry: float = 37):
        self.period = period
        self.reversal_zone_low = reversal_zone_low
        self.reversal_zone_high = reversal_zone_high
        self.ideal_entry = ideal_entry
    
    def analyze(self, df: pd.DataFrame) -> RSIAnalysis:
        """
        Analyze RSI for reversal setup
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            RSIAnalysis
        """
        if len(df) < self.period + 5:
            return self._default_analysis()
        
        # Calculate RSI
        rsi = self._calculate_rsi(df)
        current_rsi = rsi.iloc[-1]
        
        # RSI slope (rate of change)
        rsi_slope = rsi.iloc[-1] - rsi.iloc[-3] if len(rsi) >= 3 else 0
        
        # Zone checks
        in_reversal_zone = self.reversal_zone_low <= current_rsi <= self.reversal_zone_high
        oversold = current_rsi < 30
        
        # Distance to ideal entry
        distance_to_37 = abs(current_rsi - self.ideal_entry)
        
        # Check for bullish divergence
        rsi_divergence = self._check_bullish_divergence(df, rsi)
        
        return RSIAnalysis(
            current_rsi=round(current_rsi, 2),
            rsi_slope=round(rsi_slope, 2),
            in_reversal_zone=in_reversal_zone,
            oversold=oversold,
            distance_to_37=round(distance_to_37, 2),
            rsi_divergence=rsi_divergence
        )
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI using Wilder's smoothing"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        alpha = 1.0 / self.period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _check_bullish_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """Check for bullish RSI divergence (price LL, RSI HL)"""
        if len(df) < 10:
            return False
        
        # Find recent lows (last 10 bars)
        prices = df['low'].tail(10)
        rsi_vals = rsi.tail(10)
        
        # Look for lower low in price
        mid = len(prices) // 2
        first_half_low = prices.iloc[:mid].min()
        second_half_low = prices.iloc[mid:].min()
        
        first_half_rsi_low = rsi_vals.iloc[:mid].min()
        second_half_rsi_low = rsi_vals.iloc[mid:].min()
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        price_lower_low = second_half_low < first_half_low
        rsi_higher_low = second_half_rsi_low > first_half_rsi_low
        
        return price_lower_low and rsi_higher_low
    
    def _default_analysis(self) -> RSIAnalysis:
        """Return default analysis when insufficient data"""
        return RSIAnalysis(
            current_rsi=50.0,
            rsi_slope=0.0,
            in_reversal_zone=False,
            oversold=False,
            distance_to_37=13.0,
            rsi_divergence=False
        )


class ReversalCandleDetector:
    """
    Detects reversal candlestick patterns
    """
    
    def __init__(self,
                 min_wick_ratio: float = 0.5,
                 max_body_ratio: float = 0.4):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
    
    def analyze(self, df: pd.DataFrame) -> ReversalCandle:
        """
        Analyze for bullish reversal candle at support
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            ReversalCandle
        """
        if len(df) < 2:
            return self._default_candle()
        
        candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        high = candle['high']
        low = candle['low']
        open_price = candle['open']
        close = candle['close']
        
        candle_range = high - low
        if candle_range <= 0:
            return self._default_candle()
        
        body = abs(close - open_price)
        body_top = max(open_price, close)
        body_bottom = min(open_price, close)
        
        upper_wick = high - body_top
        lower_wick = body_bottom - low
        
        lower_wick_ratio = lower_wick / candle_range
        body_ratio = body / candle_range
        
        # Check for hammer pattern
        is_hammer = (
            lower_wick_ratio >= self.min_wick_ratio and
            body_ratio <= self.max_body_ratio and
            close > open_price  # Bullish close preferred
        )
        
        # Check for bullish engulfing
        is_engulfing = (
            prev_candle['close'] < prev_candle['open'] and  # Previous was bearish
            close > open_price and  # Current is bullish
            close > prev_candle['open'] and  # Body engulfs
            open_price < prev_candle['close']
        )
        
        # Check for doji at support
        is_doji = body_ratio < 0.1 and candle_range > 0
        
        # Determine pattern type
        if is_hammer:
            pattern_type = "hammer"
            detected = True
        elif is_engulfing:
            pattern_type = "bullish_engulfing"
            detected = True
        elif is_doji and lower_wick_ratio > 0.3:
            pattern_type = "doji"
            detected = True
        else:
            pattern_type = "none"
            detected = False
        
        # Confirmation strength
        if detected:
            if lower_wick_ratio >= 0.65 or is_engulfing:
                strength = "strong"
            elif lower_wick_ratio >= 0.5:
                strength = "moderate"
            else:
                strength = "weak"
        else:
            strength = "none"
        
        return ReversalCandle(
            detected=detected,
            pattern_type=pattern_type,
            lower_wick_ratio=round(lower_wick_ratio, 2),
            body_ratio=round(body_ratio, 2),
            confirmation_strength=strength
        )
    
    def _default_candle(self) -> ReversalCandle:
        """Return default when insufficient data"""
        return ReversalCandle(
            detected=False,
            pattern_type="none",
            lower_wick_ratio=0.0,
            body_ratio=0.0,
            confirmation_strength="none"
        )


# =============================================================================
# MAIN SCANNER
# =============================================================================

class CompressionReversalScanner:
    """
    Main scanner for compression reversal setups
    
    For Brokers:
    -----------
    This scanner identifies high-probability mean reversion setups where:
    1. Price is compressed (tight range, low volatility)
    2. Volume profile is normal (football shape = fair value)
    3. Price has pulled back to VAL (value area low)
    4. RSI is approaching oversold (37 zone)
    5. Reversal candle confirms buyers stepping in
    
    When all criteria align, enter CALLS with 0.65 delta, 3+ weeks out.
    Stop at -12.5% on contract. Target +1.5% price move.
    
    For Programmers:
    ---------------
    Orchestrates all analyzers and produces scored setup quality.
    """
    
    def __init__(self,
                 val_proximity_pct: float = 0.5,
                 rsi_target: float = 37,
                 rsi_tolerance: float = 5,
                 stop_loss_pct: float = 12.5,
                 target_move_pct: float = 1.5,
                 delta: float = 0.65,
                 min_dte: int = 21):
        
        self.val_proximity_pct = val_proximity_pct
        self.rsi_target = rsi_target
        self.rsi_tolerance = rsi_tolerance
        self.stop_loss_pct = stop_loss_pct
        self.target_move_pct = target_move_pct
        self.delta = delta
        self.min_dte = min_dte
        
        # Initialize analyzers
        self.compression_analyzer = CompressionAnalyzer()
        self.profile_analyzer = VolumeProfileAnalyzer()
        self.rsi_analyzer = RSIAnalyzer()
        self.candle_detector = ReversalCandleDetector()
    
    def scan(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> CompressionReversalSetup:
        """
        Scan for compression reversal setup
        
        Args:
            df: OHLCV DataFrame with sufficient history (30+ bars recommended)
            symbol: Ticker symbol
        
        Returns:
            CompressionReversalSetup with complete analysis
        """
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        
        # Run component analyses
        profile = self.profile_analyzer.analyze(df)
        compression = self.compression_analyzer.analyze(df)
        rsi = self.rsi_analyzer.analyze(df)
        reversal_candle = self.candle_detector.analyze(df)
        
        # Position relative to levels
        distance_to_val_pct = ((current_price - profile.val) / profile.val) * 100 if profile.val > 0 else 0
        distance_to_poc_pct = ((current_price - profile.poc) / profile.poc) * 100 if profile.poc > 0 else 0
        at_val = abs(distance_to_val_pct) <= self.val_proximity_pct
        
        # Check criteria
        criteria_met = {
            'normal_profile': profile.shape == ProfileShape.NORMAL,
            'compressed': compression.compression_level in [CompressionLevel.EXTREME, CompressionLevel.HIGH, CompressionLevel.MODERATE],
            'at_val': at_val or distance_to_val_pct < 0,  # At or below VAL
            'rsi_zone': rsi.in_reversal_zone or rsi.oversold or rsi.distance_to_37 <= self.rsi_tolerance,
            'reversal_candle': reversal_candle.detected
        }
        
        # Calculate score
        score = self._calculate_score(profile, compression, rsi, reversal_candle, criteria_met, distance_to_val_pct)
        
        # Determine quality
        quality = self._determine_quality(score)
        
        # Calculate levels
        entry_zone = (profile.val * 0.998, profile.val * 1.005)  # Tight zone around VAL
        atr = compression.atr_14
        stop_level = profile.val - (atr * 1.0)  # 1 ATR below VAL
        target_1 = current_price * (1 + self.target_move_pct / 100)
        target_2 = profile.poc
        
        # Options parameters
        example_entry = 4500  # Example contract price
        example_stop = example_entry * (1 - self.stop_loss_pct / 100)
        example_max_loss = example_entry - example_stop
        
        options_params = OptionsParams(
            direction="CALL",
            delta=self.delta,
            min_dte=self.min_dte,
            stop_loss_pct=self.stop_loss_pct,
            target_price_move_pct=self.target_move_pct,
            target_rsi=72.0,
            example_entry=example_entry,
            example_stop=round(example_stop, 2),
            example_max_loss=round(example_max_loss, 2)
        )
        
        # Generate notes
        notes = self._generate_notes(profile, compression, rsi, reversal_candle, criteria_met, quality)
        
        return CompressionReversalSetup(
            symbol=symbol,
            analysis_time=datetime.now(),
            current_price=round(current_price, 2),
            profile=profile,
            compression=compression,
            rsi=rsi,
            reversal_candle=reversal_candle,
            distance_to_val_pct=round(distance_to_val_pct, 2),
            distance_to_poc_pct=round(distance_to_poc_pct, 2),
            at_val=at_val,
            setup_score=score,
            setup_quality=quality,
            criteria_met=criteria_met,
            options_params=options_params,
            entry_zone=entry_zone,
            stop_level=round(stop_level, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            notes=notes
        )
    
    def _calculate_score(self, 
                         profile: ProfileAnalysis,
                         compression: CompressionMetrics,
                         rsi: RSIAnalysis,
                         candle: ReversalCandle,
                         criteria: Dict[str, bool],
                         dist_to_val: float) -> int:
        """Calculate setup score (0-100)"""
        score = 0
        
        # Profile shape (25 points max)
        if profile.shape == ProfileShape.NORMAL:
            score += 25
        elif profile.shape == ProfileShape.DOUBLE_DIST:
            score += 10
        
        # Compression (25 points max)
        score += compression.compression_level.score
        if compression.keltner_squeeze:
            score += 5  # Bonus for squeeze
        
        # RSI zone (25 points max)
        if rsi.in_reversal_zone:
            score += 25
        elif rsi.oversold:
            score += 20
        elif rsi.distance_to_37 <= 5:
            score += 15
        elif rsi.distance_to_37 <= 10:
            score += 10
        
        if rsi.rsi_divergence:
            score += 5  # Bonus for divergence
        
        # VAL proximity (15 points max)
        if dist_to_val < 0:  # Below VAL
            score += 15
        elif abs(dist_to_val) <= 0.5:
            score += 15
        elif abs(dist_to_val) <= 1.0:
            score += 10
        elif abs(dist_to_val) <= 2.0:
            score += 5
        
        # Reversal candle (10 points max)
        if candle.detected:
            if candle.confirmation_strength == "strong":
                score += 10
            elif candle.confirmation_strength == "moderate":
                score += 7
            else:
                score += 5
        
        return min(100, max(0, score))
    
    def _determine_quality(self, score: int) -> SetupQuality:
        """Determine setup quality from score"""
        if score >= 90:
            return SetupQuality.A_PLUS
        elif score >= 80:
            return SetupQuality.A
        elif score >= 70:
            return SetupQuality.B
        elif score >= 60:
            return SetupQuality.C
        else:
            return SetupQuality.NO_SETUP
    
    def _generate_notes(self,
                        profile: ProfileAnalysis,
                        compression: CompressionMetrics,
                        rsi: RSIAnalysis,
                        candle: ReversalCandle,
                        criteria: Dict[str, bool],
                        quality: SetupQuality) -> List[str]:
        """Generate actionable notes"""
        notes = []
        
        # Quality summary
        if quality.tradeable:
            notes.append(f"✅ {quality.value} SETUP - Tradeable")
        else:
            notes.append(f"⚠️ {quality.value} - Criteria not fully met")
        
        # Profile
        notes.append(f"Profile: {profile.shape.emoji} {profile.shape.value} ({profile.volume_distribution})")
        
        # Compression
        if compression.compression_level in [CompressionLevel.EXTREME, CompressionLevel.HIGH]:
            notes.append(f"🔥 {compression.compression_level.value} compression (ratio: {compression.compression_ratio})")
        if compression.keltner_squeeze:
            notes.append("📊 Keltner squeeze active")
        
        # RSI
        if rsi.in_reversal_zone:
            notes.append(f"✅ RSI in reversal zone: {rsi.current_rsi}")
        elif rsi.oversold:
            notes.append(f"⚡ RSI oversold: {rsi.current_rsi}")
        else:
            notes.append(f"RSI: {rsi.current_rsi} (target: ~37)")
        
        if rsi.rsi_divergence:
            notes.append("📈 Bullish RSI divergence detected")
        
        # Reversal candle
        if candle.detected:
            notes.append(f"🕯️ {candle.pattern_type.upper()} candle ({candle.confirmation_strength})")
        
        # Missing criteria
        missing = [k for k, v in criteria.items() if not v]
        if missing:
            notes.append(f"Missing: {', '.join(missing)}")
        
        return notes
    
    def scan_watchlist(self, 
                       symbols: List[str],
                       data_fetcher,
                       min_quality: SetupQuality = SetupQuality.B) -> List[CompressionReversalSetup]:
        """
        Scan a list of symbols for setups
        
        Args:
            symbols: List of ticker symbols
            data_fetcher: Object with fetch(symbol, days, interval) method
            min_quality: Minimum quality to include in results
        
        Returns:
            List of setups meeting minimum quality
        """
        setups = []
        
        for symbol in symbols:
            try:
                # Fetch data (need enough for VP and compression analysis)
                df = data_fetcher.fetch(symbol, days=30, interval='1h')
                
                if df is None or len(df) < 50:
                    continue
                
                setup = self.scan(df, symbol=symbol)
                
                if setup.setup_quality.tradeable and self._quality_meets_minimum(setup.setup_quality, min_quality):
                    setups.append(setup)
            
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by score (highest first)
        setups.sort(key=lambda x: x.setup_score, reverse=True)
        
        return setups
    
    def _quality_meets_minimum(self, quality: SetupQuality, minimum: SetupQuality) -> bool:
        """Check if quality meets minimum threshold"""
        quality_order = [SetupQuality.NO_SETUP, SetupQuality.C, SetupQuality.B, SetupQuality.A, SetupQuality.A_PLUS]
        return quality_order.index(quality) >= quality_order.index(minimum)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_scan(df: pd.DataFrame, symbol: str = "UNKNOWN") -> dict:
    """
    Quick scan returning dictionary result
    
    Args:
        df: OHLCV DataFrame
        symbol: Ticker symbol
    
    Returns:
        Dictionary with setup details
    """
    scanner = CompressionReversalScanner()
    setup = scanner.scan(df, symbol)
    return setup.to_dict()


def format_setup_alert(setup: CompressionReversalSetup) -> str:
    """
    Format setup as alert message
    
    Args:
        setup: CompressionReversalSetup
    
    Returns:
        Formatted alert string
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f"🎯 COMPRESSION REVERSAL: {setup.symbol}")
    lines.append(f"Quality: {setup.setup_quality.value} (Score: {setup.setup_score})")
    lines.append("=" * 50)
    
    lines.append(f"\n📊 Current Price: ${setup.current_price}")
    lines.append(f"Profile: {setup.profile.shape.value} | VAL: ${setup.profile.val} | POC: ${setup.profile.poc}")
    lines.append(f"RSI: {setup.rsi.current_rsi} | Compression: {setup.compression.compression_level.value}")
    
    if setup.reversal_candle.detected:
        lines.append(f"🕯️ Reversal: {setup.reversal_candle.pattern_type} ({setup.reversal_candle.confirmation_strength})")
    
    lines.append(f"\n💰 OPTIONS TRADE:")
    lines.append(f"   Direction: {setup.options_params.direction}")
    lines.append(f"   Delta: {setup.options_params.delta}")
    lines.append(f"   Min DTE: {setup.options_params.min_dte} days")
    lines.append(f"   Stop: -{setup.options_params.stop_loss_pct}% on contract")
    
    lines.append(f"\n🎯 LEVELS:")
    lines.append(f"   Entry Zone: ${setup.entry_zone[0]:.2f} - ${setup.entry_zone[1]:.2f}")
    lines.append(f"   Stop: ${setup.stop_level}")
    lines.append(f"   Target 1: ${setup.target_1} (+{setup.options_params.target_price_move_pct}%)")
    lines.append(f"   Target 2: ${setup.target_2} (POC)")
    
    lines.append(f"\n📝 NOTES:")
    for note in setup.notes:
        lines.append(f"   {note}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


# =============================================================================
# MAIN - Demo/Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPRESSION REVERSAL SCANNER - Demo")
    print("=" * 60)
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
    
    # Simulate compressed price action near VAL
    base_price = 150
    prices = [base_price]
    for i in range(99):
        # Small random moves (compressed)
        change = np.random.normal(0, 0.3)
        prices.append(prices[-1] + change)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
        'close': [p + np.random.normal(0, 0.2) for p in prices],
        'volume': [np.random.randint(100000, 500000) for _ in prices]
    }, index=dates)
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Run scan
    scanner = CompressionReversalScanner()
    setup = scanner.scan(df, symbol="DEMO")
    
    # Print results
    print(format_setup_alert(setup))
    
    print("\n\n📋 Full Setup Dictionary:")
    import json
    print(json.dumps(setup.to_dict(), indent=2))
