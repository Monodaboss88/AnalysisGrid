"""
S.E.F. Extension Duration Predictor V2 — C.O.R.E. Methodology
================================================================
THE KEY EDGE: It's not just WHERE price is, but HOW LONG it's been there.

The rubber band principle — the longer price stays extended from fair value,
the harder and faster it snaps back.

RETAINED FROM V1:
- Duration-based snap-back probability tables
- Extension zone classification (VWAP/VP levels)
- Streak tracking with candle counts
- Rejection candle detection (wicks pointing toward value)
- Declining volume on extension
- Trend context (with/counter/neutral)
- Session context (open/mid/close)
- Prior snap-back history tracking

NEW IN V2:
- Self-Contained Analysis — just pass a symbol, V2 calculates VP levels internally
- Volume Profile Calculation — POC/VAH/VAL computed from price data
- Weekly Structure — MTF trend context (pullback in uptrend vs breakdown)
- Wilder's RSI — consistent EMA-based RSI across all V2 scanners
- Squeeze Co-Detection — extension + squeeze = spring-loaded snap-back
- IV Percentile — options pricing context for position sizing
- Dynamic Timeframe — auto-detects candle spacing, no hardcoded 2H assumption
- Enhanced Trend Analysis — uses SMA slope + higher-high/lower-low structure
- Setup Classification — labels setup type with entry triggers
- Quality Grades — A+ through F grading system
- Batch Scanning — scan_symbols() for watchlist sweeps
- Multi-Level Tracking — tracks extension from ALL VP levels simultaneously
- Snap-Back Target Refinement — targets VP levels, not just the crossed level

Tiers:
- WATCHING:   1 candle extended — 45% snap-back
- ALERT:      2 candles — 55% snap-back
- HIGH_PROB:  3 candles — 65% snap-back 🔥
- EXTREME:    4+ candles — 75%+ snap-back 💥

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class TriggerLevel(Enum):
    """Alert levels based on extension duration"""
    NONE = 0
    WATCHING = 1      # 1 candle - just observing
    ALERT = 2         # 2 candles - prepare
    HIGH_PROB = 3     # 3 candles - look for entry
    EXTREME = 4       # 4+ candles - high conviction


class ExtensionZone(Enum):
    """Where price is relative to fair value"""
    EXTREME_ABOVE = "extreme_above"    # > 2 ATR above VWAP
    ABOVE_VALUE = "above_value"        # Above VAH or > 1 ATR above VWAP
    IN_VALUE = "in_value"              # Between VAL and VAH
    BELOW_VALUE = "below_value"        # Below VAL or > 1 ATR below VWAP
    EXTREME_BELOW = "extreme_below"    # > 2 ATR below VWAP


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VolumeProfileLevels:
    """Volume profile levels"""
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    vwap: float = 0.0
    price_zone: str = "in_value"
    at_key_level: bool = False
    nearest_level: str = ""
    nearest_level_price: float = 0.0
    distance_to_nearest_pct: float = 0.0


@dataclass
class WeeklyContext:
    """Weekly structure for MTF alignment"""
    trend: str = "NEUTRAL"
    last_week_structure: str = ""
    weekly_close_position: float = 0.5
    weekly_close_signal: str = ""
    supports_long: bool = False
    supports_short: bool = False


@dataclass
class SqueezeContext:
    """Is squeeze active alongside the extension?"""
    is_squeezed: bool = False
    squeeze_days: int = 0
    bb_width_percentile: float = 50.0
    spring_loaded: bool = False  # Extension + squeeze = loaded spring


@dataclass
class OptionsContext:
    """Options-specific context"""
    iv_percentile: float = 50.0
    iv_regime: str = "normal"     # low, normal, elevated, extreme
    suggested_delta: float = 0.65
    min_dte: int = 21
    contract_stop_pct: float = 12.5
    entry_size: str = "50%"
    scale_plan: str = "Enter 50%, add at +15% and +25%"


@dataclass
class TrendAnalysis:
    """Enhanced trend analysis"""
    direction: str = "neutral"          # uptrend, downtrend, neutral
    sma_20_slope: float = 0.0           # Slope of 20 SMA
    sma_50_slope: float = 0.0
    higher_highs: int = 0               # Count of HH in lookback
    lower_lows: int = 0                 # Count of LL in lookback
    extension_vs_trend: str = "neutral" # with_trend, counter_trend, neutral
    strength: str = "weak"              # weak, moderate, strong


@dataclass
class CandleData:
    """Single candle with extension metrics"""
    timestamp: datetime = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    
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
        
        if self.atr > 0:
            self.distance_from_vwap_atr = (mid - self.vwap) / self.atr
        
        self.zone = self._get_zone(mid)
        self._analyze_structure()
    
    def _get_zone(self, price: float) -> ExtensionZone:
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
        range_ = self.high - self.low
        if range_ == 0:
            return
        
        upper_wick = self.high - max(self.open, self.close)
        lower_wick = min(self.open, self.close) - self.low
        
        if self.zone in [ExtensionZone.ABOVE_VALUE, ExtensionZone.EXTREME_ABOVE]:
            self.is_rejection = (upper_wick / range_) > 0.5
            self.is_continuation = self.close > self.open and (lower_wick / range_) < 0.2
        elif self.zone in [ExtensionZone.BELOW_VALUE, ExtensionZone.EXTREME_BELOW]:
            self.is_rejection = (lower_wick / range_) > 0.5
            self.is_continuation = self.close < self.open and (upper_wick / range_) < 0.2


@dataclass
class ExtensionStreak:
    """Tracks consecutive candles in extension"""
    level_name: str
    direction: str
    candles: List[CandleData] = field(default_factory=list)
    candle_minutes: int = 120  # NEW: dynamic instead of hardcoded
    
    @property
    def count(self) -> int:
        return len(self.candles)
    
    @property
    def hours(self) -> float:
        return self.count * self.candle_minutes / 60
    
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
        return vols[-1] < vols[0] * 0.8


@dataclass
class ExtensionAnalysis:
    """Complete extension analysis for a symbol — V2"""
    symbol: str = ""
    current_price: float = 0.0
    timestamp: str = ""
    
    # Where price is right now
    zone: str = "in_value"
    distance_from_vwap_atr: float = 0.0
    
    # Active extension streaks
    active_streaks: Dict = field(default_factory=dict)
    hottest_streak: Dict = field(default_factory=dict)
    
    # Best trade setup (if any)
    trade_direction: str = ""          # LONG or SHORT
    snap_back_target: float = 0.0
    stop_loss: float = 0.0
    snap_back_probability: float = 0.0
    risk_reward: float = 0.0
    
    # Scoring
    extension_score: int = 0           # 0-100
    quality_grade: str = "F"           # A+ through F
    trigger_level: str = "NONE"
    
    # V2 Context
    volume_profile: VolumeProfileLevels = None
    weekly: WeeklyContext = None
    squeeze: SqueezeContext = None
    trend: TrendAnalysis = None
    options: OptionsContext = None
    
    # RSI (Wilder's)
    rsi: float = 50.0
    rsi_extreme: bool = False
    
    # Setup classification
    setup_type: str = ""               # extension_snap_long, spring_snap_short, etc.
    entry_trigger: str = ""
    factors: List[str] = field(default_factory=list)


# =============================================================================
# EXTENSION PREDICTOR V2
# =============================================================================

class ExtensionPredictorV2:
    """
    Enhanced extension duration predictor with C.O.R.E. methodology.
    
    Now self-contained: just pass a symbol and DataFrame, V2 calculates
    VP levels, weekly context, squeeze state, and everything else internally.
    
    Scoring (max ~160 pts, normalized to 100):
    - Extension Duration:     0-30 pts
    - Extension Distance:     0-20 pts
    - Rejection Candle:       0-15 pts
    - Volume Declining:       0-10 pts
    - RSI Confluence:         0-15 pts (enhanced with Wilder's)
    - Volume Profile Zone:    0-15 pts (NEW)
    - Weekly Alignment:       0-10 pts (NEW)
    - Squeeze Co-Detection:   0-10 pts (NEW)
    - Trend Context:          0-10 pts (NEW)
    - Session Context:        0-10 pts
    - IV Percentile:          0-10 pts (NEW)
    - Prior Snap-Backs:       0-5 pts
    """
    
    # Snap-back probabilities by candle count
    BASE_PROBABILITIES = {
        0: 0.40, 1: 0.45, 2: 0.55, 3: 0.65, 4: 0.75,
        5: 0.80, 6: 0.83, 7: 0.86, 8: 0.88, 9: 0.90, 10: 0.92,
    }
    
    # Probability adjustments
    REJECTION_BONUS = 0.10
    DECLINING_VOLUME_BONUS = 0.05
    EXTREME_EXTENSION_BONUS = 0.05
    COUNTER_TREND_BONUS = 0.08
    RSI_EXTREME_BONUS = 0.07
    SESSION_CLOSE_BONUS = 0.05
    PRIOR_SNAP_BACK_BONUS = 0.03
    SQUEEZE_BONUS = 0.06
    VP_KEY_LEVEL_BONUS = 0.05
    WEEKLY_ALIGNMENT_BONUS = 0.05
    
    def __init__(self):
        # Internal state
        self._streaks: Dict[str, Dict[str, ExtensionStreak]] = {}
        self._history: Dict[str, List[CandleData]] = {}
        self._snap_back_history: Dict[str, Dict[str, int]] = {}
        
        # VP parameters
        self.vp_num_bins = 50
        self.vp_value_area_pct = 0.70
        
        # Squeeze parameters
        self.bb_period = 20
        self.bb_std = 2.0
        self.kc_period = 20
        self.kc_mult = 1.5
    
    # =========================================================================
    # TECHNICAL CALCULATIONS
    # =========================================================================
    
    def _calculate_rsi_wilder(self, closes: pd.Series, period: int = 14) -> pd.Series:
        """RSI with Wilder's smoothing — consistent across all V2 scanners"""
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        val = float(atr.iloc[-1])
        return val if not pd.isna(val) else float(high_low.mean())
    
    def _calculate_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return float(df['close'].iloc[-1])
        typical = (df['high'] + df['low'] + df['close']) / 3
        vol_sum = df['volume'].sum()
        if vol_sum == 0:
            return float(df['close'].iloc[-1])
        return float((typical * df['volume']).sum() / vol_sum)
    
    def _detect_timeframe(self, df: pd.DataFrame) -> Tuple[str, int]:
        """
        Auto-detect timeframe and return (label, minutes_per_bar).
        No more hardcoded 2H assumption.
        """
        if len(df) < 3:
            return "unknown", 120
        try:
            diffs = pd.Series(df.index).diff().dropna()
            median_mins = diffs.median().total_seconds() / 60
            
            if median_mins < 10:
                return "5m", 5
            elif median_mins < 20:
                return "15m", 15
            elif median_mins < 45:
                return "30m", 30
            elif median_mins < 90:
                return "1h", 60
            elif median_mins < 180:
                return "2h", 120
            elif median_mins < 360:
                return "4h", 240
            elif median_mins < 1500:
                return "1d", 390  # Trading day ~6.5 hours
            else:
                return "1wk", 1950
        except:
            return "1d", 390
    
    # =========================================================================
    # VOLUME PROFILE
    # =========================================================================
    
    def _calculate_volume_profile(self, df: pd.DataFrame,
                                   current_price: float) -> VolumeProfileLevels:
        """Calculate POC/VAH/VAL from price data"""
        if len(df) < 10:
            return VolumeProfileLevels()
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        if price_max == price_min:
            return VolumeProfileLevels(poc=price_max, vah=price_max, val=price_min)
        
        bin_size = (price_max - price_min) / self.vp_num_bins
        bins = np.arange(price_min, price_max + bin_size, bin_size)
        volume_profile = np.zeros(len(bins) - 1)
        
        for _, row in df.iterrows():
            bar_low, bar_high, bar_vol = row['low'], row['high'], row['volume']
            for i in range(len(bins) - 1):
                overlap_low = max(bar_low, bins[i])
                overlap_high = min(bar_high, bins[i + 1])
                if overlap_high > overlap_low:
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        volume_profile[i] += bar_vol * (overlap_high - overlap_low) / bar_range
        
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        total_vol = volume_profile.sum()
        target_vol = total_vol * self.vp_value_area_pct
        va_vol = volume_profile[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while va_vol < target_vol:
            expand_low = va_low_idx > 0
            expand_high = va_high_idx < len(volume_profile) - 1
            if not expand_low and not expand_high:
                break
            low_v = volume_profile[va_low_idx - 1] if expand_low else 0
            high_v = volume_profile[va_high_idx + 1] if expand_high else 0
            if low_v >= high_v and expand_low:
                va_low_idx -= 1
                va_vol += low_v
            elif expand_high:
                va_high_idx += 1
                va_vol += high_v
            elif expand_low:
                va_low_idx -= 1
                va_vol += low_v
        
        val = bins[va_low_idx]
        vah = bins[va_high_idx + 1]
        
        # VWAP
        vwap = self._calculate_vwap(df)
        
        # Price zone
        if current_price > vah * 1.005:
            zone = "above_vah"
        elif current_price >= poc * 1.005:
            zone = "vah_poc"
        elif abs(current_price - poc) / poc < 0.005:
            zone = "at_poc"
        elif current_price >= val * 0.995:
            zone = "poc_val"
        else:
            zone = "below_val"
        
        # Nearest level
        levels = {"VAL": val, "POC": poc, "VAH": vah}
        nearest = min(levels, key=lambda k: abs(current_price - levels[k]))
        nearest_price = levels[nearest]
        dist = abs(current_price - nearest_price) / current_price * 100
        at_key = dist < 1.5
        
        return VolumeProfileLevels(
            poc=round(poc, 2), vah=round(vah, 2), val=round(val, 2),
            vwap=round(vwap, 2), price_zone=zone, at_key_level=at_key,
            nearest_level=nearest, nearest_level_price=round(nearest_price, 2),
            distance_to_nearest_pct=round(dist, 2)
        )
    
    # =========================================================================
    # WEEKLY STRUCTURE
    # =========================================================================
    
    def _calculate_weekly_context(self, symbol: str) -> WeeklyContext:
        """Fetch weekly data and classify trend"""
        try:
            from polygon_data import get_bars
            df_w = get_bars(symbol, period="6mo", interval="1wk")
            
            if df_w.empty or len(df_w) < 6:
                return WeeklyContext()
            
            df_w.columns = [c.lower() for c in df_w.columns]
            weeks = df_w.tail(8)
            
            ll_count = hh_count = lh_count = hl_count = 0
            last_structure = ""
            
            for i in range(1, len(weeks)):
                curr, prev = weeks.iloc[i], weeks.iloc[i - 1]
                structure = ""
                
                if curr['high'] > prev['high'] * 1.001:
                    structure += "HH"; hh_count += 1
                elif curr['high'] < prev['high'] * 0.999:
                    structure += "LH"; lh_count += 1
                else:
                    structure += "EQ"
                
                if curr['low'] > prev['low'] * 1.001:
                    structure += "+HL"; hl_count += 1
                elif curr['low'] < prev['low'] * 0.999:
                    structure += "+LL"; ll_count += 1
                else:
                    structure += "+EQ"
                
                last_structure = structure
            
            bearish = ll_count + lh_count
            bullish = hh_count + hl_count
            
            if bearish >= 8 and bullish <= 2:
                trend = "STRONG_DOWNTREND"
            elif bearish >= 5 and bearish > bullish * 2:
                trend = "DOWNTREND"
            elif bullish >= 8 and bearish <= 2:
                trend = "STRONG_UPTREND"
            elif bullish >= 5 and bullish > bearish * 2:
                trend = "UPTREND"
            else:
                trend = "NEUTRAL"
            
            lw = weeks.iloc[-2]
            lw_range = lw['high'] - lw['low']
            wcp = (lw['close'] - lw['low']) / lw_range if lw_range > 0 else 0.5
            
            signal = ""
            if "LL" in last_structure and wcp > 0.70:
                signal = "BULLISH_REVERSAL"
            elif "HH" in last_structure and wcp < 0.30:
                signal = "BEARISH_REVERSAL"
            elif wcp > 0.75:
                signal = "STRONG_BULL_CLOSE"
            elif wcp < 0.25:
                signal = "STRONG_BEAR_CLOSE"
            
            supports_long = (
                trend in ("UPTREND", "STRONG_UPTREND") or
                signal in ("BULLISH_REVERSAL", "STRONG_BULL_CLOSE") or
                (trend == "NEUTRAL" and wcp > 0.6)
            )
            supports_short = (
                trend in ("DOWNTREND", "STRONG_DOWNTREND") or
                signal in ("BEARISH_REVERSAL", "STRONG_BEAR_CLOSE") or
                (trend == "NEUTRAL" and wcp < 0.4)
            )
            
            return WeeklyContext(
                trend=trend, last_week_structure=last_structure,
                weekly_close_position=round(wcp, 2),
                weekly_close_signal=signal,
                supports_long=supports_long, supports_short=supports_short
            )
        except Exception as e:
            print(f"Weekly context error for {symbol}: {e}")
            return WeeklyContext()
    
    # =========================================================================
    # SQUEEZE CO-DETECTION
    # =========================================================================
    
    def _detect_squeeze(self, df: pd.DataFrame) -> SqueezeContext:
        """Check for BB inside KC alongside the extension"""
        if len(df) < self.bb_period + 5:
            return SqueezeContext()
        
        sma = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std()
        bb_upper = sma + (self.bb_std * std)
        bb_lower = sma - (self.bb_std * std)
        
        ema = df['close'].ewm(span=self.kc_period, adjust=False).mean()
        atr = self._calculate_atr_series(df, self.kc_period)
        kc_upper = ema + (self.kc_mult * atr)
        kc_lower = ema - (self.kc_mult * atr)
        
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        is_squeezed = bool(squeeze.iloc[-1]) if len(squeeze) > 0 else False
        
        squeeze_days = 0
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]:
                squeeze_days += 1
            else:
                break
        
        bb_width = (bb_upper - bb_lower) / sma
        bb_width = bb_width.dropna()
        width_pct = 50.0
        if len(bb_width) >= 20:
            current_w = bb_width.iloc[-1]
            lookback = bb_width.tail(60) if len(bb_width) >= 60 else bb_width
            width_pct = float((lookback < current_w).sum() / len(lookback) * 100)
        
        return SqueezeContext(
            is_squeezed=is_squeezed, squeeze_days=squeeze_days,
            bb_width_percentile=round(width_pct, 1),
            spring_loaded=False  # Set later based on extension state
        )
    
    # =========================================================================
    # IV PERCENTILE
    # =========================================================================
    
    def _estimate_iv_percentile(self, df: pd.DataFrame) -> OptionsContext:
        """IV regime from historical volatility proxy"""
        options = OptionsContext()
        if len(df) < 60:
            return options
        
        log_returns = np.log(df['close'] / df['close'].shift(1))
        hv_series = log_returns.rolling(window=20).std() * np.sqrt(252) * 100
        hv_series = hv_series.dropna()
        
        if len(hv_series) < 20:
            return options
        
        current_hv = float(hv_series.iloc[-1])
        lookback = hv_series.tail(60)
        percentile = float((lookback < current_hv).sum() / len(lookback) * 100)
        
        if percentile < 20:
            regime = "low"
        elif percentile < 50:
            regime = "normal"
        elif percentile < 80:
            regime = "elevated"
        else:
            regime = "extreme"
        
        options.iv_percentile = round(percentile, 1)
        options.iv_regime = regime
        
        if regime == "extreme":
            options.suggested_delta = 0.60
            options.entry_size = "25%"
            options.scale_plan = "Scale in as IV normalizes, add at +15%"
        elif regime == "elevated":
            options.suggested_delta = 0.65
            options.entry_size = "40%"
            options.scale_plan = "Enter 40%, add at +15% and +25%"
        elif regime == "low":
            options.suggested_delta = 0.70
            options.entry_size = "50%"
            options.scale_plan = "Enter 50%, add at +15% and +25% — cheap options"
        else:
            options.suggested_delta = 0.65
            options.entry_size = "50%"
            options.scale_plan = "Enter 50%, add at +15% and +25%"
        
        return options
    
    # =========================================================================
    # ENHANCED TREND ANALYSIS
    # =========================================================================
    
    def _analyze_trend(self, df: pd.DataFrame, extension_direction: str) -> TrendAnalysis:
        """
        Enhanced trend analysis using SMA slopes + HH/HL structure.
        Replaces V1's primitive 20-candle price comparison.
        """
        result = TrendAnalysis()
        
        if len(df) < 25:
            return result
        
        # SMA slopes
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean() if len(df) >= 50 else sma_20
        
        if len(sma_20.dropna()) >= 5:
            recent_sma = sma_20.dropna().tail(5)
            result.sma_20_slope = float((recent_sma.iloc[-1] - recent_sma.iloc[0]) / recent_sma.iloc[0] * 100)
        
        if len(sma_50.dropna()) >= 5:
            recent_sma50 = sma_50.dropna().tail(5)
            result.sma_50_slope = float((recent_sma50.iloc[-1] - recent_sma50.iloc[0]) / recent_sma50.iloc[0] * 100)
        
        # Higher highs / lower lows structure (5-bar pivots)
        highs = df['high'].tail(20).values
        lows = df['low'].tail(20).values
        
        hh = ll = 0
        for i in range(5, len(highs), 5):
            if highs[i] > highs[max(0, i - 5)]:
                hh += 1
            if lows[i] < lows[max(0, i - 5)]:
                ll += 1
        
        result.higher_highs = hh
        result.lower_lows = ll
        
        # Classify trend
        if result.sma_20_slope > 0.5 and hh >= 2:
            result.direction = "uptrend"
            result.strength = "strong" if result.sma_20_slope > 1.0 else "moderate"
        elif result.sma_20_slope < -0.5 and ll >= 2:
            result.direction = "downtrend"
            result.strength = "strong" if result.sma_20_slope < -1.0 else "moderate"
        else:
            result.direction = "neutral"
            result.strength = "weak"
        
        # Extension vs trend
        if extension_direction == "above":
            if result.direction == "downtrend":
                result.extension_vs_trend = "counter_trend"
            elif result.direction == "uptrend":
                result.extension_vs_trend = "with_trend"
            else:
                result.extension_vs_trend = "neutral"
        elif extension_direction == "below":
            if result.direction == "uptrend":
                result.extension_vs_trend = "counter_trend"
            elif result.direction == "downtrend":
                result.extension_vs_trend = "with_trend"
            else:
                result.extension_vs_trend = "neutral"
        
        return result
    
    # =========================================================================
    # STREAK TRACKING (from V1, enhanced)
    # =========================================================================
    
    def _track_streaks(self, df: pd.DataFrame,
                        vp: VolumeProfileLevels,
                        symbol: str,
                        candle_minutes: int) -> Dict[str, ExtensionStreak]:
        """
        Track extension streaks from VP levels.
        Processes the tail of the DataFrame to find active streaks.
        """
        atr = self._calculate_atr(df)
        
        if symbol not in self._streaks:
            self._streaks[symbol] = {}
        
        # Clear and re-track from recent data
        self._streaks[symbol] = {}
        
        levels = [
            ("vwap", vp.vwap),
            ("poc", vp.poc),
            ("vah", vp.vah),
            ("val", vp.val),
        ]
        
        # Process last 15 bars to find active streaks
        recent = df.tail(15)
        
        for level_name, level_price in levels:
            if level_price <= 0:
                continue
            
            # Track streaks from this level
            above_streak = ExtensionStreak(level_name, "above", candle_minutes=candle_minutes)
            below_streak = ExtensionStreak(level_name, "below", candle_minutes=candle_minutes)
            
            # Walk forward through recent bars
            for idx in range(len(recent)):
                row = recent.iloc[idx]
                price = row['close']
                
                candle = CandleData(
                    timestamp=row.name if hasattr(row.name, 'isoformat') else datetime.now(),
                    open=float(row['open']), high=float(row['high']),
                    low=float(row['low']), close=float(row['close']),
                    volume=int(row['volume']),
                    vwap=vp.vwap, poc=vp.poc, vah=vp.vah, val=vp.val,
                    atr=atr
                )
                candle.analyze()
                
                # Check extension
                threshold = atr * 0.5
                
                if price > level_price + threshold:
                    above_streak.candles.append(candle)
                    below_streak = ExtensionStreak(level_name, "below", candle_minutes=candle_minutes)
                elif price < level_price - threshold:
                    below_streak.candles.append(candle)
                    above_streak = ExtensionStreak(level_name, "above", candle_minutes=candle_minutes)
                else:
                    # Back in range — reset both
                    above_streak = ExtensionStreak(level_name, "above", candle_minutes=candle_minutes)
                    below_streak = ExtensionStreak(level_name, "below", candle_minutes=candle_minutes)
            
            # Store active streaks
            if above_streak.count >= 1:
                key = f"{level_name}_above"
                self._streaks[symbol][key] = above_streak
            if below_streak.count >= 1:
                key = f"{level_name}_below"
                self._streaks[symbol][key] = below_streak
        
        return self._streaks.get(symbol, {})
    
    # =========================================================================
    # QUALITY GRADE
    # =========================================================================
    
    def _quality_grade(self, score: int) -> str:
        if score >= 90: return "A+"
        elif score >= 80: return "A"
        elif score >= 70: return "B"
        elif score >= 60: return "C"
        elif score >= 50: return "D"
        else: return "F"
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> ExtensionAnalysis:
        """
        Full self-contained extension analysis.
        
        Args:
            df: DataFrame with OHLCV data (any timeframe)
            symbol: Stock symbol for weekly context
        
        Returns:
            ExtensionAnalysis with all V2 context
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if len(df) < 20:
            return ExtensionAnalysis(symbol=symbol)
        
        current_price = float(df['close'].iloc[-1])
        factors = []
        
        # Detect timeframe
        tf_label, candle_minutes = self._detect_timeframe(df)
        
        # =====================================================================
        # CORE CALCULATIONS
        # =====================================================================
        
        # 1. Volume Profile (30-bar tactical VP)
        vp_lookback = min(60, len(df))
        vp = self._calculate_volume_profile(df.tail(vp_lookback), current_price)
        
        # 2. ATR
        atr = self._calculate_atr(df)
        
        # 3. RSI (Wilder's)
        rsi_series = self._calculate_rsi_wilder(df['close'])
        current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        
        # 4. Track extension streaks
        streaks = self._track_streaks(df, vp, symbol, candle_minutes)
        
        # 5. Find the hottest streak
        hottest_key = None
        hottest_streak = None
        if streaks:
            hottest_key = max(streaks, key=lambda k: streaks[k].count)
            hottest_streak = streaks[hottest_key]
        
        # 6. Current extension zone
        if atr > 0:
            dist_from_vwap = (current_price - vp.vwap) / atr
        else:
            dist_from_vwap = 0.0
        
        zone = "in_value"
        if abs(dist_from_vwap) > 2.0:
            zone = "extreme_above" if dist_from_vwap > 0 else "extreme_below"
        elif current_price > vp.vah:
            zone = "above_vah"
        elif current_price < vp.val:
            zone = "below_val"
        elif current_price > vp.poc:
            zone = "vah_poc"
        elif current_price < vp.poc:
            zone = "poc_val"
        else:
            zone = "at_poc"
        
        # Determine primary extension direction
        if hottest_streak:
            ext_direction = hottest_streak.direction
        elif current_price > vp.vwap:
            ext_direction = "above"
        else:
            ext_direction = "below"
        
        # =====================================================================
        # V2 CONTEXT
        # =====================================================================
        
        # 7. Weekly Structure
        weekly = self._calculate_weekly_context(symbol)
        
        # 8. Squeeze Co-Detection
        squeeze = self._detect_squeeze(df)
        
        # 9. Enhanced Trend Analysis
        trend = self._analyze_trend(df, ext_direction)
        
        # 10. IV / Options
        options = self._estimate_iv_percentile(df)
        
        # 11. RSI extremes
        rsi_extreme = False
        if ext_direction == "above" and current_rsi > 70:
            rsi_extreme = True
        elif ext_direction == "below" and current_rsi < 30:
            rsi_extreme = True
        
        # 12. Spring loaded check
        if squeeze.is_squeezed and hottest_streak and hottest_streak.count >= 2:
            squeeze.spring_loaded = True
        
        # =====================================================================
        # SCORING
        # =====================================================================
        
        streak_count = hottest_streak.count if hottest_streak else 0
        
        # --- Extension Duration (0-30 pts) ---
        duration_score = 0
        if streak_count >= 4:
            duration_score = 30
            factors.append(f"EXTREME extension ({streak_count} candles, {hottest_streak.hours:.0f}h)")
        elif streak_count == 3:
            duration_score = 22
            factors.append(f"HIGH_PROB extension ({streak_count} candles, {hottest_streak.hours:.0f}h)")
        elif streak_count == 2:
            duration_score = 15
            factors.append(f"ALERT extension ({streak_count} candles)")
        elif streak_count == 1:
            duration_score = 8
        
        # --- Extension Distance (0-20 pts) ---
        distance_score = 0
        avg_ext = hottest_streak.avg_extension_atr if hottest_streak else abs(dist_from_vwap)
        if avg_ext > 2.5:
            distance_score = 20
            factors.append(f"Extreme distance ({avg_ext:.1f} ATR from level)")
        elif avg_ext > 2.0:
            distance_score = 15
        elif avg_ext > 1.5:
            distance_score = 10
        elif avg_ext > 1.0:
            distance_score = 5
        
        # --- Rejection Candle (0-15 pts) ---
        rejection_score = 0
        if hottest_streak and hottest_streak.has_rejection:
            rejection_score = 15
            factors.append("Rejection candle at extension")
        
        # --- Volume Declining (0-10 pts) ---
        vol_decline_score = 0
        if hottest_streak and hottest_streak.declining_volume:
            vol_decline_score = 10
            factors.append("Volume declining on extension")
        
        # --- RSI Confluence (0-15 pts) ---
        rsi_score = 0
        if rsi_extreme:
            if (ext_direction == "above" and current_rsi > 80) or \
               (ext_direction == "below" and current_rsi < 20):
                rsi_score = 15
                factors.append(f"RSI extreme ({current_rsi:.0f})")
            else:
                rsi_score = 10
                factors.append(f"RSI confirming ({current_rsi:.0f})")
        elif (ext_direction == "above" and current_rsi > 65) or \
             (ext_direction == "below" and current_rsi < 35):
            rsi_score = 5
        
        # --- Volume Profile Zone (0-15 pts) ---
        vp_score = 0
        if ext_direction == "above":
            if zone in ("above_vah", "extreme_above"):
                vp_score = 15
                factors.append(f"Above VAH ${vp.vah} (overvalued zone)")
            elif vp.at_key_level:
                vp_score = 8
        else:
            if zone in ("below_val", "extreme_below"):
                vp_score = 15
                factors.append(f"Below VAL ${vp.val} (undervalued zone)")
            elif vp.at_key_level:
                vp_score = 8
        
        # --- Weekly Alignment (0-10 pts) ---
        weekly_score = 0
        trade_dir = "SHORT" if ext_direction == "above" else "LONG"
        if trade_dir == "LONG" and weekly.supports_long:
            weekly_score = 10
            factors.append(f"Weekly supports long ({weekly.trend})")
        elif trade_dir == "SHORT" and weekly.supports_short:
            weekly_score = 10
            factors.append(f"Weekly supports short ({weekly.trend})")
        elif weekly.trend == "NEUTRAL":
            weekly_score = 3
        
        # --- Squeeze Co-Detection (0-10 pts) ---
        squeeze_score = 0
        if squeeze.spring_loaded:
            squeeze_score = 10
            factors.append(f"SPRING LOADED (extension + squeeze)")
        elif squeeze.is_squeezed:
            squeeze_score = 6
            factors.append(f"Squeeze active ({squeeze.squeeze_days}d)")
        elif squeeze.bb_width_percentile < 20:
            squeeze_score = 3
        
        # --- Trend Context (0-10 pts) ---
        trend_score = 0
        if trend.extension_vs_trend == "counter_trend":
            trend_score = 10
            factors.append("Counter-trend extension (snaps harder)")
        elif trend.extension_vs_trend == "neutral":
            trend_score = 5
        # With-trend gets 0 — extensions with trend are less likely to snap
        
        # --- Session Context (0-10 pts) ---
        session_score = 0
        session_ctx = "unknown"
        try:
            last_hour = df.index[-1].hour
            if last_hour >= 14:
                session_ctx = "close"
                session_score = 10
            elif last_hour <= 10:
                session_ctx = "open"
                session_score = 5
            else:
                session_ctx = "mid"
                session_score = 3
        except:
            pass
        
        # --- IV Percentile (0-10 pts) ---
        iv_score = 0
        if options.iv_regime == "low":
            iv_score = 10
            factors.append(f"Low IV ({options.iv_percentile:.0f}%ile) — cheap options")
        elif options.iv_regime == "normal":
            iv_score = 7
        elif options.iv_regime == "elevated":
            iv_score = 3
        
        # --- Prior Snap-Backs (0-5 pts) ---
        snap_history_score = 0
        if hottest_streak:
            prior = self._snap_back_history.get(symbol, {}).get(hottest_streak.level_name, 0)
            snap_history_score = min(5, prior * 2)
            if prior >= 2:
                factors.append(f"Level snapped back {prior}x before")
        
        # =====================================================================
        # TOTAL SCORE
        # =====================================================================
        raw_total = (duration_score + distance_score + rejection_score +
                    vol_decline_score + rsi_score + vp_score + weekly_score +
                    squeeze_score + trend_score + session_score + iv_score +
                    snap_history_score)
        
        # Normalize: max possible ~160, scale to 100
        total_score = min(100, int(raw_total * 100 / 160))
        
        # =====================================================================
        # SNAP-BACK PROBABILITY
        # =====================================================================
        # Normalize streak_count by bar interval — the BASE_PROBABILITIES table
        # was calibrated for ~5-15 min bars. Hourly or daily bars inflate counts.
        effective_streak = streak_count
        try:
            if len(df) >= 2 and hasattr(df.index, 'to_series'):
                diffs = df.index.to_series().diff().dropna()
                if len(diffs) > 0:
                    median_minutes = diffs.median().total_seconds() / 60
                    if median_minutes >= 50:       # hourly bars
                        effective_streak = max(0, int(streak_count / 4))
                    elif median_minutes >= 1300:    # daily bars
                        effective_streak = max(0, int(streak_count / 8))
                    # 5-15 min bars: no adjustment needed
        except Exception:
            pass  # fall back to raw streak_count

        base_prob = self.BASE_PROBABILITIES.get(min(effective_streak, 10), 0.92)
        prob = base_prob
        
        if hottest_streak and hottest_streak.has_rejection:
            prob += self.REJECTION_BONUS
        if hottest_streak and hottest_streak.declining_volume:
            prob += self.DECLINING_VOLUME_BONUS
        if avg_ext > 2.0:
            prob += self.EXTREME_EXTENSION_BONUS
        if trend.extension_vs_trend == "counter_trend":
            prob += self.COUNTER_TREND_BONUS
        if rsi_extreme:
            prob += self.RSI_EXTREME_BONUS
        if session_ctx == "close":
            prob += self.SESSION_CLOSE_BONUS
        if squeeze.is_squeezed:
            prob += self.SQUEEZE_BONUS
        if vp.at_key_level:
            prob += self.VP_KEY_LEVEL_BONUS
        if (trade_dir == "LONG" and weekly.supports_long) or \
           (trade_dir == "SHORT" and weekly.supports_short):
            prob += self.WEEKLY_ALIGNMENT_BONUS
        
        prob = min(0.90, prob)
        
        # =====================================================================
        # TRADE LEVELS
        # =====================================================================
        snap_target = 0.0
        stop = 0.0
        rr = 0.0
        
        if hottest_streak and streak_count >= 2:
            level_price = {"vwap": vp.vwap, "poc": vp.poc, "vah": vp.vah, "val": vp.val
                          }.get(hottest_streak.level_name, vp.poc)
            
            if ext_direction == "above":
                # SHORT snap-back toward level
                snap_target = level_price
                stop = current_price + atr * 0.5
                risk = stop - current_price
                reward = current_price - snap_target
            else:
                # LONG snap-back toward level
                snap_target = level_price
                stop = current_price - atr * 0.5
                risk = current_price - stop
                reward = snap_target - current_price
            
            rr = reward / risk if risk > 0 else 0
        
        # =====================================================================
        # TRIGGER LEVEL
        # =====================================================================
        if streak_count >= 4:
            trigger = "EXTREME"
        elif streak_count == 3:
            trigger = "HIGH_PROB"
        elif streak_count == 2:
            trigger = "ALERT"
        elif streak_count == 1:
            trigger = "WATCHING"
        else:
            trigger = "NONE"
        
        # =====================================================================
        # SETUP CLASSIFICATION
        # =====================================================================
        setup_type = ""
        entry_trigger = ""
        
        if streak_count >= 2:
            if squeeze.spring_loaded:
                setup_type = f"spring_snap_{'short' if ext_direction == 'above' else 'long'}"
                entry_trigger = f"Extension + squeeze: Enter {options.entry_size} {trade_dir.lower()} on rejection candle"
            elif trend.extension_vs_trend == "counter_trend" and rsi_extreme:
                setup_type = f"counter_trend_snap_{'short' if ext_direction == 'above' else 'long'}"
                entry_trigger = f"Counter-trend + RSI extreme: Enter {options.entry_size} on reversal"
            elif streak_count >= 4:
                setup_type = f"extreme_extension_{'short' if ext_direction == 'above' else 'long'}"
                entry_trigger = f"Extreme rubber band: Enter {options.entry_size}, target {hottest_streak.level_name.upper()}"
            elif streak_count >= 3:
                setup_type = f"extension_snap_{'short' if ext_direction == 'above' else 'long'}"
                entry_trigger = f"High-prob snap: Wait for rejection candle, enter {options.entry_size}"
            else:
                setup_type = f"extension_alert_{'short' if ext_direction == 'above' else 'long'}"
                entry_trigger = "Developing — monitor for continuation or rejection"
        
        # =====================================================================
        # ACTIVE STREAKS DICT
        # =====================================================================
        active_streaks_dict = {}
        for key, s in streaks.items():
            active_streaks_dict[key] = {
                'level': s.level_name,
                'direction': s.direction,
                'candles': s.count,
                'hours': round(s.hours, 1),
                'trigger': s.trigger.name,
                'avg_extension_atr': round(s.avg_extension_atr, 2),
                'has_rejection': s.has_rejection,
                'declining_volume': s.declining_volume,
                'snap_back_prob': round(
                    self.BASE_PROBABILITIES.get(min(s.count, 10), 0.92) * 100, 1
                )
            }
        
        hottest_dict = {}
        if hottest_streak:
            hottest_dict = active_streaks_dict.get(hottest_key, {})
        
        return ExtensionAnalysis(
            symbol=symbol,
            current_price=round(current_price, 2),
            timestamp=datetime.now().isoformat(),
            zone=zone,
            distance_from_vwap_atr=round(dist_from_vwap, 2),
            active_streaks=active_streaks_dict,
            hottest_streak=hottest_dict,
            trade_direction=trade_dir if streak_count >= 2 else "",
            snap_back_target=round(snap_target, 2),
            stop_loss=round(stop, 2),
            snap_back_probability=round(prob * 100, 1),
            risk_reward=round(rr, 2),
            extension_score=total_score,
            quality_grade=self._quality_grade(total_score),
            trigger_level=trigger,
            volume_profile=vp,
            weekly=weekly,
            squeeze=squeeze,
            trend=trend,
            options=options,
            rsi=round(current_rsi, 2),
            rsi_extreme=rsi_extreme,
            setup_type=setup_type,
            entry_trigger=entry_trigger,
            factors=factors[:8]
        )
    
    def to_dict(self, analysis: ExtensionAnalysis) -> Dict:
        """Convert to serializable dict"""
        return asdict(analysis)


# =============================================================================
# ALERT FORMATTER
# =============================================================================

def format_extension_alert(analysis: ExtensionAnalysis) -> str:
    """Format extension analysis as alert message"""
    emoji = {
        "NONE": "⚪", "WATCHING": "👀", "ALERT": "⚠️",
        "HIGH_PROB": "🔥", "EXTREME": "💥"
    }.get(analysis.trigger_level, "⚪")
    
    lines = [
        f"{emoji} {analysis.symbol} EXTENSION ANALYSIS (V2)",
        f"Trigger: {analysis.trigger_level} | Score: {analysis.extension_score}/100 | Grade: {analysis.quality_grade}",
        ""
    ]
    
    if analysis.setup_type:
        lines.append(f"🎯 SETUP: {analysis.setup_type}")
        lines.append(f"   {analysis.entry_trigger}")
        lines.append("")
    
    lines.append(f"📍 ZONE: {analysis.zone} | Distance: {analysis.distance_from_vwap_atr:.1f} ATR from VWAP")
    
    if analysis.hottest_streak:
        h = analysis.hottest_streak
        lines.append(f"🔄 HOTTEST: {h.get('level','?').upper()} {h.get('direction','?')} "
                     f"— {h.get('candles',0)} candles ({h.get('hours',0):.0f}h) "
                     f"| Rejection: {'✔' if h.get('has_rejection') else '✗'} "
                     f"| Vol declining: {'✔' if h.get('declining_volume') else '✗'}")
    
    if analysis.volume_profile:
        vp = analysis.volume_profile
        lines.append(f"\n📐 VP: VAH ${vp.vah} | POC ${vp.poc} | VAL ${vp.val} | VWAP ${vp.vwap}")
    
    lines.append(f"📈 RSI: {analysis.rsi:.1f} {'⚠️ EXTREME' if analysis.rsi_extreme else ''}")
    
    if analysis.trend:
        t = analysis.trend
        lines.append(f"📊 Trend: {t.direction} ({t.strength}) | vs Extension: {t.extension_vs_trend}")
    
    if analysis.weekly:
        lines.append(f"📅 Weekly: {analysis.weekly.trend} | Signal: {analysis.weekly.weekly_close_signal or 'none'}")
    
    if analysis.squeeze and analysis.squeeze.spring_loaded:
        lines.append(f"🔥 SPRING LOADED: Extension + squeeze ({analysis.squeeze.squeeze_days}d)!")
    elif analysis.squeeze and analysis.squeeze.is_squeezed:
        lines.append(f"🔲 Squeeze active: {analysis.squeeze.squeeze_days}d")
    
    if analysis.options:
        lines.append(f"📋 IV: {analysis.options.iv_percentile:.0f}%ile ({analysis.options.iv_regime}) | Size: {analysis.options.entry_size}")
    
    if analysis.trade_direction and analysis.snap_back_probability > 50:
        lines.append(f"\n💰 {analysis.trade_direction}: Snap-back {analysis.snap_back_probability:.0f}% → ${analysis.snap_back_target:.2f}")
        lines.append(f"   Stop: ${analysis.stop_loss:.2f} | R:R {analysis.risk_reward:.1f}")
    
    if analysis.active_streaks:
        lines.append(f"\n📋 ALL STREAKS:")
        for key, s in analysis.active_streaks.items():
            emoji_s = {"NONE": "⚪", "WATCHING": "👀", "ALERT": "⚠️",
                       "HIGH_PROB": "🔥", "EXTREME": "💥"}.get(s.get('trigger', 'NONE'), "⚪")
            lines.append(f"   {emoji_s} {s.get('level','?').upper()} {s.get('direction','?')}: "
                        f"{s.get('candles',0)} candles ({s.get('hours',0):.0f}h) "
                        f"| Snap-back: {s.get('snap_back_prob', 0):.0f}%")
    
    if analysis.factors:
        lines.append(f"\n✅ Factors: {', '.join(analysis.factors[:6])}")
    
    return "\n".join(lines)


# =============================================================================
# QUICK SCAN FUNCTIONS
# =============================================================================

def scan_extension(symbol: str, period: str = "1mo", interval: str = "1h") -> Optional[ExtensionAnalysis]:
    """Quick scan a single symbol for extensions"""
    from polygon_data import get_bars
    df = get_bars(symbol, period=period, interval=interval)
    if df.empty:
        return None
    predictor = ExtensionPredictorV2()
    return predictor.analyze(df, symbol)


def scan_symbols(symbols: List[str],
                 period: str = "1mo",
                 interval: str = "1h",
                 min_trigger: str = "ALERT") -> List[ExtensionAnalysis]:
    """
    Batch scan symbols for extensions.
    Returns list sorted by score, filtered by minimum trigger level.
    """
    
    trigger_order = {"NONE": 0, "WATCHING": 1, "ALERT": 2, "HIGH_PROB": 3, "EXTREME": 4}
    min_val = trigger_order.get(min_trigger, 2)
    
    predictor = ExtensionPredictorV2()
    results = []
    
    for symbol in symbols:
        try:
            from polygon_data import get_bars
            df = get_bars(symbol, period=period, interval=interval)
            if df.empty:
                continue
            
            analysis = predictor.analyze(df, symbol)
            if analysis and trigger_order.get(analysis.trigger_level, 0) >= min_val:
                results.append(analysis)
                
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
    
    results.sort(key=lambda x: x.extension_score, reverse=True)
    return results


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def add_extension_to_analysis(analysis_result: dict, extension_data: dict) -> dict:
    """Add extension data to an existing analysis result (backward compat)"""
    analysis_result['extension'] = extension_data
    
    for key, streak in extension_data.get('active_streaks', {}).items():
        if streak.get('trigger') in ['HIGH_PROB', 'EXTREME']:
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
# V1 COMPATIBILITY — ExtensionPredictor class wrapper
# =============================================================================

class ExtensionPredictor(ExtensionPredictorV2):
    """Backward-compatible wrapper. Use ExtensionPredictorV2 for new code."""
    
    def __init__(self, candle_minutes: int = 120):
        super().__init__()
        self._default_candle_minutes = candle_minutes
    
    def update(self, symbol: str, candle: CandleData) -> List:
        """V1-compatible update method"""
        candle.analyze()
        
        if symbol not in self._history:
            self._history[symbol] = []
        self._history[symbol].append(candle)
        if len(self._history[symbol]) > 50:
            self._history[symbol] = self._history[symbol][-50:]
        
        # Build minimal streak tracking
        if symbol not in self._streaks:
            self._streaks[symbol] = {}
        
        levels = [
            ("vwap", candle.vwap), ("poc", candle.poc),
            ("vah", candle.vah), ("val", candle.val),
        ]
        
        alerts = []
        for level_name, level_price in levels:
            if level_price <= 0:
                continue
            
            if candle.close > level_price + (candle.atr * 0.5):
                direction = "above"
            elif candle.close < level_price - (candle.atr * 0.5):
                direction = "below"
            else:
                direction = None
            
            key = f"{level_name}_{direction}" if direction else None
            
            if direction:
                if key not in self._streaks[symbol]:
                    self._streaks[symbol][key] = ExtensionStreak(
                        level_name=level_name, direction=direction,
                        candle_minutes=self._default_candle_minutes
                    )
                self._streaks[symbol][key].candles.append(candle)
            else:
                for d in ["above", "below"]:
                    ck = f"{level_name}_{d}"
                    if ck in self._streaks[symbol]:
                        del self._streaks[symbol][ck]
        
        return alerts
    
    def get_active_streaks(self, symbol: str) -> Dict[str, dict]:
        """V1-compatible streak getter"""
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
                'avg_extension_atr': round(streak.avg_extension_atr, 2),
                'has_rejection': streak.has_rejection,
                'snap_back_prob': round(
                    self.BASE_PROBABILITIES.get(min(streak.count, 10), 0.92) * 100, 1
                )
            }
        return result
    
    def get_hottest_setup(self, symbol: str) -> Optional[dict]:
        streaks = self.get_active_streaks(symbol)
        if not streaks:
            return None
        hottest = max(streaks.values(), key=lambda s: s['candles'])
        return hottest if hottest['candles'] >= 2 else None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  EXTENSION DURATION PREDICTOR V2 — C.O.R.E. Methodology")
    print("=" * 65)
    
    test_symbols = ["NVDA", "AAPL", "META", "TSLA", "AMD", "MSFT", "AMZN"]
    
    predictor = ExtensionPredictorV2()
    
    for symbol in test_symbols:
        print(f"\nScanning {symbol}...")
        try:
            from polygon_data import get_bars
            df = get_bars(symbol, period="1mo", interval="1h")
            
            if df.empty:
                print(f"  No data for {symbol}")
                continue
            
            result = predictor.analyze(df, symbol)
            
            if result and result.trigger_level != "NONE":
                print(format_extension_alert(result))
            else:
                print(f"  {symbol}: No active extensions (Score: {result.extension_score if result else 0})")
            
        except Exception as e:
            print(f"  Error: {e}")
        print("-" * 65)
    
    print(f"\n{'='*65}")
    print("  BATCH SCAN — ALERT+ Only")
    print(f"{'='*65}")
    
    results = scan_symbols(test_symbols, min_trigger="ALERT")
    
    if results:
        for r in results:
            emoji = {"ALERT": "⚠️", "HIGH_PROB": "🔥", "EXTREME": "💥"}.get(r.trigger_level, "")
            print(f"  {emoji} {r.symbol}: Score {r.extension_score} ({r.trigger_level}) "
                  f"— {r.setup_type or 'monitoring'} | Snap-back {r.snap_back_probability:.0f}%")
    else:
        print("  No ALERT+ extensions found")
