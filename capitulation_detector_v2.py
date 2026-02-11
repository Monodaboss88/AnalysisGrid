"""
Capitulation & Euphoria Detector V2 — C.O.R.E. Methodology
============================================================
Detects capitulation (bottoms) and euphoria (tops) aligned with
Rob's auction theory trading approach.

RETAINED FROM V1:
- Volume climax → exhaustion sequence detection
- Bullish/bearish divergence (enhanced with peak/trough finder)
- Reversal candle patterns (hammer, engulfing, shooting star)
- Consecutive day counting
- Session context (close > open > mid-session)
- Dual-mode: Capitulation (longs) + Euphoria (shorts)

NEW IN V2:
- Volume Profile Context — capitulation at VAL vs random zone
- Weekly Structure — pullback in uptrend vs breakdown in downtrend
- Squeeze Co-Detection — compressed + exhaustion = spring-loaded reversal
- Wilder's RSI — consistent with finnhub_scanner.py (EMA, not SMA)
- IV Percentile — cheap/expensive options context
- Enhanced Divergence — peak/trough finder instead of half-split
- Options Trade Levels — 0.65 delta, 3wk+ expiry, contract-based stops
- Dynamic Timeframe Detection — no more hardcoded * 8 hourly assumption
- Multi-Level Support/Resistance — VP levels, SMA, swing pivots, VWAP

Tiers:
- NONE (0-24): No capitulation/euphoria signs
- EARLY (25-44): Starting to show stress/froth
- DEVELOPING (45-59): Multiple signs aligning
- CLIMAX (60-79): Active selling/buying climax
- EXHAUSTION (80+): Sellers/buyers exhausted — ideal entry

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class CapitulationLevel(Enum):
    """Capitulation intensity levels"""
    NONE = "NONE"
    EARLY = "EARLY"
    DEVELOPING = "DEVELOPING"
    CLIMAX = "CLIMAX"
    EXHAUSTION = "EXHAUSTION"
    
    @property
    def score(self) -> int:
        return {"NONE": 0, "EARLY": 25, "DEVELOPING": 50,
                "CLIMAX": 75, "EXHAUSTION": 100}[self.value]
    
    @property
    def tradeable(self) -> bool:
        return self in [CapitulationLevel.EXHAUSTION, CapitulationLevel.CLIMAX]


class EuphoriaLevel(Enum):
    """Euphoria intensity levels (mirror for shorts)"""
    NONE = "NONE"
    EARLY = "EARLY"
    DEVELOPING = "DEVELOPING"
    CLIMAX = "CLIMAX"
    EXHAUSTION = "EXHAUSTION"
    
    @property
    def score(self) -> int:
        return {"NONE": 0, "EARLY": 25, "DEVELOPING": 50,
                "CLIMAX": 75, "EXHAUSTION": 100}[self.value]
    
    @property
    def tradeable(self) -> bool:
        return self in [EuphoriaLevel.EXHAUSTION, EuphoriaLevel.CLIMAX]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VolumeProfileContext:
    """Volume profile levels for support/resistance context"""
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    price_zone: str = "unknown"       # above_vah, vah_poc, at_poc, poc_val, below_val
    at_key_level: bool = False        # Within 1.5% of VAH/POC/VAL
    nearest_level: str = ""           # Which VP level is closest
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
    """Is a squeeze active alongside the capitulation/euphoria?"""
    is_squeezed: bool = False
    squeeze_days: int = 0
    bb_width_percentile: float = 50.0  # How tight are BBs vs recent history
    spring_loaded: bool = False         # Squeeze + exhaustion = spring


@dataclass
class SupportResistanceLevels:
    """Multi-source support/resistance confluence"""
    vp_level: float = 0.0             # Nearest VP level (VAL for cap, VAH for euph)
    vp_level_name: str = ""           # "VAL", "POC", "VAH"
    sma_200: float = 0.0
    sma_50: float = 0.0
    vwap: float = 0.0
    swing_level: float = 0.0          # Prior swing low/high
    confluence_count: int = 0          # How many levels cluster here
    confluence_score: int = 0          # 0-15 points


@dataclass
class OptionsContext:
    """Options-specific trade setup context"""
    iv_percentile: float = 50.0       # HV-based proxy
    iv_regime: str = "normal"          # low, normal, elevated, extreme
    suggested_delta: float = 0.65
    min_dte: int = 21                  # 3 weeks minimum
    contract_stop_pct: float = 12.5    # -12.5% on contract value
    entry_size: str = "50%"            # Initial position size
    scale_plan: str = ""               # When to add


@dataclass
class DivergenceDetail:
    """Enhanced divergence detection with peak/trough data"""
    detected: bool = False
    divergence_type: str = "none"      # bullish, bearish, none
    price_swing_1: float = 0.0        # First swing point
    price_swing_2: float = 0.0        # Second swing point
    rsi_swing_1: float = 0.0
    rsi_swing_2: float = 0.0
    bars_apart: int = 0               # Distance between swing points


@dataclass
class CapitulationMetrics:
    """Complete capitulation analysis — V2"""
    symbol: str = ""
    
    # Price decline
    decline_from_high_pct: float = 0.0
    days_since_high: int = 0
    
    # Volume analysis
    current_rvol: float = 1.0
    climax_volume_detected: bool = False
    volume_exhaustion: bool = False
    avg_down_volume_ratio: float = 1.0
    
    # RSI (Wilder's smoothing)
    rsi: float = 50.0
    rsi_oversold: bool = False
    rsi_extreme: bool = False
    rsi_divergence: DivergenceDetail = None
    
    # Candle patterns
    reversal_candle: bool = False
    long_lower_wick: bool = False
    reversal_pattern_name: str = ""    # NEW: specific pattern name
    
    # Additional factors
    consecutive_down_days: int = 0
    session_context: str = "unknown"
    
    # NEW V2: Volume Profile
    volume_profile: VolumeProfileContext = None
    vp_score: int = 0
    
    # NEW V2: Weekly Structure
    weekly: WeeklyContext = None
    weekly_score: int = 0
    
    # NEW V2: Squeeze Co-Detection
    squeeze: SqueezeContext = None
    squeeze_score: int = 0
    
    # NEW V2: Support Confluence
    support: SupportResistanceLevels = None
    support_score: int = 0
    
    # NEW V2: Options Context
    options: OptionsContext = None
    iv_score: int = 0
    
    # Composite
    capitulation_score: int = 0
    capitulation_level: CapitulationLevel = CapitulationLevel.NONE
    quality_grade: str = "F"           # NEW: A+ through F
    factors: List[str] = field(default_factory=list)
    
    # Trade info
    entry_zone: Tuple[float, float] = (0, 0)
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    setup_type: str = ""               # NEW: compression_reversal, climax_reversal, etc.
    entry_trigger: str = ""            # NEW: what to watch for
    
    current_price: float = 0.0
    timestamp: str = ""


@dataclass
class EuphoriaMetrics:
    """Complete euphoria analysis — V2 (mirror for shorts)"""
    symbol: str = ""
    
    # Price advance
    advance_from_low_pct: float = 0.0
    days_since_low: int = 0
    
    # Volume analysis
    current_rvol: float = 1.0
    climax_volume_detected: bool = False
    volume_exhaustion: bool = False
    avg_up_volume_ratio: float = 1.0
    
    # RSI (Wilder's smoothing)
    rsi: float = 50.0
    rsi_overbought: bool = False
    rsi_extreme: bool = False
    rsi_divergence: DivergenceDetail = None
    
    # Candle patterns
    reversal_candle: bool = False
    long_upper_wick: bool = False
    reversal_pattern_name: str = ""
    
    # Additional factors
    consecutive_up_days: int = 0
    session_context: str = "unknown"
    
    # NEW V2: Volume Profile
    volume_profile: VolumeProfileContext = None
    vp_score: int = 0
    
    # NEW V2: Weekly Structure
    weekly: WeeklyContext = None
    weekly_score: int = 0
    
    # NEW V2: Squeeze Co-Detection
    squeeze: SqueezeContext = None
    squeeze_score: int = 0
    
    # NEW V2: Resistance Confluence
    resistance: SupportResistanceLevels = None
    resistance_score: int = 0
    
    # NEW V2: Options Context
    options: OptionsContext = None
    iv_score: int = 0
    
    # Composite
    euphoria_score: int = 0
    euphoria_level: EuphoriaLevel = EuphoriaLevel.NONE
    quality_grade: str = "F"
    factors: List[str] = field(default_factory=list)
    
    # Trade info (SHORT)
    entry_zone: Tuple[float, float] = (0, 0)
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    setup_type: str = ""
    entry_trigger: str = ""
    
    current_price: float = 0.0
    timestamp: str = ""


# =============================================================================
# CAPITULATION DETECTOR V2
# =============================================================================

class CapitulationDetectorV2:
    """
    Enhanced capitulation & euphoria detector with C.O.R.E. methodology.
    
    Scoring (max ~175 pts, normalized to 100):
    - Price Decline/Advance:    0-25 pts
    - Volume Pattern:           0-25 pts  
    - RSI Condition:            0-25 pts
    - Candle Pattern:           0-25 pts
    - Volume Profile Context:   0-15 pts (NEW)
    - Weekly Alignment:         0-10 pts (NEW)
    - Squeeze Co-Detection:     0-10 pts (NEW)
    - Support/Resistance:       0-15 pts (NEW, enhanced)
    - IV Percentile:            0-10 pts (NEW)
    - Consecutive Days:         0-10 pts (bonus)
    - Session Context:          0-10 pts (bonus)
    """
    
    def __init__(self,
                 min_decline_pct: float = 10.0,
                 ideal_decline_pct: float = 20.0,
                 rsi_oversold: float = 30.0,
                 rsi_extreme: float = 25.0,
                 rsi_overbought: float = 70.0,
                 rsi_extreme_high: float = 80.0,
                 volume_climax_mult: float = 2.5,
                 volume_exhaustion_mult: float = 0.6,
                 lookback_days: int = 30):
        
        self.min_decline_pct = min_decline_pct
        self.ideal_decline_pct = ideal_decline_pct
        self.rsi_oversold = rsi_oversold
        self.rsi_extreme = rsi_extreme
        self.rsi_overbought = rsi_overbought
        self.rsi_extreme_high = rsi_extreme_high
        self.volume_climax_mult = volume_climax_mult
        self.volume_exhaustion_mult = volume_exhaustion_mult
        self.lookback_days = lookback_days
        
        # Bollinger / Keltner for squeeze co-detection
        self.bb_period = 20
        self.bb_std = 2.0
        self.kc_period = 20
        self.kc_mult = 1.5
        
        # VP parameters
        self.vp_num_bins = 50
        self.vp_value_area_pct = 0.70
        self.proximity_threshold = 0.015  # 1.5% for capitulation (wider than squeeze)
    
    # =========================================================================
    # TECHNICAL CALCULATIONS
    # =========================================================================
    
    def _calculate_rsi_wilder(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        RSI using Wilder's smoothing (EMA-based).
        Matches finnhub_scanner.py for consistency across the platform.
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _calculate_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR as series for squeeze detection"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_sma(self, series: pd.Series, period: int) -> float:
        """Calculate SMA, return last value"""
        if len(series) < period:
            return float(series.mean())
        return float(series.rolling(period).mean().iloc[-1])
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate VWAP"""
        if len(df) < 2:
            return float(df['close'].iloc[-1])
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vol_sum = df['volume'].sum()
        if vol_sum == 0:
            return float(df['close'].iloc[-1])
        return float((typical_price * df['volume']).sum() / vol_sum)
    
    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """
        Auto-detect data timeframe from index spacing.
        Fixes the hardcoded * 8 assumption from V1.
        """
        if len(df) < 3:
            return "unknown"
        
        try:
            diffs = pd.Series(df.index).diff().dropna()
            median_diff = diffs.median()
            
            minutes = median_diff.total_seconds() / 60
            
            if minutes < 10:
                return "5m"
            elif minutes < 35:
                return "30m"
            elif minutes < 90:
                return "1h"
            elif minutes < 300:
                return "4h"
            elif minutes < 1500:
                return "1d"
            else:
                return "1wk"
        except:
            return "1d"  # Safe default
    
    def _bars_per_day(self, timeframe: str) -> int:
        """How many bars per trading day for this timeframe"""
        return {
            "5m": 78, "15m": 26, "30m": 13, "1h": 7,
            "4h": 2, "1d": 1, "1wk": 0.2
        }.get(timeframe, 1)
    
    # =========================================================================
    # VOLUME PROFILE
    # =========================================================================
    
    def _calculate_volume_profile(self, df: pd.DataFrame, 
                                   current_price: float) -> VolumeProfileContext:
        """Calculate POC/VAH/VAL and determine price zone"""
        if len(df) < 10:
            return VolumeProfileContext()
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max == price_min:
            return VolumeProfileContext(poc=price_max, vah=price_max, val=price_min)
        
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
        
        # POC
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area
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
        
        # Find nearest level
        levels = {"VAL": val, "POC": poc, "VAH": vah}
        nearest_name = min(levels, key=lambda k: abs(current_price - levels[k]))
        nearest_price = levels[nearest_name]
        distance = abs(current_price - nearest_price) / current_price * 100
        at_key = distance < (self.proximity_threshold * 100)
        
        return VolumeProfileContext(
            poc=round(poc, 2), vah=round(vah, 2), val=round(val, 2),
            price_zone=zone, at_key_level=at_key,
            nearest_level=nearest_name, nearest_level_price=round(nearest_price, 2),
            distance_to_nearest_pct=round(distance, 2)
        )
    
    # =========================================================================
    # WEEKLY STRUCTURE
    # =========================================================================
    
    def _calculate_weekly_context(self, symbol: str) -> WeeklyContext:
        """Fetch weekly data and classify trend structure"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df_w = ticker.history(period="6mo", interval="1wk")
            
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
            
            # Weekly close position (completed week)
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
        """
        Check if a TTM squeeze is active alongside the capitulation/euphoria.
        Squeeze + exhaustion = spring-loaded reversal (highest probability).
        """
        if len(df) < self.bb_period + 5:
            return SqueezeContext()
        
        # Bollinger Bands
        sma = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std()
        bb_upper = sma + (self.bb_std * std)
        bb_lower = sma - (self.bb_std * std)
        
        # Keltner Channels
        ema = df['close'].ewm(span=self.kc_period, adjust=False).mean()
        atr = self._calculate_atr_series(df, self.kc_period)
        kc_upper = ema + (self.kc_mult * atr)
        kc_lower = ema - (self.kc_mult * atr)
        
        # Squeeze state
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        is_squeezed = bool(squeeze.iloc[-1]) if len(squeeze) > 0 else False
        
        # Count consecutive squeeze days
        squeeze_days = 0
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]:
                squeeze_days += 1
            else:
                break
        
        # BB width percentile (how tight vs recent history)
        bb_width = (bb_upper - bb_lower) / sma
        bb_width = bb_width.dropna()
        if len(bb_width) >= 20:
            current_width = bb_width.iloc[-1]
            lookback = bb_width.tail(60) if len(bb_width) >= 60 else bb_width
            width_pct = float((lookback < current_width).sum() / len(lookback) * 100)
        else:
            width_pct = 50.0
        
        return SqueezeContext(
            is_squeezed=is_squeezed,
            squeeze_days=squeeze_days,
            bb_width_percentile=round(width_pct, 1),
            spring_loaded=False  # Set by caller based on exhaustion state
        )
    
    # =========================================================================
    # ENHANCED DIVERGENCE (PEAK/TROUGH FINDER)
    # =========================================================================
    
    def _find_swing_points(self, series: pd.Series, 
                            mode: str = "lows",
                            lookback: int = 20,
                            min_separation: int = 3) -> List[Tuple[int, float]]:
        """
        Find swing lows or highs in a series using a local extrema approach.
        More reliable than the V1 half-split method.
        
        Args:
            series: Price or RSI series
            mode: 'lows' for swing lows, 'highs' for swing highs
            lookback: Number of bars to search
            min_separation: Minimum bars between swing points
        
        Returns:
            List of (bar_index, value) tuples, most recent last
        """
        data = series.tail(lookback).values
        swings = []
        
        for i in range(2, len(data) - 1):
            if mode == "lows":
                # Local minimum: lower than neighbors
                if data[i] < data[i-1] and data[i] < data[i+1]:
                    # Also check 2-bar neighbors if available
                    if i >= 2 and i < len(data) - 2:
                        if data[i] <= data[i-2] and data[i] <= data[i+2]:
                            swings.append((len(series) - lookback + i, float(data[i])))
                    else:
                        swings.append((len(series) - lookback + i, float(data[i])))
            else:
                # Local maximum: higher than neighbors
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    if i >= 2 and i < len(data) - 2:
                        if data[i] >= data[i-2] and data[i] >= data[i+2]:
                            swings.append((len(series) - lookback + i, float(data[i])))
                    else:
                        swings.append((len(series) - lookback + i, float(data[i])))
        
        # Filter: minimum separation between points
        if len(swings) < 2:
            return swings
        
        filtered = [swings[0]]
        for s in swings[1:]:
            if s[0] - filtered[-1][0] >= min_separation:
                filtered.append(s)
        
        return filtered
    
    def _check_bullish_divergence(self, df: pd.DataFrame, 
                                   rsi: pd.Series) -> DivergenceDetail:
        """
        Enhanced bullish divergence: Price makes lower low, RSI makes higher low.
        Uses peak/trough finder instead of naive half-split.
        """
        if len(df) < 15:
            return DivergenceDetail()
        
        price_lows = self._find_swing_points(df['low'], mode="lows", lookback=20)
        rsi_lows = self._find_swing_points(rsi, mode="lows", lookback=20)
        
        if len(price_lows) < 2 or len(rsi_lows) < 2:
            return DivergenceDetail()
        
        # Compare last two swing lows
        p1_idx, p1_val = price_lows[-2]
        p2_idx, p2_val = price_lows[-1]
        
        # Find RSI values at those price swing points (nearest RSI swing)
        r1_val = rsi.iloc[p1_idx] if p1_idx < len(rsi) else 50.0
        r2_val = rsi.iloc[p2_idx] if p2_idx < len(rsi) else 50.0
        
        # Bullish divergence: price lower low, RSI higher low
        if p2_val < p1_val and r2_val > r1_val:
            return DivergenceDetail(
                detected=True, divergence_type="bullish",
                price_swing_1=round(p1_val, 2), price_swing_2=round(p2_val, 2),
                rsi_swing_1=round(r1_val, 2), rsi_swing_2=round(r2_val, 2),
                bars_apart=p2_idx - p1_idx
            )
        
        return DivergenceDetail()
    
    def _check_bearish_divergence(self, df: pd.DataFrame, 
                                   rsi: pd.Series) -> DivergenceDetail:
        """
        Enhanced bearish divergence: Price makes higher high, RSI makes lower high.
        """
        if len(df) < 15:
            return DivergenceDetail()
        
        price_highs = self._find_swing_points(df['high'], mode="highs", lookback=20)
        rsi_highs = self._find_swing_points(rsi, mode="highs", lookback=20)
        
        if len(price_highs) < 2 or len(rsi_highs) < 2:
            return DivergenceDetail()
        
        p1_idx, p1_val = price_highs[-2]
        p2_idx, p2_val = price_highs[-1]
        
        r1_val = rsi.iloc[p1_idx] if p1_idx < len(rsi) else 50.0
        r2_val = rsi.iloc[p2_idx] if p2_idx < len(rsi) else 50.0
        
        # Bearish divergence: price higher high, RSI lower high
        if p2_val > p1_val and r2_val < r1_val:
            return DivergenceDetail(
                detected=True, divergence_type="bearish",
                price_swing_1=round(p1_val, 2), price_swing_2=round(p2_val, 2),
                rsi_swing_1=round(r1_val, 2), rsi_swing_2=round(r2_val, 2),
                bars_apart=p2_idx - p1_idx
            )
        
        return DivergenceDetail()
    
    # =========================================================================
    # SUPPORT / RESISTANCE CONFLUENCE
    # =========================================================================
    
    def _analyze_support_confluence(self, df: pd.DataFrame, 
                                     current_price: float,
                                     vp: VolumeProfileContext) -> SupportResistanceLevels:
        """
        Multi-source support confluence for capitulation.
        More levels clustering = higher probability reversal.
        """
        levels = SupportResistanceLevels()
        confluence = 0
        threshold = 0.02  # 2% proximity
        
        # 1. VP Level (VAL is key support for capitulation)
        levels.vp_level = vp.val
        levels.vp_level_name = "VAL"
        if abs(current_price - vp.val) / current_price < threshold:
            confluence += 1
        if abs(current_price - vp.poc) / current_price < threshold:
            confluence += 1
        
        # 2. SMA 200
        if len(df) >= 200:
            levels.sma_200 = self._calculate_sma(df['close'], 200)
            if abs(current_price - levels.sma_200) / current_price < threshold:
                confluence += 1
        
        # 3. SMA 50
        if len(df) >= 50:
            levels.sma_50 = self._calculate_sma(df['close'], 50)
            if abs(current_price - levels.sma_50) / current_price < threshold:
                confluence += 1
        
        # 4. VWAP
        vwap_lookback = df.tail(min(30, len(df)))
        levels.vwap = self._calculate_vwap(vwap_lookback)
        if abs(current_price - levels.vwap) / current_price < threshold:
            confluence += 1
        
        # 5. Prior swing low
        if len(df) >= 30:
            # Find swing low in bars 10-30 (not the current selloff)
            search_range = df.iloc[-30:-5] if len(df) >= 30 else df.iloc[:-5]
            if len(search_range) > 0:
                levels.swing_level = float(search_range['low'].min())
                if abs(current_price - levels.swing_level) / current_price < threshold:
                    confluence += 1
        
        levels.confluence_count = confluence
        
        # Score: each confluence point = 3 pts, max 15
        levels.confluence_score = min(15, confluence * 3)
        
        return levels
    
    def _analyze_resistance_confluence(self, df: pd.DataFrame,
                                        current_price: float,
                                        vp: VolumeProfileContext) -> SupportResistanceLevels:
        """
        Multi-source resistance confluence for euphoria.
        """
        levels = SupportResistanceLevels()
        confluence = 0
        threshold = 0.02
        
        # VAH is key resistance for euphoria
        levels.vp_level = vp.vah
        levels.vp_level_name = "VAH"
        if abs(current_price - vp.vah) / current_price < threshold:
            confluence += 1
        if abs(current_price - vp.poc) / current_price < threshold:
            confluence += 1
        
        if len(df) >= 200:
            levels.sma_200 = self._calculate_sma(df['close'], 200)
            if current_price > levels.sma_200 and abs(current_price - levels.sma_200) / current_price < threshold:
                confluence += 1
        
        if len(df) >= 50:
            levels.sma_50 = self._calculate_sma(df['close'], 50)
        
        vwap_lookback = df.tail(min(30, len(df)))
        levels.vwap = self._calculate_vwap(vwap_lookback)
        if abs(current_price - levels.vwap) / current_price < threshold:
            confluence += 1
        
        # Prior swing high
        if len(df) >= 30:
            search_range = df.iloc[-30:-5] if len(df) >= 30 else df.iloc[:-5]
            if len(search_range) > 0:
                levels.swing_level = float(search_range['high'].max())
                if abs(current_price - levels.swing_level) / current_price < threshold:
                    confluence += 1
        
        # All-time high in data
        ath = float(df['high'].max())
        if abs(current_price - ath) / current_price < threshold:
            confluence += 1
        
        levels.confluence_count = confluence
        levels.confluence_score = min(15, confluence * 3)
        
        return levels
    
    # =========================================================================
    # IV PERCENTILE (HV PROXY)
    # =========================================================================
    
    def _estimate_iv_percentile(self, df: pd.DataFrame) -> OptionsContext:
        """
        Estimate IV regime using historical volatility.
        For capitulation: high IV = expensive puts but potentially cheap calls
        (if IV mean-reverts after panic).
        """
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
        
        # IV regime classification
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
        
        # Adjust strategy based on IV
        if regime == "extreme":
            # IV is high — consider selling premium or waiting for crush
            options.suggested_delta = 0.60
            options.entry_size = "25%"
            options.scale_plan = "Scale in as IV normalizes, add at +15% contract gain"
        elif regime == "elevated":
            options.suggested_delta = 0.65
            options.entry_size = "40%"
            options.scale_plan = "Enter 40%, add at +15% and +25%"
        elif regime == "low":
            # Cheap options — full size
            options.suggested_delta = 0.70
            options.entry_size = "50%"
            options.scale_plan = "Enter 50%, add at +15% and +25%"
        else:
            options.suggested_delta = 0.65
            options.entry_size = "50%"
            options.scale_plan = "Enter 50%, add at +15% and +25%"
        
        return options
    
    # =========================================================================
    # CORE ANALYSIS COMPONENTS
    # =========================================================================
    
    def _analyze_decline(self, df: pd.DataFrame) -> Tuple[float, int, float]:
        """Analyze price decline from recent high"""
        timeframe = self._detect_timeframe(df)
        bpd = self._bars_per_day(timeframe)
        lookback_bars = int(self.lookback_days * max(bpd, 1))
        lookback_bars = min(lookback_bars, len(df))
        
        recent = df.tail(lookback_bars)
        recent_high = float(recent['high'].max())
        high_idx = recent['high'].idxmax()
        current_price = float(df['close'].iloc[-1])
        
        decline_pct = ((recent_high - current_price) / recent_high) * 100
        
        try:
            days_since = (df.index[-1] - high_idx).days
        except:
            days_since = len(df) - df.index.get_loc(high_idx)
        
        return decline_pct, days_since, recent_high
    
    def _analyze_advance(self, df: pd.DataFrame) -> Tuple[float, int, float]:
        """Analyze price advance from recent low"""
        timeframe = self._detect_timeframe(df)
        bpd = self._bars_per_day(timeframe)
        lookback_bars = int(self.lookback_days * max(bpd, 1))
        lookback_bars = min(lookback_bars, len(df))
        
        recent = df.tail(lookback_bars)
        recent_low = float(recent['low'].min())
        low_idx = recent['low'].idxmin()
        current_price = float(df['close'].iloc[-1])
        
        advance_pct = ((current_price - recent_low) / recent_low) * 100
        
        try:
            days_since = (df.index[-1] - low_idx).days
        except:
            days_since = len(df) - df.index.get_loc(low_idx)
        
        return advance_pct, days_since, recent_low
    
    def _analyze_volume(self, df: pd.DataFrame, 
                         direction: str = "down") -> Tuple[float, bool, bool, float]:
        """
        Analyze volume patterns for climax and exhaustion.
        direction='down' for capitulation, 'up' for euphoria.
        """
        if 'volume' not in df.columns:
            return 1.0, False, False, 1.0
        
        avg_vol = df['volume'].rolling(20).mean()
        current_rvol = float(df['volume'].iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 1.0
        rvol_series = df['volume'] / avg_vol
        
        df_temp = df.copy()
        if direction == "down":
            df_temp['is_target'] = df_temp['close'] < df_temp['open']  # Down bars
        else:
            df_temp['is_target'] = df_temp['close'] > df_temp['open']  # Up bars
        df_temp['rvol'] = rvol_series
        
        # Look for volume climax in recent 10 bars
        recent = df_temp.tail(10)
        climax_detected = False
        climax_bar_idx = None
        
        for i in range(len(recent) - 1, -1, -1):
            bar = recent.iloc[i]
            if bar['is_target'] and bar['rvol'] >= self.volume_climax_mult:
                climax_detected = True
                climax_bar_idx = i
                break
        
        # Exhaustion: bars AFTER climax have dried up
        exhaustion = False
        if climax_detected and climax_bar_idx is not None:
            bars_after = len(recent) - 1 - climax_bar_idx
            if bars_after >= 1:
                post_climax = recent.iloc[climax_bar_idx + 1:]
                if len(post_climax) > 0:
                    exhaustion = float(post_climax['rvol'].mean()) < self.volume_exhaustion_mult
        
        # Volume ratio
        target_vol = float(df[df_temp['is_target']]['volume'].mean()) if df_temp['is_target'].any() else 0
        other_vol = float(df[~df_temp['is_target']]['volume'].mean()) if (~df_temp['is_target']).any() else 1
        vol_ratio = target_vol / other_vol if other_vol > 0 else 1.0
        
        return current_rvol, climax_detected, exhaustion, vol_ratio
    
    def _analyze_candles_bullish(self, df: pd.DataFrame) -> Tuple[bool, bool, str]:
        """Analyze for bullish reversal candle patterns. Returns (reversal, wick, name)."""
        if len(df) < 2:
            return False, False, ""
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['close'] - last['open'])
        full_range = last['high'] - last['low']
        lower_wick = min(last['open'], last['close']) - last['low']
        
        long_wick = lower_wick > body * 1.5 and lower_wick > full_range * 0.4 if full_range > 0 else False
        bullish_candle = last['close'] > last['open']
        
        # Hammer
        hammer = bullish_candle and long_wick
        
        # Bullish engulfing
        engulfing = (
            prev['close'] < prev['open'] and
            last['close'] > last['open'] and
            last['close'] > prev['open'] and
            last['open'] < prev['close']
        )
        
        # Morning star (3-bar)
        morning_star = False
        if len(df) >= 3:
            bar3 = df.iloc[-3]
            mid = df.iloc[-2]
            mid_body = abs(mid['close'] - mid['open'])
            mid_range = mid['high'] - mid['low']
            small_body = mid_body < mid_range * 0.3 if mid_range > 0 else False
            morning_star = (
                bar3['close'] < bar3['open'] and  # First bar bearish
                small_body and                      # Middle bar small body
                last['close'] > last['open'] and    # Last bar bullish
                last['close'] > bar3['open'] * 0.99  # Closes above first bar's open
            )
        
        if engulfing:
            return True, long_wick, "bullish_engulfing"
        elif morning_star:
            return True, long_wick, "morning_star"
        elif hammer:
            return True, long_wick, "hammer"
        elif long_wick:
            return False, True, "long_lower_wick"
        
        return False, False, ""
    
    def _analyze_candles_bearish(self, df: pd.DataFrame) -> Tuple[bool, bool, str]:
        """Analyze for bearish reversal candle patterns. Returns (reversal, wick, name)."""
        if len(df) < 2:
            return False, False, ""
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['close'] - last['open'])
        full_range = last['high'] - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        
        long_wick = upper_wick > body * 1.5 and upper_wick > full_range * 0.4 if full_range > 0 else False
        bearish_candle = last['close'] < last['open']
        
        # Shooting star
        shooting_star = bearish_candle and long_wick
        
        # Bearish engulfing
        engulfing = (
            prev['close'] > prev['open'] and
            last['close'] < last['open'] and
            last['open'] > prev['close'] and
            last['close'] < prev['open']
        )
        
        # Evening star (3-bar)
        evening_star = False
        if len(df) >= 3:
            bar3 = df.iloc[-3]
            mid = df.iloc[-2]
            mid_body = abs(mid['close'] - mid['open'])
            mid_range = mid['high'] - mid['low']
            small_body = mid_body < mid_range * 0.3 if mid_range > 0 else False
            evening_star = (
                bar3['close'] > bar3['open'] and
                small_body and
                last['close'] < last['open'] and
                last['close'] < bar3['open'] * 1.01
            )
        
        if engulfing:
            return True, long_wick, "bearish_engulfing"
        elif evening_star:
            return True, long_wick, "evening_star"
        elif shooting_star:
            return True, long_wick, "shooting_star"
        elif long_wick:
            return False, True, "long_upper_wick"
        
        return False, False, ""
    
    def _count_consecutive(self, df: pd.DataFrame, direction: str = "down") -> int:
        """Count consecutive down or up bars before current bar"""
        count = 0
        for i in range(len(df) - 2, max(0, len(df) - 10), -1):
            if direction == "down":
                if df['close'].iloc[i] < df['open'].iloc[i]:
                    count += 1
                else:
                    break
            else:
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    count += 1
                else:
                    break
        return count
    
    def _get_session_context(self, df: pd.DataFrame) -> str:
        """Determine session context"""
        try:
            last_hour = df.index[-1].hour
            if last_hour >= 15:
                return 'close'
            elif last_hour <= 10:
                return 'open'
            else:
                return 'mid-session'
        except:
            return 'unknown'
    
    # =========================================================================
    # QUALITY GRADE
    # =========================================================================
    
    def _quality_grade(self, score: int) -> str:
        """Map score to quality grade"""
        if score >= 90: return "A+"
        elif score >= 80: return "A"
        elif score >= 70: return "B"
        elif score >= 60: return "C"
        elif score >= 50: return "D"
        else: return "F"
    
    # =========================================================================
    # MAIN: CAPITULATION ANALYSIS
    # =========================================================================
    
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> CapitulationMetrics:
        """
        Full capitulation analysis with C.O.R.E. methodology.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if len(df) < 20:
            return CapitulationMetrics(symbol=symbol)
        
        current_price = float(df['close'].iloc[-1])
        factors = []
        
        # =================================================================
        # CORE ANALYSIS
        # =================================================================
        
        # 1. Price decline
        decline_pct, days_since_high, recent_high = self._analyze_decline(df)
        
        # 2. Volume
        rvol, climax, exhaustion, down_vol_ratio = self._analyze_volume(df, "down")
        
        # 3. RSI (Wilder's)
        rsi_series = self._calculate_rsi_wilder(df)
        current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        oversold = current_rsi < self.rsi_oversold
        extreme = current_rsi < self.rsi_extreme
        
        # 4. Divergence (enhanced peak/trough)
        divergence = self._check_bullish_divergence(df, rsi_series)
        
        # 5. Candles
        reversal_candle, long_wick, pattern_name = self._analyze_candles_bullish(df)
        
        # 6. Consecutive down
        consecutive_down = self._count_consecutive(df, "down")
        
        # 7. Session context
        session_ctx = self._get_session_context(df)
        
        # =================================================================
        # NEW V2 ANALYSIS
        # =================================================================
        
        # 8. Volume Profile (30-day tactical VP)
        df_vp = df.tail(min(30 * self._bars_per_day(self._detect_timeframe(df)), len(df)))
        if len(df_vp) < 10:
            df_vp = df.tail(30)
        vp = self._calculate_volume_profile(df_vp, current_price)
        
        # 9. Weekly Structure
        weekly = self._calculate_weekly_context(symbol)
        
        # 10. Squeeze Co-Detection
        squeeze = self._detect_squeeze(df)
        if squeeze.is_squeezed and exhaustion:
            squeeze.spring_loaded = True
        
        # 11. Support Confluence
        support = self._analyze_support_confluence(df, current_price, vp)
        
        # 12. IV / Options Context
        options = self._estimate_iv_percentile(df)
        
        # =================================================================
        # SCORING
        # =================================================================
        
        # --- Price Decline (0-25 pts) ---
        decline_score = 0
        if decline_pct >= self.ideal_decline_pct:
            decline_score = 25
            factors.append(f"Down {decline_pct:.1f}% from high")
        elif decline_pct >= self.min_decline_pct:
            decline_score = int(15 + (decline_pct - self.min_decline_pct) /
                               (self.ideal_decline_pct - self.min_decline_pct) * 10)
            factors.append(f"Down {decline_pct:.1f}% from high")
        elif decline_pct >= 7:
            decline_score = 10
        
        # --- Volume Pattern (0-25 pts) ---
        volume_score = 0
        if exhaustion:
            volume_score = 25
            factors.append("Volume climax → exhaustion")
        elif climax:
            volume_score = 15
            factors.append("Volume climax detected")
        elif rvol < 0.6:
            volume_score = 8
            factors.append("Low volume (drying up)")
        
        # --- RSI (0-25 pts) ---
        rsi_score = 0
        if extreme:
            rsi_score = 20
            factors.append(f"RSI extreme ({current_rsi:.0f})")
        elif oversold:
            rsi_score = 15
            factors.append(f"RSI oversold ({current_rsi:.0f})")
        elif current_rsi < 35:
            rsi_score = 10
        
        if divergence.detected:
            rsi_score = min(25, rsi_score + 5)
            factors.append("Bullish RSI divergence")
        
        # --- Candle Pattern (0-25 pts) ---
        candle_score = 0
        if reversal_candle:
            candle_score = 25
            factors.append(f"Reversal: {pattern_name}")
        elif long_wick:
            candle_score = 15
            factors.append(f"Long lower wick ({pattern_name})")
        
        # --- NEW: Volume Profile (0-15 pts) ---
        vp_score = 0
        if vp.price_zone == "below_val":
            vp_score = 15
            factors.append(f"Below VAL ${vp.val} (capitulation zone)")
        elif vp.price_zone == "poc_val" and vp.at_key_level:
            vp_score = 12
            factors.append(f"At VAL ${vp.val}")
        elif vp.at_key_level:
            vp_score = 8
            factors.append(f"Near {vp.nearest_level} ${vp.nearest_level_price}")
        elif vp.price_zone in ("poc_val", "below_val"):
            vp_score = 5
        
        # --- NEW: Weekly Alignment (0-10 pts) ---
        weekly_score = 0
        if weekly.supports_long:
            if weekly.trend in ("UPTREND", "STRONG_UPTREND"):
                weekly_score = 10
                factors.append(f"Pullback in weekly {weekly.trend}")
            elif weekly.weekly_close_signal == "BULLISH_REVERSAL":
                weekly_score = 8
                factors.append("Weekly bullish reversal")
            else:
                weekly_score = 5
        elif weekly.trend == "NEUTRAL":
            weekly_score = 3
        # Note: capitulation in a strong downtrend gets 0 weekly points
        # (still valid, just lower probability for a swing long)
        
        # --- NEW: Squeeze Co-Detection (0-10 pts) ---
        squeeze_score = 0
        if squeeze.spring_loaded:
            squeeze_score = 10
            factors.append(f"SPRING LOADED (squeeze + exhaustion)")
        elif squeeze.is_squeezed:
            squeeze_score = 7
            factors.append(f"Squeeze active ({squeeze.squeeze_days}d)")
        elif squeeze.bb_width_percentile < 20:
            squeeze_score = 4
            factors.append("Tight Bollinger Bands")
        
        # --- NEW: Support Confluence (0-15 pts, replaces old basic check) ---
        # Already calculated in support.confluence_score
        support_score = support.confluence_score
        if support.confluence_count >= 3:
            factors.append(f"Strong support confluence ({support.confluence_count} levels)")
        elif support.confluence_count >= 2:
            factors.append(f"Support confluence ({support.confluence_count} levels)")
        
        # --- NEW: IV Score (0-10 pts) ---
        iv_score = 0
        if options.iv_regime == "low":
            iv_score = 10
            factors.append(f"Low IV ({options.iv_percentile:.0f}%ile) — cheap calls")
        elif options.iv_regime == "normal":
            iv_score = 7
        elif options.iv_regime == "elevated":
            iv_score = 3
        # Extreme IV = 0 pts (expensive options, poor R:R for buying)
        
        # --- Bonus: Consecutive Down (+10) ---
        consecutive_score = 0
        if consecutive_down >= 3:
            consecutive_score = 10
            factors.append(f"{consecutive_down} consecutive down bars")
        elif consecutive_down >= 2:
            consecutive_score = 5
        
        # --- Bonus: Session Context (+10) ---
        session_score = 0
        if session_ctx == 'close':
            session_score = 10
        elif session_ctx == 'open':
            session_score = 5
        
        # =================================================================
        # TOTAL SCORE
        # =================================================================
        raw_total = (decline_score + volume_score + rsi_score + candle_score +
                    vp_score + weekly_score + squeeze_score + support_score +
                    iv_score + consecutive_score + session_score)
        
        # Normalize: max possible ~180, scale to 100
        total_score = min(100, int(raw_total * 100 / 180))
        
        # Determine level
        level = self._determine_level(total_score, exhaustion, climax)
        
        # =================================================================
        # TRADE LEVELS (Options-based)
        # =================================================================
        atr = self._calculate_atr(df)
        entry_zone = (round(current_price * 0.995, 2), round(current_price * 1.005, 2))
        stop_loss = round(current_price - (atr * 1.5), 2)
        target_1 = round(current_price + (atr * 2), 2)
        target_2 = round(current_price + (recent_high - current_price) * 0.382, 2)
        
        # Setup type
        setup_type = ""
        entry_trigger = ""
        if squeeze.spring_loaded:
            setup_type = "spring_loaded_reversal"
            entry_trigger = f"Enter {options.entry_size} at RSI exhaustion, squeeze + capitulation aligned"
        elif level == CapitulationLevel.EXHAUSTION:
            setup_type = "capitulation_exhaustion"
            entry_trigger = f"Enter {options.entry_size} on reversal candle confirmation"
        elif level == CapitulationLevel.CLIMAX:
            setup_type = "climax_reversal"
            entry_trigger = "Wait for volume dry-up (1-2 bars) then enter"
        elif vp.price_zone == "below_val" and oversold:
            setup_type = "vp_exhaustion_long"
            entry_trigger = f"Below VAL ${vp.val} with RSI {current_rsi:.0f} — watch for reversal candle"
        
        return CapitulationMetrics(
            symbol=symbol,
            decline_from_high_pct=round(decline_pct, 2),
            days_since_high=days_since_high,
            current_rvol=round(rvol, 2),
            climax_volume_detected=climax,
            volume_exhaustion=exhaustion,
            avg_down_volume_ratio=round(down_vol_ratio, 2),
            rsi=round(current_rsi, 2),
            rsi_oversold=oversold,
            rsi_extreme=extreme,
            rsi_divergence=divergence,
            reversal_candle=reversal_candle,
            long_lower_wick=long_wick,
            reversal_pattern_name=pattern_name,
            consecutive_down_days=consecutive_down,
            session_context=session_ctx,
            volume_profile=vp,
            vp_score=vp_score,
            weekly=weekly,
            weekly_score=weekly_score,
            squeeze=squeeze,
            squeeze_score=squeeze_score,
            support=support,
            support_score=support_score,
            options=options,
            iv_score=iv_score,
            capitulation_score=total_score,
            capitulation_level=level,
            quality_grade=self._quality_grade(total_score),
            factors=factors[:8],
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            setup_type=setup_type,
            entry_trigger=entry_trigger,
            current_price=round(current_price, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def _determine_level(self, score: int, exhaustion: bool, climax: bool) -> CapitulationLevel:
        """Determine capitulation level"""
        if score >= 80 and exhaustion:
            return CapitulationLevel.EXHAUSTION
        elif score >= 60 and climax:
            return CapitulationLevel.CLIMAX
        elif score >= 45:
            return CapitulationLevel.DEVELOPING
        elif score >= 25:
            return CapitulationLevel.EARLY
        else:
            return CapitulationLevel.NONE
    
    # =========================================================================
    # EUPHORIA ANALYSIS (SHORT SETUPS)
    # =========================================================================
    
    def analyze_euphoria(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> EuphoriaMetrics:
        """
        Full euphoria analysis — mirror of capitulation for short setups.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if len(df) < 20:
            return EuphoriaMetrics(symbol=symbol)
        
        current_price = float(df['close'].iloc[-1])
        factors = []
        
        # =================================================================
        # CORE ANALYSIS
        # =================================================================
        
        advance_pct, days_since_low, recent_low = self._analyze_advance(df)
        rvol, climax, exhaustion, up_vol_ratio = self._analyze_volume(df, "up")
        
        rsi_series = self._calculate_rsi_wilder(df)
        current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        overbought = current_rsi > self.rsi_overbought
        extreme = current_rsi > self.rsi_extreme_high
        
        divergence = self._check_bearish_divergence(df, rsi_series)
        reversal_candle, long_wick, pattern_name = self._analyze_candles_bearish(df)
        consecutive_up = self._count_consecutive(df, "up")
        session_ctx = self._get_session_context(df)
        
        # =================================================================
        # V2 ANALYSIS
        # =================================================================
        
        df_vp = df.tail(min(30 * self._bars_per_day(self._detect_timeframe(df)), len(df)))
        if len(df_vp) < 10:
            df_vp = df.tail(30)
        vp = self._calculate_volume_profile(df_vp, current_price)
        weekly = self._calculate_weekly_context(symbol)
        
        squeeze = self._detect_squeeze(df)
        if squeeze.is_squeezed and exhaustion:
            squeeze.spring_loaded = True
        
        resistance = self._analyze_resistance_confluence(df, current_price, vp)
        options = self._estimate_iv_percentile(df)
        
        # =================================================================
        # SCORING
        # =================================================================
        
        # Price Advance (0-25)
        advance_score = 0
        if advance_pct >= self.ideal_decline_pct:
            advance_score = 25
            factors.append(f"Up {advance_pct:.1f}% from low")
        elif advance_pct >= self.min_decline_pct:
            advance_score = int(15 + (advance_pct - self.min_decline_pct) /
                               (self.ideal_decline_pct - self.min_decline_pct) * 10)
            factors.append(f"Up {advance_pct:.1f}% from low")
        elif advance_pct >= 7:
            advance_score = 10
        
        # Volume (0-25)
        volume_score = 0
        if exhaustion:
            volume_score = 25
            factors.append("Buying climax → exhaustion")
        elif climax:
            volume_score = 15
            factors.append("Buying climax detected")
        elif rvol < 0.6:
            volume_score = 8
        
        # RSI (0-25)
        rsi_score = 0
        if extreme:
            rsi_score = 20
            factors.append(f"RSI extreme ({current_rsi:.0f})")
        elif overbought:
            rsi_score = 15
            factors.append(f"RSI overbought ({current_rsi:.0f})")
        elif current_rsi > 65:
            rsi_score = 10
        
        if divergence.detected:
            rsi_score = min(25, rsi_score + 5)
            factors.append("Bearish RSI divergence")
        
        # Candle (0-25)
        candle_score = 0
        if reversal_candle:
            candle_score = 25
            factors.append(f"Reversal: {pattern_name}")
        elif long_wick:
            candle_score = 15
            factors.append(f"Long upper wick ({pattern_name})")
        
        # Volume Profile (0-15)
        vp_score = 0
        if vp.price_zone == "above_vah":
            vp_score = 15
            factors.append(f"Above VAH ${vp.vah} (euphoria zone)")
        elif vp.price_zone == "vah_poc" and vp.at_key_level:
            vp_score = 12
            factors.append(f"At VAH ${vp.vah}")
        elif vp.at_key_level:
            vp_score = 8
            factors.append(f"Near {vp.nearest_level} ${vp.nearest_level_price}")
        elif vp.price_zone in ("vah_poc", "above_vah"):
            vp_score = 5
        
        # Weekly (0-10)
        weekly_score = 0
        if weekly.supports_short:
            if weekly.trend in ("DOWNTREND", "STRONG_DOWNTREND"):
                weekly_score = 10
                factors.append(f"Rally in weekly {weekly.trend}")
            elif weekly.weekly_close_signal == "BEARISH_REVERSAL":
                weekly_score = 8
                factors.append("Weekly bearish reversal")
            else:
                weekly_score = 5
        elif weekly.trend == "NEUTRAL":
            weekly_score = 3
        
        # Squeeze (0-10)
        squeeze_score = 0
        if squeeze.spring_loaded:
            squeeze_score = 10
            factors.append("SPRING LOADED (squeeze + exhaustion)")
        elif squeeze.is_squeezed:
            squeeze_score = 7
            factors.append(f"Squeeze active ({squeeze.squeeze_days}d)")
        elif squeeze.bb_width_percentile < 20:
            squeeze_score = 4
        
        # Resistance (0-15)
        resistance_score = resistance.confluence_score
        if resistance.confluence_count >= 3:
            factors.append(f"Strong resistance confluence ({resistance.confluence_count} levels)")
        elif resistance.confluence_count >= 2:
            factors.append(f"Resistance confluence ({resistance.confluence_count} levels)")
        
        # IV (0-10)
        iv_score = 0
        if options.iv_regime == "low":
            iv_score = 10
            factors.append(f"Low IV ({options.iv_percentile:.0f}%ile) — cheap puts")
        elif options.iv_regime == "normal":
            iv_score = 7
        elif options.iv_regime == "elevated":
            iv_score = 3
        
        # Consecutive up (0-10)
        consecutive_score = 0
        if consecutive_up >= 3:
            consecutive_score = 10
            factors.append(f"{consecutive_up} consecutive up bars")
        elif consecutive_up >= 2:
            consecutive_score = 5
        
        # Session (0-10)
        session_score = 0
        if session_ctx == 'close':
            session_score = 10
        elif session_ctx == 'open':
            session_score = 5
        
        # =================================================================
        # TOTAL
        # =================================================================
        raw_total = (advance_score + volume_score + rsi_score + candle_score +
                    vp_score + weekly_score + squeeze_score + resistance_score +
                    iv_score + consecutive_score + session_score)
        
        total_score = min(100, int(raw_total * 100 / 180))
        
        if total_score >= 80 and exhaustion:
            level = EuphoriaLevel.EXHAUSTION
        elif total_score >= 60 and climax:
            level = EuphoriaLevel.CLIMAX
        elif total_score >= 45:
            level = EuphoriaLevel.DEVELOPING
        elif total_score >= 25:
            level = EuphoriaLevel.EARLY
        else:
            level = EuphoriaLevel.NONE
        
        # Trade levels (SHORT)
        atr = self._calculate_atr(df)
        entry_zone = (round(current_price * 0.995, 2), round(current_price * 1.005, 2))
        stop_loss = round(current_price + (atr * 1.5), 2)
        target_1 = round(current_price - (atr * 2), 2)
        target_2 = round(current_price - (current_price - recent_low) * 0.382, 2)
        
        setup_type = ""
        entry_trigger = ""
        if squeeze.spring_loaded:
            setup_type = "spring_loaded_reversal_short"
            entry_trigger = f"Enter {options.entry_size} at RSI exhaustion, squeeze + euphoria aligned"
        elif level == EuphoriaLevel.EXHAUSTION:
            setup_type = "euphoria_exhaustion_short"
            entry_trigger = f"Enter {options.entry_size} on reversal candle confirmation"
        elif level == EuphoriaLevel.CLIMAX:
            setup_type = "climax_reversal_short"
            entry_trigger = "Wait for volume dry-up then enter"
        elif vp.price_zone == "above_vah" and overbought:
            setup_type = "vp_exhaustion_short"
            entry_trigger = f"Above VAH ${vp.vah} with RSI {current_rsi:.0f} — watch for reversal"
        
        return EuphoriaMetrics(
            symbol=symbol,
            advance_from_low_pct=round(advance_pct, 2),
            days_since_low=days_since_low,
            current_rvol=round(rvol, 2),
            climax_volume_detected=climax,
            volume_exhaustion=exhaustion,
            avg_up_volume_ratio=round(up_vol_ratio, 2),
            rsi=round(current_rsi, 2),
            rsi_overbought=overbought,
            rsi_extreme=extreme,
            rsi_divergence=divergence,
            reversal_candle=reversal_candle,
            long_upper_wick=long_wick,
            reversal_pattern_name=pattern_name,
            consecutive_up_days=consecutive_up,
            session_context=session_ctx,
            volume_profile=vp,
            vp_score=vp_score,
            weekly=weekly,
            weekly_score=weekly_score,
            squeeze=squeeze,
            squeeze_score=squeeze_score,
            resistance=resistance,
            resistance_score=resistance_score,
            options=options,
            iv_score=iv_score,
            euphoria_score=total_score,
            euphoria_level=level,
            quality_grade=self._quality_grade(total_score),
            factors=factors[:8],
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            setup_type=setup_type,
            entry_trigger=entry_trigger,
            current_price=round(current_price, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def to_dict(self, metrics) -> Dict:
        """Convert metrics to serializable dict"""
        return asdict(metrics)


# =============================================================================
# ALERT FORMATTERS
# =============================================================================

def format_capitulation_alert(metrics: CapitulationMetrics, symbol: str = None) -> str:
    """Format capitulation analysis as alert message"""
    sym = symbol or metrics.symbol
    emoji = {
        CapitulationLevel.NONE: "⚪", CapitulationLevel.EARLY: "🟡",
        CapitulationLevel.DEVELOPING: "🟠", CapitulationLevel.CLIMAX: "🔴",
        CapitulationLevel.EXHAUSTION: "🟢"
    }[metrics.capitulation_level]
    
    lines = [
        f"{emoji} {sym} CAPITULATION ANALYSIS (V2)",
        f"Level: {metrics.capitulation_level.value} | Score: {metrics.capitulation_score}/100 | Grade: {metrics.quality_grade}",
        ""
    ]
    
    if metrics.setup_type:
        lines.append(f"🎯 SETUP: {metrics.setup_type}")
        lines.append(f"   {metrics.entry_trigger}")
        lines.append("")
    
    lines.append(f"📉 DECLINE: {metrics.decline_from_high_pct:.1f}% from high ({metrics.days_since_high}d ago)")
    lines.append(f"📊 VOLUME: RVOL {metrics.current_rvol:.1f}x | Climax: {'✔' if metrics.climax_volume_detected else '✗'} | Exhaustion: {'✔' if metrics.volume_exhaustion else '✗'}")
    lines.append(f"📈 RSI: {metrics.rsi:.1f} {'⚠️ EXTREME' if metrics.rsi_extreme else '(oversold)' if metrics.rsi_oversold else ''}")
    
    if metrics.rsi_divergence and metrics.rsi_divergence.detected:
        lines.append(f"   ✔ Bullish divergence ({metrics.rsi_divergence.bars_apart} bars)")
    
    lines.append(f"🕯️ CANDLE: {metrics.reversal_pattern_name or 'No reversal pattern'}")
    
    if metrics.volume_profile:
        vp = metrics.volume_profile
        lines.append(f"\n📐 VP: VAH ${vp.vah} | POC ${vp.poc} | VAL ${vp.val}")
        lines.append(f"   Zone: {vp.price_zone} | Score: {metrics.vp_score}")
    
    if metrics.weekly:
        lines.append(f"📅 WEEKLY: {metrics.weekly.trend} | Signal: {metrics.weekly.weekly_close_signal or 'none'}")
    
    if metrics.squeeze and metrics.squeeze.spring_loaded:
        lines.append(f"🔥 SPRING LOADED: Squeeze ({metrics.squeeze.squeeze_days}d) + Exhaustion!")
    elif metrics.squeeze and metrics.squeeze.is_squeezed:
        lines.append(f"🔲 Squeeze active: {metrics.squeeze.squeeze_days}d")
    
    if metrics.support and metrics.support.confluence_count > 0:
        lines.append(f"🛡️ Support: {metrics.support.confluence_count} levels clustering")
    
    if metrics.options:
        lines.append(f"📋 IV: {metrics.options.iv_percentile:.0f}%ile ({metrics.options.iv_regime}) | Size: {metrics.options.entry_size}")
    
    if metrics.capitulation_level.tradeable:
        lines.append(f"\n💰 TRADE: Entry ${metrics.entry_zone[0]:.2f}-${metrics.entry_zone[1]:.2f}")
        lines.append(f"   Stop: ${metrics.stop_loss:.2f} | T1: ${metrics.target_1:.2f} | T2: ${metrics.target_2:.2f}")
    
    if metrics.factors:
        lines.append(f"\n✅ Factors: {', '.join(metrics.factors[:6])}")
    
    return "\n".join(lines)


def format_euphoria_alert(metrics: EuphoriaMetrics, symbol: str = None) -> str:
    """Format euphoria analysis as alert message"""
    sym = symbol or metrics.symbol
    emoji = {
        EuphoriaLevel.NONE: "⚪", EuphoriaLevel.EARLY: "🟡",
        EuphoriaLevel.DEVELOPING: "🟠", EuphoriaLevel.CLIMAX: "🔴",
        EuphoriaLevel.EXHAUSTION: "🟢"
    }[metrics.euphoria_level]
    
    lines = [
        f"{emoji} {sym} EUPHORIA ANALYSIS (SHORT) V2",
        f"Level: {metrics.euphoria_level.value} | Score: {metrics.euphoria_score}/100 | Grade: {metrics.quality_grade}",
        ""
    ]
    
    if metrics.setup_type:
        lines.append(f"🎯 SETUP: {metrics.setup_type}")
        lines.append(f"   {metrics.entry_trigger}")
        lines.append("")
    
    lines.append(f"📈 ADVANCE: {metrics.advance_from_low_pct:.1f}% from low ({metrics.days_since_low}d ago)")
    lines.append(f"📊 VOLUME: RVOL {metrics.current_rvol:.1f}x | Climax: {'✔' if metrics.climax_volume_detected else '✗'} | Exhaustion: {'✔' if metrics.volume_exhaustion else '✗'}")
    lines.append(f"📉 RSI: {metrics.rsi:.1f} {'⚠️ EXTREME' if metrics.rsi_extreme else '(overbought)' if metrics.rsi_overbought else ''}")
    
    if metrics.rsi_divergence and metrics.rsi_divergence.detected:
        lines.append(f"   ✔ Bearish divergence ({metrics.rsi_divergence.bars_apart} bars)")
    
    lines.append(f"🕯️ CANDLE: {metrics.reversal_pattern_name or 'No reversal pattern'}")
    
    if metrics.volume_profile:
        vp = metrics.volume_profile
        lines.append(f"\n📐 VP: VAH ${vp.vah} | POC ${vp.poc} | VAL ${vp.val}")
        lines.append(f"   Zone: {vp.price_zone} | Score: {metrics.vp_score}")
    
    if metrics.weekly:
        lines.append(f"📅 WEEKLY: {metrics.weekly.trend} | Signal: {metrics.weekly.weekly_close_signal or 'none'}")
    
    if metrics.squeeze and metrics.squeeze.spring_loaded:
        lines.append(f"🔥 SPRING LOADED: Squeeze ({metrics.squeeze.squeeze_days}d) + Exhaustion!")
    
    if metrics.resistance and metrics.resistance.confluence_count > 0:
        lines.append(f"🧱 Resistance: {metrics.resistance.confluence_count} levels clustering")
    
    if metrics.options:
        lines.append(f"📋 IV: {metrics.options.iv_percentile:.0f}%ile ({metrics.options.iv_regime}) | Size: {metrics.options.entry_size}")
    
    if metrics.euphoria_level.tradeable:
        lines.append(f"\n💰 SHORT: Entry ${metrics.entry_zone[0]:.2f}-${metrics.entry_zone[1]:.2f}")
        lines.append(f"   Stop: ${metrics.stop_loss:.2f} (above) | T1: ${metrics.target_1:.2f} | T2: ${metrics.target_2:.2f}")
    
    if metrics.factors:
        lines.append(f"\n✅ Factors: {', '.join(metrics.factors[:6])}")
    
    return "\n".join(lines)


# =============================================================================
# QUICK SCAN FUNCTIONS
# =============================================================================

def scan_for_capitulation(symbol: str, period: str = "3mo", interval: str = "1d") -> Optional[CapitulationMetrics]:
    """Quick function to scan a symbol for capitulation"""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return None
    detector = CapitulationDetectorV2()
    return detector.analyze(df, symbol)


def scan_for_euphoria(symbol: str, period: str = "3mo", interval: str = "1d") -> Optional[EuphoriaMetrics]:
    """Quick function to scan a symbol for euphoria (short setup)"""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return None
    detector = CapitulationDetectorV2()
    return detector.analyze_euphoria(df, symbol)


def scan_both(symbols: List[str], 
              min_level: str = "DEVELOPING") -> Dict[str, List]:
    """
    Scan multiple symbols for both capitulation and euphoria.
    Returns dict with 'capitulation' and 'euphoria' lists.
    """
    import yfinance as yf
    
    level_order = {"NONE": 0, "EARLY": 1, "DEVELOPING": 2, "CLIMAX": 3, "EXHAUSTION": 4}
    min_val = level_order.get(min_level, 2)
    
    detector = CapitulationDetectorV2()
    results = {"capitulation": [], "euphoria": []}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            if df.empty:
                continue
            
            cap = detector.analyze(df, symbol)
            if cap and level_order.get(cap.capitulation_level.value, 0) >= min_val:
                results["capitulation"].append(cap)
            
            euph = detector.analyze_euphoria(df, symbol)
            if euph and level_order.get(euph.euphoria_level.value, 0) >= min_val:
                results["euphoria"].append(euph)
                
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
    
    results["capitulation"].sort(key=lambda x: x.capitulation_score, reverse=True)
    results["euphoria"].sort(key=lambda x: x.euphoria_score, reverse=True)
    
    return results


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  CAPITULATION & EUPHORIA DETECTOR V2 — C.O.R.E. Methodology")
    print("=" * 65)
    
    test_symbols = ["NVDA", "AAPL", "META", "TSLA", "AMD", "MSFT", "AMZN"]
    
    detector = CapitulationDetectorV2()
    
    for symbol in test_symbols:
        print(f"\nScanning {symbol}...")
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df.empty:
                print(f"  No data for {symbol}")
                continue
            
            # Capitulation
            cap = detector.analyze(df, symbol)
            if cap and cap.capitulation_level != CapitulationLevel.NONE:
                print(format_capitulation_alert(cap))
            else:
                print(f"  {symbol}: No capitulation signals (Score: {cap.capitulation_score if cap else 0})")
            
            print()
            
            # Euphoria
            euph = detector.analyze_euphoria(df, symbol)
            if euph and euph.euphoria_level != EuphoriaLevel.NONE:
                print(format_euphoria_alert(euph))
            else:
                print(f"  {symbol}: No euphoria signals (Score: {euph.euphoria_score if euph else 0})")
            
        except Exception as e:
            print(f"  Error: {e}")
        print("-" * 65)
    
    print(f"\n{'='*65}")
    print("  BATCH SCAN — DEVELOPING+ Only")
    print(f"{'='*65}")
    
    results = scan_both(test_symbols, min_level="DEVELOPING")
    
    if results["capitulation"]:
        print("\n  📉 CAPITULATION SIGNALS:")
        for r in results["capitulation"]:
            print(f"    {r.symbol}: {r.capitulation_score} ({r.capitulation_level.value}) — {r.setup_type or 'monitoring'}")
    
    if results["euphoria"]:
        print("\n  📈 EUPHORIA SIGNALS:")
        for r in results["euphoria"]:
            print(f"    {r.symbol}: {r.euphoria_score} ({r.euphoria_level.value}) — {r.setup_type or 'monitoring'}")
    
    if not results["capitulation"] and not results["euphoria"]:
        print("  No DEVELOPING+ signals found")
