"""
Market Scanner V2 — Multi-Source Data + V2 Enrichment
======================================================
Pulls real-time data from Polygon (primary), Alpaca, Finnhub, or yfinance.
Adds weekly structure, squeeze detection, IV percentile, and enhanced VP
to every analysis — creating a V2 enrichment context that downstream
scanners (integrated, dual setup, overnight) can consume.

Data Source Priority:
    1. Polygon.io (paid = real-time, free = 15-min delayed)
    2. Alpaca (real-time with account)
    3. Finnhub (15-min delayed)
    4. yfinance (fallback)

RETAINED FROM V1:
- TechnicalCalculator with VP, VWAP, RSI, ATR, RVOL, rejection candles
- Enhanced VP (profile shape, HVN/LVN, POC strength, developing flag)
- Multi-source data pipeline (Polygon > Alpaca > Finnhub > yfinance)
- Caching, resampling, quote fetching
- analyze(), analyze_mtf(), scan_symbols(), scan_mtf()
- format/print methods

NEW IN V2:
- TechnicalCalculator is now the canonical V2 version:
  * Wilder's RSI series (not just single value)
  * ATR series for squeeze detection
  * BB/KC squeeze detection
  * IV percentile estimation
  * Weekly structure analysis
  * Volume profile shape classification
- V2 Enrichment Dict: Every analysis returns a v2_context dict with:
  * weekly: trend, structure, supports_long/short
  * squeeze: is_squeezed, days, bb_width_percentile
  * iv: percentile, regime, suggested delta/size/stop
  * vp_enhanced: shape, poc_strength, HVN, LVN, developing
  * rsi_2hr: 2-hour RSI for position management
  * atr: current ATR value
  * rvol: relative volume
- analyze_enriched(): Returns (AnalysisResult, v2_context) tuple
- scan_enriched(): Batch scan returning enriched results
- MarketScanner backward compatible — existing calls still work

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import os
import time
try:
    import finnhub
except ImportError:
    finnhub = None
try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    alpaca_available = True
except ImportError:
    alpaca_available = False
try:
    from polygon import RESTClient as PolygonClient
    polygon_available = True
except ImportError:
    polygon_available = False

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import scanner components
try:
    from chart_input_analyzer import ChartInputSystem, ChartInput, AnalysisResult, MTFResult
    chart_system_available = True
except ImportError:
    chart_system_available = False
    print("⚠️ chart_input_analyzer not found — analysis methods will return raw data only")


# =============================================================================
# V2 DATA STRUCTURES
# =============================================================================

@dataclass
class WeeklyContext:
    """Weekly structure context for MTF alignment"""
    trend: str = "NEUTRAL"
    last_week_structure: str = ""
    weekly_close_position: float = 0.5
    weekly_close_signal: str = ""
    supports_long: bool = False
    supports_short: bool = False
    weekly_high: float = 0.0
    weekly_low: float = 0.0
    weekly_poc: float = 0.0


@dataclass
class SqueezeContext:
    """Bollinger Band / Keltner Channel squeeze state"""
    is_squeezed: bool = False
    squeeze_days: int = 0
    bb_width_percentile: float = 50.0
    bb_width: float = 0.0
    momentum_direction: str = "NEUTRAL"  # UP, DOWN, NEUTRAL


@dataclass
class IVContext:
    """Implied volatility context (HV proxy)"""
    iv_percentile: float = 50.0
    iv_regime: str = "normal"  # low, normal, elevated, extreme
    current_hv: float = 0.0
    suggested_delta: float = 0.65
    min_dte: int = 21
    contract_stop_pct: float = 12.5
    entry_size: str = "50%"
    scale_plan: str = "Enter 50%, add at +15% and +25%"


@dataclass
class EnhancedVP:
    """Enhanced volume profile with shape and node analysis"""
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    vwap: float = 0.0
    profile_shape: str = "normal"  # p-shape, b-shape, d-shape, normal
    developing: bool = True
    poc_strength: int = 50
    high_volume_nodes: List[float] = field(default_factory=list)
    low_volume_nodes: List[float] = field(default_factory=list)
    value_area_width_pct: float = 0.0
    price_position: str = "unknown"  # above_va, in_va, below_va, at_poc


@dataclass
class V2Context:
    """Complete V2 enrichment context — passed to downstream scanners"""
    symbol: str = ""
    current_price: float = 0.0
    timestamp: str = ""

    # Core technicals
    rsi: float = 50.0
    rsi_2hr: float = 50.0
    atr: float = 0.0
    rvol: float = 1.0
    volume_trend: str = "neutral"

    # V2 enrichment
    weekly: WeeklyContext = field(default_factory=WeeklyContext)
    squeeze: SqueezeContext = field(default_factory=SqueezeContext)
    iv: IVContext = field(default_factory=IVContext)
    vp: EnhancedVP = field(default_factory=EnhancedVP)

    def to_dict(self) -> Dict:
        """Convert to dict for API responses and downstream consumption"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'timestamp': self.timestamp,
            'rsi': self.rsi,
            'rsi_2hr': self.rsi_2hr,
            'atr': self.atr,
            'rvol': self.rvol,
            'volume_trend': self.volume_trend,
            'weekly': {
                'trend': self.weekly.trend,
                'last_week_structure': self.weekly.last_week_structure,
                'weekly_close_position': self.weekly.weekly_close_position,
                'weekly_close_signal': self.weekly.weekly_close_signal,
                'supports_long': self.weekly.supports_long,
                'supports_short': self.weekly.supports_short,
                'weekly_high': self.weekly.weekly_high,
                'weekly_low': self.weekly.weekly_low,
                'weekly_poc': self.weekly.weekly_poc,
            },
            'squeeze': {
                'is_squeezed': self.squeeze.is_squeezed,
                'squeeze_days': self.squeeze.squeeze_days,
                'bb_width_percentile': self.squeeze.bb_width_percentile,
                'momentum_direction': self.squeeze.momentum_direction,
            },
            'iv': {
                'iv_percentile': self.iv.iv_percentile,
                'iv_regime': self.iv.iv_regime,
                'current_hv': self.iv.current_hv,
                'suggested_delta': self.iv.suggested_delta,
                'min_dte': self.iv.min_dte,
                'contract_stop_pct': self.iv.contract_stop_pct,
                'entry_size': self.iv.entry_size,
                'scale_plan': self.iv.scale_plan,
            },
            'vp': {
                'poc': self.vp.poc,
                'vah': self.vp.vah,
                'val': self.vp.val,
                'vwap': self.vp.vwap,
                'profile_shape': self.vp.profile_shape,
                'developing': self.vp.developing,
                'poc_strength': self.vp.poc_strength,
                'high_volume_nodes': self.vp.high_volume_nodes,
                'low_volume_nodes': self.vp.low_volume_nodes,
                'value_area_width_pct': self.vp.value_area_width_pct,
                'price_position': self.vp.price_position,
            }
        }


# =============================================================================
# TECHNICAL CALCULATOR V2 (Canonical)
# =============================================================================

class TechnicalCalculator:
    """
    Canonical V2 technical calculator.

    All V2 scanners should import from here rather than duplicating.
    VP settings: 50 bins, 70% value area (matches charting platform).
    RSI: Wilder's smoothing (EMA alpha = 1/period).
    """

    # -------------------------------------------------------------------------
    # VOLUME PROFILE
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame,
                                  value_area_pct: float = 0.70,
                                  num_bins: int = 50) -> Tuple[float, float, float]:
        """
        Calculate POC, VAH, VAL from OHLCV data.

        Returns:
            (POC, VAH, VAL)
        """
        if len(df) < 10:
            mid = df['close'].mean()
            return round(mid, 2), round(mid * 1.01, 2), round(mid * 0.99, 2)

        price_min = df['low'].min()
        price_max = df['high'].max()
        if price_max == price_min:
            return round(price_max, 2), round(price_max, 2), round(price_min, 2)

        bin_size = (price_max - price_min) / num_bins
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

        total_volume = volume_profile.sum()
        target_volume = total_volume * value_area_pct
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
        return round(poc, 2), round(vah, 2), round(val, 2)

    @staticmethod
    def calculate_volume_profile_enhanced(df: pd.DataFrame,
                                           value_area_pct: float = 0.70,
                                           num_bins: int = 50) -> EnhancedVP:
        """
        Enhanced VP with shape analysis, HVN/LVN, POC strength.

        Returns EnhancedVP dataclass (V2 format).
        """
        result = EnhancedVP()
        if len(df) < 10:
            mid = float(df['close'].mean())
            result.poc = round(mid, 2)
            result.vah = round(mid * 1.01, 2)
            result.val = round(mid * 0.99, 2)
            return result

        price_min = df['low'].min()
        price_max = df['high'].max()
        if price_max == price_min:
            result.poc = round(price_max, 2)
            result.vah = round(price_max, 2)
            result.val = round(price_min, 2)
            return result

        bin_size = (price_max - price_min) / num_bins
        bins = np.arange(price_min, price_max + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2
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
        poc = round(float(bin_centers[poc_idx]), 2)

        # Value Area
        total_volume = volume_profile.sum()
        target_volume = total_volume * value_area_pct
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

        val = round(float(bins[va_low_idx]), 2)
        vah = round(float(bins[va_high_idx + 1]), 2)

        # Profile shape
        n = len(volume_profile)
        third = n // 3
        upper_vol = volume_profile[2 * third:].sum()
        middle_vol = volume_profile[third:2 * third].sum()
        lower_vol = volume_profile[:third].sum()
        total = upper_vol + middle_vol + lower_vol

        if total > 0:
            upper_pct = upper_vol / total
            lower_pct = lower_vol / total
            if upper_pct > 0.45:
                profile_shape = "p-shape"
            elif lower_pct > 0.45:
                profile_shape = "b-shape"
            elif middle_vol / total > 0.5:
                profile_shape = "d-shape"
            else:
                profile_shape = "normal"
        else:
            profile_shape = "normal"

        # POC strength
        avg_vol = total_volume / len(volume_profile) if len(volume_profile) > 0 else 1
        poc_strength = min(100, int((volume_profile[poc_idx] / avg_vol) * 25)) if avg_vol > 0 else 50

        # Developing
        active_bins = np.sum(volume_profile > 0)
        developing = active_bins < (num_bins * 0.5)

        # HVN / LVN
        hvn_threshold = avg_vol * 1.5
        lvn_threshold = avg_vol * 0.3
        hvn = [round(float(bin_centers[i]), 2) for i in range(len(volume_profile))
               if volume_profile[i] > hvn_threshold]
        lvn = [round(float(bin_centers[i]), 2) for i in range(len(volume_profile))
               if 0 < volume_profile[i] < lvn_threshold]

        # VA width
        full_range = price_max - price_min
        va_width_pct = round(float((vah - val) / full_range * 100), 1) if full_range > 0 else 0

        # Price position
        current_price = float(df['close'].iloc[-1])
        poc_tolerance = (vah - val) * 0.1
        if abs(current_price - poc) <= poc_tolerance:
            price_position = "at_poc"
        elif current_price > vah:
            price_position = "above_va"
        elif current_price < val:
            price_position = "below_va"
        else:
            price_position = "in_va"

        result.poc = poc
        result.vah = vah
        result.val = val
        result.profile_shape = profile_shape
        result.developing = developing
        result.poc_strength = poc_strength
        result.high_volume_nodes = hvn[:5]
        result.low_volume_nodes = lvn[:5]
        result.value_area_width_pct = va_width_pct
        result.price_position = price_position
        return result

    # -------------------------------------------------------------------------
    # VWAP
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        typical = (df['high'] + df['low'] + df['close']) / 3
        vol_sum = df['volume'].sum()
        if vol_sum == 0:
            return round(float(df['close'].iloc[-1]), 2)
        return round(float((typical * df['volume']).sum() / vol_sum), 2)

    # -------------------------------------------------------------------------
    # RSI (Wilder's)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """Single RSI value from latest close"""
        if len(df) < period + 1:
            return 50.0
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return round(float(val), 2) if not pd.isna(val) else 50.0

    @staticmethod
    def calculate_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
        """Full RSI series using Wilder's smoothing"""
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        return 100 - (100 / (1 + rs))

    # -------------------------------------------------------------------------
    # ATR
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return float((df['high'].max() - df['low'].min()) / max(len(df), 1))
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        val = atr.iloc[-1]
        return round(float(val), 4) if not pd.isna(val) else 0.0

    @staticmethod
    def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # -------------------------------------------------------------------------
    # VOLUME METRICS
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_relative_volume(df: pd.DataFrame, lookback: int = 20) -> float:
        if len(df) < lookback + 1 or 'volume' not in df.columns:
            return 1.0
        avg_volume = df['volume'].iloc[-(lookback + 1):-1].mean()
        if avg_volume <= 0:
            return 1.0
        return round(float(df['volume'].iloc[-1] / avg_volume), 2)

    @staticmethod
    def calculate_volume_trend(df: pd.DataFrame, periods: int = 5) -> str:
        if len(df) < periods + 1:
            return "neutral"
        recent_vol = df['volume'].iloc[-periods:].mean()
        prior_start = -(periods * 2) if len(df) >= periods * 2 else 0
        prior_vol = df['volume'].iloc[prior_start:-periods].mean()
        if prior_vol <= 0:
            return "neutral"
        change = (recent_vol - prior_vol) / prior_vol
        if change > 0.15:
            return "increasing"
        elif change < -0.15:
            return "decreasing"
        return "neutral"

    @staticmethod
    def detect_volume_divergence(df: pd.DataFrame, periods: int = 5) -> bool:
        if len(df) < periods + 1:
            return False
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-periods - 1]) / df['close'].iloc[-periods - 1]
        vol_start = df['volume'].iloc[-periods - 1:-1].mean()
        vol_end = df['volume'].iloc[-periods:].mean()
        vol_change = (vol_end - vol_start) / vol_start if vol_start > 0 else 0
        return abs(price_change) > 0.02 and vol_change < -0.20

    # -------------------------------------------------------------------------
    # CANDLE PATTERNS
    # -------------------------------------------------------------------------

    @staticmethod
    def is_rejection_candle(df: pd.DataFrame, direction: str, wick_ratio: float = 0.6) -> bool:
        if len(df) < 1:
            return False
        candle = df.iloc[-1]
        candle_range = candle['high'] - candle['low']
        if candle_range <= 0:
            return False
        body_top = max(candle['open'], candle['close'])
        body_bottom = min(candle['open'], candle['close'])
        if direction == "bullish":
            return (body_bottom - candle['low']) / candle_range >= wick_ratio
        else:
            return (candle['high'] - body_top) / candle_range >= wick_ratio

    @staticmethod
    def get_extension_from_level(price: float, level: float, atr: float) -> float:
        if atr <= 0:
            return 0.0
        return round((price - level) / atr, 2)

    # -------------------------------------------------------------------------
    # V2: SQUEEZE DETECTION
    # -------------------------------------------------------------------------

    @staticmethod
    def detect_squeeze(df: pd.DataFrame,
                       bb_period: int = 20, bb_std: float = 2.0,
                       kc_period: int = 20, kc_mult: float = 1.5) -> SqueezeContext:
        """Detect BB inside KC squeeze with momentum direction"""
        ctx = SqueezeContext()
        if len(df) < bb_period + 5:
            return ctx

        sma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        bb_upper = sma + (bb_std * std)
        bb_lower = sma - (bb_std * std)

        ema = df['close'].ewm(span=kc_period, adjust=False).mean()
        atr_s = TechnicalCalculator.calculate_atr_series(df, kc_period)
        kc_upper = ema + (kc_mult * atr_s)
        kc_lower = ema - (kc_mult * atr_s)

        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        ctx.is_squeezed = bool(squeeze.iloc[-1]) if len(squeeze) > 0 else False

        # Count consecutive squeeze days
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]:
                ctx.squeeze_days += 1
            else:
                break

        # BB width percentile
        bb_width = ((bb_upper - bb_lower) / sma).dropna()
        if len(bb_width) >= 20:
            current_w = float(bb_width.iloc[-1])
            lookback = bb_width.tail(60) if len(bb_width) >= 60 else bb_width
            ctx.bb_width_percentile = round(float((lookback < current_w).sum() / len(lookback) * 100), 1)
            ctx.bb_width = round(current_w, 4)

        # Momentum direction (price vs SMA slope)
        if len(sma.dropna()) >= 3:
            slope = float(sma.iloc[-1] - sma.iloc[-3])
            price_vs_sma = float(df['close'].iloc[-1] - sma.iloc[-1])
            if slope > 0 and price_vs_sma > 0:
                ctx.momentum_direction = "UP"
            elif slope < 0 and price_vs_sma < 0:
                ctx.momentum_direction = "DOWN"
            else:
                ctx.momentum_direction = "NEUTRAL"

        return ctx

    # -------------------------------------------------------------------------
    # V2: IV PERCENTILE (HV proxy)
    # -------------------------------------------------------------------------

    @staticmethod
    def estimate_iv_percentile(df: pd.DataFrame) -> IVContext:
        ctx = IVContext()
        if len(df) < 60:
            return ctx

        log_returns = np.log(df['close'] / df['close'].shift(1))
        hv_series = log_returns.rolling(window=20).std() * np.sqrt(252) * 100
        hv_series = hv_series.dropna()
        if len(hv_series) < 20:
            return ctx

        current_hv = float(hv_series.iloc[-1])
        lookback = hv_series.tail(60)
        percentile = float((lookback < current_hv).sum() / len(lookback) * 100)

        ctx.iv_percentile = round(percentile, 1)
        ctx.current_hv = round(current_hv, 2)

        if percentile < 20:
            ctx.iv_regime = "low"
            ctx.suggested_delta = 0.70
            ctx.entry_size = "50%"
            ctx.scale_plan = "Enter 50%, add at +15% and +25% — cheap options"
            ctx.contract_stop_pct = 12.5
        elif percentile < 50:
            ctx.iv_regime = "normal"
            ctx.suggested_delta = 0.65
            ctx.entry_size = "50%"
            ctx.scale_plan = "Enter 50%, add at +15% and +25%"
            ctx.contract_stop_pct = 12.5
        elif percentile < 80:
            ctx.iv_regime = "elevated"
            ctx.suggested_delta = 0.65
            ctx.entry_size = "40%"
            ctx.scale_plan = "Enter 40%, add at +15% and +25%"
            ctx.contract_stop_pct = 12.5
        else:
            ctx.iv_regime = "extreme"
            ctx.suggested_delta = 0.60
            ctx.entry_size = "25%"
            ctx.scale_plan = "Scale in as IV normalizes"
            ctx.contract_stop_pct = 15.0

        return ctx

    # -------------------------------------------------------------------------
    # V2: WEEKLY STRUCTURE
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_weekly_context(symbol: str) -> WeeklyContext:
        ctx = WeeklyContext()
        try:
            if yf is None:
                return ctx
            ticker = yf.Ticker(symbol)
            df_w = ticker.history(period="6mo", interval="1wk")
            if df_w.empty or len(df_w) < 6:
                return ctx

            df_w.columns = [c.lower() for c in df_w.columns]
            weeks = df_w.tail(8)

            ll_count = hh_count = lh_count = hl_count = 0
            last_structure = ""
            for i in range(1, len(weeks)):
                curr, prev = weeks.iloc[i], weeks.iloc[i - 1]
                s = ""
                if curr['high'] > prev['high'] * 1.001:
                    s += "HH"; hh_count += 1
                elif curr['high'] < prev['high'] * 0.999:
                    s += "LH"; lh_count += 1
                else:
                    s += "EQ"
                if curr['low'] > prev['low'] * 1.001:
                    s += "+HL"; hl_count += 1
                elif curr['low'] < prev['low'] * 0.999:
                    s += "+LL"; ll_count += 1
                else:
                    s += "+EQ"
                last_structure = s

            bearish = ll_count + lh_count
            bullish = hh_count + hl_count

            if bearish >= 8 and bullish <= 2:
                ctx.trend = "STRONG_DOWNTREND"
            elif bearish >= 5 and bearish > bullish * 2:
                ctx.trend = "DOWNTREND"
            elif bullish >= 8 and bearish <= 2:
                ctx.trend = "STRONG_UPTREND"
            elif bullish >= 5 and bullish > bearish * 2:
                ctx.trend = "UPTREND"
            else:
                ctx.trend = "NEUTRAL"

            ctx.last_week_structure = last_structure

            # Last completed week close position
            lw = weeks.iloc[-2]
            lw_range = lw['high'] - lw['low']
            ctx.weekly_close_position = round(float((lw['close'] - lw['low']) / lw_range), 2) if lw_range > 0 else 0.5
            ctx.weekly_high = round(float(lw['high']), 2)
            ctx.weekly_low = round(float(lw['low']), 2)

            # Weekly close signal
            wcp = ctx.weekly_close_position
            if "LL" in last_structure and wcp > 0.70:
                ctx.weekly_close_signal = "BULLISH_REVERSAL"
            elif "HH" in last_structure and wcp < 0.30:
                ctx.weekly_close_signal = "BEARISH_REVERSAL"
            elif wcp > 0.75:
                ctx.weekly_close_signal = "STRONG_BULL_CLOSE"
            elif wcp < 0.25:
                ctx.weekly_close_signal = "STRONG_BEAR_CLOSE"

            ctx.supports_long = (ctx.trend in ("UPTREND", "STRONG_UPTREND") or
                                ctx.weekly_close_signal in ("BULLISH_REVERSAL", "STRONG_BULL_CLOSE") or
                                (ctx.trend == "NEUTRAL" and wcp > 0.6))
            ctx.supports_short = (ctx.trend in ("DOWNTREND", "STRONG_DOWNTREND") or
                                 ctx.weekly_close_signal in ("BEARISH_REVERSAL", "STRONG_BEAR_CLOSE") or
                                 (ctx.trend == "NEUTRAL" and wcp < 0.4))

            # Weekly POC (simplified — midpoint of last week)
            ctx.weekly_poc = round(float((lw['high'] + lw['low']) / 2), 2)

        except Exception as e:
            pass  # Return default context

        return ctx


# =============================================================================
# MARKET SCANNER V2
# =============================================================================

class MarketScanner:
    """
    Multi-source market data scanner with V2 enrichment.

    Priority: Polygon > Alpaca > Finnhub > yfinance

    V2 additions:
    - analyze_enriched(): returns (AnalysisResult, V2Context)
    - scan_enriched(): batch scan with V2 context
    - build_v2_context(): standalone context builder
    """

    TIMEFRAMES = {
        "1MIN": "1", "5MIN": "5", "15MIN": "15", "30MIN": "30",
        "1HR": "60", "2HR": "60", "4HR": "60",
        "DAILY": "D", "WEEKLY": "W"
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")

        # Finnhub client
        self.client = None
        if finnhub and self.api_key:
            self.client = finnhub.Client(api_key=self.api_key)

        self.calc = TechnicalCalculator()

        # Chart input system (V1 analysis engine)
        self.system = None
        if chart_system_available:
            self.system = ChartInputSystem()

        # Cache
        self._cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_minutes = 1

        # Initialize Polygon
        self.polygon_client = None
        polygon_key = os.environ.get("POLYGON_API_KEY")
        if polygon_available and polygon_key:
            try:
                self.polygon_client = PolygonClient(polygon_key)
            except Exception:
                pass

        # Initialize Alpaca
        self.alpaca_client = None
        alpaca_key = os.environ.get("ALPACA_API_KEY")
        alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
        if alpaca_available and alpaca_key and alpaca_secret:
            try:
                self.alpaca_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
            except Exception:
                pass

    # =========================================================================
    # DATA FETCHING (unchanged from V1 — all 4 sources)
    # =========================================================================

    def _get_candles_polygon(self, symbol: str, resolution: str = "60",
                              days_back: int = 30) -> Optional[pd.DataFrame]:
        if self.polygon_client is None:
            return None
        try:
            timespan_map = {
                "1": ("1", "minute"), "5": ("5", "minute"),
                "15": ("15", "minute"), "30": ("30", "minute"),
                "60": ("1", "hour"), "D": ("1", "day"), "W": ("1", "week")
            }
            multiplier, timespan = timespan_map.get(resolution, ("1", "hour"))
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            bars = self.polygon_client.get_aggs(
                ticker=symbol, multiplier=int(multiplier), timespan=timespan,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"), limit=50000
            )
            if not bars:
                return None
            records = [{'timestamp': pd.to_datetime(b.timestamp, unit='ms'),
                        'open': float(b.open), 'high': float(b.high),
                        'low': float(b.low), 'close': float(b.close),
                        'volume': int(b.volume)} for b in bars]
            df = pd.DataFrame(records).set_index('timestamp').sort_index()
            return df
        except Exception:
            return None

    def _get_candles_alpaca(self, symbol: str, resolution: str = "60",
                             days_back: int = 30) -> Optional[pd.DataFrame]:
        if self.alpaca_client is None:
            return None
        try:
            tf_map = {
                "1": TimeFrame(1, TimeFrameUnit.Minute),
                "5": TimeFrame(5, TimeFrameUnit.Minute),
                "15": TimeFrame(15, TimeFrameUnit.Minute),
                "30": TimeFrame(30, TimeFrameUnit.Minute),
                "60": TimeFrame(1, TimeFrameUnit.Hour),
                "D": TimeFrame(1, TimeFrameUnit.Day),
                "W": TimeFrame(1, TimeFrameUnit.Week)
            }
            timeframe = tf_map.get(resolution, TimeFrame(1, TimeFrameUnit.Hour))
            end = datetime.now()
            start = end - timedelta(days=days_back)
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe,
                                       start=start, end=end)
            bars = self.alpaca_client.get_stock_bars(request)
            if symbol not in bars.data or len(bars.data[symbol]) == 0:
                return None
            records = [{'timestamp': b.timestamp, 'open': float(b.open),
                        'high': float(b.high), 'low': float(b.low),
                        'close': float(b.close), 'volume': int(b.volume)}
                       for b in bars.data[symbol]]
            df = pd.DataFrame(records).set_index('timestamp').sort_index()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception:
            return None

    def _get_candles_yfinance(self, symbol: str, resolution: str = "60",
                               days_back: int = 30) -> Optional[pd.DataFrame]:
        if yf is None:
            return None
        try:
            interval_map = {
                "1": "1m", "5": "5m", "15": "15m", "30": "30m",
                "60": "1h", "D": "1d", "W": "1wk"
            }
            interval = interval_map.get(resolution, "1h")
            period_map = {
                "1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d",
                "1h": f"{min(days_back, 729)}d", "1d": f"{days_back}d",
                "1wk": f"{days_back}d"
            }
            period = period_map.get(interval, f"{days_back}d")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return None
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                     'Close': 'close', 'Volume': 'volume'})
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index.name = 'timestamp'
            return df
        except Exception:
            return None

    def _get_candles(self, symbol: str, resolution: str = "60",
                      days_back: int = 30) -> Optional[pd.DataFrame]:
        """Fetch candles: Polygon > Alpaca > Finnhub > yfinance"""
        cache_key = f"{symbol}_{resolution}_{days_back}"
        if cache_key in self._cache:
            df, ts = self._cache[cache_key]
            if (datetime.now() - ts).seconds < self._cache_minutes * 60:
                return df

        df = None

        # 1. Polygon
        if self.polygon_client is not None:
            df = self._get_candles_polygon(symbol, resolution, days_back)

        # 2. Alpaca
        if df is None and self.alpaca_client is not None:
            df = self._get_candles_alpaca(symbol, resolution, days_back)

        # 3. Finnhub
        if df is None and self.client is not None:
            try:
                end_time = int(datetime.now().timestamp())
                start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
                data = self.client.stock_candles(
                    symbol=symbol, resolution=resolution,
                    _from=start_time, to=end_time
                )
                if data.get('s') == 'ok' and data.get('c'):
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data['t'], unit='s'),
                        'open': data['o'], 'high': data['h'],
                        'low': data['l'], 'close': data['c'],
                        'volume': data['v']
                    }).set_index('timestamp').sort_index()
            except Exception:
                pass

        # 4. yfinance
        if df is None:
            df = self._get_candles_yfinance(symbol, resolution, days_back)

        if df is not None and len(df) > 0:
            self._cache[cache_key] = (df, datetime.now())

        return df

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        resample_map = {
            "30MIN": "30min", "1HR": "1h", "2HR": "2h", "4HR": "4h", "DAILY": "1D"
        }
        rule = resample_map.get(timeframe.upper())
        if not rule:
            return df
        return df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

    # =========================================================================
    # QUOTE FETCHING (all 4 sources, unchanged)
    # =========================================================================

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Real-time quote: Polygon > Alpaca > Finnhub > yfinance"""
        # Polygon
        if self.polygon_client:
            try:
                last_trade = self.polygon_client.get_last_trade(symbol)
                if last_trade and last_trade.price:
                    return {'current': last_trade.price, 'source': 'polygon_realtime',
                            'timestamp': datetime.now()}
            except Exception:
                pass

        # Alpaca
        if self.alpaca_client:
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote_data = self.alpaca_client.get_stock_latest_quote(request)
                if symbol in quote_data:
                    q = quote_data[symbol]
                    mid = (q.bid_price + q.ask_price) / 2 if q.bid_price and q.ask_price else (q.ask_price or q.bid_price)
                    return {'current': mid, 'source': 'alpaca_realtime',
                            'timestamp': q.timestamp}
            except Exception:
                pass

        # Finnhub
        if self.client:
            try:
                quote = self.client.quote(symbol)
                if quote and quote.get('c'):
                    return {
                        'current': quote['c'], 'open': quote.get('o'),
                        'high': quote.get('h'), 'low': quote.get('l'),
                        'prev_close': quote.get('pc'),
                        'change': quote.get('d'), 'change_pct': quote.get('dp'),
                        'source': 'finnhub_delayed',
                        'timestamp': datetime.fromtimestamp(quote.get('t', 0))
                    }
            except Exception:
                pass

        # yfinance
        if yf:
            try:
                ticker = yf.Ticker(symbol)
                fi = ticker.fast_info
                if hasattr(fi, 'last_price') and fi.last_price:
                    return {
                        'current': float(fi.last_price),
                        'open': float(fi.open) if hasattr(fi, 'open') else None,
                        'high': float(fi.day_high) if hasattr(fi, 'day_high') else None,
                        'low': float(fi.day_low) if hasattr(fi, 'day_low') else None,
                        'prev_close': float(fi.previous_close) if hasattr(fi, 'previous_close') else None,
                        'source': 'yfinance_realtime',
                        'timestamp': datetime.now()
                    }
            except Exception:
                pass
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d', interval='1m')
                if hist is not None and len(hist) > 0:
                    return {
                        'current': float(hist['Close'].iloc[-1]),
                        'source': 'yfinance_history',
                        'timestamp': datetime.now()
                    }
            except Exception:
                pass

        return None

    # =========================================================================
    # V2: BUILD ENRICHMENT CONTEXT
    # =========================================================================

    def build_v2_context(self, symbol: str,
                          df: pd.DataFrame = None,
                          timeframe: str = "1HR",
                          days_back: int = 60) -> V2Context:
        """
        Build complete V2 enrichment context for a symbol.

        This is the central V2 value-add: weekly, squeeze, IV, enhanced VP
        all computed and packaged for downstream consumption.

        Args:
            symbol: Stock symbol
            df: Pre-fetched DataFrame (optional — will fetch if None)
            timeframe: Analysis timeframe
            days_back: History depth

        Returns:
            V2Context with all enrichment data
        """
        ctx = V2Context(symbol=symbol, timestamp=datetime.now().isoformat())

        # Fetch data if not provided
        if df is None:
            resolution = "D" if timeframe.upper() == "DAILY" else "60"
            df = self._get_candles(symbol, resolution, days_back)

        if df is None or len(df) < 20:
            return ctx

        # Normalize columns
        if 'Close' in df.columns:
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                     'Close': 'close', 'Volume': 'volume'})

        # Resample if needed
        if timeframe.upper() in ("2HR", "4HR", "30MIN"):
            df_tf = self._resample_to_timeframe(df, timeframe)
        else:
            df_tf = df

        if len(df_tf) < 10:
            return ctx

        # Current price
        quote = self.get_quote(symbol)
        ctx.current_price = quote['current'] if quote else float(df_tf['close'].iloc[-1])

        # Core technicals
        ctx.rsi = self.calc.calculate_rsi(df_tf)
        ctx.atr = self.calc.calculate_atr(df_tf)
        ctx.rvol = self.calc.calculate_relative_volume(df_tf)
        ctx.volume_trend = self.calc.calculate_volume_trend(df_tf)

        # 2HR RSI for position management
        df_2hr = self._resample_to_timeframe(df, "2HR") if timeframe.upper() != "2HR" else df_tf
        if len(df_2hr) >= 15:
            ctx.rsi_2hr = self.calc.calculate_rsi(df_2hr)

        # Enhanced VP
        vp_lookback = min(60, len(df_tf))
        ctx.vp = self.calc.calculate_volume_profile_enhanced(df_tf.tail(vp_lookback))
        ctx.vp.vwap = self.calc.calculate_vwap(df_tf)

        # Weekly structure
        ctx.weekly = self.calc.calculate_weekly_context(symbol)

        # Squeeze detection
        ctx.squeeze = self.calc.detect_squeeze(df_tf)

        # IV percentile (needs daily data for HV)
        df_daily = self._get_candles(symbol, "D", 90)
        if df_daily is not None and len(df_daily) >= 60:
            if 'Close' in df_daily.columns:
                df_daily = df_daily.rename(columns={'Close': 'close', 'High': 'high',
                                                     'Low': 'low', 'Open': 'open',
                                                     'Volume': 'volume'})
            ctx.iv = self.calc.estimate_iv_percentile(df_daily)

        return ctx

    # =========================================================================
    # V1 ANALYSIS METHODS (backward compatible)
    # =========================================================================

    def analyze(self, symbol: str, timeframe: str = "1HR",
                days_back: int = 20):
        """V1-compatible single timeframe analysis"""
        if self.system is None:
            return None

        resolution = "D" if timeframe.upper() == "DAILY" else "60"
        df = self._get_candles(symbol, resolution, days_back)
        if df is None or len(df) < 20:
            return None

        if timeframe.upper() in ("2HR", "4HR", "30MIN"):
            df = self._resample_to_timeframe(df, timeframe)
        if len(df) < 10:
            return None

        quote = self.get_quote(symbol)
        current_price = quote['current'] if quote else df['close'].iloc[-1]

        poc, vah, val = self.calc.calculate_volume_profile(df)
        vwap = self.calc.calculate_vwap(df)
        rsi = self.calc.calculate_rsi(df)
        rvol = self.calc.calculate_relative_volume(df)
        volume_trend = self.calc.calculate_volume_trend(df)
        volume_divergence = self.calc.detect_volume_divergence(df)
        atr = self.calc.calculate_atr(df)

        has_rejection = False
        if current_price < val:
            has_rejection = self.calc.is_rejection_candle(df, "bullish")
        elif current_price > vah:
            has_rejection = self.calc.is_rejection_candle(df, "bearish")

        return self.system.analyze(
            symbol=symbol, price=current_price,
            vah=vah, poc=poc, val=val, vwap=vwap, rsi=rsi,
            timeframe=timeframe, rvol=rvol,
            volume_trend=volume_trend, volume_divergence=volume_divergence,
            atr=atr, has_rejection=has_rejection
        )

    def analyze_mtf(self, symbol: str, timeframes: List[str] = None):
        """V1-compatible MTF analysis"""
        if self.system is None:
            return None
        if timeframes is None:
            timeframes = ["30MIN", "1HR", "2HR", "4HR"]

        df_hourly = self._get_candles(symbol, "60", days_back=30)
        if df_hourly is None or len(df_hourly) < 50:
            return None

        quote = self.get_quote(symbol)
        current_price = quote['current'] if quote else df_hourly['close'].iloc[-1]

        tf_data = {}
        for tf in timeframes:
            if tf.upper() == "30MIN":
                df = df_hourly.copy()
            else:
                df = self._resample_to_timeframe(df_hourly, tf)
            if len(df) < 10:
                continue
            poc, vah, val = self.calc.calculate_volume_profile(df)
            vwap = self.calc.calculate_vwap(df)
            rsi = self.calc.calculate_rsi(df)
            tf_data[tf.upper()] = {
                "price": current_price, "vah": vah, "poc": poc,
                "val": val, "vwap": vwap, "rsi": rsi
            }

        if not tf_data:
            return None
        return self.system.analyze_mtf(symbol, tf_data, current_price)

    # =========================================================================
    # V2: ENRICHED ANALYSIS
    # =========================================================================

    def analyze_enriched(self, symbol: str, timeframe: str = "1HR",
                          days_back: int = 60) -> Tuple[Optional[object], V2Context]:
        """
        V2 enriched analysis: returns both the V1 AnalysisResult AND V2Context.

        Usage:
            result, ctx = scanner.analyze_enriched("NVDA")
            print(f"Signal: {result.signal}, Weekly: {ctx.weekly.trend}")
        """
        # Build V2 context
        ctx = self.build_v2_context(symbol, timeframe=timeframe, days_back=days_back)

        # Run V1 analysis
        result = self.analyze(symbol, timeframe, days_back)

        return result, ctx

    def scan_symbols(self, symbols: List[str],
                      timeframe: str = "1HR") -> List:
        """V1-compatible batch scan"""
        results = []
        for i, symbol in enumerate(symbols):
            result = self.analyze(symbol, timeframe)
            if result:
                results.append(result)
            if i < len(symbols) - 1:
                time.sleep(1)

        def sort_key(r):
            signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2}
            return (signal_order.get(r.signal, 3), -r.confidence)
        results.sort(key=sort_key)
        return results

    def scan_mtf(self, symbols: List[str]) -> List:
        """V1-compatible MTF batch scan"""
        results = []
        for i, symbol in enumerate(symbols):
            result = self.analyze_mtf(symbol)
            if result:
                results.append(result)
            if i < len(symbols) - 1:
                time.sleep(2)

        def sort_key(r):
            signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2}
            return (signal_order.get(r.dominant_signal, 3), -r.confluence_pct)
        results.sort(key=sort_key)
        return results

    def scan_enriched(self, symbols: List[str],
                       timeframe: str = "1HR",
                       days_back: int = 60) -> List[Tuple[Optional[object], V2Context]]:
        """
        V2 batch scan: returns list of (AnalysisResult, V2Context) tuples
        sorted by squeeze + weekly alignment quality.
        """
        results = []
        for i, symbol in enumerate(symbols):
            try:
                result, ctx = self.analyze_enriched(symbol, timeframe, days_back)
                results.append((result, ctx))
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
            if i < len(symbols) - 1:
                time.sleep(1)

        # Sort: squeeze active first, then weekly aligned, then by RSI extremes
        def quality_key(item):
            _, ctx = item
            score = 0
            if ctx.squeeze.is_squeezed:
                score += 50
            if ctx.weekly.supports_long or ctx.weekly.supports_short:
                score += 30
            if ctx.rsi < 35 or ctx.rsi > 65:
                score += 20
            return -score

        results.sort(key=quality_key)
        return results

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def print_analysis(self, result) -> str:
        if self.system and result:
            return self.system.print_result(result)
        return ""

    def print_mtf_analysis(self, result) -> str:
        if self.system and result:
            return self.system.print_mtf_result(result)
        return ""

    def print_scan_summary(self, results: List) -> str:
        lines = ["=" * 70,
                 f"SCAN RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 "=" * 70]
        long_setups = [r for r in results if hasattr(r, 'signal') and r.signal == "LONG_SETUP"]
        short_setups = [r for r in results if hasattr(r, 'signal') and r.signal == "SHORT_SETUP"]
        yellow = [r for r in results if hasattr(r, 'signal') and r.signal == "YELLOW"]

        lines.append(f"\n🟢 LONG SETUPS: {len(long_setups)}")
        for r in long_setups:
            lines.append(f"   {r.timeframe}: Bull {r.bull_score:.0f} | Conf {r.confidence:.0f}%")
        lines.append(f"\n🔴 SHORT SETUPS: {len(short_setups)}")
        for r in short_setups:
            lines.append(f"   {r.timeframe}: Bear {r.bear_score:.0f} | Conf {r.confidence:.0f}%")
        lines.append(f"\n🟡 YELLOW (Watch): {len(yellow)}")
        for r in yellow[:5]:
            lean = "Bull" if r.bull_score > r.bear_score else "Bear"
            lines.append(f"   {r.timeframe}: {lean} lean | Conf {r.confidence:.0f}%")
        lines.append("=" * 70)
        return "\n".join(lines)

    def print_v2_context(self, ctx: V2Context) -> str:
        """Print V2 enrichment context summary"""
        lines = [
            f"{'='*60}",
            f"  V2 CONTEXT: {ctx.symbol} @ ${ctx.current_price:.2f}",
            f"{'='*60}",
            f"",
            f"📊 VOLUME PROFILE:",
            f"   POC: ${ctx.vp.poc:.2f}  VAH: ${ctx.vp.vah:.2f}  VAL: ${ctx.vp.val:.2f}",
            f"   VWAP: ${ctx.vp.vwap:.2f}  Shape: {ctx.vp.profile_shape}",
            f"   Position: {ctx.vp.price_position}  POC Strength: {ctx.vp.poc_strength}/100",
            f"   VA Width: {ctx.vp.value_area_width_pct}%  {'📍 Developing' if ctx.vp.developing else '✅ Established'}",
        ]
        if ctx.vp.high_volume_nodes:
            lines.append(f"   HVN: {', '.join(f'${x:.2f}' for x in ctx.vp.high_volume_nodes[:3])}")
        if ctx.vp.low_volume_nodes:
            lines.append(f"   LVN: {', '.join(f'${x:.2f}' for x in ctx.vp.low_volume_nodes[:3])}")

        lines.extend([
            f"",
            f"📈 TECHNICALS:",
            f"   RSI: {ctx.rsi:.1f}  RSI(2HR): {ctx.rsi_2hr:.1f}  ATR: ${ctx.atr:.2f}  RVOL: {ctx.rvol:.2f}x",
            f"   Volume Trend: {ctx.volume_trend}",
            f"",
            f"📅 WEEKLY STRUCTURE:",
            f"   Trend: {ctx.weekly.trend}  Structure: {ctx.weekly.last_week_structure}",
            f"   Close Position: {ctx.weekly.weekly_close_position:.0%}  Signal: {ctx.weekly.weekly_close_signal or 'none'}",
            f"   {'✅ Supports LONG' if ctx.weekly.supports_long else ''}{'✅ Supports SHORT' if ctx.weekly.supports_short else ''}",
        ])

        sq = ctx.squeeze
        sq_status = f"🔲 ACTIVE ({sq.squeeze_days} days)" if sq.is_squeezed else "No squeeze"
        lines.extend([
            f"",
            f"🔲 SQUEEZE: {sq_status}",
            f"   BB Width: {sq.bb_width_percentile:.0f}%ile  Momentum: {sq.momentum_direction}",
        ])

        iv = ctx.iv
        lines.extend([
            f"",
            f"📋 IV CONTEXT:",
            f"   Percentile: {iv.iv_percentile:.0f}%  Regime: {iv.iv_regime.upper()}  HV: {iv.current_hv:.1f}%",
            f"   Delta: {iv.suggested_delta}  DTE: {iv.min_dte}+  Stop: -{iv.contract_stop_pct}%",
            f"   Entry: {iv.entry_size}  Plan: {iv.scale_plan}",
            f"{'='*60}",
        ])
        return "\n".join(lines)


# =============================================================================
# V1 COMPATIBILITY ALIASES
# =============================================================================

# The old name FinnhubScanner was used in some places
FinnhubScanner = MarketScanner


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

def quick_analyze(symbol: str, api_key: str = None):
    scanner = MarketScanner(api_key)
    return scanner.analyze(symbol)


def quick_mtf(symbol: str, api_key: str = None):
    scanner = MarketScanner(api_key)
    return scanner.analyze_mtf(symbol)


def quick_scan(symbols: List[str], api_key: str = None):
    scanner = MarketScanner(api_key)
    return scanner.scan_symbols(symbols)


def quick_v2_context(symbol: str) -> V2Context:
    """Quick V2 context for any symbol"""
    scanner = MarketScanner()
    return scanner.build_v2_context(symbol)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  MARKET SCANNER V2 (Polygon > Alpaca > Finnhub > yfinance)")
    print("=" * 60)

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    try:
        scanner = MarketScanner()

        print(f"\n📊 Building V2 context for {symbol}...")
        ctx = scanner.build_v2_context(symbol)
        print(scanner.print_v2_context(ctx))

        if chart_system_available:
            print(f"\n📊 V1 Analysis for {symbol}...")
            result = scanner.analyze(symbol, "1HR")
            if result:
                print(scanner.print_analysis(result))

    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage: python market_scanner_v2.py [SYMBOL]")
        print("Set env: POLYGON_API_KEY, ALPACA_API_KEY, FINNHUB_API_KEY")
