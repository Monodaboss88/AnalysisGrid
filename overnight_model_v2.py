"""
Overnight & Gap Predictive Model V2 — C.O.R.E. Methodology
=============================================================
Analyzes overnight sessions, gaps, and pre-market activity to predict
intraday directional bias and key levels.

WHAT IT DOES:
The overnight session (6PM–9:30AM ET) reveals what institutions are doing
while retail sleeps. Gaps between sessions show conviction. This model
combines overnight behavior, gap statistics, prior day structure, and
opening location into a scored prediction for the trading day.

RETAINED FROM V1:
- RTH/ETH session separation
- Gap type classification (large/small up/down)
- Gap fill probability estimation
- Prior day type classification (trend/range/reversal/inside/outside)
- Opening scenario analysis (above all, below all, in range, etc.)
- Overnight session metrics (high/low/midpoint/delta)
- Overnight value area calculation
- Bull/bear scenario generation

NEW IN V2:
- Self-Contained VP — calculates prior day POC/VAH/VAL from data
- Weekly Structure — MTF trend context for gap behavior
- Wilder's RSI — momentum context at open
- Squeeze Co-Detection — gap into/out of squeeze = directional catalyst
- IV Percentile — options pricing for gap trades
- Enhanced Gap Fill Analysis — actual historical fill rate from data
- Gap vs ATR Normalization — gap significance relative to normal range
- Scoring System — 0-100 with quality grades (A+ through F)
- Setup Classification — gap_and_go_long, gap_fade_short, etc.
- Entry Triggers — specific action steps for each setup
- Batch Scanning — scan_symbols() for morning watchlist sweep
- Integration Ready — works with squeeze/capitulation/extension V2

Tiers:
- STRONG_BULLISH:  75+ bull score — gap & go long or buy the dip
- BULLISH:         60-74 — lean long, watch key levels
- NEUTRAL:         40-59 — wait for direction
- BEARISH:         26-39 — lean short, watch resistance
- STRONG_BEARISH:  0-25  — gap & go short or fade the rally

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class GapType(Enum):
    GAP_UP_LARGE = "GAP_UP_LARGE"
    GAP_UP_SMALL = "GAP_UP_SMALL"
    GAP_DOWN_LARGE = "GAP_DOWN_LARGE"
    GAP_DOWN_SMALL = "GAP_DOWN_SMALL"
    NO_GAP = "NO_GAP"

    @property
    def emoji(self):
        return {"GAP_UP_LARGE": "⬆️⬆️", "GAP_UP_SMALL": "⬆️",
                "GAP_DOWN_LARGE": "⬇️⬇️", "GAP_DOWN_SMALL": "⬇️",
                "NO_GAP": "➡️"}[self.value]


class GapFillProbability(Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class OvernightBias(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

    @property
    def emoji(self):
        return {"STRONG_BULLISH": "🟢🟢", "BULLISH": "🟢", "NEUTRAL": "🟡",
                "BEARISH": "🔴", "STRONG_BEARISH": "🔴🔴"}[self.value]


class DayType(Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE_DAY = "RANGE_DAY"
    REVERSAL_UP = "REVERSAL_UP"
    REVERSAL_DOWN = "REVERSAL_DOWN"
    INSIDE_DAY = "INSIDE_DAY"
    OUTSIDE_DAY = "OUTSIDE_DAY"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VolumeProfileLevels:
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    vwap: float = 0.0


@dataclass
class WeeklyContext:
    trend: str = "NEUTRAL"
    last_week_structure: str = ""
    weekly_close_position: float = 0.5
    weekly_close_signal: str = ""
    supports_long: bool = False
    supports_short: bool = False


@dataclass
class SqueezeContext:
    is_squeezed: bool = False
    squeeze_days: int = 0
    bb_width_percentile: float = 50.0
    gap_breaks_squeeze: bool = False  # Gap opened outside squeeze range


@dataclass
class OptionsContext:
    iv_percentile: float = 50.0
    iv_regime: str = "normal"
    suggested_delta: float = 0.65
    min_dte: int = 21
    entry_size: str = "50%"
    scale_plan: str = "Enter 50%, add at +15% and +25%"


@dataclass
class OvernightSession:
    session_date: datetime = None
    overnight_high: float = 0.0
    overnight_low: float = 0.0
    overnight_open: float = 0.0
    overnight_close: float = 0.0
    overnight_range: float = 0.0
    overnight_midpoint: float = 0.0
    overnight_volume: float = 0.0
    overnight_delta: float = 0.0
    overnight_poc: float = 0.0
    overnight_vah: float = 0.0
    overnight_val: float = 0.0

    @property
    def overnight_direction(self) -> str:
        change = self.overnight_close - self.overnight_open
        pct = (change / self.overnight_open * 100) if self.overnight_open else 0
        if pct > 0.3: return "UP"
        elif pct < -0.3: return "DOWN"
        return "FLAT"


@dataclass
class GapAnalysis:
    gap_size: float = 0.0
    gap_pct: float = 0.0
    gap_type: GapType = GapType.NO_GAP
    gap_atr_ratio: float = 0.0
    prior_close: float = 0.0
    current_open: float = 0.0
    gap_fill_level: float = 0.0
    gap_fill_probability: GapFillProbability = GapFillProbability.MODERATE
    partial_fill_level: float = 0.0
    similar_gaps_filled_pct: float = 0.0
    historical_fill_count: int = 0     # NEW: actual similar gaps found
    avg_fill_time_bars: int = 0        # NEW: how many bars to fill historically


@dataclass
class PriorDayContext:
    date: datetime = None
    prior_open: float = 0.0
    prior_high: float = 0.0
    prior_low: float = 0.0
    prior_close: float = 0.0
    prior_range: float = 0.0
    prior_poc: float = 0.0
    prior_vah: float = 0.0
    prior_val: float = 0.0
    day_type: DayType = DayType.RANGE_DAY
    atr: float = 0.0
    close_vs_range: float = 0.5    # NEW: 0=closed at low, 1=closed at high

    @property
    def prior_midpoint(self) -> float:
        return (self.prior_high + self.prior_low) / 2

    @property
    def close_in_value(self) -> bool:
        return self.prior_val <= self.prior_close <= self.prior_vah


@dataclass
class OpeningContext:
    open_price: float = 0.0
    vs_prior_close: str = "AT"
    vs_prior_high: str = "BELOW"
    vs_prior_low: str = "ABOVE"
    vs_prior_poc: str = "AT"
    vs_prior_vah: str = "BELOW"
    vs_prior_val: str = "ABOVE"
    vs_overnight_high: str = "BELOW"
    vs_overnight_low: str = "ABOVE"
    vs_overnight_midpoint: str = "AT"
    scenario: str = "IN_RANGE_NEUTRAL"


@dataclass
class OvernightPrediction:
    """Complete overnight/gap prediction — V2"""
    symbol: str = ""
    analysis_time: str = ""

    # Components
    overnight: OvernightSession = None
    gap: GapAnalysis = None
    prior_day: PriorDayContext = None
    opening: OpeningContext = None

    # V2 Context
    weekly: WeeklyContext = None
    squeeze: SqueezeContext = None
    options: OptionsContext = None
    rsi: float = 50.0
    rsi_zone: str = "neutral"  # oversold, neutral, overbought

    # Prediction
    bias: str = "NEUTRAL"
    bias_emoji: str = "🟡"
    confidence: float = 50.0
    prediction_score: int = 50       # 0-100 (50=neutral, >50=bullish, <50=bearish)
    quality_grade: str = "C"

    # Key levels
    key_levels: Dict[str, float] = field(default_factory=dict)

    # Setup
    setup_type: str = ""
    entry_trigger: str = ""
    trade_direction: str = ""        # LONG, SHORT, or WAIT

    # Scenarios
    bull_scenario: str = ""
    bear_scenario: str = ""

    # Scoring components
    factors: List[str] = field(default_factory=list)


# =============================================================================
# OVERNIGHT MODEL V2
# =============================================================================

class OvernightModelV2:
    """
    Enhanced overnight/gap prediction with C.O.R.E. methodology.

    Self-contained: just pass a DataFrame and symbol, V2 calculates
    VP levels, weekly context, RSI, squeeze state, and everything else.

    Scoring (directional, centered at 50):
    - Gap Analysis:         ±20 pts
    - Overnight Direction:  ±15 pts
    - Overnight Delta:      ±10 pts
    - Opening Scenario:     ±15 pts
    - Prior Day Context:    ±10 pts
    - Weekly Alignment:     ±8 pts (NEW)
    - RSI Momentum:         ±7 pts (NEW)
    - Squeeze Context:      ±5 pts (NEW)
    - VP Open Location:     ±5 pts (NEW)
    - IV Context:           ±5 pts (NEW, adjusts sizing not direction)
    """

    # Market hours (ET)
    RTH_START = time(9, 30)
    RTH_END = time(16, 0)
    ETH_START = time(18, 0)

    # Gap thresholds
    LARGE_GAP_PCT = 1.0
    SMALL_GAP_PCT = 0.3

    # Default gap fill rates (used when insufficient historical data)
    DEFAULT_FILL_RATES = {
        "GAP_UP_LARGE": 0.45, "GAP_UP_SMALL": 0.72,
        "GAP_DOWN_LARGE": 0.48, "GAP_DOWN_SMALL": 0.70,
        "NO_GAP": 1.0
    }

    # VP parameters
    VP_NUM_BINS = 50
    VP_VALUE_AREA_PCT = 0.70

    def __init__(self):
        # Squeeze parameters
        self.bb_period = 20
        self.bb_std = 2.0
        self.kc_period = 20
        self.kc_mult = 1.5

    # =========================================================================
    # TECHNICAL CALCULATIONS
    # =========================================================================

    def _calculate_rsi_wilder(self, closes: pd.Series, period: int = 14) -> pd.Series:
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
        atr = tr.rolling(min(period, len(tr))).mean()
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

    # =========================================================================
    # VOLUME PROFILE
    # =========================================================================

    def _calculate_volume_profile(self, df: pd.DataFrame) -> VolumeProfileLevels:
        """Calculate POC/VAH/VAL from price data"""
        if len(df) < 5:
            mid = float(df['close'].mean())
            return VolumeProfileLevels(poc=mid, vah=mid, val=mid, vwap=mid)

        price_min = df['low'].min()
        price_max = df['high'].max()
        if price_max == price_min:
            return VolumeProfileLevels(poc=price_max, vah=price_max, val=price_min,
                                       vwap=float(df['close'].mean()))

        bin_size = (price_max - price_min) / self.VP_NUM_BINS
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
        target_vol = total_vol * self.VP_VALUE_AREA_PCT
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
        vwap = self._calculate_vwap(df)

        return VolumeProfileLevels(
            poc=round(poc, 2), vah=round(vah, 2),
            val=round(val, 2), vwap=round(vwap, 2)
        )

    # =========================================================================
    # WEEKLY STRUCTURE
    # =========================================================================

    def _calculate_weekly_context(self, symbol: str) -> WeeklyContext:
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

            if bearish >= 8 and bullish <= 2: trend = "STRONG_DOWNTREND"
            elif bearish >= 5 and bearish > bullish * 2: trend = "DOWNTREND"
            elif bullish >= 8 and bearish <= 2: trend = "STRONG_UPTREND"
            elif bullish >= 5 and bullish > bearish * 2: trend = "UPTREND"
            else: trend = "NEUTRAL"

            lw = weeks.iloc[-2]
            lw_range = lw['high'] - lw['low']
            wcp = (lw['close'] - lw['low']) / lw_range if lw_range > 0 else 0.5

            signal = ""
            if "LL" in last_structure and wcp > 0.70: signal = "BULLISH_REVERSAL"
            elif "HH" in last_structure and wcp < 0.30: signal = "BEARISH_REVERSAL"
            elif wcp > 0.75: signal = "STRONG_BULL_CLOSE"
            elif wcp < 0.25: signal = "STRONG_BEAR_CLOSE"

            supports_long = (trend in ("UPTREND", "STRONG_UPTREND") or
                           signal in ("BULLISH_REVERSAL", "STRONG_BULL_CLOSE") or
                           (trend == "NEUTRAL" and wcp > 0.6))
            supports_short = (trend in ("DOWNTREND", "STRONG_DOWNTREND") or
                            signal in ("BEARISH_REVERSAL", "STRONG_BEAR_CLOSE") or
                            (trend == "NEUTRAL" and wcp < 0.4))

            return WeeklyContext(trend=trend, last_week_structure=last_structure,
                               weekly_close_position=round(wcp, 2),
                               weekly_close_signal=signal,
                               supports_long=supports_long, supports_short=supports_short)
        except Exception as e:
            print(f"Weekly context error for {symbol}: {e}")
            return WeeklyContext()

    # =========================================================================
    # SQUEEZE CO-DETECTION
    # =========================================================================

    def _detect_squeeze(self, df: pd.DataFrame, gap_pct: float = 0.0) -> SqueezeContext:
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

        # Does the gap break outside the squeeze?
        gap_breaks = False
        if is_squeezed and abs(gap_pct) > 0.5:
            last_bb_upper = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else 0
            last_bb_lower = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else 0
            current_price = float(df['close'].iloc[-1])
            if current_price > last_bb_upper or current_price < last_bb_lower:
                gap_breaks = True

        return SqueezeContext(
            is_squeezed=is_squeezed, squeeze_days=squeeze_days,
            bb_width_percentile=round(width_pct, 1),
            gap_breaks_squeeze=gap_breaks
        )

    # =========================================================================
    # IV PERCENTILE
    # =========================================================================

    def _estimate_iv_percentile(self, df: pd.DataFrame) -> OptionsContext:
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

        if percentile < 20: regime = "low"
        elif percentile < 50: regime = "normal"
        elif percentile < 80: regime = "elevated"
        else: regime = "extreme"

        options.iv_percentile = round(percentile, 1)
        options.iv_regime = regime

        if regime == "extreme":
            options.suggested_delta = 0.60; options.entry_size = "25%"
            options.scale_plan = "Scale in as IV normalizes"
        elif regime == "elevated":
            options.suggested_delta = 0.65; options.entry_size = "40%"
            options.scale_plan = "Enter 40%, add at +15% and +25%"
        elif regime == "low":
            options.suggested_delta = 0.70; options.entry_size = "50%"
            options.scale_plan = "Enter 50%, add at +15% and +25% — cheap options"
        else:
            options.suggested_delta = 0.65; options.entry_size = "50%"
            options.scale_plan = "Enter 50%, add at +15% and +25%"

        return options

    # =========================================================================
    # OVERNIGHT SESSION EXTRACTION
    # =========================================================================

    def _extract_overnight(self, df: pd.DataFrame,
                            target_date=None) -> OvernightSession:
        """Extract overnight session, or estimate from daily data"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return self._estimate_overnight_from_daily(df)

        # Detect daily data — no intraday bars to extract overnight from
        if len(df) > 2:
            median_gap = (df.index[1:] - df.index[:-1]).median()
            if median_gap >= pd.Timedelta(hours=12):
                return self._estimate_overnight_from_daily(df)

        if target_date is None:
            target_date = df.index[-1].date()

        prev_evening_start = datetime.combine(
            target_date - timedelta(days=1), self.ETH_START)
        morning_end = datetime.combine(target_date, self.RTH_START)

        # Handle timezone-aware indexes (yfinance returns tz-aware)
        if df.index.tz is not None:
            import pytz
            try:
                tz = df.index.tz
                prev_evening_start = tz.localize(prev_evening_start) if hasattr(tz, 'localize') else prev_evening_start.replace(tzinfo=tz)
                morning_end = tz.localize(morning_end) if hasattr(tz, 'localize') else morning_end.replace(tzinfo=tz)
            except Exception:
                # If timezone handling fails, fall back to daily estimate
                return self._estimate_overnight_from_daily(df)

        overnight_mask = (df.index >= prev_evening_start) & (df.index < morning_end)
        overnight_df = df[overnight_mask]

        if len(overnight_df) < 3:
            return self._estimate_overnight_from_daily(df)

        o_high = float(overnight_df['high'].max())
        o_low = float(overnight_df['low'].min())
        o_open = float(overnight_df['open'].iloc[0])
        o_close = float(overnight_df['close'].iloc[-1])
        o_range = o_high - o_low
        o_mid = (o_high + o_low) / 2
        o_vol = float(overnight_df['volume'].sum())
        o_delta = self._estimate_delta(overnight_df)

        vp = self._calculate_volume_profile(overnight_df)

        return OvernightSession(
            session_date=target_date,
            overnight_high=o_high, overnight_low=o_low,
            overnight_open=o_open, overnight_close=o_close,
            overnight_range=o_range, overnight_midpoint=o_mid,
            overnight_volume=o_vol, overnight_delta=o_delta,
            overnight_poc=vp.poc, overnight_vah=vp.vah, overnight_val=vp.val
        )

    def _estimate_overnight_from_daily(self, df: pd.DataFrame) -> OvernightSession:
        """Estimate overnight from daily OHLCV"""
        daily = df.resample('D').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        if len(daily) < 2:
            p = float(df['close'].iloc[-1])
            return OvernightSession(
                overnight_high=p, overnight_low=p, overnight_open=p,
                overnight_close=p, overnight_range=0, overnight_midpoint=p,
                overnight_volume=0, overnight_delta=0)

        prior_close = float(daily['close'].iloc[-2])
        current_open = float(daily['open'].iloc[-1])
        gap = abs(current_open - prior_close)

        if current_open > prior_close:
            o_high, o_low = current_open, prior_close
        else:
            o_high, o_low = prior_close, current_open

        return OvernightSession(
            overnight_high=o_high, overnight_low=o_low,
            overnight_open=prior_close, overnight_close=current_open,
            overnight_range=o_high - o_low,
            overnight_midpoint=(o_high + o_low) / 2,
            overnight_volume=0,
            overnight_delta=current_open - prior_close,
            overnight_poc=(o_high + o_low) / 2,
            overnight_vah=o_high, overnight_val=o_low
        )

    def _estimate_delta(self, df: pd.DataFrame) -> float:
        delta = 0.0
        for _, row in df.iterrows():
            bar_range = row['high'] - row['low']
            if bar_range == 0: continue
            clv = (row['close'] - row['low']) / bar_range
            delta += row['volume'] * (2 * clv - 1)
        return delta

    # =========================================================================
    # GAP ANALYSIS (Enhanced)
    # =========================================================================

    def _analyze_gap(self, prior_close: float, current_open: float,
                      atr: float, df_daily: pd.DataFrame = None) -> GapAnalysis:
        """Enhanced gap analysis with historical fill rate calculation"""
        gap_size = current_open - prior_close
        gap_pct = (gap_size / prior_close * 100) if prior_close else 0
        gap_atr_ratio = abs(gap_size) / atr if atr else 0

        # Classify
        if gap_pct >= self.LARGE_GAP_PCT: gap_type = GapType.GAP_UP_LARGE
        elif gap_pct >= self.SMALL_GAP_PCT: gap_type = GapType.GAP_UP_SMALL
        elif gap_pct <= -self.LARGE_GAP_PCT: gap_type = GapType.GAP_DOWN_LARGE
        elif gap_pct <= -self.SMALL_GAP_PCT: gap_type = GapType.GAP_DOWN_SMALL
        else: gap_type = GapType.NO_GAP

        # Fill levels
        gap_fill_level = prior_close
        partial_fill_level = prior_close + (gap_size * 0.5)

        # Historical fill rate from actual data
        fill_rate = self.DEFAULT_FILL_RATES.get(gap_type.value, 0.5)
        hist_count = 0
        avg_fill_bars = 0

        if df_daily is not None and len(df_daily) >= 30:
            fill_rate, hist_count, avg_fill_bars = self._calculate_historical_fill_rate(
                df_daily, gap_type, gap_atr_ratio)

        # Adjust for extreme gap sizes
        if gap_atr_ratio > 2.0:
            fill_rate *= 0.7
        elif gap_atr_ratio < 0.5:
            fill_rate *= 1.2
        fill_rate = min(1.0, max(0.0, fill_rate))

        if fill_rate >= 0.7: fill_prob = GapFillProbability.HIGH
        elif fill_rate >= 0.5: fill_prob = GapFillProbability.MODERATE
        else: fill_prob = GapFillProbability.LOW

        return GapAnalysis(
            gap_size=round(gap_size, 2), gap_pct=round(gap_pct, 2),
            gap_type=gap_type, gap_atr_ratio=round(gap_atr_ratio, 2),
            prior_close=round(prior_close, 2), current_open=round(current_open, 2),
            gap_fill_level=round(gap_fill_level, 2),
            gap_fill_probability=fill_prob,
            partial_fill_level=round(partial_fill_level, 2),
            similar_gaps_filled_pct=round(fill_rate * 100, 1),
            historical_fill_count=hist_count,
            avg_fill_time_bars=avg_fill_bars
        )

    def _calculate_historical_fill_rate(self, df_daily: pd.DataFrame,
                                          gap_type: GapType,
                                          gap_atr_ratio: float) -> Tuple[float, int, int]:
        """
        Calculate actual gap fill rate from historical data.
        Finds similar gaps and checks if they filled same day.
        """
        try:
            similar_count = 0
            filled_count = 0
            fill_bars_total = 0

            atr_series = self._calculate_atr_series(df_daily)

            for i in range(1, min(len(df_daily) - 1, 60)):
                prior_c = float(df_daily['close'].iloc[i - 1])
                curr_o = float(df_daily['open'].iloc[i])
                curr_atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 1

                hist_gap_pct = ((curr_o - prior_c) / prior_c * 100) if prior_c else 0
                hist_atr_ratio = abs(curr_o - prior_c) / curr_atr if curr_atr else 0

                # Match gap type
                is_same_direction = (
                    (gap_type.value.startswith("GAP_UP") and hist_gap_pct > self.SMALL_GAP_PCT) or
                    (gap_type.value.startswith("GAP_DOWN") and hist_gap_pct < -self.SMALL_GAP_PCT)
                )

                # Match approximate size (within 1.5x ATR ratio)
                if is_same_direction and abs(hist_atr_ratio - gap_atr_ratio) < 1.5:
                    similar_count += 1

                    # Did it fill same day?
                    curr_low = float(df_daily['low'].iloc[i])
                    curr_high = float(df_daily['high'].iloc[i])

                    if hist_gap_pct > 0:  # Gap up → filled if low touches prior close
                        if curr_low <= prior_c:
                            filled_count += 1
                            fill_bars_total += 1  # Same day = 1 bar on daily
                    else:  # Gap down → filled if high touches prior close
                        if curr_high >= prior_c:
                            filled_count += 1
                            fill_bars_total += 1

            if similar_count >= 3:
                rate = filled_count / similar_count
                avg_bars = fill_bars_total // max(filled_count, 1)
                return rate, similar_count, avg_bars
            else:
                return self.DEFAULT_FILL_RATES.get(gap_type.value, 0.5), similar_count, 0

        except Exception:
            return self.DEFAULT_FILL_RATES.get(gap_type.value, 0.5), 0, 0

    # =========================================================================
    # PRIOR DAY ANALYSIS (Self-Contained)
    # =========================================================================

    def _analyze_prior_day(self, df: pd.DataFrame) -> PriorDayContext:
        """Analyze prior day with self-contained VP calculation"""
        daily = df.resample('D').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        if len(daily) < 2:
            return PriorDayContext()

        prior = daily.iloc[-2]
        two_ago = daily.iloc[-3] if len(daily) >= 3 else prior

        prior_range = prior['high'] - prior['low']
        atr = self._calculate_atr(daily)

        # Self-contained VP for prior day
        prior_date = daily.index[-2]
        try:
            prior_mask = df.index.date == prior_date.date()
            prior_bars = df[prior_mask]
            if len(prior_bars) >= 5:
                vp = self._calculate_volume_profile(prior_bars)
            else:
                mid = (prior['high'] + prior['low']) / 2
                vp = VolumeProfileLevels(
                    poc=mid, vah=prior['high'] * 0.997,
                    val=prior['low'] * 1.003, vwap=mid)
        except:
            mid = (prior['high'] + prior['low']) / 2
            vp = VolumeProfileLevels(poc=mid, vah=prior['high'] * 0.997,
                                      val=prior['low'] * 1.003, vwap=mid)

        # Classify day
        day_type = self._classify_day(prior, two_ago, atr)

        # Close position in range
        close_vs_range = ((prior['close'] - prior['low']) / prior_range
                         if prior_range > 0 else 0.5)

        return PriorDayContext(
            date=prior_date,
            prior_open=round(prior['open'], 2), prior_high=round(prior['high'], 2),
            prior_low=round(prior['low'], 2), prior_close=round(prior['close'], 2),
            prior_range=round(prior_range, 2),
            prior_poc=vp.poc, prior_vah=vp.vah, prior_val=vp.val,
            day_type=day_type, atr=round(atr, 2),
            close_vs_range=round(close_vs_range, 2)
        )

    def _classify_day(self, current: pd.Series, prior: pd.Series,
                       atr: float) -> DayType:
        current_range = current['high'] - current['low']
        prior_range = prior['high'] - prior['low']

        if current['high'] <= prior['high'] and current['low'] >= prior['low']:
            return DayType.INSIDE_DAY
        if current['high'] > prior['high'] and current['low'] < prior['low']:
            return DayType.OUTSIDE_DAY

        oc_move = current['close'] - current['open']
        range_pct = abs(oc_move) / current_range if current_range else 0

        if range_pct > 0.6:
            if oc_move > 0:
                if current['open'] < prior['close'] and current['close'] > prior['close']:
                    return DayType.REVERSAL_UP
                return DayType.TREND_UP
            else:
                if current['open'] > prior['close'] and current['close'] < prior['close']:
                    return DayType.REVERSAL_DOWN
                return DayType.TREND_DOWN

        return DayType.RANGE_DAY

    # =========================================================================
    # OPENING CONTEXT
    # =========================================================================

    def _analyze_opening(self, open_price: float,
                          prior_day: PriorDayContext,
                          overnight: OvernightSession) -> OpeningContext:
        def cmp(price, level, tol=0.001):
            d = (price - level) / level if level else 0
            if d > tol: return "ABOVE"
            elif d < -tol: return "BELOW"
            return "AT"

        vs_pc = cmp(open_price, prior_day.prior_close)
        vs_ph = cmp(open_price, prior_day.prior_high)
        vs_pl = cmp(open_price, prior_day.prior_low)
        vs_poc = cmp(open_price, prior_day.prior_poc)
        vs_vah = cmp(open_price, prior_day.prior_vah)
        vs_val = cmp(open_price, prior_day.prior_val)
        vs_oh = cmp(open_price, overnight.overnight_high)
        vs_ol = cmp(open_price, overnight.overnight_low)
        vs_om = cmp(open_price, overnight.overnight_midpoint)

        if vs_ph == "ABOVE" and vs_oh in ("ABOVE", "AT"):
            scenario = "ABOVE_ALL_BULLISH"
        elif vs_pl == "BELOW" and vs_ol in ("BELOW", "AT"):
            scenario = "BELOW_ALL_BEARISH"
        elif vs_vah == "ABOVE":
            scenario = "ABOVE_VALUE_BULLISH"
        elif vs_val == "BELOW":
            scenario = "BELOW_VALUE_BEARISH"
        elif vs_poc == "AT":
            scenario = "AT_POC_NEUTRAL"
        else:
            scenario = "IN_RANGE_NEUTRAL"

        return OpeningContext(
            open_price=round(open_price, 2),
            vs_prior_close=vs_pc, vs_prior_high=vs_ph, vs_prior_low=vs_pl,
            vs_prior_poc=vs_poc, vs_prior_vah=vs_vah, vs_prior_val=vs_val,
            vs_overnight_high=vs_oh, vs_overnight_low=vs_ol,
            vs_overnight_midpoint=vs_om, scenario=scenario
        )

    # =========================================================================
    # QUALITY GRADE
    # =========================================================================

    def _quality_grade(self, confidence: float) -> str:
        if confidence >= 85: return "A+"
        elif confidence >= 75: return "A"
        elif confidence >= 65: return "B"
        elif confidence >= 55: return "C"
        elif confidence >= 45: return "D"
        else: return "F"

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> OvernightPrediction:
        """
        Full overnight/gap prediction with C.O.R.E. methodology.

        Args:
            df: OHLCV DataFrame with datetime index (intraday or daily)
            symbol: Stock symbol for weekly context

        Returns:
            OvernightPrediction with scored directional bias
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if len(df) < 20:
            return OvernightPrediction(symbol=symbol)

        current_price = float(df['close'].iloc[-1])
        factors = []

        # =====================================================================
        # CORE COMPONENTS
        # =====================================================================

        # 1. Overnight session
        overnight = self._extract_overnight(df)

        # 2. Prior day (self-contained VP)
        prior_day = self._analyze_prior_day(df)

        # 3. Gap analysis (with historical fill rate)
        daily = df.resample('D').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        current_open = float(daily['open'].iloc[-1]) if len(daily) > 0 else current_price
        gap = self._analyze_gap(prior_day.prior_close, current_open,
                                 prior_day.atr, daily)

        # 4. Opening context
        opening = self._analyze_opening(current_open, prior_day, overnight)

        # =====================================================================
        # V2 CONTEXT
        # =====================================================================

        # 5. Weekly structure
        weekly = self._calculate_weekly_context(symbol)

        # 6. RSI (Wilder's)
        rsi_series = self._calculate_rsi_wilder(df['close'])
        current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        rsi_zone = "neutral"
        if current_rsi < 30: rsi_zone = "oversold"
        elif current_rsi < 40: rsi_zone = "approaching_oversold"
        elif current_rsi > 70: rsi_zone = "overbought"
        elif current_rsi > 60: rsi_zone = "approaching_overbought"

        # 7. Squeeze co-detection
        squeeze = self._detect_squeeze(df, gap.gap_pct)

        # 8. IV/Options
        options = self._estimate_iv_percentile(df)

        # =====================================================================
        # DIRECTIONAL SCORING (centered at 50)
        # =====================================================================
        # Score > 50 = bullish, < 50 = bearish
        # Each factor pushes the score in one direction

        score = 50  # Start neutral

        # --- Gap Analysis (±20 pts) ---
        if gap.gap_type in (GapType.GAP_UP_LARGE, GapType.GAP_UP_SMALL):
            if gap.gap_fill_probability == GapFillProbability.LOW:
                score += 20
                factors.append(f"Gap up {gap.gap_pct:+.1f}% — low fill prob (Gap & Go)")
            elif gap.gap_fill_probability == GapFillProbability.HIGH:
                score += 5  # Gap up but likely to fade
                factors.append(f"Gap up {gap.gap_pct:+.1f}% — HIGH fill prob (fade risk)")
            else:
                score += 12
                factors.append(f"Gap up {gap.gap_pct:+.1f}%")
        elif gap.gap_type in (GapType.GAP_DOWN_LARGE, GapType.GAP_DOWN_SMALL):
            if gap.gap_fill_probability == GapFillProbability.LOW:
                score -= 20
                factors.append(f"Gap down {gap.gap_pct:+.1f}% — low fill prob (Gap & Go)")
            elif gap.gap_fill_probability == GapFillProbability.HIGH:
                score -= 5
                factors.append(f"Gap down {gap.gap_pct:+.1f}% — HIGH fill prob (bounce risk)")
            else:
                score -= 12
                factors.append(f"Gap down {gap.gap_pct:+.1f}%")
        else:
            factors.append("No significant gap — balanced open")

        # --- Overnight Direction (±15 pts) ---
        if overnight.overnight_direction == "UP":
            score += 15
            factors.append("Overnight session bullish")
        elif overnight.overnight_direction == "DOWN":
            score -= 15
            factors.append("Overnight session bearish")

        # --- Overnight Delta (±10 pts) ---
        if overnight.overnight_delta > 0:
            score += 10
            factors.append("Overnight buying pressure")
        elif overnight.overnight_delta < 0:
            score -= 10
            factors.append("Overnight selling pressure")

        # --- Opening Scenario (±15 pts) ---
        if "BULLISH" in opening.scenario:
            score += 15
            factors.append(f"Opening: {opening.scenario}")
        elif "BEARISH" in opening.scenario:
            score -= 15
            factors.append(f"Opening: {opening.scenario}")
        else:
            factors.append(f"Opening: {opening.scenario}")

        # --- Prior Day Context (±10 pts) ---
        if prior_day.day_type in (DayType.TREND_UP, DayType.REVERSAL_UP):
            score += 10
            factors.append(f"Prior day: {prior_day.day_type.value}")
        elif prior_day.day_type in (DayType.TREND_DOWN, DayType.REVERSAL_DOWN):
            score -= 10
            factors.append(f"Prior day: {prior_day.day_type.value}")
        elif prior_day.day_type == DayType.INSIDE_DAY:
            factors.append("Prior inside day — breakout potential")

        # --- NEW: Weekly Alignment (±8 pts) ---
        if weekly.supports_long:
            score += 8
            factors.append(f"Weekly supports long ({weekly.trend})")
        elif weekly.supports_short:
            score -= 8
            factors.append(f"Weekly supports short ({weekly.trend})")

        # --- NEW: RSI Momentum (±7 pts) ---
        if rsi_zone == "oversold":
            score += 7  # Oversold = bounce potential
            factors.append(f"RSI oversold ({current_rsi:.0f}) — bounce potential")
        elif rsi_zone == "overbought":
            score -= 7  # Overbought = fade potential
            factors.append(f"RSI overbought ({current_rsi:.0f}) — fade potential")

        # --- NEW: Squeeze Context (±5 pts) ---
        if squeeze.gap_breaks_squeeze:
            if gap.gap_pct > 0:
                score += 5
                factors.append("Gap breaks squeeze upward — momentum")
            else:
                score -= 5
                factors.append("Gap breaks squeeze downward — momentum")
        elif squeeze.is_squeezed:
            factors.append(f"Squeeze active ({squeeze.squeeze_days}d) — directional catalyst pending")

        # --- NEW: VP Open Location (±5 pts) ---
        if opening.vs_prior_vah == "ABOVE":
            score += 5
        elif opening.vs_prior_val == "BELOW":
            score -= 5

        # Clamp score to 0-100
        score = max(0, min(100, score))

        # =====================================================================
        # BIAS & CONFIDENCE
        # =====================================================================

        if score >= 75:
            bias = OvernightBias.STRONG_BULLISH
        elif score >= 60:
            bias = OvernightBias.BULLISH
        elif score <= 25:
            bias = OvernightBias.STRONG_BEARISH
        elif score <= 40:
            bias = OvernightBias.BEARISH
        else:
            bias = OvernightBias.NEUTRAL

        confidence = abs(score - 50) * 2  # 0-100 scale

        # =====================================================================
        # SETUP CLASSIFICATION
        # =====================================================================

        setup_type = ""
        entry_trigger = ""
        trade_direction = "WAIT"

        if bias in (OvernightBias.STRONG_BULLISH, OvernightBias.BULLISH):
            trade_direction = "LONG"
            if gap.gap_type in (GapType.GAP_UP_LARGE,) and gap.gap_fill_probability == GapFillProbability.LOW:
                setup_type = "gap_and_go_long"
                entry_trigger = f"Enter {options.entry_size} on first pullback to VWAP, stop below overnight low ${overnight.overnight_low:.2f}"
            elif gap.gap_type in (GapType.GAP_DOWN_SMALL, GapType.GAP_DOWN_LARGE) and gap.gap_fill_probability == GapFillProbability.HIGH:
                setup_type = "gap_fill_long"
                entry_trigger = f"Enter {options.entry_size} at gap fill test ${gap.gap_fill_level:.2f}, target prior VAH ${prior_day.prior_vah:.2f}"
            elif squeeze.gap_breaks_squeeze and gap.gap_pct > 0:
                setup_type = "squeeze_break_long"
                entry_trigger = f"Gap broke squeeze — enter {options.entry_size} on retest of breakout, target 1.5 ATR"
            elif weekly.supports_long and rsi_zone in ("oversold", "approaching_oversold"):
                setup_type = "weekly_bounce_long"
                entry_trigger = f"Weekly uptrend + oversold RSI — enter {options.entry_size} at prior VAL ${prior_day.prior_val:.2f}"
            else:
                setup_type = "lean_long"
                entry_trigger = f"Bias long — wait for pullback to overnight mid ${overnight.overnight_midpoint:.2f}"

        elif bias in (OvernightBias.STRONG_BEARISH, OvernightBias.BEARISH):
            trade_direction = "SHORT"
            if gap.gap_type in (GapType.GAP_DOWN_LARGE,) and gap.gap_fill_probability == GapFillProbability.LOW:
                setup_type = "gap_and_go_short"
                entry_trigger = f"Enter {options.entry_size} on first bounce to VWAP, stop above overnight high ${overnight.overnight_high:.2f}"
            elif gap.gap_type in (GapType.GAP_UP_SMALL, GapType.GAP_UP_LARGE) and gap.gap_fill_probability == GapFillProbability.HIGH:
                setup_type = "gap_fill_short"
                entry_trigger = f"Enter {options.entry_size} on failed gap hold, target gap fill ${gap.gap_fill_level:.2f}"
            elif squeeze.gap_breaks_squeeze and gap.gap_pct < 0:
                setup_type = "squeeze_break_short"
                entry_trigger = f"Gap broke squeeze down — enter {options.entry_size} on retest, target 1.5 ATR"
            elif weekly.supports_short and rsi_zone in ("overbought", "approaching_overbought"):
                setup_type = "weekly_fade_short"
                entry_trigger = f"Weekly downtrend + overbought RSI — enter {options.entry_size} at prior VAH ${prior_day.prior_vah:.2f}"
            else:
                setup_type = "lean_short"
                entry_trigger = f"Bias short — wait for bounce to overnight mid ${overnight.overnight_midpoint:.2f}"

        # =====================================================================
        # KEY LEVELS
        # =====================================================================

        key_levels = {
            "overnight_high": overnight.overnight_high,
            "overnight_low": overnight.overnight_low,
            "overnight_mid": overnight.overnight_midpoint,
            "prior_high": prior_day.prior_high,
            "prior_low": prior_day.prior_low,
            "prior_close": prior_day.prior_close,
            "prior_poc": prior_day.prior_poc,
            "prior_vah": prior_day.prior_vah,
            "prior_val": prior_day.prior_val,
            "gap_fill": gap.gap_fill_level,
            "partial_fill": gap.partial_fill_level
        }
        if overnight.overnight_poc:
            key_levels["overnight_poc"] = overnight.overnight_poc

        # =====================================================================
        # SCENARIOS
        # =====================================================================

        bull_scenario = (
            f"BULL: Hold above ${overnight.overnight_low:.2f} (ON low), "
            f"reclaim ${prior_day.prior_vah:.2f} (VAH), "
            f"target ${prior_day.prior_high:.2f} (prior high), "
            f"extended ${prior_day.prior_high + prior_day.atr:.2f}"
        )
        bear_scenario = (
            f"BEAR: Lose ${overnight.overnight_low:.2f} (ON low), "
            f"break ${prior_day.prior_val:.2f} (VAL), "
            f"target ${prior_day.prior_low:.2f} (prior low), "
            f"extended ${prior_day.prior_low - prior_day.atr:.2f}"
        )

        return OvernightPrediction(
            symbol=symbol,
            analysis_time=datetime.now().isoformat(),
            overnight=overnight, gap=gap, prior_day=prior_day, opening=opening,
            weekly=weekly, squeeze=squeeze, options=options,
            rsi=round(current_rsi, 2), rsi_zone=rsi_zone,
            bias=bias.value, bias_emoji=bias.emoji,
            confidence=round(confidence, 1),
            prediction_score=score,
            quality_grade=self._quality_grade(confidence),
            key_levels=key_levels,
            setup_type=setup_type, entry_trigger=entry_trigger,
            trade_direction=trade_direction,
            bull_scenario=bull_scenario, bear_scenario=bear_scenario,
            factors=factors[:10]
        )

    def to_dict(self, pred: OvernightPrediction) -> Dict:
        return asdict(pred)


# =============================================================================
# ALERT FORMATTER
# =============================================================================

def format_overnight_alert(pred: OvernightPrediction) -> str:
    """Format overnight prediction as alert message"""
    lines = [
        f"{pred.bias_emoji} {pred.symbol} OVERNIGHT/GAP PREDICTION (V2)",
        f"Bias: {pred.bias} | Score: {pred.prediction_score}/100 | "
        f"Confidence: {pred.confidence:.0f}% | Grade: {pred.quality_grade}",
        ""
    ]

    if pred.setup_type:
        lines.append(f"🎯 SETUP: {pred.setup_type}")
        lines.append(f"   {pred.entry_trigger}")
        lines.append(f"   Direction: {pred.trade_direction}")
        lines.append("")

    if pred.gap:
        g = pred.gap
        lines.append(f"📊 GAP: {g.gap_type.emoji} {g.gap_type.value}")
        lines.append(f"   Size: {g.gap_pct:+.2f}% ({g.gap_atr_ratio:.2f}x ATR)")
        lines.append(f"   Fill prob: {g.gap_fill_probability.value} ({g.similar_gaps_filled_pct:.0f}%)")
        if g.historical_fill_count > 0:
            lines.append(f"   Based on {g.historical_fill_count} similar historical gaps")
        lines.append(f"   Fill level: ${g.gap_fill_level:.2f}")

    if pred.overnight:
        o = pred.overnight
        lines.append(f"\n🌙 OVERNIGHT: {o.overnight_direction}")
        lines.append(f"   Range: ${o.overnight_low:.2f} — ${o.overnight_high:.2f}")
        lines.append(f"   Mid: ${o.overnight_midpoint:.2f}")

    if pred.opening:
        lines.append(f"\n🔔 OPENING: {pred.opening.scenario}")
        lines.append(f"   vs Prior Close: {pred.opening.vs_prior_close} | "
                     f"vs POC: {pred.opening.vs_prior_poc} | "
                     f"vs ON Mid: {pred.opening.vs_overnight_midpoint}")

    if pred.prior_day:
        pd_ = pred.prior_day
        lines.append(f"\n📅 PRIOR DAY: {pd_.day_type.value}")
        lines.append(f"   VP: VAH ${pd_.prior_vah:.2f} | POC ${pd_.prior_poc:.2f} | VAL ${pd_.prior_val:.2f}")
        lines.append(f"   Close position: {pd_.close_vs_range:.0%} of range")

    if pred.weekly:
        lines.append(f"\n📊 WEEKLY: {pred.weekly.trend} | Signal: {pred.weekly.weekly_close_signal or 'none'}")

    lines.append(f"📈 RSI: {pred.rsi:.1f} ({pred.rsi_zone})")

    if pred.squeeze and (pred.squeeze.is_squeezed or pred.squeeze.gap_breaks_squeeze):
        if pred.squeeze.gap_breaks_squeeze:
            lines.append(f"🔥 GAP BROKE SQUEEZE — directional catalyst!")
        else:
            lines.append(f"🔲 Squeeze active: {pred.squeeze.squeeze_days}d")

    if pred.options:
        lines.append(f"📋 IV: {pred.options.iv_percentile:.0f}%ile ({pred.options.iv_regime}) | "
                     f"Size: {pred.options.entry_size}")

    # Key levels sorted by price
    if pred.key_levels:
        lines.append(f"\n📍 KEY LEVELS:")
        for name, level in sorted(pred.key_levels.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"   {name:<15}: ${level:.2f}")

    lines.append(f"\n{pred.bull_scenario}")
    lines.append(f"{pred.bear_scenario}")

    if pred.factors:
        lines.append(f"\n✅ Factors: {', '.join(pred.factors[:6])}")

    return "\n".join(lines)


# =============================================================================
# QUICK SCAN FUNCTIONS
# =============================================================================

def scan_overnight(symbol: str, period: str = "3mo",
                    interval: str = "1d") -> Optional[OvernightPrediction]:
    """Quick scan a single symbol"""
    from polygon_data import get_bars
    df = get_bars(symbol, period=period, interval=interval)
    if df.empty:
        return None
    model = OvernightModelV2()
    return model.analyze(df, symbol)


def scan_symbols(symbols: List[str],
                  period: str = "3mo",
                  interval: str = "1d",
                  min_confidence: float = 30.0) -> Dict[str, List[OvernightPrediction]]:
    """
    Batch scan symbols for overnight predictions.
    Returns dict with 'bullish' and 'bearish' lists, sorted by confidence.
    """
    from polygon_data import get_bars

    model = OvernightModelV2()
    results = {"bullish": [], "bearish": [], "neutral": []}

    for symbol in symbols:
        try:
            df = get_bars(symbol, period=period, interval=interval)
            if df.empty:
                continue

            pred = model.analyze(df, symbol)
            if pred.confidence < min_confidence:
                results["neutral"].append(pred)
                continue

            if pred.prediction_score > 55:
                results["bullish"].append(pred)
            elif pred.prediction_score < 45:
                results["bearish"].append(pred)
            else:
                results["neutral"].append(pred)

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    results["bullish"].sort(key=lambda x: x.prediction_score, reverse=True)
    results["bearish"].sort(key=lambda x: x.prediction_score)

    return results


# =============================================================================
# V1 COMPATIBILITY
# =============================================================================

class OvernightPredictionEngine(OvernightModelV2):
    """Backward-compatible wrapper. Use OvernightModelV2 for new code."""

    def predict(self, df: pd.DataFrame, symbol: str = "UNKNOWN",
                prior_poc=None, prior_vah=None, prior_val=None):
        """V1-compatible predict method"""
        return self.analyze(df, symbol)

    def print_prediction(self, pred: OvernightPrediction) -> str:
        return format_overnight_alert(pred)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  OVERNIGHT & GAP PREDICTION V2 — C.O.R.E. Methodology")
    print("=" * 65)

    test_symbols = ["NVDA", "AAPL", "META", "TSLA", "AMD", "MSFT", "AMZN"]

    model = OvernightModelV2()

    for symbol in test_symbols:
        print(f"\nScanning {symbol}...")
        try:
            from polygon_data import get_bars
            df = get_bars(symbol, period="3mo", interval="1d")

            if df.empty:
                print(f"  No data for {symbol}")
                continue

            pred = model.analyze(df, symbol)
            if pred and pred.confidence > 20:
                print(format_overnight_alert(pred))
            else:
                print(f"  {symbol}: Neutral (Score: {pred.prediction_score}, Conf: {pred.confidence:.0f}%)")

        except Exception as e:
            print(f"  Error: {e}")
        print("-" * 65)

    print(f"\n{'='*65}")
    print("  MORNING SCAN — Directional Bias")
    print(f"{'='*65}")

    results = scan_symbols(test_symbols, min_confidence=30)

    if results["bullish"]:
        print("\n  🟢 BULLISH BIAS:")
        for r in results["bullish"]:
            print(f"    {r.symbol}: Score {r.prediction_score} ({r.bias}) "
                  f"— {r.setup_type or 'lean long'} | Conf {r.confidence:.0f}%")

    if results["bearish"]:
        print("\n  🔴 BEARISH BIAS:")
        for r in results["bearish"]:
            print(f"    {r.symbol}: Score {r.prediction_score} ({r.bias}) "
                  f"— {r.setup_type or 'lean short'} | Conf {r.confidence:.0f}%")

    if not results["bullish"] and not results["bearish"]:
        print("  All neutral — no strong directional bias")
