"""
Dual Setup Generator V2 — C.O.R.E. Methodology
=================================================
Outputs LONG + SHORT setups with grades, probabilities, and entry plans.
Zero API cost — deterministic rules replace GPT.

WHAT IT DOES:
For any symbol, generates both a LONG and a SHORT setup simultaneously.
Each setup has entry zones, stops, targets, R:R, probability, grade,
trigger conditions, and invalidation rules. Then picks a preferred
direction and generates an options strategy.

RETAINED FROM V1:
- Dual setup output (LONG + SHORT for every symbol)
- Grade system (A+, A, B, C, F)
- Probability calculation from VP position
- Entry zones (conservative pullback + aggressive NOW)
- Stop loss at key VP levels
- Targets at VP levels with minimum distance enforcement
- Expected Value calculation
- Verdict logic (preferred direction)
- Key decision level identification
- Bookmap order flow checklist
- Earnings context integration
- format_as_ai_text() output compatible with parseAIResponse()

NEW IN V2:
- Self-Contained Analysis — pass symbol + DataFrame, calculates VP/RSI/ATR internally
- Volume Profile Calculation — POC/VAH/VAL computed from data
- Weekly Structure — MTF trend context boosts/penalizes setups
- Wilder's RSI — consistent EMA-based RSI, replaces raw passed value
- Squeeze Co-Detection — squeeze active boosts breakout setups
- IV Percentile — options sizing adjusts to IV regime
- Single-Winner Options — replaces dual-hedge with Rob's current methodology
  (0.65 delta, 3+ weeks DTE, -12.5% contract stop, scale at +15%/+25%)
- Enhanced Probability Model — weekly, squeeze, IV all adjust probabilities
- Setup Classification — labels consistent with other V2 scanners
- Batch Scanning — scan_symbols() for watchlist sweeps
- Quality Score 0-100 — normalized scoring alongside letter grades

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

# Earnings calendar for catalyst awareness
try:
    from earnings_calendar import get_earnings_calendar
    earnings_available = True
except ImportError:
    earnings_available = False


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


@dataclass
class OptionsContext:
    iv_percentile: float = 50.0
    iv_regime: str = "normal"
    suggested_delta: float = 0.65
    min_dte: int = 21
    contract_stop_pct: float = 12.5
    entry_size: str = "50%"
    scale_plan: str = "Enter 50%, add at +15% and +25%"


@dataclass
class Setup:
    """Single direction setup"""
    direction: str = ""          # LONG or SHORT
    grade: str = "F"
    conviction: int = 1          # 1-10
    probability: int = 50
    probability_low: int = 45
    probability_high: int = 55
    probability_label: str = "Medium"
    entry_low: float = 0.0
    entry_high: float = 0.0
    aggressive_entry: float = 0.0
    stop: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    risk_reward: float = 0.0
    ev: float = 0.0
    trigger: str = ""
    invalidation: str = ""
    why: str = ""
    # V2 additions
    quality_score: int = 0       # 0-100 normalized
    setup_type: str = ""         # vp_bounce_long, compression_break_short, etc.
    weekly_aligned: bool = False
    squeeze_active: bool = False


@dataclass
class DualSetupResult:
    """Complete dual setup output — V2"""
    symbol: str = ""
    current_price: float = 0.0
    timestamp: str = ""

    # Setups
    long_setup: Setup = None
    short_setup: Setup = None

    # Verdict
    preferred_direction: str = ""
    verdict_reason: str = ""
    key_level: float = 0.0
    key_level_desc: str = ""

    # Bookmap checklist
    bookmap_long: str = ""
    bookmap_short: str = ""

    # Options strategy (single-winner V2)
    options_strategy: Optional[Dict] = None

    # Earnings context
    earnings_context: Optional[Dict] = None

    # V2 Context
    volume_profile: VolumeProfileLevels = None
    weekly: WeeklyContext = None
    squeeze: SqueezeContext = None
    options_context: OptionsContext = None
    rsi: float = 50.0
    atr: float = 0.0


# =============================================================================
# DUAL SETUP GENERATOR V2
# =============================================================================

class DualSetupGeneratorV2:
    """
    Self-contained dual setup generator with C.O.R.E. methodology.

    Pass a symbol and DataFrame, V2 handles everything:
    VP calculation, RSI, ATR, weekly context, squeeze, IV.
    """

    # Probability base rates
    BASE_VAH_REJECTION = 65
    BASE_VAL_BOUNCE = 65
    BASE_POC_RETURN = 70
    RETEST_DECAY = -15

    # Adjustments
    MTF_ALIGNED_BONUS = 15
    VWAP_TREND_BONUS = 5
    EXTENDED_PENALTY = -10
    REJECTION_CANDLE_BONUS = 10
    WEEKLY_ALIGNED_BONUS = 8
    SQUEEZE_BONUS = 7
    IV_LOW_BONUS = 5

    # VP parameters
    VP_NUM_BINS = 50
    VP_VALUE_AREA_PCT = 0.70

    def __init__(self):
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

    def _calculate_rvol(self, df: pd.DataFrame) -> float:
        """Relative volume: current volume vs average"""
        if len(df) < 20:
            return 1.0
        avg_vol = df['volume'].tail(20).mean()
        if avg_vol == 0:
            return 1.0
        return float(df['volume'].iloc[-1] / avg_vol)

    # =========================================================================
    # VOLUME PROFILE
    # =========================================================================

    def _calculate_volume_profile(self, df: pd.DataFrame) -> VolumeProfileLevels:
        if len(df) < 10:
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
        except Exception:
            return WeeklyContext()

    # =========================================================================
    # SQUEEZE CO-DETECTION
    # =========================================================================

    def _detect_squeeze(self, df: pd.DataFrame) -> SqueezeContext:
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

        return SqueezeContext(is_squeezed=is_squeezed, squeeze_days=squeeze_days,
                              bb_width_percentile=round(width_pct, 1))

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
            options.contract_stop_pct = 15.0
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
    # PROBABILITY CALCULATIONS
    # =========================================================================

    def _calculate_long_probability(self, price, vah, poc, val, vwap,
                                     rsi, rvol, weekly, squeeze, iv) -> float:
        prob = 50

        # Position-based
        if price <= val:
            prob = self.BASE_VAL_BOUNCE
        elif price <= poc:
            va_range = vah - val
            pos_pct = (price - val) / va_range * 100 if va_range > 0 else 50
            prob = 55 + (pos_pct / 100 * 10)
        elif price <= vah:
            va_range = vah - val
            pos_pct = (price - val) / va_range * 100 if va_range > 0 else 50
            prob = 50 + (50 - pos_pct) / 100 * 5
        else:
            prob = 45 + self.EXTENDED_PENALTY

        # VWAP
        if price > vwap:
            prob += self.VWAP_TREND_BONUS

        # RSI (Wilder's)
        if rsi < 30:
            prob += 12  # Oversold bounce
        elif rsi < 40:
            prob += 5   # Approaching oversold
        elif rsi > 70:
            prob -= 8   # Overbought caution

        # Volume
        if rvol > 1.5 and price > vwap:
            prob += 5
        elif rvol > 1.5 and price < vwap:
            prob -= 5

        # NEW: Weekly alignment
        if weekly.supports_long:
            prob += self.WEEKLY_ALIGNED_BONUS

        # NEW: Squeeze (breakout potential for longs above POC)
        if squeeze.is_squeezed and price > poc:
            prob += self.SQUEEZE_BONUS

        # NEW: Low IV bonus (cheap calls)
        if iv.iv_regime == "low":
            prob += self.IV_LOW_BONUS

        return max(20, min(90, prob))

    def _calculate_short_probability(self, price, vah, poc, val, vwap,
                                      rsi, rvol, weekly, squeeze, iv) -> float:
        prob = 50

        if price >= vah:
            prob = self.BASE_VAH_REJECTION
        elif price >= poc:
            va_range = vah - val
            pos_pct = (price - val) / va_range * 100 if va_range > 0 else 50
            prob = 55 + ((100 - pos_pct) / 100 * 10)
        elif price >= val:
            va_range = vah - val
            pos_pct = (price - val) / va_range * 100 if va_range > 0 else 50
            prob = 50 + (pos_pct / 100 * 5)
        else:
            prob = 45 + self.EXTENDED_PENALTY

        if price < vwap:
            prob += self.VWAP_TREND_BONUS

        if rsi > 70:
            prob += 12
        elif rsi > 60:
            prob += 5
        elif rsi < 30:
            prob -= 8

        if rvol > 1.5 and price < vwap:
            prob += 5
        elif rvol > 1.5 and price > vwap:
            prob -= 5

        if weekly.supports_short:
            prob += self.WEEKLY_ALIGNED_BONUS

        if squeeze.is_squeezed and price < poc:
            prob += self.SQUEEZE_BONUS

        if iv.iv_regime == "low":
            prob += self.IV_LOW_BONUS

        return max(20, min(90, prob))

    # =========================================================================
    # GRADE & QUALITY
    # =========================================================================

    def _calculate_grade(self, prob, rr, direction, weekly_aligned, squeeze_active) -> Tuple[str, int, int]:
        """Returns (grade, conviction, quality_score)"""
        # Quality score 0-100
        quality = 0
        quality += min(35, int(prob * 0.4))          # Probability: up to 35
        quality += min(25, int(rr * 10))              # R:R: up to 25
        if weekly_aligned: quality += 15               # Weekly alignment
        if squeeze_active: quality += 10               # Squeeze active
        quality += min(15, int((prob - 50) * 0.5))    # Excess probability bonus
        quality = max(0, min(100, quality))

        # Letter grade
        if prob >= 75 and rr >= 2.5 and weekly_aligned:
            grade = 'A+'; base_conv = 10
        elif prob >= 70 and rr >= 2.5:
            grade = 'A'; base_conv = 9
        elif prob >= 65 and rr >= 2.0:
            grade = 'A' if weekly_aligned else 'B'; base_conv = 8
        elif prob >= 60 and rr >= 2.0:
            grade = 'B'; base_conv = 7
        elif prob >= 55 and rr >= 1.5:
            grade = 'C'; base_conv = 5
        elif prob >= 50 and rr >= 1.0:
            grade = 'C'; base_conv = 4
        else:
            grade = 'F'; base_conv = 2

        if not weekly_aligned and grade != 'F':
            base_conv -= 1
        if squeeze_active:
            base_conv += 1

        conviction = max(1, min(10, base_conv))
        return grade, conviction, quality

    # =========================================================================
    # SETUP GENERATION
    # =========================================================================

    def _generate_long_setup(self, price, vah, poc, val, vwap, atr, rsi, rvol,
                              weekly, squeeze, iv) -> Setup:
        prob = self._calculate_long_probability(price, vah, poc, val, vwap,
                                                  rsi, rvol, weekly, squeeze, iv)

        # Entry levels
        if price > vah:
            entry_low, entry_high = vah, vah + atr * 0.2
        elif price > poc:
            entry_low, entry_high = poc, poc + atr * 0.3
        else:
            entry_low, entry_high = val, min(price, poc)
        aggressive = price

        # Stop
        if price > poc:
            stop = poc - atr * 0.5
        else:
            stop = val - atr * 0.5
        stop = max(stop, price * 0.95)

        # Risk
        risk = entry_high - stop if entry_high > stop else atr * 0.5

        # Targets with minimum distances
        min_t1 = max(risk * 1.5, price * 0.01, atr * 0.75)
        min_t2 = max(risk * 2.5, price * 0.02, atr * 1.5)

        if price < vah:
            target_1 = max(vah, entry_high + min_t1)
            target_2 = max(vah + atr, entry_high + min_t2)
        else:
            target_1 = max(vah + atr, entry_high + min_t1)
            target_2 = max(vah + atr * 2, entry_high + min_t2)
        if target_2 <= target_1:
            target_2 = target_1 + atr * 0.75

        reward = target_1 - entry_high
        rr = reward / risk if risk > 0 else 0
        ev = (prob / 100 * reward) - ((100 - prob) / 100 * risk)

        grade, conviction, quality = self._calculate_grade(
            prob, rr, 'LONG', weekly.supports_long, squeeze.is_squeezed)

        prob_label = "High" if prob >= 65 else "Medium" if prob >= 55 else "Low"

        # Setup classification
        setup_type = ""
        if squeeze.is_squeezed and price > poc:
            setup_type = "compression_break_long"
        elif price <= val and rsi < 35:
            setup_type = "val_exhaustion_long"
        elif price <= val:
            setup_type = "val_bounce_long"
        elif price <= poc:
            setup_type = "poc_support_long"
        elif weekly.supports_long and rsi < 40:
            setup_type = "weekly_pullback_long"
        else:
            setup_type = "continuation_long"

        # Trigger & invalidation
        if price < val:
            trigger = f"Bounce from VAL ${val:.2f} with volume > 1.5x avg"
            invalidation = f"Close below ${val:.2f} (VAL)"
        elif price < poc:
            trigger = f"Hold above VAL ${val:.2f} with bullish candle"
            invalidation = f"Break below ${stop:.2f}"
        elif price < vah:
            trigger = f"Break above POC ${poc:.2f} with increasing volume"
            invalidation = f"Break below ${stop:.2f}"
        else:
            trigger = f"Pullback to VAH ${vah:.2f} holds as support"
            invalidation = f"Break below ${stop:.2f}"

        # Why
        reasons = []
        if price > vwap: reasons.append("price above VWAP")
        if price > poc: reasons.append("above POC")
        elif price <= val: reasons.append("at VAL support")
        if weekly.supports_long: reasons.append(f"weekly {weekly.trend} supports")
        if squeeze.is_squeezed: reasons.append("squeeze active")
        if rsi < 40: reasons.append(f"RSI oversold ({rsi:.0f})")
        why = f"{', '.join(reasons[:3])} — {prob_label.lower()} probability continuation" if reasons else "Structure favors upside"

        return Setup(
            direction='LONG', grade=grade, conviction=conviction,
            probability=int(prob), probability_low=int(prob - 5), probability_high=int(prob + 5),
            probability_label=prob_label,
            entry_low=round(entry_low, 2), entry_high=round(entry_high, 2),
            aggressive_entry=round(aggressive, 2), stop=round(stop, 2),
            target_1=round(target_1, 2), target_2=round(target_2, 2),
            risk_reward=round(rr, 1), ev=round(ev, 2),
            trigger=trigger, invalidation=invalidation, why=why,
            quality_score=quality, setup_type=setup_type,
            weekly_aligned=weekly.supports_long, squeeze_active=squeeze.is_squeezed
        )

    def _generate_short_setup(self, price, vah, poc, val, vwap, atr, rsi, rvol,
                               weekly, squeeze, iv) -> Setup:
        prob = self._calculate_short_probability(price, vah, poc, val, vwap,
                                                   rsi, rvol, weekly, squeeze, iv)

        if price < val:
            entry_low, entry_high = val - atr * 0.2, val
        elif price < poc:
            entry_low, entry_high = poc - atr * 0.3, poc
        else:
            entry_low, entry_high = max(price, poc), vah
        aggressive = price

        if price < poc:
            stop = poc + atr * 0.5
        else:
            stop = vah + atr * 0.5
        stop = min(stop, price * 1.05)

        risk = stop - entry_low if stop > entry_low else atr * 0.5

        min_t1 = max(risk * 1.5, price * 0.01, atr * 0.75)
        min_t2 = max(risk * 2.5, price * 0.02, atr * 1.5)

        if price > val:
            target_1 = min(val, entry_low - min_t1)
            target_2 = min(val - atr, entry_low - min_t2)
        else:
            target_1 = min(val - atr, entry_low - min_t1)
            target_2 = min(val - atr * 2, entry_low - min_t2)
        if target_2 >= target_1:
            target_2 = target_1 - atr * 0.75

        reward = entry_low - target_1
        rr = reward / risk if risk > 0 else 0
        ev = (prob / 100 * reward) - ((100 - prob) / 100 * risk)

        grade, conviction, quality = self._calculate_grade(
            prob, rr, 'SHORT', weekly.supports_short, squeeze.is_squeezed)

        prob_label = "High" if prob >= 65 else "Medium" if prob >= 55 else "Low"

        setup_type = ""
        if squeeze.is_squeezed and price < poc:
            setup_type = "compression_break_short"
        elif price >= vah and rsi > 65:
            setup_type = "vah_exhaustion_short"
        elif price >= vah:
            setup_type = "vah_rejection_short"
        elif price >= poc:
            setup_type = "poc_resistance_short"
        elif weekly.supports_short and rsi > 60:
            setup_type = "weekly_fade_short"
        else:
            setup_type = "continuation_short"

        if price > vah:
            trigger = f"Rejection at VAH ${vah:.2f} with volume > 1.5x avg"
            invalidation = f"Close above ${vah:.2f} (VAH)"
        elif price > poc:
            trigger = f"Fail below VAH ${vah:.2f} with bearish candle"
            invalidation = f"Break above ${stop:.2f}"
        elif price > val:
            trigger = f"Break below POC ${poc:.2f} with increasing volume"
            invalidation = f"Break above ${stop:.2f}"
        else:
            trigger = f"Rally to VAL ${val:.2f} rejected as resistance"
            invalidation = f"Break above ${stop:.2f}"

        reasons = []
        if price < vwap: reasons.append("price below VWAP")
        if price < poc: reasons.append("below POC")
        elif price >= vah: reasons.append("at VAH resistance")
        if weekly.supports_short: reasons.append(f"weekly {weekly.trend} supports")
        if squeeze.is_squeezed: reasons.append("squeeze active")
        if rsi > 60: reasons.append(f"RSI overbought ({rsi:.0f})")
        why = f"{', '.join(reasons[:3])} — {prob_label.lower()} probability reversal" if reasons else "Structure favors downside"

        return Setup(
            direction='SHORT', grade=grade, conviction=conviction,
            probability=int(prob), probability_low=int(prob - 5), probability_high=int(prob + 5),
            probability_label=prob_label,
            entry_low=round(entry_low, 2), entry_high=round(entry_high, 2),
            aggressive_entry=round(aggressive, 2), stop=round(stop, 2),
            target_1=round(target_1, 2), target_2=round(target_2, 2),
            risk_reward=round(rr, 1), ev=round(ev, 2),
            trigger=trigger, invalidation=invalidation, why=why,
            quality_score=quality, setup_type=setup_type,
            weekly_aligned=weekly.supports_short, squeeze_active=squeeze.is_squeezed
        )

    # =========================================================================
    # VERDICT
    # =========================================================================

    def _determine_verdict(self, long_s, short_s, price, vah, poc, val, vwap,
                            weekly, squeeze) -> Tuple[str, str]:
        reasons = []
        long_score = 0
        short_score = 0

        # Grade comparison
        grade_val = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'F': 1}
        lg = grade_val.get(long_s.grade, 0)
        sg = grade_val.get(short_s.grade, 0)
        if lg > sg: long_score += 2
        elif sg > lg: short_score += 2

        # Probability
        if long_s.probability > short_s.probability: long_score += 1
        elif short_s.probability > long_s.probability: short_score += 1

        # VWAP
        if price > vwap:
            long_score += 1; reasons.append("VWAP support")
        else:
            short_score += 1; reasons.append("below VWAP")

        # VP position
        if price < poc and price > val:
            long_score += 1; reasons.append("near value support")
        elif price > poc and price < vah:
            short_score += 1; reasons.append("near value resistance")

        # Weekly (V2)
        if weekly.supports_long:
            long_score += 2; reasons.append(f"weekly {weekly.trend}")
        elif weekly.supports_short:
            short_score += 2; reasons.append(f"weekly {weekly.trend}")

        # Squeeze (V2) — bias toward breakout direction
        if squeeze.is_squeezed:
            if price > poc:
                long_score += 1; reasons.append("squeeze breakout bias up")
            else:
                short_score += 1; reasons.append("squeeze breakout bias down")

        if long_score > short_score:
            return 'LONG', f"{' + '.join(reasons[:3])} favor long"
        elif short_score > long_score:
            return 'SHORT', f"{' + '.join(reasons[:3])} favor short"
        else:
            return 'NEUTRAL', "conflicting signals — wait for direction"

    # =========================================================================
    # OPTIONS STRATEGY (V2: Single-Winner Scaling)
    # =========================================================================

    def _generate_options_strategy(self, direction: str, price: float,
                                     atr: float, setup: Setup,
                                     iv: OptionsContext) -> Dict:
        """
        V2: Single-winner scaling replaces dual-hedge approach.
        - Enter at suggested delta (0.60-0.70 based on IV)
        - Minimum 3 weeks DTE (21 days)
        - -12.5% contract stop
        - Scale into winners at +15% and +25%
        - Gain protection: sell 50% at target or buy protective weekly
        """
        if direction == 'NEUTRAL':
            return {'strategy': 'WAIT', 'reason': 'No clear direction — wait for setup'}

        def round_strike(p, inc=5):
            if p < 50: inc = 1
            elif p < 100: inc = 2.5
            return round(p / inc) * inc

        if direction == 'LONG':
            strike = round_strike(price * 0.98)  # Slightly ITM
            option_type = 'CALL'
            stop_desc = f"Contract value drops {iv.contract_stop_pct}% from entry"
        else:
            strike = round_strike(price * 1.02)
            option_type = 'PUT'
            stop_desc = f"Contract value drops {iv.contract_stop_pct}% from entry"

        return {
            'strategy': f'SINGLE-WINNER {option_type}',
            'bias': direction,
            'option_type': option_type,
            'strike': strike,
            'delta': f'{iv.suggested_delta:.2f}',
            'dte': f'{iv.min_dte}+ days (3+ weeks)',
            'iv_regime': iv.iv_regime,
            'iv_percentile': iv.iv_percentile,

            'entry': {
                'size': iv.entry_size,
                'timing': f"On {setup.trigger}",
                'stop': stop_desc,
            },

            'scaling': {
                'scale_1': f"+15% contract gain → add {iv.entry_size} more",
                'scale_2': f"+25% contract gain → add final position",
                'full_size': 'After 2 scale-ins at full allocation',
            },

            'management': {
                'contract_stop': f'-{iv.contract_stop_pct}% from entry',
                'gain_protection': 'At +40-50%: sell 50% of contracts OR buy ATM weekly opposite',
                'target_1': f'${setup.target_1:.2f}',
                'target_2': f'${setup.target_2:.2f}',
                'time_stop': 'If no movement in 5 days, re-evaluate thesis',
            },

            'edge': f'{iv.scale_plan}. Winners get bigger, losers get cut at -{iv.contract_stop_pct}%.'
        }

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> DualSetupResult:
        """
        Self-contained dual setup analysis.

        Args:
            df: OHLCV DataFrame (any timeframe, daily recommended)
            symbol: Stock symbol for weekly context

        Returns:
            DualSetupResult with LONG, SHORT, verdict, and options
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if len(df) < 20:
            return DualSetupResult(symbol=symbol)

        price = float(df['close'].iloc[-1])

        # Core calculations
        vp_lookback = min(60, len(df))
        vp = self._calculate_volume_profile(df.tail(vp_lookback))
        atr = self._calculate_atr(df)
        rsi_series = self._calculate_rsi_wilder(df['close'])
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        rvol = self._calculate_rvol(df)

        # V2 context
        weekly = self._calculate_weekly_context(symbol)
        squeeze = self._detect_squeeze(df)
        iv = self._estimate_iv_percentile(df)

        # Generate setups
        long_setup = self._generate_long_setup(
            price, vp.vah, vp.poc, vp.val, vp.vwap, atr, rsi, rvol,
            weekly, squeeze, iv)

        short_setup = self._generate_short_setup(
            price, vp.vah, vp.poc, vp.val, vp.vwap, atr, rsi, rvol,
            weekly, squeeze, iv)

        # Verdict
        preferred, verdict_reason = self._determine_verdict(
            long_setup, short_setup, price, vp.vah, vp.poc, vp.val, vp.vwap,
            weekly, squeeze)

        # Key level
        poc_dist = abs(price - vp.poc)
        vwap_dist = abs(price - vp.vwap)
        key_level = vp.poc if poc_dist < vwap_dist else vp.vwap

        # Bookmap checklist
        bookmap_long = f"absorption at ${long_setup.entry_low:.2f} (buyers absorbing), delta flip positive, iceberg bids"
        bookmap_short = f"absorption at ${short_setup.entry_high:.2f} (sellers absorbing), delta flip negative, iceberg offers"

        # Options strategy for preferred direction
        primary_setup = long_setup if preferred == 'LONG' else short_setup
        options_strategy = self._generate_options_strategy(
            preferred, price, atr, primary_setup, iv)

        # Earnings context
        earnings_context = None
        if earnings_available:
            try:
                cal = get_earnings_calendar()
                earnings_context = cal.get_earnings_context(symbol)
            except:
                pass

        return DualSetupResult(
            symbol=symbol, current_price=round(price, 2),
            timestamp=datetime.now().isoformat(),
            long_setup=long_setup, short_setup=short_setup,
            preferred_direction=preferred, verdict_reason=verdict_reason,
            key_level=round(key_level, 2),
            key_level_desc=f"${key_level:.2f} — Above = Long, Below = Short",
            bookmap_long=bookmap_long, bookmap_short=bookmap_short,
            options_strategy=options_strategy, earnings_context=earnings_context,
            volume_profile=vp, weekly=weekly, squeeze=squeeze,
            options_context=iv, rsi=round(rsi, 2), atr=round(atr, 2)
        )

    # =========================================================================
    # V1-COMPATIBLE generate() — accepts dict
    # =========================================================================

    def generate(self, data: Dict) -> DualSetupResult:
        """V1-compatible: accepts a dict of scanner data"""
        symbol = data.get('symbol', 'UNKNOWN')
        price = float(data.get('current_price') or data.get('price') or 0)

        # If DataFrame passed, use self-contained path
        if 'df' in data and isinstance(data['df'], pd.DataFrame):
            return self.analyze(data['df'], symbol)

        # Otherwise reconstruct from dict (V1 path)
        vah = float(data.get('vah') or 0)
        poc = float(data.get('poc') or 0)
        val = float(data.get('val') or 0)
        vwap = float(data.get('vwap') or poc)
        atr = float(data.get('atr') or (vah - val) * 0.3)
        rsi = float(data.get('rsi') or 50)
        rvol = float(data.get('rvol') or 1.0)

        # Use defaults for V2 context when not self-contained
        weekly = WeeklyContext()
        squeeze = SqueezeContext()
        iv = OptionsContext()

        long_setup = self._generate_long_setup(
            price, vah, poc, val, vwap, atr, rsi, rvol, weekly, squeeze, iv)
        short_setup = self._generate_short_setup(
            price, vah, poc, val, vwap, atr, rsi, rvol, weekly, squeeze, iv)

        preferred, verdict_reason = self._determine_verdict(
            long_setup, short_setup, price, vah, poc, val, vwap, weekly, squeeze)

        poc_dist = abs(price - poc)
        vwap_dist = abs(price - vwap)
        key_level = poc if poc_dist < vwap_dist else vwap

        bookmap_long = f"absorption at ${long_setup.entry_low:.2f}, delta flip positive"
        bookmap_short = f"absorption at ${short_setup.entry_high:.2f}, delta flip negative"

        primary_setup = long_setup if preferred == 'LONG' else short_setup
        options_strategy = self._generate_options_strategy(preferred, price, atr, primary_setup, iv)

        earnings_context = None
        if earnings_available:
            try:
                cal = get_earnings_calendar()
                earnings_context = cal.get_earnings_context(symbol)
            except:
                pass

        vp = VolumeProfileLevels(poc=poc, vah=vah, val=val, vwap=vwap)

        return DualSetupResult(
            symbol=symbol, current_price=round(price, 2),
            timestamp=datetime.now().isoformat(),
            long_setup=long_setup, short_setup=short_setup,
            preferred_direction=preferred, verdict_reason=verdict_reason,
            key_level=round(key_level, 2),
            key_level_desc=f"${key_level:.2f} — Above = Long, Below = Short",
            bookmap_long=bookmap_long, bookmap_short=bookmap_short,
            options_strategy=options_strategy, earnings_context=earnings_context,
            volume_profile=vp, weekly=weekly, squeeze=squeeze,
            options_context=iv, rsi=round(rsi, 2), atr=round(atr, 2)
        )

    # =========================================================================
    # FORMAT OUTPUT (V1 compatible with parseAIResponse)
    # =========================================================================

    def format_as_ai_text(self, result: DualSetupResult) -> str:
        """Format as AI commentary text — compatible with parseAIResponse()"""
        long = result.long_setup
        short = result.short_setup
        opts = result.options_strategy

        text = f"""🟢 LONG SETUP
⭐ GRADE: {long.grade} ({long.quality_score}/100) | 🎯 CONVICTION: {long.conviction}/10
📈 PROBABILITY: {long.probability_low}-{long.probability_high}% [{long.probability_label}]
🔍 ENTRY: ${long.entry_low:.2f} - ${long.entry_high:.2f} (conservative, wait for pullback)
⚡ AGGRESSIVE: ${long.aggressive_entry:.2f} NOW
🛑 STOP: ${long.stop:.2f}
💰 T1: ${long.target_1:.2f} | 🚀 T2: ${long.target_2:.2f}
📐 R:R: {long.risk_reward:.1f}:1 | 💹 EV: ${long.ev:.2f}
✅ TRIGGER: {long.trigger}
❌ INVALID: {long.invalidation}
💡 WHY: {long.why}
🏷️ TYPE: {long.setup_type}{"  ✅ Weekly aligned" if long.weekly_aligned else ""}{"  🔲 Squeeze" if long.squeeze_active else ""}

🔴 SHORT SETUP
⭐ GRADE: {short.grade} ({short.quality_score}/100) | 🎯 CONVICTION: {short.conviction}/10
📈 PROBABILITY: {short.probability_low}-{short.probability_high}% [{short.probability_label}]
🔍 ENTRY: ${short.entry_low:.2f} - ${short.entry_high:.2f} (conservative, wait for rally)
⚡ AGGRESSIVE: ${short.aggressive_entry:.2f} NOW
🛑 STOP: ${short.stop:.2f}
💰 T1: ${short.target_1:.2f} | 🚀 T2: ${short.target_2:.2f}
📐 R:R: {short.risk_reward:.1f}:1 | 💹 EV: ${short.ev:.2f}
✅ TRIGGER: {short.trigger}
❌ INVALID: {short.invalidation}
💡 WHY: {short.why}
🏷️ TYPE: {short.setup_type}{"  ✅ Weekly aligned" if short.weekly_aligned else ""}{"  🔲 Squeeze" if short.squeeze_active else ""}

⚖️ VERDICT: {result.preferred_direction} preferred because {result.verdict_reason}
⚠️ KEY LEVEL: {result.key_level_desc}

📊 BOOKMAP ORDER FLOW CHECKLIST (confirm before entry):
🔍 LONG: Look for {result.bookmap_long}
🔍 SHORT: Look for {result.bookmap_short}"""

        # Earnings warning
        earnings = result.earnings_context
        if earnings and earnings.get('has_upcoming') and earnings.get('days_until') is not None:
            days = earnings['days_until']
            date_str = earnings.get('date', 'N/A')
            timing = earnings.get('timing', '')
            timing_str = f" ({timing})" if timing else ""
            if days == 0:
                text += f"\n\n🚨 EARNINGS TODAY{timing_str} — IV CRUSH RISK — AVOID OPTIONS"
            elif days <= 3:
                text += f"\n\n⚠️ EARNINGS in {days} days ({date_str}{timing_str}) — IV ELEVATED"
                text += "\n   💡 Compression may be pre-earnings IV buildup, not organic"
            elif days <= 14:
                text += f"\n\n📅 EARNINGS in {days} days ({date_str}{timing_str}) — watch IV"

        # Options strategy (V2: single-winner)
        if opts and opts.get('strategy') != 'WAIT':
            e = opts.get('entry', {})
            s = opts.get('scaling', {})
            m = opts.get('management', {})

            text += f"""

🎰 OPTIONS: {opts.get('strategy', 'N/A')}
📊 Strike: ${opts.get('strike', 0):.2f} | Delta: {opts.get('delta', '0.65')} | DTE: {opts.get('dte', '21+')}
📋 IV: {opts.get('iv_percentile', 50):.0f}%ile ({opts.get('iv_regime', 'normal')})

📥 ENTRY:
   Size: {e.get('size', '50%')} | On: {e.get('timing', 'trigger confirmation')}
   Stop: {e.get('stop', '-12.5% contract value')}

📈 SCALING:
   {s.get('scale_1', '+15% → add')}
   {s.get('scale_2', '+25% → add final')}

🎯 MANAGEMENT:
   Contract stop: {m.get('contract_stop', '-12.5%')}
   Gain protection: {m.get('gain_protection', 'Sell 50% at +40-50%')}
   T1: {m.get('target_1', 'N/A')} | T2: {m.get('target_2', 'N/A')}

💡 EDGE: {opts.get('edge', 'Scale winners, cut losers fast.')}"""

        return text

    def to_dict(self, result: DualSetupResult) -> Dict:
        return asdict(result)


# =============================================================================
# QUICK SCAN FUNCTIONS
# =============================================================================

def scan_dual_setup(symbol: str, period: str = "3mo",
                     interval: str = "1d") -> Optional[DualSetupResult]:
    """Quick scan a single symbol"""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return None
    gen = DualSetupGeneratorV2()
    return gen.analyze(df, symbol)


def scan_symbols(symbols: List[str], period: str = "3mo",
                  interval: str = "1d",
                  min_grade: str = "C") -> List[DualSetupResult]:
    """Batch scan and return results sorted by preferred setup quality"""
    import yfinance as yf

    grade_order = {'A+': 6, 'A': 5, 'B': 4, 'C': 3, 'F': 1}
    min_val = grade_order.get(min_grade, 3)

    gen = DualSetupGeneratorV2()
    results = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                continue

            result = gen.analyze(df, symbol)
            if result:
                preferred = (result.long_setup if result.preferred_direction == 'LONG'
                           else result.short_setup)
                if grade_order.get(preferred.grade, 0) >= min_val:
                    results.append(result)

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    results.sort(key=lambda r: max(r.long_setup.quality_score,
                                    r.short_setup.quality_score), reverse=True)
    return results


# =============================================================================
# V1 COMPATIBILITY
# =============================================================================

class DualSetupGenerator(DualSetupGeneratorV2):
    """Backward-compatible wrapper. Use DualSetupGeneratorV2 for new code."""
    pass


def generate_dual_setup(data: Dict) -> str:
    """V1-compatible: generate formatted text from scanner data dict"""
    gen = DualSetupGeneratorV2()
    result = gen.generate(data)
    return gen.format_as_ai_text(result)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  DUAL SETUP GENERATOR V2 — C.O.R.E. Methodology")
    print("=" * 65)

    test_symbols = ["NVDA", "AAPL", "META", "TSLA", "AMD"]

    gen = DualSetupGeneratorV2()

    for symbol in test_symbols:
        print(f"\nAnalyzing {symbol}...")
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")

            if df.empty:
                print(f"  No data for {symbol}")
                continue

            result = gen.analyze(df, symbol)
            print(gen.format_as_ai_text(result))

        except Exception as e:
            print(f"  Error: {e}")
        print("-" * 65)
