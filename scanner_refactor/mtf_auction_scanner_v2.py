"""
MTF Auction Scanner V2 — Multi-Timeframe Non-Bias Setup Detection
==================================================================
Directionally neutral scanner that identifies HIGH and LOW scenario setups
across 30min to 4hr timeframes using Volume Profile, Flow Control, RSI,
VWAP, and V2 enrichment (weekly structure, squeeze, IV, enhanced VP).

V1 → V2 CHANGES:
- Import canonical TechnicalCalculator from market_scanner_v2 (VP, RSI, ATR, RVOL)
- Wilder's RSI smoothing (alpha=1/period) replaces EWM span-based
- V2Context integration: weekly trend, squeeze state, IV regime, VP shape
  all feed into scoring as contextual multipliers
- Enhanced SignalScorer: +25 pts for squeeze active, +20 for weekly alignment,
  +15 for VP shape confirmation, IV regime adjusts confidence
- scan_enriched() returns (ScanResult, V2Context) tuples
- FlowControlEngine retained (CLV-based delta is unique here)
- VWAPEngine retained (std dev bands are unique here)
- SignalScorer enhanced with V2 context bonuses
- All V1 scan/report methods preserved for backward compat

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import canonical V2 components
try:
    from market_scanner_v2 import (
        TechnicalCalculator,
        V2Context, WeeklyContext, SqueezeContext, IVContext, EnhancedVP,
        MarketScanner as BaseMarketScanner,
    )
    v2_available = True
except ImportError:
    v2_available = False
    # Stub for standalone usage
    class TechnicalCalculator:
        pass
    class V2Context:
        pass

# Import scanner config (optional — uses defaults if unavailable)
try:
    from scanner_config import (
        SwingTradeConfig, ScoringConfig, VolumeProfileConfig,
        RSIConfig, FlowConfig, TimeframeConfig,
    )
    _config_available = True
except ImportError:
    _config_available = False


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class SignalState(Enum):
    """Signal states with explicit neutral handling"""
    LONG_SETUP = "LONG_SETUP"
    SHORT_SETUP = "SHORT_SETUP"
    YELLOW = "YELLOW"
    NEUTRAL = "NEUTRAL"
    NO_DATA = "NO_DATA"

    @property
    def emoji(self) -> str:
        return {
            "LONG_SETUP": "🟢", "SHORT_SETUP": "🔴",
            "YELLOW": "🟡", "NEUTRAL": "⚪", "NO_DATA": "⬜"
        }[self.value]

    @property
    def action(self) -> str:
        return {
            "LONG_SETUP": "PREPARE LONG", "SHORT_SETUP": "PREPARE SHORT",
            "YELLOW": "WAIT - MIXED", "NEUTRAL": "NO SETUP", "NO_DATA": "SKIP"
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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VolumeProfile:
    """Volume Profile metrics"""
    poc: float
    vah: float
    val: float
    value_area_pct: float
    total_volume: float
    price_levels: Dict[float, float] = field(default_factory=dict)

    @property
    def value_width(self) -> float:
        return self.vah - self.val

    @property
    def value_center(self) -> float:
        return (self.vah + self.val) / 2


@dataclass
class FlowMetrics:
    """Order flow and delta metrics"""
    cumulative_delta: float
    delta_momentum: float
    buy_volume_pct: float
    sell_volume_pct: float
    flow_imbalance: float      # -1 to +1

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
        return "BALANCED"


@dataclass
class RSIMetrics:
    """RSI with zone classification"""
    value: float
    slope: float
    divergence: Optional[str]    # None, BULLISH_DIV, BEARISH_DIV

    @property
    def zone(self) -> str:
        if self.value >= 75:
            return "OVERBOUGHT"
        elif self.value >= 65:
            return "NEAR_OVERBOUGHT"
        elif self.value >= 55:
            return "BULLISH"
        elif self.value >= 45:
            return "NEUTRAL"
        elif self.value >= 35:
            return "BEARISH"
        elif self.value >= 30:
            return "NEAR_OVERSOLD"
        return "OVERSOLD"

    @property
    def momentum_aligned(self) -> Optional[bool]:
        if self.zone in ("BULLISH", "OVERBOUGHT", "NEAR_OVERBOUGHT") and self.slope > 0:
            return True
        elif self.zone in ("BEARISH", "OVERSOLD", "NEAR_OVERSOLD") and self.slope < 0:
            return True
        elif self.zone == "NEUTRAL":
            return None
        return False


@dataclass
class VWAPMetrics:
    """VWAP with deviation bands"""
    vwap: float
    upper_band_1: float
    lower_band_1: float
    upper_band_2: float
    lower_band_2: float
    price_vs_vwap: float
    deviation_pct: float

    @property
    def zone(self) -> str:
        if self.price_vs_vwap > self.upper_band_2 - self.vwap:
            return "EXTREME_ABOVE"
        elif self.price_vs_vwap > self.upper_band_1 - self.vwap:
            return "ABOVE_1SD"
        elif self.price_vs_vwap > 0:
            return "ABOVE_VWAP"
        elif self.price_vs_vwap > -(self.vwap - self.lower_band_1):
            return "BELOW_VWAP"
        elif self.price_vs_vwap > -(self.vwap - self.lower_band_2):
            return "BELOW_1SD"
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
    price_vs_poc: float
    price_vs_vah: float
    price_vs_val: float
    position_in_value: str       # ABOVE_VALUE, IN_VALUE, BELOW_VALUE
    bull_score: float
    bear_score: float
    signal: SignalState
    confidence: float
    vwap: Optional[VWAPMetrics] = None
    notes: List[str] = field(default_factory=list)

    # V2: context bonus points applied
    v2_bull_bonus: float = 0.0
    v2_bear_bonus: float = 0.0


@dataclass
class ScanResult:
    """Complete scan result across all timeframes"""
    symbol: str
    scan_time: datetime
    timeframe_analyses: Dict[Timeframe, TimeframeAnalysis]
    dominant_signal: SignalState
    confluence_score: float
    actionable: bool
    high_scenario_prob: float
    low_scenario_prob: float
    neutral_prob: float
    summary: str

    # V2: enrichment context (None if V2 not available)
    v2_context: Optional[object] = None


# =============================================================================
# VOLUME PROFILE ENGINE (uses canonical calculator)
# =============================================================================

class VolumeProfileEngine:
    """
    VP calculation — delegates to canonical TechnicalCalculator for core POC/VAH/VAL,
    adds price_levels histogram for this scanner's detailed analysis.
    """

    def __init__(self, value_area_pct: float = 0.70, num_bins: int = 50,
                 config: Optional[object] = None):
        if config is not None and hasattr(config, 'value_area_pct'):
            self.value_area_pct = config.value_area_pct
            self.num_bins = config.num_bins
        else:
            self.value_area_pct = value_area_pct
            self.num_bins = num_bins

    def calculate(self, df: pd.DataFrame) -> VolumeProfile:
        if len(df) < 5:
            mid = df['close'].iloc[-1] if len(df) > 0 else 0
            return VolumeProfile(poc=mid, vah=mid, val=mid, value_area_pct=0, total_volume=0)

        # Use canonical calculator for core VP
        if v2_available:
            poc, vah, val = TechnicalCalculator.calculate_volume_profile(
                df, self.value_area_pct, self.num_bins
            )
        else:
            poc, vah, val = self._fallback_vp(df)

        # Build price_levels histogram for detailed flow analysis
        price_min = df['low'].min()
        price_max = df['high'].max()
        if price_max == price_min:
            price_max = price_min * 1.001

        bin_size = (price_max - price_min) / self.num_bins
        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        volume_at_price = np.zeros(self.num_bins)

        for _, row in df.iterrows():
            bar_low, bar_high = row['low'], row['high']
            bar_volume = row['volume']
            typical_price = (row['high'] + row['low'] + row['close']) / 3

            low_bin = max(0, int((bar_low - price_min) / bin_size))
            high_bin = min(self.num_bins - 1, int((bar_high - price_min) / bin_size))
            typical_bin = min(self.num_bins - 1, max(0, int((typical_price - price_min) / bin_size)))

            for b in range(low_bin, high_bin + 1):
                distance = abs(b - typical_bin)
                weight = 1 / (1 + distance * 0.5)
                volume_at_price[b] += bar_volume * weight

        total_vol = volume_at_price.sum()
        if total_vol > 0:
            volume_at_price = volume_at_price / total_vol * df['volume'].sum()

        price_levels = {bin_centers[i]: volume_at_price[i] for i in range(self.num_bins)}

        return VolumeProfile(
            poc=poc, vah=vah, val=val,
            value_area_pct=self.value_area_pct,
            total_volume=df['volume'].sum(),
            price_levels=price_levels,
        )

    def _fallback_vp(self, df):
        """Fallback VP if canonical calculator not available"""
        mid = df['close'].mean()
        return mid, df['high'].quantile(0.85), df['low'].quantile(0.15)


# =============================================================================
# FLOW CONTROL ENGINE (unique to this scanner — CLV-based delta)
# =============================================================================

class FlowControlEngine:
    """
    Order flow analysis using Close Location Value (CLV) delta estimation.

    Without true tick data, estimates delta using price movement:
    - CLV = (close - low) / (high - low)  →  0=closed at low, 1=closed at high
    - Delta = volume × (2 × CLV - 1)
    - Cumulative delta tracks buyer/seller dominance over time
    - Delta momentum = rate of change of cumulative delta
    """

    def __init__(self, momentum_period: int = 5, config: Optional[object] = None):
        if config is not None and hasattr(config, 'momentum_period'):
            self.momentum_period = config.momentum_period
        else:
            self.momentum_period = momentum_period

    def calculate(self, df: pd.DataFrame) -> FlowMetrics:
        if len(df) < 3:
            return FlowMetrics(0, 0, 0.5, 0.5, 0)

        df = df.copy()
        df['delta'] = df.apply(self._estimate_bar_delta, axis=1)
        df['cum_delta'] = df['delta'].cumsum()

        cumulative_delta = df['cum_delta'].iloc[-1]

        if len(df) >= self.momentum_period:
            delta_momentum = (df['cum_delta'].iloc[-1] - df['cum_delta'].iloc[-self.momentum_period]) / self.momentum_period
        else:
            delta_momentum = 0

        total_volume = df['volume'].sum()
        if total_volume > 0:
            buy_volume = df[df['delta'] > 0]['volume'].sum()
            sell_volume = df[df['delta'] < 0]['volume'].sum()
            neutral_volume = df[df['delta'] == 0]['volume'].sum()
            buy_volume_pct = (buy_volume + neutral_volume * 0.5) / total_volume
            sell_volume_pct = (sell_volume + neutral_volume * 0.5) / total_volume
        else:
            buy_volume_pct = sell_volume_pct = 0.5

        flow_imbalance = buy_volume_pct - sell_volume_pct

        return FlowMetrics(
            cumulative_delta=cumulative_delta,
            delta_momentum=delta_momentum,
            buy_volume_pct=buy_volume_pct,
            sell_volume_pct=sell_volume_pct,
            flow_imbalance=flow_imbalance,
        )

    def _estimate_bar_delta(self, row: pd.Series) -> float:
        bar_range = row['high'] - row['low']
        if bar_range == 0:
            return 0
        clv = (row['close'] - row['low']) / bar_range
        return row['volume'] * (2 * clv - 1)


# =============================================================================
# RSI ENGINE (Wilder's smoothing — canonical)
# =============================================================================

class RSIEngine:
    """
    RSI with Wilder's smoothing (alpha = 1/period), slope, and divergence detection.

    V2 CHANGE: Switched from EWM span-based to Wilder's alpha=1/period
    to match canonical TechnicalCalculator and standard RSI definition.
    """

    def __init__(self, period: int = 14, slope_period: int = 3,
                 config: Optional[object] = None):
        if config is not None and hasattr(config, 'period'):
            self.period = config.period
            self.slope_period = config.slope_period
        else:
            self.period = period
            self.slope_period = slope_period

    def calculate(self, df: pd.DataFrame) -> RSIMetrics:
        if len(df) < self.period + 5:
            return RSIMetrics(value=50, slope=0, divergence=None)

        close = df['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        # V2: Wilder's smoothing (alpha = 1/period) — matches canonical
        alpha = 1.0 / self.period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))

        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # Slope
        if len(rsi.dropna()) >= self.slope_period:
            slope = float((rsi.iloc[-1] - rsi.iloc[-self.slope_period]) / self.slope_period)
        else:
            slope = 0.0

        # Divergence
        divergence = self._detect_divergence(df, rsi)

        return RSIMetrics(value=current_rsi, slope=slope, divergence=divergence)

    def _detect_divergence(self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 10) -> Optional[str]:
        if len(df) < lookback:
            return None

        recent_prices = df['close'].iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]

        price_high_idx = recent_prices.values.argmax()
        price_low_idx = recent_prices.values.argmin()

        price_highs = recent_prices.iloc[price_high_idx]
        rsi_at_high = recent_rsi.iloc[price_high_idx]
        price_lows = recent_prices.iloc[price_low_idx]
        rsi_at_low = recent_rsi.iloc[price_low_idx]

        current_price = df['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]

        if current_price >= price_highs * 0.998 and current_rsi < rsi_at_high - 5:
            return "BEARISH_DIV"
        if current_price <= price_lows * 1.002 and current_rsi > rsi_at_low + 5:
            return "BULLISH_DIV"
        return None


# =============================================================================
# VWAP ENGINE (unique — std dev bands)
# =============================================================================

class VWAPEngine:
    """
    VWAP with standard deviation bands.
    Institutional traders use VWAP as a benchmark — deviation bands
    show standard move ranges.
    """

    def __init__(self, band_mult_1: float = 1.0, band_mult_2: float = 2.0):
        self.band_mult_1 = band_mult_1
        self.band_mult_2 = band_mult_2

    def calculate(self, df: pd.DataFrame) -> VWAPMetrics:
        if len(df) < 5:
            price = df['close'].iloc[-1] if len(df) > 0 else 0
            return VWAPMetrics(price, price, price, price, price, 0, 0)

        df = df.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['cum_tp_volume'] = df['tp_volume'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        df['vwap'] = df['cum_tp_volume'] / df['cum_volume']

        df['squared_diff'] = (df['typical_price'] - df['vwap']) ** 2
        df['cum_squared_diff'] = (df['squared_diff'] * df['volume']).cumsum()
        df['variance'] = df['cum_squared_diff'] / df['cum_volume']
        df['std_dev'] = np.sqrt(df['variance'])

        vwap = float(df['vwap'].iloc[-1])
        std = float(df['std_dev'].iloc[-1])
        price = float(df['close'].iloc[-1])

        upper_1 = vwap + std * self.band_mult_1
        lower_1 = vwap - std * self.band_mult_1
        upper_2 = vwap + std * self.band_mult_2
        lower_2 = vwap - std * self.band_mult_2

        price_vs_vwap = price - vwap
        dev_pct = (price_vs_vwap / vwap * 100) if vwap != 0 else 0

        return VWAPMetrics(vwap, upper_1, lower_1, upper_2, lower_2, price_vs_vwap, dev_pct)


# =============================================================================
# SIGNAL SCORER V2 (non-bias + V2 context bonuses)
# =============================================================================

class SignalScorer:
    """
    Non-bias scoring: evaluates bullish and bearish evidence separately.

    V2 ENHANCEMENT: Optional V2Context adds contextual bonuses:
    - Squeeze active: +25 pts to dominant direction
    - Weekly trend alignment: +20 pts
    - VP shape confirmation: +15 pts (p-shape at lows, d-shape at highs)
    - RSI divergence co-detection with squeeze: +10 pts
    - IV regime adjusts confidence (elevated IV = slightly lower confidence)
    """

    # Defaults (overridden by config if provided)
    STRONG_THRESHOLD = 65
    MODERATE_THRESHOLD = 45
    MIN_SCORE_GAP = 20

    def __init__(self, config: Optional[object] = None, flow_config: Optional[object] = None):
        if config is not None and hasattr(config, 'strong_threshold'):
            self.STRONG_THRESHOLD = config.strong_threshold
            self.MODERATE_THRESHOLD = config.moderate_threshold
            self.MIN_SCORE_GAP = config.min_score_gap
        # Flow thresholds for scoring
        if flow_config is not None and hasattr(flow_config, 'strong_imbalance'):
            self._flow_strong = flow_config.strong_imbalance
            self._flow_moderate = flow_config.moderate_imbalance
            self._flow_mild = flow_config.mild_imbalance
        else:
            self._flow_strong = 0.30
            self._flow_moderate = 0.15
            self._flow_mild = 0.05

    def score(self,
              current_price: float,
              vp: VolumeProfile,
              flow: FlowMetrics,
              rsi: RSIMetrics,
              vwap: Optional[VWAPMetrics] = None,
              v2ctx: Optional[object] = None) -> Tuple[float, float, SignalState, float, List[str]]:
        """
        Score bullish and bearish evidence.

        Args:
            current_price: Current price
            vp: VolumeProfile metrics
            flow: FlowMetrics
            rsi: RSIMetrics
            vwap: Optional VWAPMetrics
            v2ctx: Optional V2Context for enrichment bonuses

        Returns:
            (bull_score, bear_score, signal, confidence, notes)
        """
        bull_score = 0.0
        bear_score = 0.0
        v2_bull_bonus = 0.0
        v2_bear_bonus = 0.0
        notes = []

        # =================================================================
        # PRICE VS VALUE AREA (25 pts max each direction)
        # =================================================================
        if current_price > vp.vah:
            distance_pct = (current_price - vp.vah) / vp.value_width if vp.value_width > 0 else 0
            if distance_pct < 0.5:
                bull_score += 20
                notes.append("Price above VAH — bullish breakout zone")
            else:
                bull_score += 8
                bear_score += 12
                notes.append("Price extended above VAH — watch for rejection")
        elif current_price < vp.val:
            distance_pct = (vp.val - current_price) / vp.value_width if vp.value_width > 0 else 0
            if distance_pct < 0.5:
                bear_score += 20
                notes.append("Price below VAL — bearish breakdown zone")
            else:
                bear_score += 8
                bull_score += 12
                notes.append("Price extended below VAL — watch for bounce")
        else:
            if current_price > vp.poc:
                bull_score += 8
                notes.append("Inside value, above POC")
            elif current_price < vp.poc:
                bear_score += 8
                notes.append("Inside value, below POC")
            else:
                notes.append("At POC — balanced")

        # =================================================================
        # VWAP ANALYSIS (20 pts max each direction)
        # =================================================================
        if vwap is not None:
            zone = vwap.zone
            dev = vwap.deviation_pct
            if zone == "EXTREME_ABOVE":
                bear_score += 15
                notes.append(f"Price extreme above VWAP (+{dev:.1f}%) — reversion risk")
            elif zone == "ABOVE_1SD":
                bull_score += 10
                notes.append(f"Price above VWAP +1SD ({dev:.1f}%)")
            elif zone == "ABOVE_VWAP":
                bull_score += 15
                notes.append(f"Price above VWAP (+{dev:.1f}%) — buyers in control")
            elif zone == "BELOW_VWAP":
                bear_score += 15
                notes.append(f"Price below VWAP ({dev:.1f}%) — sellers in control")
            elif zone == "BELOW_1SD":
                bear_score += 10
                notes.append(f"Price below VWAP -1SD ({dev:.1f}%)")
            elif zone == "EXTREME_BELOW":
                bull_score += 15
                notes.append(f"Price extreme below VWAP ({dev:.1f}%) — bounce likely")

            if abs(dev) < 0.3:
                notes.append("⚡ Price at VWAP — key decision point")

        # =================================================================
        # FLOW CONTROL (30 pts max each direction)
        # =================================================================
        fi = flow.flow_imbalance
        if fi > self._flow_strong:
            bull_score += 30; notes.append(f"Strong buy flow ({fi:.2f})")
        elif fi > self._flow_moderate:
            bull_score += 20; notes.append(f"Moderate buy flow ({fi:.2f})")
        elif fi > self._flow_mild:
            bull_score += 12; notes.append(f"Mild buy flow ({fi:.2f})")
        elif fi < -self._flow_strong:
            bear_score += 30; notes.append(f"Strong sell flow ({fi:.2f})")
        elif fi < -self._flow_moderate:
            bear_score += 20; notes.append(f"Moderate sell flow ({fi:.2f})")
        elif fi < -self._flow_mild:
            bear_score += 12; notes.append(f"Mild sell flow ({fi:.2f})")
        else:
            notes.append("Flow balanced")

        # Delta momentum bonus
        if flow.delta_momentum > 0 and fi > 0:
            bull_score += 8; notes.append("Accelerating buy pressure")
        elif flow.delta_momentum < 0 and fi < 0:
            bear_score += 8; notes.append("Accelerating sell pressure")

        # =================================================================
        # RSI (25 pts max each direction)
        # =================================================================
        rz = rsi.zone
        rv = rsi.value
        rs = rsi.slope

        if rz == "BULLISH":
            bull_score += 20; notes.append(f"RSI bullish zone ({rv:.1f})")
        elif rz == "OVERBOUGHT":
            if rs > 0:
                bull_score += 10; notes.append(f"RSI overbought but climbing ({rv:.1f})")
            else:
                bear_score += 15; notes.append(f"RSI overbought and rolling ({rv:.1f})")
        elif rz == "NEAR_OVERBOUGHT":
            if rs > 0:
                bull_score += 12; notes.append(f"RSI near overbought, climbing ({rv:.1f})")
            else:
                bear_score += 10; bull_score += 5
                notes.append(f"RSI near overbought, rolling ({rv:.1f}) ⚠️")
        elif rz == "BEARISH":
            bear_score += 15; notes.append(f"RSI bearish zone ({rv:.1f})")
        elif rz == "NEAR_OVERSOLD":
            if rs < 0:
                bear_score += 8; bull_score += 8
                notes.append(f"RSI near oversold ({rv:.1f}) — bounce potential")
            else:
                bull_score += 15; notes.append(f"RSI near oversold and turning ({rv:.1f}) ✔")
        elif rz == "OVERSOLD":
            if rs < 0:
                bear_score += 10; notes.append(f"RSI oversold but falling ({rv:.1f})")
            else:
                bull_score += 18; notes.append(f"RSI oversold and turning ({rv:.1f}) ✔")
        else:
            notes.append(f"RSI neutral ({rv:.1f})")

        # Divergence
        if rsi.divergence == "BULLISH_DIV":
            bull_score += 12; bear_score -= 8
            notes.append("⚠️ Bullish RSI divergence detected")
        elif rsi.divergence == "BEARISH_DIV":
            bear_score += 12; bull_score -= 8
            notes.append("⚠️ Bearish RSI divergence detected")

        # =================================================================
        # V2 CONTEXT BONUSES (up to +70 pts combined)
        # =================================================================
        if v2ctx is not None and v2_available and hasattr(v2ctx, 'squeeze'):
            # --- Squeeze bonus (+25 pts to dominant direction) ---
            if hasattr(v2ctx.squeeze, 'is_squeezed') and v2ctx.squeeze.is_squeezed:
                squeeze_days = getattr(v2ctx.squeeze, 'squeeze_days', 0)
                mom_dir = getattr(v2ctx.squeeze, 'momentum_direction', 'NEUTRAL')

                if mom_dir == 'UP':
                    v2_bull_bonus += 25
                    notes.append(f"🔥 SQUEEZE ACTIVE ({squeeze_days}d) — momentum UP → bull bonus +25")
                elif mom_dir == 'DOWN':
                    v2_bear_bonus += 25
                    notes.append(f"🔥 SQUEEZE ACTIVE ({squeeze_days}d) — momentum DOWN → bear bonus +25")
                else:
                    # Squeeze with no directional momentum — add to both (coiling)
                    v2_bull_bonus += 12
                    v2_bear_bonus += 12
                    notes.append(f"🔥 SQUEEZE ACTIVE ({squeeze_days}d) — neutral momentum, coiling")

            # --- Weekly trend alignment (+20 pts) ---
            if hasattr(v2ctx, 'weekly'):
                w = v2ctx.weekly
                weekly_trend = getattr(w, 'trend', 'NEUTRAL')
                supports_long = getattr(w, 'supports_long', False)
                supports_short = getattr(w, 'supports_short', False)

                if supports_long and bull_score > bear_score:
                    v2_bull_bonus += 20
                    notes.append(f"📈 Weekly {weekly_trend} supports long → bull bonus +20")
                elif supports_short and bear_score > bull_score:
                    v2_bear_bonus += 20
                    notes.append(f"📉 Weekly {weekly_trend} supports short → bear bonus +20")
                elif weekly_trend in ('UPTREND', 'STRONG_UPTREND') and bear_score > bull_score:
                    notes.append(f"⚠️ Shorting against weekly {weekly_trend} — caution")
                elif weekly_trend in ('DOWNTREND', 'STRONG_DOWNTREND') and bull_score > bear_score:
                    notes.append(f"⚠️ Going long against weekly {weekly_trend} — caution")

            # --- VP shape confirmation (+15 pts) ---
            if hasattr(v2ctx, 'vp') and v2ctx.vp is not None:
                shape = getattr(v2ctx.vp, 'profile_shape', 'normal')
                pos = getattr(v2ctx.vp, 'price_position', 'in_va')

                if shape == 'p-shape' and pos == 'below_va':
                    v2_bull_bonus += 15
                    notes.append("📊 VP p-shape at lows — accumulation → bull bonus +15")
                elif shape == 'd-shape' and pos == 'above_va':
                    v2_bear_bonus += 15
                    notes.append("📊 VP d-shape at highs — distribution → bear bonus +15")
                elif shape == 'b-shape':
                    notes.append("📊 VP b-shape — bimodal distribution, range-bound")

            # --- IV regime confidence adjustment ---
            if hasattr(v2ctx, 'iv') and v2ctx.iv is not None:
                iv_regime = getattr(v2ctx.iv, 'iv_regime', 'normal')
                if iv_regime == 'extreme':
                    notes.append("⚡ IV EXTREME — high premium, widen stops")
                elif iv_regime == 'elevated':
                    notes.append("📈 IV elevated — consider selling premium")

        # Apply V2 bonuses
        bull_score += v2_bull_bonus
        bear_score += v2_bear_bonus

        # =================================================================
        # DETERMINE SIGNAL STATE
        # =================================================================
        bull_score = max(0, min(100, bull_score))
        bear_score = max(0, min(100, bear_score))

        score_gap = abs(bull_score - bear_score)
        max_score = max(bull_score, bear_score)
        min_score = min(bull_score, bear_score)

        if max_score < 25 and min_score < 15:
            signal = SignalState.NEUTRAL
            confidence = 100 - max_score
            notes.append("Insufficient directional evidence")
        elif max_score < self.MODERATE_THRESHOLD and min_score >= 15:
            signal = SignalState.YELLOW
            confidence = 45 + (score_gap / 2) if bull_score != bear_score else 40
            lean = "bullish" if bull_score > bear_score else "bearish" if bear_score > bull_score else "no"
            notes.append(f"Mixed signals, {lean} lean — YELLOW")
        elif score_gap < self.MIN_SCORE_GAP:
            signal = SignalState.YELLOW
            confidence = 50 - (score_gap / self.MIN_SCORE_GAP * 25)
            notes.append(f"Mixed signals — gap only {score_gap:.1f} points")
        elif bull_score > bear_score and bull_score >= self.STRONG_THRESHOLD:
            signal = SignalState.LONG_SETUP
            confidence = min(95, bull_score - bear_score + 40)
            notes.append("✔ Long setup confirmed")
        elif bear_score > bull_score and bear_score >= self.STRONG_THRESHOLD:
            signal = SignalState.SHORT_SETUP
            confidence = min(95, bear_score - bull_score + 40)
            notes.append("✔ Short setup confirmed")
        elif bull_score > bear_score:
            signal = SignalState.YELLOW
            confidence = 40 + (score_gap / 2)
            notes.append("Leaning bullish but not confirmed — YELLOW")
        elif bear_score > bull_score:
            signal = SignalState.YELLOW
            confidence = 40 + (score_gap / 2)
            notes.append("Leaning bearish but not confirmed — YELLOW")
        else:
            signal = SignalState.NEUTRAL
            confidence = 50

        # IV confidence penalty
        if v2ctx is not None and v2_available and hasattr(v2ctx, 'iv') and v2ctx.iv is not None:
            iv_regime = getattr(v2ctx.iv, 'iv_regime', 'normal')
            if iv_regime == 'extreme':
                confidence *= 0.90  # 10% penalty
            elif iv_regime == 'elevated':
                confidence *= 0.95  # 5% penalty

        return bull_score, bear_score, signal, confidence, notes


# =============================================================================
# MULTI-TIMEFRAME AUCTION SCANNER V2
# =============================================================================

class MTFAuctionScanner:
    """
    Multi-Timeframe Non-Bias Auction Scanner V2.

    Scans 30min → 4hr timeframes. Each timeframe votes independently.
    Agreement = higher confidence. Disagreement = YELLOW (wait).

    V2: V2Context integration for weekly/squeeze/IV/VP shape bonuses.

    Usage:
        scanner = MTFAuctionScanner()

        # V1 compatible — pass raw DataFrame
        result = scanner.scan(df, symbol="META")
        print(scanner.print_report(result))

        # V2 enriched — pass V2Context for bonus scoring
        result = scanner.scan(df, symbol="META", v2_context=ctx)
    """

    def __init__(self, config: Optional[object] = None):
        """
        Args:
            config: Optional SwingTradeConfig (from scanner_config.py).
                    If None, uses built-in defaults (backward compatible).
        """
        self._config = config

        # Extract sub-configs if available
        vp_cfg = getattr(config, 'volume_profile', None) if config else None
        flow_cfg = getattr(config, 'flow', None) if config else None
        rsi_cfg = getattr(config, 'rsi', None) if config else None
        scoring_cfg = getattr(config, 'scoring', None) if config else None

        self.vp_engine = VolumeProfileEngine(config=vp_cfg)
        self.flow_engine = FlowControlEngine(config=flow_cfg)
        self.rsi_engine = RSIEngine(config=rsi_cfg)
        self.vwap_engine = VWAPEngine()
        self.scorer = SignalScorer(config=scoring_cfg, flow_config=flow_cfg)
        self.timeframes = [Timeframe.M30, Timeframe.H1, Timeframe.H2, Timeframe.H4]

    def scan(self,
             df: pd.DataFrame,
             symbol: str = "UNKNOWN",
             timeframes: Optional[List[Timeframe]] = None,
             v2_context: Optional[object] = None) -> ScanResult:
        """
        Run complete multi-timeframe scan.

        Args:
            df: OHLCV DataFrame (smallest available timeframe)
            symbol: Ticker symbol
            timeframes: List of Timeframe enums (default: all)
            v2_context: Optional V2Context for enrichment bonuses

        Returns:
            ScanResult with per-timeframe analysis + aggregate assessment
        """
        if timeframes is None:
            timeframes = self.timeframes

        # Normalize index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)

        df.columns = df.columns.str.lower()
        current_price = float(df['close'].iloc[-1])
        scan_time = df.index[-1]

        analyses = {}
        for tf in timeframes:
            tf_df = self._resample_to_timeframe(df, tf)
            if len(tf_df) >= 20:
                analysis = self._analyze_timeframe(tf_df, tf, current_price, v2_context)
                analyses[tf] = analysis
            else:
                analyses[tf] = self._create_no_data_analysis(tf, current_price, scan_time)

        result = self._aggregate_analyses(symbol, scan_time, analyses, v2_context)
        return result

    def scan_enriched(self,
                       df: pd.DataFrame,
                       symbol: str,
                       v2_context: object) -> Tuple[ScanResult, object]:
        """
        V2 enriched scan — returns (ScanResult, V2Context).

        Usage:
            from market_scanner_v2 import MarketScanner
            base = MarketScanner()
            ctx = base.build_v2_context("META")
            result, ctx = scanner.scan_enriched(df, "META", ctx)
        """
        result = self.scan(df, symbol, v2_context=v2_context)
        return result, v2_context

    def _resample_to_timeframe(self, df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        rule = f"{tf.minutes}min" if tf.minutes < 60 else f"{tf.minutes // 60}h"
        return df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

    def _analyze_timeframe(self,
                            df: pd.DataFrame,
                            tf: Timeframe,
                            current_price: float,
                            v2_context: Optional[object] = None) -> TimeframeAnalysis:
        """Analyze a single timeframe with optional V2 context"""
        vp = self.vp_engine.calculate(df)
        flow = self.flow_engine.calculate(df)
        rsi = self.rsi_engine.calculate(df)
        vwap = self.vwap_engine.calculate(df)

        price_vs_poc = (current_price - vp.poc) / vp.value_width if vp.value_width > 0 else 0
        price_vs_vah = current_price - vp.vah
        price_vs_val = current_price - vp.val

        if current_price > vp.vah:
            position = "ABOVE_VALUE"
        elif current_price < vp.val:
            position = "BELOW_VALUE"
        else:
            position = "IN_VALUE"

        # Score with V2 context
        bull_score, bear_score, signal, confidence, notes = self.scorer.score(
            current_price, vp, flow, rsi, vwap, v2_context
        )

        return TimeframeAnalysis(
            timeframe=tf, timestamp=df.index[-1], current_price=current_price,
            volume_profile=vp, flow=flow, rsi=rsi,
            price_vs_poc=price_vs_poc, price_vs_vah=price_vs_vah, price_vs_val=price_vs_val,
            position_in_value=position,
            bull_score=bull_score, bear_score=bear_score,
            signal=signal, confidence=confidence,
            vwap=vwap, notes=notes,
        )

    def _create_no_data_analysis(self, tf, current_price, timestamp):
        return TimeframeAnalysis(
            timeframe=tf, timestamp=timestamp, current_price=current_price,
            volume_profile=VolumeProfile(current_price, current_price, current_price, 0, 0),
            flow=FlowMetrics(0, 0, 0.5, 0.5, 0),
            rsi=RSIMetrics(50, 0, None),
            price_vs_poc=0, price_vs_vah=0, price_vs_val=0,
            position_in_value="NO_DATA",
            bull_score=0, bear_score=0,
            signal=SignalState.NO_DATA, confidence=0,
            vwap=VWAPMetrics(current_price, current_price, current_price, current_price, current_price, 0, 0),
            notes=["Insufficient data for analysis"],
        )

    def _aggregate_analyses(self, symbol, scan_time, analyses, v2_context=None):
        long_count = sum(1 for a in analyses.values() if a.signal == SignalState.LONG_SETUP)
        short_count = sum(1 for a in analyses.values() if a.signal == SignalState.SHORT_SETUP)
        yellow_count = sum(1 for a in analyses.values() if a.signal == SignalState.YELLOW)

        valid = [a for a in analyses.values() if a.signal != SignalState.NO_DATA]

        if not valid:
            return ScanResult(
                symbol=symbol, scan_time=scan_time, timeframe_analyses=analyses,
                dominant_signal=SignalState.NO_DATA, confluence_score=0,
                actionable=False, high_scenario_prob=0.33,
                low_scenario_prob=0.33, neutral_prob=0.34,
                summary="Insufficient data across all timeframes",
                v2_context=v2_context,
            )

        avg_bull = float(np.mean([a.bull_score for a in valid]))
        avg_bear = float(np.mean([a.bear_score for a in valid]))
        total_valid = len(valid)

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
        score_total = avg_bull + avg_bear + 1
        high_prob = avg_bull / score_total
        low_prob = avg_bear / score_total
        neutral_prob = max(0, 1 - high_prob - low_prob)

        prob_total = high_prob + low_prob + neutral_prob
        if prob_total > 0:
            high_prob /= prob_total
            low_prob /= prob_total
            neutral_prob /= prob_total

        actionable = (
            dominant in (SignalState.LONG_SETUP, SignalState.SHORT_SETUP) and
            confluence >= 50 and yellow_count < total_valid / 2
        )

        summary = self._generate_summary(
            dominant, confluence, long_count, short_count, yellow_count,
            avg_bull, avg_bear, high_prob, low_prob, actionable, valid,
            v2_context
        )

        return ScanResult(
            symbol=symbol, scan_time=scan_time, timeframe_analyses=analyses,
            dominant_signal=dominant, confluence_score=confluence,
            actionable=actionable,
            high_scenario_prob=high_prob, low_scenario_prob=low_prob,
            neutral_prob=neutral_prob, summary=summary,
            v2_context=v2_context,
        )

    def _generate_summary(self, dominant, confluence, long_count, short_count,
                           yellow_count, avg_bull, avg_bear, high_prob, low_prob,
                           actionable, valid_analyses, v2_context=None):
        lines = [
            f"Signal: {dominant.emoji} {dominant.value} ({confluence:.0f}% confluence)",
            f"Timeframe votes: {long_count}L / {short_count}S / {yellow_count}Y",
            f"Avg Scores: Bull {avg_bull:.1f} | Bear {avg_bear:.1f}",
            f"Scenario Odds: HIGH {high_prob:.0%} | LOW {low_prob:.0%}",
        ]

        # V2 context summary
        if v2_context is not None and v2_available and hasattr(v2_context, 'squeeze'):
            ctx_notes = []
            if hasattr(v2_context.squeeze, 'is_squeezed') and v2_context.squeeze.is_squeezed:
                ctx_notes.append(f"SQUEEZE({getattr(v2_context.squeeze, 'squeeze_days', '?')}d)")
            if hasattr(v2_context, 'weekly'):
                ctx_notes.append(f"Weekly:{getattr(v2_context.weekly, 'trend', '?')}")
            if hasattr(v2_context, 'iv') and v2_context.iv:
                ctx_notes.append(f"IV:{getattr(v2_context.iv, 'iv_regime', '?')}")
            if ctx_notes:
                lines.append(f"V2 Context: {' | '.join(ctx_notes)}")

        if actionable:
            lines.append("✅ ACTIONABLE SETUP")
        elif dominant == SignalState.YELLOW:
            lines.append("⚠️ WAIT FOR CLARITY")
        else:
            lines.append("⏸️ NO SETUP — STAND ASIDE")

        return "\n".join(lines)

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def print_report(self, result: ScanResult) -> str:
        lines = [
            "=" * 70,
            f"MTF AUCTION SCAN V2: {result.symbol}",
            f"Time: {result.scan_time}",
            "=" * 70, "",
            "TIMEFRAME BREAKDOWN:",
            "-" * 70,
        ]

        for tf in self.timeframes:
            if tf in result.timeframe_analyses:
                a = result.timeframe_analyses[tf]
                lines.append(
                    f"\n{tf.label.upper():>8} | {a.signal.emoji} {a.signal.value:12} | "
                    f"Bull: {a.bull_score:5.1f} | Bear: {a.bear_score:5.1f} | "
                    f"Conf: {a.confidence:5.1f}%"
                )
                lines.append(
                    f"         | Position: {a.position_in_value:12} | "
                    f"RSI: {a.rsi.value:5.1f} ({a.rsi.zone}) | "
                    f"Flow: {a.flow.flow_imbalance:+.2f}"
                )
                vp = a.volume_profile
                lines.append(f"         | VAH: {vp.vah:.2f} | POC: {vp.poc:.2f} | VAL: {vp.val:.2f}")

                if a.vwap:
                    lines.append(
                        f"         | VWAP: {a.vwap.vwap:.2f} ({a.vwap.zone}) | "
                        f"Dev: {a.vwap.deviation_pct:+.2f}%"
                    )

        lines.extend(["", "=" * 70, "AGGREGATE ASSESSMENT:", "-" * 70, result.summary, "=" * 70])
        return "\n".join(lines)


# =============================================================================
# DEMO DATA + CLI
# =============================================================================

def generate_demo_data(days: int = 5, interval_minutes: int = 5) -> pd.DataFrame:
    """Generate realistic demo OHLCV data for testing"""
    np.random.seed(42)
    periods = days * 24 * 60 // interval_minutes
    price = 450.0
    data = []
    timestamp = datetime.now() - timedelta(days=days)

    for i in range(periods):
        trend = 0.0001 * np.sin(i / 100)
        noise = np.random.randn() * 0.002
        hour = (i * interval_minutes // 60) % 24
        volume_mult = 1.5 if 9 <= hour <= 16 else 0.5
        returns = trend + noise

        open_price = price
        close_price = price * (1 + returns)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.001)
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.001)
        volume = int(np.random.exponential(100000) * volume_mult)

        data.append({
            'timestamp': timestamp,
            'open': open_price, 'high': high_price,
            'low': low_price, 'close': close_price,
            'volume': volume,
        })
        price = close_price
        timestamp += timedelta(minutes=interval_minutes)

    df = pd.DataFrame(data).set_index('timestamp')
    return df


if __name__ == "__main__":
    print("Generating demo data...")
    df = generate_demo_data(days=10, interval_minutes=5)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}\n")

    scanner = MTFAuctionScanner()
    result = scanner.scan(df, symbol="DEMO")
    print(scanner.print_report(result))

    print(f"\nScenario Probabilities:")
    print(f"  HIGH scenario: {result.high_scenario_prob:.1%}")
    print(f"  LOW scenario:  {result.low_scenario_prob:.1%}")
    print(f"  NEUTRAL:       {result.neutral_prob:.1%}")
