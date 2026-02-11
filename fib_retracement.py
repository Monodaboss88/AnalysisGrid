"""
Fibonacci Retracement Detection
================================
Proper swing-point detection with pivot validation,
VP+Fib confluence scoring, and trade journal integration.

V1 PROBLEMS FIXED:
- Raw max/min replaced with pivot-based swing detection
- Synthetic percentile fallback eliminated
- Fabricated 5% minimum range removed
- Legacy field bug fixed (fib_236 always mapped to bearish)
- Added: swing quality scoring, multi-swing support, confluence grading

Author: Rob's Trading Systems (Strategic Edge Flow)
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SwingPoint:
    """A validated swing high or low"""
    price: float
    date: str                          # ISO format
    bar_index: int                     # Position in DataFrame
    swing_type: str                    # "HIGH" or "LOW"
    strength: int                      # Number of bars on each side that confirm it (1-10)
    volume_at_swing: float = 0.0       # Volume on the swing bar
    is_primary: bool = False           # True = the selected swing for fib calc


@dataclass
class FibLevel:
    """A single Fibonacci level with context"""
    ratio: float                       # 0.236, 0.382, 0.500, 0.618, 0.786
    price: float
    label: str                         # "23.6%", "38.2%", etc.
    is_golden: bool = False            # True for 50%-61.8% zone
    vp_confluence: Optional[str] = None  # "VAH", "POC", "VAL" if within tolerance
    confluence_distance_pct: float = 0.0  # How close to VP level (0 = exact)


@dataclass
class FibResult:
    """Complete Fibonacci analysis for a symbol"""
    symbol: str
    analysis_time: str

    # Swing points
    swing_high: SwingPoint
    swing_low: SwingPoint
    fib_range: float                   # swing_high.price - swing_low.price
    fib_range_pct: float               # As % of swing_high

    # Trend
    trend: str                         # "UPTREND" or "DOWNTREND"
    trend_reason: str

    # Active fib set (trend-appropriate)
    levels: List[FibLevel]             # The 5 fib levels to use

    # Both sets for reference
    bull_levels: List[FibLevel]        # Support levels (uptrend pullback entries)
    bear_levels: List[FibLevel]        # Resistance levels (downtrend bounce entries)

    # Price position
    current_price: float
    price_position: str                # Human-readable position description
    price_zone: str                    # "ABOVE_SWING", "GOLDEN_ZONE", "BELOW_SWING", etc.

    # Confluence
    confluences: List[str]             # VP+Fib confluences found
    confluence_count: int = 0

    # Quality
    swing_quality: str = ""            # "STRONG", "MODERATE", "WEAK"
    swing_quality_score: int = 0       # 0-100

    # All detected swing points for reference
    all_swing_highs: List[SwingPoint] = field(default_factory=list)
    all_swing_lows: List[SwingPoint] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize for API response and trade journal"""
        active = {f"fib_{l.label.replace('.','').replace('%','')}": l.price for l in self.levels}
        bull = {f"bull_fib_{l.label.replace('.','').replace('%','')}": l.price for l in self.bull_levels}
        bear = {f"bear_fib_{l.label.replace('.','').replace('%','')}": l.price for l in self.bear_levels}

        return {
            "symbol": self.symbol,
            "swing_high": self.swing_high.price,
            "swing_high_date": self.swing_high.date,
            "swing_low": self.swing_low.price,
            "swing_low_date": self.swing_low.date,
            "fib_range": self.fib_range,
            "fib_range_pct": round(self.fib_range_pct, 2),
            "trend": self.trend,
            "current_price": self.current_price,
            "price_position": self.price_position,
            "price_zone": self.price_zone,
            "swing_quality": self.swing_quality,
            "swing_quality_score": self.swing_quality_score,
            "confluences": self.confluences,
            "confluence_count": self.confluence_count,
            # Active levels (trend-appropriate)
            **active,
            # Both sets
            **bull,
            **bear,
            # Legacy compat — FIXED: uses trend-appropriate set
            "fib_236": active.get("fib_236", 0),
            "fib_382": active.get("fib_382", 0),
            "fib_500": active.get("fib_500", 0),
            "fib_618": active.get("fib_618", 0),
            "fib_786": active.get("fib_786", 0),
            "lookback_days": 0,  # Set by caller
        }

    def to_journal_fields(self) -> Dict:
        """Fields to store with a trade journal entry"""
        return {
            "fib_swing_high": self.swing_high.price,
            "fib_swing_low": self.swing_low.price,
            "fib_trend": self.trend,
            "fib_236": next((l.price for l in self.levels if l.ratio == 0.236), 0),
            "fib_382": next((l.price for l in self.levels if l.ratio == 0.382), 0),
            "fib_500": next((l.price for l in self.levels if l.ratio == 0.500), 0),
            "fib_618": next((l.price for l in self.levels if l.ratio == 0.618), 0),
            "fib_786": next((l.price for l in self.levels if l.ratio == 0.786), 0),
            "fib_position": self.price_position,
            "fib_zone": self.price_zone,
            "fib_confluence": "; ".join(self.confluences) if self.confluences else "",
            "fib_quality": self.swing_quality,
        }


# =============================================================================
# SWING POINT DETECTION
# =============================================================================

FIB_RATIOS = [
    (0.236, "23.6%", False),
    (0.382, "38.2%", False),
    (0.500, "50%", True),      # Golden zone start
    (0.618, "61.8%", True),    # Golden ratio
    (0.786, "78.6%", False),
]


def detect_swing_points(df: pd.DataFrame,
                         min_strength: int = 3,
                         max_points: int = 10) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect swing highs and lows using pivot-point validation.

    A swing high is a bar whose high is higher than `min_strength` bars
    on BOTH sides. Same logic inverted for swing lows.

    Args:
        df: OHLCV DataFrame with datetime index
        min_strength: Minimum bars on each side to confirm a pivot (default 3)
        max_points: Maximum swing points to return per side

    Returns:
        (swing_highs, swing_lows) — sorted by date descending (most recent first)
    """
    if len(df) < min_strength * 2 + 1:
        return [], []

    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

    # Get dates
    if isinstance(df.index, pd.DatetimeIndex):
        dates = [d.isoformat() for d in df.index]
    else:
        dates = [str(d) for d in df.index]

    swing_highs = []
    swing_lows = []

    # Scan for pivots at multiple strength levels (3 through 7)
    for strength in range(min_strength, min(8, len(df) // 2)):
        for i in range(strength, len(df) - strength):
            # Swing High: bar's high > all bars within `strength` on both sides
            is_high = True
            for j in range(1, strength + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break

            if is_high:
                # Check if we already have a swing at this bar or very close
                already_found = any(abs(sh.bar_index - i) <= 1 for sh in swing_highs)
                if not already_found:
                    swing_highs.append(SwingPoint(
                        price=float(highs[i]),
                        date=dates[i],
                        bar_index=i,
                        swing_type="HIGH",
                        strength=strength,
                        volume_at_swing=float(volumes[i]),
                    ))

            # Swing Low: bar's low < all bars within `strength` on both sides
            is_low = True
            for j in range(1, strength + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break

            if is_low:
                already_found = any(abs(sl.bar_index - i) <= 1 for sl in swing_lows)
                if not already_found:
                    swing_lows.append(SwingPoint(
                        price=float(lows[i]),
                        date=dates[i],
                        bar_index=i,
                        swing_type="LOW",
                        strength=strength,
                        volume_at_swing=float(volumes[i]),
                    ))

    # For duplicates at nearby bars, keep the one with higher strength
    swing_highs = _deduplicate_swings(swing_highs, tolerance_bars=2)
    swing_lows = _deduplicate_swings(swing_lows, tolerance_bars=2)

    # Sort by date descending (most recent first)
    swing_highs.sort(key=lambda s: s.bar_index, reverse=True)
    swing_lows.sort(key=lambda s: s.bar_index, reverse=True)

    return swing_highs[:max_points], swing_lows[:max_points]


def _deduplicate_swings(swings: List[SwingPoint], tolerance_bars: int = 2) -> List[SwingPoint]:
    """Keep the strongest swing when multiple are found at nearby bars"""
    if not swings:
        return []

    swings.sort(key=lambda s: s.bar_index)
    deduped = [swings[0]]

    for s in swings[1:]:
        if abs(s.bar_index - deduped[-1].bar_index) <= tolerance_bars:
            # Keep the one with higher strength, or higher price for highs / lower for lows
            if s.strength > deduped[-1].strength:
                deduped[-1] = s
            elif s.strength == deduped[-1].strength:
                if s.swing_type == "HIGH" and s.price > deduped[-1].price:
                    deduped[-1] = s
                elif s.swing_type == "LOW" and s.price < deduped[-1].price:
                    deduped[-1] = s
        else:
            deduped.append(s)

    return deduped


def select_primary_swings(swing_highs: List[SwingPoint],
                           swing_lows: List[SwingPoint],
                           df: pd.DataFrame) -> Tuple[SwingPoint, SwingPoint, str, str]:
    """
    Select the best swing high and low pair for fib calculation.

    Rules:
    1. The swing high must be HIGHER than the swing low (obviously)
    2. Prefer higher-strength pivots
    3. The pair should represent the most recent significant move
    4. Trend determined by which extreme is more recent

    Returns:
        (swing_high, swing_low, trend, trend_reason)
    """
    if not swing_highs or not swing_lows:
        # Fallback: use raw max/min (last resort)
        max_idx = df['high'].idxmax()
        min_idx = df['low'].idxmin()
        max_pos = list(df.index).index(max_idx)
        min_pos = list(df.index).index(min_idx)

        dates = [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in df.index]
        vol = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

        sh = SwingPoint(
            price=float(df['high'].max()), date=dates[max_pos],
            bar_index=max_pos, swing_type="HIGH", strength=0,
            volume_at_swing=float(vol[max_pos])
        )
        sl = SwingPoint(
            price=float(df['low'].min()), date=dates[min_pos],
            bar_index=min_pos, swing_type="LOW", strength=0,
            volume_at_swing=float(vol[min_pos])
        )

        trend = "UPTREND" if max_pos > min_pos else "DOWNTREND"
        reason = "Fallback max/min (no pivots found)"
        return sh, sl, trend, reason

    # Strategy: find the highest-quality pair
    best_pair = None
    best_score = -1

    for sh in swing_highs[:5]:  # Top 5 most recent
        for sl in swing_lows[:5]:
            if sh.price <= sl.price:
                continue  # Invalid pair

            # Score this pair
            score = 0

            # Strength bonus (higher strength = more validated pivot)
            score += (sh.strength + sl.strength) * 10

            # Range quality: prefer meaningful ranges (2-15% of price)
            range_pct = (sh.price - sl.price) / sh.price * 100
            if 2 <= range_pct <= 15:
                score += 30  # Sweet spot
            elif 1 <= range_pct <= 20:
                score += 15  # Acceptable
            else:
                score += 0   # Too tight or too wide

            # Recency bonus: prefer more recent swings
            recency = (sh.bar_index + sl.bar_index) / 2
            max_bar = len(df) - 1
            if max_bar > 0:
                score += int((recency / max_bar) * 20)

            # Volume bonus: high volume at swing points = more significant
            avg_vol = df['volume'].mean() if 'volume' in df.columns else 1
            if avg_vol > 0:
                vol_ratio = (sh.volume_at_swing + sl.volume_at_swing) / (2 * avg_vol)
                if vol_ratio > 1.5:
                    score += 15
                elif vol_ratio > 1.0:
                    score += 8

            if score > best_score:
                best_score = score
                best_pair = (sh, sl)

    if best_pair is None:
        # This shouldn't happen but safety fallback
        best_pair = (swing_highs[0], swing_lows[0])
        if best_pair[0].price <= best_pair[1].price:
            # Swap to the widest valid pair
            for sh in swing_highs:
                for sl in swing_lows:
                    if sh.price > sl.price:
                        best_pair = (sh, sl)
                        break
                if best_pair[0].price > best_pair[1].price:
                    break

    sh, sl = best_pair
    sh.is_primary = True
    sl.is_primary = True

    # Determine trend
    if sh.bar_index > sl.bar_index:
        trend = "UPTREND"
        reason = f"Swing high ({sh.date[:10]}) more recent than swing low ({sl.date[:10]})"
    else:
        trend = "DOWNTREND"
        reason = f"Swing low ({sl.date[:10]}) more recent than swing high ({sh.date[:10]})"

    return sh, sl, trend, reason


# =============================================================================
# FIB LEVEL CALCULATION
# =============================================================================

def calculate_fib_levels(swing_high: float,
                          swing_low: float,
                          direction: str = "BULL") -> List[FibLevel]:
    """
    Calculate Fibonacci retracement levels.

    BULL direction: levels measured UP from swing_low (pullback support in uptrend)
    BEAR direction: levels measured DOWN from swing_high (bounce resistance in downtrend)
    """
    fib_range = swing_high - swing_low
    levels = []

    for ratio, label, is_golden in FIB_RATIOS:
        if direction == "BULL":
            # In uptrend, fibs are support levels for pullbacks
            # Fib 23.6% = shallow pullback (high price)
            # Fib 78.6% = deep pullback (low price)
            price = swing_high - (fib_range * ratio)
        else:
            # In downtrend, fibs are resistance levels for bounces
            # Fib 23.6% = shallow bounce (low price)
            # Fib 78.6% = deep bounce (high price)
            price = swing_low + (fib_range * ratio)

        levels.append(FibLevel(
            ratio=ratio,
            price=round(price, 2),
            label=label,
            is_golden=is_golden,
        ))

    return levels


def check_vp_confluence(fib_levels: List[FibLevel],
                         vah: float = 0, poc: float = 0, val: float = 0,
                         tolerance: float = 0.015) -> Tuple[List[FibLevel], List[str]]:
    """
    Check if VP levels (VAH/POC/VAL) align with Fib levels.

    Args:
        fib_levels: List of FibLevel objects
        vah, poc, val: Volume Profile levels
        tolerance: Max % distance to count as confluence (default 1.5%)

    Returns:
        (updated_levels, confluence_descriptions)
    """
    confluences = []
    vp_levels = {"VAH": vah, "POC": poc, "VAL": val}

    for level in fib_levels:
        for vp_name, vp_price in vp_levels.items():
            if vp_price <= 0:
                continue

            distance_pct = abs(level.price - vp_price) / vp_price
            if distance_pct < tolerance:
                level.vp_confluence = vp_name
                level.confluence_distance_pct = round(distance_pct * 100, 2)
                quality = "EXACT" if distance_pct < 0.005 else "STRONG" if distance_pct < 0.01 else "NEAR"
                confluences.append(
                    f"{vp_name} ≈ Fib {level.label} at ${vp_price:.2f} ({quality})"
                )

    return fib_levels, confluences


# =============================================================================
# PRICE POSITION CLASSIFICATION
# =============================================================================

def classify_price_position(current_price: float,
                             levels: List[FibLevel],
                             swing_high: float,
                             swing_low: float,
                             trend: str) -> Tuple[str, str]:
    """
    Classify where current price sits relative to fib levels.

    Returns:
        (human_description, zone_code)
    """
    # Get level prices in order (high to low for bull, low to high for bear)
    level_map = {l.ratio: l.price for l in levels}

    if trend == "UPTREND":
        # Bull fibs: levels are support (measured down from high)
        if current_price >= swing_high:
            return "Above swing high (extended — watch for reversal)", "ABOVE_SWING"
        elif current_price >= level_map.get(0.236, swing_high):
            return "Above Fib 23.6% (strong uptrend, shallow pullback)", "ABOVE_236"
        elif current_price >= level_map.get(0.382, swing_high):
            return "Fib 23.6%-38.2% (healthy pullback zone)", "PULLBACK_SHALLOW"
        elif current_price >= level_map.get(0.500, swing_high):
            return "Fib 38.2%-50% (pullback entry zone)", "PULLBACK_ENTRY"
        elif current_price >= level_map.get(0.618, swing_high):
            return "Fib 50%-61.8% GOLDEN ZONE (best long entry)", "GOLDEN_ZONE"
        elif current_price >= level_map.get(0.786, swing_low):
            return "Fib 61.8%-78.6% (deep retracement — caution)", "DEEP_RETRACE"
        elif current_price >= swing_low:
            return "Below Fib 78.6% (trend likely broken)", "TREND_BROKEN"
        else:
            return "Below swing low (new low — trend reversed)", "BELOW_SWING"
    else:
        # Bear fibs: levels are resistance (measured up from low)
        if current_price <= swing_low:
            return "Below swing low (extended — watch for bounce)", "BELOW_SWING"
        elif current_price <= level_map.get(0.236, swing_low):
            return "Below Fib 23.6% (strong downtrend, shallow bounce)", "BELOW_236"
        elif current_price <= level_map.get(0.382, swing_low):
            return "Fib 23.6%-38.2% (healthy bounce zone)", "BOUNCE_SHALLOW"
        elif current_price <= level_map.get(0.500, swing_low):
            return "Fib 38.2%-50% (bounce entry zone)", "BOUNCE_ENTRY"
        elif current_price <= level_map.get(0.618, swing_low):
            return "Fib 50%-61.8% GOLDEN ZONE (best short entry)", "GOLDEN_ZONE"
        elif current_price <= level_map.get(0.786, swing_high):
            return "Fib 61.8%-78.6% (deep bounce — caution)", "DEEP_BOUNCE"
        elif current_price <= swing_high:
            return "Above Fib 78.6% (downtrend likely broken)", "TREND_BROKEN"
        else:
            return "Above swing high (new high — trend reversed)", "ABOVE_SWING"


# =============================================================================
# SWING QUALITY SCORING
# =============================================================================

def score_swing_quality(swing_high: SwingPoint,
                         swing_low: SwingPoint,
                         df: pd.DataFrame) -> Tuple[str, int]:
    """
    Score the quality of the selected swing pair.

    Factors:
    - Pivot strength (higher = more validated)
    - Range relative to price (2-15% sweet spot)
    - Volume at swing points vs average
    - Separation between swings (not too close, not too far)

    Returns:
        (grade_str, score_0_100)
    """
    score = 0

    # 1. Pivot strength (0-30 pts)
    avg_strength = (swing_high.strength + swing_low.strength) / 2
    if avg_strength >= 5:
        score += 30
    elif avg_strength >= 4:
        score += 25
    elif avg_strength >= 3:
        score += 20
    elif avg_strength >= 2:
        score += 12
    elif avg_strength >= 1:
        score += 5
    # 0 strength = fallback (raw max/min), no points

    # 2. Range quality (0-30 pts)
    range_pct = (swing_high.price - swing_low.price) / swing_high.price * 100
    if 3 <= range_pct <= 12:
        score += 30   # Sweet spot for swing trading
    elif 2 <= range_pct <= 15:
        score += 20   # Acceptable
    elif 1 <= range_pct <= 20:
        score += 10   # Marginal
    # else: 0 pts — too tight or too wide

    # 3. Volume at swings (0-20 pts)
    if 'volume' in df.columns and df['volume'].mean() > 0:
        avg_vol = df['volume'].mean()
        high_vol_ratio = swing_high.volume_at_swing / avg_vol
        low_vol_ratio = swing_low.volume_at_swing / avg_vol

        if high_vol_ratio > 1.5 and low_vol_ratio > 1.5:
            score += 20  # Both swings on high volume = very significant
        elif high_vol_ratio > 1.2 or low_vol_ratio > 1.2:
            score += 12  # At least one swing has above-avg volume
        elif high_vol_ratio > 0.8 and low_vol_ratio > 0.8:
            score += 5   # Normal volume
        # Low volume swings = less reliable

    # 4. Swing separation (0-20 pts)
    total_bars = len(df)
    separation = abs(swing_high.bar_index - swing_low.bar_index)
    sep_ratio = separation / total_bars if total_bars > 0 else 0

    if 0.2 <= sep_ratio <= 0.8:
        score += 20   # Good separation
    elif 0.1 <= sep_ratio <= 0.9:
        score += 10   # Acceptable
    else:
        score += 3    # Too bunched or too spread out

    # Grade
    if score >= 80:
        grade = "STRONG"
    elif score >= 55:
        grade = "MODERATE"
    else:
        grade = "WEAK"

    return grade, min(100, score)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_fibs(df: pd.DataFrame,
                  symbol: str = "UNKNOWN",
                  vah: float = 0, poc: float = 0, val: float = 0,
                  min_pivot_strength: int = 3,
                  current_price: Optional[float] = None) -> Optional[FibResult]:
    """
    Complete Fibonacci retracement analysis.

    Args:
        df: Daily OHLCV DataFrame with datetime index
        symbol: Ticker symbol
        vah, poc, val: Volume Profile levels for confluence detection
        min_pivot_strength: Minimum bars on each side for swing validation
        current_price: Override for current price (uses last close if None)

    Returns:
        FibResult with all levels, position, confluences, and quality
        None if insufficient data
    """
    if df is None or len(df) < 10:
        return None

    # Normalize
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)

    df.columns = df.columns.str.lower()
    if current_price is None:
        current_price = float(df['close'].iloc[-1])

    # 1. Detect swing points
    swing_highs, swing_lows = detect_swing_points(df, min_strength=min_pivot_strength)

    # 2. Select primary pair
    sh, sl, trend, trend_reason = select_primary_swings(swing_highs, swing_lows, df)

    # Validate range
    fib_range = sh.price - sl.price
    if fib_range <= 0:
        return None

    fib_range_pct = (fib_range / sh.price) * 100

    # 3. Calculate both fib sets
    bull_levels = calculate_fib_levels(sh.price, sl.price, "BULL")
    bear_levels = calculate_fib_levels(sh.price, sl.price, "BEAR")

    # Active set based on trend
    active_levels = bull_levels if trend == "UPTREND" else bear_levels

    # 4. VP confluence check
    active_levels, confluences = check_vp_confluence(active_levels, vah, poc, val)
    # Also check the other set
    other_levels = bear_levels if trend == "UPTREND" else bull_levels
    other_levels, other_confluences = check_vp_confluence(other_levels, vah, poc, val)

    # 5. Price position
    position_desc, zone_code = classify_price_position(
        current_price, active_levels, sh.price, sl.price, trend
    )

    # 6. Swing quality
    quality_grade, quality_score = score_swing_quality(sh, sl, df)

    return FibResult(
        symbol=symbol,
        analysis_time=datetime.now().isoformat(),
        swing_high=sh,
        swing_low=sl,
        fib_range=round(fib_range, 2),
        fib_range_pct=round(fib_range_pct, 2),
        trend=trend,
        trend_reason=trend_reason,
        levels=active_levels,
        bull_levels=bull_levels,
        bear_levels=bear_levels,
        current_price=current_price,
        price_position=position_desc,
        price_zone=zone_code,
        confluences=confluences,
        confluence_count=len(confluences),
        swing_quality=quality_grade,
        swing_quality_score=quality_score,
        all_swing_highs=swing_highs,
        all_swing_lows=swing_lows,
    )


# =============================================================================
# FORMATTING
# =============================================================================

def format_fib_analysis(result: FibResult) -> str:
    """Human-readable fib analysis output"""
    lines = []

    lines.append("=" * 60)
    lines.append(f"📐 FIBONACCI RETRACEMENT: {result.symbol}")
    lines.append(f"   Trend: {result.trend} | Quality: {result.swing_quality} ({result.swing_quality_score}/100)")
    lines.append("=" * 60)

    lines.append(f"\n📍 SWING POINTS:")
    lines.append(f"   High: ${result.swing_high.price:.2f} ({result.swing_high.date[:10]}) strength={result.swing_high.strength}")
    lines.append(f"   Low:  ${result.swing_low.price:.2f} ({result.swing_low.date[:10]}) strength={result.swing_low.strength}")
    lines.append(f"   Range: ${result.fib_range:.2f} ({result.fib_range_pct:.1f}%)")

    lines.append(f"\n📏 ACTIVE FIB LEVELS ({result.trend}):")
    for level in result.levels:
        golden = " ⭐ GOLDEN" if level.is_golden else ""
        conf = f" 🎯 {level.vp_confluence}" if level.vp_confluence else ""
        lines.append(f"   Fib {level.label:>5}: ${level.price:.2f}{golden}{conf}")

    lines.append(f"\n📍 PRICE: ${result.current_price:.2f}")
    lines.append(f"   Position: {result.price_position}")

    if result.confluences:
        lines.append(f"\n🎯 VP+FIB CONFLUENCES ({result.confluence_count}):")
        for c in result.confluences:
            lines.append(f"   • {c}")

    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  FIBONACCI RETRACEMENT DETECTION")
    print("=" * 60)

    try:
        import yfinance as yf

        test_symbols = ["META", "AAPL", "NVDA", "TSLA"]

        for symbol in test_symbols:
            print(f"\n📊 Analyzing {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")

            if df is not None and len(df) > 10:
                result = analyze_fibs(df, symbol)
                if result:
                    print(format_fib_analysis(result))
                else:
                    print(f"  ⚠️ Insufficient data for {symbol}")
            else:
                print(f"  ⚠️ No data for {symbol}")

    except ImportError:
        print("Install yfinance for live testing: pip install yfinance")

        # Demo with synthetic data
        print("\n📊 Running synthetic demo...")
        dates = pd.date_range("2025-01-01", periods=60, freq="B")
        np.random.seed(42)

        prices = [100]
        for _ in range(59):
            prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))

        demo_df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'volume': [np.random.randint(1000000, 5000000) for _ in prices],
        }, index=dates)

        result = analyze_fibs(demo_df, "DEMO", vah=108, poc=104, val=100)
        if result:
            print(format_fib_analysis(result))
