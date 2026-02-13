"""
Absorption Detector
===================
Detects passive limit order walls absorbing aggressive flow.

While 3HER detects THAT a move is dying, the Absorption Detector
identifies WHERE and WHY — a large passive participant is sitting
at a specific price level and eating every order that hits it.

Key Signature of Absorption:
1. Repeated touches of the same price level (tight tolerance)
2. HIGH volume at that level (not declining like exhaustion)
3. Minimal price displacement despite heavy trading
4. Delta imbalance: aggressive side losing despite volume
5. Time clustering: multiple tests in a short window

Absorption Types:
- CEILING: Passive seller absorbing buyers (resistance)
- FLOOR: Passive buyer absorbing sellers (support)
- PINNING: Absorption on both sides (options-related or balancing)

Integration:
- Pairs with 3HER: exhaustion + absorption = highest conviction reversal
- Consumes same data feeds (30m/15m OHLCV from Finnhub)
- Reads volume profile levels from MTF scanner
- Independent scoring (0-100) with own classification

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
# ENUMS
# =============================================================================

class AbsorptionType(Enum):
    """Type of absorption detected"""
    CEILING = "CEILING"          # Passive seller absorbing buyers
    FLOOR = "FLOOR"              # Passive buyer absorbing sellers
    PINNING = "PINNING"          # Absorption on both sides
    NONE = "NONE"                # No absorption detected
    
    @property
    def emoji(self) -> str:
        return {
            "CEILING": "🧱",
            "FLOOR": "🛡️",
            "PINNING": "📌",
            "NONE": "⬜"
        }[self.value]
    
    @property
    def trade_implication(self) -> str:
        return {
            "CEILING": "Passive seller wall — fade longs / enter short on confirmation",
            "FLOOR": "Passive buyer wall — fade shorts / enter long on confirmation",
            "PINNING": "Both sides absorbed — range-bound, wait for breakout",
            "NONE": "No absorption — price free to move"
        }[self.value]


class AbsorptionStrength(Enum):
    """How strong the absorption wall is"""
    NONE = "NONE"
    WEAK = "WEAK"                # 1-2 touches, moderate volume
    MODERATE = "MODERATE"        # 3-4 touches, elevated volume  
    STRONG = "STRONG"            # 5+ touches, high volume, clear rejection
    INSTITUTIONAL = "INSTITUTIONAL"  # Dominant wall, extreme volume clustering
    
    @property
    def tradeable(self) -> bool:
        return self in [AbsorptionStrength.STRONG, AbsorptionStrength.INSTITUTIONAL]
    
    @property
    def score_range(self) -> Tuple[int, int]:
        return {
            "NONE": (0, 24),
            "WEAK": (25, 44),
            "MODERATE": (45, 64),
            "STRONG": (65, 84),
            "INSTITUTIONAL": (85, 100)
        }[self.value]


class WallStatus(Enum):
    """Current status of the absorption wall"""
    HOLDING = "HOLDING"          # Wall intact, absorbing
    WEAKENING = "WEAKENING"      # Wall showing cracks
    BROKEN = "BROKEN"            # Price broke through
    DEFENDED = "DEFENDED"         # Price retreated from wall


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PriceLevel:
    """A specific price level being monitored for absorption"""
    price: float                     # The exact price level
    tolerance: float                 # ± range for "touching" this level
    
    # Touch analysis
    touch_count: int = 0             # Times price reached this level
    touch_timestamps: List[datetime] = field(default_factory=list)
    time_at_level_bars: int = 0      # Candles spent at/near this level
    
    # Volume analysis
    total_volume_at_level: float = 0     # Cumulative volume when at level
    avg_volume_per_touch: float = 0      # Average volume per touch
    rvol_at_level: float = 0             # RVOL when at this level vs overall
    max_single_bar_volume: float = 0     # Largest volume bar at level
    
    # Price action
    max_penetration: float = 0           # Deepest price went past level
    avg_rejection_size: float = 0        # Avg distance price bounced back
    close_beyond_count: int = 0          # Candles that CLOSED beyond level
    wick_beyond_count: int = 0           # Candles that wicked beyond but closed inside
    
    # Delta analysis
    cumulative_delta_at_level: float = 0     # Net delta when at level
    aggressive_side_volume: float = 0        # Volume from the aggressor
    passive_side_volume: float = 0           # Volume from the absorber (winner)
    delta_imbalance: float = 0               # How much the absorber is winning


@dataclass
class AbsorptionZone:
    """A detected absorption zone (may span a small price range)"""
    center_price: float              # Center of absorption zone
    upper_bound: float               # Top of zone
    lower_bound: float               # Bottom of zone
    zone_width: float                # Width in price
    zone_width_pct: float            # Width as % of price
    
    absorption_type: AbsorptionType
    strength: AbsorptionStrength
    status: WallStatus
    
    # Aggregated metrics from constituent price levels
    total_touches: int
    total_volume: float
    rvol_ratio: float                # Volume at zone vs avg volume
    delta_imbalance: float           # Net delta bias
    time_spent_bars: int             # Candles at this zone
    
    # Scoring
    score: int                       # 0-100
    
    # Proximity to known levels
    near_vah: bool = False
    near_val: bool = False
    near_poc: bool = False
    near_vwap: bool = False
    near_fib: bool = False
    
    notes: List[str] = field(default_factory=list)


@dataclass
class AbsorptionResult:
    """Complete absorption analysis result"""
    symbol: str
    scan_time: datetime
    current_price: float
    
    # Detected zones (sorted by score)
    zones: List[AbsorptionZone]
    
    # Primary zone (highest scored)
    primary_zone: Optional[AbsorptionZone]
    
    # Overall assessment
    absorption_active: bool              # Any tradeable absorption detected
    dominant_type: AbsorptionType        # What's the main story
    
    # Dimension scores (0-25 each, total 0-100)
    touch_frequency_score: int           # How many times level was tested
    volume_concentration_score: int      # Volume clustering at level
    displacement_failure_score: int      # Price couldn't move through
    delta_absorption_score: int          # Passive side winning the delta war
    
    # Integration with 3HER
    supports_exhaustion: bool            # Does absorption explain a 3HER signal
    exhaustion_confluence_level: Optional[float]  # Price where both agree
    
    notes: List[str] = field(default_factory=list)
    
    @property
    def total_score(self) -> int:
        return (self.touch_frequency_score + self.volume_concentration_score +
                self.displacement_failure_score + self.delta_absorption_score)
    
    @property
    def score_label(self) -> str:
        s = self.total_score
        if s >= 85:
            return f"A+ | {s}/100"
        elif s >= 65:
            return f"A | {s}/100"
        elif s >= 45:
            return f"B | {s}/100"
        else:
            return f"C | {s}/100"


# =============================================================================
# ABSORPTION DETECTOR ENGINE
# =============================================================================

class AbsorptionDetector:
    """
    Detects passive absorption walls at specific price levels.
    
    Works by:
    1. Identifying candidate price levels where price clusters
    2. Analyzing volume behavior at each level
    3. Measuring price displacement failure (effort vs result)
    4. Computing delta imbalance to confirm passive dominance
    5. Scoring and classifying absorption zones
    
    Data Requirements:
    - 5-min or 15-min OHLCV candles (higher granularity = better detection)
    - Volume profile levels (POC, VAH, VAL) for context
    - Optional: Fib levels, VWAP for confluence
    
    Usage:
        detector = AbsorptionDetector()
        result = detector.analyze(
            df=fifteen_min_candles,
            symbol="NVDA",
            vah=142.50, poc=140.80, val=139.20,
            vwap=141.00
        )
    """
    
    # Level detection parameters
    PRICE_BIN_COUNT = 80             # Number of price bins for clustering
    MIN_TOUCHES = 3                  # Minimum touches to consider a level
    TOUCH_TOLERANCE_PCT = 0.0015     # 0.15% of price = "at level"
    
    # Volume thresholds
    HIGH_RVOL_THRESHOLD = 1.5        # 1.5x avg volume = elevated
    EXTREME_RVOL_THRESHOLD = 2.5     # 2.5x avg volume = institutional
    
    # Displacement failure
    MAX_CLOSE_BEYOND_PCT = 0.30      # If >30% of touches close beyond, wall is weak
    
    # Delta thresholds
    STRONG_DELTA_IMBALANCE = 0.30    # 30%+ imbalance = clear absorption
    
    # Lookback
    DEFAULT_LOOKBACK_BARS = 40       # ~10 hours of 15-min candles
    
    # Proximity for known level matching
    LEVEL_PROXIMITY_PCT = 0.003      # 0.3% to match VAH/VAL/etc
    
    def __init__(self, lookback_bars: Optional[int] = None):
        self.lookback_bars = lookback_bars or self.DEFAULT_LOOKBACK_BARS
    
    def analyze(self,
                df: pd.DataFrame,
                symbol: str,
                vah: float,
                poc: float,
                val: float,
                vwap: Optional[float] = None,
                fib_levels: Optional[List[float]] = None,
                exhaustion_price: Optional[float] = None  # Price from 3HER signal
                ) -> AbsorptionResult:
        """
        Run absorption analysis.
        
        Args:
            df: OHLCV DataFrame (5-min or 15-min candles)
            symbol: Ticker
            vah, poc, val: Volume profile from MTF scanner
            vwap: Session VWAP
            fib_levels: Fibonacci levels
            exhaustion_price: If 3HER fired, the price level to check
            
        Returns:
            AbsorptionResult with detected zones and scoring
        """
        scan_time = datetime.now()
        notes = []
        
        if len(df) < 10:
            return self._empty_result(symbol, scan_time, "Insufficient data")
        
        # Use most recent lookback window
        window = df.tail(self.lookback_bars).copy()
        current_price = window['close'].iloc[-1]
        avg_volume = window['volume'].mean()
        
        # === STEP 1: Find Candidate Price Levels ===
        candidates = self._find_candidate_levels(window, current_price)
        
        if not candidates:
            return self._empty_result(symbol, scan_time, "No price clustering detected")
        
        # === STEP 2: Analyze Each Candidate Level ===
        analyzed_levels = []
        for level_price, tolerance in candidates:
            pl = self._analyze_price_level(window, level_price, tolerance, avg_volume)
            if pl.touch_count >= self.MIN_TOUCHES:
                analyzed_levels.append(pl)
        
        if not analyzed_levels:
            return self._empty_result(symbol, scan_time, 
                f"No levels with {self.MIN_TOUCHES}+ touches")
        
        # === STEP 3: Classify and Score Absorption Zones ===
        zones = []
        for pl in analyzed_levels:
            zone = self._classify_zone(pl, current_price, avg_volume,
                                        vah, poc, val, vwap, fib_levels)
            if zone.score > 0:
                zones.append(zone)
        
        # Sort by score descending
        zones.sort(key=lambda z: z.score, reverse=True)
        
        # Primary zone = highest scored
        primary = zones[0] if zones else None
        
        # === STEP 4: Compute Dimension Scores ===
        d1, d2, d3, d4 = self._compute_dimension_scores(primary, analyzed_levels, avg_volume)
        
        # === STEP 5: 3HER Integration Check ===
        supports_exhaustion = False
        confluence_level = None
        
        if exhaustion_price and primary:
            # Check if the absorption zone is near the exhaustion price
            if abs(exhaustion_price - primary.center_price) / current_price < 0.005:
                supports_exhaustion = True
                confluence_level = primary.center_price
                notes.append(f"🎯 Absorption at ${primary.center_price:.2f} confirms 3HER exhaustion at ${exhaustion_price:.2f}")
        
        # Determine dominant type
        if primary and primary.absorption_type != AbsorptionType.NONE:
            dominant_type = primary.absorption_type
        else:
            dominant_type = AbsorptionType.NONE
        
        absorption_active = primary is not None and primary.strength.tradeable
        
        if absorption_active:
            notes.append(f"{primary.absorption_type.emoji} {primary.absorption_type.value} detected at ${primary.center_price:.2f}")
            notes.append(f"   {primary.total_touches} touches | {primary.rvol_ratio:.1f}x RVOL | {primary.status.value}")
        
        # Add zone-specific notes
        for zone in zones[:3]:  # Top 3 zones
            notes.extend(zone.notes)
        
        return AbsorptionResult(
            symbol=symbol,
            scan_time=scan_time,
            current_price=current_price,
            zones=zones,
            primary_zone=primary,
            absorption_active=absorption_active,
            dominant_type=dominant_type,
            touch_frequency_score=d1,
            volume_concentration_score=d2,
            displacement_failure_score=d3,
            delta_absorption_score=d4,
            supports_exhaustion=supports_exhaustion,
            exhaustion_confluence_level=confluence_level,
            notes=notes
        )
    
    # =========================================================================
    # STEP 1: FIND CANDIDATE LEVELS
    # =========================================================================
    
    def _find_candidate_levels(self, df: pd.DataFrame, 
                                current_price: float) -> List[Tuple[float, float]]:
        """
        Identify price levels where trading activity clusters.
        
        Uses a histogram approach: bin all high/low/close touches into
        price buckets, then find bins with outsized activity.
        
        Returns:
            List of (price_level, tolerance) tuples
        """
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max <= price_min:
            return []
        
        bin_size = (price_max - price_min) / self.PRICE_BIN_COUNT
        bins = np.linspace(price_min, price_max, self.PRICE_BIN_COUNT + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Count touches per bin (how many candles had high/low/close in this bin)
        touch_counts = np.zeros(self.PRICE_BIN_COUNT)
        volume_at_bin = np.zeros(self.PRICE_BIN_COUNT)
        
        for _, row in df.iterrows():
            # Each candle "touches" bins from its low to its high
            low_bin = max(0, int((row['low'] - price_min) / bin_size))
            high_bin = min(self.PRICE_BIN_COUNT - 1, int((row['high'] - price_min) / bin_size))
            
            # Close location - weight the bin where price settled
            close_bin = min(self.PRICE_BIN_COUNT - 1, 
                          max(0, int((row['close'] - price_min) / bin_size)))
            
            # Count high and low as touches (these are the extremes price reached)
            low_exact_bin = max(0, int((row['low'] - price_min) / bin_size))
            high_exact_bin = min(self.PRICE_BIN_COUNT - 1, int((row['high'] - price_min) / bin_size))
            
            touch_counts[low_exact_bin] += 1
            touch_counts[high_exact_bin] += 1
            touch_counts[close_bin] += 0.5  # Close is less significant than extremes
            
            # Volume distributed across range, weighted toward extremes
            for b in range(low_bin, high_bin + 1):
                if b == low_exact_bin or b == high_exact_bin:
                    volume_at_bin[b] += row['volume'] * 0.4  # 40% each extreme
                else:
                    remaining = row['volume'] * 0.2
                    span = max(1, high_bin - low_bin - 1)
                    volume_at_bin[b] += remaining / span
        
        # Find bins with outsized touch counts (above 1.5 std deviations)
        mean_touches = touch_counts.mean()
        std_touches = touch_counts.std()
        threshold = mean_touches + 1.5 * std_touches if std_touches > 0 else mean_touches + 1
        
        candidates = []
        for i in range(self.PRICE_BIN_COUNT):
            if touch_counts[i] >= threshold and touch_counts[i] >= self.MIN_TOUCHES:
                level_price = bin_centers[i]
                tolerance = current_price * self.TOUCH_TOLERANCE_PCT
                candidates.append((level_price, tolerance))
        
        # Merge nearby candidates (within 2 bins of each other)
        merged = self._merge_nearby_levels(candidates, bin_size * 2)
        
        return merged
    
    def _merge_nearby_levels(self, candidates: List[Tuple[float, float]], 
                              merge_distance: float) -> List[Tuple[float, float]]:
        """Merge candidate levels that are very close together."""
        if not candidates:
            return []
        
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        merged = [sorted_cands[0]]
        
        for price, tol in sorted_cands[1:]:
            if price - merged[-1][0] <= merge_distance:
                # Merge: take average price, wider tolerance
                avg_price = (merged[-1][0] + price) / 2
                max_tol = max(merged[-1][1], tol)
                merged[-1] = (avg_price, max_tol)
            else:
                merged.append((price, tol))
        
        return merged
    
    # =========================================================================
    # STEP 2: ANALYZE EACH PRICE LEVEL
    # =========================================================================
    
    def _analyze_price_level(self, df: pd.DataFrame, level_price: float,
                              tolerance: float, avg_volume: float) -> PriceLevel:
        """
        Deep analysis of a single price level.
        
        For each candle, determine if price "touched" this level,
        then aggregate volume, delta, and rejection behavior.
        """
        pl = PriceLevel(price=level_price, tolerance=tolerance)
        
        touches_volume = []
        rejection_sizes = []
        deltas_at_level = []
        
        for ts, row in df.iterrows():
            bar_high = row['high']
            bar_low = row['low']
            bar_close = row['close']
            bar_open = row['open']
            bar_volume = row['volume']
            
            # Does this candle touch the level?
            level_upper = level_price + tolerance
            level_lower = level_price - tolerance
            
            touches_from_above = bar_low <= level_upper and bar_low >= level_lower
            touches_from_below = bar_high >= level_lower and bar_high <= level_upper
            passes_through = bar_low <= level_price <= bar_high
            
            if touches_from_above or touches_from_below or passes_through:
                pl.touch_count += 1
                pl.touch_timestamps.append(ts)
                pl.total_volume_at_level += bar_volume
                touches_volume.append(bar_volume)
                
                if bar_volume > pl.max_single_bar_volume:
                    pl.max_single_bar_volume = bar_volume
                
                # Did price CLOSE beyond the level?
                # For ceiling: close above level = broken
                # For floor: close below level = broken
                # We don't know type yet, so track both
                closed_above = bar_close > level_upper
                closed_below = bar_close < level_lower
                
                if closed_above or closed_below:
                    pl.close_beyond_count += 1
                
                # Wick beyond but close inside = absorption
                wicked_above = bar_high > level_upper and bar_close <= level_upper
                wicked_below = bar_low < level_lower and bar_close >= level_lower
                
                if wicked_above or wicked_below:
                    pl.wick_beyond_count += 1
                
                # Penetration depth
                penetration_above = max(0, bar_high - level_upper)
                penetration_below = max(0, level_lower - bar_low)
                max_pen = max(penetration_above, penetration_below)
                pl.max_penetration = max(pl.max_penetration, max_pen)
                
                # Rejection: how far did price bounce back from the level?
                if touches_from_below:
                    rejection = max(0, level_price - bar_close)
                    rejection_sizes.append(rejection)
                elif touches_from_above:
                    rejection = max(0, bar_close - level_price)
                    rejection_sizes.append(rejection)
                
                # Delta at level (CLV method, consistent with FlowControlEngine)
                bar_range = bar_high - bar_low
                if bar_range > 0:
                    clv = (bar_close - bar_low) / bar_range
                    delta = bar_volume * (2 * clv - 1)
                else:
                    delta = 0
                deltas_at_level.append(delta)
                
                # Time at level
                pl.time_at_level_bars += 1
        
        # Aggregate
        if touches_volume:
            pl.avg_volume_per_touch = np.mean(touches_volume)
            pl.rvol_at_level = pl.avg_volume_per_touch / avg_volume if avg_volume > 0 else 1.0
        
        if rejection_sizes:
            pl.avg_rejection_size = np.mean(rejection_sizes)
        
        if deltas_at_level:
            pl.cumulative_delta_at_level = sum(deltas_at_level)
            positive_delta = sum(d for d in deltas_at_level if d > 0)
            negative_delta = sum(abs(d) for d in deltas_at_level if d < 0)
            total_delta_vol = positive_delta + negative_delta
            
            if total_delta_vol > 0:
                # Imbalance: which side is winning the delta war?
                pl.delta_imbalance = (positive_delta - negative_delta) / total_delta_vol
                
                # Passive vs aggressive depends on context
                # Positive delta = buyers aggressive, negative = sellers aggressive
                pl.aggressive_side_volume = max(positive_delta, negative_delta)
                pl.passive_side_volume = min(positive_delta, negative_delta)
        
        return pl
    
    # =========================================================================
    # STEP 3: CLASSIFY AND SCORE ZONES
    # =========================================================================
    
    def _classify_zone(self, pl: PriceLevel, current_price: float,
                        avg_volume: float,
                        vah: float, poc: float, val: float,
                        vwap: Optional[float],
                        fib_levels: Optional[List[float]]) -> AbsorptionZone:
        """
        Classify a price level as an absorption zone with type, strength, score.
        """
        notes = []
        
        # --- Determine Absorption Type ---
        # If price is below the level and keeps getting rejected = CEILING
        # If price is above the level and keeps bouncing = FLOOR
        price_relative = current_price - pl.price
        
        # Delta tells us who's winning
        # Negative delta at level = sellers winning (absorbing buyers) = CEILING
        # Positive delta at level = buyers winning (absorbing sellers) = FLOOR
        if pl.delta_imbalance < -0.1 and pl.wick_beyond_count >= 2:
            abs_type = AbsorptionType.CEILING
        elif pl.delta_imbalance > 0.1 and pl.wick_beyond_count >= 2:
            abs_type = AbsorptionType.FLOOR
        elif pl.wick_beyond_count >= 2 and abs(pl.delta_imbalance) <= 0.1:
            abs_type = AbsorptionType.PINNING
        elif pl.touch_count >= self.MIN_TOUCHES:
            # Infer from price position relative to level
            if current_price < pl.price:
                abs_type = AbsorptionType.CEILING
            elif current_price > pl.price:
                abs_type = AbsorptionType.FLOOR
            else:
                abs_type = AbsorptionType.PINNING
        else:
            abs_type = AbsorptionType.NONE
        
        # --- Score the Zone (4 sub-dimensions, 0-25 each) ---
        
        # Sub-score 1: Touch Frequency (0-25)
        touch_score = 0
        if pl.touch_count >= 8:
            touch_score = 25
        elif pl.touch_count >= 6:
            touch_score = 20
        elif pl.touch_count >= 5:
            touch_score = 15
        elif pl.touch_count >= 4:
            touch_score = 10
        elif pl.touch_count >= 3:
            touch_score = 5
        
        # Bonus for recent touches (clustering in time)
        if len(pl.touch_timestamps) >= 3:
            recent_3 = pl.touch_timestamps[-3:]
            if hasattr(recent_3[0], 'minute'):
                try:
                    time_span = (recent_3[-1] - recent_3[0]).total_seconds() / 60
                    if time_span < 120:  # 3 touches within 2 hours
                        touch_score = min(25, touch_score + 5)
                        notes.append(f"⏱️ Rapid retesting — 3 touches in {time_span:.0f}min")
                except:
                    pass
        
        # Sub-score 2: Volume Concentration (0-25)
        vol_score = 0
        if pl.rvol_at_level >= self.EXTREME_RVOL_THRESHOLD:
            vol_score = 25
            notes.append(f"📊 Extreme volume at level ({pl.rvol_at_level:.1f}x RVOL)")
        elif pl.rvol_at_level >= self.HIGH_RVOL_THRESHOLD:
            vol_score = 18
        elif pl.rvol_at_level >= 1.2:
            vol_score = 10
        elif pl.rvol_at_level >= 1.0:
            vol_score = 5
        
        # Max single bar spike adds conviction
        if avg_volume > 0 and pl.max_single_bar_volume / avg_volume >= 3.0:
            vol_score = min(25, vol_score + 5)
            notes.append(f"💥 Volume spike at level ({pl.max_single_bar_volume / avg_volume:.1f}x single bar)")
        
        # Sub-score 3: Displacement Failure (0-25)
        disp_score = 0
        if pl.touch_count > 0:
            close_beyond_ratio = pl.close_beyond_count / pl.touch_count
            wick_beyond_ratio = pl.wick_beyond_count / pl.touch_count
            
            # High wick-beyond but low close-beyond = strong absorption
            if close_beyond_ratio <= 0.1 and wick_beyond_ratio >= 0.5:
                disp_score = 25
                notes.append("🚫 Price repeatedly rejected — wicks beyond but never closes through")
            elif close_beyond_ratio <= 0.2 and wick_beyond_ratio >= 0.4:
                disp_score = 20
            elif close_beyond_ratio <= self.MAX_CLOSE_BEYOND_PCT:
                disp_score = 12
            elif close_beyond_ratio <= 0.5:
                disp_score = 5
            # If >50% close beyond, wall is weak/broken
            
            # Low penetration depth = wall is firm
            if pl.max_penetration > 0:
                pen_pct = pl.max_penetration / pl.price
                if pen_pct < 0.001:  # Less than 0.1% penetration
                    disp_score = min(25, disp_score + 5)
        
        # Sub-score 4: Delta Absorption (0-25)
        delta_score = 0
        abs_imbalance = abs(pl.delta_imbalance)
        
        if abs_imbalance >= 0.5:
            delta_score = 25
            notes.append(f"⚡ Dominant delta absorption ({pl.delta_imbalance:+.2f} imbalance)")
        elif abs_imbalance >= self.STRONG_DELTA_IMBALANCE:
            delta_score = 20
        elif abs_imbalance >= 0.2:
            delta_score = 12
        elif abs_imbalance >= 0.1:
            delta_score = 5
        
        total_score = touch_score + vol_score + disp_score + delta_score
        
        # --- Determine Strength ---
        if total_score >= 85:
            strength = AbsorptionStrength.INSTITUTIONAL
        elif total_score >= 65:
            strength = AbsorptionStrength.STRONG
        elif total_score >= 45:
            strength = AbsorptionStrength.MODERATE
        elif total_score >= 25:
            strength = AbsorptionStrength.WEAK
        else:
            strength = AbsorptionStrength.NONE
        
        # --- Determine Wall Status ---
        if pl.touch_count > 0:
            close_beyond_ratio = pl.close_beyond_count / pl.touch_count
            if close_beyond_ratio > 0.5:
                status = WallStatus.BROKEN
            elif close_beyond_ratio > 0.3:
                status = WallStatus.WEAKENING
            elif current_price < pl.price - pl.tolerance and abs_type == AbsorptionType.CEILING:
                status = WallStatus.DEFENDED
            elif current_price > pl.price + pl.tolerance and abs_type == AbsorptionType.FLOOR:
                status = WallStatus.DEFENDED
            else:
                status = WallStatus.HOLDING
        else:
            status = WallStatus.HOLDING
        
        # --- Check Proximity to Known Levels ---
        near_vah = abs(pl.price - vah) / pl.price < self.LEVEL_PROXIMITY_PCT
        near_val = abs(pl.price - val) / pl.price < self.LEVEL_PROXIMITY_PCT
        near_poc = abs(pl.price - poc) / pl.price < self.LEVEL_PROXIMITY_PCT
        near_vwap = vwap is not None and abs(pl.price - vwap) / pl.price < self.LEVEL_PROXIMITY_PCT
        near_fib = False
        if fib_levels:
            for fib in fib_levels:
                if abs(pl.price - fib) / pl.price < self.LEVEL_PROXIMITY_PCT:
                    near_fib = True
                    break
        
        # Confluence bonuses
        confluences = []
        if near_vah:
            confluences.append("VAH")
        if near_val:
            confluences.append("VAL")
        if near_poc:
            confluences.append("POC")
        if near_vwap:
            confluences.append("VWAP")
        if near_fib:
            confluences.append("Fib")
        
        if confluences:
            bonus = min(10, len(confluences) * 5)
            total_score = min(100, total_score + bonus)
            notes.append(f"🔗 Confluence with {', '.join(confluences)} (+{bonus}pts)")
        
        zone_width = pl.tolerance * 2
        
        return AbsorptionZone(
            center_price=pl.price,
            upper_bound=pl.price + pl.tolerance,
            lower_bound=pl.price - pl.tolerance,
            zone_width=zone_width,
            zone_width_pct=zone_width / pl.price if pl.price > 0 else 0,
            absorption_type=abs_type,
            strength=strength,
            status=status,
            total_touches=pl.touch_count,
            total_volume=pl.total_volume_at_level,
            rvol_ratio=pl.rvol_at_level,
            delta_imbalance=pl.delta_imbalance,
            time_spent_bars=pl.time_at_level_bars,
            score=total_score,
            near_vah=near_vah,
            near_val=near_val,
            near_poc=near_poc,
            near_vwap=near_vwap,
            near_fib=near_fib,
            notes=notes
        )
    
    # =========================================================================
    # STEP 4: DIMENSION SCORES
    # =========================================================================
    
    def _compute_dimension_scores(self, primary: Optional[AbsorptionZone],
                                   levels: List[PriceLevel],
                                   avg_volume: float) -> Tuple[int, int, int, int]:
        """
        Compute the 4 global dimension scores for the AbsorptionResult.
        These reflect the overall absorption picture, not just one zone.
        """
        if not primary or not levels:
            return 0, 0, 0, 0
        
        # D1: Touch Frequency — how aggressively is the market testing levels?
        max_touches = max(pl.touch_count for pl in levels)
        total_levels_tested = len([pl for pl in levels if pl.touch_count >= 3])
        
        d1 = 0
        if max_touches >= 8:
            d1 = 25
        elif max_touches >= 6:
            d1 = 20
        elif max_touches >= 4:
            d1 = 12
        elif max_touches >= 3:
            d1 = 7
        
        # Multiple tested levels adds conviction
        if total_levels_tested >= 3:
            d1 = min(25, d1 + 3)
        
        # D2: Volume Concentration — is volume bunching at levels?
        max_rvol = max(pl.rvol_at_level for pl in levels)
        d2 = 0
        if max_rvol >= self.EXTREME_RVOL_THRESHOLD:
            d2 = 25
        elif max_rvol >= self.HIGH_RVOL_THRESHOLD:
            d2 = 18
        elif max_rvol >= 1.2:
            d2 = 10
        elif max_rvol >= 1.0:
            d2 = 5
        
        # D3: Displacement Failure — are levels holding?
        # Look at the primary zone
        if primary.status == WallStatus.HOLDING:
            d3 = 22
        elif primary.status == WallStatus.DEFENDED:
            d3 = 25
        elif primary.status == WallStatus.WEAKENING:
            d3 = 10
        else:  # BROKEN
            d3 = 3
        
        # D4: Delta Absorption — passive side winning?
        max_abs_imbalance = max(abs(pl.delta_imbalance) for pl in levels)
        d4 = 0
        if max_abs_imbalance >= 0.5:
            d4 = 25
        elif max_abs_imbalance >= self.STRONG_DELTA_IMBALANCE:
            d4 = 20
        elif max_abs_imbalance >= 0.2:
            d4 = 12
        elif max_abs_imbalance >= 0.1:
            d4 = 5
        
        return d1, d2, d3, d4
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _empty_result(self, symbol: str, scan_time: datetime, reason: str) -> AbsorptionResult:
        return AbsorptionResult(
            symbol=symbol, scan_time=scan_time,
            current_price=0, zones=[], primary_zone=None,
            absorption_active=False, dominant_type=AbsorptionType.NONE,
            touch_frequency_score=0, volume_concentration_score=0,
            displacement_failure_score=0, delta_absorption_score=0,
            supports_exhaustion=False, exhaustion_confluence_level=None,
            notes=[reason]
        )
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def print_report(self, result: AbsorptionResult) -> str:
        """Generate human-readable absorption report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  ABSORPTION SCAN: {result.symbol}")
        lines.append(f"  {result.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Price: ${result.current_price:.2f}")
        lines.append("=" * 60)
        
        if not result.zones:
            lines.append("\n  ⬜ No absorption detected")
            if result.notes:
                for note in result.notes:
                    lines.append(f"  {note}")
            return "\n".join(lines)
        
        # Dimension scores
        lines.append(f"\n📊 ABSORPTION SCORING:")
        lines.append(f"   D1 Touch Frequency:      {result.touch_frequency_score:>2}/25")
        lines.append(f"   D2 Volume Concentration:  {result.volume_concentration_score:>2}/25")
        lines.append(f"   D3 Displacement Failure:  {result.displacement_failure_score:>2}/25")
        lines.append(f"   D4 Delta Absorption:      {result.delta_absorption_score:>2}/25")
        lines.append(f"   {'─' * 40}")
        lines.append(f"   TOTAL:                    {result.total_score:>2}/100  [{result.score_label}]")
        
        # Primary zone
        if result.primary_zone:
            z = result.primary_zone
            lines.append(f"\n{z.absorption_type.emoji} PRIMARY ZONE: {z.absorption_type.value} at ${z.center_price:.2f}")
            lines.append(f"   Range: ${z.lower_bound:.2f} — ${z.upper_bound:.2f}")
            lines.append(f"   Strength: {z.strength.value} | Status: {z.status.value}")
            lines.append(f"   Touches: {z.total_touches} | RVOL: {z.rvol_ratio:.1f}x | Delta: {z.delta_imbalance:+.2f}")
            lines.append(f"   Implication: {z.absorption_type.trade_implication}")
        
        # Additional zones
        if len(result.zones) > 1:
            lines.append(f"\n📍 SECONDARY ZONES:")
            for z in result.zones[1:4]:  # Show up to 3 more
                lines.append(f"   {z.absorption_type.emoji} ${z.center_price:.2f} | "
                            f"{z.strength.value} | {z.total_touches} touches | "
                            f"Score: {z.score}")
        
        # 3HER integration
        if result.supports_exhaustion:
            lines.append(f"\n🎯 3HER CONFLUENCE:")
            lines.append(f"   Absorption confirms exhaustion at ${result.exhaustion_confluence_level:.2f}")
            lines.append(f"   → HIGHEST conviction reversal setup")
        
        # Notes
        if result.notes:
            lines.append(f"\n📝 NOTES:")
            for note in result.notes:
                lines.append(f"   {note}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# COMBINED 3HER + ABSORPTION ANALYZER
# =============================================================================

class ExhaustionAbsorptionAnalyzer:
    """
    Combines 3HER exhaustion detection with absorption analysis
    for maximum conviction reversal setups.
    
    When both fire at the same price level, that's the signal
    that tells you WHAT (exhaustion) and WHERE/WHY (absorption).
    
    Usage:
        from three_hour_exhaustion import ThreeHourExhaustionScanner
        
        analyzer = ExhaustionAbsorptionAnalyzer()
        combined = analyzer.analyze(
            df_30m=..., df_15m=..., df_detail=...,
            symbol="NVDA", daily_atr=5.20,
            vah=142.50, poc=140.80, val=139.20
        )
    """
    
    def __init__(self):
        self.absorption_detector = AbsorptionDetector()
        # 3HER scanner imported at runtime to avoid circular deps
        self._exhaustion_scanner = None
    
    def _get_exhaustion_scanner(self):
        if self._exhaustion_scanner is None:
            try:
                from three_hour_exhaustion import ThreeHourExhaustionScanner
                self._exhaustion_scanner = ThreeHourExhaustionScanner()
            except ImportError:
                return None
        return self._exhaustion_scanner
    
    def analyze(self,
                df_30m: pd.DataFrame,
                df_15m: pd.DataFrame,
                df_detail: pd.DataFrame,  # 5-min or 15-min for absorption
                symbol: str,
                daily_atr: float,
                vah: float,
                poc: float,
                val: float,
                vwap: Optional[float] = None,
                fib_levels: Optional[List[float]] = None,
                active_mtf_direction: Optional[str] = None
                ) -> Dict:
        """
        Run both analyses and combine results.
        
        Returns dict with:
        - exhaustion: ExhaustionResult (or None)
        - absorption: AbsorptionResult
        - confluence: bool (both agree at same level)
        - combined_score: int (0-100, weighted blend)
        - recommendation: str
        """
        # Run absorption
        absorption = self.absorption_detector.analyze(
            df=df_detail, symbol=symbol,
            vah=vah, poc=poc, val=val,
            vwap=vwap, fib_levels=fib_levels
        )
        
        # Run 3HER
        exhaustion = None
        scanner = self._get_exhaustion_scanner()
        if scanner:
            exhaustion = scanner.scan(
                df_30m=df_30m, df_15m=df_15m,
                symbol=symbol, daily_atr=daily_atr,
                vah=vah, poc=poc, val=val,
                vwap=vwap, fib_levels=fib_levels,
                active_mtf_direction=active_mtf_direction
            )
        
        # Check confluence
        confluence = False
        confluence_level = None
        
        if (exhaustion and absorption.primary_zone and 
            exhaustion.level.tradeable and absorption.absorption_active):
            
            exh_price = exhaustion.entry_price or exhaustion.move.window_high
            abs_price = absorption.primary_zone.center_price
            
            if abs(exh_price - abs_price) / abs_price < 0.005:  # Within 0.5%
                confluence = True
                confluence_level = abs_price
        
        # Combined score: 60% exhaustion + 40% absorption when both present
        if exhaustion and exhaustion.adjusted_score > 0:
            combined_score = int(exhaustion.adjusted_score * 0.6 + absorption.total_score * 0.4)
            if confluence:
                combined_score = min(100, combined_score + 10)  # Confluence bonus
        else:
            combined_score = absorption.total_score
        
        # Recommendation
        if confluence and combined_score >= 75:
            recommendation = "🔴 HIGH CONVICTION — Exhaustion + Absorption confluence. Full 3HER entry."
        elif exhaustion and exhaustion.level.tradeable and absorption.absorption_active:
            recommendation = "🟠 STRONG — Both signals active but at different levels. Enter 3HER, use absorption as stop reference."
        elif exhaustion and exhaustion.level.tradeable:
            recommendation = "🟡 MODERATE — Exhaustion only, no absorption wall detected. Standard 3HER entry."
        elif absorption.absorption_active:
            recommendation = f"🟡 MODERATE — Absorption wall at ${absorption.primary_zone.center_price:.2f} but no exhaustion. Watch for bounce/rejection."
        else:
            recommendation = "⬜ NO SETUP — Neither exhaustion nor absorption detected."
        
        return {
            'exhaustion': exhaustion,
            'absorption': absorption,
            'confluence': confluence,
            'confluence_level': confluence_level,
            'combined_score': combined_score,
            'recommendation': recommendation
        }


# =============================================================================
# DEMO / TEST
# =============================================================================

def generate_absorption_demo() -> pd.DataFrame:
    """
    Generate 15-min candles showing a ceiling absorption pattern.
    Price pushes up repeatedly to ~460 but gets rejected every time.
    """
    np.random.seed(123)
    
    base_price = 455.0
    base_time = datetime(2026, 2, 12, 9, 30)
    ceiling_level = 460.0
    
    records = []
    price = base_price
    
    for i in range(40):  # 40 x 15min = 10 hours
        # Create natural-looking price action that repeatedly tests 460
        cycle_position = i % 8  # 8-candle cycles
        
        if cycle_position < 4:
            # Push up toward ceiling
            drift = np.random.uniform(0.3, 1.2)
            noise = np.random.randn() * 0.2
        elif cycle_position == 4:
            # Hit the ceiling — big volume, rejected
            drift = np.random.uniform(0.0, 0.5)
            noise = np.random.randn() * 0.1
            price = min(price + drift, ceiling_level + np.random.uniform(0, 0.3))
        else:
            # Rejected — pull back
            drift = np.random.uniform(-1.5, -0.3)
            noise = np.random.randn() * 0.3
        
        open_p = price
        
        if cycle_position == 4:
            # Absorption candle: price goes up but closes below ceiling
            high_p = ceiling_level + np.random.uniform(0.1, 0.5)
            close_p = ceiling_level - np.random.uniform(0.2, 1.0)
            low_p = close_p - np.random.uniform(0, 0.3)
            vol = int(np.random.uniform(200000, 400000))  # HIGH volume
        elif cycle_position < 4:
            close_p = price + drift + noise
            high_p = max(open_p, close_p) + abs(np.random.randn()) * 0.2
            low_p = min(open_p, close_p) - abs(np.random.randn()) * 0.15
            vol = int(np.random.uniform(80000, 150000))
        else:
            close_p = price + drift + noise
            high_p = max(open_p, close_p) + abs(np.random.randn()) * 0.2
            low_p = min(open_p, close_p) - abs(np.random.randn()) * 0.3
            vol = int(np.random.uniform(60000, 120000))
        
        records.append({
            'timestamp': base_time + timedelta(minutes=15 * i),
            'open': round(open_p, 2),
            'high': round(high_p, 2),
            'low': round(low_p, 2),
            'close': round(close_p, 2),
            'volume': vol
        })
        
        price = close_p
        # Keep price in realistic range
        price = max(450, min(price, 462))
    
    df = pd.DataFrame(records)
    df.set_index('timestamp', inplace=True)
    return df


if __name__ == "__main__":
    print("Generating absorption demo data...")
    df = generate_absorption_demo()
    
    print(f"15m candles: {len(df)}")
    print(f"Price range: ${df['low'].min():.2f} — ${df['high'].max():.2f}")
    print()
    
    # Run detector
    detector = AbsorptionDetector()
    result = detector.analyze(
        df=df,
        symbol="DEMO",
        vah=460.0,
        poc=456.0,
        val=452.0,
        vwap=457.0,
        fib_levels=[453.5, 457.0, 460.5],
        exhaustion_price=459.5  # Simulate 3HER signal near ceiling
    )
    
    # Print report
    report = detector.print_report(result)
    print(report)
