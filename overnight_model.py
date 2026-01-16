"""
Overnight & Gap Predictive Model
================================
Analyzes overnight sessions, gaps, and pre-market activity to predict
intraday directional bias and key levels.

For Brokers:
-----------
Overnight moves (6PM-9:30AM ET) show what institutions are doing while
retail sleeps. Gap analysis reveals:
- Gap Up: Buyers aggressive overnight → continuation or fade?
- Gap Down: Sellers aggressive overnight → continuation or fade?
- Gap Fill Probability: Statistical likelihood of filling the gap

For Programmers:
---------------
Separates RTH (Regular Trading Hours) from ETH (Extended Trading Hours),
calculates gap metrics, overnight value areas, and generates predictive
signals based on historical gap behavior patterns.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class GapType(Enum):
    """Classification of gap types"""
    GAP_UP_LARGE = "GAP_UP_LARGE"       # > 1% gap up
    GAP_UP_SMALL = "GAP_UP_SMALL"       # 0.3-1% gap up
    GAP_DOWN_LARGE = "GAP_DOWN_LARGE"   # > 1% gap down
    GAP_DOWN_SMALL = "GAP_DOWN_SMALL"   # 0.3-1% gap down
    NO_GAP = "NO_GAP"                   # < 0.3% either direction
    
    @property
    def emoji(self) -> str:
        return {
            "GAP_UP_LARGE": "⬆️⬆️",
            "GAP_UP_SMALL": "⬆️",
            "GAP_DOWN_LARGE": "⬇️⬇️",
            "GAP_DOWN_SMALL": "⬇️",
            "NO_GAP": "➡️"
        }[self.value]


class GapFillProbability(Enum):
    """Likelihood of gap fill"""
    HIGH = "HIGH"           # >70% historical fill rate
    MODERATE = "MODERATE"   # 50-70% fill rate
    LOW = "LOW"             # <50% fill rate


class OvernightBias(Enum):
    """Overnight session directional bias"""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    
    @property
    def emoji(self) -> str:
        return {
            "STRONG_BULLISH": "🟢🟢",
            "BULLISH": "🟢",
            "NEUTRAL": "🟡",
            "BEARISH": "🔴",
            "STRONG_BEARISH": "🔴🔴"
        }[self.value]


class DayType(Enum):
    """Prior day classification for context"""
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE_DAY = "RANGE_DAY"
    REVERSAL_UP = "REVERSAL_UP"
    REVERSAL_DOWN = "REVERSAL_DOWN"
    INSIDE_DAY = "INSIDE_DAY"
    OUTSIDE_DAY = "OUTSIDE_DAY"


@dataclass
class OvernightSession:
    """Overnight/Globex session metrics"""
    session_date: datetime
    
    # Price levels
    overnight_high: float
    overnight_low: float
    overnight_open: float
    overnight_close: float
    overnight_range: float
    overnight_midpoint: float
    
    # Volume (if available)
    overnight_volume: float
    overnight_delta: float  # Estimated buying vs selling
    
    # Value area (if enough data)
    overnight_poc: Optional[float] = None
    overnight_vah: Optional[float] = None
    overnight_val: Optional[float] = None
    
    @property
    def overnight_direction(self) -> str:
        """Direction of overnight move"""
        change = self.overnight_close - self.overnight_open
        pct = (change / self.overnight_open * 100) if self.overnight_open else 0
        if pct > 0.3:
            return "UP"
        elif pct < -0.3:
            return "DOWN"
        return "FLAT"


@dataclass
class GapAnalysis:
    """Gap analysis between sessions"""
    # Gap metrics
    gap_size: float              # Absolute gap in points
    gap_pct: float               # Gap as percentage
    gap_type: GapType
    gap_atr_ratio: float         # Gap size relative to ATR
    
    # Prior close reference
    prior_close: float
    current_open: float
    
    # Fill analysis
    gap_fill_level: float        # Price that would fill the gap
    gap_fill_probability: GapFillProbability
    partial_fill_level: float    # 50% fill level
    
    # Historical context (if available)
    similar_gaps_filled_pct: float = 0.0
    avg_fill_time_minutes: float = 0.0


@dataclass
class PriorDayContext:
    """Prior day metrics for context"""
    date: datetime
    
    # OHLC
    prior_open: float
    prior_high: float
    prior_low: float
    prior_close: float
    prior_range: float
    
    # Value area
    prior_poc: float
    prior_vah: float
    prior_val: float
    
    # Classification
    day_type: DayType
    
    # ATR for normalization
    atr: float
    
    @property
    def prior_midpoint(self) -> float:
        return (self.prior_high + self.prior_low) / 2
    
    @property
    def close_in_value(self) -> bool:
        return self.prior_val <= self.prior_close <= self.prior_vah


@dataclass  
class OpeningContext:
    """Where price opens relative to key levels"""
    open_price: float
    
    # Relative to prior day
    vs_prior_close: str          # "ABOVE", "BELOW", "AT"
    vs_prior_high: str
    vs_prior_low: str
    vs_prior_poc: str
    vs_prior_vah: str
    vs_prior_val: str
    
    # Relative to overnight
    vs_overnight_high: str
    vs_overnight_low: str
    vs_overnight_midpoint: str
    
    # Opening scenario
    scenario: str                # "ABOVE_ALL", "IN_RANGE", "BELOW_ALL", etc.


@dataclass
class OvernightPrediction:
    """Complete overnight/gap predictive assessment"""
    symbol: str
    analysis_time: datetime
    
    # Components
    overnight: OvernightSession
    gap: GapAnalysis
    prior_day: PriorDayContext
    opening: OpeningContext
    
    # Prediction
    bias: OvernightBias
    confidence: float            # 0-100
    
    # Key levels for the day
    key_levels: Dict[str, float]
    
    # Scenarios
    bull_scenario: str
    bear_scenario: str
    
    # Notes
    notes: List[str] = field(default_factory=list)


# =============================================================================
# OVERNIGHT SESSION ANALYZER
# =============================================================================

class OvernightAnalyzer:
    """
    Analyzes overnight/ETH sessions
    
    For Brokers:
    -----------
    The overnight session (Globex) runs from 6PM to 9:30AM ET.
    Key insights:
    - Overnight high/low often act as support/resistance
    - Overnight POC shows where institutions found value
    - Direction of overnight move hints at sentiment
    
    For Programmers:
    ---------------
    Filters data to ETH hours, calculates session metrics.
    RTH: 9:30 AM - 4:00 PM ET
    ETH: 6:00 PM - 9:30 AM ET (next day)
    """
    
    # Market hours (ET)
    RTH_START = time(9, 30)
    RTH_END = time(16, 0)
    ETH_START = time(18, 0)
    
    def __init__(self):
        pass
    
    def extract_overnight_session(self, 
                                   df: pd.DataFrame, 
                                   target_date: Optional[datetime] = None) -> Optional[OvernightSession]:
        """
        Extract overnight session data
        
        Args:
            df: Full OHLCV dataframe with datetime index
            target_date: Date to analyze (default: most recent)
        
        Returns:
            OvernightSession object or None
        """
        if len(df) < 10:
            return None
        
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        
        # Get target date
        if target_date is None:
            target_date = df.index[-1].date()
        
        # Filter for overnight session
        # Overnight for day D = evening of D-1 (6PM+) + morning of D (until 9:30AM)
        prev_evening_start = datetime.combine(target_date - timedelta(days=1), self.ETH_START)
        morning_end = datetime.combine(target_date, self.RTH_START)
        
        overnight_mask = (df.index >= prev_evening_start) & (df.index < morning_end)
        overnight_df = df[overnight_mask]
        
        if len(overnight_df) < 3:
            # Not enough overnight data, estimate from daily
            return self._estimate_overnight_from_daily(df, target_date)
        
        # Calculate overnight metrics
        overnight_high = overnight_df['high'].max()
        overnight_low = overnight_df['low'].min()
        overnight_open = overnight_df['open'].iloc[0]
        overnight_close = overnight_df['close'].iloc[-1]
        overnight_range = overnight_high - overnight_low
        overnight_midpoint = (overnight_high + overnight_low) / 2
        overnight_volume = overnight_df['volume'].sum()
        
        # Estimate delta from price action
        overnight_delta = self._estimate_session_delta(overnight_df)
        
        # Simple overnight POC (highest volume price level)
        overnight_poc, overnight_vah, overnight_val = self._calculate_session_value_area(overnight_df)
        
        return OvernightSession(
            session_date=target_date,
            overnight_high=overnight_high,
            overnight_low=overnight_low,
            overnight_open=overnight_open,
            overnight_close=overnight_close,
            overnight_range=overnight_range,
            overnight_midpoint=overnight_midpoint,
            overnight_volume=overnight_volume,
            overnight_delta=overnight_delta,
            overnight_poc=overnight_poc,
            overnight_vah=overnight_vah,
            overnight_val=overnight_val
        )
    
    def _estimate_overnight_from_daily(self, 
                                        df: pd.DataFrame, 
                                        target_date: datetime) -> OvernightSession:
        """Estimate overnight session from daily data when intraday not available"""
        # Get prior day close and current day open
        daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(daily) < 2:
            # Return minimal session
            last_price = df['close'].iloc[-1]
            return OvernightSession(
                session_date=target_date,
                overnight_high=last_price,
                overnight_low=last_price,
                overnight_open=last_price,
                overnight_close=last_price,
                overnight_range=0,
                overnight_midpoint=last_price,
                overnight_volume=0,
                overnight_delta=0
            )
        
        prior_close = daily['close'].iloc[-2]
        current_open = daily['open'].iloc[-1]
        
        # Estimate overnight range as gap + small buffer
        gap = abs(current_open - prior_close)
        estimated_range = gap * 1.5 if gap > 0 else daily['high'].iloc[-1] - daily['low'].iloc[-1]
        
        if current_open > prior_close:
            overnight_high = current_open
            overnight_low = prior_close
        else:
            overnight_high = prior_close
            overnight_low = current_open
        
        return OvernightSession(
            session_date=target_date,
            overnight_high=overnight_high,
            overnight_low=overnight_low,
            overnight_open=prior_close,
            overnight_close=current_open,
            overnight_range=estimated_range,
            overnight_midpoint=(overnight_high + overnight_low) / 2,
            overnight_volume=0,
            overnight_delta=current_open - prior_close
        )
    
    def _estimate_session_delta(self, df: pd.DataFrame) -> float:
        """Estimate session delta from price action"""
        delta = 0
        for _, row in df.iterrows():
            bar_range = row['high'] - row['low']
            if bar_range == 0:
                continue
            clv = (row['close'] - row['low']) / bar_range
            delta += row['volume'] * (2 * clv - 1)
        return delta
    
    def _calculate_session_value_area(self, 
                                       df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate POC, VAH, VAL for a session"""
        if len(df) < 5:
            mid = df['close'].mean()
            return mid, mid, mid
        
        # Simple histogram approach
        price_min = df['low'].min()
        price_max = df['high'].max()
        num_bins = 20
        
        if price_max == price_min:
            return price_min, price_min, price_min
        
        bin_size = (price_max - price_min) / num_bins
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        volume_at_price = np.zeros(num_bins)
        
        for _, row in df.iterrows():
            typical = (row['high'] + row['low'] + row['close']) / 3
            bin_idx = min(num_bins - 1, max(0, int((typical - price_min) / bin_size)))
            volume_at_price[bin_idx] += row['volume']
        
        # POC
        poc_idx = np.argmax(volume_at_price)
        poc = bin_centers[poc_idx]
        
        # Value area (70%)
        total_vol = volume_at_price.sum()
        target_vol = total_vol * 0.70
        
        captured = volume_at_price[poc_idx]
        val_idx = poc_idx
        vah_idx = poc_idx
        
        while captured < target_vol and (val_idx > 0 or vah_idx < num_bins - 1):
            below = volume_at_price[val_idx - 1] if val_idx > 0 else 0
            above = volume_at_price[vah_idx + 1] if vah_idx < num_bins - 1 else 0
            
            if above >= below and vah_idx < num_bins - 1:
                vah_idx += 1
                captured += above
            elif val_idx > 0:
                val_idx -= 1
                captured += below
            else:
                vah_idx += 1
                captured += above
        
        return poc, bin_centers[vah_idx], bin_centers[val_idx]


# =============================================================================
# GAP ANALYZER
# =============================================================================

class GapAnalyzer:
    """
    Analyzes gaps between sessions
    
    For Brokers:
    -----------
    Gaps reveal overnight sentiment:
    - Large gaps (>1%): Strong conviction, often continue
    - Small gaps (0.3-1%): Moderate conviction, often fill
    - Gap and Go: Gap continues in direction
    - Gap and Fade: Gap fills back to prior close
    
    Historical gap fill rates vary by:
    - Gap size (larger gaps less likely to fill same day)
    - Market regime (trending vs mean-reverting)
    - Day of week (Monday gaps behave differently)
    
    For Programmers:
    ---------------
    Calculates gap metrics and uses historical patterns for fill probability.
    """
    
    # Gap size thresholds (percentage)
    LARGE_GAP_THRESHOLD = 1.0
    SMALL_GAP_THRESHOLD = 0.3
    
    # Historical gap fill rates (approximate)
    GAP_FILL_RATES = {
        "GAP_UP_LARGE": 0.45,
        "GAP_UP_SMALL": 0.72,
        "GAP_DOWN_LARGE": 0.48,
        "GAP_DOWN_SMALL": 0.70,
        "NO_GAP": 1.0
    }
    
    def __init__(self):
        pass
    
    def analyze_gap(self, 
                    prior_close: float, 
                    current_open: float,
                    atr: float) -> GapAnalysis:
        """
        Analyze the gap between prior close and current open
        
        Args:
            prior_close: Previous session close price
            current_open: Current session open price
            atr: Average True Range for normalization
        
        Returns:
            GapAnalysis object
        """
        # Gap metrics
        gap_size = current_open - prior_close
        gap_pct = (gap_size / prior_close * 100) if prior_close else 0
        gap_atr_ratio = abs(gap_size) / atr if atr else 0
        
        # Classify gap type
        gap_type = self._classify_gap(gap_pct)
        
        # Fill levels
        gap_fill_level = prior_close
        partial_fill_level = prior_close + (gap_size * 0.5)
        
        # Fill probability based on gap type
        fill_rate = self.GAP_FILL_RATES.get(gap_type.value, 0.5)
        
        # Adjust fill probability based on gap size relative to ATR
        if gap_atr_ratio > 2.0:
            fill_rate *= 0.7  # Very large gaps less likely to fill
        elif gap_atr_ratio < 0.5:
            fill_rate *= 1.2  # Small gaps more likely to fill
        
        fill_rate = min(1.0, max(0.0, fill_rate))
        
        if fill_rate >= 0.7:
            fill_prob = GapFillProbability.HIGH
        elif fill_rate >= 0.5:
            fill_prob = GapFillProbability.MODERATE
        else:
            fill_prob = GapFillProbability.LOW
        
        return GapAnalysis(
            gap_size=gap_size,
            gap_pct=gap_pct,
            gap_type=gap_type,
            gap_atr_ratio=gap_atr_ratio,
            prior_close=prior_close,
            current_open=current_open,
            gap_fill_level=gap_fill_level,
            gap_fill_probability=fill_prob,
            partial_fill_level=partial_fill_level,
            similar_gaps_filled_pct=fill_rate * 100
        )
    
    def _classify_gap(self, gap_pct: float) -> GapType:
        """Classify gap by size"""
        if gap_pct >= self.LARGE_GAP_THRESHOLD:
            return GapType.GAP_UP_LARGE
        elif gap_pct >= self.SMALL_GAP_THRESHOLD:
            return GapType.GAP_UP_SMALL
        elif gap_pct <= -self.LARGE_GAP_THRESHOLD:
            return GapType.GAP_DOWN_LARGE
        elif gap_pct <= -self.SMALL_GAP_THRESHOLD:
            return GapType.GAP_DOWN_SMALL
        else:
            return GapType.NO_GAP


# =============================================================================
# PRIOR DAY ANALYZER
# =============================================================================

class PriorDayAnalyzer:
    """
    Analyzes prior day for context
    
    For Brokers:
    -----------
    Prior day context sets expectations:
    - Trend day: Likely continuation or exhaustion
    - Range day: Breakout potential building
    - Reversal day: New direction establishing
    - Inside day: Compression, explosion coming
    - Outside day: Volatility, key reversal signal
    
    For Programmers:
    ---------------
    Classifies prior day type and extracts key levels.
    """
    
    def __init__(self):
        pass
    
    def analyze_prior_day(self, 
                          df: pd.DataFrame,
                          prior_poc: float,
                          prior_vah: float,
                          prior_val: float) -> Optional[PriorDayContext]:
        """
        Analyze prior trading day
        
        Args:
            df: OHLCV dataframe
            prior_poc/vah/val: Prior day value area levels
        
        Returns:
            PriorDayContext object
        """
        # Resample to daily
        daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(daily) < 2:
            return None
        
        prior = daily.iloc[-2]
        two_days_ago = daily.iloc[-3] if len(daily) >= 3 else prior
        
        prior_range = prior['high'] - prior['low']
        
        # Calculate ATR
        atr = self._calculate_atr(daily)
        
        # Classify day type
        day_type = self._classify_day(prior, two_days_ago, atr)
        
        return PriorDayContext(
            date=daily.index[-2],
            prior_open=prior['open'],
            prior_high=prior['high'],
            prior_low=prior['low'],
            prior_close=prior['close'],
            prior_range=prior_range,
            prior_poc=prior_poc,
            prior_vah=prior_vah,
            prior_val=prior_val,
            day_type=day_type,
            atr=atr
        )
    
    def _calculate_atr(self, daily: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from daily data"""
        if len(daily) < 2:
            return daily['high'].iloc[-1] - daily['low'].iloc[-1]
        
        high = daily['high']
        low = daily['low']
        close = daily['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(min(period, len(tr))).mean().iloc[-1]
        return atr if not np.isnan(atr) else tr.iloc[-1]
    
    def _classify_day(self, 
                      current: pd.Series, 
                      prior: pd.Series, 
                      atr: float) -> DayType:
        """Classify day type"""
        current_range = current['high'] - current['low']
        prior_range = prior['high'] - prior['low']
        
        # Inside day: current range within prior range
        if current['high'] <= prior['high'] and current['low'] >= prior['low']:
            return DayType.INSIDE_DAY
        
        # Outside day: current range engulfs prior range
        if current['high'] > prior['high'] and current['low'] < prior['low']:
            return DayType.OUTSIDE_DAY
        
        # Trend vs reversal
        open_close_move = current['close'] - current['open']
        range_pct = abs(open_close_move) / current_range if current_range else 0
        
        if range_pct > 0.6:  # Strong directional close
            if open_close_move > 0:
                # Check if reversal (opened near low, closed near high after down move)
                if current['open'] < prior['close'] and current['close'] > prior['close']:
                    return DayType.REVERSAL_UP
                return DayType.TREND_UP
            else:
                if current['open'] > prior['close'] and current['close'] < prior['close']:
                    return DayType.REVERSAL_DOWN
                return DayType.TREND_DOWN
        
        return DayType.RANGE_DAY


# =============================================================================
# OPENING CONTEXT ANALYZER
# =============================================================================

class OpeningContextAnalyzer:
    """
    Analyzes where price opens relative to key levels
    
    For Brokers:
    -----------
    Opening location is critical for daily bias:
    - Open above all levels: Strong bullish
    - Open below all levels: Strong bearish
    - Open inside range: Auction day, wait for direction
    - Open at key level: Test/rejection likely
    
    For Programmers:
    ---------------
    Compares opening price to multiple reference levels.
    """
    
    def analyze_opening(self,
                        open_price: float,
                        prior_day: PriorDayContext,
                        overnight: OvernightSession) -> OpeningContext:
        """
        Analyze opening price relative to key levels
        
        Returns:
            OpeningContext object
        """
        def compare(price: float, level: float, tolerance: float = 0.001) -> str:
            pct_diff = (price - level) / level if level else 0
            if pct_diff > tolerance:
                return "ABOVE"
            elif pct_diff < -tolerance:
                return "BELOW"
            return "AT"
        
        # Vs prior day
        vs_prior_close = compare(open_price, prior_day.prior_close)
        vs_prior_high = compare(open_price, prior_day.prior_high)
        vs_prior_low = compare(open_price, prior_day.prior_low)
        vs_prior_poc = compare(open_price, prior_day.prior_poc)
        vs_prior_vah = compare(open_price, prior_day.prior_vah)
        vs_prior_val = compare(open_price, prior_day.prior_val)
        
        # Vs overnight
        vs_overnight_high = compare(open_price, overnight.overnight_high)
        vs_overnight_low = compare(open_price, overnight.overnight_low)
        vs_overnight_midpoint = compare(open_price, overnight.overnight_midpoint)
        
        # Determine scenario
        if vs_prior_high == "ABOVE" and vs_overnight_high in ["ABOVE", "AT"]:
            scenario = "ABOVE_ALL_BULLISH"
        elif vs_prior_low == "BELOW" and vs_overnight_low in ["BELOW", "AT"]:
            scenario = "BELOW_ALL_BEARISH"
        elif vs_prior_vah == "ABOVE":
            scenario = "ABOVE_VALUE_BULLISH"
        elif vs_prior_val == "BELOW":
            scenario = "BELOW_VALUE_BEARISH"
        elif vs_prior_poc == "AT":
            scenario = "AT_POC_NEUTRAL"
        else:
            scenario = "IN_RANGE_NEUTRAL"
        
        return OpeningContext(
            open_price=open_price,
            vs_prior_close=vs_prior_close,
            vs_prior_high=vs_prior_high,
            vs_prior_low=vs_prior_low,
            vs_prior_poc=vs_prior_poc,
            vs_prior_vah=vs_prior_vah,
            vs_prior_val=vs_prior_val,
            vs_overnight_high=vs_overnight_high,
            vs_overnight_low=vs_overnight_low,
            vs_overnight_midpoint=vs_overnight_midpoint,
            scenario=scenario
        )


# =============================================================================
# OVERNIGHT PREDICTION ENGINE
# =============================================================================

class OvernightPredictionEngine:
    """
    Combines all overnight/gap analysis into predictive assessment
    
    For Brokers:
    -----------
    This engine synthesizes:
    1. Overnight session direction and levels
    2. Gap size and fill probability
    3. Prior day context
    4. Opening location
    
    Into actionable predictions:
    - Directional bias for the day
    - Key levels to watch
    - Bull/bear scenarios
    
    For Programmers:
    ---------------
    Orchestrates component analyzers and generates unified prediction.
    """
    
    def __init__(self):
        self.overnight_analyzer = OvernightAnalyzer()
        self.gap_analyzer = GapAnalyzer()
        self.prior_day_analyzer = PriorDayAnalyzer()
        self.opening_analyzer = OpeningContextAnalyzer()
    
    def predict(self, 
                df: pd.DataFrame, 
                symbol: str = "UNKNOWN",
                prior_poc: Optional[float] = None,
                prior_vah: Optional[float] = None,
                prior_val: Optional[float] = None) -> Optional[OvernightPrediction]:
        """
        Generate overnight/gap prediction
        
        Args:
            df: OHLCV dataframe with datetime index
            symbol: Ticker symbol
            prior_poc/vah/val: Prior day value area (if known)
        
        Returns:
            OvernightPrediction object
        """
        if len(df) < 20:
            return None
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        
        # Get components
        overnight = self.overnight_analyzer.extract_overnight_session(df)
        if overnight is None:
            return None
        
        # If prior value area not provided, estimate from data
        if prior_poc is None:
            # Use overnight POC or estimate
            prior_poc = overnight.overnight_poc or overnight.overnight_midpoint
            prior_vah = overnight.overnight_vah or overnight.overnight_high
            prior_val = overnight.overnight_val or overnight.overnight_low
        
        prior_day = self.prior_day_analyzer.analyze_prior_day(df, prior_poc, prior_vah, prior_val)
        if prior_day is None:
            return None
        
        # Gap analysis
        current_open = df['open'].iloc[-1]
        gap = self.gap_analyzer.analyze_gap(
            prior_day.prior_close, 
            current_open, 
            prior_day.atr
        )
        
        # Opening context
        opening = self.opening_analyzer.analyze_opening(current_open, prior_day, overnight)
        
        # Generate prediction
        bias, confidence, notes = self._generate_prediction(overnight, gap, prior_day, opening)
        
        # Key levels
        key_levels = self._compile_key_levels(overnight, prior_day, gap)
        
        # Scenarios
        bull_scenario, bear_scenario = self._generate_scenarios(gap, prior_day, overnight)
        
        return OvernightPrediction(
            symbol=symbol,
            analysis_time=datetime.now(),
            overnight=overnight,
            gap=gap,
            prior_day=prior_day,
            opening=opening,
            bias=bias,
            confidence=confidence,
            key_levels=key_levels,
            bull_scenario=bull_scenario,
            bear_scenario=bear_scenario,
            notes=notes
        )
    
    def _generate_prediction(self,
                             overnight: OvernightSession,
                             gap: GapAnalysis,
                             prior_day: PriorDayContext,
                             opening: OpeningContext) -> Tuple[OvernightBias, float, List[str]]:
        """Generate bias prediction from components"""
        bull_points = 0
        bear_points = 0
        notes = []
        
        # Gap influence
        if gap.gap_type in [GapType.GAP_UP_LARGE, GapType.GAP_UP_SMALL]:
            if gap.gap_fill_probability == GapFillProbability.LOW:
                bull_points += 30
                notes.append(f"Gap up {gap.gap_pct:.1f}% - low fill probability (Gap & Go)")
            else:
                bull_points += 15
                bear_points += 10
                notes.append(f"Gap up {gap.gap_pct:.1f}% - may fill to {gap.gap_fill_level:.2f}")
        elif gap.gap_type in [GapType.GAP_DOWN_LARGE, GapType.GAP_DOWN_SMALL]:
            if gap.gap_fill_probability == GapFillProbability.LOW:
                bear_points += 30
                notes.append(f"Gap down {gap.gap_pct:.1f}% - low fill probability (Gap & Go)")
            else:
                bear_points += 15
                bull_points += 10
                notes.append(f"Gap down {gap.gap_pct:.1f}% - may fill to {gap.gap_fill_level:.2f}")
        else:
            notes.append("No significant gap - balanced open")
        
        # Overnight direction
        if overnight.overnight_direction == "UP":
            bull_points += 15
            notes.append("Overnight session bullish")
        elif overnight.overnight_direction == "DOWN":
            bear_points += 15
            notes.append("Overnight session bearish")
        
        # Overnight delta (if significant)
        if overnight.overnight_delta > 0:
            bull_points += 10
            notes.append("Overnight buying pressure")
        elif overnight.overnight_delta < 0:
            bear_points += 10
            notes.append("Overnight selling pressure")
        
        # Opening scenario
        if "BULLISH" in opening.scenario:
            bull_points += 20
            notes.append(f"Opening scenario: {opening.scenario}")
        elif "BEARISH" in opening.scenario:
            bear_points += 20
            notes.append(f"Opening scenario: {opening.scenario}")
        else:
            notes.append(f"Opening scenario: {opening.scenario} - watch for direction")
        
        # Prior day context
        if prior_day.day_type in [DayType.TREND_UP, DayType.REVERSAL_UP]:
            bull_points += 10
            notes.append(f"Prior day: {prior_day.day_type.value}")
        elif prior_day.day_type in [DayType.TREND_DOWN, DayType.REVERSAL_DOWN]:
            bear_points += 10
            notes.append(f"Prior day: {prior_day.day_type.value}")
        elif prior_day.day_type == DayType.INSIDE_DAY:
            notes.append("Prior inside day - breakout potential")
        
        # Determine bias
        total = bull_points + bear_points
        if total == 0:
            return OvernightBias.NEUTRAL, 50, notes
        
        bull_pct = bull_points / (bull_points + bear_points) * 100
        
        if bull_pct >= 70:
            bias = OvernightBias.STRONG_BULLISH
        elif bull_pct >= 55:
            bias = OvernightBias.BULLISH
        elif bull_pct <= 30:
            bias = OvernightBias.STRONG_BEARISH
        elif bull_pct <= 45:
            bias = OvernightBias.BEARISH
        else:
            bias = OvernightBias.NEUTRAL
        
        confidence = abs(bull_pct - 50) * 2  # 0-100 scale
        
        return bias, confidence, notes
    
    def _compile_key_levels(self,
                            overnight: OvernightSession,
                            prior_day: PriorDayContext,
                            gap: GapAnalysis) -> Dict[str, float]:
        """Compile key levels for the day"""
        levels = {
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
            levels["overnight_poc"] = overnight.overnight_poc
        
        return levels
    
    def _generate_scenarios(self,
                            gap: GapAnalysis,
                            prior_day: PriorDayContext,
                            overnight: OvernightSession) -> Tuple[str, str]:
        """Generate bull and bear scenarios"""
        
        bull_scenario = f"""
        BULL SCENARIO:
        - Hold above {overnight.overnight_low:.2f} (overnight low)
        - Reclaim {prior_day.prior_val:.2f} (prior VAL) if below
        - Target {prior_day.prior_vah:.2f} (prior VAH)
        - Extended target: {prior_day.prior_high:.2f} (prior high)
        """
        
        bear_scenario = f"""
        BEAR SCENARIO:
        - Lose {overnight.overnight_low:.2f} (overnight low)
        - Break below {prior_day.prior_val:.2f} (prior VAL)
        - Target {prior_day.prior_low:.2f} (prior low)
        - Extended target: {prior_day.prior_low - prior_day.atr:.2f}
        """
        
        return bull_scenario.strip(), bear_scenario.strip()
    
    def print_prediction(self, pred: OvernightPrediction) -> str:
        """Format prediction for display"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"OVERNIGHT/GAP PREDICTION: {pred.symbol}")
        lines.append(f"Analysis Time: {pred.analysis_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        
        # Main prediction
        lines.append(f"\n{pred.bias.emoji} BIAS: {pred.bias.value} ({pred.confidence:.0f}% confidence)")
        
        # Gap info
        lines.append(f"\n📊 GAP ANALYSIS:")
        lines.append(f"   Type: {pred.gap.gap_type.emoji} {pred.gap.gap_type.value}")
        lines.append(f"   Size: {pred.gap.gap_size:+.2f} ({pred.gap.gap_pct:+.2f}%)")
        lines.append(f"   ATR Ratio: {pred.gap.gap_atr_ratio:.2f}x")
        lines.append(f"   Fill Probability: {pred.gap.gap_fill_probability.value}")
        lines.append(f"   Fill Level: ${pred.gap.gap_fill_level:.2f}")
        
        # Overnight session
        lines.append(f"\n🌙 OVERNIGHT SESSION:")
        lines.append(f"   Direction: {pred.overnight.overnight_direction}")
        lines.append(f"   Range: ${pred.overnight.overnight_low:.2f} - ${pred.overnight.overnight_high:.2f}")
        lines.append(f"   Midpoint: ${pred.overnight.overnight_midpoint:.2f}")
        
        # Opening context
        lines.append(f"\n🔔 OPENING CONTEXT:")
        lines.append(f"   Scenario: {pred.opening.scenario}")
        lines.append(f"   vs Prior Close: {pred.opening.vs_prior_close}")
        lines.append(f"   vs Prior POC: {pred.opening.vs_prior_poc}")
        lines.append(f"   vs Overnight Mid: {pred.opening.vs_overnight_midpoint}")
        
        # Key levels
        lines.append(f"\n📍 KEY LEVELS:")
        for name, level in sorted(pred.key_levels.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"   {name:<15}: ${level:.2f}")
        
        # Notes
        lines.append(f"\n📝 ANALYSIS NOTES:")
        for note in pred.notes:
            lines.append(f"   • {note}")
        
        # Scenarios
        lines.append(f"\n{pred.bull_scenario}")
        lines.append(f"\n{pred.bear_scenario}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

def generate_overnight_demo_data() -> pd.DataFrame:
    """Generate demo data with clear overnight session"""
    np.random.seed(123)
    
    data = []
    
    # 5 days of data, 5-min bars
    for day in range(5):
        base_date = datetime.now() - timedelta(days=5-day)
        
        # Prior close
        if day == 0:
            price = 450
        else:
            price = data[-1]['close']
        
        # Overnight session (6PM - 9:30AM) - simulate gap
        overnight_bias = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        overnight_move = overnight_bias * np.random.uniform(0.5, 2.0)
        
        # Evening session
        evening_start = datetime.combine(base_date.date() - timedelta(days=1), time(18, 0))
        for i in range(90):  # 7.5 hours = 90 5-min bars
            timestamp = evening_start + timedelta(minutes=i*5)
            
            trend = overnight_move * 0.005  # Gradual overnight move
            noise = np.random.randn() * 0.001
            
            open_p = price
            close_p = price * (1 + trend + noise)
            high_p = max(open_p, close_p) * 1.0005
            low_p = min(open_p, close_p) * 0.9995
            volume = int(np.random.exponential(30000))  # Low overnight volume
            
            data.append({
                'timestamp': timestamp,
                'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p,
                'volume': volume
            })
            price = close_p
        
        # RTH session (9:30AM - 4PM)
        rth_start = datetime.combine(base_date.date(), time(9, 30))
        for i in range(78):  # 6.5 hours = 78 5-min bars
            timestamp = rth_start + timedelta(minutes=i*5)
            
            # Higher volume, more volatility during RTH
            noise = np.random.randn() * 0.002
            
            open_p = price
            close_p = price * (1 + noise)
            high_p = max(open_p, close_p) * 1.001
            low_p = min(open_p, close_p) * 0.999
            volume = int(np.random.exponential(150000))
            
            data.append({
                'timestamp': timestamp,
                'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p,
                'volume': volume
            })
            price = close_p
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


if __name__ == "__main__":
    print("Generating demo data with overnight sessions...")
    df = generate_overnight_demo_data()
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Run prediction
    engine = OvernightPredictionEngine()
    prediction = engine.predict(df, symbol="DEMO")
    
    if prediction:
        report = engine.print_prediction(prediction)
        print(report)
    else:
        print("Could not generate prediction")
