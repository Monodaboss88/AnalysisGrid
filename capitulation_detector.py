"""
Capitulation Detector
=====================
Detects market capitulation conditions - ideal for catching bottoms.

Capitulation occurs when:
1. Price has declined significantly (20%+ from recent high)
2. Volume spikes on final selloff (selling climax)
3. Volume then dries up (sellers exhausted)
4. RSI deeply oversold (<30, ideally <25)
5. Reversal candle appears (hammer, bullish engulfing)

This is WHERE the big moves start - when everyone has given up.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CapitulationLevel(Enum):
    """Capitulation intensity levels"""
    NONE = "NONE"                    # No capitulation signs
    EARLY = "EARLY"                  # Starting to show stress
    DEVELOPING = "DEVELOPING"        # Multiple signs aligning
    CLIMAX = "CLIMAX"               # Active selling climax
    EXHAUSTION = "EXHAUSTION"        # Sellers exhausted - ideal entry
    
    @property
    def score(self) -> int:
        return {
            "NONE": 0,
            "EARLY": 25,
            "DEVELOPING": 50,
            "CLIMAX": 75,
            "EXHAUSTION": 100
        }[self.value]
    
    @property
    def tradeable(self) -> bool:
        """Is this a tradeable setup?"""
        return self in [CapitulationLevel.EXHAUSTION, CapitulationLevel.CLIMAX]


class EuphoriaLevel(Enum):
    """Euphoria intensity levels (mirror of CapitulationLevel for shorts)"""
    NONE = "NONE"                    # No euphoria signs
    EARLY = "EARLY"                  # Starting to show froth
    DEVELOPING = "DEVELOPING"        # Multiple signs aligning
    CLIMAX = "CLIMAX"               # Active buying climax / blow-off top
    EXHAUSTION = "EXHAUSTION"        # Buyers exhausted - ideal SHORT entry
    
    @property
    def score(self) -> int:
        return {
            "NONE": 0,
            "EARLY": 25,
            "DEVELOPING": 50,
            "CLIMAX": 75,
            "EXHAUSTION": 100
        }[self.value]
    
    @property
    def tradeable(self) -> bool:
        """Is this a tradeable setup?"""
        return self in [EuphoriaLevel.EXHAUSTION, EuphoriaLevel.CLIMAX]


@dataclass
class CapitulationMetrics:
    """Detailed capitulation analysis"""
    # Price decline
    decline_from_high_pct: float      # % decline from recent high
    days_since_high: int              # How long ago was the high
    
    # Volume analysis
    current_rvol: float               # Current relative volume
    climax_volume_detected: bool      # Did we see a volume spike on selloff
    volume_exhaustion: bool           # Has volume dried up after climax
    avg_down_volume_ratio: float      # Ratio of volume on down days vs up days
    
    # RSI
    rsi: float
    rsi_oversold: bool
    rsi_extreme: bool                 # RSI < 25
    rsi_divergence: bool              # Price LL, RSI HL
    
    # Candle patterns
    reversal_candle: bool             # Hammer, bullish engulfing
    long_lower_wick: bool             # Strong buyer rejection
    
    # NEW: Additional factors
    consecutive_down_days: int        # Multi-day selloff pattern
    at_support_level: bool            # Near 200 SMA, prior pivot, etc.
    session_context: str              # 'close', 'open', 'mid-session'
    
    # Composite
    capitulation_score: int           # 0-100
    capitulation_level: CapitulationLevel
    
    # Trade info
    entry_zone: Tuple[float, float]
    stop_loss: float
    target_1: float
    target_2: float


@dataclass
class EuphoriaMetrics:
    """Detailed euphoria analysis (mirror of CapitulationMetrics for shorts)"""
    # Price advance
    advance_from_low_pct: float       # % advance from recent low
    days_since_low: int               # How long ago was the low
    
    # Volume analysis
    current_rvol: float               # Current relative volume
    climax_volume_detected: bool      # Volume spike on rally (blow-off top)
    volume_exhaustion: bool           # Has volume dried up after climax
    avg_up_volume_ratio: float        # Ratio of volume on up days vs down days
    
    # RSI
    rsi: float
    rsi_overbought: bool              # RSI > 70
    rsi_extreme: bool                 # RSI > 80
    rsi_divergence: bool              # Price HH, RSI LH (bearish divergence)
    
    # Candle patterns
    reversal_candle: bool             # Shooting star, bearish engulfing
    long_upper_wick: bool             # Strong seller rejection
    
    # Additional factors
    consecutive_up_days: int          # Multi-day rally pattern
    at_resistance_level: bool         # Near prior high, 200 SMA above, etc.
    session_context: str              # 'close', 'open', 'mid-session'
    
    # Composite
    euphoria_score: int               # 0-100
    euphoria_level: EuphoriaLevel
    
    # Trade info (SHORT)
    entry_zone: Tuple[float, float]
    stop_loss: float                  # Above recent high
    target_1: float                   # First target down
    target_2: float                   # Extended target


class CapitulationDetector:
    """
    Detects capitulation conditions across multiple factors.
    
    For Traders:
    -----------
    This finds the setups where "everyone has given up" - 
    the best risk/reward entries. When you see EXHAUSTION level,
    that's when the smart money is stepping in.
    
    Key signals:
    - Price down 20%+ from high
    - Volume climax followed by dry-up
    - RSI < 25 with divergence
    - Reversal candle confirmation
    
    For Programmers:
    ---------------
    Calculates composite score from:
    - Price decline magnitude (25 pts)
    - Volume pattern (25 pts)  
    - RSI condition (25 pts)
    - Candle pattern (25 pts)
    """
    
    def __init__(self,
                 min_decline_pct: float = 10.0,      # Lowered for large caps
                 ideal_decline_pct: float = 20.0,    # Adjusted from 25%
                 rsi_oversold: float = 30.0,
                 rsi_extreme: float = 25.0,
                 volume_climax_mult: float = 2.5,    # RVOL for climax
                 volume_exhaustion_mult: float = 0.6, # RVOL for exhaustion (tightened)
                 lookback_days: int = 30):
        
        self.min_decline_pct = min_decline_pct
        self.ideal_decline_pct = ideal_decline_pct
        self.rsi_oversold = rsi_oversold
        self.rsi_extreme = rsi_extreme
        self.volume_climax_mult = volume_climax_mult
        self.volume_exhaustion_mult = volume_exhaustion_mult
        self.lookback_days = lookback_days
    
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> CapitulationMetrics:
        """
        Analyze for capitulation conditions
        
        Args:
            df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
            symbol: Ticker symbol
        
        Returns:
            CapitulationMetrics with complete analysis
        """
        # Normalize column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if len(df) < 20:
            return self._default_metrics()
        
        current_price = df['close'].iloc[-1]
        
        # 1. PRICE DECLINE ANALYSIS
        decline_pct, days_since_high, recent_high = self._analyze_decline(df)
        
        # 2. VOLUME ANALYSIS
        rvol, climax_detected, exhaustion, down_vol_ratio = self._analyze_volume(df)
        
        # 3. RSI ANALYSIS
        rsi, oversold, extreme, divergence = self._analyze_rsi(df)
        
        # 4. CANDLE PATTERN ANALYSIS
        reversal_candle, long_wick = self._analyze_candles(df)
        
        # 5. NEW: ADDITIONAL FACTORS
        consecutive_down = self._count_consecutive_down_days(df)
        at_support = self._check_support_confluence(df, current_price)
        session_ctx = self._get_session_context(df)
        
        # 6. CALCULATE COMPOSITE SCORE (with new factors)
        score = self._calculate_score(
            decline_pct, rvol, climax_detected, exhaustion,
            rsi, oversold, extreme, divergence,
            reversal_candle, long_wick,
            consecutive_down, at_support, session_ctx
        )
        
        # 7. DETERMINE LEVEL (fixed logic)
        level = self._determine_level(score, exhaustion, climax_detected)
        
        # 8. CALCULATE TRADE LEVELS
        atr = self._calculate_atr(df)
        entry_zone = (current_price * 0.995, current_price * 1.005)
        stop_loss = current_price - (atr * 1.5)
        target_1 = current_price + (atr * 2)
        target_2 = current_price + (recent_high - current_price) * 0.382  # 38.2% retrace of decline
        
        return CapitulationMetrics(
            decline_from_high_pct=round(decline_pct, 2),
            days_since_high=days_since_high,
            current_rvol=round(rvol, 2),
            climax_volume_detected=climax_detected,
            volume_exhaustion=exhaustion,
            avg_down_volume_ratio=round(down_vol_ratio, 2),
            rsi=round(rsi, 2),
            rsi_oversold=oversold,
            rsi_extreme=extreme,
            rsi_divergence=divergence,
            reversal_candle=reversal_candle,
            long_lower_wick=long_wick,
            consecutive_down_days=consecutive_down,
            at_support_level=at_support,
            session_context=session_ctx,
            capitulation_score=score,
            capitulation_level=level,
            entry_zone=entry_zone,
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2)
        )
    
    def _analyze_decline(self, df: pd.DataFrame) -> Tuple[float, int, float]:
        """Analyze price decline from recent high"""
        lookback = min(len(df), self.lookback_days * 8)  # Assuming hourly data
        recent = df.tail(lookback)
        
        recent_high = recent['high'].max()
        high_idx = recent['high'].idxmax()
        current_price = df['close'].iloc[-1]
        
        decline_pct = ((recent_high - current_price) / recent_high) * 100
        
        # Calculate days since high
        try:
            days_since = (df.index[-1] - high_idx).days
        except:
            days_since = len(df) - df.index.get_loc(high_idx)
        
        return decline_pct, days_since, recent_high
    
    def _analyze_volume(self, df: pd.DataFrame) -> Tuple[float, bool, bool, float]:
        """
        Analyze volume patterns for climax and exhaustion.
        
        Key insight: The SEQUENCE matters.
        Climax = RVOL ≥ 2.5x on a DOWN day
        Exhaustion = Climax followed by 1-3 bars of RVOL < 0.6x
        """
        if 'volume' not in df.columns:
            return 1.0, False, False, 1.0
        
        # Current relative volume
        avg_vol = df['volume'].rolling(20).mean()
        current_rvol = df['volume'].iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1.0
        
        # Calculate RVOL for each bar
        rvol_series = df['volume'] / avg_vol
        
        # Identify down bars
        df_temp = df.copy()
        df_temp['is_down'] = df_temp['close'] < df_temp['open']
        df_temp['rvol'] = rvol_series
        
        # Look for volume climax on DOWN day in recent 10 bars
        recent = df_temp.tail(10)
        climax_detected = False
        climax_bar_idx = None
        
        for i in range(len(recent) - 1, -1, -1):
            bar = recent.iloc[i]
            if bar['is_down'] and bar['rvol'] >= self.volume_climax_mult:
                climax_detected = True
                climax_bar_idx = i
                break
        
        # Check for exhaustion: bars AFTER climax have dried up (RVOL < 0.6x)
        exhaustion = False
        if climax_detected and climax_bar_idx is not None:
            # Check 1-3 bars after climax
            bars_after_climax = len(recent) - 1 - climax_bar_idx
            if bars_after_climax >= 1:
                post_climax = recent.iloc[climax_bar_idx + 1:]
                if len(post_climax) > 0:
                    # All post-climax bars should have low volume
                    post_climax_rvol = post_climax['rvol'].mean()
                    exhaustion = post_climax_rvol < self.volume_exhaustion_mult
        
        # Volume ratio on down days vs up days (overall)
        down_vol = df[df_temp['is_down']]['volume'].mean() if df_temp['is_down'].any() else 0
        up_vol = df[~df_temp['is_down']]['volume'].mean() if (~df_temp['is_down']).any() else 1
        down_vol_ratio = down_vol / up_vol if up_vol > 0 else 1.0
        
        return current_rvol, climax_detected, exhaustion, down_vol_ratio
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Tuple[float, bool, bool, bool]:
        """Analyze RSI conditions"""
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        oversold = current_rsi < self.rsi_oversold
        extreme = current_rsi < self.rsi_extreme
        
        # Check for bullish divergence
        divergence = self._check_divergence(df, rsi)
        
        return current_rsi, oversold, extreme, divergence
    
    def _check_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """Check for bullish RSI divergence"""
        if len(df) < 10:
            return False
        
        prices = df['low'].tail(10)
        rsi_vals = rsi.tail(10)
        
        # Find lows
        mid = len(prices) // 2
        first_price_low = prices.iloc[:mid].min()
        second_price_low = prices.iloc[mid:].min()
        
        first_rsi_low = rsi_vals.iloc[:mid].min()
        second_rsi_low = rsi_vals.iloc[mid:].min()
        
        # Divergence: price lower low, RSI higher low
        return second_price_low < first_price_low and second_rsi_low > first_rsi_low
    
    def _analyze_candles(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Analyze candlestick patterns"""
        if len(df) < 2:
            return False, False
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['close'] - last['open'])
        full_range = last['high'] - last['low']
        lower_wick = min(last['open'], last['close']) - last['low']
        
        # Long lower wick (hammer-like)
        long_wick = lower_wick > body * 1.5 and lower_wick > full_range * 0.4
        
        # Bullish reversal candle
        bullish_candle = last['close'] > last['open']
        
        # Bullish engulfing
        engulfing = (
            prev['close'] < prev['open'] and  # Prev was red
            last['close'] > last['open'] and  # Current is green
            last['close'] > prev['open'] and  # Close above prev open
            last['open'] < prev['close']      # Open below prev close
        )
        
        reversal = (bullish_candle and long_wick) or engulfing
        
        return reversal, long_wick
    
    def _count_consecutive_down_days(self, df: pd.DataFrame) -> int:
        """Count consecutive down days/bars before current bar"""
        count = 0
        for i in range(len(df) - 2, max(0, len(df) - 10), -1):
            if df['close'].iloc[i] < df['open'].iloc[i]:
                count += 1
            else:
                break
        return count
    
    def _check_support_confluence(self, df: pd.DataFrame, current_price: float) -> bool:
        """Check if price is at a support level (200 SMA, prior pivot, etc.)"""
        if len(df) < 50:
            return False
        
        # Calculate 200-period SMA (or use available data)
        sma_period = min(200, len(df))
        sma = df['close'].rolling(sma_period).mean().iloc[-1]
        
        # Check if within 2% of SMA
        near_sma = abs(current_price - sma) / sma < 0.02
        
        # Check if at prior swing low (lowest low in last 20 bars)
        recent_low = df['low'].tail(20).min()
        near_swing_low = abs(current_price - recent_low) / recent_low < 0.01
        
        return near_sma or near_swing_low
    
    def _get_session_context(self, df: pd.DataFrame) -> str:
        """Determine session context (more meaningful at close)"""
        if len(df) < 2:
            return 'unknown'
        
        # Check if last bar is near session end (approximation)
        try:
            last_hour = df.index[-1].hour
            if last_hour >= 15:  # After 3 PM
                return 'close'
            elif last_hour <= 10:  # Before 10 AM
                return 'open'
            else:
                return 'mid-session'
        except:
            return 'unknown'
    
    def _calculate_score(self, 
                        decline_pct: float,
                        rvol: float,
                        climax: bool,
                        exhaustion: bool,
                        rsi: float,
                        oversold: bool,
                        extreme: bool,
                        divergence: bool,
                        reversal_candle: bool,
                        long_wick: bool,
                        consecutive_down: int,
                        at_support: bool,
                        session_ctx: str) -> int:
        """
        Calculate composite capitulation score (0-100+)
        
        Base factors: 100 pts max
        Bonus factors: up to +25 pts
        """
        score = 0
        
        # PRICE DECLINE (25 points max)
        # Adjusted: 10% = significant for large caps, 20% = full capitulation
        if decline_pct >= self.ideal_decline_pct:
            score += 25
        elif decline_pct >= self.min_decline_pct:
            score += int(15 + (decline_pct - self.min_decline_pct) / 
                        (self.ideal_decline_pct - self.min_decline_pct) * 10)
        elif decline_pct >= 7:
            score += 10
        
        # VOLUME (25 points max)
        # The SEQUENCE is key: climax THEN exhaustion
        if exhaustion:
            score += 25  # Ideal: climax followed by dry-up
        elif climax:
            score += 15  # Climax but not yet exhausted
        elif rvol < 0.6:
            score += 8   # Low volume (could be drying up)
        
        # RSI (25 points max)
        if extreme:
            score += 20
        elif oversold:
            score += 15
        elif rsi < 35:
            score += 10
        
        if divergence:
            score += 5  # Bonus for divergence
        
        # CANDLE PATTERN (25 points max)
        if reversal_candle:
            score += 25
        elif long_wick:
            score += 15
        
        # ========== NEW BONUS FACTORS ==========
        
        # MULTI-DAY SELLOFF PATTERN (+10)
        # 3+ consecutive down days before reversal = more meaningful
        if consecutive_down >= 3:
            score += 10
        elif consecutive_down >= 2:
            score += 5
        
        # SUPPORT CONFLUENCE (+10)
        # Exhaustion AT a level (200 SMA, prior pivot) is higher quality
        if at_support:
            score += 10
        
        # SESSION CONTEXT (+5-10)
        # Exhaustion into session close = more meaningful
        if session_ctx == 'close':
            score += 10
        elif session_ctx == 'open':
            score += 5  # Opening capitulation can be strong too
        
        return min(120, score)  # Cap at 120 (100 base + 20 bonus)
    
    def _determine_level(self, score: int, exhaustion: bool, climax: bool) -> CapitulationLevel:
        """
        Determine capitulation level from score and conditions.
        
        FIXED: CLIMAX now requires score >= 60 AND climax detected
        (previously OR logic caused noise from single volume spikes)
        """
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
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def _default_metrics(self) -> CapitulationMetrics:
        """Return default metrics for insufficient data"""
        return CapitulationMetrics(
            decline_from_high_pct=0.0,
            days_since_high=0,
            current_rvol=1.0,
            climax_volume_detected=False,
            volume_exhaustion=False,
            avg_down_volume_ratio=1.0,
            rsi=50.0,
            rsi_oversold=False,
            rsi_extreme=False,
            rsi_divergence=False,
            reversal_candle=False,
            long_lower_wick=False,
            consecutive_down_days=0,
            at_support_level=False,
            session_context='unknown',
            capitulation_score=0,
            capitulation_level=CapitulationLevel.NONE,
            entry_zone=(0, 0),
            stop_loss=0,
            target_1=0,
            target_2=0
        )

    # =========================================================================
    # EUPHORIA DETECTION (Short-side logic - mirror of capitulation)
    # =========================================================================
    
    def analyze_euphoria(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> EuphoriaMetrics:
        """
        Analyze for euphoria/blow-off top conditions (SHORT setups)
        
        Euphoria occurs when:
        1. Price has advanced significantly (20%+ from recent low)
        2. Volume spikes on final rally (buying climax)
        3. Volume then dries up (buyers exhausted)
        4. RSI deeply overbought (>75, ideally >80)
        5. Reversal candle appears (shooting star, bearish engulfing)
        
        This is WHERE the short setups start - when everyone is euphoric.
        
        Args:
            df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
            symbol: Ticker symbol
        
        Returns:
            EuphoriaMetrics with complete analysis
        """
        # Normalize column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if len(df) < 20:
            return self._default_euphoria_metrics()
        
        current_price = df['close'].iloc[-1]
        
        # 1. PRICE ADVANCE ANALYSIS
        advance_pct, days_since_low, recent_low = self._analyze_advance(df)
        
        # 2. VOLUME ANALYSIS (buying climax)
        rvol, climax_detected, exhaustion, up_vol_ratio = self._analyze_volume_euphoria(df)
        
        # 3. RSI ANALYSIS
        rsi, overbought, extreme, divergence = self._analyze_rsi_euphoria(df)
        
        # 4. CANDLE PATTERN ANALYSIS (bearish)
        reversal_candle, long_upper_wick = self._analyze_candles_bearish(df)
        
        # 5. ADDITIONAL FACTORS
        consecutive_up = self._count_consecutive_up_days(df)
        at_resistance = self._check_resistance_confluence(df, current_price)
        session_ctx = self._get_session_context(df)
        
        # 6. CALCULATE COMPOSITE SCORE
        score = self._calculate_euphoria_score(
            advance_pct, rvol, climax_detected, exhaustion,
            rsi, overbought, extreme, divergence,
            reversal_candle, long_upper_wick,
            consecutive_up, at_resistance, session_ctx
        )
        
        # 7. DETERMINE LEVEL
        level = self._determine_euphoria_level(score, exhaustion, climax_detected)
        
        # 8. CALCULATE TRADE LEVELS (SHORT)
        atr = self._calculate_atr(df)
        entry_zone = (current_price * 0.995, current_price * 1.005)
        stop_loss = current_price + (atr * 1.5)  # Stop ABOVE for shorts
        target_1 = current_price - (atr * 2)     # First target DOWN
        target_2 = current_price - (current_price - recent_low) * 0.382  # 38.2% retrace of advance
        
        return EuphoriaMetrics(
            advance_from_low_pct=round(advance_pct, 2),
            days_since_low=days_since_low,
            current_rvol=round(rvol, 2),
            climax_volume_detected=climax_detected,
            volume_exhaustion=exhaustion,
            avg_up_volume_ratio=round(up_vol_ratio, 2),
            rsi=round(rsi, 2),
            rsi_overbought=overbought,
            rsi_extreme=extreme,
            rsi_divergence=divergence,
            reversal_candle=reversal_candle,
            long_upper_wick=long_upper_wick,
            consecutive_up_days=consecutive_up,
            at_resistance_level=at_resistance,
            session_context=session_ctx,
            euphoria_score=score,
            euphoria_level=level,
            entry_zone=entry_zone,
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2)
        )
    
    def _analyze_advance(self, df: pd.DataFrame) -> Tuple[float, int, float]:
        """Analyze price advance from recent low"""
        lookback = min(len(df), self.lookback_days * 8)
        recent = df.tail(lookback)
        
        recent_low = recent['low'].min()
        low_idx = recent['low'].idxmin()
        current_price = df['close'].iloc[-1]
        
        advance_pct = ((current_price - recent_low) / recent_low) * 100
        
        try:
            days_since = (df.index[-1] - low_idx).days
        except:
            days_since = len(df) - df.index.get_loc(low_idx)
        
        return advance_pct, days_since, recent_low
    
    def _analyze_volume_euphoria(self, df: pd.DataFrame) -> Tuple[float, bool, bool, float]:
        """
        Analyze volume patterns for buying climax and exhaustion.
        
        Climax = RVOL ≥ 2.5x on an UP day
        Exhaustion = Climax followed by 1-3 bars of RVOL < 0.6x
        """
        if 'volume' not in df.columns:
            return 1.0, False, False, 1.0
        
        avg_vol = df['volume'].rolling(20).mean()
        current_rvol = df['volume'].iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1.0
        
        rvol_series = df['volume'] / avg_vol
        
        df_temp = df.copy()
        df_temp['is_up'] = df_temp['close'] > df_temp['open']
        df_temp['rvol'] = rvol_series
        
        # Look for volume climax on UP day in recent 10 bars
        recent = df_temp.tail(10)
        climax_detected = False
        climax_bar_idx = None
        
        for i in range(len(recent) - 1, -1, -1):
            bar = recent.iloc[i]
            if bar['is_up'] and bar['rvol'] >= self.volume_climax_mult:
                climax_detected = True
                climax_bar_idx = i
                break
        
        # Check for exhaustion: bars AFTER climax have dried up
        exhaustion = False
        if climax_detected and climax_bar_idx is not None:
            bars_after_climax = len(recent) - 1 - climax_bar_idx
            if bars_after_climax >= 1:
                post_climax = recent.iloc[climax_bar_idx + 1:]
                if len(post_climax) > 0:
                    post_climax_rvol = post_climax['rvol'].mean()
                    exhaustion = post_climax_rvol < self.volume_exhaustion_mult
        
        # Volume ratio on up days vs down days
        up_vol = df[df_temp['is_up']]['volume'].mean() if df_temp['is_up'].any() else 0
        down_vol = df[~df_temp['is_up']]['volume'].mean() if (~df_temp['is_up']).any() else 1
        up_vol_ratio = up_vol / down_vol if down_vol > 0 else 1.0
        
        return current_rvol, climax_detected, exhaustion, up_vol_ratio
    
    def _analyze_rsi_euphoria(self, df: pd.DataFrame) -> Tuple[float, bool, bool, bool]:
        """Analyze RSI conditions for overbought"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        overbought = current_rsi > 70
        extreme = current_rsi > 80
        
        # Check for bearish divergence (price HH, RSI LH)
        divergence = self._check_bearish_divergence(df, rsi)
        
        return current_rsi, overbought, extreme, divergence
    
    def _check_bearish_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """Check for bearish RSI divergence (price HH, RSI LH)"""
        if len(df) < 10:
            return False
        
        prices = df['high'].tail(10)
        rsi_vals = rsi.tail(10)
        
        mid = len(prices) // 2
        first_price_high = prices.iloc[:mid].max()
        second_price_high = prices.iloc[mid:].max()
        
        first_rsi_high = rsi_vals.iloc[:mid].max()
        second_rsi_high = rsi_vals.iloc[mid:].max()
        
        # Bearish divergence: price higher high, RSI lower high
        return second_price_high > first_price_high and second_rsi_high < first_rsi_high
    
    def _analyze_candles_bearish(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Analyze candlestick patterns for bearish reversal"""
        if len(df) < 2:
            return False, False
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['close'] - last['open'])
        full_range = last['high'] - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        
        # Long upper wick (shooting star-like)
        long_wick = upper_wick > body * 1.5 and upper_wick > full_range * 0.4
        
        # Bearish reversal candle
        bearish_candle = last['close'] < last['open']
        
        # Bearish engulfing
        engulfing = (
            prev['close'] > prev['open'] and  # Prev was green
            last['close'] < last['open'] and  # Current is red
            last['open'] > prev['close'] and  # Open above prev close
            last['close'] < prev['open']      # Close below prev open
        )
        
        reversal = (bearish_candle and long_wick) or engulfing
        
        return reversal, long_wick
    
    def _count_consecutive_up_days(self, df: pd.DataFrame) -> int:
        """Count consecutive up days/bars before current bar"""
        count = 0
        for i in range(len(df) - 2, max(0, len(df) - 10), -1):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                count += 1
            else:
                break
        return count
    
    def _check_resistance_confluence(self, df: pd.DataFrame, current_price: float) -> bool:
        """Check if price is at a resistance level"""
        if len(df) < 50:
            return False
        
        # Check if at prior swing high
        recent_high = df['high'].tail(20).max()
        near_swing_high = abs(current_price - recent_high) / recent_high < 0.01
        
        # Check all-time high in lookback
        all_time_high = df['high'].max()
        near_ath = abs(current_price - all_time_high) / all_time_high < 0.02
        
        return near_swing_high or near_ath
    
    def _calculate_euphoria_score(self,
                                   advance_pct: float,
                                   rvol: float,
                                   climax: bool,
                                   exhaustion: bool,
                                   rsi: float,
                                   overbought: bool,
                                   extreme: bool,
                                   divergence: bool,
                                   reversal_candle: bool,
                                   long_wick: bool,
                                   consecutive_up: int,
                                   at_resistance: bool,
                                   session_ctx: str) -> int:
        """Calculate composite euphoria score (0-100+)"""
        score = 0
        
        # PRICE ADVANCE (25 points max)
        if advance_pct >= self.ideal_decline_pct:  # Use same threshold
            score += 25
        elif advance_pct >= self.min_decline_pct:
            score += int(15 + (advance_pct - self.min_decline_pct) / 
                        (self.ideal_decline_pct - self.min_decline_pct) * 10)
        elif advance_pct >= 7:
            score += 10
        
        # VOLUME (25 points max)
        if exhaustion:
            score += 25
        elif climax:
            score += 15
        elif rvol < 0.6:
            score += 8
        
        # RSI (25 points max)
        if extreme:
            score += 20
        elif overbought:
            score += 15
        elif rsi > 65:
            score += 10
        
        if divergence:
            score += 5
        
        # CANDLE PATTERN (25 points max)
        if reversal_candle:
            score += 25
        elif long_wick:
            score += 15
        
        # BONUS: MULTI-DAY RALLY (+10)
        if consecutive_up >= 3:
            score += 10
        elif consecutive_up >= 2:
            score += 5
        
        # BONUS: RESISTANCE CONFLUENCE (+10)
        if at_resistance:
            score += 10
        
        # BONUS: SESSION CONTEXT (+5-10)
        if session_ctx == 'close':
            score += 10
        elif session_ctx == 'open':
            score += 5
        
        return min(120, score)
    
    def _determine_euphoria_level(self, score: int, exhaustion: bool, climax: bool) -> EuphoriaLevel:
        """Determine euphoria level from score and conditions"""
        if score >= 80 and exhaustion:
            return EuphoriaLevel.EXHAUSTION
        elif score >= 60 and climax:
            return EuphoriaLevel.CLIMAX
        elif score >= 45:
            return EuphoriaLevel.DEVELOPING
        elif score >= 25:
            return EuphoriaLevel.EARLY
        else:
            return EuphoriaLevel.NONE
    
    def _default_euphoria_metrics(self) -> EuphoriaMetrics:
        """Return default euphoria metrics for insufficient data"""
        return EuphoriaMetrics(
            advance_from_low_pct=0.0,
            days_since_low=0,
            current_rvol=1.0,
            climax_volume_detected=False,
            volume_exhaustion=False,
            avg_up_volume_ratio=1.0,
            rsi=50.0,
            rsi_overbought=False,
            rsi_extreme=False,
            rsi_divergence=False,
            reversal_candle=False,
            long_upper_wick=False,
            consecutive_up_days=0,
            at_resistance_level=False,
            session_context='unknown',
            euphoria_score=0,
            euphoria_level=EuphoriaLevel.NONE,
            entry_zone=(0, 0),
            stop_loss=0,
            target_1=0,
            target_2=0
        )


def format_capitulation_alert(metrics: CapitulationMetrics, symbol: str) -> str:
    """Format capitulation analysis as alert message"""
    lines = []
    
    emoji = {
        CapitulationLevel.NONE: "⚪",
        CapitulationLevel.EARLY: "🟡",
        CapitulationLevel.DEVELOPING: "🟠",
        CapitulationLevel.CLIMAX: "🔴",
        CapitulationLevel.EXHAUSTION: "🟢"
    }[metrics.capitulation_level]
    
    lines.append(f"{emoji} {symbol} CAPITULATION ANALYSIS")
    lines.append(f"Level: {metrics.capitulation_level.value} | Score: {metrics.capitulation_score}/100")
    lines.append("")
    
    lines.append("📉 DECLINE:")
    lines.append(f"   Down {metrics.decline_from_high_pct:.1f}% from high ({metrics.days_since_high} days ago)")
    
    lines.append("📊 VOLUME:")
    lines.append(f"   Current RVOL: {metrics.current_rvol:.1f}x")
    lines.append(f"   Climax Detected: {'✓' if metrics.climax_volume_detected else '✗'}")
    lines.append(f"   Volume Exhaustion: {'✓' if metrics.volume_exhaustion else '✗'}")
    
    lines.append("📈 RSI:")
    lines.append(f"   RSI: {metrics.rsi:.1f} {'⚠️ EXTREME' if metrics.rsi_extreme else '(oversold)' if metrics.rsi_oversold else ''}")
    lines.append(f"   Divergence: {'✓ BULLISH' if metrics.rsi_divergence else '✗'}")
    
    lines.append("🕯️ CANDLES:")
    lines.append(f"   Reversal Candle: {'✓' if metrics.reversal_candle else '✗'}")
    lines.append(f"   Long Lower Wick: {'✓' if metrics.long_lower_wick else '✗'}")
    
    if metrics.capitulation_level.tradeable:
        lines.append("")
        lines.append("💰 TRADE SETUP:")
        lines.append(f"   Entry Zone: ${metrics.entry_zone[0]:.2f} - ${metrics.entry_zone[1]:.2f}")
        lines.append(f"   Stop Loss: ${metrics.stop_loss:.2f}")
        lines.append(f"   Target 1: ${metrics.target_1:.2f}")
        lines.append(f"   Target 2: ${metrics.target_2:.2f}")
    
    return "\n".join(lines)


def format_euphoria_alert(metrics: EuphoriaMetrics, symbol: str) -> str:
    """Format euphoria analysis as alert message (SHORT setup)"""
    lines = []
    
    emoji = {
        EuphoriaLevel.NONE: "⚪",
        EuphoriaLevel.EARLY: "🟡",
        EuphoriaLevel.DEVELOPING: "🟠",
        EuphoriaLevel.CLIMAX: "🔴",
        EuphoriaLevel.EXHAUSTION: "🟢"
    }[metrics.euphoria_level]
    
    lines.append(f"{emoji} {symbol} EUPHORIA ANALYSIS (SHORT)")
    lines.append(f"Level: {metrics.euphoria_level.value} | Score: {metrics.euphoria_score}/100")
    lines.append("")
    
    lines.append("📈 ADVANCE:")
    lines.append(f"   Up {metrics.advance_from_low_pct:.1f}% from low ({metrics.days_since_low} days ago)")
    
    lines.append("📊 VOLUME:")
    lines.append(f"   Current RVOL: {metrics.current_rvol:.1f}x")
    lines.append(f"   Buying Climax: {'✓' if metrics.climax_volume_detected else '✗'}")
    lines.append(f"   Volume Exhaustion: {'✓' if metrics.volume_exhaustion else '✗'}")
    
    lines.append("📉 RSI:")
    lines.append(f"   RSI: {metrics.rsi:.1f} {'⚠️ EXTREME' if metrics.rsi_extreme else '(overbought)' if metrics.rsi_overbought else ''}")
    lines.append(f"   Divergence: {'✓ BEARISH' if metrics.rsi_divergence else '✗'}")
    
    lines.append("🕯️ CANDLES:")
    lines.append(f"   Reversal Candle: {'✓' if metrics.reversal_candle else '✗'}")
    lines.append(f"   Long Upper Wick: {'✓' if metrics.long_upper_wick else '✗'}")
    
    if metrics.euphoria_level.tradeable:
        lines.append("")
        lines.append("💰 SHORT SETUP:")
        lines.append(f"   Entry Zone: ${metrics.entry_zone[0]:.2f} - ${metrics.entry_zone[1]:.2f}")
        lines.append(f"   Stop Loss: ${metrics.stop_loss:.2f} (above)")
        lines.append(f"   Target 1: ${metrics.target_1:.2f}")
        lines.append(f"   Target 2: ${metrics.target_2:.2f}")
    
    return "\n".join(lines)


# =============================================================================
# QUICK SCAN FUNCTION
# =============================================================================

def scan_for_capitulation(symbol: str, period: str = "30d", interval: str = "1h") -> CapitulationMetrics:
    """Quick function to scan a symbol for capitulation"""
    import yfinance as yf
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        return None
    
    detector = CapitulationDetector()
    return detector.analyze(df, symbol)


def scan_for_euphoria(symbol: str, period: str = "30d", interval: str = "1h") -> EuphoriaMetrics:
    """Quick function to scan a symbol for euphoria (short setup)"""
    import yfinance as yf
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        return None
    
    detector = CapitulationDetector()
    return detector.analyze_euphoria(df, symbol)


# =============================================================================
# MAIN - Demo/Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CAPITULATION DETECTOR - Scan")
    print("=" * 60)
    
    # Test on some symbols
    test_symbols = ["NVDA", "NFLX", "CRM", "COIN", "MSFT"]
    
    for symbol in test_symbols:
        print(f"\nScanning {symbol}...")
        try:
            metrics = scan_for_capitulation(symbol)
            if metrics:
                print(format_capitulation_alert(metrics, symbol))
            else:
                print(f"  Could not fetch data for {symbol}")
        except Exception as e:
            print(f"  Error: {e}")
        print("-" * 60)
