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
    
    # Composite
    capitulation_score: int           # 0-100
    capitulation_level: CapitulationLevel
    
    # Trade info
    entry_zone: Tuple[float, float]
    stop_loss: float
    target_1: float
    target_2: float


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
                 min_decline_pct: float = 15.0,      # Min decline to consider
                 ideal_decline_pct: float = 25.0,    # "Full" capitulation decline
                 rsi_oversold: float = 30.0,
                 rsi_extreme: float = 25.0,
                 volume_climax_mult: float = 2.5,    # RVOL for climax
                 volume_exhaustion_mult: float = 0.7, # RVOL for exhaustion
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
        
        # 5. CALCULATE COMPOSITE SCORE
        score = self._calculate_score(
            decline_pct, rvol, climax_detected, exhaustion,
            rsi, oversold, extreme, divergence,
            reversal_candle, long_wick
        )
        
        # 6. DETERMINE LEVEL
        level = self._determine_level(score, exhaustion, climax_detected)
        
        # 7. CALCULATE TRADE LEVELS
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
        """Analyze volume patterns for climax and exhaustion"""
        if 'volume' not in df.columns:
            return 1.0, False, False, 1.0
        
        # Current relative volume
        avg_vol = df['volume'].rolling(20).mean()
        current_rvol = df['volume'].iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1.0
        
        # Look for volume climax in recent bars
        recent_rvol = df['volume'].tail(10) / avg_vol.tail(10)
        climax_detected = recent_rvol.max() >= self.volume_climax_mult
        
        # Volume exhaustion: current volume dried up after climax
        exhaustion = climax_detected and current_rvol < self.volume_exhaustion_mult
        
        # Volume ratio on down days vs up days
        df['direction'] = (df['close'] - df['open']).apply(lambda x: 'down' if x < 0 else 'up')
        down_vol = df[df['direction'] == 'down']['volume'].mean()
        up_vol = df[df['direction'] == 'up']['volume'].mean()
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
                        long_wick: bool) -> int:
        """Calculate composite capitulation score (0-100)"""
        score = 0
        
        # PRICE DECLINE (25 points max)
        if decline_pct >= self.ideal_decline_pct:
            score += 25
        elif decline_pct >= self.min_decline_pct:
            score += int(15 + (decline_pct - self.min_decline_pct) / 
                        (self.ideal_decline_pct - self.min_decline_pct) * 10)
        elif decline_pct >= 10:
            score += 10
        
        # VOLUME (25 points max)
        if exhaustion:
            score += 25  # Ideal: climax followed by exhaustion
        elif climax:
            score += 15  # Climax but not yet exhausted
        elif rvol < 0.7:
            score += 10  # Low volume (could be drying up)
        
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
        
        return min(100, score)
    
    def _determine_level(self, score: int, exhaustion: bool, climax: bool) -> CapitulationLevel:
        """Determine capitulation level from score and conditions"""
        if score >= 80 and exhaustion:
            return CapitulationLevel.EXHAUSTION
        elif score >= 70 or climax:
            return CapitulationLevel.CLIMAX
        elif score >= 50:
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
            capitulation_score=0,
            capitulation_level=CapitulationLevel.NONE,
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
