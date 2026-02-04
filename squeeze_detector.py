"""
Enhanced Squeeze Detector
=========================
Detects volatility compression setups with multiple confirmation factors.

Factors:
- TTM Squeeze (BB inside Keltner) - The gold standard
- ATR Compression - Current ATR < 0.7x 20-period average
- ADX < 20 - Low directional movement
- RSI Neutral Zone - 40-60 range
- Tight Range relative to ATR
- Squeeze Duration - How long the compression has been active
- Direction Bias - Which way is it likely to break

Tiers:
- 50-69: FORMING - Squeeze is developing
- 70-84: ACTIVE - Squeeze is tight
- 85+: EXTREME - High priority watchlist
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime


@dataclass
class SqueezeMetrics:
    """Complete squeeze analysis result"""
    symbol: str
    score: int
    tier: str  # FORMING, ACTIVE, EXTREME
    factors: List[str]
    
    # Individual scores
    ttm_squeeze: bool
    ttm_score: int
    atr_compression: float  # ratio of current ATR to average ATR
    atr_score: int
    adx: float
    adx_score: int
    rsi: float
    rsi_score: int
    range_vs_atr: float  # day range as % of ATR
    range_score: int
    rvol: float
    rvol_score: int
    squeeze_duration: int  # days in squeeze
    duration_score: int
    
    # Direction bias
    direction_bias: str  # 'long', 'short', 'neutral'
    bias_score: int  # How confident in direction (0-100)
    price_drift: str  # 'up', 'down', 'flat'
    volume_bias: str  # 'accumulation', 'distribution', 'neutral'
    
    # Price levels
    current_price: float
    upper_band: float  # Upper Keltner/BB
    lower_band: float  # Lower Keltner/BB
    atr: float
    avg_daily_range: float
    
    timestamp: str


class SqueezeDetector:
    """Detects volatility squeeze setups"""
    
    def __init__(self):
        self.bb_period = 20
        self.bb_std = 2.0
        self.kc_period = 20
        self.kc_mult = 1.5
        self.atr_period = 14
        self.adx_period = 14
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        return upper, sma, lower
    
    def calculate_keltner_channels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        ema = df['close'].ewm(span=self.kc_period, adjust=False).mean()
        atr = self.calculate_atr(df, self.kc_period)
        upper = ema + (self.kc_mult * atr)
        lower = ema - (self.kc_mult * atr)
        return upper, ema, lower
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When +DM > -DM, set -DM to 0 and vice versa
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0
        
        # Calculate ATR for smoothing
        atr = self.calculate_atr(df, period)
        
        # Smooth DM
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        
        # Smooth DX to get ADX
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def detect_ttm_squeeze(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """
        TTM Squeeze: Bollinger Bands inside Keltner Channels
        Returns: (is_squeeze, days_in_squeeze)
        """
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(df)
        kc_upper, kc_mid, kc_lower = self.calculate_keltner_channels(df)
        
        # Check if BB is inside KC
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        # Count consecutive squeeze days
        squeeze_days = 0
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]:
                squeeze_days += 1
            else:
                break
        
        is_squeeze_now = squeeze.iloc[-1] if len(squeeze) > 0 else False
        
        return bool(is_squeeze_now), squeeze_days
    
    def calculate_direction_bias(self, df: pd.DataFrame) -> Dict:
        """
        Calculate likely breakout direction based on:
        - Price drift within squeeze
        - Volume on up vs down moves
        - Recent momentum
        """
        if len(df) < 10:
            return {'bias': 'neutral', 'score': 0, 'drift': 'flat', 'volume_bias': 'neutral'}
        
        recent = df.tail(10)
        
        # Price drift: Compare first half to second half
        first_half_avg = recent['close'].iloc[:5].mean()
        second_half_avg = recent['close'].iloc[5:].mean()
        price_change_pct = (second_half_avg - first_half_avg) / first_half_avg * 100
        
        if price_change_pct > 0.5:
            price_drift = 'up'
        elif price_change_pct < -0.5:
            price_drift = 'down'
        else:
            price_drift = 'flat'
        
        # Volume on up vs down moves
        up_volume = 0
        down_volume = 0
        
        for i in range(1, len(recent)):
            if recent['close'].iloc[i] > recent['close'].iloc[i-1]:
                up_volume += recent['volume'].iloc[i]
            else:
                down_volume += recent['volume'].iloc[i]
        
        total_volume = up_volume + down_volume
        if total_volume > 0:
            up_vol_pct = up_volume / total_volume
            down_vol_pct = down_volume / total_volume
            
            if up_vol_pct > 0.6:
                volume_bias = 'accumulation'
            elif down_vol_pct > 0.6:
                volume_bias = 'distribution'
            else:
                volume_bias = 'neutral'
        else:
            volume_bias = 'neutral'
        
        # Calculate overall bias score
        bias_score = 0
        
        # Drift contribution (+/- 40 points)
        if price_drift == 'up':
            bias_score += min(40, int(abs(price_change_pct) * 20))
        elif price_drift == 'down':
            bias_score -= min(40, int(abs(price_change_pct) * 20))
        
        # Volume contribution (+/- 30 points)
        if volume_bias == 'accumulation':
            bias_score += 30
        elif volume_bias == 'distribution':
            bias_score -= 30
        
        # Recent close vs open (momentum, +/- 30 points)
        last_3_momentum = sum(1 for i in range(-3, 0) if len(recent) > abs(i) and recent['close'].iloc[i] > recent['open'].iloc[i])
        momentum_score = (last_3_momentum - 1.5) * 20  # -30 to +30
        bias_score += int(momentum_score)
        
        # Determine direction
        if bias_score > 20:
            direction = 'long'
        elif bias_score < -20:
            direction = 'short'
        else:
            direction = 'neutral'
        
        return {
            'bias': direction,
            'score': min(100, max(0, abs(bias_score))),
            'drift': price_drift,
            'volume_bias': volume_bias
        }
    
    def analyze(self, symbol: str) -> Optional[SqueezeMetrics]:
        """
        Analyze a symbol for squeeze conditions
        
        Returns SqueezeMetrics with all factors and overall score
        """
        try:
            # Fetch data - need 60 days for proper calculation
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df.empty or len(df) < 30:
                return None
            
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Calculate all indicators
            bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(df)
            kc_upper, kc_mid, kc_lower = self.calculate_keltner_channels(df)
            atr_series = self.calculate_atr(df, self.atr_period)
            adx_series = self.calculate_adx(df, self.adx_period)
            
            # Current values
            current_price = float(df['close'].iloc[-1])
            current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0
            avg_atr_20 = float(atr_series.tail(20).mean()) if len(atr_series) >= 20 else current_atr
            current_adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 25
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 0.0001)
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # Today's range vs ATR
            today_range = float(df['high'].iloc[-1] - df['low'].iloc[-1])
            range_vs_atr = today_range / current_atr if current_atr > 0 else 1.0
            
            # Average daily range (5-day)
            avg_daily_range = float((df['high'].tail(5) - df['low'].tail(5)).mean())
            
            # Relative volume
            avg_volume = float(df['volume'].tail(20).mean())
            current_volume = float(df['volume'].iloc[-1])
            rvol = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # TTM Squeeze
            is_ttm_squeeze, squeeze_days = self.detect_ttm_squeeze(df)
            
            # Direction bias
            bias_data = self.calculate_direction_bias(df)
            
            # ============== SCORING ==============
            factors = []
            
            # 1. TTM Squeeze (+25 points) - THE GOLD STANDARD
            ttm_score = 0
            if is_ttm_squeeze:
                ttm_score = 25
                factors.append(f'TTM Squeeze ({squeeze_days}d)')
            
            # 2. ATR Compression (+20 points)
            atr_compression = current_atr / avg_atr_20 if avg_atr_20 > 0 else 1.0
            atr_score = 0
            if atr_compression < 0.6:
                atr_score = 20
                factors.append('ATR compressed')
            elif atr_compression < 0.7:
                atr_score = 15
                factors.append('ATR compressing')
            elif atr_compression < 0.8:
                atr_score = 10
                factors.append('Low ATR')
            
            # 3. ADX < 20 (+15 points)
            adx_score = 0
            if current_adx < 15:
                adx_score = 15
                factors.append('No trend (ADX)')
            elif current_adx < 20:
                adx_score = 10
                factors.append('Weak trend')
            elif current_adx < 25:
                adx_score = 5
            
            # 4. RSI Neutral Zone (+15 points) - WIDENED to 40-60
            rsi_score = 0
            if 45 <= rsi <= 55:
                rsi_score = 15
                factors.append('RSI coiling')
            elif 40 <= rsi <= 60:
                rsi_score = 10
                factors.append('RSI neutral')
            
            # 5. Tight Range relative to ATR (+15 points)
            range_score = 0
            if range_vs_atr < 0.5:
                range_score = 15
                factors.append('Very tight range')
            elif range_vs_atr < 0.7:
                range_score = 10
                factors.append('Tight range')
            elif range_vs_atr < 0.9:
                range_score = 5
                factors.append('Narrow range')
            
            # 6. Low Volume (+10 points)
            rvol_score = 0
            if rvol < 0.5:
                rvol_score = 10
                factors.append('Very low vol')
            elif rvol < 0.7:
                rvol_score = 7
                factors.append('Low vol')
            elif rvol < 0.9:
                rvol_score = 4
                factors.append('Quiet vol')
            
            # 7. Squeeze Duration (+5 to +15 points)
            duration_score = 0
            if squeeze_days >= 5:
                duration_score = 15
                factors.append(f'{squeeze_days}d squeeze')
            elif squeeze_days >= 3:
                duration_score = 10
            elif squeeze_days >= 2:
                duration_score = 5
            
            # Calculate total score
            total_score = ttm_score + atr_score + adx_score + rsi_score + range_score + rvol_score + duration_score
            
            # Determine tier
            if total_score >= 85:
                tier = 'EXTREME'
            elif total_score >= 70:
                tier = 'ACTIVE'
            elif total_score >= 50:
                tier = 'FORMING'
            else:
                tier = 'NONE'
            
            return SqueezeMetrics(
                symbol=symbol.upper(),
                score=total_score,
                tier=tier,
                factors=factors[:4],  # Top 4 factors
                
                ttm_squeeze=is_ttm_squeeze,
                ttm_score=ttm_score,
                atr_compression=round(atr_compression, 2),
                atr_score=atr_score,
                adx=round(current_adx, 1),
                adx_score=adx_score,
                rsi=round(rsi, 1),
                rsi_score=rsi_score,
                range_vs_atr=round(range_vs_atr, 2),
                range_score=range_score,
                rvol=round(rvol, 2),
                rvol_score=rvol_score,
                squeeze_duration=squeeze_days,
                duration_score=duration_score,
                
                direction_bias=bias_data['bias'],
                bias_score=bias_data['score'],
                price_drift=bias_data['drift'],
                volume_bias=bias_data['volume_bias'],
                
                current_price=round(current_price, 2),
                upper_band=round(float(kc_upper.iloc[-1]), 2),
                lower_band=round(float(kc_lower.iloc[-1]), 2),
                atr=round(current_atr, 2),
                avg_daily_range=round(avg_daily_range, 2),
                
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Squeeze analysis error for {symbol}: {e}")
            return None


def scan_for_squeezes(symbols: List[str]) -> List[SqueezeMetrics]:
    """
    Scan multiple symbols for squeeze setups
    Returns list sorted by score (highest first)
    """
    detector = SqueezeDetector()
    results = []
    
    for symbol in symbols:
        try:
            metrics = detector.analyze(symbol)
            if metrics and metrics.tier != 'NONE':
                results.append(metrics)
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
    
    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    return results


# Quick test
if __name__ == "__main__":
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'META']
    
    print("🎰 Enhanced Squeeze Detector")
    print("=" * 60)
    
    detector = SqueezeDetector()
    
    for symbol in test_symbols:
        result = detector.analyze(symbol)
        if result:
            print(f"\n{symbol}: Score {result.score} - {result.tier}")
            print(f"  TTM Squeeze: {result.ttm_squeeze} ({result.squeeze_duration}d)")
            print(f"  ATR Compression: {result.atr_compression}x")
            print(f"  ADX: {result.adx}")
            print(f"  RSI: {result.rsi}")
            print(f"  Direction Bias: {result.direction_bias} ({result.bias_score}%)")
            print(f"  Factors: {', '.join(result.factors)}")
        else:
            print(f"\n{symbol}: No data")
