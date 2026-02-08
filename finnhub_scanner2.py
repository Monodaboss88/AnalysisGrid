"""
Market Scanner - Multi-Source Data Integration
==============================================
Pulls market data from multiple sources with priority order:
    1. Polygon.io (paid = real-time, free = 15-min delayed)
    2. Alpaca (real-time with account)
    3. Finnhub (15-min delayed)
    4. yfinance (fallback)

Setup:
    Set environment variables (at least one):
    - POLYGON_API_KEY (recommended for real-time)
    - ALPACA_API_KEY + ALPACA_SECRET_KEY
    - FINNHUB_API_KEY

Note: File kept as finnhub_scanner.py for backwards compatibility.
      Class renamed to MarketScanner with FinnhubScanner as alias.

Author: Rob's Trading Systems
Version: 2.0.0
"""

import os
import time
try:
    import finnhub  # finnhub-python
except ImportError:  # pragma: no cover
    finnhub = None
try:
    import yfinance as yf  # Free fallback for candle data
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
from dataclasses import dataclass

# Import our scanner components
from chart_input_analyzer import ChartInputSystem, ChartInput, AnalysisResult, MTFResult, RangeContext


# =============================================================================
# TECHNICAL CALCULATORS
# =============================================================================

class TechnicalCalculator:
    """
    Calculates VP, VWAP, RSI from OHLCV data
    
    These match YOUR charting platform settings (VP 20,70)
    """
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, 
                                  value_area_pct: float = 0.70,
                                  num_bins: int = 50) -> Tuple[float, float, float]:
        """
        Calculate POC, VAH, VAL from price/volume data
        
        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume'
            value_area_pct: Percentage for value area (0.70 = 70%)
            num_bins: Number of price bins
        
        Returns:
            (POC, VAH, VAL)
        """
        if len(df) < 10:
            mid = df['close'].mean()
            return mid, mid * 1.01, mid * 0.99
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max == price_min:
            return price_max, price_max, price_min
        
        bin_size = (price_max - price_min) / num_bins
        bins = np.arange(price_min, price_max + bin_size, bin_size)
        
        # Distribute volume across price levels
        volume_profile = np.zeros(len(bins) - 1)
        
        for _, row in df.iterrows():
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']
            
            # Find bins this bar touches
            for i in range(len(bins) - 1):
                bin_low = bins[i]
                bin_high = bins[i + 1]
                
                # Check overlap
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)
                
                if overlap_high > overlap_low:
                    # Distribute volume proportionally
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        volume_profile[i] += bar_volume * overlap_pct
        
        # Find POC (highest volume bin)
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * value_area_pct
        
        # Start from POC and expand
        va_volume = volume_profile[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while va_volume < target_volume:
            # Check which direction to expand
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
    def calculate_vwap(df: pd.DataFrame) -> float:
        """Calculate VWAP"""
        if len(df) == 0:
            return 0
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return round(vwap, 2)
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI using Wilder's smoothing (EMA-based, standard method)"""
        if len(df) < period + 1:
            return 50.0
        
        # Use only the last 100 candles max for RSI calculation
        df_rsi = df.tail(100).copy()
        
        delta = df_rsi['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing (EMA with alpha = 1/period) - this is the standard RSI method
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility-based stops
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period (default 14)
        
        Returns:
            float: ATR value
        """
        if len(df) < period + 1:
            # Fallback: use simple range
            return (df['high'].max() - df['low'].min()) / len(df) if len(df) > 0 else 0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range = max of: H-L, |H-Prev_C|, |L-Prev_C|
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return round(atr.iloc[-1], 4) if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def is_rejection_candle(df: pd.DataFrame, direction: str, wick_ratio: float = 0.6) -> bool:
        """
        Check if the latest candle shows rejection pattern
        
        Args:
            df: DataFrame with OHLC data
            direction: "bullish" (rejection of lows) or "bearish" (rejection of highs)
            wick_ratio: Minimum wick ratio to consider rejection (0.6 = 60% of range)
        
        Returns:
            bool: True if rejection candle detected
        """
        if len(df) < 1:
            return False
        
        candle = df.iloc[-1]
        high = candle['high']
        low = candle['low']
        open_price = candle['open']
        close = candle['close']
        
        candle_range = high - low
        if candle_range <= 0:
            return False
        
        body_top = max(open_price, close)
        body_bottom = min(open_price, close)
        
        upper_wick = high - body_top
        lower_wick = body_bottom - low
        
        if direction == "bullish":
            # Bullish rejection: large lower wick (hammer pattern)
            lower_wick_ratio = lower_wick / candle_range
            return lower_wick_ratio >= wick_ratio
        else:
            # Bearish rejection: large upper wick (shooting star pattern)
            upper_wick_ratio = upper_wick / candle_range
            return upper_wick_ratio >= wick_ratio
    
    @staticmethod
    def get_extension_from_level(price: float, level: float, atr: float) -> float:
        """
        Calculate how many ATR price is extended from a level
        
        Args:
            price: Current price
            level: Reference level (VAH, VAL, POC, etc.)
            atr: Average True Range
        
        Returns:
            float: Extension in ATR units (negative = below level)
        """
        if atr <= 0:
            return 0
        return round((price - level) / atr, 2)
    
    @staticmethod
    def calculate_relative_volume(df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Calculate Relative Volume (RVOL) - current volume vs average
        
        Returns:
            float: RVOL multiplier (1.0 = average, 2.0 = 2x average)
        """
        if len(df) < lookback + 1 or 'volume' not in df.columns:
            return 1.0
        
        # Average volume over lookback period (excluding current bar)
        avg_volume = df['volume'].iloc[-(lookback+1):-1].mean()
        current_volume = df['volume'].iloc[-1]
        
        if avg_volume <= 0:
            return 1.0
        
        return round(current_volume / avg_volume, 2)
    
    @staticmethod
    def calculate_volume_trend(df: pd.DataFrame, periods: int = 5) -> str:
        """
        Determine if volume is increasing or decreasing
        
        Returns:
            str: "increasing", "decreasing", or "neutral"
        """
        if len(df) < periods + 1 or 'volume' not in df.columns:
            return "neutral"
        
        # Compare recent volume average to prior period
        recent_vol = df['volume'].iloc[-periods:].mean()
        prior_vol = df['volume'].iloc[-(periods*2):-periods].mean() if len(df) >= periods * 2 else df['volume'].iloc[:-periods].mean()
        
        if prior_vol <= 0:
            return "neutral"
        
        change = (recent_vol - prior_vol) / prior_vol
        
        if change > 0.15:  # 15% increase
            return "increasing"
        elif change < -0.15:  # 15% decrease
            return "decreasing"
        return "neutral"
    
    @staticmethod
    def detect_volume_divergence(df: pd.DataFrame, periods: int = 5) -> bool:
        """
        Detect volume divergence (price going one way, volume going the other)
        
        Returns:
            bool: True if divergence detected
        """
        if len(df) < periods + 1:
            return False
        
        # Price trend
        price_start = df['close'].iloc[-periods-1]
        price_end = df['close'].iloc[-1]
        price_change = (price_end - price_start) / price_start
        
        # Volume trend
        vol_start = df['volume'].iloc[-periods-1:-1].mean()
        vol_end = df['volume'].iloc[-periods:].mean()
        vol_change = (vol_end - vol_start) / vol_start if vol_start > 0 else 0
        
        # Divergence: price up + volume down, or price down + volume down significantly
        if price_change > 0.02 and vol_change < -0.20:  # Price up 2%+, volume down 20%+
            return True
        if price_change < -0.02 and vol_change < -0.20:  # Price down 2%+, volume down 20%+
            return True
        
        return False
    
    @staticmethod
    def calculate_range_structure(df_weekly: pd.DataFrame, 
                                   df_daily: pd.DataFrame = None,
                                   current_price: float = None) -> 'RangeContext':
        """
        Calculate range structure using WEEKLY candles for macro structure
        and DAILY candles for tactical proximity.
        
        Weekly candles represent complete auction cycles, so HH/HL/LH/LL
        patterns on weekly bars are much more meaningful than daily period
        comparisons. A Lower-Low on a weekly bar = an entire week's auction
        failed to hold prior week's low.
        
        Structure (weekly): Compare each week's high/low to prior week
        Compression (weekly): Current week range vs 8-week average range
        Proximity (daily): How close price is to key levels right now
        
        Args:
            df_weekly: DataFrame with weekly OHLCV data (at least 8 bars)
            df_daily: Optional DataFrame with daily OHLCV for proximity checks
            current_price: Current price for proximity calculations
            
        Returns:
            RangeContext with structural analysis
        """
        if df_weekly is None or len(df_weekly) < 6:
            return RangeContext()
        
        price = current_price or df_weekly['close'].iloc[-1]
        
        # =====================================================================
        # WEEKLY STRUCTURE: Compare consecutive weekly candles
        # Each week = complete auction cycle, so structure is high-conviction
        # =====================================================================
        
        # Use last 8 weeks for structure analysis
        weeks = df_weekly.tail(8)
        
        ll_count = 0
        hh_count = 0
        lh_count = 0
        hl_count = 0
        structures = {}
        
        for i in range(1, len(weeks)):
            curr = weeks.iloc[i]
            prev = weeks.iloc[i - 1]
            
            curr_high = curr['high']
            curr_low = curr['low']
            prev_high = prev['high']
            prev_low = prev['low']
            
            structure = ""
            
            # Compare highs: Higher High or Lower High
            if curr_high > prev_high * 1.001:
                structure += "HH"
                hh_count += 1
            elif curr_high < prev_high * 0.999:
                structure += "LH"
                lh_count += 1
            else:
                structure += "EQ"  # Equal high
            
            # Compare lows: Higher Low or Lower Low
            if curr_low > prev_low * 1.001:
                structure += "+HL"
                hl_count += 1
            elif curr_low < prev_low * 0.999:
                structure += "+LL"
                ll_count += 1
            else:
                structure += "+EQ"  # Equal low
            
            structures[f"W{i}"] = structure
        
        total_periods = len(structures)
        
        # =====================================================================
        # WEEKLY TREND CLASSIFICATION
        # Based on consecutive weekly HH/HL/LH/LL patterns
        # =====================================================================
        
        bearish_signals = ll_count + lh_count
        bullish_signals = hh_count + hl_count
        total_signals = bearish_signals + bullish_signals
        
        # Check recent bias (last 3 weeks weighted heavier)
        recent_weeks = df_weekly.tail(4)
        recent_bearish = 0
        recent_bullish = 0
        for i in range(1, len(recent_weeks)):
            curr = recent_weeks.iloc[i]
            prev = recent_weeks.iloc[i - 1]
            if curr['high'] < prev['high'] * 0.999:
                recent_bearish += 1
            if curr['high'] > prev['high'] * 1.001:
                recent_bullish += 1
            if curr['low'] < prev['low'] * 0.999:
                recent_bearish += 1
            if curr['low'] > prev['low'] * 1.001:
                recent_bullish += 1
        
        if bearish_signals >= 8 and bullish_signals <= 2:
            trend = "STRONG_DOWNTREND"
        elif bearish_signals >= 5 and bearish_signals > bullish_signals * 2:
            trend = "DOWNTREND"
        elif bullish_signals >= 8 and bearish_signals <= 2:
            trend = "STRONG_UPTREND"
        elif bullish_signals >= 5 and bullish_signals > bearish_signals * 2:
            trend = "UPTREND"
        # Use recent bias to break ties
        elif recent_bearish >= 4 and recent_bullish <= 1:
            trend = "DOWNTREND"
        elif recent_bullish >= 4 and recent_bearish <= 1:
            trend = "UPTREND"
        else:
            trend = "NEUTRAL"
        
        # =====================================================================
        # WEEKLY COMPRESSION: Current week range vs 8-week average range
        # More meaningful than daily 3D/30D ratio - measures if weekly
        # auction cycles are contracting
        # =====================================================================
        
        week_ranges = []
        for i in range(len(weeks)):
            w = weeks.iloc[i]
            week_ranges.append(w['high'] - w['low'])
        
        current_week_range = week_ranges[-1] if week_ranges else 0
        avg_8w_range = sum(week_ranges) / len(week_ranges) if week_ranges else 1
        
        # Also check if ranges are progressively narrowing (true compression)
        if len(week_ranges) >= 4:
            first_half_avg = sum(week_ranges[:len(week_ranges)//2]) / (len(week_ranges)//2)
            second_half_avg = sum(week_ranges[len(week_ranges)//2:]) / (len(week_ranges) - len(week_ranges)//2)
            narrowing = second_half_avg < first_half_avg * 0.80  # Ranges shrinking 20%+
        else:
            narrowing = False
        
        compression_ratio = current_week_range / avg_8w_range if avg_8w_range > 0 else 1.0
        
        if compression_ratio < 0.50 or narrowing:
            range_state = "COMPRESSING"
        elif compression_ratio > 1.50:
            range_state = "EXPANDING"
        else:
            range_state = "NORMAL"
        
        # =====================================================================
        # DAILY PROXIMITY: Tactical level checks using daily data
        # Weekly tells us the structure, daily tells us where we are NOW
        # =====================================================================
        
        # Use 8-week high/low as the structural range
        high_8w = weeks['high'].max()
        low_8w = weeks['low'].min()
        price_range = high_8w - low_8w if high_8w > low_8w else 1
        
        near_support = (price - low_8w) / price_range < 0.10 if price_range > 0 else False
        near_resistance = (high_8w - price) / price_range < 0.10 if price_range > 0 else False
        
        # If daily data available, use it for tighter proximity checks
        if df_daily is not None and len(df_daily) >= 5:
            daily_5d = df_daily.tail(5)
            recent_low = daily_5d['low'].min()
            recent_high = daily_5d['high'].max()
            
            # Breakdown/breakout watch = recent daily levels
            breakdown_watch = round(min(recent_low, low_8w), 2)
            breakout_watch = round(max(recent_high, high_8w), 2)
            
            # Tighten proximity using daily position within weekly range
            daily_pos_in_range = (price - low_8w) / price_range if price_range > 0 else 0.5
            if daily_pos_in_range < 0.10:
                near_support = True
            elif daily_pos_in_range > 0.90:
                near_resistance = True
        else:
            # Fallback: use weekly levels
            current_week_low = weeks.iloc[-1]['low']
            current_week_high = weeks.iloc[-1]['high']
            breakdown_watch = round(min(current_week_low, low_8w), 2)
            breakout_watch = round(max(current_week_high, high_8w), 2)
        
        return RangeContext(
            trend=trend,
            range_state=range_state,
            compression_ratio=round(compression_ratio, 3),
            ll_count=ll_count,
            hh_count=hh_count,
            lh_count=lh_count,
            hl_count=hl_count,
            total_periods=total_periods,
            near_support=near_support,
            near_resistance=near_resistance,
            breakout_watch=breakout_watch,
            breakdown_watch=breakdown_watch
        )


class MarketScanner:
    """
    Multi-source market data scanner.
    
    Data Priority: Polygon > Alpaca > Finnhub > yfinance
    
    Usage:
        scanner = MarketScanner()  # Uses env vars for API keys
        
        # Single symbol analysis
        result = scanner.analyze("META")
        print(result)
        
        # Multi-timeframe analysis  
        mtf = scanner.analyze_mtf("META")
        print(mtf)
        
        # Scan watchlist
        results = scanner.scan_symbols(["META", "AAPL", "NVDA"])
    """
    
    # Timeframe mappings (Finnhub resolution codes)
    TIMEFRAMES = {
        "1MIN": "1",
        "5MIN": "5",
        "15MIN": "15",
        "30MIN": "30",
        "1HR": "60",
        "2HR": "60",   # Will aggregate from 1HR
        "4HR": "60",   # Will aggregate from 1HR
        "DAILY": "D",
        "WEEKLY": "W"
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize with Finnhub API key
        
        Args:
            api_key: Your Finnhub API key (or set FINNHUB_API_KEY env var)
        """
        if finnhub is None:
            raise ValueError(
                "Missing dependency: finnhub-python.\n"
                "Install with: pip install finnhub-python\n"
                "(or: pip install -r requirements.txt)"
            )

        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. Either pass api_key or set FINNHUB_API_KEY environment variable.\n"
                "Get free key at: https://finnhub.io"
            )
        
        self.client = finnhub.Client(api_key=self.api_key)
        self.calc = TechnicalCalculator()
        self.system = ChartInputSystem()
        
        # Cache for rate limiting
        self._cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_minutes = 1  # Cache data for 1 minute
        
        # Initialize Polygon client if credentials available
        self.polygon_client = None
        polygon_key = os.environ.get("POLYGON_API_KEY")
        if polygon_available and polygon_key:
            try:
                self.polygon_client = PolygonClient(polygon_key)
                print("âœ… Polygon.io real-time data enabled")
            except Exception as e:
                print(f"âš ï¸ Polygon init failed: {e}")
        
        # Initialize Alpaca client if credentials available
        self.alpaca_client = None
        alpaca_key = os.environ.get("ALPACA_API_KEY")
        alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
        if alpaca_available and alpaca_key and alpaca_secret:
            try:
                self.alpaca_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                print("âœ… Alpaca real-time data enabled")
            except Exception as e:
                print(f"âš ï¸ Alpaca init failed: {e}")
    
    def _get_candles_polygon(self,
                             symbol: str,
                             resolution: str = "60",
                             days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch candle data from Polygon.io
        Free tier: 15-min delayed | Paid: Real-time
        """
        if self.polygon_client is None:
            return None
        
        try:
            # Map resolution to Polygon multiplier/timespan
            timespan_map = {
                "1": ("1", "minute"),
                "5": ("5", "minute"),
                "15": ("15", "minute"),
                "30": ("30", "minute"),
                "60": ("1", "hour"),
                "D": ("1", "day"),
                "W": ("1", "week")
            }
            multiplier, timespan = timespan_map.get(resolution, ("1", "hour"))
            
            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch bars using Polygon REST client
            bars = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=int(multiplier),
                timespan=timespan,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            )
            
            if not bars:
                print(f"âš ï¸ Polygon: No data for {symbol}")
                return None
            
            # Convert to DataFrame
            records = []
            for bar in bars:
                records.append({
                    'timestamp': pd.to_datetime(bar.timestamp, unit='ms'),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
            
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"âœ… Polygon: Got {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"âŒ Polygon error for {symbol}: {e}")
            return None
    
    def _get_candles_alpaca(self,
                            symbol: str,
                            resolution: str = "60",
                            days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch REAL-TIME candle data from Alpaca (free with account)
        """
        if self.alpaca_client is None:
            return None
        
        try:
            # Map resolution to Alpaca TimeFrame
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
            
            # Build request
            end = datetime.now()
            start = end - timedelta(days=days_back)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            
            # Fetch bars
            bars = self.alpaca_client.get_stock_bars(request)
            
            if symbol not in bars.data or len(bars.data[symbol]) == 0:
                print(f"âš ï¸ Alpaca: No data for {symbol}")
                return None
            
            # Convert to DataFrame
            records = []
            for bar in bars.data[symbol]:
                records.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
            
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Remove timezone for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            print(f"âœ… Alpaca: Got {len(df)} real-time candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"âŒ Alpaca error for {symbol}: {e}")
            return None
    
    def _get_candles_yfinance(self,
                              symbol: str,
                              resolution: str = "60",
                              days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Fallback: Fetch candle data from Yahoo Finance (free, no API key needed)
        """
        if yf is None:
            print("âš ï¸ yfinance not installed. Run: pip install yfinance")
            return None
        
        try:
            # Map resolution to yfinance interval
            interval_map = {
                "1": "1m", "5": "5m", "15": "15m", "30": "30m",
                "60": "1h", "D": "1d", "W": "1wk"
            }
            interval = interval_map.get(resolution, "1h")
            
            # yfinance has limits: 1m=7d, 5m/15m/30m=60d, 1h=730d
            period_map = {
                "1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d",
                "1h": f"{min(days_back, 729)}d", "1d": f"{days_back}d", "1wk": f"{days_back}d"
            }
            period = period_map.get(interval, f"{days_back}d")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"âš ï¸ yfinance: No data for {symbol}")
                return None
            
            # Normalize column names
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index.name = 'timestamp'
            
            print(f"âœ… yfinance: Got {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"âŒ yfinance error for {symbol}: {e}")
            return None
    
    def _get_candles(self, 
                     symbol: str, 
                     resolution: str = "60",
                     days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch candle data. Priority: Alpaca (real-time) > Finnhub > yfinance (delayed)
        
        Args:
            symbol: Stock symbol
            resolution: Resolution (1, 5, 15, 30, 60, D, W, M)
            days_back: Number of days of history
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{resolution}_{days_back}"
        
        # Check cache
        if cache_key in self._cache:
            df, timestamp = self._cache[cache_key]
            if (datetime.now() - timestamp).seconds < self._cache_minutes * 60:
                return df
        
        df = None
        
        # 1. Try Polygon first (real-time with paid, delayed on free tier)
        if self.polygon_client is not None:
            df = self._get_candles_polygon(symbol, resolution, days_back)
        
        # 2. Try Alpaca (real-time, free with account)
        if df is None and self.alpaca_client is not None:
            df = self._get_candles_alpaca(symbol, resolution, days_back)
        
        # 3. Try Finnhub if Alpaca unavailable
        if df is None:
            try:
                end_time = int(datetime.now().timestamp())
                start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
                
                data = self.client.stock_candles(
                    symbol=symbol,
                    resolution=resolution,
                    _from=start_time,
                    to=end_time
                )
                
                if data.get('s') == 'ok' and data.get('c'):
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data['t'], unit='s'),
                        'open': data['o'],
                        'high': data['h'],
                        'low': data['l'],
                        'close': data['c'],
                        'volume': data['v']
                    })
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    print(f"âœ… Finnhub: Got {len(df)} candles for {symbol}")
                else:
                    print(f"âš ï¸ Finnhub: No data for {symbol}")
                
            except Exception as e:
                error_str = str(e)
                if "403" in error_str:
                    print(f"âš ï¸ Finnhub 403 (no candle access)")
                else:
                    print(f"âŒ Finnhub error: {e}")
        
        # 4. Fallback to yfinance (delayed but free)
        if df is None:
            df = self._get_candles_yfinance(symbol, resolution, days_back)
        
        # Cache successful result
        if df is not None and len(df) > 0:
            self._cache[cache_key] = (df, datetime.now())
        
        return df
        
        return df
    
    def _resample_to_timeframe(self, 
                                df: pd.DataFrame, 
                                timeframe: str) -> pd.DataFrame:
        """Resample data to higher timeframe"""
        
        # Note: pandas 2.x uses lowercase 'h' instead of 'H'
        resample_map = {
            "30MIN": "30min",
            "1HR": "1h",
            "2HR": "2h",
            "4HR": "4h",
            "DAILY": "1D"
        }
        
        rule = resample_map.get(timeframe.upper())
        if not rule:
            return df
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote - tries Polygon real-time endpoints first"""
        
        # Try Polygon real-time endpoints (paid subscription = true real-time)
        if self.polygon_client:
            # Use snapshot - it includes min bar, quote, and day data
            try:
                snapshot = self.polygon_client.get_snapshot_ticker("stocks", symbol.upper())
                if snapshot:
                    # snapshot has: day, lastTrade, lastQuote, min, prevDay
                    current_price = None
                    source = None
                    
                    # Priority 1: Current minute bar (min) - most real-time on Stocks Starter
                    if hasattr(snapshot, 'min') and snapshot.min and hasattr(snapshot.min, 'close'):
                        current_price = float(snapshot.min.close)
                        source = 'polygon_min_bar'
                        # Log the minute bar timestamp if available
                        if hasattr(snapshot.min, 't'):
                            print(f"ðŸ“Š Min bar timestamp: {snapshot.min.t}")
                    
                    # Priority 2: Last quote midpoint (bid/ask average)
                    if current_price is None and hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                        bid = getattr(snapshot.last_quote, 'bid', None) or getattr(snapshot.last_quote, 'p', None)
                        ask = getattr(snapshot.last_quote, 'ask', None) or getattr(snapshot.last_quote, 'P', None)
                        if bid and ask:
                            current_price = (float(bid) + float(ask)) / 2
                            source = 'polygon_quote_mid'
                    
                    # Priority 3: Last trade price (may require higher tier)
                    if current_price is None and hasattr(snapshot, 'last_trade') and snapshot.last_trade and hasattr(snapshot.last_trade, 'price'):
                        current_price = float(snapshot.last_trade.price)
                        source = 'polygon_snapshot_trade'
                    
                    # Priority 4: Day's close/last (during market hours)
                    if current_price is None and hasattr(snapshot, 'day') and snapshot.day:
                        if hasattr(snapshot.day, 'close') and snapshot.day.close:
                            current_price = float(snapshot.day.close)
                            source = 'polygon_snapshot_day'
                    
                    # Priority 3: Previous day close
                    if current_price is None and hasattr(snapshot, 'prev_day') and snapshot.prev_day:
                        if hasattr(snapshot.prev_day, 'close') and snapshot.prev_day.close:
                            current_price = float(snapshot.prev_day.close)
                            source = 'polygon_snapshot_prev'
                    
                    if current_price:
                        # Get additional data
                        open_price = None
                        high_price = None
                        low_price = None
                        prev_close = None
                        
                        if hasattr(snapshot, 'day') and snapshot.day:
                            open_price = float(snapshot.day.open) if hasattr(snapshot.day, 'open') and snapshot.day.open else None
                            high_price = float(snapshot.day.high) if hasattr(snapshot.day, 'high') and snapshot.day.high else None
                            low_price = float(snapshot.day.low) if hasattr(snapshot.day, 'low') and snapshot.day.low else None
                        
                        if hasattr(snapshot, 'prev_day') and snapshot.prev_day:
                            prev_close = float(snapshot.prev_day.close) if hasattr(snapshot.prev_day, 'close') and snapshot.prev_day.close else None
                        
                        change = current_price - prev_close if prev_close else None
                        change_pct = (change / prev_close * 100) if prev_close and change else None
                        
                        print(f"âœ… Polygon snapshot for {symbol}: ${current_price:.2f} ({source})")
                        return {
                            'current': current_price,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'prev_close': prev_close,
                            'change': change,
                            'change_pct': change_pct,
                            'timestamp': datetime.now(),
                            'source': source
                        }
            except Exception as e:
                print(f"âš ï¸ Polygon snapshot failed for {symbol}: {e}")
            
            # Fallback: Get latest 1-minute bar (works on FREE tier - 15 min delayed)
            try:
                from datetime import date
                today = date.today()
                aggs = self.polygon_client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="minute",
                    from_=today - timedelta(days=1),
                    to=today,
                    limit=5,
                    sort="desc"
                )
                if aggs and len(aggs) > 0:
                    latest = aggs[0]
                    latest_price = float(latest.close)
                    print(f"âœ… Polygon latest bar for {symbol}: ${latest_price:.2f} (15-min delayed)")
                    return {
                        'current': latest_price,
                        'open': float(latest.open) if hasattr(latest, 'open') else None,
                        'high': float(latest.high) if hasattr(latest, 'high') else None,
                        'low': float(latest.low) if hasattr(latest, 'low') else None,
                        'prev_close': None,
                        'change': None,
                        'change_pct': None,
                        'timestamp': datetime.fromtimestamp(latest.timestamp / 1000) if hasattr(latest, 'timestamp') else datetime.now(),
                        'source': 'polygon_latest_bar'
                    }
            except Exception as e:
                print(f"âš ï¸ Polygon latest bar failed for {symbol}: {e}")
            
            # Fallback: Previous close (last resort)
            try:
                prev_close = self.polygon_client.get_previous_close(symbol)
                if prev_close and prev_close.results and len(prev_close.results) > 0:
                    result = prev_close.results[0]
                    close_price = float(result.close)
                    print(f"âš ï¸ Polygon prev_close for {symbol}: ${close_price:.2f} (STALE - yesterday's close)")
                    return {
                        'current': close_price,
                        'open': float(result.open) if hasattr(result, 'open') else None,
                        'high': float(result.high) if hasattr(result, 'high') else None,
                        'low': float(result.low) if hasattr(result, 'low') else None,
                        'prev_close': close_price,
                        'change': None,
                        'change_pct': None,
                        'timestamp': datetime.now(),
                        'source': 'polygon_prev_close'
                    }
            except Exception as e:
                print(f"âš ï¸ Polygon prev_close failed for {symbol}: {e}")
        
        # Try Alpaca as backup (free real-time for US stocks)
        if self.alpaca_client:
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote_data = self.alpaca_client.get_stock_latest_quote(request)
                if symbol in quote_data:
                    q = quote_data[symbol]
                    mid_price = (q.bid_price + q.ask_price) / 2 if q.bid_price and q.ask_price else q.ask_price or q.bid_price
                    return {
                        'current': mid_price,
                        'open': None,
                        'high': None,
                        'low': None,
                        'prev_close': None,
                        'change': None,
                        'change_pct': None,
                        'timestamp': q.timestamp,
                        'source': 'alpaca_realtime'
                    }
            except Exception as e:
                print(f"âš ï¸ Alpaca quote failed for {symbol}: {e}")
        
        # Fall back to Finnhub (15-min delayed on free tier)
        try:
            quote = self.client.quote(symbol)
            if quote and quote.get('c'):
                return {
                    'current': quote.get('c'),
                    'open': quote.get('o'),
                    'high': quote.get('h'),
                    'low': quote.get('l'),
                    'prev_close': quote.get('pc'),
                    'change': quote.get('d'),
                    'change_pct': quote.get('dp'),
                    'timestamp': datetime.fromtimestamp(quote.get('t', 0)),
                    'source': 'finnhub_delayed'
                }
        except Exception as e:
            print(f"âš ï¸ Finnhub quote failed for {symbol}: {e}")
        
        # Fall back to yfinance (near real-time during market hours)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Try fast_info.lastPrice (most reliable for real-time)
            try:
                fi = ticker.fast_info
                if hasattr(fi, 'last_price') and fi.last_price:
                    print(f"âœ… yfinance fast_info for {symbol}: ${fi.last_price:.2f}")
                    return {
                        'current': float(fi.last_price),
                        'open': float(fi.open) if hasattr(fi, 'open') else None,
                        'high': float(fi.day_high) if hasattr(fi, 'day_high') else None,
                        'low': float(fi.day_low) if hasattr(fi, 'day_low') else None,
                        'prev_close': float(fi.previous_close) if hasattr(fi, 'previous_close') else None,
                        'change': None,
                        'change_pct': None,
                        'timestamp': datetime.now(),
                        'source': 'yfinance_realtime'
                    }
            except Exception as e:
                print(f"âš ï¸ yfinance fast_info failed: {e}")
            
            # Fallback: Use history with 1-minute interval for most recent price
            try:
                hist = ticker.history(period='1d', interval='1m')
                if hist is not None and len(hist) > 0:
                    last_close = float(hist['Close'].iloc[-1])
                    print(f"âœ… yfinance history for {symbol}: ${last_close:.2f}")
                    return {
                        'current': last_close,
                        'open': float(hist['Open'].iloc[0]) if 'Open' in hist.columns else None,
                        'high': float(hist['High'].max()) if 'High' in hist.columns else None,
                        'low': float(hist['Low'].min()) if 'Low' in hist.columns else None,
                        'prev_close': None,
                        'change': None,
                        'change_pct': None,
                        'timestamp': datetime.now(),
                        'source': 'yfinance_history'
                    }
            except Exception as e:
                print(f"âš ï¸ yfinance history failed: {e}")
                
        except Exception as e:
            print(f"âŒ yfinance quote failed for {symbol}: {e}")
        
        print(f"âŒ All quote sources failed for {symbol}")
        return None
    
    def analyze(self, 
                symbol: str, 
                timeframe: str = "1HR",
                days_back: int = None,
                vp_bars: int = 30) -> Optional[AnalysisResult]:
        """
        Analyze a single symbol/timeframe
        
        Args:
            symbol: Stock symbol
            timeframe: "5MIN", "15MIN", "30MIN", "1HR", "2HR", "4HR", "DAILY"
            days_back: Days of history for calculations (auto-set if None)
            vp_bars: Number of bars to use for VP (like Webull visible range)
        
        Returns:
            AnalysisResult with signal and levels
        """
        # Map timeframe to Polygon resolution
        # Use native resolution for accurate VP calculations
        resolution_map = {
            "5MIN": "5",
            "15MIN": "15",
            "30MIN": "30",
            "1HR": "60",
            "2HR": "60",   # Resample from hourly
            "4HR": "60",   # Resample from hourly
            "DAILY": "D"
        }
        resolution = resolution_map.get(timeframe.upper(), "60")
        
        # Fetch enough days to get vp_bars candles
        if days_back is None:
            days_map = {
                "5MIN": 1, "15MIN": 2, "30MIN": 5,
                "1HR": 7, "2HR": 15, "4HR": 30, "DAILY": 60
            }
            days_back = days_map.get(timeframe.upper(), 7)
        
        df = self._get_candles(symbol, resolution, days_back)
        
        if df is None or len(df) < 10:
            print(f"âš ï¸ Insufficient data for {symbol}")
            return None
        
        # Resample only for 2HR and 4HR (up from hourly)
        if timeframe.upper() in ["2HR", "4HR"]:
            df = self._resample_to_timeframe(df, timeframe)
        
        if len(df) < 10:
            return None
        
        # Trim to last vp_bars for consistent VP (like Webull visible range)
        if len(df) > vp_bars:
            df = df.tail(vp_bars)
        
        # Get current price (from latest candle or real-time quote)
        quote = self.get_quote(symbol)
        current_price = quote['current'] if quote else df['close'].iloc[-1]
        
        # Calculate technicals
        poc, vah, val = self.calc.calculate_volume_profile(df)
        vwap = self.calc.calculate_vwap(df)
        rsi = self.calc.calculate_rsi(df)
        
        # Calculate volume metrics
        rvol = self.calc.calculate_relative_volume(df)
        volume_trend = self.calc.calculate_volume_trend(df)
        volume_divergence = self.calc.detect_volume_divergence(df)
        
        # NEW: Calculate ATR and check for rejection candle (from concept system)
        atr = self.calc.calculate_atr(df)
        
        # Check for rejection candle based on price position
        has_rejection = False
        if current_price < val:
            # Below VAL, check for bullish rejection (hammer)
            has_rejection = self.calc.is_rejection_candle(df, "bullish")
        elif current_price > vah:
            # Above VAH, check for bearish rejection (shooting star)
            has_rejection = self.calc.is_rejection_candle(df, "bearish")
        
        # Run through analyzer with enhanced data
        result = self.system.analyze(
            symbol=symbol,
            price=current_price,
            vah=vah,
            poc=poc,
            val=val,
            vwap=vwap,
            rsi=rsi,
            timeframe=timeframe,
            rvol=rvol,
            volume_trend=volume_trend,
            volume_divergence=volume_divergence,
            atr=atr,
            has_rejection=has_rejection
        )
        
        return result
    
    def analyze_mtf(self, 
                    symbol: str,
                    timeframes: List[str] = None) -> Optional[MTFResult]:
        """
        Multi-timeframe analysis with range structure context
        
        Improvements:
        - FIX #1: Range structure feeds into signal scoring
        - FIX #2: Directional vs YELLOW confluence distinction
        - FIX #3: Composite compression+trend signals
        - FIX #4: Actual 30MIN candles instead of hourly copy
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes (default: ["30MIN", "1HR", "2HR", "4HR"])
        
        Returns:
            MTFResult with combined signal including range structure
        """
        if timeframes is None:
            timeframes = ["30MIN", "1HR", "2HR", "4HR"]
        
        # Get base hourly data (for 1HR, 2HR, 4HR)
        df_hourly = self._get_candles(symbol, "60", days_back=30)
        
        if df_hourly is None or len(df_hourly) < 50:
            print(f"  Insufficient data for MTF analysis on {symbol}")
            return None
        
        # FIX #4: Fetch actual 30MIN candles for distinct 30MIN analysis
        df_30min = None
        if "30MIN" in [tf.upper() for tf in timeframes]:
            df_30min = self._get_candles(symbol, "30", days_back=10)
            if df_30min is not None and len(df_30min) < 20:
                df_30min = None  # Fall back if insufficient
        
        # Get weekly + daily data for range structure calculation
        # Weekly = macro structure (trend, compression, HH/HL/LH/LL)
        # Daily = tactical proximity (support/resistance nearness)
        df_weekly = self._get_candles(symbol, "W", days_back=90)
        df_daily = self._get_candles(symbol, "D", days_back=15)
        
        # Get current price
        quote = self.get_quote(symbol)
        current_price = quote['current'] if quote else df_hourly['close'].iloc[-1]
        
        # FIX #1: Calculate range structure context (weekly-driven)
        range_ctx = None
        if df_weekly is not None and len(df_weekly) >= 6:
            range_ctx = self.calc.calculate_range_structure(df_weekly, df_daily, current_price)
            print(f"   Range (weekly): {range_ctx.trend} | {range_ctx.range_state} | "
                  f"LL:{range_ctx.ll_count} HH:{range_ctx.hh_count} LH:{range_ctx.lh_count} HL:{range_ctx.hl_count} | "
                  f"Compression: {range_ctx.compression_ratio:.2f}")
        
        # Calculate for each timeframe
        tf_data = {}
        
        for tf in timeframes:
            tf_upper = tf.upper()
            
            # FIX #4: Use actual 30MIN data instead of hourly copy
            if tf_upper == "30MIN":
                if df_30min is not None:
                    df = df_30min.copy()
                else:
                    # Fallback: resample hourly to 30min won't help (can't upsample)
                    # Use hourly but at least it's labeled correctly now
                    df = df_hourly.copy()
                    print(f"   30MIN: Falling back to hourly data (30min candles unavailable)")
            else:
                df = self._resample_to_timeframe(df_hourly, tf)
            
            if len(df) < 10:
                continue
            
            # Calculate technicals
            poc, vah, val = self.calc.calculate_volume_profile(df)
            vwap = self.calc.calculate_vwap(df)
            rsi = self.calc.calculate_rsi(df)
            
            tf_data[tf_upper] = {
                "price": current_price,
                "vah": vah,
                "poc": poc,
                "val": val,
                "vwap": vwap,
                "rsi": rsi
            }
        
        if not tf_data:
            return None
        
        # Run MTF analysis WITH range context (FIX #1 + #2 + #3)
        result = self.system.analyze_mtf(symbol, tf_data, current_price, range_ctx=range_ctx)
        
        return result
    
    def scan_symbols(self, 
                     symbols: List[str],
                     timeframe: str = "1HR") -> List[AnalysisResult]:
        """
        Scan multiple symbols
        
        Args:
            symbols: List of symbols to scan
            timeframe: Timeframe for analysis
        
        Returns:
            List of AnalysisResult sorted by actionability
        """
        results = []
        
        for i, symbol in enumerate(symbols):
            print(f"Scanning {symbol} ({i+1}/{len(symbols)})...", end=" ")
            
            result = self.analyze(symbol, timeframe)
            
            if result:
                results.append(result)
                print(f"{result.signal_emoji} {result.signal}")
            else:
                print("âš ï¸ No data")
            
            # Rate limiting (Finnhub free tier: 60 calls/min)
            if i < len(symbols) - 1:
                time.sleep(1)
        
        # Sort by signal quality
        def sort_key(r):
            signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2}
            return (signal_order.get(r.signal, 3), -r.confidence)
        
        results.sort(key=sort_key)
        
        return results
    
    def scan_mtf(self, 
                 symbols: List[str]) -> List[MTFResult]:
        """
        Multi-timeframe scan of multiple symbols
        
        Args:
            symbols: List of symbols to scan
        
        Returns:
            List of MTFResult sorted by actionability
        """
        results = []
        
        for i, symbol in enumerate(symbols):
            print(f"MTF Scanning {symbol} ({i+1}/{len(symbols)})...")
            
            result = self.analyze_mtf(symbol)
            
            if result:
                results.append(result)
                print(f"   {result.signal_emoji} {result.dominant_signal} ({result.confluence_pct:.0f}% confluence)")
            else:
                print("   âš ï¸ No data")
            
            # Rate limiting
            if i < len(symbols) - 1:
                time.sleep(2)  # More calls per symbol for MTF
        
        # Sort by signal quality
        def sort_key(r):
            signal_order = {"LONG_SETUP": 0, "SHORT_SETUP": 1, "YELLOW": 2}
            return (signal_order.get(r.dominant_signal, 3), -r.confluence_pct)
        
        results.sort(key=sort_key)
        
        return results
    
    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================
    
    def print_analysis(self, result: AnalysisResult) -> str:
        """Print single timeframe analysis"""
        return self.system.print_result(result)
    
    def print_mtf_analysis(self, result: MTFResult) -> str:
        """Print MTF analysis"""
        return self.system.print_mtf_result(result)
    
    def print_scan_summary(self, results: List[AnalysisResult]) -> str:
        """Print scan summary"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"SCAN RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        
        long_setups = [r for r in results if r.signal == "LONG_SETUP"]
        short_setups = [r for r in results if r.signal == "SHORT_SETUP"]
        yellow = [r for r in results if r.signal == "YELLOW"]
        
        lines.append(f"\nðŸŸ¢ LONG SETUPS: {len(long_setups)}")
        for r in long_setups:
            lines.append(f"   {r.timeframe}: Bull {r.bull_score:.0f} | Conf {r.confidence:.0f}%")
        
        lines.append(f"\nðŸ”´ SHORT SETUPS: {len(short_setups)}")
        for r in short_setups:
            lines.append(f"   {r.timeframe}: Bear {r.bear_score:.0f} | Conf {r.confidence:.0f}%")
        
        lines.append(f"\nðŸŸ¡ YELLOW (Watch): {len(yellow)}")
        for r in yellow[:5]:  # Top 5
            lean = "Bull" if r.bull_score > r.bear_score else "Bear"
            lines.append(f"   {r.timeframe}: {lean} lean | Conf {r.confidence:.0f}%")
        
        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

def quick_analyze(symbol: str, api_key: str = None) -> Optional[AnalysisResult]:
    """Quick single-symbol analysis"""
    scanner = MarketScanner(api_key)
    return scanner.analyze(symbol)


def quick_mtf(symbol: str, api_key: str = None) -> Optional[MTFResult]:
    """Quick MTF analysis"""
    scanner = MarketScanner(api_key)
    return scanner.analyze_mtf(symbol)


def quick_scan(symbols: List[str], api_key: str = None) -> List[AnalysisResult]:
    """Quick scan multiple symbols"""
    scanner = MarketScanner(api_key)
    return scanner.scan_symbols(symbols)


# =============================================================================
# BACKWARDS COMPATIBILITY ALIAS
# =============================================================================
# FinnhubScanner is the old name, MarketScanner is the new name
# Both work identically - use MarketScanner for new code
FinnhubScanner = MarketScanner


# =============================================================================
# DEMO / CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("MARKET SCANNER (Polygon > Alpaca > Finnhub > yfinance)")
    print("=" * 70)
    
    # Check for API keys
    polygon_key = os.environ.get("POLYGON_API_KEY")
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    
    if not any([polygon_key, alpaca_key, finnhub_key]):
        print("\nâš ï¸  No API keys found!")
        print("\nData sources (set at least one):")
        print("  - POLYGON_API_KEY (recommended - real-time with paid plan)")
        print("  - ALPACA_API_KEY + ALPACA_SECRET_KEY")
        print("  - FINNHUB_API_KEY")
        print("\nFallback: yfinance (delayed but free)")
        print("=" * 70)
        
        # Demo with placeholder
        print("\nðŸ“‹ DEMO MODE (showing structure only):")
        print("""
Usage Examples:

    from finnhub_scanner import MarketScanner
    
    # Initialize (uses env vars for API keys)
    scanner = MarketScanner()
    
    # Single symbol
    result = scanner.analyze("META", timeframe="2HR")
    print(scanner.print_analysis(result))
    
    # Multi-timeframe
    mtf = scanner.analyze_mtf("META")
    print(scanner.print_mtf_analysis(mtf))
    
    # Scan watchlist
    symbols = ["META", "AAPL", "NVDA", "GOOGL", "MSFT"]
    results = scanner.scan_symbols(symbols)
    print(scanner.print_scan_summary(results))
    
    # MTF scan
    mtf_results = scanner.scan_mtf(symbols)
    for r in mtf_results:
        print(f"{r.symbol}: {r.signal_emoji} {r.dominant_signal}")
""")
    else:
        sources = []
        if polygon_key: sources.append("Polygon")
        if alpaca_key: sources.append("Alpaca")
        if finnhub_key: sources.append("Finnhub")
        print(f"\nâœ… Data sources: {', '.join(sources)}")
        
        # Run demo
        scanner = MarketScanner(finnhub_key or "demo")
        
        symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        
        print(f"\nðŸ“Š Analyzing {symbol}...")
        
        # Single timeframe
        result = scanner.analyze(symbol, "1HR")
        if result:
            print(scanner.print_analysis(result))
        
        print(f"\nðŸ“Š MTF Analysis for {symbol}...")
        
        # MTF
        mtf = scanner.analyze_mtf(symbol)
        if mtf:
            print(scanner.print_mtf_analysis(mtf))
        
        print("\nâœ… Scanner ready!")
