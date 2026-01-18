"""
Finnhub Integration for MTF Auction Scanner
============================================
Pulls real-time data from Finnhub and runs through the scanner.

Setup:
    1. Get free API key from https://finnhub.io
    2. Set environment variable: export FINNHUB_API_KEY=your_key
    3. Or pass key directly to FinnhubScanner()

Author: Rob's Trading Systems
Version: 1.0.0
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
from chart_input_analyzer import ChartInputSystem, ChartInput, AnalysisResult, MTFResult


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


# =============================================================================
# FINNHUB SCANNER
# =============================================================================

class FinnhubScanner:
    """
    Real-time scanner using Finnhub data
    
    Usage:
        scanner = FinnhubScanner("YOUR_API_KEY")
        
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
                print("✅ Polygon.io real-time data enabled")
            except Exception as e:
                print(f"⚠️ Polygon init failed: {e}")
        
        # Initialize Alpaca client if credentials available
        self.alpaca_client = None
        alpaca_key = os.environ.get("ALPACA_API_KEY")
        alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
        if alpaca_available and alpaca_key and alpaca_secret:
            try:
                self.alpaca_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                print("✅ Alpaca real-time data enabled")
            except Exception as e:
                print(f"⚠️ Alpaca init failed: {e}")
    
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
                print(f"⚠️ Polygon: No data for {symbol}")
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
            
            print(f"✅ Polygon: Got {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Polygon error for {symbol}: {e}")
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
                print(f"⚠️ Alpaca: No data for {symbol}")
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
            
            print(f"✅ Alpaca: Got {len(df)} real-time candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Alpaca error for {symbol}: {e}")
            return None
    
    def _get_candles_yfinance(self,
                              symbol: str,
                              resolution: str = "60",
                              days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Fallback: Fetch candle data from Yahoo Finance (free, no API key needed)
        """
        if yf is None:
            print("⚠️ yfinance not installed. Run: pip install yfinance")
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
                print(f"⚠️ yfinance: No data for {symbol}")
                return None
            
            # Normalize column names
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index.name = 'timestamp'
            
            print(f"✅ yfinance: Got {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ yfinance error for {symbol}: {e}")
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
                    print(f"✅ Finnhub: Got {len(df)} candles for {symbol}")
                else:
                    print(f"⚠️ Finnhub: No data for {symbol}")
                
            except Exception as e:
                error_str = str(e)
                if "403" in error_str:
                    print(f"⚠️ Finnhub 403 (no candle access)")
                else:
                    print(f"❌ Finnhub error: {e}")
        
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
        
        resample_map = {
            "30MIN": "30T",
            "1HR": "1H",
            "2HR": "2H",
            "4HR": "4H",
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
        """Get real-time quote"""
        try:
            quote = self.client.quote(symbol)
            return {
                'current': quote.get('c'),
                'open': quote.get('o'),
                'high': quote.get('h'),
                'low': quote.get('l'),
                'prev_close': quote.get('pc'),
                'change': quote.get('d'),
                'change_pct': quote.get('dp'),
                'timestamp': datetime.fromtimestamp(quote.get('t', 0))
            }
        except Exception as e:
            print(f"❌ Error getting quote for {symbol}: {e}")
            return None
    
    def analyze(self, 
                symbol: str, 
                timeframe: str = "1HR",
                days_back: int = 20) -> Optional[AnalysisResult]:
        """
        Analyze a single symbol/timeframe
        
        Args:
            symbol: Stock symbol
            timeframe: "30MIN", "1HR", "2HR", "4HR", "DAILY"
            days_back: Days of history for calculations
        
        Returns:
            AnalysisResult with signal and levels
        """
        # Get base data (hourly for most, daily for daily)
        if timeframe.upper() == "DAILY":
            resolution = "D"
        else:
            resolution = "60"  # Get hourly, resample as needed
        
        df = self._get_candles(symbol, resolution, days_back)
        
        if df is None or len(df) < 20:
            print(f"⚠️ Insufficient data for {symbol}")
            return None
        
        # Resample if needed
        if timeframe.upper() in ["2HR", "4HR", "30MIN"]:
            df = self._resample_to_timeframe(df, timeframe)
        
        if len(df) < 10:
            return None
        
        # Get current price (from latest candle or real-time quote)
        quote = self.get_quote(symbol)
        current_price = quote['current'] if quote else df['close'].iloc[-1]
        
        # Calculate technicals
        poc, vah, val = self.calc.calculate_volume_profile(df)
        vwap = self.calc.calculate_vwap(df)
        rsi = self.calc.calculate_rsi(df)
        
        # Run through analyzer
        result = self.system.analyze(
            symbol=symbol,
            price=current_price,
            vah=vah,
            poc=poc,
            val=val,
            vwap=vwap,
            rsi=rsi,
            timeframe=timeframe
        )
        
        return result
    
    def analyze_mtf(self, 
                    symbol: str,
                    timeframes: List[str] = None) -> Optional[MTFResult]:
        """
        Multi-timeframe analysis
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes (default: ["30MIN", "1HR", "2HR", "4HR"])
        
        Returns:
            MTFResult with combined signal
        """
        if timeframes is None:
            timeframes = ["30MIN", "1HR", "2HR", "4HR"]
        
        # Get base hourly data
        df_hourly = self._get_candles(symbol, "60", days_back=30)
        
        if df_hourly is None or len(df_hourly) < 50:
            print(f"⚠️ Insufficient data for MTF analysis on {symbol}")
            return None
        
        # Get current price
        quote = self.get_quote(symbol)
        current_price = quote['current'] if quote else df_hourly['close'].iloc[-1]
        
        # Calculate for each timeframe
        tf_data = {}
        
        for tf in timeframes:
            # Resample data
            if tf.upper() == "30MIN":
                # For 30min, we'd need 30min data from Finnhub
                # Using hourly as approximation (or fetch 30min separately)
                df = df_hourly.copy()
            else:
                df = self._resample_to_timeframe(df_hourly, tf)
            
            if len(df) < 10:
                continue
            
            # Calculate technicals
            poc, vah, val = self.calc.calculate_volume_profile(df)
            vwap = self.calc.calculate_vwap(df)
            rsi = self.calc.calculate_rsi(df)
            
            tf_data[tf.upper()] = {
                "price": current_price,
                "vah": vah,
                "poc": poc,
                "val": val,
                "vwap": vwap,
                "rsi": rsi
            }
        
        if not tf_data:
            return None
        
        # Run MTF analysis
        result = self.system.analyze_mtf(symbol, tf_data, current_price)
        
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
                print("⚠️ No data")
            
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
                print("   ⚠️ No data")
            
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
        
        lines.append(f"\n🟢 LONG SETUPS: {len(long_setups)}")
        for r in long_setups:
            lines.append(f"   {r.timeframe}: Bull {r.bull_score:.0f} | Conf {r.confidence:.0f}%")
        
        lines.append(f"\n🔴 SHORT SETUPS: {len(short_setups)}")
        for r in short_setups:
            lines.append(f"   {r.timeframe}: Bear {r.bear_score:.0f} | Conf {r.confidence:.0f}%")
        
        lines.append(f"\n🟡 YELLOW (Watch): {len(yellow)}")
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
    scanner = FinnhubScanner(api_key)
    return scanner.analyze(symbol)


def quick_mtf(symbol: str, api_key: str = None) -> Optional[MTFResult]:
    """Quick MTF analysis"""
    scanner = FinnhubScanner(api_key)
    return scanner.analyze_mtf(symbol)


def quick_scan(symbols: List[str], api_key: str = None) -> List[AnalysisResult]:
    """Quick scan multiple symbols"""
    scanner = FinnhubScanner(api_key)
    return scanner.scan_symbols(symbols)


# =============================================================================
# DEMO / CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("FINNHUB SCANNER")
    print("=" * 70)
    
    # Check for API key
    api_key = os.environ.get("FINNHUB_API_KEY")
    
    if not api_key:
        print("\n⚠️  FINNHUB_API_KEY not set!")
        print("\nTo use this scanner:")
        print("  1. Get free API key from https://finnhub.io")
        print("  2. Set environment variable:")
        print("     export FINNHUB_API_KEY=your_key_here")
        print("  3. Run again")
        print("\nOr pass key directly:")
        print('  scanner = FinnhubScanner("your_key_here")')
        print("=" * 70)
        
        # Demo with placeholder
        print("\n📋 DEMO MODE (showing structure only):")
        print("""
Usage Examples:

    from finnhub_scanner import FinnhubScanner
    
    # Initialize
    scanner = FinnhubScanner("YOUR_API_KEY")
    
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
        print(f"\n✅ API Key found!")
        
        # Run demo
        scanner = FinnhubScanner(api_key)
        
        symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        
        print(f"\n📊 Analyzing {symbol}...")
        
        # Single timeframe
        result = scanner.analyze(symbol, "1HR")
        if result:
            print(scanner.print_analysis(result))
        
        print(f"\n📊 MTF Analysis for {symbol}...")
        
        # MTF
        mtf = scanner.analyze_mtf(symbol)
        if mtf:
            print(scanner.print_mtf_analysis(mtf))
        
        print("\n✅ Scanner ready!")
