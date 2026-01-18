"""
SEF Trading System - Polygon.io Data Client
Handles all market data retrieval from Polygon.io API

Polygon Advantages:
- Better tick-level data
- Faster updates
- More reliable historical data
- WebSocket support for real-time
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
import logging
from config import APIConfig, SessionConfig

logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """Standard OHLCV bar structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None  # Polygon provides VWAP per bar
    transactions: Optional[int] = None  # Number of transactions
    
    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3
    
    @property
    def ohlc4(self) -> float:
        return (self.open + self.high + self.low + self.close) / 4
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


class PolygonClient:
    """
    Async client for Polygon.io API
    Handles rate limiting and data normalization
    
    API Docs: https://polygon.io/docs/stocks
    """
    
    BASE_URL = "https://api.polygon.io"
    
    # Polygon timespan mapping
    TIMESPAN_MAP = {
        1: "minute",
        5: "minute",
        15: "minute",
        30: "minute",
        60: "hour",
        120: "hour",
        240: "hour",
        "D": "day",
        "W": "week",
        "M": "month"
    }
    
    # Multiplier for timespan
    MULTIPLIER_MAP = {
        1: 1,
        5: 5,
        15: 15,
        30: 30,
        60: 1,
        120: 2,
        240: 4,
        "D": 1,
        "W": 1,
        "M": 1
    }
    
    def __init__(self, config: APIConfig, session_config: SessionConfig):
        self.api_key = config.polygon_api_key
        self.session_config = session_config
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 100
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Make rate-limited API request"""
        session = await self._get_session()
        
        params = params or {}
        params["apiKey"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("Rate limit hit, waiting...")
                    await asyncio.sleep(60)  # Polygon rate limit reset
                    return await self._request(endpoint, params)
                
                if response.status == 403:
                    logger.error("API key invalid or insufficient permissions")
                    raise PermissionError("Invalid Polygon API key")
                
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"Polygon API error: {e}")
            raise
    
    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: Union[datetime, date, str],
        to_date: Union[datetime, date, str],
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000
    ) -> List[OHLCV]:
        """
        Get aggregate bars for a stock
        
        Args:
            symbol: Stock ticker
            multiplier: Size of the timespan multiplier (e.g., 5 for 5-minute)
            timespan: minute, hour, day, week, month
            from_date: Start date
            to_date: End date
            adjusted: Whether to adjust for splits
            sort: asc or desc
            limit: Max results (up to 50000)
            
        Returns:
            List of OHLCV bars
        """
        # Format dates
        if isinstance(from_date, datetime):
            from_str = from_date.strftime("%Y-%m-%d")
        elif isinstance(from_date, date):
            from_str = from_date.strftime("%Y-%m-%d")
        else:
            from_str = from_date
            
        if isinstance(to_date, datetime):
            to_str = to_date.strftime("%Y-%m-%d")
        elif isinstance(to_date, date):
            to_str = to_date.strftime("%Y-%m-%d")
        else:
            to_str = to_date
        
        endpoint = f"v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
        
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit
        }
        
        data = await self._request(endpoint, params)
        
        if data.get("status") != "OK" or "results" not in data:
            logger.warning(f"No data returned for {symbol}: {data.get('status')}")
            return []
        
        candles = []
        for bar in data["results"]:
            candle = OHLCV(
                timestamp=datetime.fromtimestamp(bar["t"] / 1000),  # Polygon uses milliseconds
                open=bar["o"],
                high=bar["h"],
                low=bar["l"],
                close=bar["c"],
                volume=int(bar["v"]),
                vwap=bar.get("vw"),  # Volume-weighted average price
                transactions=bar.get("n")  # Number of transactions
            )
            candles.append(candle)
        
        return candles
    
    async def get_candles(
        self,
        symbol: str,
        resolution: Union[int, str],
        from_ts: datetime,
        to_ts: datetime
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candles for a symbol (compatible interface with old client)
        
        Args:
            symbol: Stock ticker
            resolution: Candle resolution in minutes (1, 5, 15, 30, 60, 120, 240) or D/W/M
            from_ts: Start datetime
            to_ts: End datetime
            
        Returns:
            List of OHLCV bars
        """
        timespan = self.TIMESPAN_MAP.get(resolution, "minute")
        multiplier = self.MULTIPLIER_MAP.get(resolution, 1)
        
        return await self.get_aggregates(
            symbol=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_ts,
            to_date=to_ts
        )
    
    async def get_quote(self, symbol: str) -> Dict:
        """Get last trade (real-time quote)"""
        endpoint = f"v2/last/trade/{symbol.upper()}"
        data = await self._request(endpoint)
        
        if "results" in data:
            return {
                "price": data["results"].get("p"),
                "size": data["results"].get("s"),
                "timestamp": data["results"].get("t"),
                "exchange": data["results"].get("x")
            }
        return {}
    
    async def get_previous_close(self, symbol: str) -> Dict:
        """Get previous day's OHLCV"""
        endpoint = f"v2/aggs/ticker/{symbol.upper()}/prev"
        data = await self._request(endpoint)
        
        if "results" in data and data["results"]:
            bar = data["results"][0]
            return {
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "vwap": bar.get("vw")
            }
        return {}
    
    async def get_snapshot(self, symbol: str) -> Dict:
        """Get current snapshot (quote, day stats, prev day)"""
        endpoint = f"v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}"
        return await self._request(endpoint)
    
    async def get_intraday_candles(
        self,
        symbol: str,
        resolution_minutes: int = 5,
        days_back: int = 1
    ) -> pd.DataFrame:
        """
        Fetch intraday candles and return as DataFrame
        Filters to RTH only if configured
        """
        to_ts = datetime.now()
        from_ts = to_ts - timedelta(days=days_back)
        
        candles = await self.get_candles(symbol, resolution_minutes, from_ts, to_ts)
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "vwap": c.vwap,
                "transactions": c.transactions
            }
            for c in candles
        ])
        
        df.set_index("timestamp", inplace=True)
        
        # Filter to RTH if configured
        if self.session_config.use_rth_only:
            df = self._filter_rth(df)
        
        return df
    
    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to Regular Trading Hours only"""
        if df.empty:
            return df
            
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        market_open = self.session_config.market_open
        market_close = self.session_config.market_close
        
        mask = (df.index.time >= market_open) & (df.index.time < market_close)
        return df[mask]
    
    async def get_historical_daily(
        self,
        symbol: str,
        days_back: int = 252
    ) -> pd.DataFrame:
        """Fetch daily candles for longer-term analysis"""
        to_ts = datetime.now()
        from_ts = to_ts - timedelta(days=days_back)
        
        candles = await self.get_candles(symbol, "D", from_ts, to_ts)
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "vwap": c.vwap
            }
            for c in candles
        ])
        
        df.set_index("timestamp", inplace=True)
        return df
    
    async def get_grouped_daily(self, date_str: str) -> Dict[str, Dict]:
        """
        Get daily bars for all stocks on a given date
        Useful for scanning
        
        Args:
            date_str: Date in YYYY-MM-DD format
        """
        endpoint = f"v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        data = await self._request(endpoint)
        
        if "results" not in data:
            return {}
        
        return {
            bar["T"]: {
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"],
                "vwap": bar.get("vw")
            }
            for bar in data["results"]
        }


class PolygonWebSocket:
    """
    WebSocket client for real-time Polygon data
    
    Channels:
    - T.* : Trades
    - Q.* : Quotes
    - A.* : Second aggregates
    - AM.* : Minute aggregates
    """
    
    WS_URL = "wss://socket.polygon.io/stocks"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._ws = None
        self._subscriptions: List[str] = []
        self._callbacks: Dict[str, callable] = {}
        self._running = False
    
    async def connect(self):
        """Establish WebSocket connection"""
        import websockets
        
        self._ws = await websockets.connect(self.WS_URL)
        
        # Authenticate
        auth_msg = {"action": "auth", "params": self.api_key}
        await self._ws.send(str(auth_msg).replace("'", '"'))
        
        response = await self._ws.recv()
        logger.info(f"WebSocket auth response: {response}")
        
        self._running = True
    
    async def subscribe(self, channels: List[str], callback: callable):
        """
        Subscribe to channels
        
        Args:
            channels: List of channels like ["AM.SPY", "AM.QQQ"]
            callback: Function to call with each message
        """
        if not self._ws:
            await self.connect()
        
        sub_msg = {"action": "subscribe", "params": ",".join(channels)}
        await self._ws.send(str(sub_msg).replace("'", '"'))
        
        self._subscriptions.extend(channels)
        for channel in channels:
            self._callbacks[channel] = callback
        
        logger.info(f"Subscribed to: {channels}")
    
    async def listen(self):
        """Listen for messages"""
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                await self._handle_message(message)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    async def _handle_message(self, message: str):
        """Handle incoming message"""
        import json
        data = json.loads(message)
        
        for item in data:
            event_type = item.get("ev")
            symbol = item.get("sym")
            
            channel = f"{event_type}.{symbol}"
            if channel in self._callbacks:
                await self._callbacks[channel](item)
    
    async def close(self):
        """Close WebSocket connection"""
        self._running = False
        if self._ws:
            await self._ws.close()


class CandleAggregator:
    """
    Aggregates lower timeframe candles into higher timeframes
    Specifically handles 2-hour candle construction from minute data
    """
    
    def __init__(self, target_minutes: int = 120):
        self.target_minutes = target_minutes
        self._current_bar: Optional[OHLCV] = None
        self._bar_start_time: Optional[datetime] = None
    
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate minute/5-min data into target timeframe
        
        Args:
            df: DataFrame with OHLCV columns and datetime index
            
        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df
        
        # Resample to target timeframe
        rule = f"{self.target_minutes}min"
        
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        
        # Add vwap aggregation if available
        if "vwap" in df.columns:
            # Volume-weighted VWAP aggregation
            df["vwap_volume"] = df["vwap"] * df["volume"]
            agg_dict["vwap_volume"] = "sum"
        
        if "transactions" in df.columns:
            agg_dict["transactions"] = "sum"
        
        aggregated = df.resample(rule).agg(agg_dict).dropna()
        
        # Calculate proper VWAP for aggregated bars
        if "vwap_volume" in aggregated.columns:
            aggregated["vwap"] = aggregated["vwap_volume"] / aggregated["volume"]
            aggregated.drop("vwap_volume", axis=1, inplace=True)
        
        return aggregated
    
    def update_live(self, candle: OHLCV) -> Optional[OHLCV]:
        """
        Update with a new candle, return completed bar if ready
        
        Args:
            candle: New incoming candle
            
        Returns:
            Completed OHLCV bar if target period complete, else None
        """
        if self._current_bar is None:
            # Start new bar
            self._bar_start_time = candle.timestamp
            self._current_bar = OHLCV(
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                vwap=candle.vwap,
                transactions=candle.transactions
            )
            return None
        
        # Check if we've completed the target period
        elapsed = (candle.timestamp - self._bar_start_time).total_seconds() / 60
        
        if elapsed >= self.target_minutes:
            # Complete current bar and return it
            completed = self._current_bar
            
            # Start new bar
            self._bar_start_time = candle.timestamp
            self._current_bar = OHLCV(
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                vwap=candle.vwap,
                transactions=candle.transactions
            )
            
            return completed
        
        # Update current bar
        # Calculate running VWAP
        total_volume = self._current_bar.volume + candle.volume
        if total_volume > 0 and candle.vwap and self._current_bar.vwap:
            new_vwap = (
                (self._current_bar.vwap * self._current_bar.volume + candle.vwap * candle.volume)
                / total_volume
            )
        else:
            new_vwap = candle.vwap
        
        self._current_bar = OHLCV(
            timestamp=self._current_bar.timestamp,
            open=self._current_bar.open,
            high=max(self._current_bar.high, candle.high),
            low=min(self._current_bar.low, candle.low),
            close=candle.close,
            volume=total_volume,
            vwap=new_vwap,
            transactions=(
                (self._current_bar.transactions or 0) + (candle.transactions or 0)
            )
        )
        
        return None
    
    @property
    def current_bar(self) -> Optional[OHLCV]:
        """Get the current in-progress bar"""
        return self._current_bar


# Utility functions
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_vwap_from_bars(df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP from OHLCV data
    Note: Polygon provides bar-level VWAP, but this calculates cumulative session VWAP
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    
    return cum_pv / cum_vol
