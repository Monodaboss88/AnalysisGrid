"""
SEF Trading System - Real-Time WebSocket Stream
Polygon.io WebSocket integration for live data

Channels:
- AM.*  : Minute aggregates (what we want for live bars)
- A.*   : Second aggregates  
- T.*   : Trades
- Q.*   : Quotes

This replaces polling with push-based real-time data.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class PolygonChannel(Enum):
    """Polygon WebSocket channels"""
    TRADES = "T"           # Individual trades
    QUOTES = "Q"           # NBBO quotes
    SECOND_AGG = "A"       # Per-second aggregates
    MINUTE_AGG = "AM"      # Per-minute aggregates (main one we use)


@dataclass
class LiveBar:
    """Real-time bar from WebSocket"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trades: int  # Number of trades in bar
    
    # For building higher timeframes
    is_complete: bool = True  # Minute bar is complete
    
    @classmethod
    def from_polygon_message(cls, msg: Dict) -> 'LiveBar':
        """Parse Polygon AM.* message into LiveBar"""
        return cls(
            symbol=msg.get("sym", ""),
            timestamp=datetime.fromtimestamp(msg.get("s", 0) / 1000),  # Start timestamp
            open=msg.get("o", 0),
            high=msg.get("h", 0),
            low=msg.get("l", 0),
            close=msg.get("c", 0),
            volume=int(msg.get("v", 0)),
            vwap=msg.get("vw", 0),
            trades=int(msg.get("n", 0)),
            is_complete=True
        )
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trades": self.trades
        }


@dataclass
class LiveQuote:
    """Real-time quote from WebSocket"""
    symbol: str
    timestamp: datetime
    bid: float
    bid_size: int
    ask: float
    ask_size: int
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @classmethod
    def from_polygon_message(cls, msg: Dict) -> 'LiveQuote':
        return cls(
            symbol=msg.get("sym", ""),
            timestamp=datetime.fromtimestamp(msg.get("t", 0) / 1000000000),  # Nanoseconds
            bid=msg.get("bp", 0),
            bid_size=int(msg.get("bs", 0)),
            ask=msg.get("ap", 0),
            ask_size=int(msg.get("as", 0))
        )


@dataclass
class LiveTrade:
    """Real-time trade from WebSocket"""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: int
    conditions: List[int]
    
    @classmethod
    def from_polygon_message(cls, msg: Dict) -> 'LiveTrade':
        return cls(
            symbol=msg.get("sym", ""),
            timestamp=datetime.fromtimestamp(msg.get("t", 0) / 1000000000),
            price=msg.get("p", 0),
            size=int(msg.get("s", 0)),
            exchange=msg.get("x", 0),
            conditions=msg.get("c", [])
        )


class BarAggregator:
    """
    Aggregates minute bars into higher timeframes in real-time
    Maintains running 2H bars from incoming 1-minute data
    """
    
    def __init__(self, target_minutes: int = 120):
        self.target_minutes = target_minutes
        
        # Current building bars by symbol
        self._building_bars: Dict[str, Dict] = {}
        
        # Completed bars
        self._completed_bars: Dict[str, List[LiveBar]] = defaultdict(list)
        
        # Callbacks for completed bars
        self._on_bar_complete: Optional[Callable] = None
    
    def set_callback(self, callback: Callable[[str, LiveBar], Any]):
        """Set callback for when a higher-TF bar completes"""
        self._on_bar_complete = callback
    
    def add_minute_bar(self, bar: LiveBar) -> Optional[LiveBar]:
        """
        Add a minute bar and return completed higher-TF bar if ready
        
        Args:
            bar: Incoming 1-minute bar
            
        Returns:
            Completed higher-TF bar if period complete, else None
        """
        symbol = bar.symbol
        
        if symbol not in self._building_bars:
            # Start new aggregation period
            self._building_bars[symbol] = {
                "start_time": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap_volume": bar.vwap * bar.volume,
                "trades": bar.trades,
                "bar_count": 1
            }
            return None
        
        building = self._building_bars[symbol]
        
        # Check if we've completed the target period
        elapsed = (bar.timestamp - building["start_time"]).total_seconds() / 60
        
        if elapsed >= self.target_minutes:
            # Complete the bar
            completed = self._finalize_bar(symbol, building)
            
            # Start new bar
            self._building_bars[symbol] = {
                "start_time": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap_volume": bar.vwap * bar.volume,
                "trades": bar.trades,
                "bar_count": 1
            }
            
            # Store and callback
            self._completed_bars[symbol].append(completed)
            
            if self._on_bar_complete:
                asyncio.create_task(self._async_callback(symbol, completed))
            
            return completed
        
        # Update building bar
        building["high"] = max(building["high"], bar.high)
        building["low"] = min(building["low"], bar.low)
        building["close"] = bar.close
        building["volume"] += bar.volume
        building["vwap_volume"] += bar.vwap * bar.volume
        building["trades"] += bar.trades
        building["bar_count"] += 1
        
        return None
    
    def _finalize_bar(self, symbol: str, building: Dict) -> LiveBar:
        """Create completed LiveBar from building data"""
        total_volume = building["volume"]
        vwap = building["vwap_volume"] / total_volume if total_volume > 0 else building["close"]
        
        return LiveBar(
            symbol=symbol,
            timestamp=building["start_time"],
            open=building["open"],
            high=building["high"],
            low=building["low"],
            close=building["close"],
            volume=total_volume,
            vwap=vwap,
            trades=building["trades"],
            is_complete=True
        )
    
    async def _async_callback(self, symbol: str, bar: LiveBar):
        """Async wrapper for callback"""
        try:
            result = self._on_bar_complete(symbol, bar)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Bar callback error: {e}")
    
    def get_current_bar(self, symbol: str) -> Optional[Dict]:
        """Get the current in-progress bar for a symbol"""
        if symbol in self._building_bars:
            b = self._building_bars[symbol]
            return {
                "symbol": symbol,
                "timestamp": b["start_time"],
                "open": b["open"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "volume": b["volume"],
                "elapsed_minutes": (datetime.now() - b["start_time"]).total_seconds() / 60,
                "bars_aggregated": b["bar_count"]
            }
        return None
    
    def get_completed_bars(self, symbol: str, n: int = 10) -> List[LiveBar]:
        """Get last N completed bars for a symbol"""
        return self._completed_bars[symbol][-n:]


class PolygonWebSocketClient:
    """
    Production-ready WebSocket client for Polygon.io
    
    Features:
    - Auto-reconnect
    - Heartbeat monitoring
    - Message queuing
    - Multiple symbol subscriptions
    - Bar aggregation to 2H
    """
    
    WS_URL = "wss://socket.polygon.io/stocks"
    RECONNECT_DELAY = 5  # Seconds before reconnect attempt
    HEARTBEAT_INTERVAL = 30  # Seconds between heartbeats
    
    def __init__(self, api_key: str, target_timeframe_minutes: int = 120):
        self.api_key = api_key
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # State
        self._running = False
        self._authenticated = False
        self._subscriptions: List[str] = []
        
        # Bar aggregation
        self.aggregator = BarAggregator(target_timeframe_minutes)
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "bar": [],      # 1-minute bars
            "bar_2h": [],   # 2-hour bars (aggregated)
            "quote": [],
            "trade": [],
            "status": [],   # Connection status changes
        }
        
        # Message queue for processing
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
        # Latest data cache
        self._latest_bars: Dict[str, LiveBar] = {}
        self._latest_quotes: Dict[str, LiveQuote] = {}
        
        # Stats
        self._messages_received = 0
        self._bars_received = 0
        self._last_message_time: Optional[datetime] = None
    
    def on(self, event: str, callback: Callable):
        """
        Register callback for events
        
        Events:
        - 'bar': Called with (LiveBar) for each minute bar
        - 'bar_2h': Called with (symbol, LiveBar) for each 2H bar
        - 'quote': Called with (LiveQuote) for each quote
        - 'trade': Called with (LiveTrade) for each trade
        - 'status': Called with (status_dict) for connection changes
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        
        # Special handling for 2H bars
        if event == "bar_2h":
            self.aggregator.set_callback(callback)
    
    async def connect(self):
        """Establish WebSocket connection"""
        logger.info("Connecting to Polygon WebSocket...")
        
        self._session = aiohttp.ClientSession()
        
        try:
            self._ws = await self._session.ws_connect(self.WS_URL)
            self._running = True
            
            # Wait for connection message
            msg = await self._ws.receive()
            data = json.loads(msg.data)
            logger.info(f"Connected: {data}")
            
            # Authenticate
            await self._authenticate()
            
            # Notify status
            await self._emit("status", {"status": "connected", "authenticated": self._authenticated})
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self._emit("status", {"status": "error", "error": str(e)})
            raise
    
    async def _authenticate(self):
        """Authenticate with API key"""
        auth_msg = {"action": "auth", "params": self.api_key}
        await self._ws.send_json(auth_msg)
        
        msg = await self._ws.receive()
        data = json.loads(msg.data)
        
        if isinstance(data, list) and len(data) > 0:
            if data[0].get("status") == "auth_success":
                self._authenticated = True
                logger.info("Authentication successful")
            else:
                logger.error(f"Authentication failed: {data}")
                raise PermissionError("Polygon authentication failed")
    
    async def subscribe(self, symbols: List[str], channels: List[PolygonChannel] = None):
        """
        Subscribe to symbols
        
        Args:
            symbols: List of stock symbols
            channels: List of channels (default: MINUTE_AGG only)
        """
        if not self._authenticated:
            raise RuntimeError("Must connect and authenticate first")
        
        channels = channels or [PolygonChannel.MINUTE_AGG]
        
        # Build subscription list
        subs = []
        for symbol in symbols:
            for channel in channels:
                subs.append(f"{channel.value}.{symbol.upper()}")
        
        # Send subscription
        sub_msg = {"action": "subscribe", "params": ",".join(subs)}
        await self._ws.send_json(sub_msg)
        
        self._subscriptions.extend(subs)
        logger.info(f"Subscribed to: {subs}")
    
    async def unsubscribe(self, symbols: List[str], channels: List[PolygonChannel] = None):
        """Unsubscribe from symbols"""
        channels = channels or [PolygonChannel.MINUTE_AGG]
        
        unsubs = []
        for symbol in symbols:
            for channel in channels:
                unsubs.append(f"{channel.value}.{symbol.upper()}")
        
        unsub_msg = {"action": "unsubscribe", "params": ",".join(unsubs)}
        await self._ws.send_json(unsub_msg)
        
        for unsub in unsubs:
            if unsub in self._subscriptions:
                self._subscriptions.remove(unsub)
        
        logger.info(f"Unsubscribed from: {unsubs}")
    
    async def start(self):
        """Start listening for messages"""
        if not self._running:
            await self.connect()
        
        # Start background tasks
        asyncio.create_task(self._listen_loop())
        asyncio.create_task(self._process_loop())
        asyncio.create_task(self._heartbeat_loop())
        
        logger.info("WebSocket stream started")
    
    async def stop(self):
        """Stop the WebSocket client"""
        self._running = False
        
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        if self._session and not self._session.closed:
            await self._session.close()
        
        logger.info(f"WebSocket stopped. Messages received: {self._messages_received}, Bars: {self._bars_received}")
    
    async def _listen_loop(self):
        """Main loop for receiving messages"""
        while self._running:
            try:
                if self._ws is None or self._ws.closed:
                    await self._reconnect()
                    continue
                
                msg = await asyncio.wait_for(self._ws.receive(), timeout=60)
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._message_queue.put(msg.data)
                    self._messages_received += 1
                    self._last_message_time = datetime.now()
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket closed, reconnecting...")
                    await self._reconnect()
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    await self._reconnect()
                    
            except asyncio.TimeoutError:
                # No message in 60 seconds, check connection
                logger.debug("No messages for 60s, connection may be stale")
                
            except Exception as e:
                logger.error(f"Listen loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_loop(self):
        """Process messages from queue"""
        while self._running:
            try:
                data = await asyncio.wait_for(self._message_queue.get(), timeout=1)
                await self._handle_message(data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Process loop error: {e}")
    
    async def _heartbeat_loop(self):
        """Monitor connection health"""
        while self._running:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            
            if self._last_message_time:
                silence = (datetime.now() - self._last_message_time).total_seconds()
                if silence > 120:  # 2 minutes of silence
                    logger.warning(f"No messages for {silence:.0f}s, reconnecting...")
                    await self._reconnect()
    
    async def _reconnect(self):
        """Reconnect to WebSocket"""
        logger.info(f"Reconnecting in {self.RECONNECT_DELAY}s...")
        
        await self._emit("status", {"status": "reconnecting"})
        
        # Close existing connection
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        await asyncio.sleep(self.RECONNECT_DELAY)
        
        try:
            await self.connect()
            
            # Resubscribe
            if self._subscriptions:
                sub_msg = {"action": "subscribe", "params": ",".join(self._subscriptions)}
                await self._ws.send_json(sub_msg)
                logger.info(f"Resubscribed to: {self._subscriptions}")
            
            await self._emit("status", {"status": "reconnected"})
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await self._emit("status", {"status": "error", "error": str(e)})
    
    async def _handle_message(self, raw_data: str):
        """Parse and route incoming message"""
        try:
            data = json.loads(raw_data)
            
            if not isinstance(data, list):
                return
            
            for item in data:
                event_type = item.get("ev")
                
                if event_type == "AM":  # Minute aggregate
                    bar = LiveBar.from_polygon_message(item)
                    self._bars_received += 1
                    self._latest_bars[bar.symbol] = bar
                    
                    # Emit minute bar
                    await self._emit("bar", bar)
                    
                    # Aggregate to 2H
                    completed = self.aggregator.add_minute_bar(bar)
                    if completed:
                        await self._emit("bar_2h", (bar.symbol, completed))
                
                elif event_type == "Q":  # Quote
                    quote = LiveQuote.from_polygon_message(item)
                    self._latest_quotes[quote.symbol] = quote
                    await self._emit("quote", quote)
                
                elif event_type == "T":  # Trade
                    trade = LiveTrade.from_polygon_message(item)
                    await self._emit("trade", trade)
                
                elif event_type == "status":
                    logger.info(f"Status message: {item.get('message')}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def _emit(self, event: str, data: Any):
        """Emit event to all registered callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    # ========== Public Getters ==========
    
    def get_latest_bar(self, symbol: str) -> Optional[LiveBar]:
        """Get most recent minute bar for symbol"""
        return self._latest_bars.get(symbol.upper())
    
    def get_latest_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get most recent quote for symbol"""
        return self._latest_quotes.get(symbol.upper())
    
    def get_current_2h_bar(self, symbol: str) -> Optional[Dict]:
        """Get the in-progress 2H bar"""
        return self.aggregator.get_current_bar(symbol.upper())
    
    def get_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "running": self._running,
            "authenticated": self._authenticated,
            "subscriptions": self._subscriptions,
            "messages_received": self._messages_received,
            "bars_received": self._bars_received,
            "last_message": self._last_message_time.isoformat() if self._last_message_time else None,
            "symbols_tracking": list(self._latest_bars.keys())
        }


# ========== High-Level Stream Manager ==========

class LiveDataStream:
    """
    High-level manager for real-time data streaming
    Integrates with the trading system
    """
    
    def __init__(self, api_key: str, symbols: List[str], timeframe_minutes: int = 120):
        self.api_key = api_key
        self.symbols = [s.upper() for s in symbols]
        self.timeframe_minutes = timeframe_minutes
        
        self.ws_client = PolygonWebSocketClient(api_key, timeframe_minutes)
        
        # User callbacks
        self._on_bar_callbacks: List[Callable] = []
        self._on_2h_bar_callbacks: List[Callable] = []
    
    def on_minute_bar(self, callback: Callable[[LiveBar], Any]):
        """Register callback for minute bars"""
        self._on_bar_callbacks.append(callback)
    
    def on_2h_bar(self, callback: Callable[[str, LiveBar], Any]):
        """Register callback for completed 2H bars"""
        self._on_2h_bar_callbacks.append(callback)
    
    async def start(self):
        """Start the live data stream"""
        # Register internal handlers
        self.ws_client.on("bar", self._handle_bar)
        self.ws_client.on("bar_2h", self._handle_2h_bar)
        self.ws_client.on("status", self._handle_status)
        
        # Connect and subscribe
        await self.ws_client.connect()
        await self.ws_client.subscribe(self.symbols)
        await self.ws_client.start()
        
        logger.info(f"Live stream started for: {self.symbols}")
    
    async def stop(self):
        """Stop the live data stream"""
        await self.ws_client.stop()
    
    async def _handle_bar(self, bar: LiveBar):
        """Handle incoming minute bar"""
        for callback in self._on_bar_callbacks:
            try:
                result = callback(bar)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Minute bar callback error: {e}")
    
    async def _handle_2h_bar(self, data: tuple):
        """Handle completed 2H bar"""
        symbol, bar = data
        logger.info(f"2H BAR COMPLETE: {symbol} O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:,}")
        
        for callback in self._on_2h_bar_callbacks:
            try:
                result = callback(symbol, bar)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"2H bar callback error: {e}")
    
    def _handle_status(self, status: Dict):
        """Handle connection status changes"""
        logger.info(f"Stream status: {status}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from latest bar"""
        bar = self.ws_client.get_latest_bar(symbol)
        return bar.close if bar else None
    
    def get_stats(self) -> Dict:
        """Get stream statistics"""
        return self.ws_client.get_stats()


# ========== Example Usage ==========

async def example():
    """Example of using the live data stream"""
    import os
    
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("Set POLYGON_API_KEY environment variable")
        return
    
    symbols = ["SPY", "QQQ", "AAPL"]
    
    # Create stream
    stream = LiveDataStream(api_key, symbols, timeframe_minutes=120)
    
    # Register callbacks
    def on_minute(bar: LiveBar):
        print(f"[1m] {bar.symbol}: {bar.close:.2f} vol:{bar.volume:,}")
    
    def on_2h(symbol: str, bar: LiveBar):
        print(f"\n{'='*50}")
        print(f"[2H COMPLETE] {symbol}")
        print(f"  O: {bar.open:.2f}  H: {bar.high:.2f}  L: {bar.low:.2f}  C: {bar.close:.2f}")
        print(f"  Volume: {bar.volume:,}  VWAP: {bar.vwap:.2f}")
        print(f"{'='*50}\n")
    
    stream.on_minute_bar(on_minute)
    stream.on_2h_bar(on_2h)
    
    # Start streaming
    await stream.start()
    
    # Run for a while
    try:
        while True:
            await asyncio.sleep(60)
            print(f"\nStats: {stream.get_stats()}\n")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await stream.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example())
