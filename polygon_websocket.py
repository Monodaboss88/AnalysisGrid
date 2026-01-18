"""
Polygon.io WebSocket Streaming Client
======================================
Real-time minute bars streamed directly to your scanner - no polling!

Features:
- Live minute bar aggregates (AM.* messages)
- Auto-reconnect on disconnect
- Symbol subscription management
- Callback system for real-time updates

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠️ websockets not installed. Run: pip install websockets")


@dataclass
class MinuteBar:
    """Real-time minute bar data"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    timestamp: datetime
    trades: int = 0
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'timestamp': self.timestamp.isoformat(),
            'trades': self.trades
        }


@dataclass
class StreamingState:
    """Tracks the state of streaming connections"""
    connected: bool = False
    authenticated: bool = False
    subscribed_symbols: Set[str] = field(default_factory=set)
    last_message_time: Optional[datetime] = None
    reconnect_count: int = 0
    error_message: Optional[str] = None


class PolygonWebSocket:
    """
    Real-time WebSocket streaming from Polygon.io
    
    Usage:
        streamer = PolygonWebSocket(api_key="your_key")
        
        # Add callback for live bars
        streamer.on_bar(lambda bar: print(f"Live: {bar.symbol} @ {bar.close}"))
        
        # Subscribe to symbols
        await streamer.subscribe(["AAPL", "TSLA", "NVDA"])
        
        # Start streaming (runs in background)
        await streamer.start()
    """
    
    POLYGON_WS_URL = "wss://socket.polygon.io/stocks"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key required for WebSocket streaming")
        
        self.state = StreamingState()
        self._callbacks: List[Callable[[MinuteBar], None]] = []
        self._status_callbacks: List[Callable[[StreamingState], None]] = []
        self._ws = None
        self._running = False
        self._loop = None
        self._thread = None
        
        # Store latest bars per symbol
        self.latest_bars: Dict[str, MinuteBar] = {}
        
        # Accumulated bars for scanner (keyed by symbol -> list of bars)
        self.bar_history: Dict[str, List[MinuteBar]] = defaultdict(list)
        self.max_history = 500  # Keep last 500 bars per symbol
        
    def on_bar(self, callback: Callable[[MinuteBar], None]):
        """Register callback for live minute bars"""
        self._callbacks.append(callback)
        
    def on_status(self, callback: Callable[[StreamingState], None]):
        """Register callback for connection status changes"""
        self._status_callbacks.append(callback)
        
    def _notify_bar(self, bar: MinuteBar):
        """Notify all registered callbacks"""
        for cb in self._callbacks:
            try:
                cb(bar)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")
                
    def _notify_status(self):
        """Notify status change"""
        for cb in self._status_callbacks:
            try:
                cb(self.state)
            except Exception as e:
                print(f"⚠️ Status callback error: {e}")
    
    async def _authenticate(self):
        """Authenticate with Polygon"""
        auth_msg = {"action": "auth", "params": self.api_key}
        await self._ws.send(json.dumps(auth_msg))
        print("🔐 Sent authentication...")
        
    async def _subscribe(self, symbols: List[str]):
        """Subscribe to minute aggregates for symbols"""
        if not symbols:
            return
            
        # AM.* = Minute Aggregates
        params = ",".join([f"AM.{s.upper()}" for s in symbols])
        sub_msg = {"action": "subscribe", "params": params}
        await self._ws.send(json.dumps(sub_msg))
        
        self.state.subscribed_symbols.update(s.upper() for s in symbols)
        print(f"📡 Subscribed to {len(symbols)} symbols: {params}")
        self._notify_status()
        
    async def _unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if not symbols:
            return
            
        params = ",".join([f"AM.{s.upper()}" for s in symbols])
        unsub_msg = {"action": "unsubscribe", "params": params}
        await self._ws.send(json.dumps(unsub_msg))
        
        for s in symbols:
            self.state.subscribed_symbols.discard(s.upper())
        print(f"🔕 Unsubscribed from: {params}")
        self._notify_status()
        
    def _parse_minute_bar(self, data: dict) -> Optional[MinuteBar]:
        """Parse AM (minute aggregate) message"""
        try:
            return MinuteBar(
                symbol=data.get('sym', ''),
                open=float(data.get('o', 0)),
                high=float(data.get('h', 0)),
                low=float(data.get('l', 0)),
                close=float(data.get('c', 0)),
                volume=int(data.get('v', 0)),
                vwap=float(data.get('vw', 0)),
                timestamp=datetime.fromtimestamp(data.get('s', 0) / 1000),
                trades=int(data.get('n', 0))
            )
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return None
            
    async def _handle_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle array of messages
            if isinstance(data, list):
                for item in data:
                    await self._process_single_message(item)
            else:
                await self._process_single_message(data)
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")
            
    async def _process_single_message(self, msg: dict):
        """Process a single message"""
        ev = msg.get('ev')
        
        if ev == 'status':
            status = msg.get('status')
            message = msg.get('message', '')
            
            if status == 'connected':
                print("✅ WebSocket connected")
                self.state.connected = True
                await self._authenticate()
                
            elif status == 'auth_success':
                print("✅ Authenticated successfully")
                self.state.authenticated = True
                self.state.error_message = None
                self._notify_status()
                
                # Re-subscribe if we have pending symbols
                if self.state.subscribed_symbols:
                    await self._subscribe(list(self.state.subscribed_symbols))
                    
            elif status == 'auth_failed':
                print(f"❌ Authentication failed: {message}")
                self.state.authenticated = False
                self.state.error_message = f"Auth failed: {message}"
                self._notify_status()
                
            elif status == 'success':
                print(f"✅ {message}")
                
            else:
                print(f"📨 Status: {status} - {message}")
                
        elif ev == 'AM':
            # Minute Aggregate bar
            bar = self._parse_minute_bar(msg)
            if bar:
                self.state.last_message_time = datetime.now()
                self.latest_bars[bar.symbol] = bar
                
                # Add to history
                self.bar_history[bar.symbol].append(bar)
                if len(self.bar_history[bar.symbol]) > self.max_history:
                    self.bar_history[bar.symbol] = self.bar_history[bar.symbol][-self.max_history:]
                
                # Notify callbacks
                self._notify_bar(bar)
                
        elif ev == 'A':
            # Second aggregate (we don't need this, but log it)
            pass
            
        elif ev == 'T':
            # Trade tick (we don't need this for minute bars)
            pass
            
    async def _connect_loop(self):
        """Main connection loop with auto-reconnect"""
        while self._running:
            try:
                print(f"🔌 Connecting to Polygon WebSocket...")
                
                async with websockets.connect(
                    self.POLYGON_WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self._ws = ws
                    self.state.connected = True
                    self.state.error_message = None
                    self._notify_status()
                    
                    # Listen for messages
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed as e:
                print(f"🔌 Connection closed: {e}")
                self.state.connected = False
                self.state.authenticated = False
                self._notify_status()
                
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                self.state.connected = False
                self.state.authenticated = False
                self.state.error_message = str(e)
                self._notify_status()
                
            if self._running:
                self.state.reconnect_count += 1
                wait_time = min(30, 2 ** self.state.reconnect_count)
                print(f"🔄 Reconnecting in {wait_time}s (attempt {self.state.reconnect_count})...")
                await asyncio.sleep(wait_time)
                
    async def start(self):
        """Start the WebSocket connection"""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets library not installed. Run: pip install websockets")
            
        self._running = True
        await self._connect_loop()
        
    async def stop(self):
        """Stop the WebSocket connection"""
        self._running = False
        if self._ws:
            await self._ws.close()
        self.state.connected = False
        self.state.authenticated = False
        self._notify_status()
        print("🛑 WebSocket stopped")
        
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        if self.state.authenticated and self._ws:
            await self._subscribe(symbols)
        else:
            # Store for when we connect
            self.state.subscribed_symbols.update(s.upper() for s in symbols)
            print(f"📋 Queued {len(symbols)} symbols for subscription")
            
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if self.state.authenticated and self._ws:
            await self._unsubscribe(symbols)
        else:
            for s in symbols:
                self.state.subscribed_symbols.discard(s.upper())
                
    def start_background(self):
        """Start WebSocket in a background thread"""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.start())
            
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        print("🚀 WebSocket streaming started in background")
        
    def stop_background(self):
        """Stop background WebSocket"""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self.stop(), self._loop)
            
    def get_latest_bar(self, symbol: str) -> Optional[MinuteBar]:
        """Get the latest bar for a symbol"""
        return self.latest_bars.get(symbol.upper())
        
    def get_bar_history(self, symbol: str, count: int = 100) -> List[MinuteBar]:
        """Get recent bar history for a symbol"""
        bars = self.bar_history.get(symbol.upper(), [])
        return bars[-count:] if bars else []
        
    def get_status(self) -> dict:
        """Get current streaming status"""
        return {
            'connected': self.state.connected,
            'authenticated': self.state.authenticated,
            'subscribed_count': len(self.state.subscribed_symbols),
            'subscribed_symbols': list(self.state.subscribed_symbols),
            'last_message': self.state.last_message_time.isoformat() if self.state.last_message_time else None,
            'reconnect_count': self.state.reconnect_count,
            'error': self.state.error_message,
            'bars_cached': len(self.latest_bars)
        }


# =============================================================================
# STREAMING MANAGER - Singleton for app-wide streaming
# =============================================================================

class StreamingManager:
    """
    Singleton manager for WebSocket streaming
    
    Usage:
        manager = StreamingManager.get_instance()
        manager.start()
        manager.subscribe(["AAPL", "TSLA"])
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'StreamingManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        self.streamer: Optional[PolygonWebSocket] = None
        self._bar_callbacks: List[Callable[[MinuteBar], None]] = []
        
    def initialize(self, api_key: str = None):
        """Initialize the streamer"""
        try:
            self.streamer = PolygonWebSocket(api_key)
            
            # Forward bar events
            self.streamer.on_bar(self._on_bar)
            
            print("✅ StreamingManager initialized")
            return True
        except Exception as e:
            print(f"❌ StreamingManager init failed: {e}")
            return False
            
    def _on_bar(self, bar: MinuteBar):
        """Handle incoming bar"""
        for cb in self._bar_callbacks:
            try:
                cb(bar)
            except Exception as e:
                print(f"⚠️ Bar callback error: {e}")
                
    def on_bar(self, callback: Callable[[MinuteBar], None]):
        """Register a callback for live bars"""
        self._bar_callbacks.append(callback)
        
    def start(self):
        """Start streaming in background"""
        if self.streamer:
            self.streamer.start_background()
            
    def stop(self):
        """Stop streaming"""
        if self.streamer:
            self.streamer.stop_background()
            
    def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        if self.streamer and self.streamer._loop:
            asyncio.run_coroutine_threadsafe(
                self.streamer.subscribe(symbols),
                self.streamer._loop
            )
        elif self.streamer:
            self.streamer.state.subscribed_symbols.update(s.upper() for s in symbols)
            
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if self.streamer and self.streamer._loop:
            asyncio.run_coroutine_threadsafe(
                self.streamer.unsubscribe(symbols),
                self.streamer._loop
            )
            
    def get_latest(self, symbol: str) -> Optional[dict]:
        """Get latest bar as dict"""
        if self.streamer:
            bar = self.streamer.get_latest_bar(symbol)
            return bar.to_dict() if bar else None
        return None
        
    def get_status(self) -> dict:
        """Get streaming status"""
        if self.streamer:
            return self.streamer.get_status()
        return {'connected': False, 'error': 'Not initialized'}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    async def test():
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            print("Set POLYGON_API_KEY environment variable")
            sys.exit(1)
            
        streamer = PolygonWebSocket(api_key)
        
        # Print live bars
        streamer.on_bar(lambda bar: print(f"📊 {bar.symbol}: ${bar.close:.2f} (vol: {bar.volume:,})"))
        
        # Subscribe to some symbols
        await streamer.subscribe(["AAPL", "TSLA", "NVDA", "META", "GOOGL"])
        
        # Run for 60 seconds
        print("🎯 Streaming for 60 seconds...")
        try:
            await asyncio.wait_for(streamer.start(), timeout=60)
        except asyncio.TimeoutError:
            print("⏱️ Test complete")
        finally:
            await streamer.stop()
            
    asyncio.run(test())
