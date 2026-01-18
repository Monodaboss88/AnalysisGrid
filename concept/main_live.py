"""
SEF Trading System - Live Real-Time Version
Uses Polygon WebSocket for push-based data instead of polling

Key difference from main.py:
- Receives minute bars in real-time via WebSocket
- Aggregates to 2H bars on the fly
- Triggers analysis when 2H bar completes
- No polling loops - event-driven
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Optional, Dict, List
from dataclasses import dataclass
import pytz

from config import SystemConfig, DEFAULT_CONFIG
from polygon_client import PolygonClient, calculate_atr
from realtime_stream import LiveDataStream, LiveBar, PolygonWebSocketClient
from volume_profile import MultiTimeframeVPEngine, MultiTimeframeVP
from vwap_engine import MultiTimeframeVWAPEngine, MultiTimeframeVWAP
from signal_generator import (
    SignalGenerator, SignalContext, Signal, SignalFilter,
    SignalType, SignalStrength
)
from execution import (
    TradierClient, PositionManager, ExecutionEngine,
    OrderSide, Position
)
from extension_predictor import (
    ExtensionDurationPredictor, CandleInExtension, 
    ExtensionAlert, TriggerLevel
)
from extension_dashboard import ExtensionDashboard, MultiSymbolDashboard


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SymbolState:
    """Per-symbol tracking state"""
    symbol: str
    
    # Latest data
    last_bar: Optional[LiveBar] = None
    last_2h_bar: Optional[LiveBar] = None
    last_signal: Optional[Signal] = None
    last_extension_alert: Optional[ExtensionAlert] = None
    
    # Volume Profile (needs historical data)
    vp: Optional[MultiTimeframeVP] = None
    vwap: Optional[MultiTimeframeVWAP] = None
    atr: float = 0.0
    
    # Stats
    bars_received: int = 0
    bars_2h_completed: int = 0
    signals_generated: int = 0


class LiveTradingSystem:
    """
    Real-time trading system using WebSocket streaming
    
    Event-driven architecture:
    1. WebSocket receives minute bars
    2. Bars aggregated to 2H
    3. On 2H bar complete -> run full analysis
    4. Check signals and extension predictor
    5. Execute if conditions met
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Timezone
        self.tz = pytz.timezone(self.config.session.timezone)
        
        # State
        self._running = False
        self._symbol_states: Dict[str, SymbolState] = {}
        
        # Stats
        self.total_signals = 0
        self.total_trades = 0
        self.total_alerts = 0
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all system components"""
        
        # REST client for historical data
        self.rest_client = PolygonClient(
            self.config.api,
            self.config.session
        )
        
        # WebSocket stream
        self.stream = LiveDataStream(
            self.config.api.polygon_api_key,
            self.config.symbols,
            self.config.candles.primary_timeframe_minutes
        )
        
        # Volume Profile engine
        self.vp_engine = MultiTimeframeVPEngine(self.config.volume_profile)
        
        # VWAP engine
        self.vwap_engine = MultiTimeframeVWAPEngine(self.config.vwap)
        
        # Signal generator
        self.signal_gen = SignalGenerator(
            self.config.signals,
            self.config.risk
        )
        
        # Execution client
        self.exec_client = TradierClient(self.config.api)
        
        # Position manager
        self.position_mgr = PositionManager(self.config.risk)
        
        # Execution engine
        self.execution = ExecutionEngine(
            self.exec_client,
            self.position_mgr
        )
        
        # Extension Dashboard
        self.extension_dashboard = MultiSymbolDashboard(self.config.symbols)
        
        # Initialize symbol states
        for symbol in self.config.symbols:
            self._symbol_states[symbol] = SymbolState(symbol=symbol)
        
        logger.info("Live trading system components initialized")
    
    async def start(self):
        """Start the live trading system"""
        logger.info("=" * 60)
        logger.info("SEF LIVE TRADING SYSTEM")
        logger.info("Mode: Real-Time WebSocket Streaming")
        logger.info("=" * 60)
        
        self._running = True
        
        # Load initial historical data for VP/VWAP
        await self._load_historical_data()
        
        # Register stream callbacks
        self.stream.on_minute_bar(self._on_minute_bar)
        self.stream.on_2h_bar(self._on_2h_bar_complete)
        
        # Start the stream
        await self.stream.start()
        
        logger.info(f"Streaming: {self.config.symbols}")
        logger.info("Waiting for market data...")
        
        # Keep running
        try:
            while self._running:
                await asyncio.sleep(60)
                self._print_status()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the live trading system"""
        logger.info("Shutting down...")
        self._running = False
        
        await self.stream.stop()
        await self.rest_client.close()
        await self.exec_client.close()
        
        # Final stats
        logger.info(f"Total signals: {self.total_signals}")
        logger.info(f"Total alerts: {self.total_alerts}")
        logger.info(f"Total trades: {self.total_trades}")
        logger.info("Shutdown complete")
    
    async def _load_historical_data(self):
        """Load historical data for initial VP/VWAP calculation"""
        logger.info("Loading historical data...")
        
        for symbol in self.config.symbols:
            try:
                # Get intraday data
                intraday_df = await self.rest_client.get_intraday_candles(
                    symbol,
                    resolution_minutes=5,
                    days_back=5
                )
                
                # Get daily data
                daily_df = await self.rest_client.get_historical_daily(
                    symbol,
                    days_back=60
                )
                
                if intraday_df.empty:
                    logger.warning(f"No historical data for {symbol}")
                    continue
                
                # Calculate VP
                vp = self.vp_engine.calculate_all(
                    intraday_df,
                    daily_df,
                    datetime.now(self.tz)
                )
                
                # Calculate VWAP
                vwap = self.vwap_engine.calculate_all(
                    intraday_df,
                    datetime.now(self.tz)
                )
                
                # Calculate ATR
                atr_series = calculate_atr(daily_df, self.config.risk.atr_period)
                atr = atr_series.iloc[-1] if not atr_series.empty else 1.0
                
                # Store in state
                state = self._symbol_states[symbol]
                state.vp = vp
                state.vwap = vwap
                state.atr = atr
                
                logger.info(f"{symbol}: VP loaded (POC: {vp.daily.poc:.2f}, VAH: {vp.daily.vah:.2f}, VAL: {vp.daily.val:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {e}")
        
        logger.info("Historical data loaded")
    
    async def _on_minute_bar(self, bar: LiveBar):
        """Handle incoming minute bar"""
        symbol = bar.symbol
        
        if symbol not in self._symbol_states:
            return
        
        state = self._symbol_states[symbol]
        state.last_bar = bar
        state.bars_received += 1
        
        # Log every 5 minutes
        if state.bars_received % 5 == 0:
            logger.debug(f"[1m] {symbol}: {bar.close:.2f} vol:{bar.volume:,}")
    
    async def _on_2h_bar_complete(self, symbol: str, bar: LiveBar):
        """
        Handle completed 2H bar - THIS IS WHERE THE MAGIC HAPPENS
        
        When a 2H bar completes:
        1. Update VP/VWAP incrementally
        2. Run extension predictor
        3. Generate signals
        4. Execute if conditions met
        """
        if symbol not in self._symbol_states:
            return
        
        state = self._symbol_states[symbol]
        state.last_2h_bar = bar
        state.bars_2h_completed += 1
        
        logger.info("=" * 60)
        logger.info(f"2H BAR COMPLETE: {symbol}")
        logger.info(f"  O: {bar.open:.2f}  H: {bar.high:.2f}  L: {bar.low:.2f}  C: {bar.close:.2f}")
        logger.info(f"  Volume: {bar.volume:,}  VWAP: {bar.vwap:.2f}")
        logger.info("=" * 60)
        
        # Check market hours
        if not self._is_market_open():
            logger.info("Market closed, skipping analysis")
            return
        
        try:
            await self._analyze_and_signal(symbol, bar)
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
    
    async def _analyze_and_signal(self, symbol: str, bar: LiveBar):
        """Run full analysis on 2H bar completion"""
        state = self._symbol_states[symbol]
        
        # Ensure we have VP/VWAP
        if state.vp is None or state.vwap is None:
            logger.warning(f"{symbol}: Missing VP/VWAP, refreshing...")
            await self._refresh_historical(symbol)
            if state.vp is None:
                return
        
        # 1. Update Extension Predictor
        ext_candle = CandleInExtension(
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            vwap=state.vwap.daily.vwap if state.vwap.daily else bar.vwap,
            poc=state.vp.daily.poc if state.vp.daily else bar.close,
            vah=state.vp.daily.vah if state.vp.daily else bar.high,
            val=state.vp.daily.val if state.vp.daily else bar.low,
            atr=state.atr
        )
        
        extension_alerts = self.extension_dashboard.update(symbol, ext_candle)
        
        # Process extension alerts
        for alert in extension_alerts:
            self.total_alerts += 1
            state.last_extension_alert = alert
            self._log_extension_alert(symbol, alert)
        
        # Print extension status
        print(self.extension_dashboard.dashboards[symbol].get_status_text())
        
        # 2. Build signal context
        from polygon_client import OHLCV
        current_candle = OHLCV(
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            vwap=bar.vwap
        )
        
        context = SignalContext(
            current_price=bar.close,
            current_candle=current_candle,
            vp=state.vp,
            vwap=state.vwap,
            atr=state.atr,
            atr_percent=(state.atr / bar.close * 100) if bar.close > 0 else 0,
            time_of_day=self._get_time_of_day(),
            minutes_into_session=self._minutes_into_session()
        )
        
        # 3. Generate signal
        signal = self.signal_gen.generate(context)
        
        if signal:
            state.signals_generated += 1
            state.last_signal = signal
            self.total_signals += 1
            
            # Apply filters
            if not self._should_take_signal(signal, context):
                logger.info(f"{symbol}: Signal filtered - {signal.signal_type.value}")
                return
            
            # Check extension confirmation
            if self.config.signals.use_extension_duration:
                if not self._extension_confirms(symbol, signal):
                    logger.info(f"{symbol}: Signal lacks extension confirmation")
                    return
            
            # Log and execute
            self._log_signal(symbol, signal)
            await self._execute_signal(symbol, signal, state.atr)
    
    async def _refresh_historical(self, symbol: str):
        """Refresh historical VP/VWAP for a symbol"""
        try:
            intraday_df = await self.rest_client.get_intraday_candles(
                symbol, resolution_minutes=5, days_back=5
            )
            daily_df = await self.rest_client.get_historical_daily(
                symbol, days_back=60
            )
            
            if not intraday_df.empty:
                state = self._symbol_states[symbol]
                state.vp = self.vp_engine.calculate_all(
                    intraday_df, daily_df, datetime.now(self.tz)
                )
                state.vwap = self.vwap_engine.calculate_all(
                    intraday_df, datetime.now(self.tz)
                )
                atr_series = calculate_atr(daily_df, self.config.risk.atr_period)
                state.atr = atr_series.iloc[-1] if not atr_series.empty else 1.0
                
        except Exception as e:
            logger.error(f"Failed to refresh {symbol}: {e}")
    
    def _extension_confirms(self, symbol: str, signal: Signal) -> bool:
        """Check if extension predictor confirms signal direction"""
        dashboard = self.extension_dashboard.dashboards.get(symbol)
        if not dashboard:
            return True
        
        actionable = dashboard.predictor.get_actionable_setups()
        if not actionable:
            return True
        
        for streak in actionable:
            if streak.direction == "above" and signal.is_short:
                logger.info(f"{symbol}: Extension confirms SHORT ({streak.streak_count} candles above)")
                return True
            if streak.direction == "below" and signal.is_long:
                logger.info(f"{symbol}: Extension confirms LONG ({streak.streak_count} candles below)")
                return True
        
        return False
    
    def _should_take_signal(self, signal: Signal, context: SignalContext) -> bool:
        """Apply signal filters"""
        if not SignalFilter.time_filter(signal, context):
            return False
        if not SignalFilter.quality_filter(signal, min_score=50):
            return False
        if not SignalFilter.risk_reward_filter(signal, min_rr=1.5):
            return False
        return True
    
    async def _execute_signal(self, symbol: str, signal: Signal, atr: float):
        """Execute trading signal"""
        # Check existing position
        existing = await self.exec_client.get_position(symbol)
        if existing:
            logger.info(f"{symbol}: Already have position, skipping")
            return
        
        side = "long" if signal.is_long else "short"
        targets = [t for t in [signal.target_1, signal.target_2, signal.target_3] if t > 0]
        
        orders = await self.execution.execute_signal(
            symbol=symbol,
            side=side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            targets=targets,
            atr=atr
        )
        
        if orders:
            self.total_trades += 1
            logger.info(f"🚀 TRADE EXECUTED: {symbol} {side.upper()}")
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now(self.tz)
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return self.config.session.market_open <= current_time < self.config.session.market_close
    
    def _get_time_of_day(self) -> str:
        """Get session time category"""
        mins = self._minutes_into_session()
        if mins < 60:
            return "open"
        elif mins > 330:
            return "close"
        return "mid"
    
    def _minutes_into_session(self) -> int:
        """Minutes since market open"""
        now = datetime.now(self.tz)
        market_open = datetime.combine(now.date(), self.config.session.market_open)
        market_open = self.tz.localize(market_open)
        delta = now - market_open
        return max(0, int(delta.total_seconds() / 60))
    
    def _print_status(self):
        """Print periodic status"""
        stats = self.stream.get_stats()
        
        logger.info("-" * 40)
        logger.info(f"Stream: {stats['messages_received']} msgs, {stats['bars_received']} bars")
        
        for symbol, state in self._symbol_states.items():
            current = self.stream.ws_client.get_current_2h_bar(symbol)
            if current:
                logger.info(
                    f"  {symbol}: {current['close']:.2f} | "
                    f"2H bar: {current['elapsed_minutes']:.0f}min / 120min | "
                    f"Signals: {state.signals_generated}"
                )
        logger.info("-" * 40)
    
    def _log_signal(self, symbol: str, signal: Signal):
        """Log signal details"""
        logger.info("🎯 " + "=" * 50)
        logger.info(f"SIGNAL: {symbol}")
        logger.info(f"Type: {signal.signal_type.value}")
        logger.info(f"Strength: {signal.strength.name}")
        logger.info(f"Score: {signal.score:.1f}")
        logger.info(f"Entry: ${signal.entry_price:.2f}")
        logger.info(f"Stop: ${signal.stop_loss:.2f}")
        logger.info(f"Target: ${signal.target_1:.2f} (R:R {signal.risk_reward_1:.2f})")
        logger.info(f"Trigger: {signal.trigger_reason}")
        logger.info("=" * 50)
    
    def _log_extension_alert(self, symbol: str, alert: ExtensionAlert):
        """Log extension alert"""
        logger.info("🚨 " + "=" * 50)
        logger.info(f"EXTENSION ALERT: {symbol}")
        logger.info(f"Level: {alert.level_name} ({alert.direction})")
        logger.info(f"Duration: {alert.candle_count} candles ({alert.hours_extended:.1f}h)")
        logger.info(f"Trigger: {alert.trigger_level.name}")
        logger.info(f"Snap-back: {alert.snap_back_probability:.0%}")
        logger.info(f"Quality: {alert.quality_score:.0f}")
        logger.info("=" * 50)


# ========== Entry Point ==========

async def main():
    """Main entry point for live system"""
    import os
    
    # Check for API key
    if not os.getenv("POLYGON_API_KEY"):
        print("ERROR: Set POLYGON_API_KEY environment variable")
        print("  export POLYGON_API_KEY='your_key'")
        return
    
    config = DEFAULT_CONFIG
    
    # Create and start live system
    system = LiveTradingSystem(config)
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())
