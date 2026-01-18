"""
SEF Trading System - Main Orchestrator
Ties together all components for live trading
Now with Polygon.io data and Extension Duration Predictor
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Optional, Dict, List
from dataclasses import dataclass
import pytz

from config import SystemConfig, DEFAULT_CONFIG
from polygon_client import PolygonClient, CandleAggregator, OHLCV, calculate_atr
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
class SystemState:
    """Current system state"""
    is_running: bool = False
    is_market_open: bool = False
    current_time: Optional[datetime] = None
    active_symbols: List[str] = None
    
    # Per-symbol state
    last_signals: Dict[str, Signal] = None
    active_positions: Dict[str, Position] = None
    extension_alerts: Dict[str, ExtensionAlert] = None
    
    # Stats
    signals_generated: int = 0
    trades_executed: int = 0
    extension_alerts_generated: int = 0
    
    def __post_init__(self):
        self.active_symbols = []
        self.last_signals = {}
        self.active_positions = {}
        self.extension_alerts = {}


class TradingSystem:
    """
    Main trading system orchestrator
    
    Lifecycle:
    1. Initialize -> Load config, setup components
    2. Start -> Begin monitoring loop
    3. Monitor -> Fetch data, generate signals, execute
    4. Stop -> Clean shutdown
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.state = SystemState()
        
        # Initialize components
        self._init_components()
        
        # Timezone
        self.tz = pytz.timezone(self.config.session.timezone)
    
    def _init_components(self):
        """Initialize all system components"""
        # Data client (Polygon.io)
        self.data_client = PolygonClient(
            self.config.api,
            self.config.session
        )
        
        # Candle aggregator for 2H bars
        self.aggregator = CandleAggregator(
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
        
        # Extension Dashboard (multi-symbol)
        self.extension_dashboard = MultiSymbolDashboard(self.config.symbols)
        
        logger.info("All components initialized (Polygon.io + Extension Predictor)")
    
    async def start(self):
        """Start the trading system"""
        logger.info("=" * 60)
        logger.info("SEF Trading System Starting")
        logger.info("Data Provider: Polygon.io")
        logger.info("Extension Duration Predictor: ENABLED")
        logger.info("=" * 60)
        
        self.state.is_running = True
        self.state.active_symbols = self.config.symbols.copy()
        
        logger.info(f"Monitoring symbols: {self.state.active_symbols}")
        logger.info(f"Sandbox mode: {self.config.api.use_sandbox}")
        
        # Main loop
        try:
            await self._run_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"System error: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading system"""
        logger.info("Shutting down...")
        
        self.state.is_running = False
        
        # Close connections
        await self.data_client.close()
        await self.exec_client.close()
        
        # Log stats
        logger.info(f"Signals generated: {self.state.signals_generated}")
        logger.info(f"Extension alerts: {self.state.extension_alerts_generated}")
        logger.info(f"Trades executed: {self.state.trades_executed}")
        logger.info("Shutdown complete")
    
    async def _run_loop(self):
        """Main monitoring loop"""
        while self.state.is_running:
            try:
                # Update time
                self.state.current_time = datetime.now(self.tz)
                
                # Check market hours
                self.state.is_market_open = self._is_market_open()
                
                if not self.state.is_market_open:
                    logger.debug("Market closed, waiting...")
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                # Process each symbol
                for symbol in self.state.active_symbols:
                    await self._process_symbol(symbol)
                
                # Print extension dashboard
                self._print_extension_status()
                
                # Wait for next check interval (every 5 minutes for 2H candles)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol for signals"""
        logger.debug(f"Processing {symbol}")
        
        try:
            # 1. Fetch data from Polygon
            intraday_df = await self.data_client.get_intraday_candles(
                symbol,
                resolution_minutes=self.config.volume_profile.resolution_minutes,
                days_back=5  # 5 days of intraday for daily/session VP
            )
            
            daily_df = await self.data_client.get_historical_daily(
                symbol,
                days_back=60  # 60 days for weekly/monthly VP
            )
            
            if intraday_df.empty:
                logger.warning(f"No intraday data for {symbol}")
                return
            
            # 2. Calculate VP
            vp = self.vp_engine.calculate_all(
                intraday_df,
                daily_df,
                self.state.current_time
            )
            
            # 3. Calculate VWAP
            vwap = self.vwap_engine.calculate_all(
                intraday_df,
                self.state.current_time
            )
            
            # 4. Get current price and build 2H candle
            current_price = intraday_df["close"].iloc[-1]
            
            # Aggregate recent data into 2H candle
            aggregated = self.aggregator.aggregate(intraday_df.tail(48))  # ~4 hours of 5-min bars
            
            if not aggregated.empty:
                latest_2h = aggregated.iloc[-1]
                current_candle = OHLCV(
                    timestamp=aggregated.index[-1],
                    open=latest_2h["open"],
                    high=latest_2h["high"],
                    low=latest_2h["low"],
                    close=latest_2h["close"],
                    volume=int(latest_2h["volume"]),
                    vwap=latest_2h.get("vwap")
                )
            else:
                # Fallback to recent bars
                recent = intraday_df.tail(24)
                current_candle = OHLCV(
                    timestamp=recent.index[-1],
                    open=recent["open"].iloc[0],
                    high=recent["high"].max(),
                    low=recent["low"].min(),
                    close=recent["close"].iloc[-1],
                    volume=int(recent["volume"].sum())
                )
            
            # 5. Calculate ATR
            atr_series = calculate_atr(daily_df, self.config.risk.atr_period)
            atr = atr_series.iloc[-1] if not atr_series.empty else current_price * 0.02
            
            # 6. Update Extension Predictor
            ext_candle = CandleInExtension(
                timestamp=current_candle.timestamp,
                open=current_candle.open,
                high=current_candle.high,
                low=current_candle.low,
                close=current_candle.close,
                volume=current_candle.volume,
                vwap=vwap.daily.vwap if vwap.daily else current_price,
                poc=vp.daily.poc if vp.daily else current_price,
                vah=vp.daily.vah if vp.daily else current_price * 1.01,
                val=vp.daily.val if vp.daily else current_price * 0.99,
                atr=atr
            )
            
            extension_alerts = self.extension_dashboard.update(symbol, ext_candle)
            
            # Process extension alerts
            for alert in extension_alerts:
                self.state.extension_alerts_generated += 1
                self.state.extension_alerts[symbol] = alert
                self._log_extension_alert(symbol, alert)
            
            # 7. Build signal context
            context = SignalContext(
                current_price=current_price,
                current_candle=current_candle,
                vp=vp,
                vwap=vwap,
                atr=atr,
                atr_percent=(atr / current_price * 100) if current_price > 0 else 0,
                time_of_day=self._get_time_of_day(),
                minutes_into_session=self._minutes_into_session()
            )
            
            # 8. Generate signal (traditional VP/VWAP)
            signal = self.signal_gen.generate(context)
            
            if signal:
                self.state.signals_generated += 1
                self.state.last_signals[symbol] = signal
                
                # Apply filters
                if not self._should_take_signal(signal, context):
                    logger.info(f"{symbol}: Signal filtered out - {signal.signal_type.value}")
                    return
                
                # Check for extension confirmation if enabled
                if self.config.signals.use_extension_duration:
                    if not self._extension_confirms_signal(symbol, signal):
                        logger.info(f"{symbol}: Signal lacks extension confirmation")
                        return
                
                # Log signal
                self._log_signal(symbol, signal)
                
                # 9. Execute if conditions met
                await self._execute_signal(symbol, signal, atr)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def _extension_confirms_signal(self, symbol: str, signal: Signal) -> bool:
        """Check if extension predictor confirms the signal direction"""
        dashboard = self.extension_dashboard.dashboards.get(symbol)
        if not dashboard:
            return True  # No extension data, allow signal
        
        actionable = dashboard.predictor.get_actionable_setups()
        if not actionable:
            return True  # No extension setup, allow signal
        
        # Check if extension direction matches signal
        for streak in actionable:
            # Extension above + SHORT signal = confirmed
            if streak.direction == "above" and signal.is_short:
                logger.info(f"{symbol}: Extension confirms SHORT (above for {streak.streak_count} candles)")
                return True
            
            # Extension below + LONG signal = confirmed
            if streak.direction == "below" and signal.is_long:
                logger.info(f"{symbol}: Extension confirms LONG (below for {streak.streak_count} candles)")
                return True
        
        return False
    
    def _should_take_signal(self, signal: Signal, context: SignalContext) -> bool:
        """Apply filters to determine if signal should be taken"""
        # Time filter
        if not SignalFilter.time_filter(signal, context):
            return False
        
        # Quality filter
        if not SignalFilter.quality_filter(signal, min_score=50):
            return False
        
        # Risk/reward filter
        if not SignalFilter.risk_reward_filter(signal, min_rr=1.5):
            return False
        
        return True
    
    async def _execute_signal(self, symbol: str, signal: Signal, atr: float):
        """Execute a trading signal"""
        # Check if we already have a position
        existing = await self.exec_client.get_position(symbol)
        
        if existing:
            logger.info(f"{symbol}: Already have position, skipping new signal")
            return
        
        # Determine side
        side = "long" if signal.is_long else "short"
        
        # Collect targets
        targets = [t for t in [signal.target_1, signal.target_2, signal.target_3] if t > 0]
        
        # Execute
        orders = await self.execution.execute_signal(
            symbol=symbol,
            side=side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            targets=targets,
            atr=atr
        )
        
        if orders:
            self.state.trades_executed += 1
            logger.info(f"{symbol}: Trade executed - {side.upper()}")
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = self.state.current_time
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        current_time = now.time()
        return (
            self.config.session.market_open <= current_time < 
            self.config.session.market_close
        )
    
    def _get_time_of_day(self) -> str:
        """Categorize current time of day"""
        if not self.state.current_time:
            return "mid"
        
        mins = self._minutes_into_session()
        
        if mins < 60:
            return "open"
        elif mins > 330:  # Last 60 mins
            return "close"
        return "mid"
    
    def _minutes_into_session(self) -> int:
        """Calculate minutes since market open"""
        if not self.state.current_time:
            return 0
        
        now = self.state.current_time
        market_open = datetime.combine(
            now.date(),
            self.config.session.market_open
        )
        market_open = self.tz.localize(market_open)
        
        delta = now - market_open
        return max(0, int(delta.total_seconds() / 60))
    
    def _print_extension_status(self):
        """Print extension dashboard status"""
        print("\n" + self.extension_dashboard.get_all_status() + "\n")
    
    def _log_signal(self, symbol: str, signal: Signal):
        """Log signal details"""
        logger.info("=" * 50)
        logger.info(f"SIGNAL: {symbol}")
        logger.info(f"Type: {signal.signal_type.value}")
        logger.info(f"Strength: {signal.strength.name}")
        logger.info(f"Score: {signal.score:.1f}")
        logger.info(f"Entry: ${signal.entry_price:.2f}")
        logger.info(f"Stop: ${signal.stop_loss:.2f}")
        logger.info(f"Target 1: ${signal.target_1:.2f} (R:R {signal.risk_reward_1:.2f})")
        logger.info(f"Trigger: {signal.trigger_reason}")
        logger.info(f"Bias: {signal.bias}")
        logger.info("=" * 50)
    
    def _log_extension_alert(self, symbol: str, alert: ExtensionAlert):
        """Log extension alert"""
        logger.info("=" * 50)
        logger.info(f"🚨 EXTENSION ALERT: {symbol}")
        logger.info(f"Level: {alert.level_name} ({alert.direction})")
        logger.info(f"Duration: {alert.candle_count} candles ({alert.hours_extended:.1f}h)")
        logger.info(f"Trigger: {alert.trigger_level.name}")
        logger.info(f"Snap-back probability: {alert.snap_back_probability:.0%}")
        logger.info(f"Quality Score: {alert.quality_score:.0f}")
        logger.info("=" * 50)


class SystemMonitor:
    """
    Utility for monitoring system state and performance
    """
    
    def __init__(self, system: TradingSystem):
        self.system = system
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "is_running": self.system.state.is_running,
            "is_market_open": self.system.state.is_market_open,
            "current_time": str(self.system.state.current_time),
            "symbols": self.system.state.active_symbols,
            "signals_generated": self.system.state.signals_generated,
            "extension_alerts": self.system.state.extension_alerts_generated,
            "trades_executed": self.system.state.trades_executed,
            "last_signals": {
                sym: {
                    "type": sig.signal_type.value,
                    "score": sig.score,
                    "timestamp": str(sig.timestamp)
                }
                for sym, sig in self.system.state.last_signals.items()
            }
        }
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        positions = await self.system.exec_client.get_positions()
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl
            }
            for p in positions
        ]


# ========== Entry Point ==========

async def main():
    """Main entry point"""
    # Load config (can be customized)
    config = DEFAULT_CONFIG
    
    # Override with your API keys (or use environment variables)
    # config.api.polygon_api_key = "your_polygon_key"
    # config.api.tradier_api_key = "your_tradier_key"
    # config.api.tradier_account_id = "your_account"
    
    # Create and start system
    system = TradingSystem(config)
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())
