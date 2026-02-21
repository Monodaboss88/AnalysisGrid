"""
Trade Monitor — Auto-Close Engine
===================================
Background service that watches open trades against live prices.
Auto-closes when price hits target (WIN) or stop (LOSS).

Architecture:
  - Runs as asyncio task inside unified_server.py
  - Polls Polygon for latest prices every ~30s during market hours
  - Reads open trades from Firestore (all users)
  - Closes trades via FirestoreManager.close_trade()
  - Broker-ready: plug in Tradier/Alpaca at the execute_close() hook

Usage:
  monitor = get_trade_monitor()
  monitor.start_background()          # launch loop
  monitor.stop()                      # graceful shutdown
  monitor.get_status()                # current state
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("trade_monitor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [TradeMonitor] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class MonitorEvent:
    """Record of an auto-close event"""
    user_id: str
    trade_id: str
    symbol: str
    direction: str
    entry: float
    exit_price: float
    pnl: float
    result: str           # "WIN", "LOSS", "TRAILING_STOP"
    trigger: str           # "target_hit", "stop_hit", "trailing_stop", "manual"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MonitoredTrade:
    """A trade being actively watched"""
    user_id: str
    trade_id: str
    symbol: str
    direction: str         # LONG / SHORT
    entry: float
    stop: float
    target: float
    target2: float = 0
    highest_since_entry: float = 0   # for trailing stop
    lowest_since_entry: float = 999999
    last_price: float = 0
    last_checked: str = ""
    checks: int = 0


# ============================================================================
# TRADE MONITOR ENGINE
# ============================================================================

class TradeMonitor:
    """
    Background engine that auto-closes trades on target/stop hit.

    Flow:
      1. Every `interval` seconds, fetch all open trades across all users
      2. Batch-fetch prices for unique symbols
      3. Compare price vs entry/stop/target
      4. Auto-close via Firestore (and optionally broker API)
      5. Log event to history buffer
    """

    def __init__(
        self,
        interval: int = 30,
        max_events: int = 500,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 0.02,      # 2% trailing from high
    ):
        self.interval = interval
        self.max_events = max_events
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_pct = trailing_stop_pct

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._monitored: Dict[str, MonitoredTrade] = {}  # key = f"{user_id}:{trade_id}"
        self._events: List[MonitorEvent] = []
        self._cycle_count = 0
        self._last_cycle: str = ""
        self._errors: List[str] = []
        self._executor = ThreadPoolExecutor(max_workers=3)

        # Force-run flag (for testing outside market hours)
        self._force_run = False

        # Callbacks — plug broker here
        self._on_close_callbacks: List[Callable] = []

        # Stats
        self._total_closes = 0
        self._total_wins = 0
        self._total_losses = 0

        logger.info("TradeMonitor initialized (interval=%ds, trailing=%s)", interval, trailing_stop_enabled)

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    async def start(self):
        """Main monitoring loop"""
        self._running = True
        logger.info("🟢 Trade monitor STARTED (checking every %ds)", self.interval)

        while self._running:
            try:
                if self._is_market_hours() or self._force_run:
                    await self._run_cycle()
                    if self._force_run:
                        self._force_run = False  # one-shot
                else:
                    # During off-hours, still do a slow check every 5 min
                    # in case of after-hours movers
                    pass
            except Exception as e:
                err = f"Cycle error: {e}"
                logger.error(err)
                self._errors.append(f"{datetime.now(timezone.utc).isoformat()} {err}")
                if len(self._errors) > 100:
                    self._errors = self._errors[-50:]

            await asyncio.sleep(self.interval)

        logger.info("🔴 Trade monitor STOPPED")

    def start_background(self):
        """Launch as asyncio task"""
        if self._task and not self._task.done():
            logger.warning("Monitor already running")
            return

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.start())
        logger.info("Background task created")

    def stop(self):
        """Graceful shutdown"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Stop requested")

    def on_close(self, callback: Callable):
        """Register callback for trade close events (for broker integration)"""
        self._on_close_callbacks.append(callback)

    # ========================================================================
    # CORE CYCLE
    # ========================================================================

    async def _run_cycle(self):
        """One monitoring cycle: fetch trades → fetch prices → check → close"""
        self._cycle_count += 1
        self._last_cycle = datetime.now(timezone.utc).isoformat()

        # 1. Get all open trades across all users
        open_trades = await self._fetch_open_trades()
        if not open_trades:
            return

        # 2. Collect unique symbols
        symbols = list(set(t.symbol for t in open_trades))

        # 3. Batch-fetch prices
        prices = await self._fetch_prices(symbols)
        if not prices:
            return

        # 4. Check each trade
        closes = []
        for trade in open_trades:
            price = prices.get(trade.symbol)
            if price is None or price <= 0:
                continue

            trade.last_price = price
            trade.last_checked = datetime.now(timezone.utc).isoformat()
            trade.checks += 1

            # Track high/low for trailing stops
            if trade.direction.upper() == "LONG":
                trade.highest_since_entry = max(trade.highest_since_entry, price)
            else:
                trade.lowest_since_entry = min(trade.lowest_since_entry, price)

            # Check for close conditions
            result = self._check_trade(trade, price)
            if result:
                closes.append((trade, price, result))

            # Update monitored map
            key = f"{trade.user_id}:{trade.trade_id}"
            self._monitored[key] = trade

        # 5. Execute closes
        for trade, price, result in closes:
            await self._execute_close(trade, price, result)

        if closes:
            logger.info("Cycle %d: %d trades monitored, %d closed",
                       self._cycle_count, len(open_trades), len(closes))

    def _check_trade(self, trade: MonitoredTrade, price: float) -> Optional[str]:
        """
        Check if trade should be closed.
        Returns: "WIN", "LOSS", "WIN_T2", "TRAILING_STOP", or None
        """
        direction = trade.direction.upper()

        if direction == "LONG":
            # === LONG: price >= target → WIN, price <= stop → LOSS ===

            # Target 2 hit (if set, takes priority as bigger win)
            if trade.target2 > 0 and price >= trade.target2:
                return "WIN_T2"

            # Target 1 hit
            if trade.target > 0 and price >= trade.target:
                return "WIN"

            # Stop hit
            if trade.stop > 0 and price <= trade.stop:
                return "LOSS"

            # Trailing stop (only if price has moved past entry)
            if self.trailing_stop_enabled and trade.highest_since_entry > trade.entry:
                trail_level = trade.highest_since_entry * (1 - self.trailing_stop_pct)
                if price <= trail_level and trail_level > trade.entry:
                    return "TRAILING_STOP"

        elif direction == "SHORT":
            # === SHORT: price <= target → WIN, price >= stop → LOSS ===

            if trade.target2 > 0 and price <= trade.target2:
                return "WIN_T2"

            if trade.target > 0 and price <= trade.target:
                return "WIN"

            if trade.stop > 0 and price >= trade.stop:
                return "LOSS"

            if self.trailing_stop_enabled and trade.lowest_since_entry < trade.entry:
                trail_level = trade.lowest_since_entry * (1 + self.trailing_stop_pct)
                if price >= trail_level and trail_level < trade.entry:
                    return "TRAILING_STOP"

        return None

    async def _execute_close(self, trade: MonitoredTrade, exit_price: float, result: str):
        """Close a trade in Firestore (or local) + fire callbacks"""
        try:
            # Map result to status
            if result in ("WIN", "WIN_T2"):
                status = "WIN"
            elif result == "TRAILING_STOP":
                status = "WIN" if self._calc_pnl(trade, exit_price) > 0 else "LOSS"
            else:
                status = "LOSS"

            pnl = self._calc_pnl(trade, exit_price)
            closed = None

            # Try Firestore close
            if trade.user_id != 'local':
                try:
                    from firestore_store import get_firestore
                    fs = get_firestore()
                    closed = fs.close_trade(trade.user_id, trade.trade_id, exit_price, status=status)
                    if closed:
                        fs.update_trade(trade.user_id, trade.trade_id, {
                            'monitor_closed': True,
                            'monitor_trigger': result,
                            'monitor_exit_price': exit_price,
                            'monitor_timestamp': datetime.now(timezone.utc).isoformat(),
                            'result': status
                        })
                        pnl = closed.get('pnl', pnl)
                except Exception as e:
                    logger.warning("Firestore close failed: %s", e)

            # Local trade close (update in-memory)
            if trade.user_id == 'local' or (not closed and trade.user_id != 'local'):
                try:
                    from chart_input_analyzer import ChartInputSystem
                    cs = ChartInputSystem(data_dir="./scanner_data")
                    idx = int(trade.trade_id.replace('local_', '')) if trade.trade_id.startswith('local_') else -1
                    if 0 <= idx < len(cs.tracker.trades):
                        t = cs.tracker.trades[idx]
                        if hasattr(t, 'status'):
                            t.status = status
                        if hasattr(t, 'exit_price'):
                            t.exit_price = exit_price
                        if hasattr(t, 'exit_time'):
                            t.exit_time = datetime.now(timezone.utc).isoformat()
                        if hasattr(t, 'result_pct'):
                            t.result_pct = round(pnl / trade.entry * 100, 2) if trade.entry else 0
                        cs.tracker._save()
                        closed = {'pnl': pnl, 'status': status}
                        logger.info("Local trade %s closed as %s", trade.trade_id, status)
                except Exception as e:
                    logger.warning("Local close fallback: %s", e)

            if closed or True:  # always log the event even if storage update fails

                # Create event
                event = MonitorEvent(
                    user_id=trade.user_id,
                    trade_id=trade.trade_id,
                    symbol=trade.symbol,
                    direction=trade.direction,
                    entry=trade.entry,
                    exit_price=exit_price,
                    pnl=pnl,
                    result=status,
                    trigger=result.lower()
                )
                self._events.append(event)
                if len(self._events) > self.max_events:
                    self._events = self._events[-self.max_events:]

                # Update stats
                self._total_closes += 1
                if status == "WIN":
                    self._total_wins += 1
                else:
                    self._total_losses += 1

                # Remove from monitored
                key = f"{trade.user_id}:{trade.trade_id}"
                self._monitored.pop(key, None)

                logger.info("✅ AUTO-CLOSED %s %s %s @ %.2f → %.2f | PnL: %.2f | %s",
                           trade.direction, trade.symbol, trade.trade_id[:8],
                           trade.entry, exit_price, pnl, result)

                # Fire broker/notification callbacks
                for cb in self._on_close_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(event)
                        else:
                            cb(event)
                    except Exception as e:
                        logger.error("Callback error: %s", e)

        except Exception as e:
            logger.error("Failed to close %s %s: %s", trade.symbol, trade.trade_id[:8], e)

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    async def _fetch_open_trades(self) -> List[MonitoredTrade]:
        """Fetch all open trades from Firestore (primary) or local storage (fallback)"""
        trades = []

        # Try Firestore first
        try:
            from firestore_store import get_firestore
            fs = get_firestore()
            if fs.db:
                loop = asyncio.get_event_loop()

                def _fetch_firestore():
                    all_trades = []
                    try:
                        users_ref = fs.db.collection('users')
                        user_docs = users_ref.stream()

                        for user_doc in user_docs:
                            user_id = user_doc.id
                            for status in ["pending", "active"]:
                                user_trades = fs.get_trades(user_id, status=status)
                                for t in user_trades:
                                    entry = t.get('entry', 0)
                                    stop = t.get('stop', 0)
                                    target = t.get('target', 0)
                                    if not entry or (not stop and not target):
                                        continue
                                    mt = MonitoredTrade(
                                        user_id=user_id,
                                        trade_id=t.get('id', ''),
                                        symbol=t.get('symbol', '').upper(),
                                        direction=t.get('direction', 'LONG').upper(),
                                        entry=entry,
                                        stop=stop,
                                        target=target,
                                        target2=t.get('target2', 0) or t.get('target_2', 0) or 0,
                                    )
                                    key = f"{user_id}:{t.get('id', '')}"
                                    prev = self._monitored.get(key)
                                    if prev:
                                        mt.highest_since_entry = prev.highest_since_entry
                                        mt.lowest_since_entry = prev.lowest_since_entry
                                        mt.checks = prev.checks
                                    all_trades.append(mt)
                    except Exception as e:
                        logger.error("Firestore fetch error: %s", e)
                    return all_trades

                trades = await loop.run_in_executor(self._executor, _fetch_firestore)
        except Exception as e:
            logger.warning("Firestore not available: %s", e)

        # Fallback: pull from local chart_system if no Firestore trades
        if not trades:
            try:
                trades = await self._fetch_local_trades()
            except Exception as e:
                logger.warning("Local trade fetch error: %s", e)

        return trades

    async def _fetch_local_trades(self) -> List[MonitoredTrade]:
        """Fetch open trades from the local chart_system storage"""
        try:
            # Import the chart system from unified_server context
            import importlib, sys
            # The local trades are accessible via the /api/trades endpoint logic
            # We replicate the same local read here
            from chart_input_analyzer import ChartInputSystem
            cs = ChartInputSystem(data_dir="./scanner_data")
            local_trades = cs.tracker.trades
            result = []
            for i, t in enumerate(local_trades):
                trade_dict = t if isinstance(t, dict) else (t.__dict__ if hasattr(t, '____dict__') else {})
                if hasattr(t, '__dataclass_fields__'):
                    from dataclasses import asdict
                    trade_dict = asdict(t)

                status = str(trade_dict.get('status', '')).upper()
                if status in ('CLOSED', 'WIN', 'LOSS', 'CANCELLED'):
                    continue

                entry = trade_dict.get('entry_price', 0) or trade_dict.get('entry', 0)
                stop = trade_dict.get('stop_loss', 0) or trade_dict.get('stop', 0)
                target = trade_dict.get('target_1', 0) or trade_dict.get('target', 0)

                if not entry or (not stop and not target):
                    continue

                mt = MonitoredTrade(
                    user_id='local',
                    trade_id=f'local_{i}',
                    symbol=str(trade_dict.get('symbol', '')).upper(),
                    direction=str(trade_dict.get('direction', 'LONG')).upper(),
                    entry=float(entry),
                    stop=float(stop) if stop else 0,
                    target=float(target) if target else 0,
                    target2=float(trade_dict.get('target_2', 0) or 0),
                )
                key = f"local:{mt.trade_id}"
                prev = self._monitored.get(key)
                if prev:
                    mt.highest_since_entry = prev.highest_since_entry
                    mt.lowest_since_entry = prev.lowest_since_entry
                    mt.checks = prev.checks
                result.append(mt)
            return result
        except Exception as e:
            logger.warning("Local trades fallback error: %s", e)
            return []

    async def _fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Batch-fetch current prices from Polygon"""
        if not symbols:
            return {}

        try:
            from polygon_data import get_price_quote

            loop = asyncio.get_event_loop()
            prices = {}

            def _fetch_batch():
                result = {}
                for sym in symbols:
                    try:
                        quote = get_price_quote(sym)
                        if quote and quote.get('price'):
                            result[sym.upper()] = float(quote['price'])
                    except Exception as e:
                        logger.warning("Price fetch failed for %s: %s", sym, e)
                return result

            prices = await loop.run_in_executor(self._executor, _fetch_batch)
            return prices

        except Exception as e:
            logger.error("_fetch_prices error: %s", e)
            return {}

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _is_market_hours() -> bool:
        """Check if US stock market is open (ET: 9:30 AM - 4:00 PM, Mon-Fri)"""
        now_utc = datetime.now(timezone.utc)
        et_offset = timedelta(hours=-5)  # EST (simplification, doesn't handle DST)
        now_et = now_utc + et_offset

        # Weekends
        if now_et.weekday() >= 5:
            return False

        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)

        # Allow 15 min pre-market buffer and 15 min post-market
        extended_open = dt_time(9, 15)
        extended_close = dt_time(16, 15)

        return extended_open <= now_et.time() <= extended_close

    @staticmethod
    def _calc_pnl(trade: MonitoredTrade, exit_price: float) -> float:
        if trade.direction.upper() == "LONG":
            return exit_price - trade.entry
        else:
            return trade.entry - exit_price

    # ========================================================================
    # STATUS / API
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Full status snapshot"""
        return {
            "running": self._running,
            "interval_seconds": self.interval,
            "cycle_count": self._cycle_count,
            "last_cycle": self._last_cycle,
            "monitored_trades": len(self._monitored),
            "total_closes": self._total_closes,
            "total_wins": self._total_wins,
            "total_losses": self._total_losses,
            "win_rate": round(self._total_wins / self._total_closes * 100, 1) if self._total_closes else 0,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_pct": self.trailing_stop_pct,
            "market_hours": self._is_market_hours(),
            "recent_errors": self._errors[-5:] if self._errors else [],
        }

    def get_monitored_trades(self) -> List[Dict]:
        """All trades currently being watched"""
        return [
            {
                "user_id": t.user_id[:8] + "...",
                "trade_id": t.trade_id[:8] + "...",
                "symbol": t.symbol,
                "direction": t.direction,
                "entry": t.entry,
                "stop": t.stop,
                "target": t.target,
                "target2": t.target2,
                "last_price": t.last_price,
                "last_checked": t.last_checked,
                "checks": t.checks,
                "distance_to_target": self._distance_pct(t, "target"),
                "distance_to_stop": self._distance_pct(t, "stop"),
            }
            for t in self._monitored.values()
        ]

    def get_events(self, limit: int = 50) -> List[Dict]:
        """Recent auto-close events"""
        return [e.to_dict() for e in self._events[-limit:]]

    def _distance_pct(self, trade: MonitoredTrade, side: str) -> str:
        """Calculate distance to target or stop as percentage"""
        price = trade.last_price
        if not price:
            return "—"

        if side == "target":
            ref = trade.target
        else:
            ref = trade.stop

        if not ref:
            return "—"

        pct = ((ref - price) / price) * 100
        if trade.direction.upper() == "SHORT":
            pct = -pct

        return f"{pct:+.1f}%"


# ============================================================================
# SINGLETON
# ============================================================================

_monitor_instance: Optional[TradeMonitor] = None


def get_trade_monitor(
    interval: int = 30,
    trailing_stop: bool = False,
    trailing_stop_pct: float = 0.02
) -> TradeMonitor:
    """Get or create the singleton TradeMonitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = TradeMonitor(
            interval=interval,
            trailing_stop_enabled=trailing_stop,
            trailing_stop_pct=trailing_stop_pct
        )
    return _monitor_instance
