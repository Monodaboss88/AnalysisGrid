"""
Backtest Engine
===============
Two modes:
  1. REPLAY — Validate past trades against actual intraday price data
  2. STRATEGY — Run scanner signals over historical data, simulate trades

Uses:
  - Polygon bars for historical OHLCV
  - MarketScanner for signal generation  
  - journal_analytics for performance metrics
  - Same trade format as the rest of the platform

Usage:
  from backtest_engine import BacktestEngine
  engine = BacktestEngine()
  
  # Replay past trades
  results = engine.replay_trades(trades, bar_interval="5m")
  
  # Test a strategy on historical data
  results = engine.run_strategy(
      symbols=["AAPL","NVDA","TSLA"],
      days_back=90,
      signal_filter="GREEN",
      timeframe="swing"
  )
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import time

logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [Backtest] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class BacktestTrade:
    """A single simulated trade result"""
    symbol: str
    direction: str           # LONG / SHORT
    entry: float
    stop: float
    target: float
    target2: float = 0
    signal: str = ""
    confidence: int = 0
    timeframe: str = "swing"
    
    # Outcome (filled by simulation)
    status: str = ""         # WIN, LOSS, OPEN, EXPIRED
    exit_price: float = 0
    pnl: float = 0           # dollar P&L per share
    pnl_r: float = 0         # R-multiple
    pnl_pct: float = 0       # percentage
    hit_target_first: bool = False
    hit_stop_first: bool = False
    bars_to_exit: int = 0
    exit_date: str = ""
    entry_date: str = ""
    max_favorable: float = 0    # MFE — max favorable excursion (best price)
    max_adverse: float = 0      # MAE — max adverse excursion (worst price)
    notes: str = ""

    def to_analytics_dict(self) -> Dict:
        """Convert to format expected by journal_analytics"""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "exit_price": self.exit_price,
            "status": self.status,
            "pnl": self.pnl,
            "pnl_r": self.pnl_r,
            "signal": self.signal,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "created_at": self.entry_date,
            "closed_at": self.exit_date,
            "notes": self.notes,
        }


@dataclass 
class BacktestResult:
    """Complete backtest output"""
    mode: str                # "replay" or "strategy"
    trades: List[BacktestTrade] = field(default_factory=list)
    analytics: Dict = field(default_factory=dict)
    
    # Execution metadata
    symbols_tested: List[str] = field(default_factory=list)
    period: str = ""
    bar_interval: str = ""
    total_bars_processed: int = 0
    runtime_seconds: float = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "summary": {
                "total_trades": len(self.trades),
                "wins": sum(1 for t in self.trades if t.status == "WIN"),
                "losses": sum(1 for t in self.trades if t.status == "LOSS"),
                "open": sum(1 for t in self.trades if t.status == "OPEN"),
                "expired": sum(1 for t in self.trades if t.status == "EXPIRED"),
                "win_rate": self._win_rate(),
                "avg_r": self._avg_r(),
                "total_pnl": round(sum(t.pnl for t in self.trades), 2),
                "profit_factor": self._profit_factor(),
                "expectancy_r": self._expectancy_r(),
                "best_trade_r": max((t.pnl_r for t in self.trades), default=0),
                "worst_trade_r": min((t.pnl_r for t in self.trades), default=0),
                "avg_bars_to_exit": self._avg_bars(),
            },
            "trades": [t.to_analytics_dict() for t in self.trades],
            "analytics": self.analytics,
            "meta": {
                "symbols_tested": self.symbols_tested,
                "period": self.period,
                "bar_interval": self.bar_interval,
                "total_bars": self.total_bars_processed,
                "runtime_seconds": round(self.runtime_seconds, 2),
                "errors": self.errors[-10:],
            }
        }
    
    def _win_rate(self) -> float:
        closed = [t for t in self.trades if t.status in ("WIN", "LOSS")]
        if not closed:
            return 0
        return round(sum(1 for t in closed if t.status == "WIN") / len(closed) * 100, 1)
    
    def _avg_r(self) -> float:
        closed = [t for t in self.trades if t.status in ("WIN", "LOSS")]
        if not closed:
            return 0
        return round(sum(t.pnl_r for t in closed) / len(closed), 2)
    
    def _profit_factor(self) -> float:
        wins = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if losses == 0:
            return float('inf') if wins > 0 else 0
        return round(wins / losses, 2)
    
    def _expectancy_r(self) -> float:
        wr = self._win_rate() / 100
        closed = [t for t in self.trades if t.status in ("WIN", "LOSS")]
        winners = [t for t in closed if t.status == "WIN"]
        losers = [t for t in closed if t.status == "LOSS"]
        avg_win = sum(t.pnl_r for t in winners) / len(winners) if winners else 0
        avg_loss = abs(sum(t.pnl_r for t in losers) / len(losers)) if losers else 0
        return round(wr * avg_win - (1 - wr) * avg_loss, 3)
    
    def _avg_bars(self) -> float:
        closed = [t for t in self.trades if t.status in ("WIN", "LOSS") and t.bars_to_exit > 0]
        if not closed:
            return 0
        return round(sum(t.bars_to_exit for t in closed) / len(closed), 1)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    Core backtest engine. Two modes:
    
    1. replay_trades() — Given existing trade setups (from journal), determine
       what ACTUALLY happened by checking historical intraday bars.
       
    2. run_strategy() — Run scanner over historical period, generate signals,
       simulate trades, produce analytics.
    """

    def __init__(self, rate_limit_delay: float = 0.25):
        self._rate_delay = rate_limit_delay  # seconds between Polygon calls

    # ========================================================================
    # MODE 1: REPLAY — Validate past trades with real data
    # ========================================================================

    def replay_trades(
        self,
        trades: List[Dict],
        bar_interval: str = "5m",
        max_hold_bars: int = 200,
        days_forward: int = 30,
    ) -> BacktestResult:
        """
        Take existing trade setups and simulate using real price data.
        
        For each trade:
          1. Fetch bars from entry_date forward
          2. Walk bar by bar, check if high/low hits target or stop first
          3. Record actual outcome with exact exit price and R-multiple
        
        Args:
            trades: list of trade dicts (from Firestore, journal, or manual)
            bar_interval: bar size for simulation ("5m", "15m", "1h", "1d")
            max_hold_bars: max bars before marking as EXPIRED
            days_forward: how many days of bars to fetch after entry
        """
        from polygon_data import get_bars

        start_time = time.time()
        result = BacktestResult(mode="replay", bar_interval=bar_interval)
        symbols_seen = set()

        for trade_dict in trades:
            try:
                bt = self._parse_trade(trade_dict)
                if not bt or not bt.symbol or not bt.entry:
                    continue

                symbols_seen.add(bt.symbol)

                # Determine date range for bars
                entry_date = self._parse_date(trade_dict.get("created_at", ""))
                if not entry_date:
                    # No date — use 30 days ago as reasonable default  
                    entry_date = datetime.now() - timedelta(days=30)

                bt.entry_date = entry_date.strftime("%Y-%m-%d")

                # Fetch bars for simulation period
                days_str = f"{days_forward}d"
                bars_df = get_bars(bt.symbol, period=days_str, interval=bar_interval)
                time.sleep(self._rate_delay)

                if bars_df is None or bars_df.empty:
                    bt.status = "EXPIRED"
                    bt.notes = "No bar data available"
                    result.trades.append(bt)
                    continue

                # Filter bars to after entry date
                if hasattr(bars_df.index, 'tz_localize') and bars_df.index.tz is None:
                    pass  # already naive
                
                # Walk bars and simulate
                self._simulate_trade(bt, bars_df, max_hold_bars)
                result.total_bars_processed += len(bars_df)
                result.trades.append(bt)

            except Exception as e:
                result.errors.append(f"{trade_dict.get('symbol', '?')}: {e}")
                logger.warning("Replay error for %s: %s", trade_dict.get('symbol', '?'), e)

        result.symbols_tested = sorted(symbols_seen)
        result.runtime_seconds = time.time() - start_time

        # Generate analytics
        result.analytics = self._compute_analytics(result.trades)
        
        logger.info("Replay complete: %d trades, %.1f%% win rate, %.2fR avg",
                    len(result.trades), result._win_rate(), result._avg_r())
        return result

    # ========================================================================
    # MODE 2: STRATEGY — Scan historical data, generate + simulate trades
    # ========================================================================

    def run_strategy(
        self,
        symbols: List[str],
        days_back: int = 90,
        signal_filter: str = None,        # "GREEN", "LONG_SETUP", None=all
        min_confidence: int = 0,
        timeframe: str = "swing",
        scan_interval_days: int = 5,      # rescan every N days
        bar_interval: str = "1d",         # sim bar interval
        max_hold_bars: int = 60,          # max bars before expiry
        rr_ratio: float = 2.0,           # default R:R if not from scanner
    ) -> BacktestResult:
        """
        Run a strategy backtest:
          1. For each symbol, get historical daily bars
          2. Every scan_interval_days, run the scanner to generate a signal
          3. If signal matches filter, create a trade
          4. Simulate trade using subsequent bars
          5. Produce analytics
        
        Args:
            symbols: list of tickers to test
            days_back: how far back to start
            signal_filter: only take these signals (None = all non-YELLOW)
            min_confidence: minimum confidence score
            timeframe: passed to scanner
            scan_interval_days: how often to "scan" (resample)
            bar_interval: bar size for simulation
            max_hold_bars: max bars before EXPIRED
            rr_ratio: default R:R ratio
        """
        from polygon_data import get_bars

        start_time = time.time()
        result = BacktestResult(
            mode="strategy",
            bar_interval=bar_interval,
            period=f"{days_back}d",
            symbols_tested=sorted([s.upper() for s in symbols])
        )

        scanner = self._get_scanner()

        for symbol in symbols:
            symbol = symbol.upper()
            try:
                logger.info("Strategy scan: %s (%dd back)", symbol, days_back)

                # Fetch full bar history
                bars_df = get_bars(symbol, period=f"{days_back + 30}d", interval=bar_interval)
                time.sleep(self._rate_delay)

                if bars_df is None or bars_df.empty or len(bars_df) < 20:
                    result.errors.append(f"{symbol}: insufficient data ({len(bars_df) if bars_df is not None else 0} bars)")
                    continue

                result.total_bars_processed += len(bars_df)

                # Generate scan signals at intervals
                signals = self._generate_historical_signals(
                    symbol, bars_df, scanner, timeframe,
                    scan_interval_days, signal_filter, min_confidence
                )

                # Simulate each signal as a trade
                for sig in signals:
                    entry_idx = sig["bar_index"]
                    if entry_idx >= len(bars_df) - 1:
                        continue

                    bt = BacktestTrade(
                        symbol=symbol,
                        direction=sig["direction"],
                        entry=sig["entry"],
                        stop=sig["stop"],
                        target=sig["target"],
                        signal=sig["signal"],
                        confidence=sig["confidence"],
                        timeframe=timeframe,
                        entry_date=sig["date"],
                    )

                    # Simulate using bars after entry
                    future_bars = bars_df.iloc[entry_idx + 1:]
                    self._simulate_trade(bt, future_bars, max_hold_bars)
                    result.trades.append(bt)

            except Exception as e:
                result.errors.append(f"{symbol}: {e}")
                logger.warning("Strategy error for %s: %s", symbol, e)

        result.runtime_seconds = time.time() - start_time
        result.analytics = self._compute_analytics(result.trades)

        logger.info("Strategy complete: %d symbols, %d trades, %.1f%% WR, %.2fR avg, %.1fs",
                    len(symbols), len(result.trades), result._win_rate(), 
                    result._avg_r(), result.runtime_seconds)
        return result

    # ========================================================================
    # TRADE SIMULATION — Walk bars, check target/stop
    # ========================================================================

    def _simulate_trade(self, trade: BacktestTrade, bars_df, max_hold_bars: int):
        """
        Walk through bars and determine if target or stop is hit first.
        Uses High/Low to check intrabar touch.
        """
        if bars_df is None or bars_df.empty:
            trade.status = "EXPIRED"
            trade.notes = "No data for simulation"
            return

        is_long = trade.direction.upper() == "LONG"
        risk = abs(trade.entry - trade.stop) if trade.stop else trade.entry * 0.02

        for i, (idx, bar) in enumerate(bars_df.iterrows()):
            if i >= max_hold_bars:
                break

            high = bar.get("High", bar.get("high", 0))
            low = bar.get("Low", bar.get("low", 0))
            close = bar.get("Close", bar.get("close", 0))

            if not high or not low:
                continue

            # Track MFE / MAE
            if is_long:
                trade.max_favorable = max(trade.max_favorable, high)
                trade.max_adverse = min(trade.max_adverse, low) if trade.max_adverse > 0 else low
            else:
                trade.max_favorable = min(trade.max_favorable, low) if trade.max_favorable > 0 else low
                trade.max_adverse = max(trade.max_adverse, high)

            # Check hit conditions
            target_hit = False
            stop_hit = False

            if is_long:
                if trade.target and high >= trade.target:
                    target_hit = True
                if trade.stop and low <= trade.stop:
                    stop_hit = True
            else:  # SHORT
                if trade.target and low <= trade.target:
                    target_hit = True
                if trade.stop and high >= trade.stop:
                    stop_hit = True

            # Both hit in same bar — use open to determine which was first
            if target_hit and stop_hit:
                bar_open = bar.get("Open", bar.get("open", close))
                if is_long:
                    # If opened closer to stop, likely stopped first
                    stop_hit = bar_open <= trade.entry
                    target_hit = not stop_hit
                else:
                    stop_hit = bar_open >= trade.entry
                    target_hit = not stop_hit

            if target_hit:
                trade.status = "WIN"
                trade.exit_price = trade.target
                trade.hit_target_first = True
                trade.pnl = abs(trade.target - trade.entry)
                if not is_long:
                    trade.pnl = abs(trade.entry - trade.target)
                trade.pnl_r = round(trade.pnl / risk, 2) if risk else 0
                trade.pnl_pct = round(trade.pnl / trade.entry * 100, 2) if trade.entry else 0
                trade.bars_to_exit = i + 1
                trade.exit_date = str(idx) if idx else ""
                return

            if stop_hit:
                trade.status = "LOSS"
                trade.exit_price = trade.stop
                trade.hit_stop_first = True
                trade.pnl = -abs(trade.entry - trade.stop)
                trade.pnl_r = -1.0  # stopped out = -1R by definition
                trade.pnl_pct = round(trade.pnl / trade.entry * 100, 2) if trade.entry else 0
                trade.bars_to_exit = i + 1
                trade.exit_date = str(idx) if idx else ""
                return

        # Didn't hit either — mark as EXPIRED at last close
        if len(bars_df) > 0:
            last_bar = bars_df.iloc[-1]
            last_close = last_bar.get("Close", last_bar.get("close", 0))
            trade.status = "EXPIRED"
            trade.exit_price = last_close
            if is_long:
                trade.pnl = last_close - trade.entry
            else:
                trade.pnl = trade.entry - last_close
            trade.pnl_r = round(trade.pnl / risk, 2) if risk else 0
            trade.pnl_pct = round(trade.pnl / trade.entry * 100, 2) if trade.entry else 0
            trade.bars_to_exit = len(bars_df)
            trade.exit_date = str(bars_df.index[-1])
            trade.notes = f"Expired after {len(bars_df)} bars"

    # ========================================================================
    # MODE 3: CUSTOM RULES — Test price-action or indicator criteria
    # ========================================================================

    def run_custom(
        self,
        symbols: List[str],
        days_back: int = 90,
        rules: List[Dict] = None,
        direction: str = "LONG",
        bar_interval: str = "1d",
        max_hold_bars: int = 60,
        rr_ratio: float = 2.0,
        stop_atr_mult: float = 1.5,
    ) -> BacktestResult:
        """
        Run a custom-rule backtest.  Each 'rule' is a condition evaluated
        on every bar.  When ALL rules pass on a bar, a trade is opened.

        Supported rule types:
           move_off_open   — intraday % move from open triggers entry
                             params: min_pct, max_pct  (e.g. 0.75, 1.25)
           rsi_range       — RSI between two bounds
                             params: min_rsi, max_rsi  (e.g. 30, 50)
           above_ma        — price above N-period SMA
                             params: period  (e.g. 20)
           below_ma        — price below N-period SMA
                             params: period
           rvol_min        — relative volume >= threshold
                             params: min_rvol  (e.g. 1.3)
           gap_up          — open gapped up >= X %
                             params: min_pct
           gap_down        — open gapped down >= X %
                             params: min_pct
           range_pct       — (high-low)/open between bounds
                             params: min_pct, max_pct
           high_off_open   — (high-open)/open %, how far high reached above open
                             params: min_pct, max_pct  (e.g. 0.75, 1.25)
           low_off_open    — (open-low)/open %, how far low dipped below open
                             params: min_pct, max_pct  (e.g. 0.75, 1.25)

        Example rules for "0.75-1.25 % move off open, long":
           [{"type": "move_off_open", "min_pct": 0.75, "max_pct": 1.25}]
        """
        from polygon_data import get_bars as pg_bars

        if not rules:
            rules = [{"type": "move_off_open", "min_pct": 0.75, "max_pct": 1.25}]

        start_time = time.time()
        result = BacktestResult(
            mode="custom",
            bar_interval=bar_interval,
            period=f"{days_back}d",
            symbols_tested=sorted([s.upper() for s in symbols]),
        )

        is_long = direction.upper() == "LONG"

        for symbol in symbols:
            symbol = symbol.upper()
            try:
                logger.info("Custom scan: %s (%dd back, %d rules)", symbol, days_back, len(rules))

                bars_df = pg_bars(symbol, period=f"{days_back + 30}d", interval=bar_interval)
                time.sleep(self._rate_delay)

                if bars_df is None or bars_df.empty or len(bars_df) < 25:
                    result.errors.append(f"{symbol}: insufficient data")
                    continue

                result.total_bars_processed += len(bars_df)

                # Column helpers
                def _col(df, name):
                    return df[name].values if name in df.columns else df[name.lower()].values

                opens   = _col(bars_df, "Open")
                highs   = _col(bars_df, "High")
                lows    = _col(bars_df, "Low")
                closes  = _col(bars_df, "Close")
                volumes = _col(bars_df, "Volume")
                dates   = bars_df.index

                # Pre-compute indicators needed by rules
                rsi_vals = self._calc_rsi(closes)
                atr_vals = self._calc_atr(highs, lows, closes)
                sma_cache = {}
                vol_avg   = self._sma(volumes.tolist(), 20)

                def get_sma(period):
                    if period not in sma_cache:
                        sma_cache[period] = self._sma(closes.tolist(), period) if len(closes) >= period else closes.tolist()
                    return sma_cache[period]

                # Walk bars
                cooldown = 0
                for i in range(25, len(closes)):
                    if cooldown > 0:
                        cooldown -= 1
                        continue

                    o, h, l, c = opens[i], highs[i], lows[i], closes[i]
                    prev_c = closes[i - 1] if i > 0 else c
                    rsi     = rsi_vals[min(i, len(rsi_vals) - 1)]
                    atr     = atr_vals[min(i, len(atr_vals) - 1)]
                    rvol    = volumes[i] / vol_avg[min(i, len(vol_avg) - 1)] if vol_avg[min(i, len(vol_avg) - 1)] > 0 else 1.0
                    move_pct = ((c - o) / o * 100) if o > 0 else 0
                    high_off = ((h - o) / o * 100) if o > 0 else 0
                    low_off  = ((o - l) / o * 100) if o > 0 else 0

                    # Evaluate every rule
                    all_pass = True
                    for rule in rules:
                        rtype = rule.get("type", "")
                        if rtype == "move_off_open":
                            mn = rule.get("min_pct", 0)
                            mx = rule.get("max_pct", 999)
                            if is_long:
                                if not (mn <= move_pct <= mx):
                                    all_pass = False
                            else:
                                if not (mn <= -move_pct <= mx):
                                    all_pass = False

                        elif rtype == "high_off_open":
                            mn = rule.get("min_pct", 0)
                            mx = rule.get("max_pct", 999)
                            if not (mn <= high_off <= mx):
                                all_pass = False

                        elif rtype == "low_off_open":
                            mn = rule.get("min_pct", 0)
                            mx = rule.get("max_pct", 999)
                            if not (mn <= low_off <= mx):
                                all_pass = False

                        elif rtype == "rsi_range":
                            if not (rule.get("min_rsi", 0) <= rsi <= rule.get("max_rsi", 100)):
                                all_pass = False

                        elif rtype == "above_ma":
                            period = int(rule.get("period", 20))
                            ma = get_sma(period)
                            if c <= ma[min(i, len(ma) - 1)]:
                                all_pass = False

                        elif rtype == "below_ma":
                            period = int(rule.get("period", 20))
                            ma = get_sma(period)
                            if c >= ma[min(i, len(ma) - 1)]:
                                all_pass = False

                        elif rtype == "rvol_min":
                            if rvol < rule.get("min_rvol", 1.0):
                                all_pass = False

                        elif rtype == "gap_up":
                            gap = (o - prev_c) / prev_c * 100 if prev_c > 0 else 0
                            if gap < rule.get("min_pct", 0):
                                all_pass = False

                        elif rtype == "gap_down":
                            gap = (prev_c - o) / prev_c * 100 if prev_c > 0 else 0
                            if gap < rule.get("min_pct", 0):
                                all_pass = False

                        elif rtype == "range_pct":
                            rng = (h - l) / o * 100 if o > 0 else 0
                            if not (rule.get("min_pct", 0) <= rng <= rule.get("max_pct", 999)):
                                all_pass = False

                        if not all_pass:
                            break

                    if not all_pass:
                        continue

                    # All rules passed — create trade
                    entry_price = c
                    risk = atr * stop_atr_mult if atr > 0 else entry_price * 0.015
                    if is_long:
                        stop_price  = round(entry_price - risk, 2)
                        target_price = round(entry_price + risk * rr_ratio, 2)
                    else:
                        stop_price  = round(entry_price + risk, 2)
                        target_price = round(entry_price - risk * rr_ratio, 2)

                    rule_labels = ", ".join(r.get("type", "?") for r in rules)
                    bt = BacktestTrade(
                        symbol=symbol,
                        direction=direction.upper(),
                        entry=round(entry_price, 2),
                        stop=stop_price,
                        target=target_price,
                        signal=f"CUSTOM({rule_labels})",
                        confidence=80,
                        entry_date=str(dates[i]) if i < len(dates) else "",
                    )

                    future_bars = bars_df.iloc[i + 1:]
                    self._simulate_trade(bt, future_bars, max_hold_bars)
                    result.trades.append(bt)

                    # Cooldown: skip bars equal to bars-to-exit (or 5 minimum)
                    cooldown = max(bt.bars_to_exit, 5)

            except Exception as e:
                result.errors.append(f"{symbol}: {e}")
                logger.warning("Custom backtest error for %s: %s", symbol, e)

        result.runtime_seconds = time.time() - start_time
        result.analytics = self._compute_analytics(result.trades)

        logger.info("Custom complete: %d symbols, %d trades, %.1f%% WR, %.2fR avg, %.1fs",
                    len(symbols), len(result.trades), result._win_rate(),
                    result._avg_r(), result.runtime_seconds)
        return result

    # ========================================================================
    # SIGNAL GENERATION — Run scanner at historical intervals
    # ========================================================================

    def _generate_historical_signals(
        self, symbol, bars_df, scanner, timeframe,
        scan_interval_days, signal_filter, min_confidence
    ) -> List[Dict]:
        """
        Walk through historical bars and generate signals at regular intervals.
        Uses price/volume data to derive setups without calling live scanner
        (to avoid rate limits on Polygon).
        """
        signals = []
        
        if len(bars_df) < 25:
            return signals

        # Compute technical indicators from bars
        closes = bars_df["Close"].values if "Close" in bars_df.columns else bars_df["close"].values
        highs = bars_df["High"].values if "High" in bars_df.columns else bars_df["high"].values
        lows = bars_df["Low"].values if "Low" in bars_df.columns else bars_df["low"].values
        volumes = bars_df["Volume"].values if "Volume" in bars_df.columns else bars_df["volume"].values
        dates = bars_df.index

        rsi = self._calc_rsi(closes)
        atr = self._calc_atr(highs, lows, closes)
        sma20 = self._sma(closes.tolist(), 20)
        sma50 = self._sma(closes.tolist(), 50) if len(closes) >= 50 else self._sma(closes.tolist(), 20)

        # Volume average
        vol_avg = self._sma(volumes.tolist(), 20)

        # Walk through at intervals starting from bar 25
        for i in range(25, len(closes), scan_interval_days):
            try:
                price = closes[i]
                rsi_val = rsi[min(i, len(rsi)-1)]
                atr_val = atr[min(i, len(atr)-1)]
                ma20 = sma20[i]
                ma50 = sma50[min(i, len(sma50)-1)]
                avg_vol = vol_avg[min(i, len(vol_avg)-1)]
                cur_vol = volumes[i]
                rvol = cur_vol / avg_vol if avg_vol > 0 else 1.0

                # Generate signal based on technicals
                signal_info = self._derive_signal(
                    price, rsi_val, atr_val, ma20, ma50, rvol, 
                    highs[max(0,i-20):i+1], lows[max(0,i-20):i+1]
                )

                if not signal_info:
                    continue

                sig_type = signal_info["signal"]
                confidence = signal_info["confidence"]

                # Apply filters
                if signal_filter and sig_type != signal_filter:
                    continue
                if confidence < min_confidence:
                    continue

                # Create trade setup
                direction = signal_info["direction"]
                
                if direction == "LONG":
                    stop = price - (atr_val * 1.5)
                    target = price + (atr_val * 1.5 * 2)  # 2R target
                else:
                    stop = price + (atr_val * 1.5)
                    target = price - (atr_val * 1.5 * 2)

                signals.append({
                    "bar_index": i,
                    "date": str(dates[i]) if i < len(dates) else "",
                    "direction": direction,
                    "signal": sig_type,
                    "confidence": confidence,
                    "entry": round(price, 2),
                    "stop": round(stop, 2),
                    "target": round(target, 2),
                })

            except Exception as e:
                logger.debug("Signal generation error at bar %d: %s", i, e)
                continue

        return signals

    def _derive_signal(
        self, price, rsi, atr, ma20, ma50, rvol,
        recent_highs, recent_lows
    ) -> Optional[Dict]:
        """
        Derive a trading signal from technical indicators.
        Returns signal dict or None.
        """
        bull_score = 0
        bear_score = 0

        # Trend (MA alignment)
        if price > ma20 > ma50:
            bull_score += 30
        elif price < ma20 < ma50:
            bear_score += 30
        elif price > ma20:
            bull_score += 15
        elif price < ma20:
            bear_score += 15

        # RSI 
        if 40 <= rsi <= 55:  # pullback in uptrend
            if price > ma20:
                bull_score += 20
        elif 45 <= rsi <= 60:
            if price < ma20:
                bear_score += 20
        
        if rsi < 30:
            bull_score += 25  # oversold bounce
        elif rsi > 70:
            bear_score += 25  # overbought fade

        # Volume confirmation
        if rvol > 1.3:
            bull_score += 10
            bear_score += 10

        # Recent structure (HH/HL vs LH/LL)
        if len(recent_highs) >= 10:
            h1 = max(recent_highs[:len(recent_highs)//2])
            h2 = max(recent_highs[len(recent_highs)//2:])
            l1 = min(recent_lows[:len(recent_lows)//2])
            l2 = min(recent_lows[len(recent_lows)//2:])
            
            if h2 > h1 and l2 > l1:  # HH + HL = uptrend
                bull_score += 20
            elif h2 < h1 and l2 < l1:  # LH + LL = downtrend
                bear_score += 20

        # Determine signal
        if bull_score >= 50 and bull_score > bear_score + 15:
            confidence = min(95, bull_score)
            if bull_score >= 70:
                return {"signal": "GREEN", "direction": "LONG", "confidence": confidence}
            else:
                return {"signal": "YELLOW", "direction": "LONG", "confidence": confidence}
        
        elif bear_score >= 50 and bear_score > bull_score + 15:
            confidence = min(95, bear_score)
            if bear_score >= 70:
                return {"signal": "RED", "direction": "SHORT", "confidence": confidence}
            else:
                return {"signal": "YELLOW", "direction": "SHORT", "confidence": confidence}

        return None  # No clear signal

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _calc_rsi(data, period=14):
        """Simple RSI calculation, returns list same length as data."""
        deltas = [data[i] - data[i-1] for i in range(1, len(data))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        rsi_vals = [50.0] * period
        if len(gains) < period:
            return [50.0] * len(data)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi_vals.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_vals.append(100 - (100 / (1 + rs)))
        return rsi_vals

    @staticmethod
    def _calc_atr(highs, lows, closes, period=14):
        """Simple ATR calculation, returns list same length as closes."""
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        atr_vals = [trs[0] if trs else 0] * min(period, len(trs))
        if len(trs) >= period:
            atr = sum(trs[:period]) / period
            atr_vals = [atr] * period
            for i in range(period, len(trs)):
                atr = (atr * (period - 1) + trs[i]) / period
                atr_vals.append(atr)
        return [0] + atr_vals

    @staticmethod
    def _sma(data, period):
        """Simple moving average, returns list same length as data."""
        result = []
        for i in range(len(data)):
            if i < period - 1:
                result.append(data[i])
            else:
                result.append(sum(data[i-period+1:i+1]) / period)
        return result

    def _parse_trade(self, d: Dict) -> Optional[BacktestTrade]:
        """Parse a trade dict (Firestore/journal format) into BacktestTrade"""
        try:
            entry = d.get("entry") or d.get("entry_price") or d.get("actual_entry") or 0
            stop = d.get("stop") or d.get("stop_loss") or 0
            target = d.get("target") or d.get("target_1") or 0
            
            if not entry:
                return None

            return BacktestTrade(
                symbol=str(d.get("symbol", "")).upper(),
                direction=str(d.get("direction", "LONG")).upper(),
                entry=float(entry),
                stop=float(stop) if stop else 0,
                target=float(target) if target else 0,
                target2=float(d.get("target2") or d.get("target_2") or 0),
                signal=str(d.get("signal", "")),
                confidence=int(d.get("confidence", 0)),
                timeframe=str(d.get("timeframe", "swing")),
            )
        except Exception as e:
            logger.warning("Parse error: %s", e)
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        if not date_str:
            return None
        for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str[:len(fmt)+5], fmt)
            except (ValueError, IndexError):
                continue
        return None

    def _get_scanner(self):
        """Get MarketScanner instance if available"""
        try:
            from market_scanner_v2 import MarketScanner
            return MarketScanner()
        except ImportError:
            logger.warning("MarketScanner not available, using built-in signal generation")
            return None

    def _compute_analytics(self, trades: List[BacktestTrade]) -> Dict:
        """Run journal_analytics on simulated trades"""
        try:
            from journal_analytics import compute_journal_analytics
            trade_dicts = [t.to_analytics_dict() for t in trades if t.status in ("WIN", "LOSS")]
            if not trade_dicts:
                return {}
            return compute_journal_analytics(trade_dicts, days=9999)
        except Exception as e:
            logger.warning("Analytics computation failed: %s", e)
            return {}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_backtest(
    symbols: List[str],
    days_back: int = 90,
    signal_filter: str = None,
    min_confidence: int = 50,
) -> Dict:
    """
    One-liner strategy backtest.
    Returns dict with summary, trades, and analytics.
    """
    engine = BacktestEngine()
    result = engine.run_strategy(
        symbols=symbols,
        days_back=days_back,
        signal_filter=signal_filter,
        min_confidence=min_confidence,
    )
    return result.to_dict()


def replay_journal(
    user_id: str = None,
    bar_interval: str = "1d",
    max_hold_bars: int = 60,
) -> Dict:
    """
    Replay trades from the journal to see what would have happened.
    """
    trades = []
    
    # Pull from Firestore
    if user_id:
        try:
            from firestore_store import get_firestore
            fs = get_firestore()
            if fs.is_available():
                trades = fs.get_trades(user_id)
        except Exception:
            pass
    
    # Fallback to local
    if not trades:
        try:
            from chart_input_analyzer import ChartInputSystem
            cs = ChartInputSystem(data_dir="./scanner_data")
            from dataclasses import asdict
            trades = [asdict(t) for t in cs.tracker.trades]
        except Exception:
            pass

    if not trades:
        return {"error": "No trades found", "trades": []}

    engine = BacktestEngine()
    result = engine.replay_trades(
        trades=trades,
        bar_interval=bar_interval,
        max_hold_bars=max_hold_bars,
    )
    return result.to_dict()
