"""
Regime Scanner — Generic Cross-Gate Strategy Engine
====================================================
Converts the IWM V6 dip-buy / Mode A system into a universal scanner
that works on ANY stock by replacing fixed dollar thresholds with
ATR-based and percentage-based levels that auto-scale to price.

Plugs into:
  - backtest_engine.run_cross_analysis()  → raw cross/VWAP data
  - backtest_engine.run_ohlc_analysis()   → daily OHLC + gap data
  - polygon_data.get_bars()               → price/volume bars

Usage:
  from regime_scanner import RegimeScanner
  
  scanner = RegimeScanner()
  
  # Single symbol — full analysis
  result = scanner.scan("AAPL", days_back=30)
  
  # Multi-symbol watchlist scan
  results = scanner.scan_watchlist(
      ["AAPL", "NVDA", "TSLA", "META", "MSFT", "AMD", "AMZN", "GOOG", "IWM", "SPY"],
      days_back=30
  )
  
  # Get strategy levels for tomorrow's session
  levels = scanner.get_strategy_levels("NVDA")
  
  # Run full V6 backtest with auto-scaled levels
  backtest = scanner.run_regime_backtest("TSLA", days_back=60)

Output format matches the Regime Tracker UI — bull/bear/choppy counts,
rolling windows, strategy implications, and per-day detail.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("regime_scanner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [RegimeScanner] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)


def _sanitize(obj):
    """Recursively convert numpy/pandas types to native Python for JSON."""
    import math
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isinf(v) or math.isnan(v):
                return 0
            return v
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_sanitize(v) for v in obj.tolist()]
    except ImportError:
        pass
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return 0
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ============================================================================
# IWM BASELINE — The calibration reference
# ============================================================================
# All thresholds were discovered on IWM at ~$265. We convert to percentages
# so they scale to any price level.
#
# IWM $0.50 validation  = 0.189%  →  use ~0.19% of price
# IWM $0.75 half-exit   = 0.283%  →  use ~0.28% of price
# IWM $1.25 full target  = 0.472%  →  use ~0.47% of price
# IWM -$0.20 dip entry  = 0.075%  →  use ~0.08% of price
# IWM -$0.35 dip max    = 0.132%  →  use ~0.13% of price
# IWM -$0.40 fast drop  = 0.151%  →  use ~0.15% of price
# IWM -$0.55 put flip   = 0.208%  →  use ~0.21% of price
# IWM $0.86 stop (-30%) = 0.325%  →  use ~0.33% of price (or 30% of option)
#
# HOWEVER: For higher-volatility stocks (TSLA, AMD), these % thresholds
# need to be ATR-adjusted. A 0.19% move on NVDA ($140) is $0.27 — nothing.
# On TSLA ($350), 0.19% is $0.67 — also small. But ATR captures the real
# volatility of each name.
#
# FINAL APPROACH: Use ATR multipliers as the primary scaling, with the
# IWM-derived percentages as a floor/ceiling sanity check.
# ============================================================================

# ATR multipliers (calibrated from IWM: ATR ~$3.50, price ~$265)
# IWM dip entry ($0.20-$0.35) = 0.057-0.100 × ATR
# IWM validation ($0.50) = 0.143 × ATR
# IWM half exit ($0.75) = 0.214 × ATR
# IWM full target ($1.25) = 0.357 × ATR
# IWM fast drop ($0.40) = 0.114 × ATR
# IWM put flip ($0.55) = 0.157 × ATR

ATR_MULT = {
    "dip_entry_min":   0.06,    # minimum dip to enter (ATR × this)
    "dip_entry_max":   0.10,    # maximum dip to enter
    "validation":      0.14,    # must reach this above open to confirm
    "half_exit":       0.21,    # close half position
    "full_target":     0.36,    # close remaining / 33% option profit
    "fast_drop":       0.11,    # fast drop defense exit
    "put_flip":        0.16,    # flip to put level
    "vwap_ext":        0.01,    # 1% VWAP extension (percentage, not ATR)
    "mode_a_entry":    0.14,    # Mode A confirmation level
    "stop_pct":        0.30,    # 30% of option premium (fixed)
}

# Cross gate thresholds (same for all tickers — this is price behavior, not price level)
CROSS_GATE = {
    "directional_max":  2,      # ≤2 crosses at 9:35 = directional
    "moderate_max":     5,      # 3-5 = moderate
    "choppy_min":       6,      # ≥6 = choppy (dip-buy territory)
    "observe_minutes":  5,      # count crosses for first 5 minutes
}

# Regime classification thresholds
REGIME_THRESHOLDS = {
    "bull_close_pct":   0.19,   # close > open + 0.19% = bull (IWM $0.50/$265)
    "bear_close_pct":  -0.19,   # close < open - 0.19% = bear
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StrategyLevels:
    """Auto-scaled strategy levels for a specific symbol"""
    symbol: str
    price: float                # current/reference price
    atr: float                  # 14-period ATR
    atr_pct: float              # ATR as % of price
    
    # Dip-buy levels (dollar amounts)
    dip_entry_min: float = 0    # minimum dip to enter
    dip_entry_max: float = 0    # maximum dip to enter
    validation: float = 0       # must reach above open
    half_exit: float = 0        # close half
    full_target: float = 0      # close remaining
    fast_drop: float = 0        # fast drop defense
    put_flip: float = 0         # flip to put
    vwap_ext_pct: float = 0.01  # 1% VWAP extension
    
    # Mode A levels
    mode_a_entry: float = 0     # confirmation level
    mode_a_target: float = 0    # same as full_target
    
    # Option sizing reference
    option_premium_est: float = 0    # estimated ATM call price
    stop_dollar: float = 0           # 30% of premium
    
    # Percentage equivalents (for display)
    dip_entry_min_pct: float = 0
    dip_entry_max_pct: float = 0
    validation_pct: float = 0
    half_exit_pct: float = 0
    full_target_pct: float = 0
    
    def to_dict(self) -> Dict:
        return _sanitize(asdict(self))


@dataclass
class DayClassification:
    """Classification for a single trading day"""
    date: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    crosses: int
    vwap: float
    vwap_crosses: int
    pct_above_open: float
    pct_above_vwap: float
    
    # Classification
    regime: str = ""              # BULL, BEAR, CHOPPY
    strategy: str = ""            # DIP-BUY, MODE_A_LONG, MODE_A_SHORT, NO_DIP, SKIP
    
    # V6 filter results
    cross_gate: str = ""          # PASSED, BLOCKED
    hit_validation: bool = False  # did price reach validation level
    hit_half_exit: bool = False
    hit_full_target: bool = False
    hit_fast_drop: bool = False
    hit_put_flip: bool = False
    
    # P&L simulation
    simulated_pnl_pct: float = 0
    simulated_outcome: str = ""   # WIN_FULL, WIN_HALF, LOSS_FILTERED, LOSS_STOPPED, SKIP
    
    # Raw data
    high_off_open: float = 0
    low_off_open: float = 0
    close_vs_open: float = 0
    range_pct: float = 0
    
    def to_dict(self) -> Dict:
        return _sanitize(asdict(self))


@dataclass
class RegimeScanResult:
    """Complete scan result for one symbol"""
    symbol: str
    days_analyzed: int = 0
    lookback: int = 30
    
    # Strategy levels
    levels: Optional[StrategyLevels] = None
    
    # Regime counts
    bull_days: int = 0
    bear_days: int = 0
    choppy_days: int = 0
    
    # Strategy counts
    dip_buy_days: int = 0
    mode_a_days: int = 0
    no_dip_days: int = 0
    skip_days: int = 0
    
    # Performance
    dip_buy_wins: int = 0
    dip_buy_losses: int = 0
    mode_a_wins: int = 0
    mode_a_losses: int = 0
    total_pnl_pct: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    
    # Aggregates from cross analysis
    avg_crosses: float = 0
    avg_vwap_crosses: float = 0
    green_pct: float = 0
    avg_high_off_open_pct: float = 0
    avg_low_off_open_pct: float = 0
    
    # Day-by-day detail
    days: List[DayClassification] = field(default_factory=list)
    
    # Rolling windows
    windows: Dict = field(default_factory=dict)
    
    # Streak
    streak_type: str = ""
    streak_count: int = 0
    
    # Current month
    current_month_counts: Dict = field(default_factory=dict)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    runtime: float = 0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["days"] = [day.to_dict() if hasattr(day, "to_dict") else _sanitize(day) for day in self.days]
        if self.levels:
            d["levels"] = self.levels.to_dict()
        return _sanitize(d)


# ============================================================================
# REGIME SCANNER
# ============================================================================

class RegimeScanner:
    """
    Generic regime scanner that applies the V6 cross-gate / dip-buy / Mode A
    strategy to any stock by auto-scaling thresholds based on ATR.
    """
    
    def __init__(self, rate_limit_delay: float = 0.1):
        self._rate_delay = rate_limit_delay
        self._atr_mult = ATR_MULT.copy()
        self._cross_gate = CROSS_GATE.copy()
        self._cache: Dict[str, tuple] = {}   # symbol -> (timestamp, result)
        self._cache_ttl = 300                 # 5 minutes

    def _get_cached(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _set_cached(self, key: str, value):
        self._cache[key] = (time.time(), value)
    
    # ========================================================================
    # PUBLIC: Scan a single symbol
    # ========================================================================
    
    def scan(self, symbol: str, days_back: int = 30) -> RegimeScanResult:
        """
        Full regime scan for a single symbol.
        
        1. Pull cross analysis data (1-min bars, open crosses, VWAP)
        2. Pull daily bars for ATR calculation
        3. Compute strategy levels (ATR-scaled)
        4. Classify each day
        5. Simulate V6 strategy
        6. Compute regime stats and rolling windows
        
        Returns RegimeScanResult with everything the tracker UI needs.
        """
        symbol = symbol.upper()
        cache_key = f"{symbol}:{days_back}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        start_time = time.time()
        result = RegimeScanResult(symbol=symbol, lookback=days_back)
        
        try:
            # Step 1: Get cross analysis data
            cross_data = self._get_cross_data(symbol, days_back)
            if not cross_data or not cross_data.get("days"):
                result.errors.append(f"No cross data for {symbol}")
                return result
            
            # Step 2: Get ATR from daily bars
            atr, avg_price = self._get_atr(symbol, days_back)
            if atr <= 0:
                # Fallback: estimate ATR from cross data range
                days = cross_data["days"]
                ranges = [d["high"] - d["low"] for d in days if d["high"] > d["low"]]
                atr = sum(ranges) / len(ranges) if ranges else avg_price * 0.015
                logger.warning("%s: ATR fallback from range data: $%.2f", symbol, atr)
            
            # Step 3: Compute strategy levels
            levels = self._compute_levels(symbol, avg_price, atr)
            result.levels = levels
            
            # Step 4: Classify each day and simulate
            classified_days = []
            for day_data in cross_data["days"]:
                day = self._classify_day(day_data, levels)
                classified_days.append(day)
            
            result.days = sorted(classified_days, key=lambda d: d.date, reverse=True)
            result.days_analyzed = len(classified_days)
            
            # Step 5: Compute regime stats
            self._compute_stats(result)
            
            # Step 6: Rolling windows
            self._compute_windows(result)
            
            # Step 7: Streak detection
            self._compute_streak(result)
            
            # Step 8: Current month breakdown
            self._compute_monthly(result)
            
            # Copy cross analysis aggregates
            summary = cross_data.get("summary", {})
            result.avg_crosses = summary.get("avg_crosses", 0)
            result.avg_vwap_crosses = summary.get("avg_vwap_crosses", 0)
            result.green_pct = summary.get("green_pct", 0)
            result.avg_high_off_open_pct = summary.get("avg_high_off_open_pct", 0)
            result.avg_low_off_open_pct = summary.get("avg_low_off_open_pct", 0)
            
        except Exception as e:
            result.errors.append(f"{symbol}: {e}")
            logger.error("Scan error for %s: %s", symbol, e, exc_info=True)
        
        result.runtime = round(time.time() - start_time, 2)
        logger.info("%s scan complete: %d days, %d bull/%d bear/%d choppy, %.1f%% WR, %.1fs",
                    symbol, result.days_analyzed, result.bull_days, result.bear_days,
                    result.choppy_days, result.win_rate, result.runtime)
        self._set_cached(cache_key, result)
        return result
    
    # ========================================================================
    # PUBLIC: Scan a watchlist
    # ========================================================================
    
    def scan_watchlist(
        self,
        symbols: List[str],
        days_back: int = 30,
    ) -> Dict[str, RegimeScanResult]:
        """
        Scan multiple symbols in parallel and return results keyed by ticker.
        Phase 1: Batch-prefetch all cross data + ATR in parallel (I/O bound).
        Phase 2: Run classification (CPU-only, instant) sequentially.
        """
        results = {}
        start_time = time.time()
        clean = [s.strip().upper() for s in symbols if s.strip()]

        # ── Phase 1: Parallel I/O — prefetch cross data + ATR into cache ──
        def _prefetch_cross(sym):
            return (f"cross:{sym}", self._get_cross_data(sym, days_back))

        def _prefetch_atr(sym):
            return (f"atr:{sym}", self._get_atr(sym, days_back))

        tasks = []
        for sym in clean:
            tasks.append(("cross", sym))
            tasks.append(("atr", sym))

        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {}
            for kind, sym in tasks:
                if kind == "cross":
                    futs[pool.submit(_prefetch_cross, sym)] = (kind, sym)
                else:
                    futs[pool.submit(_prefetch_atr, sym)] = (kind, sym)
            for fut in as_completed(futs):
                kind, sym = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.error("Prefetch %s for %s failed: %s", kind, sym, e)

        prefetch_time = time.time() - start_time
        logger.info("Prefetch done for %d symbols in %.1fs", len(clean), prefetch_time)

        # ── Phase 2: Classify (all data is cached, no I/O) ──
        for sym in clean:
            try:
                results[sym] = self.scan(sym, days_back)
            except Exception as e:
                logger.error("Scan error for %s: %s", sym, e)
                results[sym] = RegimeScanResult(symbol=sym, errors=[str(e)])

        total_time = time.time() - start_time
        logger.info("Watchlist scan complete: %d symbols in %.1fs (prefetch %.1fs)",
                    len(results), total_time, prefetch_time)

        return results
    
    # ========================================================================
    # PUBLIC: Get strategy levels for tomorrow's session
    # ========================================================================
    
    def get_strategy_levels(self, symbol: str) -> Dict:
        """
        Quick lookup: compute and return the ATR-scaled strategy levels
        for a symbol without running full day classification.
        
        Returns a dict with dollar amounts for each threshold.
        """
        symbol = symbol.upper()
        atr, price = self._get_atr(symbol, 30)
        if atr <= 0 or price <= 0:
            return {"error": f"Could not get ATR for {symbol}"}
        
        levels = self._compute_levels(symbol, price, atr)
        
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "atr": round(atr, 2),
            "atr_pct": round(atr / price * 100, 2),
            "levels": {
                "dip_entry_zone": f"-${levels.dip_entry_min:.2f} to -${levels.dip_entry_max:.2f}",
                "dip_entry_min": round(levels.dip_entry_min, 2),
                "dip_entry_max": round(levels.dip_entry_max, 2),
                "validation": round(levels.validation, 2),
                "half_exit": round(levels.half_exit, 2),
                "full_target": round(levels.full_target, 2),
                "fast_drop": round(levels.fast_drop, 2),
                "put_flip": round(levels.put_flip, 2),
                "mode_a_entry": round(levels.mode_a_entry, 2),
                "vwap_1pct": round(price * 0.01, 2),
            },
            "option_reference": {
                "est_atm_premium": round(levels.option_premium_est, 2),
                "stop_30pct": round(levels.stop_dollar, 2),
                "target_33pct": round(levels.option_premium_est * 0.33, 2),
            },
            "iwm_equivalent": {
                "note": "These levels correspond to the IWM V6 system at ~$265",
                "dip_entry": "$0.20 – $0.35",
                "validation": "$0.50",
                "half_exit": "$0.75",
                "full_target": "$1.25",
                "fast_drop": "$0.40",
                "put_flip": "$0.55",
            },
        }
    
    # ========================================================================
    # PUBLIC: Generate watchlist comparison table
    # ========================================================================
    
    def compare_watchlist(self, results: Dict[str, RegimeScanResult]) -> Dict:
        """
        Given scan results from scan_watchlist(), produce a comparison table
        ranking symbols by strategy fit.
        """
        comparison = []
        for sym, r in results.items():
            if r.days_analyzed == 0:
                continue
            comparison.append({
                "symbol": sym,
                "price": r.levels.price if r.levels else 0,
                "atr": r.levels.atr if r.levels else 0,
                "atr_pct": r.levels.atr_pct if r.levels else 0,
                "days": r.days_analyzed,
                "bull": r.bull_days,
                "bear": r.bear_days,
                "choppy": r.choppy_days,
                "choppy_pct": round(r.choppy_days / r.days_analyzed * 100, 1) if r.days_analyzed else 0,
                "dip_buy_days": r.dip_buy_days,
                "mode_a_days": r.mode_a_days,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "total_pnl_pct": r.total_pnl_pct,
                "green_pct": r.green_pct,
                "avg_crosses": r.avg_crosses,
                # Regime score: higher choppy% + higher win rate = better dip-buy candidate
                "dip_buy_score": round(
                    (r.choppy_days / r.days_analyzed * 50 if r.days_analyzed else 0) +
                    (r.win_rate * 0.5),
                    1
                ),
                "streak": f"{r.streak_count} {r.streak_type}",
            })
        
        # Sort by dip_buy_score descending
        comparison.sort(key=lambda x: x["dip_buy_score"], reverse=True)
        
        return {
            "comparison": comparison,
            "best_dip_buy": comparison[0]["symbol"] if comparison else None,
            "best_mode_a": max(comparison, key=lambda x: x["mode_a_days"])["symbol"] if comparison else None,
            "highest_win_rate": max(comparison, key=lambda x: x["win_rate"])["symbol"] if comparison else None,
        }
    
    # ========================================================================
    # INTERNAL: Get cross analysis data
    # ========================================================================
    
    def _get_cross_data(self, symbol: str, days_back: int) -> Optional[Dict]:
        """Pull cross analysis from backtest engine (with cache)"""
        cache_key = f"cross:{symbol}:{days_back}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        try:
            from backtest_engine import BacktestEngine
            engine = BacktestEngine(rate_limit_delay=self._rate_delay)
            data = engine.run_cross_analysis(symbols=[symbol], days_back=days_back)
            if data:
                self._set_cached(cache_key, data)
            return data
        except ImportError:
            logger.error("backtest_engine not available")
            return None
        except Exception as e:
            logger.error("Cross analysis failed for %s: %s", symbol, e)
            return None
    
    # ========================================================================
    # INTERNAL: Get ATR from daily bars
    # ========================================================================
    
    def _get_atr(self, symbol: str, days_back: int, period: int = 14) -> tuple:
        """
        Pull daily bars and compute 14-period ATR.
        Returns (atr_value, average_price).
        """
        cache_key = f"atr:{symbol}:{days_back}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        try:
            from polygon_data import get_bars
            bars = get_bars(symbol, period=f"{days_back + 20}d", interval="1d")
            
            if bars is None or bars.empty or len(bars) < period + 1:
                return (0.0, 0.0)
            
            highs = bars["High"].values if "High" in bars.columns else bars["high"].values
            lows = bars["Low"].values if "Low" in bars.columns else bars["low"].values
            closes = bars["Close"].values if "Close" in bars.columns else bars["close"].values
            
            # True Range
            trs = []
            for i in range(1, len(closes)):
                tr = max(
                    float(highs[i]) - float(lows[i]),
                    abs(float(highs[i]) - float(closes[i-1])),
                    abs(float(lows[i]) - float(closes[i-1]))
                )
                trs.append(tr)
            
            if len(trs) < period:
                atr = sum(trs) / len(trs) if trs else 0
            else:
                atr = sum(trs[-period:]) / period
            
            avg_price = float(closes[-1])  # use last close as reference
            result = (round(atr, 4), round(avg_price, 2))
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.warning("ATR calculation failed for %s: %s", symbol, e)
            return (0.0, 0.0)
    
    # ========================================================================
    # INTERNAL: Compute ATR-scaled strategy levels
    # ========================================================================
    
    def _compute_levels(self, symbol: str, price: float, atr: float) -> StrategyLevels:
        """
        Convert ATR multipliers into dollar amounts for this symbol.
        """
        m = self._atr_mult
        
        levels = StrategyLevels(
            symbol=symbol,
            price=round(price, 2),
            atr=round(atr, 2),
            atr_pct=round(atr / price * 100, 2) if price > 0 else 0,
        )
        
        # Dollar amounts = ATR × multiplier
        levels.dip_entry_min = round(atr * m["dip_entry_min"], 2)
        levels.dip_entry_max = round(atr * m["dip_entry_max"], 2)
        levels.validation    = round(atr * m["validation"], 2)
        levels.half_exit     = round(atr * m["half_exit"], 2)
        levels.full_target   = round(atr * m["full_target"], 2)
        levels.fast_drop     = round(atr * m["fast_drop"], 2)
        levels.put_flip      = round(atr * m["put_flip"], 2)
        levels.vwap_ext_pct  = m["vwap_ext"]
        levels.mode_a_entry  = round(atr * m["mode_a_entry"], 2)
        levels.mode_a_target = levels.full_target
        
        # Percentage equivalents
        if price > 0:
            levels.dip_entry_min_pct = round(levels.dip_entry_min / price * 100, 3)
            levels.dip_entry_max_pct = round(levels.dip_entry_max / price * 100, 3)
            levels.validation_pct    = round(levels.validation / price * 100, 3)
            levels.half_exit_pct     = round(levels.half_exit / price * 100, 3)
            levels.full_target_pct   = round(levels.full_target / price * 100, 3)
        
        # Option premium estimate (~1.1% of stock price for ATM 2-3 DTE)
        # This is rough — actual premium depends on IV, days, etc.
        levels.option_premium_est = round(price * 0.011, 2)
        levels.stop_dollar = round(levels.option_premium_est * m["stop_pct"], 2)
        
        return levels
    
    # ========================================================================
    # INTERNAL: Classify a single day
    # ========================================================================
    
    def _classify_day(self, day_data: Dict, levels: StrategyLevels) -> DayClassification:
        """
        Take raw cross analysis day data and classify it using V6 rules.
        """
        d = day_data
        price = levels.price
        
        day = DayClassification(
            date=d.get("date", ""),
            symbol=d.get("symbol", levels.symbol),
            open=d.get("open", 0),
            high=d.get("high", 0),
            low=d.get("low", 0),
            close=d.get("close", 0),
            crosses=d.get("crosses", 0),
            vwap=d.get("eod_vwap", 0) or 0,
            vwap_crosses=d.get("vwap_crosses", 0),
            pct_above_open=d.get("pct_above_open", 50),
            pct_above_vwap=d.get("pct_above_vwap", 50),
            high_off_open=d.get("high_off_open_pct", 0),
            low_off_open=d.get("low_off_open_pct", 0),
            close_vs_open=d.get("close_vs_open_pct", 0),
            range_pct=round((d.get("high", 0) - d.get("low", 0)) / d.get("open", 1) * 100, 2) if d.get("open", 0) > 0 else 0,
        )
        
        o = day.open
        h = day.high
        l = day.low
        c = day.close
        
        if o <= 0:
            day.regime = "SKIP"
            day.strategy = "SKIP"
            return day
        
        # Dollar moves from open
        high_off = h - o
        low_off = o - l
        close_off = c - o
        
        # ── CROSS GATE ──
        xs = day.crosses
        if xs <= self._cross_gate["directional_max"]:
            day.cross_gate = "BLOCKED"
            # Mode A: trade direction
            if c > o:
                day.regime = "BULL"
                day.strategy = "MODE_A_LONG"
            else:
                day.regime = "BEAR"
                day.strategy = "MODE_A_SHORT"
        else:
            day.cross_gate = "PASSED"
            # Choppy: dip-buy territory
            day.regime = "CHOPPY"
            
            # Did it dip enough?
            if low_off >= levels.dip_entry_min:
                day.strategy = "DIP-BUY"
            else:
                day.strategy = "NO_DIP"
                day.regime = "BULL" if c > o else "CHOPPY"
        
        # ── V6 FILTER CHECKS (for simulation) ──
        # These use the daily high/low as proxy (no intraday time data)
        
        # Validation: did high reach open + validation?
        day.hit_validation = high_off >= levels.validation
        
        # Half exit: did high reach open + half_exit?
        day.hit_half_exit = high_off >= levels.half_exit
        
        # Full target: did high reach open + full_target?
        day.hit_full_target = high_off >= levels.full_target
        
        # Fast drop: did price drop to fast_drop without bouncing?
        day.hit_fast_drop = low_off >= levels.fast_drop
        
        # Put flip: did price drop to put_flip?
        day.hit_put_flip = low_off >= levels.put_flip
        
        # ── SIMULATE P&L ──
        self._simulate_day_pnl(day, levels)
        
        return day
    
    # ========================================================================
    # INTERNAL: Simulate day P&L using V6 rules
    # ========================================================================
    
    def _simulate_day_pnl(self, day: DayClassification, levels: StrategyLevels):
        """
        Simulate the V6 strategy outcome for this day.
        P&L is in percentage of stock price (not option premium).
        """
        if day.strategy == "SKIP" or day.strategy == "NO_DIP":
            day.simulated_pnl_pct = 0
            day.simulated_outcome = "SKIP"
            return
        
        price = day.open
        if price <= 0:
            return
        
        if day.strategy == "DIP-BUY":
            # Dip-buy simulation
            if day.hit_full_target:
                # Full win: hit $1.25-equivalent target
                day.simulated_pnl_pct = round(levels.full_target / price * 100, 3)
                day.simulated_outcome = "WIN_FULL"
            elif day.hit_half_exit:
                # Partial win: hit half exit but not full
                day.simulated_pnl_pct = round(levels.half_exit / price * 100 * 0.75, 3)  # avg of half + trail
                day.simulated_outcome = "WIN_HALF"
            elif not day.hit_validation:
                # Filter 2: validation not reached = early exit
                if day.hit_fast_drop:
                    # Filter 3: fast drop = exit at fast_drop level
                    day.simulated_pnl_pct = round(-levels.fast_drop / price * 100, 3)
                    day.simulated_outcome = "LOSS_FAST_DROP"
                else:
                    # Slow grind: exit at validation failure (small loss)
                    day.simulated_pnl_pct = round(-levels.validation / price * 100 * 0.3, 3)
                    day.simulated_outcome = "LOSS_VALIDATION"
            else:
                # Hit validation but not half exit: small loss/flat
                day.simulated_pnl_pct = round(-levels.dip_entry_min / price * 100 * 0.5, 3)
                day.simulated_outcome = "LOSS_STOPPED"
        
        elif day.strategy in ("MODE_A_LONG", "MODE_A_SHORT"):
            # Mode A: directional trade
            is_long = day.strategy == "MODE_A_LONG"
            if is_long:
                move = (day.close - day.open) / day.open * 100
            else:
                move = (day.open - day.close) / day.open * 100
            
            target_pct = levels.full_target / price * 100
            stop_pct = levels.validation / price * 100  # use validation as stop distance
            
            if move >= target_pct:
                day.simulated_pnl_pct = round(target_pct, 3)
                day.simulated_outcome = "WIN_FULL"
            elif move >= target_pct * 0.5:
                day.simulated_pnl_pct = round(move * 0.75, 3)
                day.simulated_outcome = "WIN_HALF"
            elif move <= -stop_pct:
                day.simulated_pnl_pct = round(-stop_pct, 3)
                day.simulated_outcome = "LOSS_STOPPED"
            else:
                day.simulated_pnl_pct = round(move * 0.5, 3)  # partial capture
                day.simulated_outcome = "WIN_HALF" if move > 0 else "LOSS_STOPPED"
    
    # ========================================================================
    # INTERNAL: Compute regime stats
    # ========================================================================
    
    def _compute_stats(self, result: RegimeScanResult):
        """Compute aggregate stats from classified days"""
        days = result.days
        if not days:
            return
        
        result.bull_days = sum(1 for d in days if d.regime == "BULL")
        result.bear_days = sum(1 for d in days if d.regime == "BEAR")
        result.choppy_days = sum(1 for d in days if d.regime == "CHOPPY")
        
        result.dip_buy_days = sum(1 for d in days if d.strategy == "DIP-BUY")
        result.mode_a_days = sum(1 for d in days if "MODE_A" in d.strategy)
        result.no_dip_days = sum(1 for d in days if d.strategy == "NO_DIP")
        
        # Win/loss counts
        dip_buys = [d for d in days if d.strategy == "DIP-BUY"]
        mode_as = [d for d in days if "MODE_A" in d.strategy]
        
        result.dip_buy_wins = sum(1 for d in dip_buys if "WIN" in d.simulated_outcome)
        result.dip_buy_losses = sum(1 for d in dip_buys if "LOSS" in d.simulated_outcome)
        result.mode_a_wins = sum(1 for d in mode_as if "WIN" in d.simulated_outcome)
        result.mode_a_losses = sum(1 for d in mode_as if "LOSS" in d.simulated_outcome)
        
        # Overall stats
        traded = [d for d in days if d.simulated_outcome != "SKIP"]
        wins = [d for d in traded if "WIN" in d.simulated_outcome]
        losses = [d for d in traded if "LOSS" in d.simulated_outcome]
        
        result.total_pnl_pct = round(sum(d.simulated_pnl_pct for d in traded), 2)
        result.win_rate = round(len(wins) / len(traded) * 100, 1) if traded else 0
        
        total_wins = sum(d.simulated_pnl_pct for d in wins)
        total_losses = abs(sum(d.simulated_pnl_pct for d in losses))
        result.profit_factor = round(total_wins / total_losses, 2) if total_losses > 0 else (999.99 if total_wins > 0 else 0)
    
    # ========================================================================
    # INTERNAL: Rolling windows
    # ========================================================================
    
    def _compute_windows(self, result: RegimeScanResult):
        """Compute 5D/10D/20D rolling regime counts"""
        days_sorted = sorted(result.days, key=lambda d: d.date, reverse=True)
        
        windows = {}
        for w in [5, 10, 20]:
            window_days = days_sorted[:min(w, len(days_sorted))]
            windows[f"{w}D"] = {
                "bull": sum(1 for d in window_days if d.regime == "BULL"),
                "bear": sum(1 for d in window_days if d.regime == "BEAR"),
                "choppy": sum(1 for d in window_days if d.regime == "CHOPPY"),
                "total": len(window_days),
            }
        
        result.windows = windows
    
    # ========================================================================
    # INTERNAL: Streak detection
    # ========================================================================
    
    def _compute_streak(self, result: RegimeScanResult):
        """Find current regime streak"""
        days_sorted = sorted(result.days, key=lambda d: d.date, reverse=True)
        if not days_sorted:
            return
        
        streak_type = days_sorted[0].regime
        streak_count = 1
        for d in days_sorted[1:]:
            if d.regime == streak_type:
                streak_count += 1
            else:
                break
        
        result.streak_type = streak_type
        result.streak_count = streak_count
    
    # ========================================================================
    # INTERNAL: Monthly breakdown
    # ========================================================================
    
    def _compute_monthly(self, result: RegimeScanResult):
        """Compute current month regime counts"""
        now = datetime.now()
        current_month = now.month
        current_year = now.year
        
        month_days = [
            d for d in result.days
            if self._parse_month_year(d.date) == (current_month, current_year)
        ]
        
        result.current_month_counts = {
            "month": now.strftime("%B %Y"),
            "sessions": len(month_days),
            "bull": sum(1 for d in month_days if d.regime == "BULL"),
            "bear": sum(1 for d in month_days if d.regime == "BEAR"),
            "choppy": sum(1 for d in month_days if d.regime == "CHOPPY"),
            "dip_buy": sum(1 for d in month_days if d.strategy == "DIP-BUY"),
            "mode_a": sum(1 for d in month_days if "MODE_A" in d.strategy),
        }
    
    @staticmethod
    def _parse_month_year(date_str: str) -> tuple:
        """Extract (month, year) from date string"""
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return (dt.month, dt.year)
        except (ValueError, IndexError):
            return (0, 0)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_regime_scan(
    symbols: List[str],
    days_back: int = 30,
) -> Dict:
    """
    One-liner: scan multiple symbols, return comparison + individual results.
    
    Usage:
        from regime_scanner import quick_regime_scan
        data = quick_regime_scan(["AAPL", "NVDA", "TSLA", "IWM"], days_back=30)
        print(data["comparison"])
        print(data["results"]["AAPL"])
    """
    scanner = RegimeScanner()
    results = scanner.scan_watchlist(symbols, days_back)
    comparison = scanner.compare_watchlist(results)
    
    return {
        "comparison": comparison,
        "results": {sym: r.to_dict() for sym, r in results.items()},
    }


def get_levels(symbol: str) -> Dict:
    """
    Quick lookup: get strategy levels for a symbol.
    
    Usage:
        from regime_scanner import get_levels
        levels = get_levels("NVDA")
        print(f"Dip entry: -${levels['levels']['dip_entry_min']} to -${levels['levels']['dip_entry_max']}")
    """
    scanner = RegimeScanner()
    return scanner.get_strategy_levels(symbol)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python regime_scanner.py SYMBOL [SYMBOL2 ...] [--days N]")
        print("Example: python regime_scanner.py AAPL NVDA TSLA --days 30")
        print("         python regime_scanner.py IWM --levels")
        sys.exit(1)
    
    symbols = []
    days = 30
    levels_only = False
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--days" and i + 1 < len(sys.argv):
            days = int(sys.argv[i + 1])
            i += 2
        elif arg == "--levels":
            levels_only = True
            i += 1
        else:
            symbols.append(arg.upper())
            i += 1
    
    if not symbols:
        symbols = ["IWM"]
    
    scanner = RegimeScanner()
    
    if levels_only:
        for sym in symbols:
            levels = scanner.get_strategy_levels(sym)
            print(f"\n{'='*60}")
            print(f"  {sym} Strategy Levels")
            print(f"{'='*60}")
            print(json.dumps(levels, indent=2))
    else:
        results = scanner.scan_watchlist(symbols, days)
        comparison = scanner.compare_watchlist(results)
        
        # Print comparison table
        print(f"\n{'='*100}")
        print(f"  REGIME SCANNER — {days}D Lookback")
        print(f"{'='*100}")
        print(f"{'Symbol':<8} {'Price':>8} {'ATR':>6} {'Bull':>5} {'Bear':>5} {'Chop':>5} {'WR%':>6} {'PF':>6} {'P&L%':>7} {'Score':>6}")
        print("-" * 100)
        for row in comparison["comparison"]:
            print(f"{row['symbol']:<8} ${row['price']:>7.2f} ${row['atr']:>5.2f} "
                  f"{row['bull']:>5} {row['bear']:>5} {row['choppy']:>5} "
                  f"{row['win_rate']:>5.1f}% {row['profit_factor']:>5.1f}x "
                  f"{row['total_pnl_pct']:>+6.2f}% {row['dip_buy_score']:>5.1f}")
        
        print(f"\nBest dip-buy candidate: {comparison['best_dip_buy']}")
        print(f"Highest win rate: {comparison['highest_win_rate']}")
        print(f"Most Mode A days: {comparison['best_mode_a']}")
