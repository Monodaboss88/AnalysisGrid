"""
Live Auction Scanner - Real Data Integration
=============================================
Connects the MTF Auction Scanner to real market data via yfinance
with watchlist scanning and alert capabilities.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")

from mtf_auction_scanner import (
    MTFAuctionScanner, ScanResult, SignalState, Timeframe
)


# =============================================================================
# DATA FETCHER
# =============================================================================

class MarketDataFetcher:
    """
    Fetches market data from various sources
    
    For Brokers:
    -----------
    This module pulls real price/volume data for analysis.
    - Uses Yahoo Finance for free delayed data
    - Supports any ticker: stocks, ETFs, futures, crypto
    - Fetches enough history for multi-timeframe analysis
    
    For Programmers:
    ---------------
    Wraps yfinance with caching and error handling.
    Returns standardized OHLCV DataFrames with DatetimeIndex.
    """
    
    def __init__(self, cache_minutes: int = 5):
        self.cache_minutes = cache_minutes
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
    
    def fetch(self, 
              symbol: str, 
              days: int = 15, 
              interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol
        
        Args:
            symbol: Ticker symbol (e.g., "SPY", "AAPL", "BTC-USD")
            days: Number of days of history (max 60 for 5m data)
            interval: Data interval ("1m", "5m", "15m", "30m", "1h")
        
        Returns:
            DataFrame with columns [open, high, low, close, volume]
        """
        if not YFINANCE_AVAILABLE:
            print(f"Cannot fetch {symbol}: yfinance not installed")
            return None
        
        # Check cache
        cache_key = f"{symbol}_{interval}_{days}"
        if cache_key in self._cache:
            cached_time, cached_df = self._cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_minutes):
                return cached_df.copy()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # yfinance limits: 5m data max 60 days, 1m data max 7 days
            if interval in ["1m", "2m", "5m"]:
                days = min(days, 60)
            
            df = ticker.history(period=f"{days}d", interval=interval)
            
            if df.empty:
                print(f"No data returned for {symbol}")
                return None
            
            # Standardize columns
            df.columns = df.columns.str.lower()
            
            # Keep only OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in required_cols if c in df.columns]]
            
            # Remove any rows with NaN
            df = df.dropna()
            
            # Cache
            self._cache[cache_key] = (datetime.now(), df.copy())
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None


# =============================================================================
# WATCHLIST SCANNER
# =============================================================================

@dataclass
class WatchlistItem:
    """Configuration for a single watchlist symbol"""
    symbol: str
    name: str = ""
    category: str = "General"
    enabled: bool = True


class WatchlistScanner:
    """
    Scans multiple symbols and prioritizes setups
    
    For Brokers:
    -----------
    Run this to scan your entire watchlist at once.
    - Returns ranked list of setups
    - Highlights actionable trades
    - Filters out yellow/neutral states if desired
    
    For Programmers:
    ---------------
    Iterates through watchlist, fetches data, runs scanner,
    collects results and sorts by signal strength.
    """
    
    # Default watchlists
    DEFAULT_INDICES = [
        WatchlistItem("SPY", "S&P 500 ETF", "Index"),
        WatchlistItem("QQQ", "Nasdaq 100 ETF", "Index"),
        WatchlistItem("IWM", "Russell 2000 ETF", "Index"),
        WatchlistItem("DIA", "Dow Jones ETF", "Index"),
    ]
    
    DEFAULT_SECTORS = [
        WatchlistItem("XLF", "Financials", "Sector"),
        WatchlistItem("XLK", "Technology", "Sector"),
        WatchlistItem("XLE", "Energy", "Sector"),
        WatchlistItem("XLV", "Healthcare", "Sector"),
        WatchlistItem("XLI", "Industrials", "Sector"),
        WatchlistItem("XLY", "Consumer Disc", "Sector"),
        WatchlistItem("XLP", "Consumer Staples", "Sector"),
        WatchlistItem("XLU", "Utilities", "Sector"),
        WatchlistItem("XLB", "Materials", "Sector"),
        WatchlistItem("XLRE", "Real Estate", "Sector"),
    ]
    
    DEFAULT_MEGA_CAPS = [
        WatchlistItem("AAPL", "Apple", "Mega Cap"),
        WatchlistItem("MSFT", "Microsoft", "Mega Cap"),
        WatchlistItem("GOOGL", "Google", "Mega Cap"),
        WatchlistItem("AMZN", "Amazon", "Mega Cap"),
        WatchlistItem("NVDA", "Nvidia", "Mega Cap"),
        WatchlistItem("META", "Meta", "Mega Cap"),
        WatchlistItem("TSLA", "Tesla", "Mega Cap"),
    ]
    
    def __init__(self, 
                 watchlist: Optional[List[WatchlistItem]] = None,
                 use_defaults: bool = True):
        """
        Initialize scanner with watchlist
        
        Args:
            watchlist: Custom watchlist items
            use_defaults: Include default indices/sectors
        """
        self.watchlist: List[WatchlistItem] = []
        
        if use_defaults:
            self.watchlist.extend(self.DEFAULT_INDICES)
            self.watchlist.extend(self.DEFAULT_SECTORS)
        
        if watchlist:
            self.watchlist.extend(watchlist)
        
        self.fetcher = MarketDataFetcher()
        self.scanner = MTFAuctionScanner()
    
    def add_symbol(self, symbol: str, name: str = "", category: str = "Custom"):
        """Add a symbol to the watchlist"""
        self.watchlist.append(WatchlistItem(symbol, name, category))
    
    def add_symbols(self, symbols: List[str], category: str = "Custom"):
        """Add multiple symbols"""
        for sym in symbols:
            self.add_symbol(sym, category=category)
    
    def scan_all(self, 
                 timeframes: Optional[List[Timeframe]] = None,
                 show_progress: bool = True) -> List[Tuple[WatchlistItem, ScanResult]]:
        """
        Scan all symbols in watchlist
        
        Returns:
            List of (WatchlistItem, ScanResult) tuples, sorted by signal strength
        """
        results = []
        total = len([w for w in self.watchlist if w.enabled])
        
        for i, item in enumerate(self.watchlist):
            if not item.enabled:
                continue
            
            if show_progress:
                print(f"Scanning {item.symbol} ({i+1}/{total})...", end="\r")
            
            df = self.fetcher.fetch(item.symbol, days=15, interval="5m")
            
            if df is not None and len(df) > 100:
                result = self.scanner.scan(df, symbol=item.symbol, timeframes=timeframes)
                results.append((item, result))
        
        if show_progress:
            print(" " * 50, end="\r")  # Clear progress line
        
        # Sort by actionability and confidence
        results.sort(key=lambda x: self._score_result(x[1]), reverse=True)
        
        return results
    
    def _score_result(self, result: ScanResult) -> float:
        """Score a result for sorting (higher = more interesting)"""
        score = 0
        
        # Actionable setups get highest priority
        if result.actionable:
            score += 1000
        
        # Signal type
        if result.dominant_signal == SignalState.LONG_SETUP:
            score += 500
        elif result.dominant_signal == SignalState.SHORT_SETUP:
            score += 500
        elif result.dominant_signal == SignalState.YELLOW:
            score += 100
        
        # Confluence
        score += result.confluence_score * 2
        
        # Scenario probability extremes
        if result.high_scenario_prob > 0.7 or result.low_scenario_prob > 0.7:
            score += 100
        
        return score
    
    def scan_for_setups(self, 
                        direction: Optional[str] = None,
                        min_confluence: float = 50) -> List[Tuple[WatchlistItem, ScanResult]]:
        """
        Scan and filter for actionable setups only
        
        Args:
            direction: "LONG", "SHORT", or None for both
            min_confluence: Minimum confluence score
        
        Returns:
            Filtered list of actionable setups
        """
        all_results = self.scan_all(show_progress=True)
        
        filtered = []
        for item, result in all_results:
            if not result.actionable:
                continue
            
            if result.confluence_score < min_confluence:
                continue
            
            if direction == "LONG" and result.dominant_signal != SignalState.LONG_SETUP:
                continue
            
            if direction == "SHORT" and result.dominant_signal != SignalState.SHORT_SETUP:
                continue
            
            filtered.append((item, result))
        
        return filtered
    
    def print_scan_report(self, results: List[Tuple[WatchlistItem, ScanResult]]) -> str:
        """Generate formatted scan report"""
        lines = []
        lines.append("=" * 90)
        lines.append(f"WATCHLIST SCAN REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 90)
        
        # Group by signal type
        actionable = [(i, r) for i, r in results if r.actionable]
        yellows = [(i, r) for i, r in results if r.dominant_signal == SignalState.YELLOW]
        neutrals = [(i, r) for i, r in results if r.dominant_signal in [SignalState.NEUTRAL, SignalState.NO_DATA]]
        
        # Actionable setups
        if actionable:
            lines.append("\n🎯 ACTIONABLE SETUPS:")
            lines.append("-" * 90)
            lines.append(f"{'Symbol':<8} {'Name':<20} {'Signal':<15} {'Confluence':>10} {'High%':>8} {'Low%':>8}")
            lines.append("-" * 90)
            
            for item, result in actionable:
                signal_str = f"{result.dominant_signal.emoji} {result.dominant_signal.value}"
                lines.append(
                    f"{item.symbol:<8} {item.name[:20]:<20} {signal_str:<15} "
                    f"{result.confluence_score:>9.0f}% "
                    f"{result.high_scenario_prob:>7.0%} {result.low_scenario_prob:>7.0%}"
                )
        
        # Yellow/Watch
        if yellows:
            lines.append("\n🟡 YELLOW - WATCH FOR CLARITY:")
            lines.append("-" * 90)
            
            for item, result in yellows[:10]:  # Top 10
                avg_bull = np.mean([a.bull_score for a in result.timeframe_analyses.values()])
                avg_bear = np.mean([a.bear_score for a in result.timeframe_analyses.values()])
                lines.append(
                    f"{item.symbol:<8} {item.name[:20]:<20} "
                    f"Bull: {avg_bull:>5.1f} | Bear: {avg_bear:>5.1f} | "
                    f"Leaning: {'LONG' if avg_bull > avg_bear else 'SHORT' if avg_bear > avg_bull else 'FLAT'}"
                )
        
        # Summary stats
        lines.append("\n" + "=" * 90)
        lines.append("SUMMARY:")
        lines.append(f"  Total scanned: {len(results)}")
        lines.append(f"  Actionable setups: {len(actionable)}")
        lines.append(f"  Yellow/Watch: {len(yellows)}")
        lines.append(f"  Neutral/No setup: {len(neutrals)}")
        lines.append("=" * 90)
        
        return "\n".join(lines)


# =============================================================================
# SINGLE SYMBOL DEEP DIVE
# =============================================================================

class SymbolAnalyzer:
    """
    Deep dive analysis on a single symbol
    
    For Brokers:
    -----------
    Use this for detailed analysis before entering a trade.
    Shows all timeframe details, key levels, and trade plan.
    
    For Programmers:
    ---------------
    Wraps scanner with additional context and level calculations.
    """
    
    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self.scanner = MTFAuctionScanner()
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        """
        Run deep analysis on a single symbol
        
        Returns:
            Dictionary with complete analysis
        """
        df = self.fetcher.fetch(symbol, days=15, interval="5m")
        
        if df is None or len(df) < 100:
            return None
        
        result = self.scanner.scan(df, symbol=symbol)
        
        # Calculate additional levels
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)
        
        # Key levels across timeframes
        levels = {}
        for tf, analysis in result.timeframe_analyses.items():
            vp = analysis.volume_profile
            levels[tf.label] = {
                'poc': vp.poc,
                'vah': vp.vah,
                'val': vp.val,
                'value_width': vp.value_width
            }
        
        # Trade plan if actionable
        trade_plan = None
        if result.actionable:
            trade_plan = self._generate_trade_plan(result, current_price, atr, levels)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'atr': atr,
            'scan_result': result,
            'levels': levels,
            'trade_plan': trade_plan
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    def _generate_trade_plan(self, 
                             result: ScanResult, 
                             price: float,
                             atr: float,
                             levels: Dict) -> Dict:
        """Generate a trade plan for actionable setups"""
        
        # Use 4hr value area for swing trade levels
        h4_levels = levels.get('4hour', levels.get('2hour', levels.get('1hour', {})))
        
        if result.dominant_signal == SignalState.LONG_SETUP:
            entry_zone = f"{h4_levels.get('poc', price):.2f} - {price:.2f}"
            stop = h4_levels.get('val', price - atr * 2) - atr * 0.5
            target1 = h4_levels.get('vah', price + atr * 2) + atr
            target2 = price + atr * 4
            direction = "LONG"
            
        else:  # SHORT_SETUP
            entry_zone = f"{price:.2f} - {h4_levels.get('poc', price):.2f}"
            stop = h4_levels.get('vah', price + atr * 2) + atr * 0.5
            target1 = h4_levels.get('val', price - atr * 2) - atr
            target2 = price - atr * 4
            direction = "SHORT"
        
        risk = abs(price - stop)
        reward1 = abs(target1 - price)
        rr1 = reward1 / risk if risk > 0 else 0
        
        return {
            'direction': direction,
            'entry_zone': entry_zone,
            'stop_loss': stop,
            'target_1': target1,
            'target_2': target2,
            'risk_per_share': risk,
            'reward_to_risk_t1': rr1,
            'swing_days': "3-5",
            'notes': [
                f"Stop below/above value area ({'VAL' if direction == 'LONG' else 'VAH'})",
                f"Target 1 at opposite value area edge",
                f"Target 2 at 2x value area extension",
                f"ATR(14): {atr:.2f}"
            ]
        }
    
    def print_deep_analysis(self, analysis: Dict) -> str:
        """Format deep analysis for display"""
        if analysis is None:
            return "No analysis available"
        
        lines = []
        result = analysis['scan_result']
        
        lines.append("=" * 80)
        lines.append(f"DEEP ANALYSIS: {analysis['symbol']}")
        lines.append(f"Current Price: ${analysis['current_price']:.2f}")
        lines.append(f"ATR(14): ${analysis['atr']:.2f}")
        lines.append("=" * 80)
        
        # Main signal
        lines.append(f"\n{result.dominant_signal.emoji} SIGNAL: {result.dominant_signal.value}")
        lines.append(f"Confluence: {result.confluence_score:.0f}%")
        lines.append(f"Actionable: {'YES ✅' if result.actionable else 'NO'}")
        
        # Scenario probabilities
        lines.append(f"\nSCENARIO PROBABILITIES:")
        lines.append(f"  HIGH scenario: {result.high_scenario_prob:.0%}")
        lines.append(f"  LOW scenario:  {result.low_scenario_prob:.0%}")
        lines.append(f"  Neutral/Chop:  {result.neutral_prob:.0%}")
        
        # Key levels
        lines.append(f"\nKEY LEVELS BY TIMEFRAME:")
        lines.append("-" * 80)
        lines.append(f"{'Timeframe':<12} {'VAH':>12} {'POC':>12} {'VAL':>12} {'Width':>12}")
        lines.append("-" * 80)
        
        for tf_label, lvls in analysis['levels'].items():
            lines.append(
                f"{tf_label:<12} "
                f"${lvls['vah']:>11.2f} "
                f"${lvls['poc']:>11.2f} "
                f"${lvls['val']:>11.2f} "
                f"${lvls['value_width']:>11.2f}"
            )
        
        # Timeframe breakdown
        lines.append(f"\nTIMEFRAME SIGNALS:")
        lines.append("-" * 80)
        
        for tf, a in result.timeframe_analyses.items():
            lines.append(
                f"{tf.label:>8}: {a.signal.emoji} {a.signal.value:<12} | "
                f"Bull: {a.bull_score:>5.1f} | Bear: {a.bear_score:>5.1f} | "
                f"RSI: {a.rsi.value:>5.1f} | Flow: {a.flow.flow_imbalance:>+.2f}"
            )
        
        # Trade plan
        if analysis['trade_plan']:
            tp = analysis['trade_plan']
            lines.append(f"\n{'='*80}")
            lines.append(f"📋 TRADE PLAN - {tp['direction']}")
            lines.append(f"{'='*80}")
            lines.append(f"  Entry Zone:  {tp['entry_zone']}")
            lines.append(f"  Stop Loss:   ${tp['stop_loss']:.2f}")
            lines.append(f"  Target 1:    ${tp['target_1']:.2f} ({tp['reward_to_risk_t1']:.1f}R)")
            lines.append(f"  Target 2:    ${tp['target_2']:.2f}")
            lines.append(f"  Swing Days:  {tp['swing_days']}")
            lines.append(f"\n  Notes:")
            for note in tp['notes']:
                lines.append(f"    • {note}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_watchlist_scan():
    """Run a full watchlist scan"""
    print("\n🔍 Starting Watchlist Scan...\n")
    
    scanner = WatchlistScanner(use_defaults=True)
    
    # Add any custom symbols
    # scanner.add_symbols(["COIN", "MARA", "RIOT"], category="Crypto-Related")
    
    results = scanner.scan_all()
    report = scanner.print_scan_report(results)
    print(report)
    
    return results


def analyze_symbol(symbol: str):
    """Run deep analysis on a single symbol"""
    print(f"\n🔬 Analyzing {symbol}...\n")
    
    analyzer = SymbolAnalyzer()
    analysis = analyzer.analyze(symbol)
    
    if analysis:
        report = analyzer.print_deep_analysis(analysis)
        print(report)
        return analysis
    else:
        print(f"Could not analyze {symbol}")
        return None


def scan_for_longs():
    """Scan specifically for long setups"""
    print("\n🟢 Scanning for LONG setups...\n")
    
    scanner = WatchlistScanner(use_defaults=True)
    setups = scanner.scan_for_setups(direction="LONG", min_confluence=50)
    
    if setups:
        print(f"Found {len(setups)} long setups:\n")
        for item, result in setups:
            print(f"  {result.dominant_signal.emoji} {item.symbol:<8} - {result.confluence_score:.0f}% confluence")
    else:
        print("No long setups found meeting criteria")
    
    return setups


def scan_for_shorts():
    """Scan specifically for short setups"""
    print("\n🔴 Scanning for SHORT setups...\n")
    
    scanner = WatchlistScanner(use_defaults=True)
    setups = scanner.scan_for_setups(direction="SHORT", min_confluence=50)
    
    if setups:
        print(f"Found {len(setups)} short setups:\n")
        for item, result in setups:
            print(f"  {result.dominant_signal.emoji} {item.symbol:<8} - {result.confluence_score:.0f}% confluence")
    else:
        print("No short setups found meeting criteria")
    
    return setups


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MTF AUCTION SCANNER - LIVE DATA INTEGRATION")
    print("=" * 80)
    
    if not YFINANCE_AVAILABLE:
        print("\n⚠️  yfinance not installed. Install with: pip install yfinance")
        print("Running with demo data instead...\n")
        
        # Fall back to demo
        from mtf_auction_scanner import generate_demo_data
        
        df = generate_demo_data(days=15, interval_minutes=5)
        scanner = MTFAuctionScanner()
        result = scanner.scan(df, symbol="DEMO")
        print(scanner.print_report(result))
        
    else:
        # Run real scan on SPY as example
        print("\n📊 Demo: Analyzing SPY...\n")
        analyze_symbol("SPY")
        
        print("\n" + "=" * 80)
        print("AVAILABLE FUNCTIONS:")
        print("=" * 80)
        print("""
    run_watchlist_scan()   - Scan entire watchlist
    analyze_symbol("AAPL") - Deep dive on single symbol
    scan_for_longs()       - Find long setups only
    scan_for_shorts()      - Find short setups only
    
    Usage in Python:
        from live_scanner import *
        results = run_watchlist_scan()
        analysis = analyze_symbol("NVDA")
        """)
