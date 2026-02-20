"""
Live Scanner with Watchlist Integration
=======================================
Combines the MTF Scanner, Overnight Model, and Watchlist Manager
into a complete scanning solution.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json

# Import our modules
from watchlist_manager import WatchlistManager, WatchlistSymbol, quick_scan_list
from integrated_scanner import IntegratedScanner, IntegratedAnalysis
from mtf_auction_scanner import MTFAuctionScanner, ScanResult, SignalState

# Import polygon_data for market data
try:
    from polygon_data import get_bars
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("⚠️ polygon_data not available - using demo data")


# =============================================================================
# DATA FETCHER
# =============================================================================

class MarketDataFetcher:
    """
    Fetches market data with caching
    """
    
    def __init__(self, cache_minutes: int = 5):
        self.cache: Dict[str, tuple] = {}  # symbol -> (data, timestamp)
        self.cache_minutes = cache_minutes
    
    def fetch(self, 
              symbol: str, 
              days: int = 15, 
              interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Fetch market data for a symbol
        
        Args:
            symbol: Ticker symbol
            days: Days of history
            interval: Bar interval
        
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()
        
        # Check cache
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            age = (datetime.now() - timestamp).total_seconds() / 60
            if age < self.cache_minutes:
                return data
        
        # Fetch new data
        if POLYGON_AVAILABLE:
            df = self._fetch_polygon(symbol, days, interval)
        else:
            df = self._generate_demo_data(symbol, days)
        
        if df is not None and len(df) > 0:
            self.cache[symbol] = (df, datetime.now())
        
        return df
    
    def _fetch_polygon(self, 
                        symbol: str, 
                        days: int, 
                        interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Polygon"""
        try:
            df = get_bars(symbol, period=f"{days}d", interval=interval)
            
            if df.empty:
                return None
            
            df.columns = df.columns.str.lower()
            required = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in required if c in df.columns]]
            df = df.dropna()
            
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def _generate_demo_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate demo data when Polygon unavailable"""
        np.random.seed(hash(symbol) % 2**32)
        
        periods = days * 24 * 12  # 5-min bars
        base_price = 50 + (hash(symbol) % 500)
        
        data = []
        timestamp = datetime.now() - timedelta(days=days)
        price = base_price
        
        for i in range(periods):
            trend = 0.0001 * np.sin(i / 200)
            noise = np.random.randn() * 0.002
            
            open_p = price
            close_p = price * (1 + trend + noise)
            high_p = max(open_p, close_p) * (1 + abs(np.random.randn()) * 0.001)
            low_p = min(open_p, close_p) * (1 - abs(np.random.randn()) * 0.001)
            volume = int(np.random.exponential(100000))
            
            data.append({
                'timestamp': timestamp,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': close_p,
                'volume': volume
            })
            
            price = close_p
            timestamp += timedelta(minutes=5)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()


# =============================================================================
# SCAN RESULT FORMATTER
# =============================================================================

@dataclass
class FormattedScanResult:
    """Formatted scan result for display/API"""
    symbol: str
    name: str
    category: str
    sector: str
    
    # Signal
    signal: str
    signal_emoji: str
    combined_bias: str
    confidence: float
    actionable: bool
    
    # Scores
    bull_score: float
    bear_score: float
    confluence: float
    
    # Probabilities
    high_prob: float
    low_prob: float
    chop_prob: float
    
    # Overnight
    gap_type: str = ""
    gap_pct: float = 0.0
    gap_fill_prob: str = ""
    overnight_bias: str = ""
    
    # Key levels
    key_levels: Dict[str, float] = field(default_factory=dict)
    
    # Trade plan
    trade_plan: Optional[Dict] = None
    
    # Metadata
    scan_time: str = ""
    notes: List[str] = field(default_factory=list)
    
    # Index membership
    in_sp500: bool = False
    in_nasdaq100: bool = False
    in_dow30: bool = False


# =============================================================================
# LIVE SCANNER
# =============================================================================

class LiveScanner:
    """
    Complete live scanning solution
    
    For Brokers:
    -----------
    - Scan any watchlist with one command
    - Pre-built scans: "indices", "sectors", "dow", "nasdaq", "sp500"
    - Filter results by signal type
    - Rank by actionability and confluence
    - Full overnight/gap context
    
    For Programmers:
    ---------------
    Orchestrates WatchlistManager, IntegratedScanner, and MarketDataFetcher.
    """
    
    def __init__(self):
        self.watchlist_manager = WatchlistManager()
        self.scanner = IntegratedScanner()
        self.data_fetcher = MarketDataFetcher()
        self.last_scan_results: List[FormattedScanResult] = []
        self.last_scan_time: Optional[datetime] = None
    
    # =========================================================================
    # WATCHLIST MANAGEMENT (delegated)
    # =========================================================================
    
    def get_watchlist_names(self) -> List[str]:
        """Get all watchlist names"""
        return self.watchlist_manager.get_watchlist_names()
    
    def get_watchlist(self, name: str):
        """Get a specific watchlist"""
        return self.watchlist_manager.get_watchlist(name)
    
    def search_symbols(self, query: str, limit: int = 20) -> List[WatchlistSymbol]:
        """Search for symbols"""
        return self.watchlist_manager.search(query, limit=limit)
    
    def add_to_watchlist(self, 
                         symbol: str, 
                         watchlist: str = "Custom",
                         name: str = "",
                         sector: str = "") -> WatchlistSymbol:
        """Add symbol to a watchlist"""
        return self.watchlist_manager.add_symbol(watchlist, symbol, name=name, sector=sector)
    
    def remove_from_watchlist(self, symbol: str, watchlist: str = "Custom") -> bool:
        """Remove symbol from watchlist"""
        return self.watchlist_manager.remove_symbol(watchlist, symbol)
    
    def toggle_favorite(self, symbol: str) -> bool:
        """Toggle favorite status"""
        return self.watchlist_manager.toggle_favorite(symbol)
    
    def create_watchlist(self, name: str, description: str = ""):
        """Create a new custom watchlist"""
        return self.watchlist_manager.create_watchlist(name, description)
    
    def get_symbols_by_index(self, index: str) -> List[str]:
        """Get symbols in an index (SPY, QQQ, DIA)"""
        symbols = self.watchlist_manager.get_by_index(index)
        return [s.symbol for s in symbols]
    
    def get_symbols_by_sector(self, sector: str) -> List[str]:
        """Get symbols in a sector"""
        symbols = self.watchlist_manager.get_by_sector(sector)
        return [s.symbol for s in symbols]
    
    def get_all_sectors(self) -> List[str]:
        """Get list of all sectors"""
        return self.watchlist_manager.get_all_sectors()
    
    # =========================================================================
    # SCANNING
    # =========================================================================
    
    def scan_symbol(self, symbol: str) -> Optional[FormattedScanResult]:
        """
        Scan a single symbol
        
        Returns:
            FormattedScanResult or None if data unavailable
        """
        df = self.data_fetcher.fetch(symbol)
        
        if df is None or len(df) < 100:
            return None
        
        # Get symbol info from watchlist
        sym_info = self.watchlist_manager.all_symbols.get(symbol.upper())
        
        # Run integrated analysis
        analysis = self.scanner.analyze(df, symbol=symbol.upper())
        
        # Format result
        return self._format_result(analysis, sym_info)
    
    def scan_watchlist(self, 
                       watchlist_name: str,
                       enabled_only: bool = True) -> List[FormattedScanResult]:
        """
        Scan all symbols in a watchlist
        
        Args:
            watchlist_name: Name of watchlist to scan
            enabled_only: Only scan enabled symbols
        
        Returns:
            List of FormattedScanResult
        """
        if enabled_only:
            symbols = self.watchlist_manager.get_enabled_symbols(watchlist_name)
        else:
            wl = self.watchlist_manager.get_watchlist(watchlist_name)
            symbols = [s.symbol for s in wl.symbols] if wl else []
        
        return self._scan_symbols(symbols)
    
    def scan_quick(self, list_type: str) -> List[FormattedScanResult]:
        """
        Quick scan using predefined list
        
        Args:
            list_type: One of "indices", "sectors", "mega", "dow", "nasdaq", "sp500", "all"
        
        Returns:
            List of FormattedScanResult
        """
        symbols = quick_scan_list(self.watchlist_manager, list_type)
        return self._scan_symbols(symbols)
    
    def scan_custom(self, symbols: List[str]) -> List[FormattedScanResult]:
        """
        Scan a custom list of symbols
        
        Args:
            symbols: List of ticker symbols
        
        Returns:
            List of FormattedScanResult
        """
        return self._scan_symbols([s.upper() for s in symbols])
    
    def scan_index_components(self, index: str) -> List[FormattedScanResult]:
        """
        Scan all components of an index
        
        Args:
            index: "SPY", "QQQ", "DIA"
        
        Returns:
            List of FormattedScanResult
        """
        symbols = self.get_symbols_by_index(index)
        return self._scan_symbols(symbols)
    
    def scan_sector(self, sector: str) -> List[FormattedScanResult]:
        """
        Scan all symbols in a sector
        
        Args:
            sector: Sector name (e.g., "Technology", "Healthcare")
        
        Returns:
            List of FormattedScanResult
        """
        symbols = self.get_symbols_by_sector(sector)
        return self._scan_symbols(symbols)
    
    def _scan_symbols(self, symbols: List[str]) -> List[FormattedScanResult]:
        """Internal method to scan a list of symbols"""
        results = []
        
        for symbol in symbols:
            result = self.scan_symbol(symbol)
            if result:
                results.append(result)
        
        # Sort by actionability and confidence
        results.sort(key=lambda x: (
            x.actionable,
            x.signal in ['LONG_SETUP', 'SHORT_SETUP'],
            x.confidence
        ), reverse=True)
        
        self.last_scan_results = results
        self.last_scan_time = datetime.now()
        
        return results
    
    def _format_result(self, 
                       analysis: IntegratedAnalysis,
                       sym_info: Optional[WatchlistSymbol]) -> FormattedScanResult:
        """Format IntegratedAnalysis into FormattedScanResult"""
        
        mtf = analysis.mtf_scan
        overnight = analysis.overnight
        
        # Get primary timeframe for scores
        primary_tf = list(mtf.timeframe_analyses.values())[0] if mtf.timeframe_analyses else None
        
        result = FormattedScanResult(
            symbol=analysis.symbol,
            name=sym_info.name if sym_info else "",
            category=sym_info.category if sym_info else "",
            sector=sym_info.sector if sym_info else "",
            
            signal=mtf.dominant_signal.value,
            signal_emoji=mtf.dominant_signal.emoji,
            combined_bias=analysis.combined_bias,
            confidence=analysis.combined_confidence,
            actionable=analysis.trade_plan is not None,
            
            bull_score=primary_tf.bull_score if primary_tf else 0,
            bear_score=primary_tf.bear_score if primary_tf else 0,
            confluence=mtf.confluence_score,
            
            high_prob=analysis.high_scenario_prob * 100,
            low_prob=analysis.low_scenario_prob * 100,
            chop_prob=analysis.chop_scenario_prob * 100,
            
            key_levels=analysis.key_levels,
            trade_plan=analysis.trade_plan,
            
            scan_time=analysis.analysis_time.isoformat(),
            notes=analysis.notes,
            
            in_sp500=sym_info.in_sp500 if sym_info else False,
            in_nasdaq100=sym_info.in_nasdaq100 if sym_info else False,
            in_dow30=sym_info.in_dow30 if sym_info else False
        )
        
        # Add overnight info
        if overnight:
            result.gap_type = overnight.gap.gap_type.value
            result.gap_pct = overnight.gap.gap_pct
            result.gap_fill_prob = overnight.gap.gap_fill_probability.value
            result.overnight_bias = overnight.bias.value
        
        return result
    
    # =========================================================================
    # FILTERING AND SORTING
    # =========================================================================
    
    def filter_results(self,
                       results: List[FormattedScanResult] = None,
                       signal: str = None,
                       min_confidence: float = None,
                       actionable_only: bool = False,
                       sector: str = None,
                       in_index: str = None) -> List[FormattedScanResult]:
        """
        Filter scan results
        
        Args:
            results: Results to filter (default: last scan results)
            signal: Filter by signal type ("LONG_SETUP", "SHORT_SETUP", "YELLOW", etc.)
            min_confidence: Minimum confidence threshold
            actionable_only: Only show actionable setups
            sector: Filter by sector
            in_index: Filter by index membership ("SPY", "QQQ", "DIA")
        
        Returns:
            Filtered list
        """
        if results is None:
            results = self.last_scan_results
        
        filtered = []
        
        for r in results:
            if signal and r.signal != signal:
                continue
            if min_confidence and r.confidence < min_confidence:
                continue
            if actionable_only and not r.actionable:
                continue
            if sector and r.sector != sector:
                continue
            if in_index:
                idx = in_index.upper()
                if idx in ["SPY", "SPX", "S&P500"] and not r.in_sp500:
                    continue
                if idx in ["QQQ", "NDX", "NASDAQ"] and not r.in_nasdaq100:
                    continue
                if idx in ["DIA", "DOW"] and not r.in_dow30:
                    continue
            
            filtered.append(r)
        
        return filtered
    
    def get_long_setups(self, min_confidence: float = 50) -> List[FormattedScanResult]:
        """Get all LONG_SETUP signals"""
        return self.filter_results(signal="LONG_SETUP", min_confidence=min_confidence)
    
    def get_short_setups(self, min_confidence: float = 50) -> List[FormattedScanResult]:
        """Get all SHORT_SETUP signals"""
        return self.filter_results(signal="SHORT_SETUP", min_confidence=min_confidence)
    
    def get_yellow_alerts(self) -> List[FormattedScanResult]:
        """Get all YELLOW signals (potential setups forming)"""
        return self.filter_results(signal="YELLOW")
    
    def get_actionable(self) -> List[FormattedScanResult]:
        """Get all actionable setups with trade plans"""
        return self.filter_results(actionable_only=True)
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def print_scan_summary(self, results: List[FormattedScanResult] = None) -> str:
        """Print summary of scan results"""
        if results is None:
            results = self.last_scan_results
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"SCAN SUMMARY - {self.last_scan_time.strftime('%Y-%m-%d %H:%M') if self.last_scan_time else 'N/A'}")
        lines.append("=" * 80)
        
        total = len(results)
        actionable = sum(1 for r in results if r.actionable)
        long_setups = sum(1 for r in results if r.signal == "LONG_SETUP")
        short_setups = sum(1 for r in results if r.signal == "SHORT_SETUP")
        yellow = sum(1 for r in results if r.signal == "YELLOW")
        neutral = sum(1 for r in results if r.signal in ["NEUTRAL", "NO_DATA"])
        
        lines.append(f"\nTotal Scanned: {total}")
        lines.append(f"Actionable:    {actionable}")
        lines.append(f"")
        lines.append(f"🟢 LONG:    {long_setups}")
        lines.append(f"🔴 SHORT:   {short_setups}")
        lines.append(f"🟡 YELLOW:  {yellow}")
        lines.append(f"⚪ NEUTRAL: {neutral}")
        
        # Top actionable
        if actionable > 0:
            lines.append(f"\n{'='*80}")
            lines.append("TOP ACTIONABLE SETUPS:")
            lines.append("=" * 80)
            
            for r in results[:10]:
                if r.actionable:
                    lines.append(f"\n{r.signal_emoji} {r.symbol:<6} | {r.combined_bias:<12} | Conf: {r.confidence:.0f}%")
                    lines.append(f"   {r.name[:40]}")
                    if r.trade_plan:
                        lines.append(f"   Direction: {r.trade_plan['direction']}")
                        lines.append(f"   Entry: {r.trade_plan['entry_zone']}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def print_result_detail(self, result: FormattedScanResult) -> str:
        """Print detailed view of a single result"""
        lines = []
        
        lines.append("=" * 70)
        lines.append(f"{result.signal_emoji} {result.symbol} - {result.name}")
        lines.append("=" * 70)
        
        lines.append(f"\nCombined Bias: {result.combined_bias} ({result.confidence:.0f}% confidence)")
        lines.append(f"MTF Signal:    {result.signal}")
        lines.append(f"Confluence:    {result.confluence:.0f}%")
        lines.append(f"Actionable:    {'YES ✅' if result.actionable else 'NO'}")
        
        lines.append(f"\n📊 SCORES:")
        lines.append(f"   Bull: {result.bull_score:.1f}")
        lines.append(f"   Bear: {result.bear_score:.1f}")
        
        lines.append(f"\n📈 SCENARIOS:")
        lines.append(f"   HIGH: {result.high_prob:.0f}%")
        lines.append(f"   LOW:  {result.low_prob:.0f}%")
        lines.append(f"   CHOP: {result.chop_prob:.0f}%")
        
        if result.gap_type:
            lines.append(f"\n🌙 OVERNIGHT:")
            lines.append(f"   Gap: {result.gap_type} ({result.gap_pct:+.2f}%)")
            lines.append(f"   Fill Prob: {result.gap_fill_prob}")
            lines.append(f"   Bias: {result.overnight_bias}")
        
        if result.key_levels:
            lines.append(f"\n📍 KEY LEVELS:")
            for name, level in sorted(result.key_levels.items(), key=lambda x: x[1], reverse=True)[:8]:
                lines.append(f"   {name:<18}: ${level:.2f}")
        
        if result.trade_plan:
            tp = result.trade_plan
            lines.append(f"\n📋 TRADE PLAN: {tp['direction']}")
            lines.append(f"   Entry: {tp['entry_zone']}")
            lines.append(f"   Stop:  ${tp['stop_loss']:.2f}")
            lines.append(f"   T1:    ${tp['target_1']:.2f}")
            lines.append(f"   T2:    ${tp['target_2']:.2f}")
            lines.append(f"   R:R:   {tp['reward_risk_ratio']:.2f}")
        
        if result.notes:
            lines.append(f"\n📝 NOTES:")
            for note in result.notes:
                lines.append(f"   • {note}")
        
        # Index membership
        indices = []
        if result.in_sp500:
            indices.append("S&P 500")
        if result.in_nasdaq100:
            indices.append("Nasdaq 100")
        if result.in_dow30:
            indices.append("Dow 30")
        if indices:
            lines.append(f"\n🏛️ INDEX MEMBERSHIP: {', '.join(indices)}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_scanner() -> LiveScanner:
    """Create and return a LiveScanner instance"""
    return LiveScanner()


def scan_indices() -> List[FormattedScanResult]:
    """Quick scan of major index ETFs"""
    scanner = LiveScanner()
    return scanner.scan_quick("indices")


def scan_sectors() -> List[FormattedScanResult]:
    """Quick scan of sector ETFs"""
    scanner = LiveScanner()
    return scanner.scan_quick("sectors")


def scan_dow() -> List[FormattedScanResult]:
    """Quick scan of Dow 30"""
    scanner = LiveScanner()
    return scanner.scan_quick("dow")


def scan_mega_tech() -> List[FormattedScanResult]:
    """Quick scan of mega cap tech"""
    scanner = LiveScanner()
    return scanner.scan_quick("mega")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("Creating Live Scanner...")
    scanner = LiveScanner()
    
    # Show available watchlists
    print("\n📋 Available Watchlists:")
    for name in scanner.get_watchlist_names():
        wl = scanner.get_watchlist(name)
        print(f"   {name}: {wl.symbol_count} symbols")
    
    # Search demo
    print("\n🔍 Search 'TECH':")
    results = scanner.search_symbols("TECH", limit=5)
    for r in results:
        print(f"   {r.symbol}: {r.name}")
    
    # Quick scan demo
    print("\n🔄 Scanning Index ETFs...")
    results = scanner.scan_quick("indices")
    
    print(scanner.print_scan_summary(results))
    
    # Show first result detail
    if results:
        print("\n" + scanner.print_result_detail(results[0]))
    
    # Filter examples
    print("\n🎯 FILTER EXAMPLES:")
    print(f"   Long setups: {len(scanner.get_long_setups())}")
    print(f"   Short setups: {len(scanner.get_short_setups())}")
    print(f"   Yellow alerts: {len(scanner.get_yellow_alerts())}")
    print(f"   Actionable: {len(scanner.get_actionable())}")
