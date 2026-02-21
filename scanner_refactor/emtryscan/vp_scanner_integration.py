"""
Volume Profile Scanner Integration
==================================
Example integration with Finnhub data feed.

This shows how to:
1. Calculate VP levels from intraday data
2. Run entry detection on your watchlist
3. Generate alerts
"""

from .vp_entry_detector import (
    VolumeProfileEntryDetector, 
    VolumeProfileLevels, 
    PriceBar,
    ProfileType,
    classify_profile,
    EntrySignal,
    Direction
)
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class ScanResult:
    """Result from scanning a single symbol"""
    symbol: str
    signals: List[EntrySignal]
    levels: VolumeProfileLevels
    current_price: float
    timestamp: str


class VolumeProfileScanner:
    """
    Scans watchlist for volume profile entry setups.
    
    Integrates with your existing data feed (Finnhub, etc.)
    """
    
    def __init__(self, detector: VolumeProfileEntryDetector = None):
        self.detector = detector or VolumeProfileEntryDetector()
        self.alerts = []
    
    def calculate_volume_profile(self,
                                  bars: List[PriceBar],
                                  num_bins: int = 20) -> VolumeProfileLevels:
        """
        Calculate VAH, POC, VAL from price bars.

        Uses TPO-style calculation:
        1. Divide price range into bins
        2. Sum volume in each bin
        3. Find POC (highest volume bin)
        4. Find Value Area (70% of total volume around POC)

        NOTE: This is a standalone VP calculation using PriceBar inputs
        (not DataFrames). The canonical VP lives in market_scanner_v2
        TechnicalCalculator.calculate_volume_profile(). This version exists
        because the entry detector pipeline uses PriceBar objects, not DataFrames.
        """
        if not bars:
            return None
        
        # Get price range
        all_highs = [b.high for b in bars]
        all_lows = [b.low for b in bars]
        price_high = max(all_highs)
        price_low = min(all_lows)
        price_range = price_high - price_low
        
        if price_range == 0:
            return None
        
        bin_size = price_range / num_bins
        
        # Create volume profile histogram
        volume_profile = {}
        for i in range(num_bins):
            bin_low = price_low + (i * bin_size)
            bin_high = bin_low + bin_size
            bin_mid = (bin_low + bin_high) / 2
            volume_profile[bin_mid] = 0
        
        # Distribute volume across bins each bar touched
        for bar in bars:
            bar_range = bar.high - bar.low
            if bar_range == 0:
                # Doji - assign to single bin
                for bin_mid in volume_profile:
                    if abs(bar.close - bin_mid) <= bin_size / 2:
                        volume_profile[bin_mid] += bar.volume
                        break
            else:
                # Distribute proportionally
                for bin_mid in volume_profile:
                    bin_low = bin_mid - bin_size / 2
                    bin_high = bin_mid + bin_size / 2
                    
                    # Calculate overlap
                    overlap_low = max(bar.low, bin_low)
                    overlap_high = min(bar.high, bin_high)
                    
                    if overlap_high > overlap_low:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        volume_profile[bin_mid] += bar.volume * overlap_pct
        
        # Find POC (highest volume bin)
        poc_price = max(volume_profile, key=volume_profile.get)
        
        # Calculate Value Area (70% of total volume)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.70
        
        # Start from POC, expand up and down
        sorted_bins = sorted(volume_profile.keys())
        poc_idx = sorted_bins.index(poc_price)
        
        included_volume = volume_profile[poc_price]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while included_volume < target_volume and (low_idx > 0 or high_idx < len(sorted_bins) - 1):
            # Check which direction adds more volume
            vol_below = volume_profile[sorted_bins[low_idx - 1]] if low_idx > 0 else 0
            vol_above = volume_profile[sorted_bins[high_idx + 1]] if high_idx < len(sorted_bins) - 1 else 0
            
            if vol_below >= vol_above and low_idx > 0:
                low_idx -= 1
                included_volume += vol_below
            elif high_idx < len(sorted_bins) - 1:
                high_idx += 1
                included_volume += vol_above
            else:
                break
        
        val_price = sorted_bins[low_idx] - bin_size / 2
        vah_price = sorted_bins[high_idx] + bin_size / 2
        
        # Classify profile type
        vah_volume = sum(volume_profile[b] for b in sorted_bins[high_idx:] if b in volume_profile)
        val_volume = sum(volume_profile[b] for b in sorted_bins[:low_idx+1] if b in volume_profile)
        poc_volume = volume_profile[poc_price]
        
        profile_type = classify_profile(vah_volume, poc_volume, val_volume)
        
        return VolumeProfileLevels(
            vah=vah_price,
            poc=poc_price,
            val=val_price,
            profile_type=profile_type
        )
    
    def scan_symbol(self, 
                    symbol: str, 
                    bars: List[PriceBar],
                    profile_bars: List[PriceBar] = None,
                    avg_volume: float = None) -> ScanResult:
        """
        Scan a single symbol for entry signals.
        
        Args:
            symbol: Ticker symbol
            bars: Recent price bars for signal detection
            profile_bars: Bars to calculate VP from (default: use same bars)
            avg_volume: Average volume for comparison
        """
        # Calculate VP levels
        vp_bars = profile_bars or bars
        levels = self.calculate_volume_profile(vp_bars)
        
        if not levels:
            return ScanResult(
                symbol=symbol,
                signals=[],
                levels=None,
                current_price=bars[-1].close if bars else 0,
                timestamp=datetime.now().isoformat()
            )
        
        # Detect entries
        signals = self.detector.detect_entries(levels, bars, avg_volume)
        
        return ScanResult(
            symbol=symbol,
            signals=signals,
            levels=levels,
            current_price=bars[-1].close,
            timestamp=datetime.now().isoformat()
        )
    
    def scan_watchlist(self, 
                       watchlist: Dict[str, List[PriceBar]],
                       avg_volumes: Dict[str, float] = None) -> List[ScanResult]:
        """
        Scan entire watchlist for entry signals.
        
        Args:
            watchlist: Dict of symbol -> price bars
            avg_volumes: Dict of symbol -> average volume
        
        Returns:
            List of ScanResults with signals
        """
        results = []
        avg_volumes = avg_volumes or {}
        
        for symbol, bars in watchlist.items():
            avg_vol = avg_volumes.get(symbol)
            result = self.scan_symbol(symbol, bars, avg_volume=avg_vol)
            
            if result.signals:
                results.append(result)
        
        return results
    
    def format_alert(self, result: ScanResult) -> str:
        """Format scan result as alert message"""
        lines = [
            f"\n{'='*60}",
            f"🎯 ALERT: {result.symbol} @ ${result.current_price:.2f}",
            f"{'='*60}",
            f"Profile: {result.levels.profile_type.value.upper()}",
            f"VAH: ${result.levels.vah:.2f} | POC: ${result.levels.poc:.2f} | VAL: ${result.levels.val:.2f}",
            ""
        ]
        
        for signal in result.signals:
            direction_emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
            lines.extend([
                f"{direction_emoji} {signal.entry_type.value}",
                f"   Entry: ${signal.entry_price:.2f}",
                f"   Stop: ${signal.stop_price:.2f} | Target: ${signal.target_1:.2f}",
                f"   R:R: {signal.rr_ratio:.2f} | Confidence: {signal.confidence:.0f}%",
                f"   {signal.notes}",
                ""
            ])
        
        return "\n".join(lines)


# === FINNHUB INTEGRATION EXAMPLE ===

def convert_finnhub_candles(candles: dict) -> List[PriceBar]:
    """
    Convert Finnhub candle response to PriceBar list.
    
    Finnhub format:
    {
        'c': [close prices],
        'h': [high prices],
        'l': [low prices],
        'o': [open prices],
        'v': [volumes],
        't': [timestamps]
    }
    """
    bars = []
    if candles and candles.get('c'):
        for i in range(len(candles['c'])):
            bars.append(PriceBar(
                open=candles['o'][i],
                high=candles['h'][i],
                low=candles['l'][i],
                close=candles['c'][i],
                volume=candles['v'][i],
                timestamp=str(candles['t'][i])
            ))
    return bars


def scan_with_finnhub(api_key: str, symbols: List[str], resolution: str = "15"):
    """
    Example: Scan symbols using Finnhub data.
    
    Args:
        api_key: Your Finnhub API key
        symbols: List of symbols to scan
        resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M)
    """
    import requests
    from datetime import datetime, timedelta
    
    scanner = VolumeProfileScanner()
    
    # Time range: last 5 days of intraday data
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=5)).timestamp())
    
    results = []
    
    for symbol in symbols:
        try:
            # Fetch candles from Finnhub
            url = f"https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': start_time,
                'to': end_time,
                'token': api_key
            }
            
            response = requests.get(url, params=params)
            candles = response.json()
            
            if candles.get('s') == 'ok':
                bars = convert_finnhub_candles(candles)
                
                if len(bars) >= 10:
                    # Calculate average volume
                    avg_vol = np.mean([b.volume for b in bars])
                    
                    # Scan
                    result = scanner.scan_symbol(symbol, bars, avg_volume=avg_vol)
                    
                    if result.signals:
                        results.append(result)
                        print(scanner.format_alert(result))
        
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
    
    return results


# === MANUAL VP LEVELS INPUT ===

def scan_with_manual_levels(symbol: str, 
                            vah: float, 
                            poc: float, 
                            val: float,
                            bars: List[PriceBar],
                            profile_type: str = "normal") -> ScanResult:
    """
    Scan using manually input VP levels from your chart.
    
    Use this when you've already identified levels visually.
    """
    scanner = VolumeProfileScanner()
    
    ptype = ProfileType.NORMAL if profile_type.lower() == "normal" else ProfileType.INVERTED
    
    levels = VolumeProfileLevels(
        vah=vah,
        poc=poc,
        val=val,
        profile_type=ptype
    )
    
    avg_vol = np.mean([b.volume for b in bars]) if bars else None
    signals = scanner.detector.detect_entries(levels, bars, avg_vol)
    
    result = ScanResult(
        symbol=symbol,
        signals=signals,
        levels=levels,
        current_price=bars[-1].close if bars else 0,
        timestamp=datetime.now().isoformat()
    )
    
    if signals:
        print(scanner.format_alert(result))
    
    return result


# === MAIN EXAMPLE ===

if __name__ == "__main__":
    # Example 1: Manual levels with simulated data
    print("\n" + "="*60)
    print("EXAMPLE: Manual VP Levels Scan")
    print("="*60)
    
    # Simulated price bars approaching VAL
    test_bars = [
        PriceBar(open=151.00, high=152.00, low=150.50, close=151.50, volume=1000000),
        PriceBar(open=151.50, high=151.75, low=149.00, close=149.50, volume=1200000),
        PriceBar(open=149.50, high=150.00, low=148.00, close=148.25, volume=1100000),
        PriceBar(open=148.25, high=148.50, low=147.00, close=147.50, volume=1500000),
        # Rejection candle at VAL
        PriceBar(open=147.50, high=149.00, low=147.25, close=148.75, volume=1800000),
    ]
    
    result = scan_with_manual_levels(
        symbol="AAPL",
        vah=152.00,
        poc=150.00,
        val=147.50,
        bars=test_bars,
        profile_type="normal"
    )
    
    # Example 2: Inverted profile breakout
    print("\n" + "="*60)
    print("EXAMPLE: Inverted Profile Breakout")
    print("="*60)
    
    breakout_bars = [
        PriceBar(open=99.00, high=100.00, low=98.50, close=99.50, volume=800000),
        PriceBar(open=99.50, high=100.25, low=99.00, close=100.00, volume=900000),
        PriceBar(open=100.00, high=101.50, low=99.75, close=101.25, volume=1500000),
        PriceBar(open=101.25, high=102.00, low=100.50, close=100.75, volume=1200000),
        # Retesting VAH as support
        PriceBar(open=100.75, high=101.50, low=100.00, close=101.25, volume=1100000),
    ]
    
    result = scan_with_manual_levels(
        symbol="NVDA",
        vah=100.00,
        poc=98.00,
        val=96.00,
        bars=breakout_bars,
        profile_type="inverted"
    )
    
    print("\n" + "="*60)
    print("To use with Finnhub, call:")
    print("  scan_with_finnhub('YOUR_API_KEY', ['AAPL', 'NVDA', 'META'])")
    print("="*60)
