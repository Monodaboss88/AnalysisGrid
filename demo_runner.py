"""
Demo Runner - Shows the scanner in action with realistic scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mtf_auction_scanner import MTFAuctionScanner, Timeframe


def generate_bullish_scenario(days: int = 10) -> pd.DataFrame:
    """Generate data showing a bullish setup forming"""
    np.random.seed(100)
    
    periods = days * 24 * 12  # 5-min bars
    price = 445.0
    
    data = []
    timestamp = datetime.now() - timedelta(days=days)
    
    for i in range(periods):
        # Create bullish structure: price finding support at value low, pushing to POC then VAH
        progress = i / periods
        
        # Trend component (bullish bias in second half)
        if progress < 0.3:
            trend = -0.0002  # Initial selloff
        elif progress < 0.5:
            trend = 0  # Base building
        else:
            trend = 0.0003 + (progress - 0.5) * 0.001  # Bullish resolution
        
        # Add noise
        noise = np.random.randn() * 0.002
        
        # Volume spike on breakout
        hour = (i * 5 // 60) % 24
        base_volume = 1.5 if 9 <= hour <= 16 else 0.5
        
        if progress > 0.7:
            base_volume *= 1.8  # Volume expansion
        
        returns = trend + noise
        
        open_price = price
        close_price = price * (1 + returns)
        
        # Make buying pressure visible (close near highs in bullish phase)
        if progress > 0.5:
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.002)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.0005)
        else:
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.001)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.001)
        
        volume = int(np.random.exponential(150000) * base_volume)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
        timestamp += timedelta(minutes=5)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def generate_bearish_scenario(days: int = 10) -> pd.DataFrame:
    """Generate data showing a bearish setup forming"""
    np.random.seed(200)
    
    periods = days * 24 * 12
    price = 460.0
    
    data = []
    timestamp = datetime.now() - timedelta(days=days)
    
    for i in range(periods):
        progress = i / periods
        
        # Create bearish structure: rally rejected at VAH, breaking POC
        if progress < 0.3:
            trend = 0.0002  # Initial rally
        elif progress < 0.5:
            trend = 0  # Distribution
        else:
            trend = -0.0003 - (progress - 0.5) * 0.001  # Bearish breakdown
        
        noise = np.random.randn() * 0.002
        
        hour = (i * 5 // 60) % 24
        base_volume = 1.5 if 9 <= hour <= 16 else 0.5
        
        if progress > 0.7:
            base_volume *= 1.8
        
        returns = trend + noise
        
        open_price = price
        close_price = price * (1 + returns)
        
        # Make selling pressure visible
        if progress > 0.5:
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.0005)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.002)
        else:
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.001)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.001)
        
        volume = int(np.random.exponential(150000) * base_volume)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
        timestamp += timedelta(minutes=5)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def generate_yellow_scenario(days: int = 10) -> pd.DataFrame:
    """Generate data showing a choppy/neutral scenario (YELLOW)"""
    np.random.seed(300)
    
    periods = days * 24 * 12
    price = 452.0
    
    data = []
    timestamp = datetime.now() - timedelta(days=days)
    
    for i in range(periods):
        progress = i / periods
        
        # Choppy, mean-reverting price action
        trend = 0.0001 * np.sin(i / 50)  # Oscillation
        noise = np.random.randn() * 0.003
        
        hour = (i * 5 // 60) % 24
        base_volume = 1.2 if 9 <= hour <= 16 else 0.6
        
        returns = trend + noise
        
        open_price = price
        close_price = price * (1 + returns)
        
        # Mixed candle types
        if np.random.rand() > 0.5:
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.001)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.001)
        else:
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.0015)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.0015)
        
        volume = int(np.random.exponential(100000) * base_volume)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
        timestamp += timedelta(minutes=5)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def run_all_demos():
    """Run scanner on all three scenarios"""
    scanner = MTFAuctionScanner()
    
    print("=" * 80)
    print("MTF AUCTION SCANNER - DEMO SCENARIOS")
    print("=" * 80)
    
    # Scenario 1: Bullish
    print("\n" + "🟢" * 20)
    print("SCENARIO 1: BULLISH SETUP")
    print("🟢" * 20)
    
    df_bull = generate_bullish_scenario()
    result_bull = scanner.scan(df_bull, symbol="BULL_DEMO")
    print(scanner.print_report(result_bull))
    
    # Scenario 2: Bearish
    print("\n" + "🔴" * 20)
    print("SCENARIO 2: BEARISH SETUP")
    print("🔴" * 20)
    
    df_bear = generate_bearish_scenario()
    result_bear = scanner.scan(df_bear, symbol="BEAR_DEMO")
    print(scanner.print_report(result_bear))
    
    # Scenario 3: Yellow/Choppy
    print("\n" + "🟡" * 20)
    print("SCENARIO 3: YELLOW/NEUTRAL - NO CLEAR SETUP")
    print("🟡" * 20)
    
    df_yellow = generate_yellow_scenario()
    result_yellow = scanner.scan(df_yellow, symbol="CHOP_DEMO")
    print(scanner.print_report(result_yellow))
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print(f"""
    BULLISH scenario:  {result_bull.dominant_signal.emoji} {result_bull.dominant_signal.value}
                       HIGH prob: {result_bull.high_scenario_prob:.0%} | Actionable: {result_bull.actionable}
    
    BEARISH scenario:  {result_bear.dominant_signal.emoji} {result_bear.dominant_signal.value}
                       LOW prob: {result_bear.low_scenario_prob:.0%} | Actionable: {result_bear.actionable}
    
    CHOPPY scenario:   {result_yellow.dominant_signal.emoji} {result_yellow.dominant_signal.value}
                       Neutral prob: {result_yellow.neutral_prob:.0%} | Actionable: {result_yellow.actionable}
    """)
    
    return result_bull, result_bear, result_yellow


if __name__ == "__main__":
    run_all_demos()
