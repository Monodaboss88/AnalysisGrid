"""
SEF Trading System - Utilities
Helper functions and common utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ========== Time Utilities ==========

def get_session_boundaries(
    date: datetime,
    market_open: time = time(9, 30),
    market_close: time = time(16, 0)
) -> Tuple[datetime, datetime]:
    """Get session open and close times for a date"""
    open_dt = datetime.combine(date.date(), market_open)
    close_dt = datetime.combine(date.date(), market_close)
    return open_dt, close_dt


def is_trading_day(date: datetime) -> bool:
    """Check if date is a trading day (not weekend)"""
    return date.weekday() < 5


def get_previous_trading_day(date: datetime) -> datetime:
    """Get the previous trading day"""
    prev = date - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev


def minutes_since_open(
    current_time: datetime,
    market_open: time = time(9, 30)
) -> int:
    """Calculate minutes since market open"""
    open_dt = datetime.combine(current_time.date(), market_open)
    delta = current_time - open_dt
    return max(0, int(delta.total_seconds() / 60))


def minutes_until_close(
    current_time: datetime,
    market_close: time = time(16, 0)
) -> int:
    """Calculate minutes until market close"""
    close_dt = datetime.combine(current_time.date(), market_close)
    delta = close_dt - current_time
    return max(0, int(delta.total_seconds() / 60))


# ========== Price Utilities ==========

def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """Round price to nearest tick size"""
    return round(price / tick_size) * tick_size


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def price_to_ticks(price: float, tick_size: float = 0.01) -> int:
    """Convert price to tick count"""
    return int(price / tick_size)


# ========== Technical Utilities ==========

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (middle, upper, lower)"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_typical_price(df: pd.DataFrame) -> pd.Series:
    """Calculate Typical Price (H+L+C)/3"""
    return (df["high"] + df["low"] + df["close"]) / 3


def calculate_money_flow(df: pd.DataFrame) -> pd.Series:
    """Calculate Money Flow (typical price * volume)"""
    return calculate_typical_price(df) * df["volume"]


# ========== Candle Pattern Utilities ==========

def is_doji(row: pd.Series, threshold: float = 0.1) -> bool:
    """Check if candle is a doji (small body)"""
    body = abs(row["close"] - row["open"])
    range_ = row["high"] - row["low"]
    if range_ == 0:
        return True
    return (body / range_) < threshold


def is_hammer(row: pd.Series, wick_ratio: float = 2.0) -> bool:
    """Check if candle is a hammer (bullish reversal)"""
    body = abs(row["close"] - row["open"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    upper_wick = row["high"] - max(row["open"], row["close"])
    
    if body == 0:
        return False
    
    return (lower_wick / body >= wick_ratio) and (upper_wick < body)


def is_shooting_star(row: pd.Series, wick_ratio: float = 2.0) -> bool:
    """Check if candle is a shooting star (bearish reversal)"""
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    
    if body == 0:
        return False
    
    return (upper_wick / body >= wick_ratio) and (lower_wick < body)


def is_engulfing_bullish(curr: pd.Series, prev: pd.Series) -> bool:
    """Check for bullish engulfing pattern"""
    # Previous candle bearish, current candle bullish
    prev_bearish = prev["close"] < prev["open"]
    curr_bullish = curr["close"] > curr["open"]
    
    # Current body engulfs previous body
    engulfs = (
        curr["open"] <= prev["close"] and
        curr["close"] >= prev["open"]
    )
    
    return prev_bearish and curr_bullish and engulfs


def is_engulfing_bearish(curr: pd.Series, prev: pd.Series) -> bool:
    """Check for bearish engulfing pattern"""
    # Previous candle bullish, current candle bearish
    prev_bullish = prev["close"] > prev["open"]
    curr_bearish = curr["close"] < curr["open"]
    
    # Current body engulfs previous body
    engulfs = (
        curr["open"] >= prev["close"] and
        curr["close"] <= prev["open"]
    )
    
    return prev_bullish and curr_bearish and engulfs


# ========== Volume Utilities ==========

def calculate_vwap_manual(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP from scratch"""
    typical = calculate_typical_price(df)
    cum_pv = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_pv / cum_vol


def calculate_relative_volume(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Calculate relative volume vs average"""
    avg_vol = df["volume"].rolling(window=lookback).mean()
    return df["volume"] / avg_vol


def is_high_volume(df: pd.DataFrame, threshold: float = 1.5, lookback: int = 20) -> pd.Series:
    """Identify high volume bars"""
    rvol = calculate_relative_volume(df, lookback)
    return rvol >= threshold


# ========== Risk Utilities ==========

def calculate_position_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float
) -> int:
    """Calculate position size based on risk"""
    risk_amount = account_equity * risk_pct
    risk_per_share = abs(entry_price - stop_price)
    
    if risk_per_share == 0:
        return 0
    
    return int(risk_amount / risk_per_share)


def calculate_risk_reward(
    entry: float,
    stop: float,
    target: float
) -> float:
    """Calculate risk:reward ratio"""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    
    if risk == 0:
        return 0.0
    
    return reward / risk


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly Criterion for position sizing
    Returns fraction of capital to risk
    """
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    return (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio


# ========== Formatting Utilities ==========

def format_price(price: float) -> str:
    """Format price for display"""
    return f"${price:,.2f}"


def format_pnl(pnl: float) -> str:
    """Format P&L with color indicator"""
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:,.2f}"


def format_percentage(pct: float) -> str:
    """Format percentage for display"""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def format_volume(volume: int) -> str:
    """Format volume with K/M suffix"""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.1f}K"
    return str(volume)


# ========== Logging Utilities ==========

def log_trade_entry(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    stop: float,
    target: float
):
    """Log trade entry details"""
    rr = calculate_risk_reward(price, stop, target)
    
    logger.info("=" * 50)
    logger.info(f"TRADE ENTRY: {symbol}")
    logger.info(f"Side: {side.upper()}")
    logger.info(f"Quantity: {quantity}")
    logger.info(f"Entry: {format_price(price)}")
    logger.info(f"Stop: {format_price(stop)}")
    logger.info(f"Target: {format_price(target)}")
    logger.info(f"Risk/Reward: {rr:.2f}")
    logger.info("=" * 50)


def log_trade_exit(
    symbol: str,
    quantity: int,
    entry_price: float,
    exit_price: float,
    pnl: float
):
    """Log trade exit details"""
    pnl_pct = calculate_percentage_change(entry_price, exit_price)
    
    logger.info("=" * 50)
    logger.info(f"TRADE EXIT: {symbol}")
    logger.info(f"Quantity: {quantity}")
    logger.info(f"Entry: {format_price(entry_price)}")
    logger.info(f"Exit: {format_price(exit_price)}")
    logger.info(f"P&L: {format_pnl(pnl)} ({format_percentage(pnl_pct)})")
    logger.info("=" * 50)


# ========== Data Validation ==========

def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Validate OHLCV dataframe has required columns and format"""
    required_cols = ["open", "high", "low", "close", "volume"]
    
    # Check columns exist
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return False
    
    # Check for nulls
    if df[required_cols].isnull().any().any():
        logger.warning("OHLCV data contains null values")
    
    # Check OHLC relationship
    invalid = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    
    if invalid.any():
        logger.warning(f"Found {invalid.sum()} bars with invalid OHLC relationships")
    
    return True


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize OHLCV data"""
    df = df.copy()
    
    # Forward fill nulls
    df = df.ffill()
    
    # Fix invalid OHLC relationships
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    
    # Ensure volume is non-negative
    df["volume"] = df["volume"].clip(lower=0)
    
    return df
