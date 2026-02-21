"""
Polygon Data — Shared data layer replacing yfinance across the entire system.
==============================================================================
Single source of truth for all OHLCV bar data and price quotes.

Provides yfinance-compatible DataFrames (columns: Open, High, Low, Close, Volume)
so every existing module works with zero analysis code changes.

Usage:
    from polygon_data import get_bars, get_price_quote

    # Daily bars (replaces yf.Ticker("AAPL").history(period="6mo", interval="1d"))
    df = get_bars("AAPL", period="6mo", interval="1d")

    # Weekly bars (replaces yf.Ticker("AAPL").history(period="6mo", interval="1wk"))
    df = get_bars("AAPL", period="6mo", interval="1wk")

    # Intraday bars (replaces yf.Ticker("AAPL").history(period="1mo", interval="1h"))
    df = get_bars("AAPL", period="1mo", interval="1h")

    # Quick price quote (replaces yf.Ticker("AAPL").history(period="2d"))
    quote = get_price_quote("AAPL")
    # → {"symbol": "AAPL", "price": 185.50, "prev_close": 184.20, "change": 1.30, "change_pct": 0.71}

Env: POLYGON_API_KEY (required)
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
POLYGON_BASE = "https://api.polygon.io"
_session: Optional[requests.Session] = None
_rate_limit_until: float = 0  # epoch time to resume after a 429

# ─────────────────────────────────────────────
#  Shared Rate Limiter
# ─────────────────────────────────────────────
import threading

class _RateLimiter:
    """
    Thread-safe token bucket rate limiter for Polygon API calls.
    Prevents concurrent scanners from exceeding API limits.

    Default: 5 requests/second (Polygon free tier).
    Paid tiers can increase via set_rate().
    """

    def __init__(self, requests_per_second: float = 5.0):
        self._lock = threading.Lock()
        self._min_interval = 1.0 / requests_per_second
        self._last_request: float = 0.0
        self._total_requests: int = 0
        self._total_waits: int = 0

    def set_rate(self, requests_per_second: float):
        """Adjust rate limit (e.g., for paid tier)."""
        with self._lock:
            self._min_interval = 1.0 / requests_per_second

    def acquire(self):
        """Block until a request slot is available."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                self._total_waits += 1
                time.sleep(wait_time)
            self._last_request = time.time()
            self._total_requests += 1

    @property
    def stats(self) -> Dict:
        return {
            "total_requests": self._total_requests,
            "total_waits": self._total_waits,
            "rate_limit": round(1.0 / self._min_interval, 1),
        }


# Global rate limiter — shared across all scanners
_limiter = _RateLimiter()


def _get_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    return key


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Accept": "application/json"})
    return _session


# ─────────────────────────────────────────────
#  Period / Interval Mapping
# ─────────────────────────────────────────────

def _period_to_days(period: str) -> int:
    """Convert yfinance-style period string to days back."""
    p = period.lower().strip()
    mapping = {
        "1d": 1, "2d": 2, "5d": 5, "7d": 7,
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825,
        "ytd": (datetime.now() - datetime(datetime.now().year, 1, 1)).days or 1,
        "max": 3650,
    }
    # Handle numeric days like "60d"
    if p.endswith("d") and p[:-1].isdigit():
        return int(p[:-1])
    return mapping.get(p, 180)


def _interval_to_polygon(interval: str) -> Tuple[int, str]:
    """Convert yfinance-style interval to Polygon multiplier + timespan."""
    i = interval.lower().strip()
    mapping = {
        "1m":  (1,  "minute"),
        "5m":  (5,  "minute"),
        "5min": (5, "minute"),
        "15m": (15, "minute"),
        "15min": (15, "minute"),
        "30m": (30, "minute"),
        "30min": (30, "minute"),
        "1h":  (1,  "hour"),
        "60m": (1,  "hour"),
        "1d":  (1,  "day"),
        "1wk": (1,  "week"),
        "1mo_bars": (1, "month"),
    }
    if i in mapping:
        return mapping[i]
    # Default to daily
    return (1, "day")


# ─────────────────────────────────────────────
#  Core Fetcher
# ─────────────────────────────────────────────

def _fetch_aggs(ticker: str, from_date: str, to_date: str,
                multiplier: int = 1, timespan: str = "day",
                limit: int = 50000) -> pd.DataFrame:
    """
    Fetch aggregated bars from Polygon REST API.
    Returns DataFrame with yfinance-compatible columns:
    Open, High, Low, Close, Volume (and DatetimeIndex).
    """
    global _rate_limit_until

    # Shared rate limiter — prevents concurrent scanners from exceeding API limits
    _limiter.acquire()

    # Respect 429 backoff
    now = time.time()
    if now < _rate_limit_until:
        wait = _rate_limit_until - now
        logger.debug(f"Rate limited, waiting {wait:.1f}s")
        time.sleep(wait)

    key = _get_key()
    session = _get_session()
    
    url = (f"{POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}/range/"
           f"{multiplier}/{timespan}/{from_date}/{to_date}"
           f"?adjusted=true&sort=asc&limit={limit}&apiKey={key}")

    try:
        resp = session.get(url, timeout=30)
        
        if resp.status_code == 429:
            _rate_limit_until = time.time() + 12  # back off 12s
            logger.warning(f"Polygon 429 rate limit for {ticker}, backing off 12s")
            time.sleep(12)
            resp = session.get(url, timeout=30)
        
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Convert Polygon columns → yfinance columns
        # Polygon: o, h, l, c, v, vw, t, n
        rename = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
        df = df.rename(columns=rename)
        
        # Timestamp → DatetimeIndex (ET for intraday, date for daily/weekly)
        if timespan in ("minute", "hour"):
            df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df["Date"] = df["Date"].dt.tz_convert("America/New_York")
        else:
            df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
            df["Date"] = pd.to_datetime(df["Date"])
        
        df = df.set_index("Date")
        
        # Keep only the standard columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep]
        
        # Ensure numeric
        for col in keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
        
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Polygon HTTP error for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Polygon fetch error for {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def get_bars(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Drop-in replacement for yf.Ticker(symbol).history(period=period, interval=interval).
    
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a DatetimeIndex — identical to yfinance output.
    
    Args:
        ticker:   Stock symbol (e.g. "AAPL")
        period:   Lookback period — "5d", "1mo", "3mo", "6mo", "1y", "2y", or "60d" etc.
        interval: Bar size — "1d", "1wk", "1h", "5m", "15m", "30m"
    """
    days = _period_to_days(period)
    multiplier, timespan = _interval_to_polygon(interval)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # For intraday, Polygon free tier limits lookback; for paid tier, we're fine
    # But add a small buffer for weekends/holidays
    if timespan in ("minute", "hour"):
        start_date -= timedelta(days=5)  # buffer for weekends
    else:
        start_date -= timedelta(days=7)  # buffer for daily/weekly
    
    from_str = start_date.strftime("%Y-%m-%d")
    to_str = end_date.strftime("%Y-%m-%d")
    
    df = _fetch_aggs(ticker, from_str, to_str, multiplier, timespan)
    
    if df.empty:
        return df
    
    # For weekly, Polygon may return extra bars; trim to requested window
    if timespan == "day" and len(df) > days + 20:
        df = df.tail(days + 10)
    
    return df


def get_price_quote(symbol: str) -> Optional[Dict]:
    """
    Quick price + prev close + change.
    Uses Polygon previous day close + today's snapshot.
    
    Drop-in replacement for the yfinance price pattern:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="2d")
        price = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]

    Returns:
        {"symbol": "AAPL", "price": 185.50, "prev_close": 184.20, 
         "open": 184.50, "high": 186.00, "low": 184.10,
         "change": 1.30, "change_pct": 0.71, "volume": 45123456}
        or None on failure.
    """
    key = _get_key()
    session = _get_session()
    sym = symbol.upper()

    # Shared rate limiter
    _limiter.acquire()

    try:
        # Try snapshot first (real-time during market hours)
        snap_url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{sym}?apiKey={key}"
        resp = session.get(snap_url, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            t = data.get("ticker", {})
            day = t.get("day", {})
            prev = t.get("prevDay", {})
            
            # Current price — best available: last trade > day close > min bar close
            price = None
            lt = t.get("lastTrade", {})
            if lt and lt.get("p"):
                price = float(lt["p"])
            if not price and day.get("c"):
                price = float(day["c"])
            mn = t.get("min", {})
            if not price and mn and mn.get("c"):
                price = float(mn["c"])
            if not price and prev.get("c"):
                price = float(prev["c"])  # after-hours fallback
            
            prev_close = float(prev.get("c", 0)) if prev.get("c") else 0
            
            if price and prev_close:
                return {
                    "symbol": sym,
                    "price": round(price, 4),
                    "prev_close": round(prev_close, 4),
                    "open": round(float(day.get("o", 0) or 0), 4),
                    "high": round(float(day.get("h", 0) or 0), 4),
                    "low": round(float(day.get("l", 0) or 0), 4),
                    "change": round(price - prev_close, 4),
                    "change_pct": round((price - prev_close) / prev_close * 100, 4),
                    "volume": int(day.get("v", 0) or 0),
                }
        
        # Fallback: fetch 2 daily bars
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        df = _fetch_aggs(sym, start, end, 1, "day")
        
        if df.empty or len(df) < 1:
            return None
        
        price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else float(df["Open"].iloc[0])
        
        return {
            "symbol": sym,
            "price": round(price, 4),
            "prev_close": round(prev_close, 4),
            "open": round(float(df["Open"].iloc[-1]), 4),
            "high": round(float(df["High"].iloc[-1]), 4),
            "low": round(float(df["Low"].iloc[-1]), 4),
            "change": round(price - prev_close, 4),
            "change_pct": round((price - prev_close) / prev_close * 100, 4) if prev_close else 0,
            "volume": int(df["Volume"].iloc[-1]),
        }
        
    except Exception as e:
        logger.warning(f"Price quote error for {sym}: {e}")
        return None


# ─────────────────────────────────────────────
#  CLI Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    
    print("=" * 50)
    print("  Polygon Data — Shared Data Layer Test")
    print("=" * 50)
    
    # Test daily bars
    print("\n1. Daily bars (AAPL, 3mo):")
    df = get_bars("AAPL", "3mo", "1d")
    print(f"   {len(df)} bars, cols: {list(df.columns)}")
    if not df.empty:
        print(f"   Range: {df.index[0]} → {df.index[-1]}")
        print(f"   Last close: ${df['Close'].iloc[-1]:.2f}")
    
    # Test weekly bars
    print("\n2. Weekly bars (NVDA, 6mo):")
    df = get_bars("NVDA", "6mo", "1wk")
    print(f"   {len(df)} bars")
    if not df.empty:
        print(f"   Last close: ${df['Close'].iloc[-1]:.2f}")
    
    # Test intraday
    print("\n3. Hourly bars (META, 1mo):")
    df = get_bars("META", "1mo", "1h")
    print(f"   {len(df)} bars")
    if not df.empty:
        print(f"   Last close: ${df['Close'].iloc[-1]:.2f}")
    
    # Test price quote
    print("\n4. Price quote (SPY):")
    q = get_price_quote("SPY")
    print(f"   {json.dumps(q, indent=2)}")
