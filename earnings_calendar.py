"""
Earnings Calendar - Fetch upcoming earnings dates

Sources (in priority order):
1. Polygon.io reference/tickers endpoint
2. Finnhub earnings calendar endpoint
3. Cache to minimize API calls

Usage:
    from earnings_calendar import EarningsCalendar
    
    # Get days until next earnings
    cal = EarningsCalendar()
    days = cal.days_until_earnings("NFLX")  # Returns int or None
    
    # Get full earnings info
    info = cal.get_earnings_info("AAPL")
    # Returns: {"date": "2024-02-01", "days_until": 3, "timing": "AMC"}
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import json
import time

# Try to import yfinance for free fallback
try:
    import yfinance as yf
    yf_available = True
except ImportError:
    yf_available = False


@dataclass
class EarningsInfo:
    """Earnings information for a symbol"""
    symbol: str
    date: str  # YYYY-MM-DD
    days_until: int
    timing: str  # BMO (before market), AMC (after market close), or ""
    quarter: str  # Q1, Q2, Q3, Q4
    year: int
    is_confirmed: bool


class EarningsCalendar:
    """
    Fetches and caches earnings dates from Polygon/Finnhub.
    
    Cache persists in memory with 4-hour TTL to minimize API calls.
    """
    
    # Cache TTL in seconds (4 hours)
    CACHE_TTL = 4 * 60 * 60
    
    # Days to consider "near earnings"
    NEAR_EARNINGS_THRESHOLD = 5
    
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._cache_time: Dict[str, float] = {}
        
        # API keys
        self.polygon_key = os.environ.get("POLYGON_API_KEY")
        self.finnhub_key = os.environ.get("FINNHUB_API_KEY")
        
        # Track API source for debugging
        self.last_source = None
    
    def days_until_earnings(self, symbol: str) -> Optional[int]:
        """
        Get days until next earnings for a symbol.
        
        Returns:
            int: Days until earnings (0 = today, negative = past)
            None: No earnings found or API error
        """
        info = self.get_earnings_info(symbol)
        return info.days_until if info else None
    
    def is_near_earnings(self, symbol: str, days: int = None) -> bool:
        """
        Check if symbol has earnings within N days.
        
        Args:
            symbol: Stock ticker
            days: Days threshold (default: NEAR_EARNINGS_THRESHOLD = 5)
        
        Returns:
            True if earnings within threshold
        """
        if days is None:
            days = self.NEAR_EARNINGS_THRESHOLD
            
        days_until = self.days_until_earnings(symbol)
        if days_until is None:
            return False
        return 0 <= days_until <= days
    
    def get_earnings_info(self, symbol: str) -> Optional[EarningsInfo]:
        """
        Get full earnings info for a symbol.
        
        Tries Polygon first, then Finnhub, then yfinance.
        Results are cached for CACHE_TTL seconds.
        """
        symbol = symbol.upper()
        
        # Check cache
        if self._is_cache_valid(symbol):
            cached = self._cache.get(symbol)
            if cached:
                return EarningsInfo(**cached)
            return None
        
        # Try Polygon first
        info = self._fetch_polygon(symbol)
        
        # Fallback to Finnhub
        if info is None and self.finnhub_key:
            info = self._fetch_finnhub(symbol)
        
        # Fallback to yfinance (free, no key needed)
        if info is None and yf_available:
            info = self._fetch_yfinance(symbol)
        
        # Cache result (even if None to avoid repeated failed lookups)
        self._cache_result(symbol, info)
        
        return info
    
    def get_earnings_batch(self, symbols: list) -> Dict[str, Optional[EarningsInfo]]:
        """Get earnings info for multiple symbols efficiently."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_earnings_info(symbol)
        return results
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache entry is still valid."""
        if symbol not in self._cache_time:
            return False
        age = time.time() - self._cache_time[symbol]
        return age < self.CACHE_TTL
    
    def _cache_result(self, symbol: str, info: Optional[EarningsInfo]):
        """Store result in cache."""
        self._cache_time[symbol] = time.time()
        if info:
            self._cache[symbol] = {
                "symbol": info.symbol,
                "date": info.date,
                "days_until": info.days_until,
                "timing": info.timing,
                "quarter": info.quarter,
                "year": info.year,
                "is_confirmed": info.is_confirmed
            }
        else:
            self._cache[symbol] = None
    
    def _fetch_polygon(self, symbol: str) -> Optional[EarningsInfo]:
        """
        Fetch earnings from Polygon.io reference endpoint.
        
        Uses: GET /v3/reference/tickers/{ticker}
        The response includes next_earnings_date field.
        """
        if not self.polygon_key:
            return None
        
        try:
            # Polygon ticker details endpoint
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": self.polygon_key}
            
            resp = requests.get(url, params=params, timeout=5)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            results = data.get("results", {})
            
            # Check for earnings date in different possible fields
            earnings_date = None
            
            # Try next_earnings_date first
            if "next_earnings_date" in results:
                earnings_date = results["next_earnings_date"]
            
            if not earnings_date:
                return None
            
            # Parse date and calculate days until
            try:
                earn_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                days_until = (earn_dt - today).days
                
                # Determine quarter
                quarter = f"Q{(earn_dt.month - 1) // 3 + 1}"
                
                self.last_source = "polygon"
                
                return EarningsInfo(
                    symbol=symbol,
                    date=earnings_date,
                    days_until=days_until,
                    timing="",  # Polygon doesn't provide timing
                    quarter=quarter,
                    year=earn_dt.year,
                    is_confirmed=True
                )
            except ValueError:
                return None
                
        except Exception as e:
            print(f"⚠️ Polygon earnings fetch error for {symbol}: {e}")
            return None
    
    def _fetch_finnhub(self, symbol: str) -> Optional[EarningsInfo]:
        """
        Fetch earnings from Finnhub calendar endpoint.
        
        Uses: GET /calendar/earnings
        """
        if not self.finnhub_key:
            return None
        
        try:
            # Get earnings for next 60 days
            today = datetime.now()
            from_date = today.strftime("%Y-%m-%d")
            to_date = (today + timedelta(days=60)).strftime("%Y-%m-%d")
            
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {
                "from": from_date,
                "to": to_date,
                "symbol": symbol,
                "token": self.finnhub_key
            }
            
            resp = requests.get(url, params=params, timeout=5)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            earnings_list = data.get("earningsCalendar", [])
            
            if not earnings_list:
                return None
            
            # Get the next upcoming earnings
            next_earnings = None
            for earn in earnings_list:
                if earn.get("symbol") == symbol:
                    next_earnings = earn
                    break
            
            if not next_earnings:
                return None
            
            earnings_date = next_earnings.get("date")
            if not earnings_date:
                return None
            
            # Parse date
            try:
                earn_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
                today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                days_until = (earn_dt - today_dt).days
                
                # Extract timing (BMO/AMC)
                hour = next_earnings.get("hour", "")
                timing = ""
                if hour:
                    hour_lower = hour.lower()
                    if "bmo" in hour_lower or "before" in hour_lower:
                        timing = "BMO"
                    elif "amc" in hour_lower or "after" in hour_lower:
                        timing = "AMC"
                
                # Determine quarter
                quarter = f"Q{next_earnings.get('quarter', (earn_dt.month - 1) // 3 + 1)}"
                year = next_earnings.get('year', earn_dt.year)
                
                self.last_source = "finnhub"
                
                return EarningsInfo(
                    symbol=symbol,
                    date=earnings_date,
                    days_until=days_until,
                    timing=timing,
                    quarter=quarter,
                    year=year,
                    is_confirmed=True
                )
            except ValueError:
                return None
                
        except Exception as e:
            print(f"⚠️ Finnhub earnings fetch error for {symbol}: {e}")
            return None
    
    def _fetch_yfinance(self, symbol: str) -> Optional[EarningsInfo]:
        """
        Fetch earnings from yfinance (free, no API key needed).
        
        Uses the calendar property of the Ticker object.
        """
        if not yf_available:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get calendar data (includes earnings)
            calendar = ticker.calendar
            
            if calendar is None:
                return None
            
            # yfinance returns different formats - handle dict and DataFrame
            earnings_date = None
            
            if isinstance(calendar, dict):
                # Dict format: {'Earnings Date': [Timestamp(...), ...], 'Revenue Estimate': ...}
                if 'Earnings Date' in calendar:
                    dates = calendar['Earnings Date']
                    if dates:
                        earnings_date = dates[0] if isinstance(dates, list) else dates
                # Also check 'earnings_date' lowercase
                elif 'earnings_date' in calendar:
                    dates = calendar['earnings_date']
                    if dates:
                        earnings_date = dates[0] if isinstance(dates, list) else dates
            else:
                # DataFrame format
                if hasattr(calendar, 'empty') and not calendar.empty:
                    if 'Earnings Date' in calendar.index:
                        earnings_row = calendar.loc['Earnings Date']
                        if hasattr(earnings_row, 'iloc'):
                            earnings_date = earnings_row.iloc[0]
                        else:
                            earnings_date = earnings_row
            
            if earnings_date is None:
                return None
            
            # Handle timestamp or string date
            if hasattr(earnings_date, 'strftime'):
                # It's a pandas Timestamp or datetime
                if hasattr(earnings_date, 'to_pydatetime'):
                    earn_dt = earnings_date.to_pydatetime()
                    if hasattr(earn_dt, 'tzinfo') and earn_dt.tzinfo:
                        earn_dt = earn_dt.replace(tzinfo=None)
                else:
                    earn_dt = earnings_date
                date_str = earn_dt.strftime("%Y-%m-%d")
            elif isinstance(earnings_date, str):
                earn_dt = datetime.strptime(earnings_date[:10], "%Y-%m-%d")
                date_str = earnings_date[:10]
            else:
                return None
            
            # Calculate days until
            today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            # Normalize earn_dt to date only
            if hasattr(earn_dt, 'date'):
                earn_date = earn_dt.date()
            else:
                earn_date = earn_dt
            today_date = today_dt.date()
            
            days_until = (earn_date - today_date).days
            
            # Determine quarter from the date
            if hasattr(earn_date, 'month'):
                month = earn_date.month
                year = earn_date.year
            else:
                month = earn_dt.month
                year = earn_dt.year
            quarter = f"Q{(month - 1) // 3 + 1}"
            
            self.last_source = "yfinance"
            
            return EarningsInfo(
                symbol=symbol,
                date=date_str,
                days_until=days_until,
                timing="",  # yfinance doesn't provide timing
                quarter=quarter,
                year=year,
                is_confirmed=False  # yfinance dates are estimates
            )
                
        except Exception as e:
            print(f"⚠️ yfinance earnings fetch error for {symbol}: {e}")
            return None
    
    def format_earnings_warning(self, symbol: str) -> Optional[str]:
        """
        Get a formatted warning string for earnings proximity.
        
        Returns:
            "⚠️ EARNINGS in 3 days (Feb 8 AMC)" or None
        """
        info = self.get_earnings_info(symbol)
        if not info:
            return None
        
        if info.days_until < 0:
            return None  # Past earnings
        
        if info.days_until == 0:
            timing_str = f" ({info.timing})" if info.timing else ""
            return f"🚨 EARNINGS TODAY{timing_str}"
        
        if info.days_until <= self.NEAR_EARNINGS_THRESHOLD:
            try:
                earn_dt = datetime.strptime(info.date, "%Y-%m-%d")
                date_str = earn_dt.strftime("%b %d")
                timing_str = f" {info.timing}" if info.timing else ""
                return f"⚠️ EARNINGS in {info.days_until} days ({date_str}{timing_str})"
            except:
                return f"⚠️ EARNINGS in {info.days_until} days"
        
        return None
    
    def get_earnings_context(self, symbol: str) -> Dict:
        """
        Get earnings context for setup generator.
        
        Returns dict with:
        - has_upcoming: bool
        - days_until: int or None
        - warning: str or None
        - should_avoid: bool (within AVOID_EARNINGS_DAYS)
        - is_iv_inflated: bool (likely if <3 days out)
        """
        info = self.get_earnings_info(symbol)
        
        if not info or info.days_until < 0:
            return {
                "has_upcoming": False,
                "days_until": None,
                "warning": None,
                "should_avoid": False,
                "is_iv_inflated": False,
                "date": None,
                "timing": None
            }
        
        return {
            "has_upcoming": True,
            "days_until": info.days_until,
            "warning": self.format_earnings_warning(symbol),
            "should_avoid": info.days_until <= 3,  # Match AVOID_EARNINGS_DAYS
            "is_iv_inflated": info.days_until <= 3,
            "date": info.date,
            "timing": info.timing
        }


# Singleton instance for easy import
_calendar = None

def get_earnings_calendar() -> EarningsCalendar:
    """Get shared EarningsCalendar instance."""
    global _calendar
    if _calendar is None:
        _calendar = EarningsCalendar()
    return _calendar


# Quick access functions
def days_until_earnings(symbol: str) -> Optional[int]:
    """Quick access: Get days until earnings."""
    return get_earnings_calendar().days_until_earnings(symbol)

def is_near_earnings(symbol: str, days: int = 5) -> bool:
    """Quick access: Check if near earnings."""
    return get_earnings_calendar().is_near_earnings(symbol, days)

def get_earnings_warning(symbol: str) -> Optional[str]:
    """Quick access: Get formatted warning."""
    return get_earnings_calendar().format_earnings_warning(symbol)


if __name__ == "__main__":
    # Test the calendar
    cal = EarningsCalendar()
    
    test_symbols = ["NFLX", "AAPL", "TSLA", "META", "NVDA"]
    
    print("=" * 60)
    print("EARNINGS CALENDAR TEST")
    print("=" * 60)
    
    for symbol in test_symbols:
        info = cal.get_earnings_info(symbol)
        warning = cal.format_earnings_warning(symbol)
        
        print(f"\n{symbol}:")
        if info:
            print(f"  Date: {info.date}")
            print(f"  Days until: {info.days_until}")
            print(f"  Timing: {info.timing or 'N/A'}")
            print(f"  Quarter: {info.quarter} {info.year}")
            print(f"  Source: {cal.last_source}")
        else:
            print("  No earnings data found")
        
        if warning:
            print(f"  {warning}")
        
        context = cal.get_earnings_context(symbol)
        if context["should_avoid"]:
            print(f"  ❌ AVOID - Too close to earnings")
