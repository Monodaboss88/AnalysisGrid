"""
Polygon Options API Client
===========================
Fetches options chain snapshots, Greeks, IV, OI, and volume
from Polygon.io's Options Starter/Developer tier.

Requires POLYGON_API_KEY env var.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"


def _get_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    return key


def _fetch_stock_price(ticker: str) -> Optional[float]:
    """Fetch current stock price from Polygon prev-close or snapshot."""
    key = _get_key()
    symbol = ticker.strip().upper()
    # Try previous close endpoint (most reliable)
    try:
        resp = requests.get(
            f"{BASE_URL}/v2/aggs/ticker/{symbol}/prev",
            params={"apiKey": key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            return results[0].get("c")  # close price
    except Exception as e:
        logger.debug(f"Prev-close price fetch failed for {symbol}: {e}")
    # Fallback: stock snapshot
    try:
        resp = requests.get(
            f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
            params={"apiKey": key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        ticker_data = data.get("ticker", {})
        day = ticker_data.get("day", {})
        return day.get("c") or day.get("o") or ticker_data.get("lastTrade", {}).get("p")
    except Exception as e:
        logger.debug(f"Snapshot price fetch failed for {symbol}: {e}")
    return None


# ── Options Chain Snapshot ──

def fetch_options_snapshot(
    ticker: str,
    limit: int = 250,
    max_pages: int = 6,
    contract_type: Optional[str] = None,
    expiration_gte: Optional[str] = None,
    expiration_lte: Optional[str] = None,
    strike_gte: Optional[float] = None,
    strike_lte: Optional[float] = None,
) -> Dict:
    """
    Fetch full options chain snapshot for a ticker via Polygon v3 API.
    Returns all contracts with Greeks, IV, volume, OI, quotes.
    Paginates automatically up to max_pages.
    """
    key = _get_key()
    symbol = ticker.strip().upper()

    params = {
        "apiKey": key,
        "limit": min(limit, 250),
        "order": "asc",
        "sort": "ticker",
    }
    if contract_type:
        params["contract_type"] = contract_type
    if expiration_gte:
        params["expiration_date.gte"] = expiration_gte
    if expiration_lte:
        params["expiration_date.lte"] = expiration_lte
    if strike_gte is not None:
        params["strike_price.gte"] = strike_gte
    if strike_lte is not None:
        params["strike_price.lte"] = strike_lte

    all_results = []
    url = f"{BASE_URL}/v3/snapshot/options/{symbol}"
    pages = 0

    while url and pages < max_pages:
        try:
            resp = requests.get(url, params=params if pages == 0 else {"apiKey": key}, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            all_results.extend(results)

            url = data.get("next_url")
            pages += 1

            if not results:
                break
        except requests.exceptions.HTTPError as e:
            logger.error(f"Polygon options HTTP error for {symbol}: {e}")
            break
        except Exception as e:
            logger.error(f"Polygon options error for {symbol}: {e}")
            break

    # Extract underlying price from first result
    underlying_price = None
    if all_results:
        ua = all_results[0].get("underlying_asset", {})
        underlying_price = ua.get("price")

    # Fallback: fetch stock price separately if not in options data
    if not underlying_price:
        underlying_price = _fetch_stock_price(symbol)

    return {
        "ticker": symbol,
        "underlyingPrice": underlying_price,
        "contracts": all_results,
        "contractCount": len(all_results),
        "pages": pages,
    }


def fetch_options_snapshot_filtered(
    ticker: str,
    dte_min: int = 0,
    dte_max: int = 60,
    strike_range_pct: float = 0.20,
) -> Dict:
    """
    Fetches options snapshot filtered to relevant strikes (±20% of price)
    and expirations (next 0-60 DTE by default). More focused = faster.
    """
    today = datetime.now(timezone.utc).date()
    exp_gte = (today + timedelta(days=dte_min)).strftime("%Y-%m-%d")
    exp_lte = (today + timedelta(days=dte_max)).strftime("%Y-%m-%d")

    # First get underlying price for strike filtering
    key = _get_key()
    symbol = ticker.strip().upper()

    # Fetch underlying price for strike filtering
    price = _fetch_stock_price(symbol)

    strike_gte = None
    strike_lte = None
    if price:
        strike_gte = round(price * (1 - strike_range_pct), 2)
        strike_lte = round(price * (1 + strike_range_pct), 2)

    return fetch_options_snapshot(
        ticker,
        expiration_gte=exp_gte,
        expiration_lte=exp_lte,
        strike_gte=strike_gte,
        strike_lte=strike_lte,
    )


# ── Helpers ──

def parse_contract(c: Dict) -> Dict:
    """Flatten a single Polygon snapshot contract into a clean dict."""
    details = c.get("details", {})
    day = c.get("day", {})
    greeks = c.get("greeks", {})
    quote = c.get("last_quote", {})
    trade = c.get("last_trade", {})
    ua = c.get("underlying_asset", {})

    return {
        "optionTicker": details.get("ticker", ""),
        "contractType": details.get("contract_type", ""),
        "strike": details.get("strike_price", 0),
        "expiration": details.get("expiration_date", ""),
        "exerciseStyle": details.get("exercise_style", ""),

        # Price / bar
        "dayOpen": day.get("open"),
        "dayHigh": day.get("high"),
        "dayLow": day.get("low"),
        "dayClose": day.get("close"),
        "dayVolume": day.get("volume", 0),
        "dayChange": day.get("change"),
        "dayChangePct": day.get("change_percent"),
        "prevClose": day.get("previous_close"),
        "vwap": day.get("vwap"),

        # Greeks
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "iv": c.get("implied_volatility"),

        # OI
        "openInterest": c.get("open_interest", 0),

        # Quote
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "bidSize": quote.get("bid_size"),
        "askSize": quote.get("ask_size"),
        "midpoint": quote.get("midpoint"),

        # Last trade
        "lastPrice": trade.get("price"),
        "lastSize": trade.get("size"),

        # Break-even
        "breakEven": c.get("break_even_price"),

        # Underlying
        "underlyingPrice": ua.get("price"),
        "changeToBreakEven": ua.get("change_to_break_even"),
    }


def group_by_expiration(contracts: List[Dict]) -> Dict[str, List[Dict]]:
    """Group parsed contracts by expiration date string."""
    groups = {}
    for c in contracts:
        exp = c.get("expiration", "unknown")
        groups.setdefault(exp, []).append(c)
    return dict(sorted(groups.items()))


# ── Async wrappers ──

async def async_fetch_snapshot(ticker: str, **kwargs) -> Dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fetch_options_snapshot(ticker, **kwargs))


async def async_fetch_snapshot_filtered(ticker: str, **kwargs) -> Dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fetch_options_snapshot_filtered(ticker, **kwargs))
