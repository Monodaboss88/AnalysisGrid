"""
Options Flow Stream — Real-Time Flow Detection via Snapshot Diffing
====================================================================
Polls Polygon options snapshots periodically, diffs volume numbers
to detect new flow events, and pushes them to SSE subscribers.

Gives near-real-time (~30-60 s) options flow detection without
requiring per-contract WebSocket subscriptions.

Author: Rob's Trading Systems
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── safe imports ──
try:
    from polygon_options import fetch_options_snapshot_filtered, parse_contract
except ImportError:
    fetch_options_snapshot_filtered = None
    parse_contract = None

try:
    from options_flow_scanner import _calc_flow_score, _calc_gex
except ImportError:
    _calc_flow_score = None
    _calc_gex = None


# ── Data classes ──

@dataclass
class FlowEvent:
    """A detected options flow event (volume delta on a specific contract)."""
    ticker: str
    contract_symbol: str
    contract_type: str          # "call" or "put"
    strike: float
    expiration: str
    new_volume: int             # Volume delta since last poll
    total_volume: int
    open_interest: int
    price: float                # Last trade price or midpoint
    underlying_price: float
    iv: Optional[float]
    delta: Optional[float]
    premium_usd: float          # new_volume * 100 * price
    timestamp: str
    direction: str              # "BUY" / "SELL" / "MIXED" — heuristic

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "contract": self.contract_symbol,
            "type": self.contract_type,
            "strike": self.strike,
            "expiration": self.expiration,
            "newVolume": self.new_volume,
            "totalVolume": self.total_volume,
            "oi": self.open_interest,
            "price": self.price,
            "underlyingPrice": self.underlying_price,
            "iv": round(self.iv * 100, 1) if self.iv else None,
            "delta": round(self.delta, 3) if self.delta else None,
            "premiumUSD": round(self.premium_usd),
            "timestamp": self.timestamp,
            "direction": self.direction,
        }


@dataclass
class FlowSnapshot:
    """Aggregated summary pushed with each poll cycle."""
    ticker: str
    call_flow: int              # New call volume this cycle
    put_flow: int               # New put volume this cycle
    call_premium: float
    put_premium: float
    top_events: List[FlowEvent]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "type": "snapshot",
            "ticker": self.ticker,
            "callFlow": self.call_flow,
            "putFlow": self.put_flow,
            "callPremium": round(self.call_premium),
            "putPremium": round(self.put_premium),
            "netSentiment": "BULLISH" if self.call_premium > self.put_premium * 1.2
                else "BEARISH" if self.put_premium > self.call_premium * 1.2
                else "NEUTRAL",
            "topEvents": [e.to_dict() for e in self.top_events[:10]],
            "timestamp": self.timestamp,
        }


# ── Main Streamer ──

class OptionsFlowStream:
    """
    Background poller that detects new options flow and delivers events
    to asyncio.Queue subscribers (used by SSE endpoint).

    Usage:
        stream = OptionsFlowStream()
        stream.set_tickers(["AAPL", "TSLA", "NVDA"])
        asyncio.create_task(stream.start())

        # In SSE handler:
        q = stream.subscribe()
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\\n\\n"
    """

    def __init__(self, interval: int = 45, dte_max: int = 45, strike_range: float = 0.15):
        self.interval = interval          # seconds between polls
        self.dte_max = dte_max
        self.strike_range = strike_range
        self._tickers: List[str] = []
        self._running = False
        self._prev_volumes: Dict[str, int] = {}      # contract_sym -> last known volume
        self._prev_prices: Dict[str, float] = {}      # contract_sym -> last known price
        self._flow_buffer: List[dict] = []             # Rolling buffer of all events (dicts)
        self._max_buffer = 500
        self._subscribers: List[asyncio.Queue] = []
        self._poll_count = 0
        self._last_poll_time: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._task: Optional[asyncio.Task] = None

    # ── Public API ──

    def set_tickers(self, tickers: List[str]):
        self._tickers = [t.strip().upper() for t in tickers if t.strip()]
        logger.info(f"Flow stream tickers set: {self._tickers}")

    def get_tickers(self) -> List[str]:
        return list(self._tickers)

    @property
    def is_running(self) -> bool:
        return self._running

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        logger.info(f"SSE subscriber added (total: {len(self._subscribers)})")
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)
            logger.info(f"SSE subscriber removed (total: {len(self._subscribers)})")

    def get_buffer(self, limit: int = 100) -> List[dict]:
        """Return last N flow events from the buffer."""
        return self._flow_buffer[-limit:]

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "tickers": self._tickers,
            "interval": self.interval,
            "pollCount": self._poll_count,
            "lastPoll": self._last_poll_time,
            "bufferSize": len(self._flow_buffer),
            "subscribers": len(self._subscribers),
            "trackedContracts": len(self._prev_volumes),
        }

    # ── Lifecycle ──

    async def start(self):
        """Start the polling loop (runs forever until stop() is called)."""
        if self._running:
            return
        if not fetch_options_snapshot_filtered:
            logger.error("polygon_options not available — cannot start flow stream")
            return
        self._running = True
        logger.info(f"Options flow stream started — polling every {self.interval}s for {self._tickers}")
        try:
            while self._running:
                if self._tickers:
                    await self._poll_all()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            logger.info("Options flow stream stopped")

    def start_background(self):
        """Create an asyncio task for the polling loop."""
        if self._task and not self._task.done():
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.start())

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    # ── Polling logic ──

    async def _poll_all(self):
        """Poll all tickers and detect volume deltas."""
        now_ts = datetime.now(timezone.utc).isoformat()
        self._poll_count += 1
        self._last_poll_time = now_ts

        loop = asyncio.get_event_loop()
        for ticker in self._tickers:
            try:
                raw = await loop.run_in_executor(
                    self._executor,
                    lambda t=ticker: fetch_options_snapshot_filtered(
                        t, dte_min=0, dte_max=self.dte_max, strike_range_pct=self.strike_range
                    ),
                )
                if not raw or not raw.get("contracts"):
                    continue
                underlying_price = raw.get("underlyingPrice", 0)
                events = self._diff_contracts(ticker, raw["contracts"], underlying_price, now_ts)
                if events:
                    snapshot = self._build_snapshot(ticker, events, now_ts)
                    await self._broadcast(snapshot.to_dict())
                    # Also broadcast individual big events (premium >= $10k)
                    for ev in events:
                        if ev.premium_usd >= 10_000:
                            await self._broadcast(ev.to_dict())
            except Exception as e:
                logger.warning(f"Flow poll error for {ticker}: {e}")

    def _diff_contracts(
        self, ticker: str, contracts: list, underlying_price: float, ts: str
    ) -> List[FlowEvent]:
        """Compare contract volumes to previous poll, return list of FlowEvents for deltas."""
        events: List[FlowEvent] = []
        for c in contracts:
            sym = c.get("ticker") or c.get("symbol", "")
            vol = c.get("day", {}).get("volume", 0) or c.get("volume", 0)
            if not sym or not vol:
                continue

            prev_vol = self._prev_volumes.get(sym, 0)
            prev_price = self._prev_prices.get(sym)

            # Detect the contract details
            details = c.get("details", {})
            strike = details.get("strike_price", 0) or c.get("strike", 0)
            ctype = details.get("contract_type", "").lower()
            if not ctype:
                ctype = "call" if "C" in sym.split(":")[-1][-9:] else "put"
            expiry = details.get("expiration_date", "") or c.get("expiration", "")

            # Price info
            last_price = (
                c.get("last_quote", {}).get("midpoint")
                or c.get("last_trade", {}).get("price")
                or c.get("day", {}).get("close", 0)
                or c.get("price", 0)
            )
            iv = c.get("implied_volatility") or c.get("iv")
            greeks = c.get("greeks", {})
            delta_val = greeks.get("delta")
            oi = c.get("open_interest", 0) or c.get("oi", 0)

            # Update tracked price
            self._prev_prices[sym] = last_price

            if self._poll_count == 1:
                # First poll — just store baseline, don't generate events
                self._prev_volumes[sym] = vol
                continue

            delta_vol = vol - prev_vol
            self._prev_volumes[sym] = vol

            if delta_vol <= 0:
                continue

            # Heuristic direction: if price went up vs previous, likely buy side
            direction = "MIXED"
            if prev_price is not None and last_price:
                if last_price > prev_price * 1.005:
                    direction = "BUY"
                elif last_price < prev_price * 0.995:
                    direction = "SELL"

            premium = delta_vol * 100 * (last_price or 0)

            ev = FlowEvent(
                ticker=ticker,
                contract_symbol=sym,
                contract_type=ctype,
                strike=strike,
                expiration=expiry,
                new_volume=delta_vol,
                total_volume=vol,
                open_interest=oi,
                price=round(last_price, 2) if last_price else 0,
                underlying_price=underlying_price,
                iv=iv,
                delta=delta_val,
                premium_usd=premium,
                timestamp=ts,
                direction=direction,
            )
            events.append(ev)

        # Sort by premium descending
        events.sort(key=lambda e: e.premium_usd, reverse=True)

        # Buffer the top events
        for ev in events[:20]:
            d = ev.to_dict()
            d["type"] = "trade"
            self._flow_buffer.append(d)
        if len(self._flow_buffer) > self._max_buffer:
            self._flow_buffer = self._flow_buffer[-self._max_buffer:]

        return events

    def _build_snapshot(self, ticker: str, events: List[FlowEvent], ts: str) -> FlowSnapshot:
        call_events = [e for e in events if e.contract_type == "call"]
        put_events = [e for e in events if e.contract_type == "put"]
        return FlowSnapshot(
            ticker=ticker,
            call_flow=sum(e.new_volume for e in call_events),
            put_flow=sum(e.new_volume for e in put_events),
            call_premium=sum(e.premium_usd for e in call_events),
            put_premium=sum(e.premium_usd for e in put_events),
            top_events=events[:10],
            timestamp=ts,
        )

    # ── Broadcasting ──

    async def _broadcast(self, data: dict):
        """Push event dict to all SSE subscribers."""
        dead: List[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                # Slow consumer — drop oldest and retry
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except Exception:
                    dead.append(q)
        for q in dead:
            self._subscribers.remove(q)


# ── Singleton ──

_flow_stream_instance: Optional[OptionsFlowStream] = None


def get_flow_stream() -> OptionsFlowStream:
    global _flow_stream_instance
    if _flow_stream_instance is None:
        _flow_stream_instance = OptionsFlowStream()
    return _flow_stream_instance
