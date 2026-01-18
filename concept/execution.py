"""
SEF Trading System - Tradier Execution Client
Handles order routing and position management via Tradier API
"""

import asyncio
import aiohttp
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import logging

from config import APIConfig, RiskConfig

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderDuration(Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    PRE = "pre"  # Pre-market
    POST = "post"  # Post-market


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation"""
    order_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    duration: OrderDuration = OrderDuration.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    created_at: Optional[datetime] = None
    
    # Linked orders (for brackets)
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    account_type: str
    cash: float
    buying_power: float
    equity: float
    positions: List[Position] = field(default_factory=list)
    pending_orders: List[Order] = field(default_factory=list)


class TradierClient:
    """
    Async client for Tradier API
    Handles order routing, position management, and account info
    """
    
    def __init__(self, config: APIConfig):
        self.api_key = config.tradier_api_key
        self.account_id = config.tradier_account_id
        self.base_url = (
            config.tradier_sandbox_url 
            if config.use_sandbox 
            else config.tradier_base_url
        )
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Dict:
        """Make API request"""
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with session.get(url, params=data) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                async with session.post(url, data=data) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            logger.error(f"Tradier API error: {e}")
            raise
    
    # ========== Account Methods ==========
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        data = await self._request("GET", f"accounts/{self.account_id}/balances")
        balances = data.get("balances", {})
        
        return AccountInfo(
            account_id=self.account_id,
            account_type=balances.get("account_type", ""),
            cash=float(balances.get("cash", {}).get("cash_available", 0)),
            buying_power=float(balances.get("margin", {}).get("buying_power", 0)),
            equity=float(balances.get("total_equity", 0))
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all current positions"""
        data = await self._request("GET", f"accounts/{self.account_id}/positions")
        positions_data = data.get("positions", {})
        
        if positions_data == "null" or not positions_data:
            return []
        
        positions = positions_data.get("position", [])
        if isinstance(positions, dict):
            positions = [positions]
        
        return [
            Position(
                symbol=p.get("symbol", ""),
                quantity=int(p.get("quantity", 0)),
                avg_cost=float(p.get("cost_basis", 0)) / int(p.get("quantity", 1)),
                current_price=float(p.get("current_price", 0)),
                market_value=float(p.get("market_value", 0)),
                unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                unrealized_pnl_pct=float(p.get("unrealized_pnl_percent", 0))
            )
            for p in positions
        ]
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None
    
    # ========== Order Methods ==========
    
    async def place_order(self, order: Order) -> Order:
        """
        Place an order
        
        Args:
            order: Order object with trade details
            
        Returns:
            Order with order_id populated
        """
        data = {
            "class": "equity",
            "symbol": order.symbol.upper(),
            "side": order.side.value,
            "quantity": str(order.quantity),
            "type": order.order_type.value,
            "duration": order.duration.value
        }
        
        if order.limit_price is not None:
            data["price"] = str(order.limit_price)
        
        if order.stop_price is not None:
            data["stop"] = str(order.stop_price)
        
        result = await self._request(
            "POST",
            f"accounts/{self.account_id}/orders",
            data
        )
        
        order_response = result.get("order", {})
        order.order_id = str(order_response.get("id", ""))
        order.status = OrderStatus.PENDING
        order.created_at = datetime.now()
        
        logger.info(f"Order placed: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}")
        
        return order
    
    async def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float],
        stop_loss: float,
        take_profit: float,
        order_type: OrderType = OrderType.LIMIT,
        duration: OrderDuration = OrderDuration.DAY
    ) -> Dict[str, Order]:
        """
        Place a bracket order (entry + stop loss + take profit)
        
        Note: Tradier requires OCO/bracket orders through OTOCO
        For simplicity, we'll place as separate orders and track them
        """
        orders = {}
        
        # Entry order
        entry = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=entry_price,
            duration=duration
        )
        entry = await self.place_order(entry)
        orders["entry"] = entry
        
        # Determine exit sides
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY_TO_COVER
        
        # Stop loss order
        stop = Order(
            symbol=symbol,
            side=exit_side,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_loss,
            duration=OrderDuration.GTC  # Stop loss stays until canceled
        )
        stop = await self.place_order(stop)
        orders["stop_loss"] = stop
        
        # Take profit order
        tp = Order(
            symbol=symbol,
            side=exit_side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=take_profit,
            duration=OrderDuration.GTC
        )
        tp = await self.place_order(tp)
        orders["take_profit"] = tp
        
        # Link orders
        entry.stop_loss_order_id = stop.order_id
        entry.take_profit_order_id = tp.order_id
        
        return orders
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            await self._request(
                "DELETE",
                f"accounts/{self.account_id}/orders/{order_id}"
            )
            logger.info(f"Order canceled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of an order"""
        data = await self._request(
            "GET",
            f"accounts/{self.account_id}/orders/{order_id}"
        )
        
        order_data = data.get("order", {})
        if not order_data:
            return None
        
        return Order(
            order_id=str(order_data.get("id", "")),
            symbol=order_data.get("symbol", ""),
            side=OrderSide(order_data.get("side", "buy")),
            order_type=OrderType(order_data.get("type", "market")),
            quantity=int(order_data.get("quantity", 0)),
            limit_price=float(order_data.get("price", 0)) or None,
            stop_price=float(order_data.get("stop", 0)) or None,
            status=OrderStatus(order_data.get("status", "pending")),
            filled_qty=int(order_data.get("exec_quantity", 0)),
            avg_fill_price=float(order_data.get("avg_fill_price", 0))
        )
    
    async def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        data = await self._request(
            "GET",
            f"accounts/{self.account_id}/orders"
        )
        
        orders_data = data.get("orders", {})
        if orders_data == "null" or not orders_data:
            return []
        
        orders = orders_data.get("order", [])
        if isinstance(orders, dict):
            orders = [orders]
        
        return [
            Order(
                order_id=str(o.get("id", "")),
                symbol=o.get("symbol", ""),
                side=OrderSide(o.get("side", "buy")),
                order_type=OrderType(o.get("type", "market")),
                quantity=int(o.get("quantity", 0)),
                limit_price=float(o.get("price", 0)) or None,
                stop_price=float(o.get("stop", 0)) or None,
                status=OrderStatus(o.get("status", "pending")),
                filled_qty=int(o.get("exec_quantity", 0)),
                avg_fill_price=float(o.get("avg_fill_price", 0))
            )
            for o in orders
            if o.get("status") in ["pending", "open", "partially_filled"]
        ]
    
    # ========== Quote Methods ==========
    
    async def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        data = await self._request(
            "GET",
            "markets/quotes",
            {"symbols": symbol.upper()}
        )
        
        quotes = data.get("quotes", {}).get("quote", {})
        return quotes
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols"""
        data = await self._request(
            "GET",
            "markets/quotes",
            {"symbols": ",".join(s.upper() for s in symbols)}
        )
        
        quotes = data.get("quotes", {}).get("quote", [])
        if isinstance(quotes, dict):
            quotes = [quotes]
        
        return {q.get("symbol", ""): q for q in quotes}


class PositionManager:
    """
    Manages position sizing and risk
    """
    
    def __init__(self, risk_config: RiskConfig):
        self.risk = risk_config
    
    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_price: float,
        atr: Optional[float] = None
    ) -> int:
        """
        Calculate position size based on risk parameters
        
        Args:
            account_equity: Total account equity
            entry_price: Planned entry price
            stop_price: Stop loss price
            atr: Average True Range (optional, for volatility adjustment)
            
        Returns:
            Number of shares to trade
        """
        # Risk per trade
        risk_amount = account_equity * self.risk.max_risk_per_trade_pct
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share == 0:
            return 0
        
        # Basic position size
        shares = int(risk_amount / risk_per_share)
        
        # Cap by max position size
        max_position_value = account_equity * self.risk.max_position_pct
        max_shares_by_value = int(max_position_value / entry_price)
        
        shares = min(shares, max_shares_by_value)
        
        # Volatility adjustment if ATR provided
        if atr and atr > 0:
            # Reduce size for high volatility
            volatility_factor = entry_price / (atr * 10)  # Normalize
            volatility_factor = max(0.5, min(1.5, volatility_factor))
            shares = int(shares * volatility_factor)
        
        return max(1, shares)  # At least 1 share
    
    def calculate_scale_out_quantities(self, total_qty: int) -> List[int]:
        """
        Calculate quantities for each scale-out level
        
        Returns:
            List of quantities for each exit level
        """
        quantities = []
        remaining = total_qty
        
        for pct in self.risk.scale_out_pcts[:-1]:
            qty = int(total_qty * pct)
            quantities.append(qty)
            remaining -= qty
        
        # Last exit gets remaining
        quantities.append(remaining)
        
        return quantities


class ExecutionEngine:
    """
    High-level execution engine
    Combines position sizing with order execution
    """
    
    def __init__(
        self,
        client: TradierClient,
        position_manager: PositionManager
    ):
        self.client = client
        self.pm = position_manager
        self._active_trades: Dict[str, Dict] = {}  # symbol -> trade info
    
    async def execute_signal(
        self,
        symbol: str,
        side: str,  # "long" or "short"
        entry_price: float,
        stop_loss: float,
        targets: List[float],
        atr: float
    ) -> Optional[Dict[str, Order]]:
        """
        Execute a trading signal with full bracket
        
        Args:
            symbol: Stock symbol
            side: "long" or "short"
            entry_price: Entry price
            stop_loss: Stop loss price
            targets: List of target prices
            atr: ATR for position sizing
            
        Returns:
            Dict of orders if successful
        """
        # Get account info for position sizing
        account = await self.client.get_account_info()
        
        # Calculate position size
        qty = self.pm.calculate_position_size(
            account.equity,
            entry_price,
            stop_loss,
            atr
        )
        
        if qty == 0:
            logger.warning(f"Position size is 0 for {symbol}, skipping")
            return None
        
        # Determine order side
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL_SHORT
        
        # Place bracket order
        orders = await self.client.place_bracket_order(
            symbol=symbol,
            side=order_side,
            quantity=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=targets[0] if targets else entry_price
        )
        
        # Track the trade
        self._active_trades[symbol] = {
            "orders": orders,
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "targets": targets,
            "quantity": qty,
            "opened_at": datetime.now()
        }
        
        logger.info(f"Executed {side} signal for {symbol}: {qty} shares @ {entry_price}")
        
        return orders
    
    async def close_position(self, symbol: str) -> Optional[Order]:
        """Close an existing position"""
        position = await self.client.get_position(symbol)
        
        if not position:
            logger.warning(f"No position found for {symbol}")
            return None
        
        # Cancel any open orders for this symbol
        open_orders = await self.client.get_open_orders()
        for order in open_orders:
            if order.symbol.upper() == symbol.upper():
                await self.client.cancel_order(order.order_id)
        
        # Place market order to close
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY_TO_COVER
        
        close_order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity)
        )
        
        close_order = await self.client.place_order(close_order)
        
        # Remove from active trades
        if symbol in self._active_trades:
            del self._active_trades[symbol]
        
        return close_order
    
    async def update_stops(self, symbol: str, new_stop: float) -> bool:
        """Update stop loss for an active trade (trailing stop)"""
        if symbol not in self._active_trades:
            return False
        
        trade = self._active_trades[symbol]
        old_stop_id = trade["orders"].get("stop_loss", {}).order_id
        
        if old_stop_id:
            # Cancel old stop
            await self.client.cancel_order(old_stop_id)
        
        # Place new stop
        position = await self.client.get_position(symbol)
        if not position:
            return False
        
        exit_side = OrderSide.SELL if position.is_long else OrderSide.BUY_TO_COVER
        
        new_stop_order = Order(
            symbol=symbol,
            side=exit_side,
            order_type=OrderType.STOP,
            quantity=abs(position.quantity),
            stop_price=new_stop,
            duration=OrderDuration.GTC
        )
        
        new_stop_order = await self.client.place_order(new_stop_order)
        trade["orders"]["stop_loss"] = new_stop_order
        trade["stop_loss"] = new_stop
        
        return True
