"""
Telegram Bot Integration for SEF Trading Terminal
==================================================
Handles inbound commands and outbound alert delivery via Telegram.
Uses webhook mode with FastAPI integration.

Setup:
1. Message @BotFather on Telegram → /newbot → get token
2. Set TELEGRAM_BOT_TOKEN in environment
3. Set TELEGRAM_CHAT_ID (your personal chat ID)
4. Bot auto-registers webhook on startup

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import httpx
import asyncio
import hashlib
import hmac
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field


# =============================================================================
# CONFIGURATION
# =============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # Your personal chat ID
TELEGRAM_API_BASE = "https://api.telegram.org/bot"
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TelegramMessage:
    """Parsed incoming Telegram message"""
    chat_id: str
    user_id: str
    username: str
    text: str
    message_id: int
    timestamp: str = ""
    is_command: bool = False
    command: str = ""
    command_args: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.text and self.text.startswith("/"):
            self.is_command = True
            parts = self.text.split(" ", 1)
            self.command = parts[0].lower().replace("@", "").split("@")[0]  # strip bot username
            self.command_args = parts[1] if len(parts) > 1 else ""


@dataclass
class OutboundMessage:
    """Message to send via Telegram"""
    chat_id: str
    text: str
    parse_mode: str = "Markdown"
    disable_notification: bool = False
    reply_markup: Optional[Dict] = None


# =============================================================================
# TELEGRAM CLIENT
# =============================================================================

class TelegramClient:
    """Async Telegram Bot API client"""

    def __init__(self, token: str = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.base_url = f"{TELEGRAM_API_BASE}{self.token}"
        self._client = None
        self._command_handlers: Dict[str, Callable] = {}
        self._default_handler: Optional[Callable] = None
        self.bot_info = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # =========================================================================
    # API CALLS
    # =========================================================================

    async def api_call(self, method: str, data: Dict = None) -> Dict:
        """Make a Telegram Bot API call"""
        client = await self._get_client()
        url = f"{self.base_url}/{method}"

        try:
            if data:
                response = await client.post(url, json=data)
            else:
                response = await client.get(url)

            result = response.json()
            if not result.get("ok"):
                print(f"❌ Telegram API error [{method}]: {result.get('description', 'Unknown')}")
            return result
        except Exception as e:
            print(f"❌ Telegram API call failed [{method}]: {e}")
            return {"ok": False, "error": str(e)}

    async def get_me(self) -> Dict:
        """Get bot info"""
        result = await self.api_call("getMe")
        if result.get("ok"):
            self.bot_info = result["result"]
        return result

    async def set_webhook(self, url: str, secret_token: str = None) -> Dict:
        """Register webhook URL with Telegram"""
        data = {
            "url": url,
            "allowed_updates": ["message", "callback_query"],
            "drop_pending_updates": True
        }
        if secret_token:
            data["secret_token"] = secret_token

        result = await self.api_call("setWebhook", data)
        if result.get("ok"):
            print(f"✅ Telegram webhook set: {url}")
        return result

    async def delete_webhook(self) -> Dict:
        """Remove webhook"""
        return await self.api_call("deleteWebhook", {"drop_pending_updates": True})

    async def get_webhook_info(self) -> Dict:
        """Get current webhook status"""
        return await self.api_call("getWebhookInfo")

    # =========================================================================
    # SENDING MESSAGES
    # =========================================================================

    async def send_message(self, chat_id: str = None, text: str = "",
                           parse_mode: str = "Markdown",
                           disable_notification: bool = False,
                           reply_markup: Dict = None) -> Dict:
        """Send a text message"""
        data = {
            "chat_id": chat_id or TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification
        }
        if reply_markup:
            data["reply_markup"] = reply_markup

        return await self.api_call("sendMessage", data)

    async def send_alert(self, symbol: str, level: float, direction: str,
                          action: str, note: str = "", price: float = None) -> Dict:
        """Send a formatted trading alert"""
        emoji = "🟢" if action in ["LONG", "BUY"] else "🔴" if action in ["SHORT", "SELL"] else "🔔"
        direction_arrow = "⬆️" if direction == "above" else "⬇️"

        text = (
            f"{emoji} *ALERT TRIGGERED*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"*{symbol}* {direction_arrow} ${level:.2f}\n"
            f"Action: *{action}*\n"
        )

        if price:
            text += f"Current: ${price:.2f}\n"
        if note:
            text += f"Note: _{note}_\n"

        text += f"\n⏰ {datetime.now().strftime('%I:%M %p ET')}"

        # Add quick action buttons
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": f"📊 Setup {symbol}", "callback_data": f"setup_{symbol}"},
                    {"text": "📋 All Alerts", "callback_data": "alerts_list"}
                ],
                [
                    {"text": f"✅ Log Trade", "callback_data": f"log_{symbol}_{action}_{level}"},
                    {"text": "❌ Dismiss", "callback_data": "dismiss"}
                ]
            ]
        }

        return await self.send_message(
            text=text,
            reply_markup=reply_markup,
            disable_notification=False
        )

    async def send_scanner_brief(self, setups: List[Dict]) -> Dict:
        """Send morning scanner summary"""
        if not setups:
            return await self.send_message(text="📡 *Morning Scan Complete*\nNo setups found matching criteria.")

        text = f"📡 *Morning Scanner Brief*\n"
        text += f"━━━━━━━━━━━━━━━━━━\n"
        text += f"Found *{len(setups)}* setups\n\n"

        for i, setup in enumerate(setups[:10], 1):  # Max 10 in one message
            symbol = setup.get("symbol", "?")
            signal = setup.get("signal", "?")
            score = setup.get("score", 0)
            entry = setup.get("entry", 0)

            signal_emoji = "🟢" if signal == "GREEN" else "🟡" if signal == "YELLOW" else "🔴"
            text += f"{i}. {signal_emoji} *{symbol}* — Score: {score} | Entry: ${entry:.2f}\n"

        if len(setups) > 10:
            text += f"\n_...and {len(setups) - 10} more_"

        text += f"\n\n⏰ {datetime.now().strftime('%I:%M %p ET')}"

        return await self.send_message(text=text)

    async def send_trade_stats(self, stats: Dict) -> Dict:
        """Send trade performance stats"""
        wr = stats.get("win_rate", 0)
        wr_emoji = "🔥" if wr >= 60 else "✅" if wr >= 50 else "⚠️"

        text = (
            f"📊 *Trade Performance*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Total Trades: {stats.get('total', 0)}\n"
            f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
            f"{wr_emoji} Win Rate: *{wr:.1f}%*\n"
            f"Total P&L: *${stats.get('total_pnl', 0):.2f}*\n"
            f"Avg Win: ${stats.get('avg_win', 0):.2f}\n"
            f"Avg Loss: ${stats.get('avg_loss', 0):.2f}\n"
        )

        return await self.send_message(text=text)

    async def send_error(self, error_msg: str) -> Dict:
        """Send an error notification"""
        text = f"⚠️ *System Alert*\n{error_msg}"
        return await self.send_message(text=text)

    # =========================================================================
    # COMMAND HANDLING
    # =========================================================================

    def command(self, cmd: str):
        """Decorator to register a command handler"""
        def decorator(func):
            self._command_handlers[cmd] = func
            return func
        return decorator

    def default_handler(self, func):
        """Decorator to register a default message handler"""
        self._default_handler = func
        return func

    def parse_update(self, update: Dict) -> Optional[TelegramMessage]:
        """Parse a Telegram update into a TelegramMessage"""
        message = update.get("message")
        if not message:
            # Could be a callback query
            callback = update.get("callback_query")
            if callback:
                return TelegramMessage(
                    chat_id=str(callback["message"]["chat"]["id"]),
                    user_id=str(callback["from"]["id"]),
                    username=callback["from"].get("username", ""),
                    text=callback.get("data", ""),
                    message_id=callback["message"]["message_id"],
                    is_command=True,
                    command=f"callback_{callback.get('data', '').split('_')[0]}",
                    command_args=callback.get("data", "")
                )
            return None

        text = message.get("text", "")
        from_user = message.get("from", {})

        return TelegramMessage(
            chat_id=str(message["chat"]["id"]),
            user_id=str(from_user.get("id", "")),
            username=from_user.get("username", ""),
            text=text,
            message_id=message.get("message_id", 0)
        )

    async def process_update(self, update: Dict) -> Optional[str]:
        """Process an incoming update and route to handlers"""
        msg = self.parse_update(update)
        if not msg:
            return None

        # Security: verify this is from authorized user
        if TELEGRAM_CHAT_ID and msg.chat_id != TELEGRAM_CHAT_ID:
            print(f"⚠️ Unauthorized Telegram message from chat {msg.chat_id}")
            return None

        # Route to command handler
        if msg.is_command and msg.command in self._command_handlers:
            return await self._command_handlers[msg.command](msg)

        # Route to default handler (for natural language / Haiku parsing)
        if self._default_handler:
            return await self._default_handler(msg)

        return None

    async def answer_callback(self, callback_query_id: str, text: str = "") -> Dict:
        """Answer a callback query (dismiss the loading state)"""
        return await self.api_call("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text": text
        })


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_telegram_client: Optional[TelegramClient] = None

def get_telegram() -> TelegramClient:
    """Get the global Telegram client instance"""
    global _telegram_client
    if _telegram_client is None:
        _telegram_client = TelegramClient()
    return _telegram_client
