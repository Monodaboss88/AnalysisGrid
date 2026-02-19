"""
Discord Bot Integration for SEF Trading Terminal
=================================================
Sends trading alerts and notifications to Discord via webhook.
Dead simple — just POST JSON to the webhook URL.

No bot token needed. No OAuth. No permissions headache.

Setup:
1. Create webhook in Discord channel settings → Integrations → Webhooks
2. Set DISCORD_WEBHOOK_URL in environment
3. That's it.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import httpx
import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

_ET = ZoneInfo("America/New_York")

def _now_et() -> datetime:
    return datetime.now(_ET)


# =============================================================================
# CONFIGURATION
# =============================================================================

DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1473815224825155686/LaIO4JSZzLfpRf_dPU64IOsuAKr4obDFKusGgC76MxMvA8dKgOdKiNectI5AxfooZ1Tt"
)
BOT_USERNAME = "SEF Trading Bot"
BOT_AVATAR = None  # Optional avatar URL


# =============================================================================
# DISCORD CLIENT
# =============================================================================

class DiscordClient:
    """Async Discord webhook client for sending trading alerts"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or DISCORD_WEBHOOK_URL
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # =========================================================================
    # CORE SEND
    # =========================================================================

    async def send_message(self, content: str = "", embeds: List[Dict] = None,
                           username: str = None) -> bool:
        """Send a message via webhook"""
        client = await self._get_client()

        payload = {
            "username": username or BOT_USERNAME,
        }

        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds
        if BOT_AVATAR:
            payload["avatar_url"] = BOT_AVATAR

        try:
            resp = await client.post(self.webhook_url, json=payload)
            if resp.status_code in (200, 204):
                return True
            else:
                print(f"❌ Discord webhook error: {resp.status_code} — {resp.text[:200]}")
                return False
        except Exception as e:
            print(f"❌ Discord send failed: {e}")
            return False

    # =========================================================================
    # TRADING ALERTS
    # =========================================================================

    async def send_alert(self, symbol: str, level: float, direction: str,
                          action: str, note: str = "", price: float = None) -> bool:
        """Send a formatted trading alert with embed"""
        color = 0x00FF88 if action in ["LONG", "BUY"] else 0xFF4444 if action in ["SHORT", "SELL"] else 0x00D9FF
        direction_arrow = "⬆️" if direction == "above" else "⬇️"

        embed = {
            "title": f"🔔 ALERT TRIGGERED — {symbol}",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": f"**{symbol}**", "inline": True},
                {"name": "Level", "value": f"${level:.2f} {direction_arrow}", "inline": True},
                {"name": "Action", "value": f"**{action}**", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Trading Terminal"}
        }

        if price:
            embed["fields"].append({"name": "Current Price", "value": f"${price:.2f}", "inline": True})
        if note:
            embed["fields"].append({"name": "Note", "value": note, "inline": False})

        return await self.send_message(
            content=f"@here **{symbol}** alert triggered!",
            embeds=[embed]
        )

    async def send_scanner_brief(self, setups: List[Dict]) -> bool:
        """Send morning scanner summary"""
        if not setups:
            embed = {
                "title": "📡 Morning Scan Complete",
                "description": "No setups found matching criteria.",
                "color": 0x888888,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return await self.send_message(embeds=[embed])

        description = ""
        for i, setup in enumerate(setups[:15], 1):
            symbol = setup.get("symbol", "?")
            signal = setup.get("signal", "?")
            score = setup.get("score", 0)
            entry = setup.get("entry", 0)

            signal_emoji = "🟢" if signal == "GREEN" else "🟡" if signal == "YELLOW" else "🔴"
            description += f"{i}. {signal_emoji} **{symbol}** — Score: {score} | Entry: ${entry:.2f}\n"

        if len(setups) > 15:
            description += f"\n*...and {len(setups) - 15} more*"

        embed = {
            "title": f"📡 Morning Scanner Brief — {len(setups)} Setups",
            "description": description,
            "color": 0x00D9FF,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Trading Terminal"}
        }

        return await self.send_message(embeds=[embed])

    async def send_price(self, symbol: str, price: float, change: float,
                          change_pct: float) -> bool:
        """Send a price update"""
        color = 0x00FF88 if change >= 0 else 0xFF4444
        emoji = "🟢" if change >= 0 else "🔴"
        sign = "+" if change >= 0 else ""

        embed = {
            "title": f"{emoji} {symbol} — ${price:.2f}",
            "description": f"Change: {sign}{change:.2f} ({sign}{change_pct:.2f}%)",
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self.send_message(embeds=[embed])

    async def send_trade_stats(self, stats: Dict) -> bool:
        """Send trade performance stats"""
        wr = stats.get("win_rate", 0)
        color = 0x00FF88 if wr >= 60 else 0xFFC800 if wr >= 50 else 0xFF4444
        wr_emoji = "🔥" if wr >= 60 else "✅" if wr >= 50 else "⚠️"

        embed = {
            "title": "📊 Trade Performance",
            "color": color,
            "fields": [
                {"name": "Total Trades", "value": str(stats.get("total", 0)), "inline": True},
                {"name": "Wins", "value": str(stats.get("wins", 0)), "inline": True},
                {"name": "Losses", "value": str(stats.get("losses", 0)), "inline": True},
                {"name": f"{wr_emoji} Win Rate", "value": f"**{wr:.1f}%**", "inline": True},
                {"name": "Total P&L", "value": f"**${stats.get('total_pnl', 0):.2f}**", "inline": True},
                {"name": "Avg Win / Loss", "value": f"${stats.get('avg_win', 0):.2f} / ${stats.get('avg_loss', 0):.2f}", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Trading Terminal"}
        }

        return await self.send_message(embeds=[embed])

    async def send_alerts_list(self, alerts: List[Dict]) -> bool:
        """Send current alerts list"""
        if not alerts:
            return await self.send_message(content="📭 No active alerts.")

        description = ""
        for a in alerts[:20]:
            direction = "⬆️" if a.get("direction") == "above" else "⬇️"
            description += f"{direction} **{a.get('symbol')}** ${a.get('level', 0):.2f} — {a.get('action', '?')}"
            if a.get("note"):
                description += f" *{a['note']}*"
            description += "\n"

        if len(alerts) > 20:
            description += f"\n*...and {len(alerts) - 20} more*"

        embed = {
            "title": f"🔔 Active Alerts ({len(alerts)})",
            "description": description,
            "color": 0x00D9FF,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Trading Terminal"}
        }

        return await self.send_message(embeds=[embed])

    async def send_market_brief(self, prices: List[Dict]) -> bool:
        """Send market overview with major indices"""
        description = ""
        for p in prices:
            emoji = "🟢" if p.get("change", 0) >= 0 else "🔴"
            sign = "+" if p.get("change", 0) >= 0 else ""
            description += (
                f"{emoji} **{p['symbol']}** ${p['price']:.2f} "
                f"({sign}{p['change_pct']:.2f}%)\n"
            )

        embed = {
            "title": "📡 Market Brief",
            "description": description or "No data available.",
            "color": 0x00D9FF,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"SEF Trading Terminal — {_now_et().strftime('%I:%M %p ET')}"}
        }

        return await self.send_message(embeds=[embed])

    async def send_queue_stats(self, stats: Dict) -> bool:
        """Send task queue status"""
        embed = {
            "title": "📋 Task Queue Status",
            "color": 0x888888,
            "fields": [
                {"name": "⏳ Pending", "value": str(stats.get("pending", 0)), "inline": True},
                {"name": "⚙️ Processing", "value": str(stats.get("processing", 0)), "inline": True},
                {"name": "✅ Completed", "value": str(stats.get("completed", 0)), "inline": True},
                {"name": "❌ Failed", "value": str(stats.get("failed", 0)), "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self.send_message(embeds=[embed])

    async def send_error(self, error_msg: str) -> bool:
        """Send an error notification"""
        embed = {
            "title": "⚠️ System Alert",
            "description": error_msg,
            "color": 0xFF4444,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return await self.send_message(embeds=[embed])

    async def send_setup_analysis(self, symbol: str, price: float, change: float,
                                    change_pct: float, alerts: List[Dict] = None) -> bool:
        """Send setup analysis for a symbol"""
        color = 0x00FF88 if change >= 0 else 0xFF4444
        sign = "+" if change >= 0 else ""

        fields = [
            {"name": "Price", "value": f"${price:.2f}", "inline": True},
            {"name": "Change", "value": f"{sign}{change:.2f} ({sign}{change_pct:.2f}%)", "inline": True},
        ]

        if alerts:
            alert_text = ""
            for a in alerts[:5]:
                d = "⬆️" if a.get("direction") == "above" else "⬇️"
                alert_text += f"{d} ${a.get('level', 0):.2f} — {a.get('action')}\n"
            fields.append({"name": "Active Alerts", "value": alert_text or "None", "inline": False})

        embed = {
            "title": f"📊 Setup Analysis — {symbol}",
            "color": color,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Trading Terminal — Full analysis in web UI"}
        }

        return await self.send_message(embeds=[embed])


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_discord_client: Optional[DiscordClient] = None

def get_discord() -> DiscordClient:
    """Get the global Discord client instance"""
    global _discord_client
    if _discord_client is None:
        _discord_client = DiscordClient()
    return _discord_client
