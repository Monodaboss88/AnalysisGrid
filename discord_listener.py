"""
Discord Bot Listener — Two-Way Communication
=============================================
Full Discord bot that reads messages from your channel and responds.
Works alongside the webhook (which handles outbound alerts).

Setup:
1. Go to https://discord.com/developers/applications
2. Create application → Bot → Copy token
3. Enable MESSAGE CONTENT intent in Bot settings
4. Invite bot to server with URL from OAuth2 → URL Generator
   (Scopes: bot | Permissions: Send Messages, Read Messages, Embed Links)
5. Set DISCORD_BOT_TOKEN environment variable

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import asyncio
import threading
import discord
from discord import Intents
from datetime import datetime
from typing import Optional, Dict

# Import our existing command processor
from discord_bot import get_discord


# =============================================================================
# CONFIGURATION
# =============================================================================

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
COMMAND_PREFIX = "!"  # e.g., !price SPY, !alerts, !help
ALLOWED_CHANNEL_IDS = os.getenv("DISCORD_CHANNEL_IDS", "").split(",")  # Optional: restrict to specific channels


# =============================================================================
# BOT CLIENT
# =============================================================================

class SEFDiscordBot(discord.Client):
    """Discord bot that listens to messages and responds with trading data"""

    def __init__(self):
        intents = Intents.default()
        intents.message_content = True  # Required to read message text
        super().__init__(intents=intents)
        self._command_handlers = {}
        self._ready = False
        self._register_commands()

    def _register_commands(self):
        """Register all command handlers"""
        self._command_handlers = {
            "help": self.cmd_help,
            "price": self.cmd_price,
            "p": self.cmd_price,
            "alerts": self.cmd_alerts,
            "alert": self.cmd_alerts,
            "stats": self.cmd_stats,
            "brief": self.cmd_brief,
            "market": self.cmd_brief,
            "scan": self.cmd_scan,
            "setup": self.cmd_setup,
            "queue": self.cmd_queue,
            "autoscan": self.cmd_autoscan,
        }

    # =========================================================================
    # EVENTS
    # =========================================================================

    async def on_ready(self):
        self._ready = True
        print(f"✅ Discord bot connected as {self.user.name} ({self.user.id})")
        print(f"   Listening in {len(self.guilds)} server(s)")

    async def on_message(self, message: discord.Message):
        # Ignore own messages
        if message.author == self.user:
            return
        # Ignore other bots
        if message.author.bot:
            return

        content = message.content.strip()

        # Check for command prefix (!)
        if content.startswith(COMMAND_PREFIX):
            await self._handle_prefixed_command(message, content[len(COMMAND_PREFIX):])
            return

        # Also respond to natural language if bot is mentioned
        if self.user in message.mentions:
            clean_text = content.replace(f"<@{self.user.id}>", "").strip()
            await self._handle_natural_language(message, clean_text)
            return

        # Check for direct commands without prefix (price, scan, etc.)
        first_word = content.split()[0].lower() if content else ""
        if first_word in self._command_handlers:
            await self._handle_prefixed_command(message, content)
            return

    async def _handle_prefixed_command(self, message: discord.Message, command_text: str):
        """Handle a prefixed command like !price SPY"""
        parts = command_text.strip().split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = self._command_handlers.get(cmd)
        if handler:
            try:
                await handler(message, args)
            except Exception as e:
                await message.channel.send(f"⚠️ Error: {str(e)[:200]}")
        else:
            await message.channel.send(
                f"🤔 Unknown command: `{cmd}`\n"
                f"Try: `!help`, `!price SPY`, `!alerts`, `!brief`, `!scan NVDA`"
            )

    async def _handle_natural_language(self, message: discord.Message, text: str):
        """Handle natural language via the hybrid router"""
        try:
            from hybrid_router import get_router
            router = get_router()
            classification = await router.classify_message(text)

            task_type = classification.get("task_type")
            symbol = classification.get("symbol")
            quick_response = classification.get("quick_response")

            if quick_response:
                await message.channel.send(quick_response)
                return

            # Route to appropriate handler based on classification
            fake_args = symbol or ""
            type_to_cmd = {
                "price_check": self.cmd_price,
                "alert_lookup": self.cmd_alerts,
                "trade_stats": self.cmd_stats,
                "market_brief": self.cmd_brief,
                "scanner_run": self.cmd_scan,
                "setup_analysis": self.cmd_setup,
            }

            handler = type_to_cmd.get(task_type)
            if handler:
                await handler(message, fake_args)
            else:
                await message.channel.send(
                    f"🤔 Not sure what you need. Try:\n"
                    f"`!price SPY` — Price check\n"
                    f"`!scan NVDA` — Run scanner\n"
                    f"`!alerts` — View alerts\n"
                    f"`!help` — All commands"
                )
        except Exception as e:
            await message.channel.send(f"⚠️ Error processing: {str(e)[:200]}")

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    async def cmd_help(self, message: discord.Message, args: str):
        embed = discord.Embed(
            title="🤖 SEF Trading Bot — Commands",
            color=0x00D9FF,
            description=(
                "**Prefix commands** (use `!` or just type the command):\n\n"
                "`!price SYMBOL` — Quick price check\n"
                "`!alerts` — View active alerts\n"
                "`!alerts SPY` — Alerts for specific symbol\n"
                "`!stats` — Trade performance stats\n"
                "`!scan NVDA` — Run scanner on symbol\n"
                "`!setup TSLA` — Setup analysis with levels\n"
                "`!brief` — Market brief (SPY, QQQ, IWM, DIA)\n"
                "`!autoscan` — Auto-scanner status\n"
                "`!autoscan trigger` — Run scan now\n"
                "`!autoscan start/stop` — Control scanner\n"
                "`!queue` — Task queue status\n"
                "`!help` — This message\n\n"
                "**Or mention me** with natural language:\n"
                f"<@{self.user.id}> how is NVDA looking?\n"
                f"<@{self.user.id}> show me my alerts"
            )
        )
        embed.set_footer(text="SEF Trading Terminal")
        await message.channel.send(embed=embed)

    async def cmd_price(self, message: discord.Message, args: str):
        if not args:
            await message.channel.send("⚠️ Specify a symbol: `!price SPY`")
            return

        raw = args.strip().upper()
        for filler in ["FOR ", "ON ", "OF ", "CHECK "]:
            if raw.startswith(filler):
                raw = raw[len(filler):]
        symbol = raw.strip()
        price_data = await _get_price_data(symbol)

        if not price_data:
            await message.channel.send(f"⚠️ Could not get price for {symbol}")
            return

        price = price_data["price"]
        change = price_data["change"]
        change_pct = price_data["change_pct"]

        color = 0x00FF88 if change >= 0 else 0xFF4444
        emoji = "🟢" if change >= 0 else "🔴"
        sign = "+" if change >= 0 else ""

        embed = discord.Embed(
            title=f"{emoji} {symbol} — ${price:.2f}",
            description=f"Change: {sign}{change:.2f} ({sign}{change_pct:.2f}%)",
            color=color,
            timestamp=datetime.utcnow()
        )
        await message.channel.send(embed=embed)

    async def cmd_alerts(self, message: discord.Message, args: str):
        try:
            import httpx
            # Strip filler words like "for", "on", "of" before the symbol
            raw = args.strip().upper()
            debug_mode = False
            if raw.startswith("DEBUG"):
                debug_mode = True
                raw = raw[5:].strip()
            for filler in ["FOR ", "ON ", "OF ", "CHECK "]:
                if raw.startswith(filler):
                    raw = raw[len(filler):]
            symbol = raw.strip() if raw else None

            port = os.environ.get("PORT", "8000")
            base = f"http://localhost:{port}"
            debug_log = []

            all_alerts = []

            # 1) Try local/chart_system alerts (no user_id)
            try:
                params = {}
                if symbol:
                    params["symbol"] = symbol
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{base}/api/alerts", params=params)
                    debug_log.append(f"Local API: status={resp.status_code}")
                    if resp.status_code == 200:
                        data = resp.json()
                        debug_log.append(f"Local: {data.get('count', 0)} alerts, storage={data.get('storage', '?')}")
                        for a in data.get("alerts", []):
                            a["_source"] = data.get("storage", "local")
                            all_alerts.append(a)
            except Exception as e:
                debug_log.append(f"Local API error: {e}")

            # 2) Try Firestore — iterate all user docs for their alerts
            try:
                from firestore_store import get_firestore
                fs = get_firestore()
                fs_available = fs.is_available() and fs.db is not None
                debug_log.append(f"Firestore: available={fs_available}")

                if fs_available:
                    seen_ids = {a.get("id") for a in all_alerts if a.get("id")}
                    users_ref = fs.db.collection('users')
                    user_count = 0
                    fs_alert_count = 0

                    for user_doc in users_ref.stream():
                        user_count += 1
                        alerts_ref = users_ref.document(user_doc.id).collection('alerts')
                        q = alerts_ref
                        if symbol:
                            q = q.where('symbol', '==', symbol)
                        for doc in q.stream():
                            fs_alert_count += 1
                            alert = doc.to_dict()
                            alert['id'] = doc.id
                            if doc.id not in seen_ids and not alert.get('triggered', False):
                                alert["_source"] = "firestore"
                                all_alerts.append(alert)
                                seen_ids.add(doc.id)

                    debug_log.append(f"Firestore: {user_count} users, {fs_alert_count} raw alerts, {len([a for a in all_alerts if a.get('_source')=='firestore'])} kept")

                    # Also try common user IDs directly
                    for uid in ["default", "anonymous"]:
                        try:
                            alerts_ref = users_ref.document(uid).collection('alerts')
                            q = alerts_ref
                            if symbol:
                                q = q.where('symbol', '==', symbol)
                            for doc in q.stream():
                                alert = doc.to_dict()
                                alert['id'] = doc.id
                                if doc.id not in seen_ids and not alert.get('triggered', False):
                                    alert["_source"] = f"firestore/{uid}"
                                    all_alerts.append(alert)
                                    seen_ids.add(doc.id)
                        except Exception:
                            pass

            except ImportError:
                debug_log.append("Firestore: ImportError (firebase-admin not installed)")
            except Exception as e:
                debug_log.append(f"Firestore error: {e}")
                import traceback; traceback.print_exc()

            # Debug mode: show diagnostics
            if debug_mode:
                diag = "\n".join(f"• {l}" for l in debug_log)
                await message.channel.send(
                    f"🔧 **Alert Debug** (symbol={symbol or 'ALL'}):\n{diag}\n"
                    f"**Total found: {len(all_alerts)}**"
                )
                # Also show raw alert data if any
                if all_alerts:
                    for a in all_alerts[:5]:
                        await message.channel.send(f"```\n{a}\n```")
                return

            if not all_alerts:
                await message.channel.send(f"📭 No active alerts{f' for {symbol}' if symbol else ''}.")
                return

            description = ""
            for a in all_alerts[:20]:
                direction = "⬆️" if a.get("direction") == "above" else "⬇️"
                description += f"{direction} **{a.get('symbol')}** ${a.get('level', 0):.2f} — {a.get('action', '?')}"
                if a.get("note"):
                    description += f" *{a['note']}*"
                description += "\n"

            embed = discord.Embed(
                title=f"🔔 Active Alerts ({len(all_alerts)})",
                description=description,
                color=0x00D9FF,
                timestamp=datetime.utcnow()
            )
            await message.channel.send(embed=embed)

        except ImportError:
            await message.channel.send("⚠️ httpx not available — alerts require httpx package")
        except Exception as e:
            await message.channel.send(f"⚠️ Error fetching alerts: {str(e)[:200]}")

    async def cmd_stats(self, message: discord.Message, args: str):
        try:
            from firestore_store import get_firestore
            fs = get_firestore()
            stats = fs.get_trade_stats(user_id="default")

            if stats.get("total", 0) == 0:
                await message.channel.send("📊 No closed trades yet.")
                return

            wr = stats.get("win_rate", 0)
            color = 0x00FF88 if wr >= 60 else 0xFFC800 if wr >= 50 else 0xFF4444
            wr_emoji = "🔥" if wr >= 60 else "✅" if wr >= 50 else "⚠️"

            embed = discord.Embed(title="📊 Trade Performance", color=color, timestamp=datetime.utcnow())
            embed.add_field(name="Total Trades", value=str(stats.get("total", 0)), inline=True)
            embed.add_field(name="Wins", value=str(stats.get("wins", 0)), inline=True)
            embed.add_field(name="Losses", value=str(stats.get("losses", 0)), inline=True)
            embed.add_field(name=f"{wr_emoji} Win Rate", value=f"**{wr:.1f}%**", inline=True)
            embed.add_field(name="Total P&L", value=f"**${stats.get('total_pnl', 0):.2f}**", inline=True)
            embed.set_footer(text="SEF Trading Terminal")
            await message.channel.send(embed=embed)

        except ImportError:
            await message.channel.send("⚠️ Firestore not available — stats require Firebase setup")
        except Exception as e:
            await message.channel.send(f"⚠️ Error: {str(e)[:200]}")

    async def cmd_brief(self, message: discord.Message, args: str):
        indices = ["SPY", "QQQ", "IWM", "DIA"]
        description = ""

        await message.channel.send("📡 Fetching market data...")

        for symbol in indices:
            data = await _get_price_data(symbol)
            if data:
                emoji = "🟢" if data["change"] >= 0 else "🔴"
                sign = "+" if data["change"] >= 0 else ""
                description += (
                    f"{emoji} **{data['symbol']}** ${data['price']:.2f} "
                    f"({sign}{data['change_pct']:.2f}%)\n"
                )

        embed = discord.Embed(
            title="📡 Market Brief",
            description=description or "No data available.",
            color=0x00D9FF,
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text=f"SEF Trading Terminal — {datetime.now().strftime('%I:%M %p ET')}")
        await message.channel.send(embed=embed)

    async def cmd_scan(self, message: discord.Message, args: str):
        if not args:
            await message.channel.send("⚠️ Specify a symbol: `!scan NVDA`")
            return

        raw = args.strip().upper()
        for filler in ["FOR ", "ON ", "OF "]:
            if raw.startswith(filler):
                raw = raw[len(filler):]
        symbol = raw.strip()
        await message.channel.send(f"🔍 Scanning **{symbol}**...")

        # Get price data + any available analysis
        price_data = await _get_price_data(symbol)
        if not price_data:
            await message.channel.send(f"⚠️ Could not get data for {symbol}")
            return

        price = price_data["price"]
        change = price_data["change"]
        change_pct = price_data["change_pct"]
        color = 0x00FF88 if change >= 0 else 0xFF4444
        sign = "+" if change >= 0 else ""

        embed = discord.Embed(
            title=f"🔍 Scanner — {symbol}",
            color=color,
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="Price", value=f"${price:.2f}", inline=True)
        embed.add_field(name="Change", value=f"{sign}{change:.2f} ({sign}{change_pct:.2f}%)", inline=True)

        # Try to get scanner analysis
        try:
            from finnhub_scanner_v2 import FinnhubScanner
            api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("FINNHUB_API_KEY")
            if api_key:
                scanner = FinnhubScanner(api_key)
                result = scanner.analyze(symbol, "2HR")
                if result:
                    embed.add_field(name="Signal", value=f"{result.signal_emoji} {result.signal}", inline=True)
                    embed.add_field(name="Score", value=str(getattr(result, 'score', 'N/A')), inline=True)
                    if hasattr(result, 'vah') and result.vah:
                        embed.add_field(name="VAH", value=f"${result.vah:.2f}", inline=True)
                    if hasattr(result, 'poc') and result.poc:
                        embed.add_field(name="POC", value=f"${result.poc:.2f}", inline=True)
                    if hasattr(result, 'val') and result.val:
                        embed.add_field(name="VAL", value=f"${result.val:.2f}", inline=True)
        except Exception as e:
            embed.add_field(name="Scanner", value=f"Scanner unavailable: {str(e)[:100]}", inline=False)

        embed.set_footer(text="SEF Trading Terminal — Full analysis at localhost:8000")
        await message.channel.send(embed=embed)

    async def cmd_setup(self, message: discord.Message, args: str):
        if not args:
            await message.channel.send("⚠️ Specify a symbol: `!setup TSLA`")
            return

        raw = args.strip().upper()
        for filler in ["FOR ", "ON ", "OF "]:
            if raw.startswith(filler):
                raw = raw[len(filler):]
        symbol = raw.strip()
        # Reuse scan for now — setup = scan + alerts
        await self.cmd_scan(message, symbol)

        # Also show alerts for the symbol
        try:
            from firestore_store import get_firestore
            fs = get_firestore()
            alerts = fs.get_alerts(user_id="default", symbol=symbol)
            if alerts:
                alert_text = ""
                for a in alerts[:5]:
                    d = "⬆️" if a.get("direction") == "above" else "⬇️"
                    alert_text += f"{d} ${a.get('level', 0):.2f} — {a.get('action')}\n"

                embed = discord.Embed(
                    title=f"🔔 Alerts for {symbol}",
                    description=alert_text,
                    color=0x00D9FF
                )
                await message.channel.send(embed=embed)
        except Exception:
            pass  # Alerts are optional

    async def cmd_queue(self, message: discord.Message, args: str):
        try:
            from task_queue import get_task_queue
            queue = get_task_queue()
            stats = queue.get_queue_stats()

            if not stats:
                await message.channel.send("📋 Task queue empty or unavailable.")
                return

            embed = discord.Embed(title="📋 Task Queue Status", color=0x888888, timestamp=datetime.utcnow())
            embed.add_field(name="⏳ Pending", value=str(stats.get("pending", 0)), inline=True)
            embed.add_field(name="⚙️ Processing", value=str(stats.get("processing", 0)), inline=True)
            embed.add_field(name="✅ Completed", value=str(stats.get("completed", 0)), inline=True)
            embed.add_field(name="❌ Failed", value=str(stats.get("failed", 0)), inline=True)
            await message.channel.send(embed=embed)
        except Exception as e:
            await message.channel.send(f"⚠️ Queue error: {str(e)[:200]}")

    async def cmd_autoscan(self, message: discord.Message, args: str):
        """Auto-scanner control: status, trigger, start, stop"""
        try:
            from auto_scanner import get_auto_scanner
            scanner = get_auto_scanner()

            if not scanner:
                await message.channel.send("⚠️ Auto-scanner not initialized.")
                return

            action = args.strip().lower() if args else "status"

            if action == "trigger" or action == "run" or action == "now":
                await message.channel.send("🔍 Triggering manual scan...")
                result = await scanner.run_now()
                await message.channel.send(
                    f"✅ Scan complete — "
                    f"Squeezes: {len(result.squeeze_setups)} | "
                    f"Setups: {len(result.dual_setups)} | "
                    f"Capitulations: {len(result.capitulation_signals)}"
                )

            elif action == "start":
                scanner.start()
                await message.channel.send("▶️ Auto-scanner started (30-min interval)")

            elif action == "stop":
                scanner.stop()
                await message.channel.send("⏹️ Auto-scanner stopped")

            else:  # status
                status = scanner.status
                running_emoji = "🟢" if status["running"] else "🔴"
                embed = discord.Embed(
                    title="🔄 Auto-Scanner Status",
                    color=0x00D9FF if status["running"] else 0x888888,
                    timestamp=datetime.utcnow()
                )
                embed.add_field(name="Status", value=f"{running_emoji} {'Running' if status['running'] else 'Stopped'}", inline=True)
                embed.add_field(name="Interval", value=f"{status['interval_minutes']} min", inline=True)
                embed.add_field(name="Cycles", value=str(status['cycle_count']), inline=True)
                embed.add_field(name="Last Scan", value=status['last_scan'] or "Never", inline=False)
                embed.add_field(name="Squeezes", value=str(status['last_squeeze_count']), inline=True)
                embed.add_field(name="Setups", value=str(status['last_dual_count']), inline=True)
                embed.add_field(name="Extremes", value=f"{status['last_cap_count']}C / {status['last_euph_count']}E", inline=True)
                await message.channel.send(embed=embed)

        except ImportError:
            await message.channel.send("⚠️ Auto-scanner module not available.")
        except Exception as e:
            await message.channel.send(f"⚠️ Auto-scan error: {str(e)[:200]}")


# =============================================================================
# PRICE HELPER (async, uses yfinance)
# =============================================================================

async def _get_price_data(symbol: str) -> Optional[Dict]:
    """Get price data — runs yfinance in executor to avoid blocking"""
    import yfinance as yf

    def _fetch():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if hist.empty:
                return None

            price = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else hist["Open"].iloc[0]
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0

            return {
                "symbol": symbol,
                "price": float(price),
                "change": float(change),
                "change_pct": float(change_pct)
            }
        except Exception as e:
            print(f"⚠️ Price fetch error for {symbol}: {e}")
            return None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)


# =============================================================================
# BOT LIFECYCLE
# =============================================================================

_bot_instance: Optional[SEFDiscordBot] = None
_bot_thread: Optional[threading.Thread] = None


def get_bot() -> Optional[SEFDiscordBot]:
    """Get the global bot instance"""
    return _bot_instance


async def start_discord_bot():
    """
    Start the Discord bot in a background thread.
    Called from unified_server.py startup event.
    """
    global _bot_instance, _bot_thread

    if not DISCORD_BOT_TOKEN:
        print("⚠️ DISCORD_BOT_TOKEN not set — two-way Discord bot disabled")
        print("   Webhook-only mode active (one-way: server → Discord)")
        print("   To enable two-way: set DISCORD_BOT_TOKEN environment variable")
        return

    _bot_instance = SEFDiscordBot()

    def _run_bot():
        """Run the bot in its own event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_bot_instance.start(DISCORD_BOT_TOKEN))
        except Exception as e:
            print(f"❌ Discord bot error: {e}")
        finally:
            loop.close()

    _bot_thread = threading.Thread(target=_run_bot, daemon=True, name="discord-bot")
    _bot_thread.start()
    print("🚀 Discord bot listener started (two-way mode)")


async def stop_discord_bot():
    """Stop the Discord bot gracefully"""
    global _bot_instance
    if _bot_instance and not _bot_instance.is_closed():
        await _bot_instance.close()
        print("🛑 Discord bot stopped")
