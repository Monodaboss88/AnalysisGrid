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
Version: 1.0.3
"""

import os
import asyncio
import json
import threading
import discord
from collections import OrderedDict
from discord import Intents
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict

_ET = ZoneInfo("America/New_York")

def _now_et() -> datetime:
    return datetime.now(_ET)


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
        self._processed_messages: OrderedDict = OrderedDict()  # ordered dedup guard
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
            "options": self.cmd_options,
            "opts": self.cmd_options,
            "flow": self.cmd_options,
            "fullscan": self.cmd_fullscan,
            "unusual": self.cmd_unusual,
            "warroom": self.cmd_warroom,
            "war": self.cmd_warroom,
            "wr": self.cmd_warroom,
            "odds": self.cmd_odds,
            "prob": self.cmd_odds,
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

        # Channel allowlist — if configured, only respond in those channels
        if ALLOWED_CHANNEL_IDS and ALLOWED_CHANNEL_IDS != [""]:
            if str(message.channel.id) not in ALLOWED_CHANNEL_IDS:
                return

        # Dedup guard — prevent processing the same message twice
        # (reconnection can cause Discord gateway to redeliver messages)
        if message.id in self._processed_messages:
            return
        self._processed_messages[message.id] = None  # OrderedDict preserves insertion order
        # Keep bounded (last 200 messages) — prune oldest entries
        while len(self._processed_messages) > 200:
            self._processed_messages.popitem(last=False)

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
                "war_room": self.cmd_warroom,
                "probability_check": self.cmd_odds,
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
                "`!options NVDA` — Options flow for a symbol\n"
                "`!fullscan` — Full universe scan + options → Discord\n"
                "`!warroom` — Pre-market War Room (extension DNA)\n"
                "`!odds NVDA` — Historical probability context\n"
                "`!unusual` — Unusual options activity alerts\n"
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
            timestamp=datetime.now(timezone.utc)
        )
        await message.channel.send(embed=embed)

    async def cmd_alerts(self, message: discord.Message, args: str):
        try:
            import httpx
            # Check for debug flag FIRST
            raw = args.strip()
            debug_mode = raw.lower().startswith("debug")
            if debug_mode:
                raw = raw[5:].strip()

            # Strip filler words
            raw = raw.upper()
            for filler in ["FOR ", "ON ", "OF ", "CHECK "]:
                if raw.startswith(filler):
                    raw = raw[len(filler):]
            symbol = raw.strip() if raw else None

            port = os.environ.get("PORT", "8000")
            base = f"http://localhost:{port}"
            debug_log = [f"v4-REST | symbol={symbol or 'ALL'} | debug={debug_mode}"]

            all_alerts = []

            # 1) Try Firestore REST API (no service account needed)
            #    Wrapped in run_in_executor to avoid blocking the event loop
            #    (sync HTTP calls block heartbeats → reconnect → message redelivery)
            try:
                from firestore_rest import search_all_alerts, is_available as rest_available, get_status as rest_status
                debug_log.append(f"REST client: available={rest_available()}")
                
                if rest_available():
                    loop = asyncio.get_running_loop()
                    rest_alerts = await loop.run_in_executor(None, search_all_alerts, symbol)
                    debug_log.append(f"REST: found {len(rest_alerts)} alerts")
                    for a in rest_alerts:
                        if not a.get('triggered', False):
                            a["_source"] = "firestore-rest"
                            all_alerts.append(a)
                else:
                    status = rest_status()
                    debug_log.append(f"REST status: {status}")
            except ImportError:
                debug_log.append("REST client: not available (import error)")
            except Exception as e:
                debug_log.append(f"REST error: {type(e).__name__}: {e}")

            # 2) Try firebase-admin SDK (if FIREBASE_SERVICE_ACCOUNT is set)
            if not all_alerts:
                try:
                    from firestore_store import get_firestore
                    fs = get_firestore()
                    fs_available = fs.is_available() and fs.db is not None
                    debug_log.append(f"Admin SDK: available={fs_available}")

                    if fs_available:
                        seen_ids = set()
                        users_ref = fs.db.collection('users')
                        user_count = 0
                        fs_alert_count = 0

                        for user_doc_ref in users_ref.list_documents():
                            user_count += 1
                            uid = user_doc_ref.id
                            alerts_ref = users_ref.document(uid).collection('alerts')
                            q = alerts_ref
                            if symbol:
                                q = q.where('symbol', '==', symbol)
                            for doc in q.stream():
                                fs_alert_count += 1
                                alert = doc.to_dict()
                                alert['id'] = doc.id
                                if doc.id not in seen_ids and not alert.get('triggered', False):
                                    alert["_source"] = "firestore-admin"
                                    all_alerts.append(alert)
                                    seen_ids.add(doc.id)

                        debug_log.append(f"Admin SDK: {user_count} users, {fs_alert_count} alerts")

                except ImportError:
                    debug_log.append("Admin SDK: not installed")
                except Exception as e:
                    debug_log.append(f"Admin SDK error: {type(e).__name__}: {e}")

            # 3) Fallback: local chart_system alerts
            if not all_alerts:
                try:
                    params = {}
                    if symbol:
                        params["symbol"] = symbol
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(f"{base}/api/alerts", params=params)
                        debug_log.append(f"Local API: status={resp.status_code}")
                        if resp.status_code == 200:
                            data = resp.json()
                            debug_log.append(f"Local: {data.get('count', 0)} alerts")
                            for a in data.get("alerts", []):
                                a["_source"] = "local"
                                all_alerts.append(a)
                except Exception as e:
                    debug_log.append(f"Local API error: {e}")

            # Debug mode: show diagnostics in Discord
            if debug_mode:
                diag = "\n".join(f"• {l}" for l in debug_log)
                await message.channel.send(
                    f"🔧 **Alert Debug**:\n{diag}\n"
                    f"**Total found: {len(all_alerts)}**"
                )
                if all_alerts:
                    for a in all_alerts[:3]:
                        await message.channel.send(f"```\n{json.dumps(a, indent=2, default=str)[:1800]}\n```")
                return

            if not all_alerts:
                await message.channel.send(f"📭 No active alerts{f' for {symbol}' if symbol else ''}.")
                return

            description = ""
            for a in all_alerts[:20]:
                direction = "⬆️" if a.get("direction") == "above" else "⬇️"
                sym = a.get("symbol", "?")
                level = a.get("level", 0)
                try:
                    level = float(level)
                    level_str = f"${level:.2f}"
                except (ValueError, TypeError):
                    level_str = str(level)
                action = a.get("action", "?")
                description += f"{direction} **{sym}** {level_str} — {action}"
                if a.get("note"):
                    description += f" *{a['note']}*"
                description += "\n"

            embed = discord.Embed(
                title=f"🔔 Active Alerts ({len(all_alerts)})",
                description=description,
                color=0x00D9FF,
                timestamp=datetime.now(timezone.utc)
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

            embed = discord.Embed(title="📊 Trade Performance", color=color, timestamp=datetime.now(timezone.utc))
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
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_footer(text=f"SEF Trading Terminal — {_now_et().strftime('%I:%M %p ET')}")
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
            timestamp=datetime.now(timezone.utc)
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

        embed.set_footer(text=f"SEF Trading Terminal — {_now_et().strftime('%I:%M %p ET')}")
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

    async def cmd_options(self, message: discord.Message, args: str):
        """Options flow scan for a symbol: !options NVDA"""
        if not args:
            await message.channel.send("⚠️ Specify a symbol: `!options NVDA`")
            return

        raw = args.strip().upper()
        for filler in ["FOR ", "ON ", "OF "]:
            if raw.startswith(filler):
                raw = raw[len(filler):]
        symbol = raw.strip()
        await message.channel.send(f"📊 Pulling options flow for **{symbol}**...")

        try:
            loop = asyncio.get_running_loop()
            from options_flow_scanner import _scan_single as opts_scan
            r = await loop.run_in_executor(None, lambda: opts_scan(symbol, 30, 0.10))

            if r.get("error"):
                await message.channel.send(f"⚠️ {symbol}: {r['error']}")
                return

            price = r.get("price", 0)
            flow = r.get("flowScore", 0)
            sentiment = r.get("sentiment", "NEUTRAL")
            iv_pct = r.get("avgIVPct")
            unusual = r.get("unusualCount", 0)
            mp = r.get("maxPain")
            em = r.get("expectedMovePct")
            total_vol = r.get("totalVolume", 0)
            pc_vol = r.get("pcVolumeRatio", 0)
            pc_oi = r.get("pcOIRatio", 0)

            sent_map = {"BULLISH": 0x00FF88, "LEAN BULLISH": 0x00FF88,
                        "BEARISH": 0xFF4444, "LEAN BEARISH": 0xFF4444}
            color = sent_map.get(sentiment, 0x00D9FF)
            sent_emoji = {"BULLISH": "🟢", "LEAN BULLISH": "🟢",
                          "BEARISH": "🔴", "LEAN BEARISH": "🔴"}.get(sentiment, "⚪")

            embed = discord.Embed(
                title=f"📊 Options Flow — {symbol} ${price:.2f}",
                color=color,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name="Flow Score", value=f"**{flow}**/100", inline=True)
            embed.add_field(name="Sentiment", value=f"{sent_emoji} {sentiment}", inline=True)
            embed.add_field(name="IV", value=f"{iv_pct:.1f}% ({r.get('ivLevel','?')})" if iv_pct else "N/A", inline=True)
            embed.add_field(name="Opt Volume", value=f"{total_vol:,}", inline=True)
            embed.add_field(name="P/C Vol", value=f"{pc_vol}", inline=True)
            embed.add_field(name="P/C OI", value=f"{pc_oi}", inline=True)
            embed.add_field(name="Unusual", value=f"**{unusual}** contracts", inline=True)
            embed.add_field(name="Max Pain", value=f"${mp:.2f}" if mp else "?", inline=True)
            embed.add_field(name="Exp Move", value=f"{em:.1f}%" if em else "N/A", inline=True)

            # Top unusual contracts
            unusual_list = r.get("unusualContracts", [])[:5]
            if unusual_list:
                lines = []
                for u in unusual_list:
                    iv_tag = f" IV:{u['iv']*100:.0f}%" if u.get('iv') else ""
                    lines.append(
                        f"`${u['strike']}` {u['type'].upper()} {u['expiration']} — "
                        f"Vol:{u['volume']:,} OI:{u['oi']:,} (**{u['volOiRatio']}x**){iv_tag}"
                    )
                embed.add_field(name="🔥 Unusual Activity", value="\n".join(lines), inline=False)

            # OI walls
            oi_walls = r.get("oiWalls", [])[:4]
            if oi_walls:
                lines = []
                for w in oi_walls:
                    tag = "↑" if w['strike'] > price else "↓"
                    lines.append(f"${w['strike']} {tag} — C:{w['call_oi']:,} P:{w['put_oi']:,}")
                embed.add_field(name="🧱 OI Walls", value="\n".join(lines), inline=False)

            embed.set_footer(text=f"SEF Trading Terminal — {_now_et().strftime('%I:%M %p ET')}")
            await message.channel.send(embed=embed)

        except Exception as e:
            await message.channel.send(f"⚠️ Options error: {str(e)[:200]}")

    async def cmd_warroom(self, message: discord.Message, args: str):
        """Pre-market War Room — extension DNA scanner: !warroom [preset|tickers]"""
        from war_room import async_run_war_room, PRESETS as WR_PRESETS

        raw = args.strip().upper() if args else ""

        # Determine tickers: preset name OR comma/space-separated symbols
        if not raw or raw in ("HELP", "?"):
            preset_list = ", ".join(f"`{k}`" for k in WR_PRESETS)
            await message.channel.send(
                f"⚔️ **War Room — Usage**\n"
                f"`!warroom mag7` — Run a preset watchlist\n"
                f"`!warroom NVDA TSLA AMD` — Custom tickers\n\n"
                f"**Presets:** {preset_list}"
            )
            return

        # Check if it's a preset
        preset_key = raw.lower().replace(" ", "_")
        if preset_key in WR_PRESETS:
            tickers = WR_PRESETS[preset_key]
            label = preset_key.replace("_", " ").title()
        else:
            # Parse as ticker list (comma or space separated)
            tickers = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
            label = "Custom"

        if not tickers:
            await message.channel.send("⚠️ No tickers provided. Try `!warroom mag7` or `!warroom NVDA TSLA`")
            return

        await message.channel.send(f"⚔️ Running War Room **{label}** ({len(tickers)} tickers)...\nAnalyzing 30-60d of intraday extension DNA — this may take a moment.")

        try:
            data = await async_run_war_room(tickers)
            results = data.get("results", [])
            meta = data.get("meta", {})
            errors = data.get("errors", [])

            if not results:
                err_msg = ", ".join(e["ticker"] for e in errors[:5]) if errors else "unknown"
                await message.channel.send(f"⚠️ War Room returned no results. Failed: {err_msg}")
                return

            # --- Summary embed ---
            embed = discord.Embed(
                title=f"⚔️ Daily War Room — {label}",
                color=0xEF4444,
                timestamp=datetime.now(timezone.utc)
            )
            embed.description = (
                f"**{meta.get('scanned', 0)}** scanned · "
                f"**{meta.get('hot_count', 0)}** HOT · "
                f"**{meta.get('cold_count', 0)}** COLD · "
                f"SPY avg ext: **{meta.get('spy_avg_up', 'N/A')}%**"
            )

            # --- Top results (max 10 to fit embed) ---
            for r in results[:10]:
                ticker = r.get("ticker", "?")
                conviction = r.get("fade_conviction", 0)
                avg_up = r.get("avg_up", 0)
                avg_down = r.get("avg_down", 0)
                exhaustion = r.get("exhaustion", 0)
                regime = r.get("regime", {}).get("ext_regime", "NORMAL")
                signals = r.get("signals", [])
                peak_hour = r.get("peak_hour", 0)
                peak_h = int(peak_hour)
                peak_m = int((peak_hour - peak_h) * 60)
                thin_top = r.get("thin_top_pct", 0)
                close_pos = r.get("avg_close_pos", 0)

                # Conviction bar
                bar_fill = int(conviction / 10)
                bar = "█" * bar_fill + "░" * (10 - bar_fill)

                regime_emoji = "🔥" if regime == "HOT" else ("❄️" if regime == "COLD" else "⚖️")
                signal_tags = " ".join(f"`{s}`" for s in signals[:5]) if signals else "—"

                field_val = (
                    f"{regime_emoji} **{regime}** · Fade: `{bar}` {conviction}%\n"
                    f"Avg Ext: ↑{avg_up:.2f}% ↓{avg_down:.2f}% · Exhaust: {exhaustion:.2f}%\n"
                    f"Peak: {peak_h}:{peak_m:02d} · ThinTop: {thin_top:.0f}% · ClosePos: {close_pos:.0f}%\n"
                    f"{signal_tags}"
                )
                embed.add_field(name=f"**{ticker}**", value=field_val, inline=False)

            if errors:
                err_tickers = ", ".join(e["ticker"] for e in errors[:5])
                embed.set_footer(text=f"⚠️ Failed: {err_tickers} · {_now_et().strftime('%I:%M %p ET')}")
            else:
                embed.set_footer(text=f"SEF Trading Terminal — {_now_et().strftime('%I:%M %p ET')}")

            await message.channel.send(embed=embed)

        except Exception as e:
            await message.channel.send(f"⚠️ War Room error: {str(e)[:300]}")

    async def cmd_odds(self, message: discord.Message, args: str):
        """Historical probability context: !odds NVDA"""
        if not args:
            await message.channel.send("⚠️ Specify a symbol: `!odds SPY`")
            return

        symbol = args.strip().upper().split()[0]
        await message.channel.send(f"🎲 Pulling historical odds for **{symbol}**...")

        try:
            import sys, os
            tool_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polygon_signal_tool")
            if tool_dir not in sys.path:
                sys.path.insert(0, tool_dir)

            from signal_endpoints import _run_analysis
            loop = asyncio.get_running_loop()
            analysis = await loop.run_in_executor(None, lambda: _run_analysis(symbol, 365))

            if not analysis:
                await message.channel.send(f"⚠️ No data for {symbol}. Check Polygon API key or ticker.")
                return

            sig = analysis["signal"]
            stats = analysis["all_stats"]
            straddle = analysis["straddle"]
            today = sig["today"]

            # Pick scenario based on today's condition
            if today["color"] == "RED":
                rs = today["rstreak"]
                call_key = f"call_red{rs}" if rs >= 2 and f"call_red{rs}" in stats else "call_red"
                put_key = f"put_red{rs}" if rs >= 2 and f"put_red{rs}" in stats else "put_red"
                streak_label = f"{rs} red day{'s' if rs > 1 else ''}"
            else:
                gs = today["gstreak"]
                call_key = f"call_green{gs}" if gs >= 2 and f"call_green{gs}" in stats else "call_green"
                put_key = f"put_green{gs}" if gs >= 2 and f"put_green{gs}" in stats else "put_green"
                streak_label = f"{gs} green day{'s' if gs > 1 else ''}"

            cs = stats.get(call_key, stats.get("call_all", {}))
            ps = stats.get(put_key, stats.get("put_all", {}))

            embed = discord.Embed(
                title=f"🎲 {symbol} — Historical Odds",
                color=0x6366F1,
                timestamp=datetime.now(timezone.utc)
            )
            embed.description = (
                f"After **{streak_label}** · {analysis['n']} days analyzed\n"
                f"Last close: **${today['close']:.2f}** · "
                f"{'🟢' if today['color'] == 'GREEN' else '🔴'} {today['color']}"
            )

            # Call odds
            if cs:
                c_hit1 = cs.get("rate_1d", 0) * 100
                c_hit3 = cs.get("rate_3d", 0) * 100
                c_best = cs.get("avg_best_pct_3d", 0)
                c_win = cs.get("close_win_1d", 0) * 100
                c_n = cs.get("count", 0)
                embed.add_field(
                    name="📈 Call Odds",
                    value=(
                        f"1D scalp: **{c_hit1:.1f}%** ({c_n} trades)\n"
                        f"3D scalp: **{c_hit3:.1f}%**\n"
                        f"Avg best 3D: **{c_best:.2f}%**\n"
                        f"Close win 1D: {c_win:.1f}%"
                    ),
                    inline=True
                )

            # Put odds
            if ps:
                p_hit1 = ps.get("rate_1d", 0) * 100
                p_hit3 = ps.get("rate_3d", 0) * 100
                p_best = ps.get("avg_best_pct_3d", 0)
                p_win = ps.get("close_win_1d", 0) * 100
                p_n = ps.get("count", 0)
                embed.add_field(
                    name="📉 Put Odds",
                    value=(
                        f"1D scalp: **{p_hit1:.1f}%** ({p_n} trades)\n"
                        f"3D scalp: **{p_hit3:.1f}%**\n"
                        f"Avg best 3D: **{p_best:.2f}%**\n"
                        f"Close win 1D: {p_win:.1f}%"
                    ),
                    inline=True
                )

            # Expected range + straddle
            up1 = sig.get("expected_upside", 0)
            dn1 = sig.get("expected_downside", 0)
            up3 = sig.get("expected_upside_3d", 0)
            dn3 = sig.get("expected_downside_3d", 0)
            strad = straddle.get("at_least_one_rate", 0) * 100

            embed.add_field(
                name="📊 Expected Range",
                value=(
                    f"1D: +{up1:.2f}% / -{dn1:.2f}%\n"
                    f"3D: +{up3:.2f}% / -{dn3:.2f}%\n"
                    f"Straddle hit: **{strad:.1f}%**"
                ),
                inline=False
            )

            # ── Predictability Map ──
            cl = analysis.get("close_location", {})
            ga = analysis.get("gap_analysis", {})
            vr = analysis.get("vol_regime", {})
            ext = analysis.get("extension", {})
            opx = analysis.get("opex", {})

            pred_lines = []
            # Close Location
            clv = cl.get("today_clv", 0)
            clv_emoji = "🟢" if clv > 70 else ("🔴" if clv < 30 else "🟡")
            pred_lines.append(f"{clv_emoji} Close position: **{clv}%** ({cl.get('trend_cluster', '-')})")
            # Gap
            gdir = ga.get("today_gap_direction", "FLAT")
            gpct = abs(ga.get("today_gap_pct", 0))
            gfill = "✅" if ga.get("today_gap_filled") else "❌"
            gu_fill = ga.get("gap_ups", {}).get("fill_rate", 0)
            gd_fill = ga.get("gap_downs", {}).get("fill_rate", 0)
            pred_lines.append(f"Gap: **{gdir} {gpct:.2f}%** {gfill} | Fill rates: ↑{gu_fill}% ↓{gd_fill}%")
            # Regime
            regime = vr.get("regime", "-")
            r_emoji = {"SQUEEZE": "⏸️", "STABLE": "🟢", "EXPANDING": "🟡", "EXTREME": "🔴"}.get(regime, "⚪")
            pred_lines.append(f"{r_emoji} Regime: **{regime}** (ATR ratio: {vr.get('atr_ratio', 0)})")
            # Extension
            zscore = ext.get("zscore", 0)
            z_emoji = "🔴" if abs(zscore) >= 2 else ("🟡" if abs(zscore) >= 1.5 else "🟢")
            pred_lines.append(f"{z_emoji} Z-Score: **{zscore:.2f}** — {ext.get('extension_pct', 0)}% of avg range")
            if ext.get("revert_after_extreme_rate", 0) > 0:
                pred_lines.append(f"   Revert after Z>2: **{ext['revert_after_extreme_rate']}%**")
            # OpEx
            if opx.get("today_is_opex"):
                pred_lines.append(f"⚡ **OpEx Day** — Pin rate: {opx.get('opex', {}).get('pin_rate', 0)}%")

            embed.add_field(
                name="🗺️ Predictability Map",
                value="\n".join(pred_lines),
                inline=False
            )

            # VWAP Magnet (optional — may not load if War Room fails)
            try:
                from signal_endpoints import _get_vwap_magnet
                vm = await loop.run_in_executor(None, lambda: _get_vwap_magnet(symbol))
                if vm and vm.get("vwap_revert_rate", 0) > 0:
                    embed.add_field(
                        name="🧲 VWAP Magnet",
                        value=(
                            f"Revert to VWAP: **{vm['vwap_revert_rate']}%** of days\n"
                            f"Avg max distance: **{vm['avg_max_vwap_dist']}%**\n"
                            f"VWAP crosses: **{vm['avg_vwap_crosses']}/day**\n"
                            f"Snap-back to: **{vm['avg_min_dist_after']}%** from VWAP"
                        ),
                        inline=False
                    )
            except Exception:
                pass  # VWAP magnet is optional

            embed.set_footer(text=f"Historical only · Not financial advice · Scenario: {call_key}/{put_key}")
            await message.channel.send(embed=embed)

        except Exception as e:
            await message.channel.send(f"⚠️ Odds error: {str(e)[:300]}")

    async def cmd_fullscan(self, message: discord.Message, args: str):
        """Full universe scan + options → Discord: !fullscan"""
        await message.channel.send("🔍 Running full 72-symbol scan + options overlay...\nThis takes ~2-3 minutes.")

        try:
            loop = asyncio.get_running_loop()
            from full_scan_discord import run_full_scan
            result = await loop.run_in_executor(
                None, lambda: run_full_scan(timeframe="2HR", include_options=True, webhook=True, quiet=True)
            )

            bulls = len(result.get("bullish", []))
            bears = len(result.get("bearish", []))
            yellows = len(result.get("yellow", []))
            opts_count = len(result.get("options", {}))
            best = result.get("best_setup", "none")

            await message.channel.send(
                f"✅ **Full Scan Complete**\n"
                f"🟢 {bulls} longs | 🔴 {bears} shorts | 🟡 {yellows} yellows\n"
                f"📊 Options data on {opts_count} setups\n"
                f"🏆 Best setup: **{best}**\n"
                f"Results posted to channel ↑"
            )
        except Exception as e:
            await message.channel.send(f"⚠️ Full scan error: {str(e)[:300]}")

    async def cmd_unusual(self, message: discord.Message, args: str):
        """Unusual options activity: !unusual or !unusual AAPL,NVDA,TSLA"""
        custom_symbols = None
        if args:
            custom_symbols = [s.strip().upper() for s in args.replace(",", " ").split() if s.strip()]

        label = f"{len(custom_symbols)} symbols" if custom_symbols else "top 15 liquid names"
        await message.channel.send(f"⚡ Checking unusual options activity ({label})...")

        try:
            loop = asyncio.get_running_loop()
            from full_scan_discord import check_unusual_activity, push_unusual_alerts
            alerts = await loop.run_in_executor(
                None, lambda: check_unusual_activity(symbols=custom_symbols)
            )

            if not alerts:
                await message.channel.send("📭 No unusual activity detected right now.")
                return

            # Push to webhook
            await loop.run_in_executor(None, lambda: push_unusual_alerts(alerts, quiet=True))

            summary = "\n".join([
                f"⚡ **{a['symbol']}** — Flow:{a['flowScore']} | Unusual:{a['unusualCount']} | {a['sentiment']}"
                for a in alerts[:8]
            ])
            await message.channel.send(
                f"**{len(alerts)} tickers with unusual activity:**\n{summary}\n"
                f"Full details posted ↑"
            )
        except Exception as e:
            await message.channel.send(f"⚠️ Unusual scan error: {str(e)[:200]}")

    async def cmd_queue(self, message: discord.Message, args: str):
        try:
            from task_queue import get_task_queue
            queue = get_task_queue()
            stats = queue.get_queue_stats()

            if not stats:
                await message.channel.send("📋 Task queue empty or unavailable.")
                return

            embed = discord.Embed(title="📋 Task Queue Status", color=0x888888, timestamp=datetime.now(timezone.utc))
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
                    timestamp=datetime.now(timezone.utc)
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

    loop = asyncio.get_running_loop()
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
    """Stop the Discord bot gracefully (cross-thread safe)"""
    global _bot_instance
    if _bot_instance and not _bot_instance.is_closed():
        # Bot runs on a separate thread's event loop — schedule close there
        bot_loop = _bot_instance.loop
        if bot_loop and bot_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_bot_instance.close(), bot_loop)
            try:
                future.result(timeout=10)  # wait up to 10s for graceful shutdown
            except Exception as e:
                print(f"⚠️ Discord bot stop error: {e}")
        else:
            await _bot_instance.close()
        print("🛑 Discord bot stopped")
