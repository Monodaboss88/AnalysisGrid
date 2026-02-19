"""
Discord Endpoint Router
=======================
FastAPI router that handles:
  - Alert delivery pipeline (scanner → Discord)
  - API endpoints for triggering Discord notifications
  - Task queue worker callbacks
  - Background alert checker

Mount this router in unified_server.py:
  from discord_endpoints import discord_router, setup_discord
  app.include_router(discord_router, prefix="/discord")
  # In startup: await setup_discord(app)

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from discord_bot import get_discord, DiscordClient
from task_queue import get_task_queue, Task, TaskType, TaskPriority, QUICK_TASKS
from hybrid_router import get_router

# Try to import existing modules
try:
    from firestore_store import get_firestore
    firestore_available = True
except ImportError:
    firestore_available = False

try:
    import yfinance as yf
    yf_available = True
except ImportError:
    yf_available = False


# =============================================================================
# ROUTER
# =============================================================================

discord_router = APIRouter(tags=["discord"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class CommandRequest(BaseModel):
    command: str            # e.g., "alerts", "stats", "price SPY", "scan NVDA"

class AlertNotification(BaseModel):
    symbol: str
    level: float
    direction: str          # "above" or "below"
    action: str             # "LONG", "SHORT", "ALERT", "EXIT"
    note: str = ""
    price: float = 0


# =============================================================================
# API ENDPOINTS
# =============================================================================

@discord_router.post("/command")
async def handle_command(req: CommandRequest, background_tasks: BackgroundTasks):
    """
    Process a command and send results to Discord.
    Can be called from: web UI, cron jobs, or other services.

    Examples:
      POST /discord/command {"command": "alerts"}
      POST /discord/command {"command": "price SPY"}
      POST /discord/command {"command": "stats"}
      POST /discord/command {"command": "brief"}
      POST /discord/command {"command": "scan NVDA"}
    """
    background_tasks.add_task(process_command, req.command)
    return {"status": "queued", "command": req.command}


@discord_router.post("/alert")
async def send_alert_notification(alert: AlertNotification):
    """Send an alert notification to Discord immediately"""
    dc = get_discord()
    success = await dc.send_alert(
        symbol=alert.symbol,
        level=alert.level,
        direction=alert.direction,
        action=alert.action,
        note=alert.note,
        price=alert.price if alert.price > 0 else None
    )
    return {"status": "sent" if success else "failed"}


@discord_router.get("/test")
async def test_webhook():
    """Test Discord webhook connection"""
    dc = get_discord()
    success = await dc.send_message(content="✅ **Webhook test successful!** SEF Trading Bot is connected.")
    return {"status": "connected" if success else "failed"}


@discord_router.get("/queue")
async def get_queue_status():
    """Get task queue stats and send to Discord"""
    queue = get_task_queue()
    stats = queue.get_queue_stats()

    dc = get_discord()
    await dc.send_queue_stats(stats)

    return {"status": "sent", "stats": stats}


@discord_router.post("/check-alerts")
async def trigger_alert_check(background_tasks: BackgroundTasks):
    """Manually trigger an alert check"""
    background_tasks.add_task(check_alerts_against_prices)
    return {"status": "checking"}


# =============================================================================
# COMMAND PROCESSOR
# =============================================================================

async def process_command(command_text: str):
    """Process a command string and send results to Discord"""
    dc = get_discord()
    parts = command_text.strip().split(" ", 1)
    cmd = parts[0].lower().strip("/")
    args = parts[1].strip() if len(parts) > 1 else ""

    try:
        if cmd in ("alerts", "alert"):
            await cmd_alerts(args)
        elif cmd in ("stats", "stat", "performance"):
            await cmd_stats()
        elif cmd in ("price", "quote", "p"):
            await cmd_price(args)
        elif cmd in ("brief", "market", "morning"):
            await cmd_brief()
        elif cmd in ("scan", "scanner"):
            await cmd_scan(args)
        elif cmd in ("setup", "levels"):
            await cmd_setup(args)
        elif cmd in ("queue", "tasks"):
            queue = get_task_queue()
            stats = queue.get_queue_stats()
            await dc.send_queue_stats(stats)
        elif cmd in ("help", "start", "commands"):
            await cmd_help()
        else:
            # Try hybrid router for natural language
            router = get_router()
            classification = await router.classify_message(command_text)
            task_type = classification.get("task_type")
            symbol = classification.get("symbol")

            if task_type == TaskType.PRICE_CHECK and symbol:
                await cmd_price(symbol)
            elif task_type == TaskType.ALERT_LOOKUP:
                await cmd_alerts(symbol or "")
            elif task_type == TaskType.TRADE_STATS:
                await cmd_stats()
            elif task_type == TaskType.MARKET_BRIEF:
                await cmd_brief()
            else:
                await dc.send_message(content=f"🤔 Unknown command: `{command_text}`\nTry: alerts, stats, price SPY, brief, scan NVDA")

    except Exception as e:
        await dc.send_error(f"Command error: {str(e)[:200]}")


async def cmd_help():
    dc = get_discord()
    embed = {
        "title": "🤖 SEF Trading Bot — Commands",
        "color": 0x00D9FF,
        "description": (
            "**Available commands** (via API or web UI):\n\n"
            "`alerts` — View active alerts\n"
            "`alerts SPY` — Alerts for specific symbol\n"
            "`stats` — Trade performance stats\n"
            "`price AAPL` — Quick price check\n"
            "`scan NVDA` — Run scanner (queued)\n"
            "`setup TSLA` — Setup analysis\n"
            "`brief` — Market brief (SPY, QQQ, IWM)\n"
            "`queue` — Task queue status\n"
            "`help` — This message\n\n"
            "**API:**\n"
            "`POST /discord/command {\"command\": \"price SPY\"}`\n"
            "`GET /discord/test` — Test webhook\n"
            "`POST /discord/check-alerts` — Manual alert check"
        ),
        "footer": {"text": "SEF Trading Terminal"}
    }
    await dc.send_message(embeds=[embed])


async def cmd_alerts(symbol: str = ""):
    dc = get_discord()
    symbol_upper = symbol.strip().upper() if symbol else None
    all_alerts = []

    # 1) Try Firestore REST API (primary — no service account needed)
    try:
        from firestore_rest import search_all_alerts, is_available as rest_available
        if rest_available():
            rest_alerts = search_all_alerts(symbol_upper)
            for a in rest_alerts:
                if not a.get('triggered', False):
                    a["_source"] = "firestore-rest"
                    all_alerts.append(a)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ REST alert fetch: {e}")

    # 2) Fallback: firebase-admin SDK
    if not all_alerts and firestore_available:
        try:
            fs = get_firestore()
            if fs.is_available() and fs.db:
                seen_ids = set()
                users_ref = fs.db.collection('users')
                for user_doc in users_ref.stream():
                    alerts_ref = users_ref.document(user_doc.id).collection('alerts')
                    q = alerts_ref
                    if symbol_upper:
                        q = q.where('symbol', '==', symbol_upper)
                    for doc in q.stream():
                        alert = doc.to_dict()
                        alert['id'] = doc.id
                        if doc.id not in seen_ids and not alert.get('triggered', False):
                            all_alerts.append(alert)
                            seen_ids.add(doc.id)
        except Exception as e:
            print(f"⚠️ Firestore alert fetch: {e}")

    # 3) Fallback: local chart_system alerts
    if not all_alerts:
        try:
            import httpx
            port = os.environ.get("PORT", "8000")
            params = {}
            if symbol_upper:
                params["symbol"] = symbol_upper
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"http://localhost:{port}/api/alerts", params=params)
                if resp.status_code == 200:
                    all_alerts.extend(resp.json().get("alerts", []))
        except Exception as e:
            print(f"⚠️ Local alert fetch: {e}")

    await dc.send_alerts_list(all_alerts)


async def cmd_stats():
    dc = get_discord()
    if not firestore_available:
        await dc.send_message(content="⚠️ Firestore not available")
        return

    fs = get_firestore()
    stats = fs.get_trade_stats(user_id="default")

    if stats.get("total", 0) == 0:
        await dc.send_message(content="📊 No closed trades yet.")
        return

    await dc.send_trade_stats(stats)


async def cmd_price(symbol: str):
    dc = get_discord()
    if not symbol:
        await dc.send_message(content="⚠️ Specify a symbol: `price SPY`")
        return

    symbol = symbol.strip().upper()
    price_data = await _get_price_data(symbol)

    if price_data:
        await dc.send_price(**price_data)
    else:
        await dc.send_message(content=f"⚠️ Could not get price for {symbol}")


async def cmd_brief():
    dc = get_discord()
    indices = ["SPY", "QQQ", "IWM", "DIA"]
    prices = []

    for symbol in indices:
        data = await _get_price_data(symbol)
        if data:
            prices.append(data)

    await dc.send_market_brief(prices)


async def cmd_scan(symbol: str):
    dc = get_discord()
    if not symbol:
        await dc.send_message(content="⚠️ Specify a symbol: `scan NVDA`")
        return

    symbol = symbol.strip().upper()
    queue = get_task_queue()
    task = Task(
        task_type=TaskType.SCANNER_RUN,
        payload={"symbol": symbol},
        priority=TaskPriority.HIGH,
        source="discord"
    )
    task_id = queue.enqueue(task)

    if task_id:
        await dc.send_message(content=f"🔍 Scanner queued for **{symbol}** — Task: `{task_id[:8]}`")
    else:
        await dc.send_message(content=f"⚠️ Failed to queue scan for {symbol}")


async def cmd_setup(symbol: str):
    dc = get_discord()
    if not symbol:
        await dc.send_message(content="⚠️ Specify a symbol: `setup TSLA`")
        return

    symbol = symbol.strip().upper()
    price_data = await _get_price_data(symbol)

    if not price_data:
        await dc.send_message(content=f"⚠️ Could not get data for {symbol}")
        return

    # Get alerts for this symbol
    alerts = []
    if firestore_available:
        fs = get_firestore()
        alerts = fs.get_alerts(user_id="default", symbol=symbol)

    await dc.send_setup_analysis(
        symbol=symbol,
        price=price_data["price"],
        change=price_data["change"],
        change_pct=price_data["change_pct"],
        alerts=alerts
    )


# =============================================================================
# PRICE HELPER
# =============================================================================

async def _get_price_data(symbol: str) -> Optional[Dict]:
    """Get price data for a symbol"""
    if not yf_available:
        return None

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


# =============================================================================
# ALERT DELIVERY PIPELINE
# =============================================================================

async def deliver_alert(symbol: str, level: float, direction: str,
                         action: str, note: str = "", price: float = None):
    """
    Called by the alert checker when an alert triggers.
    Sends notification to Discord.
    """
    dc = get_discord()
    try:
        await dc.send_alert(
            symbol=symbol,
            level=level,
            direction=direction,
            action=action,
            note=note,
            price=price
        )
        print(f"📨 Alert delivered via Discord: {symbol} @ {level}")
    except Exception as e:
        print(f"❌ Alert delivery failed: {e}")


# =============================================================================
# ALERT CHECKER (runs on interval)
# =============================================================================

async def check_alerts_against_prices():
    """
    Check all active alerts against current prices.
    Called by background task on interval.
    """
    if not firestore_available or not yf_available:
        return

    fs = get_firestore()
    alerts = fs.get_alerts(user_id="default")

    if not alerts:
        return

    # Get untriggered alerts grouped by symbol
    symbols = set(a.get("symbol", "") for a in alerts if not a.get("triggered"))
    if not symbols:
        return

    for symbol in symbols:
        try:
            price_data = await _get_price_data(symbol)
            if not price_data:
                continue

            current_price = price_data["price"]

            for alert in alerts:
                if alert.get("symbol") != symbol or alert.get("triggered"):
                    continue

                level = alert.get("level", 0)
                direction = alert.get("direction", "above")

                triggered = False
                if direction == "above" and current_price >= level:
                    triggered = True
                elif direction == "below" and current_price <= level:
                    triggered = True

                if triggered:
                    # Mark as triggered / delete
                    if alert.get("id"):
                        fs.delete_alert_by_id("default", alert["id"])

                    # Deliver via Discord
                    await deliver_alert(
                        symbol=symbol,
                        level=level,
                        direction=direction,
                        action=alert.get("action", "ALERT"),
                        note=alert.get("note", ""),
                        price=current_price
                    )

        except Exception as e:
            print(f"⚠️ Alert check error for {symbol}: {e}")


# =============================================================================
# BACKGROUND LOOPS
# =============================================================================

async def alert_checker_loop():
    """Check alerts every 60 seconds during market hours"""
    while True:
        try:
            now = datetime.now()
            # Only check during extended market hours (8 AM - 5 PM ET, weekdays)
            if now.weekday() < 5 and 8 <= now.hour <= 17:
                await check_alerts_against_prices()
        except Exception as e:
            print(f"⚠️ Alert checker error: {e}")

        await asyncio.sleep(60)


# =============================================================================
# TASK WORKERS
# =============================================================================

def register_task_workers(queue):
    """Register handlers for each task type"""
    dc_getter = get_discord

    async def worker_scanner_run(task: Task) -> Dict:
        symbol = task.payload.get("symbol", "")
        dc = dc_getter()
        try:
            price_data = await _get_price_data(symbol)
            if price_data:
                await dc.send_setup_analysis(
                    symbol=symbol,
                    price=price_data["price"],
                    change=price_data["change"],
                    change_pct=price_data["change_pct"]
                )
            else:
                await dc.send_message(content=f"⚠️ Scanner: Could not get data for {symbol}")
            return {"symbol": symbol, "status": "scanned"}
        except Exception as e:
            await dc.send_error(f"Scanner error for {symbol}: {str(e)[:100]}")
            raise

    async def worker_setup_analysis(task: Task) -> Dict:
        symbol = task.payload.get("symbol", "")
        dc = dc_getter()
        price_data = await _get_price_data(symbol)
        if price_data:
            alerts = []
            if firestore_available:
                fs = get_firestore()
                alerts = fs.get_alerts(user_id="default", symbol=symbol)
            await dc.send_setup_analysis(
                symbol=symbol,
                price=price_data["price"],
                change=price_data["change"],
                change_pct=price_data["change_pct"],
                alerts=alerts
            )
        return {"symbol": symbol, "status": "analyzed"}

    async def worker_alert_check(task: Task) -> Dict:
        await check_alerts_against_prices()
        dc = dc_getter()
        await dc.send_message(content="✅ Alert check complete.")
        return {"status": "checked"}

    async def worker_market_brief(task: Task) -> Dict:
        await cmd_brief()
        return {"status": "sent"}

    async def worker_custom_query(task: Task) -> Dict:
        dc = dc_getter()
        raw = task.payload.get("raw_text", "")
        await dc.send_message(
            content=f"🤔 Received: `{raw[:100]}`\nTry: alerts, stats, price SYMBOL, brief, scan SYMBOL"
        )
        return {"status": "unhandled", "raw_text": raw}

    async def worker_trade_plan(task: Task) -> Dict:
        dc = dc_getter()
        symbol = task.payload.get("symbol", "")
        await dc.send_message(content=f"📋 Trade plan for **{symbol}** — check web UI for full AI-powered plans.")
        return {"symbol": symbol, "status": "done"}

    async def worker_full_analysis(task: Task) -> Dict:
        dc = dc_getter()
        symbol = task.payload.get("symbol", "")
        await dc.send_message(content=f"🧠 Deep analysis for **{symbol}** — check web UI for full report.")
        return {"symbol": symbol, "status": "done"}

    queue.register_worker(TaskType.SCANNER_RUN, worker_scanner_run)
    queue.register_worker(TaskType.SETUP_ANALYSIS, worker_setup_analysis)
    queue.register_worker(TaskType.ALERT_CHECK, worker_alert_check)
    queue.register_worker(TaskType.MARKET_BRIEF, worker_market_brief)
    queue.register_worker(TaskType.CUSTOM_QUERY, worker_custom_query)
    queue.register_worker(TaskType.TRADE_PLAN, worker_trade_plan)
    queue.register_worker(TaskType.FULL_ANALYSIS, worker_full_analysis)


# =============================================================================
# SETUP & LIFECYCLE
# =============================================================================

async def setup_discord(app=None):
    """
    Initialize Discord integration:
    - Test webhook
    - Connect task queue to Firestore
    - Register workers
    - Start background loops
    """
    dc = get_discord()
    queue = get_task_queue()

    # Connect task queue to Firestore
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            queue.set_db(fs.db)
            print("✅ Task queue connected to Firestore")

    # Test webhook
    success = await dc.send_message(
        content="🤖 **SEF Trading Bot Online**\nAlert monitoring active. Type `help` for commands."
    )

    if success:
        print("✅ Discord webhook connected")
    else:
        print("⚠️ Discord webhook test failed — check DISCORD_WEBHOOK_URL")

    # Register workers
    register_task_workers(queue)

    # Start background loops
    asyncio.create_task(queue.run_worker_loop())
    asyncio.create_task(alert_checker_loop())

    # Start two-way Discord bot listener (if token is set)
    try:
        from discord_listener import start_discord_bot
        await start_discord_bot()
    except ImportError:
        print("⚠️ discord_listener.py not found — two-way bot disabled")
    except Exception as e:
        print(f"⚠️ Discord bot listener error: {e}")

    print("✅ Discord integration ready")
