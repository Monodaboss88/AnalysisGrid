"""
Telegram Endpoint Router
========================
FastAPI router that handles:
  - Telegram webhook (inbound messages)
  - Command handlers (/alerts, /stats, /scan, etc.)
  - Alert delivery pipeline (scanner → Telegram)
  - Task queue worker callbacks
  - Scheduled morning brief

Mount this router in unified_server.py:
  from telegram_endpoints import telegram_router, setup_telegram
  app.include_router(telegram_router, prefix="/telegram")
  # In startup: await setup_telegram(app)

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from telegram_bot import get_telegram, TelegramClient, TelegramMessage
from task_queue import get_task_queue, Task, TaskType, TaskPriority, QUICK_TASKS
from hybrid_router import get_router

# Try to import existing modules
try:
    from firestore_store import get_firestore
    firestore_available = True
except ImportError:
    firestore_available = False

try:
    from polygon_data import get_price_quote
    polygon_price_available = True
except ImportError:
    polygon_price_available = False


# =============================================================================
# ROUTER
# =============================================================================

telegram_router = APIRouter(tags=["telegram"])

# Webhook secret for verification
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
BASE_URL = os.getenv("BASE_URL", "")  # e.g., https://your-domain.com


# =============================================================================
# WEBHOOK ENDPOINT
# =============================================================================

@telegram_router.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive updates from Telegram.
    This is the main entry point for all incoming messages.
    """
    # Verify secret token if configured
    if WEBHOOK_SECRET:
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if token != WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret token")

    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Process in background so we return 200 quickly (Telegram expects <30s)
    background_tasks.add_task(process_telegram_update, update)

    return JSONResponse({"ok": True})


async def process_telegram_update(update: Dict):
    """Process a Telegram update — route through hybrid system"""
    tg = get_telegram()
    router = get_router()
    queue = get_task_queue()

    msg = tg.parse_update(update)
    if not msg:
        return

    # Handle callback queries (inline button presses)
    if update.get("callback_query"):
        await handle_callback(update["callback_query"])
        return

    # Check for slash commands first
    if msg.is_command:
        response = await handle_command(msg)
        if response:
            await tg.send_message(chat_id=msg.chat_id, text=response)
            return

    # Route through Haiku hybrid classifier
    response_text, task = await router.route_message(
        text=msg.text,
        chat_id=msg.chat_id,
        user_id=msg.user_id
    )

    if task is None:
        # Quick response (greeting, etc.)
        await tg.send_message(chat_id=msg.chat_id, text=response_text)
        return

    if response_text == "__QUICK__":
        # Handle quick task inline
        result = await handle_quick_task(task)
        await tg.send_message(chat_id=msg.chat_id, text=result)
        return

    # Heavy task — send acknowledgment and queue it
    await tg.send_message(chat_id=msg.chat_id, text=response_text)
    queue.enqueue(task)


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

async def handle_command(msg: TelegramMessage) -> Optional[str]:
    """Handle slash commands"""

    commands = {
        "/start": cmd_start,
        "/help": cmd_help,
        "/alerts": cmd_alerts,
        "/stats": cmd_stats,
        "/scan": cmd_scan,
        "/price": cmd_price,
        "/brief": cmd_brief,
        "/queue": cmd_queue,
        "/watchlist": cmd_watchlist,
    }

    handler = commands.get(msg.command)
    if handler:
        return await handler(msg)
    return None


async def cmd_start(msg: TelegramMessage) -> str:
    return (
        "🤖 *SEF Trading Terminal — Telegram*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Commands:\n"
        "/alerts — View your active alerts\n"
        "/stats — Trade performance stats\n"
        "/scan SYMBOL — Run scanner\n"
        "/price SYMBOL — Quick price check\n"
        "/brief — Morning market brief\n"
        "/queue — Task queue status\n"
        "/watchlist — Your watchlist\n"
        "/help — Show this message\n\n"
        "Or just type naturally:\n"
        "_'How is SPY looking?'_\n"
        "_'Show me TSLA setup'_\n"
        "_'What's my win rate?'_"
    )


async def cmd_help(msg: TelegramMessage) -> str:
    return await cmd_start(msg)


async def cmd_alerts(msg: TelegramMessage) -> str:
    """Show active alerts"""
    if not firestore_available:
        return "⚠️ Firestore not available"

    fs = get_firestore()
    symbol = msg.command_args.strip().upper() if msg.command_args else None
    alerts = fs.get_alerts(user_id="default", symbol=symbol)

    if not alerts:
        return f"📭 No active alerts{f' for {symbol}' if symbol else ''}."

    text = f"🔔 *Active Alerts*{f' — {symbol}' if symbol else ''}\n━━━━━━━━━━━━━━━━━━\n"

    for a in alerts[:15]:
        direction = "⬆️" if a.get("direction") == "above" else "⬇️"
        text += f"{direction} *{a.get('symbol')}* ${a.get('level', 0):.2f} — {a.get('action', '?')}"
        if a.get("note"):
            text += f" _{a['note']}_"
        text += "\n"

    if len(alerts) > 15:
        text += f"\n_...and {len(alerts) - 15} more_"

    return text


async def cmd_stats(msg: TelegramMessage) -> str:
    """Show trade stats"""
    if not firestore_available:
        return "⚠️ Firestore not available"

    fs = get_firestore()
    stats = fs.get_trade_stats(user_id="default")

    if stats.get("total", 0) == 0:
        return "📊 No closed trades yet."

    wr = stats.get("win_rate", 0)
    wr_emoji = "🔥" if wr >= 60 else "✅" if wr >= 50 else "⚠️"

    return (
        f"📊 *Trade Performance*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Total: {stats.get('total', 0)} trades\n"
        f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
        f"{wr_emoji} Win Rate: *{wr:.1f}%*\n"
        f"Total P&L: *${stats.get('total_pnl', 0):.2f}*\n"
        f"Avg Win: ${stats.get('avg_win', 0):.2f}\n"
        f"Avg Loss: ${stats.get('avg_loss', 0):.2f}"
    )


async def cmd_price(msg: TelegramMessage) -> str:
    """Quick price check"""
    symbol = msg.command_args.strip().upper() if msg.command_args else None
    if not symbol:
        return "Usage: /price SYMBOL\nExample: `/price SPY`"

    return await _get_price(symbol)


async def cmd_scan(msg: TelegramMessage) -> str:
    """Queue a scanner run"""
    symbol = msg.command_args.strip().upper() if msg.command_args else None
    if not symbol:
        return "Usage: /scan SYMBOL\nExample: `/scan NVDA`"

    queue = get_task_queue()
    task = Task(
        task_type=TaskType.SCANNER_RUN,
        payload={"symbol": symbol},
        priority=TaskPriority.HIGH,
        source="telegram",
        chat_id=msg.chat_id,
        user_id=msg.user_id
    )
    task_id = queue.enqueue(task)

    if task_id:
        return f"🔍 Scanner queued for *{symbol}*\nTask ID: `{task_id[:8]}`"
    return "⚠️ Failed to queue scan. Check task queue."


async def cmd_brief(msg: TelegramMessage) -> str:
    """Queue a market brief"""
    queue = get_task_queue()
    task = Task(
        task_type=TaskType.MARKET_BRIEF,
        payload={},
        priority=TaskPriority.NORMAL,
        source="telegram",
        chat_id=msg.chat_id,
        user_id=msg.user_id
    )
    task_id = queue.enqueue(task)

    if task_id:
        return "📡 Market brief queued. I'll send it shortly."
    return "⚠️ Failed to queue brief."


async def cmd_queue(msg: TelegramMessage) -> str:
    """Show queue stats"""
    queue = get_task_queue()
    stats = queue.get_queue_stats()

    if not stats:
        return "⚠️ Could not retrieve queue stats."

    return (
        f"📋 *Task Queue Status*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⏳ Pending: {stats.get('pending', 0)}\n"
        f"⚙️ Processing: {stats.get('processing', 0)}\n"
        f"✅ Completed: {stats.get('completed', 0)}\n"
        f"❌ Failed: {stats.get('failed', 0)}"
    )


async def cmd_watchlist(msg: TelegramMessage) -> str:
    """Show watchlist summary"""
    return (
        "📋 *Watchlist*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "Use the web UI for full watchlist management.\n"
        "Quick commands:\n"
        "`/scan SYMBOL` — Scan a specific stock\n"
        "`/price SYMBOL` — Check current price"
    )


# =============================================================================
# QUICK TASK HANDLERS
# =============================================================================

async def handle_quick_task(task: Task) -> str:
    """Handle quick tasks inline (no queue needed)"""
    handlers = {
        TaskType.ALERT_LOOKUP: _quick_alert_lookup,
        TaskType.TRADE_STATS: _quick_trade_stats,
        TaskType.PRICE_CHECK: _quick_price_check,
        TaskType.WATCHLIST: _quick_watchlist,
    }

    handler = handlers.get(task.task_type)
    if handler:
        return await handler(task)
    return "⚠️ Unknown quick task type."


async def _quick_alert_lookup(task: Task) -> str:
    if not firestore_available:
        return "⚠️ Firestore not available"

    fs = get_firestore()
    symbol = task.payload.get("symbol")
    alerts = fs.get_alerts(user_id="default", symbol=symbol)

    if not alerts:
        return f"📭 No alerts{f' for {symbol}' if symbol else ''}."

    text = f"🔔 *Alerts*{f' — {symbol}' if symbol else ''}\n"
    for a in alerts[:10]:
        direction = "⬆️" if a.get("direction") == "above" else "⬇️"
        text += f"{direction} *{a.get('symbol')}* ${a.get('level', 0):.2f} — {a.get('action')}\n"
    return text


async def _quick_trade_stats(task: Task) -> str:
    if not firestore_available:
        return "⚠️ Firestore not available"

    fs = get_firestore()
    stats = fs.get_trade_stats(user_id="default")
    if stats.get("total", 0) == 0:
        return "📊 No closed trades yet."

    wr = stats.get("win_rate", 0)
    return f"📊 *{stats['total']} trades* | WR: *{wr:.1f}%* | P&L: *${stats['total_pnl']:.2f}*"


async def _quick_price_check(task: Task) -> str:
    symbol = task.payload.get("symbol")
    if not symbol:
        return "⚠️ No symbol specified. Try: 'price SPY'"
    return await _get_price(symbol)


async def _quick_watchlist(task: Task) -> str:
    return "📋 Use `/watchlist` for details or visit the web UI."


async def _get_price(symbol: str) -> str:
    """Get quick price quote via Polygon"""
    if not polygon_price_available:
        return f"⚠️ Polygon not available for price lookup"

    try:
        q = get_price_quote(symbol)
        if not q:
            return f"⚠️ Could not get price for {symbol}"

        price = q["price"]
        prev_close = q["prev_close"]
        change = q["change"]
        change_pct = q["change_pct"]
        emoji = "🟢" if change >= 0 else "🔴"

        return (
            f"{emoji} *{symbol}*: ${price:.2f}\n"
            f"Change: {'+' if change >= 0 else ''}{change:.2f} ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)"
        )
    except Exception as e:
        return f"⚠️ Error getting price for {symbol}: {str(e)[:100]}"


# =============================================================================
# CALLBACK HANDLER (Inline button presses)
# =============================================================================

async def handle_callback(callback_query: Dict):
    """Handle inline keyboard button presses"""
    tg = get_telegram()
    data = callback_query.get("data", "")
    chat_id = str(callback_query["message"]["chat"]["id"])
    callback_id = callback_query["id"]

    # Acknowledge the callback
    await tg.answer_callback(callback_id)

    if data == "dismiss":
        return

    if data == "alerts_list":
        msg = TelegramMessage(chat_id=chat_id, user_id="", username="", text="/alerts", message_id=0)
        response = await cmd_alerts(msg)
        await tg.send_message(chat_id=chat_id, text=response)

    elif data.startswith("setup_"):
        symbol = data.replace("setup_", "")
        queue = get_task_queue()
        task = Task(
            task_type=TaskType.SETUP_ANALYSIS,
            payload={"symbol": symbol},
            priority=TaskPriority.HIGH,
            source="telegram",
            chat_id=chat_id
        )
        queue.enqueue(task)
        await tg.send_message(chat_id=chat_id, text=f"📊 Analyzing *{symbol}* setup...")

    elif data.startswith("log_"):
        parts = data.split("_")
        if len(parts) >= 4:
            symbol, action, level = parts[1], parts[2], parts[3]
            await tg.send_message(
                chat_id=chat_id,
                text=f"📝 To log this trade, use the web UI or send:\n`/log {symbol} {action} {level}`"
            )


# =============================================================================
# ALERT DELIVERY PIPELINE
# =============================================================================

async def deliver_alert(symbol: str, level: float, direction: str,
                         action: str, note: str = "", price: float = None):
    """
    Called by the alert checker when an alert triggers.
    Sends notification to Telegram.
    """
    tg = get_telegram()
    try:
        await tg.send_alert(
            symbol=symbol,
            level=level,
            direction=direction,
            action=action,
            note=note,
            price=price
        )
        print(f"📨 Alert delivered via Telegram: {symbol} @ {level}")
    except Exception as e:
        print(f"❌ Alert delivery failed: {e}")


# =============================================================================
# ALERT CHECKER (runs on interval)
# =============================================================================

async def check_alerts_against_prices():
    """
    Check all active alerts against current prices.
    Called by a background task on interval.
    """
    if not firestore_available or not polygon_price_available:
        return

    fs = get_firestore()
    alerts = fs.get_alerts(user_id="default")

    if not alerts:
        return

    # Group alerts by symbol
    symbols = set(a.get("symbol", "") for a in alerts if not a.get("triggered"))
    if not symbols:
        return

    # Batch price lookup
    for symbol in symbols:
        try:
            q = get_price_quote(symbol)
            if not q:
                continue

            current_price = q["price"]

            # Check each alert for this symbol
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
                    # Mark as triggered
                    if alert.get("id"):
                        fs.delete_alert_by_id("default", alert["id"])

                    # Deliver via Telegram
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
# SETUP & LIFECYCLE
# =============================================================================

async def setup_telegram(app=None):
    """
    Initialize Telegram bot:
    - Register webhook
    - Start task queue worker
    - Start alert checker loop
    """
    tg = get_telegram()
    queue = get_task_queue()

    # Connect task queue to Firestore
    if firestore_available:
        fs = get_firestore()
        if fs.is_available():
            queue.set_db(fs.db)
            print("✅ Task queue connected to Firestore")

    # Get bot info
    bot_info = await tg.get_me()
    if bot_info.get("ok"):
        bot_name = bot_info["result"].get("username", "unknown")
        print(f"✅ Telegram bot: @{bot_name}")
    else:
        print("⚠️ Telegram bot token not configured or invalid")
        return

    # Set webhook if BASE_URL is configured
    if BASE_URL:
        webhook_url = f"{BASE_URL}/telegram/webhook"
        await tg.set_webhook(webhook_url, secret_token=WEBHOOK_SECRET)
    else:
        print("⚠️ BASE_URL not set — webhook not registered. Use polling or set BASE_URL.")

    # Register task workers
    register_task_workers(queue)

    # Start background loops
    asyncio.create_task(queue.run_worker_loop())
    asyncio.create_task(alert_checker_loop())

    print("✅ Telegram integration ready")


async def alert_checker_loop():
    """Background loop to check alerts every 60 seconds during market hours"""
    while True:
        try:
            now = datetime.now()
            # Only check during market hours (9:30 AM - 4:00 PM ET, weekdays)
            if now.weekday() < 5 and 9 <= now.hour <= 16:
                await check_alerts_against_prices()
        except Exception as e:
            print(f"⚠️ Alert checker error: {e}")

        await asyncio.sleep(60)  # Check every 60 seconds


def register_task_workers(queue):
    """Register handlers for each task type"""

    async def worker_scanner_run(task: Task) -> Dict:
        """Run scanner on a symbol"""
        symbol = task.payload.get("symbol", "")
        tg = get_telegram()

        # Use your existing scanner infrastructure
        try:
            # Import scanner modules
            from chart_input_analyzer import ChartInputSystem
            scanner = ChartInputSystem()

            # This is a placeholder — hook into your actual scanner
            result = {"symbol": symbol, "status": "scanned", "message": f"Scanner results for {symbol} — check web UI for details"}

            # Send result to Telegram
            await tg.send_message(
                chat_id=task.chat_id,
                text=f"🔍 *Scanner Results — {symbol}*\nCheck the web UI for full details."
            )
            return result
        except Exception as e:
            await tg.send_message(chat_id=task.chat_id, text=f"⚠️ Scanner error for {symbol}: {str(e)[:100]}")
            raise

    async def worker_setup_analysis(task: Task) -> Dict:
        symbol = task.payload.get("symbol", "")
        tg = get_telegram()

        price_text = await _get_price(symbol)
        await tg.send_message(
            chat_id=task.chat_id,
            text=f"📊 *Setup Analysis — {symbol}*\n━━━━━━━━━━━━━━━━━━\n{price_text}\n\n_Full analysis available in web UI._"
        )
        return {"symbol": symbol, "status": "analyzed"}

    async def worker_alert_check(task: Task) -> Dict:
        await check_alerts_against_prices()
        tg = get_telegram()
        await tg.send_message(chat_id=task.chat_id, text="✅ Alert check complete.")
        return {"status": "checked"}

    async def worker_market_brief(task: Task) -> Dict:
        tg = get_telegram()
        # Quick market brief using major indices
        indices = ["SPY", "QQQ", "IWM"]
        text = "📡 *Market Brief*\n━━━━━━━━━━━━━━━━━━\n"

        for symbol in indices:
            price_info = await _get_price(symbol)
            text += f"{price_info}\n\n"

        text += f"⏰ {datetime.now().strftime('%I:%M %p ET')}"
        await tg.send_message(chat_id=task.chat_id, text=text)
        return {"status": "sent"}

    async def worker_custom_query(task: Task) -> Dict:
        tg = get_telegram()
        raw = task.payload.get("raw_text", "")
        await tg.send_message(
            chat_id=task.chat_id,
            text=f"🤔 I received your message but couldn't determine the exact request.\n\nTry:\n/help — See available commands\n/scan SYMBOL — Run scanner\n/price SYMBOL — Price check"
        )
        return {"status": "unhandled", "raw_text": raw}

    async def worker_trade_plan(task: Task) -> Dict:
        symbol = task.payload.get("symbol", "")
        tg = get_telegram()
        await tg.send_message(
            chat_id=task.chat_id,
            text=f"📋 *Trade Plan — {symbol}*\n━━━━━━━━━━━━━━━━━━\n_Trade plan generation requires the full AI advisor. Check the web UI for AI-powered trade plans._"
        )
        return {"symbol": symbol, "status": "plan_generated"}

    async def worker_full_analysis(task: Task) -> Dict:
        symbol = task.payload.get("symbol", "")
        tg = get_telegram()
        await tg.send_message(
            chat_id=task.chat_id,
            text=f"🧠 *Deep Analysis — {symbol}*\n━━━━━━━━━━━━━━━━━━\n_Full AI analysis available in web UI._"
        )
        return {"symbol": symbol, "status": "analyzed"}

    # Register all workers
    queue.register_worker(TaskType.SCANNER_RUN, worker_scanner_run)
    queue.register_worker(TaskType.SETUP_ANALYSIS, worker_setup_analysis)
    queue.register_worker(TaskType.ALERT_CHECK, worker_alert_check)
    queue.register_worker(TaskType.MARKET_BRIEF, worker_market_brief)
    queue.register_worker(TaskType.CUSTOM_QUERY, worker_custom_query)
    queue.register_worker(TaskType.TRADE_PLAN, worker_trade_plan)
    queue.register_worker(TaskType.FULL_ANALYSIS, worker_full_analysis)
