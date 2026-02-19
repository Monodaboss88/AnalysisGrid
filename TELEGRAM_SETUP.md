# Telegram Bot + Hybrid Task Queue Setup

## Quick Start (5 minutes)

### Step 1: Create Your Bot
1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Give it a name: `SEF Trading Bot`
4. Give it a username: `sef_trading_bot` (must be unique, try adding numbers)
5. Copy the **API token** BotFather gives you

### Step 2: Get Your Chat ID
1. Search for **@userinfobot** on Telegram
2. Send it any message
3. Copy your **chat ID** (numeric, like `123456789`)

### Step 3: Set Environment Variables

**Windows PowerShell:**
```powershell
$env:TELEGRAM_BOT_TOKEN = "your_bot_token_here"
$env:TELEGRAM_CHAT_ID = "your_chat_id_here"
$env:TELEGRAM_WEBHOOK_SECRET = "any_random_string_here"
$env:ANTHROPIC_API_KEY = "your_anthropic_key"  # For Haiku classification
```

**Or add to a `.env` file:**
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_WEBHOOK_SECRET=any_random_string_here
ANTHROPIC_API_KEY=your_anthropic_key
BASE_URL=https://your-domain.com  # Only needed if exposing publicly
```

### Step 4: Start the Server
```bash
python unified_server.py
```

You should see:
```
✅ Telegram Bot + Hybrid Task Queue enabled
✅ Telegram bot: @sef_trading_bot
✅ Task queue connected to Firestore
✅ Telegram integration ready
```

### Step 5: Test It
Open your bot in Telegram and send:
- `/start` — See all commands
- `/price SPY` — Quick price check
- `/alerts` — View your alerts
- `/stats` — Trade performance
- `How is NVDA looking?` — Natural language (uses Haiku)

---

## Architecture

```
You (Telegram) → Bot Webhook → FastAPI Server
                                    │
                        ┌───────────┼───────────┐
                        ▼           ▼           ▼
                  Quick Task   Haiku Parse   Slash Command
                  (instant)    (~$0.0003)    (instant)
                        │           │           │
                        ▼           ▼           ▼
                  Direct Reply  Queue Task   Direct Reply
                                    │
                                    ▼
                            Task Worker
                            (processes)
                                    │
                                    ▼
                         Telegram Reply
```

## Available Commands

| Command | What it does |
|---------|-------------|
| `/start` | Welcome message + command list |
| `/alerts` | Show active alerts |
| `/alerts SPY` | Alerts for specific symbol |
| `/stats` | Trade performance stats |
| `/price AAPL` | Quick price check |
| `/scan NVDA` | Run scanner (queued task) |
| `/brief` | Market brief (SPY, QQQ, IWM) |
| `/queue` | Task queue status |
| `/watchlist` | Watchlist info |

## Natural Language (Haiku-powered)

Just type normally:
- "How is SPY doing?" → Price check
- "Show me my alerts" → Alert list
- "What's my win rate?" → Trade stats
- "Scan TSLA for me" → Scanner run (queued)
- "Give me a market update" → Market brief (queued)
- "Deep dive on AAPL" → Full analysis (queued)

## Alert Delivery

When your alerts trigger, you'll get a Telegram notification with:
- Symbol, price level, and action
- Current price
- Quick-action buttons (View Setup, Log Trade, Dismiss)

Alert checking runs every 60 seconds during market hours (9 AM - 4 PM ET, weekdays).

## Webhook vs Polling

**Local development (no public URL):**
The bot works without a webhook using the task queue polling. Alerts still deliver.
You just won't receive inbound messages from Telegram without a webhook.

**For full two-way communication, you need a public URL:**
- Use ngrok: `ngrok http 8000` → set BASE_URL to the ngrok URL
- Or deploy to Railway/Render with your existing setup

## Cost Estimate

| Component | Cost |
|-----------|------|
| Telegram Bot API | Free |
| Haiku classification (~100 msgs/day) | ~$0.01/day |
| Firestore task queue | Free tier |
| Alert checking (yfinance) | Free |
| **Total** | **~$0.30/month** |

## Files Created

| File | Purpose |
|------|---------|
| `telegram_bot.py` | Telegram API client, message formatting |
| `task_queue.py` | Firestore-backed task queue |
| `hybrid_router.py` | Haiku message classifier + routing |
| `telegram_endpoints.py` | FastAPI routes, commands, workers |
| `TELEGRAM_SETUP.md` | This file |

## Troubleshooting

**"Telegram bot token not configured"**
→ Set `TELEGRAM_BOT_TOKEN` environment variable

**"Firestore not available"**
→ Set `FIREBASE_SERVICE_ACCOUNT` environment variable

**"Haiku classification error"**
→ Set `ANTHROPIC_API_KEY` — falls back to keyword matching if unavailable

**Bot doesn't respond to messages**
→ Set `BASE_URL` and ensure webhook is registered, or use ngrok for local dev
