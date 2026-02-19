# Discord Bot + Hybrid Task Queue Setup

## Quick Start (2 minutes)

Your webhook is already configured in the code. Just start the server:

```bash
python unified_server.py
```

You should see:
```
✅ Discord Bot + Hybrid Task Queue enabled
✅ Discord webhook connected
✅ Task queue connected to Firestore
✅ Discord integration ready
```

And a message appears in your Discord channel: **"SEF Trading Bot Online"**

## How It Works

```
Your Trading Server (unified_server.py)
        │
        ├── Alert Checker (every 60s during market hours)
        │       └── Alert triggers → Discord notification
        │
        ├── Task Queue (Firestore-backed)
        │       └── Queued tasks → Worker processes → Discord result
        │
        └── API Endpoints
                └── POST /discord/command → Discord response
```

## API Endpoints

Test your webhook:
```
GET http://localhost:8000/discord/test
```

Send commands:
```
POST http://localhost:8000/discord/command
{"command": "price SPY"}
```

```
POST http://localhost:8000/discord/command
{"command": "alerts"}
```

```
POST http://localhost:8000/discord/command
{"command": "brief"}
```

```
POST http://localhost:8000/discord/command
{"command": "stats"}
```

```
POST http://localhost:8000/discord/command
{"command": "scan NVDA"}
```

Send a manual alert:
```
POST http://localhost:8000/discord/alert
{
  "symbol": "SPY",
  "level": 450.00,
  "direction": "above",
  "action": "LONG",
  "note": "Breakout above VAH"
}
```

Manually trigger alert check:
```
POST http://localhost:8000/discord/check-alerts
```

## Available Commands

| Command | What it does |
|---------|-------------|
| `alerts` | Show active alerts |
| `alerts SPY` | Alerts for specific symbol |
| `stats` | Trade performance stats |
| `price AAPL` | Quick price check |
| `scan NVDA` | Run scanner (queued task) |
| `setup TSLA` | Setup analysis with alerts |
| `brief` | Market brief (SPY, QQQ, IWM, DIA) |
| `queue` | Task queue status |
| `help` | Command list |

## Alert Delivery

When your alerts trigger, Discord gets a rich embed with:
- Symbol, price level, direction, and action
- Current price
- Color-coded (green for LONG, red for SHORT, blue for ALERT)
- @here ping so your phone buzzes

Alert checking runs every 60 seconds during market hours (8 AM - 5 PM ET, weekdays).

## Environment Variables (Optional)

The webhook URL is hardcoded in `discord_bot.py`. To override:

```powershell
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/your/url"
$env:ANTHROPIC_API_KEY = "your_key"  # For Haiku message classification
```

## Cost Estimate

| Component | Cost |
|-----------|------|
| Discord webhook | Free |
| Haiku classification | ~$0.01/day |
| Firestore task queue | Free tier |
| yfinance price checks | Free |
| **Total** | **~$0.30/month** |

## Files

| File | Purpose |
|------|---------|
| `discord_bot.py` | Discord webhook client, rich embed formatting |
| `discord_endpoints.py` | FastAPI routes, command processor, alert checker |
| `task_queue.py` | Firestore-backed task queue with workers |
| `hybrid_router.py` | Haiku message classifier + routing |
| `telegram_bot.py` | Telegram client (available if you want to add later) |
| `telegram_endpoints.py` | Telegram routes (available if you want to add later) |

## Two-Way Bot Setup (Type in Discord, Get Responses)

The webhook is one-way (server → Discord). To type commands directly in Discord:

### Step 1: Create a Discord Bot
1. Go to https://discord.com/developers/applications
2. Click **New Application** → name it `SEF Trading Bot`
3. Go to **Bot** tab → click **Reset Token** → **Copy** the token
4. Under **Privileged Gateway Intents**, enable **MESSAGE CONTENT INTENT**
5. Save changes

### Step 2: Invite Bot to Your Server
1. Go to **OAuth2 → URL Generator**
2. Scopes: check `bot`
3. Bot Permissions: check `Send Messages`, `Read Messages/View Channels`, `Embed Links`
4. Copy the generated URL → open in browser → select your server → Authorize

### Step 3: Set the Token
```powershell
$env:DISCORD_BOT_TOKEN = "your_bot_token_here"
```

### Step 4: Restart Server
```bash
python unified_server.py
```

You should see:
```
✅ Discord bot connected as SEF Trading Bot (123456789)
   Listening in 1 server(s)
🚀 Discord bot listener started (two-way mode)
```

### Two-Way Commands (type in Discord)
| Command | What it does |
|---------|-------------|
| `!help` | Show all commands |
| `!price SPY` | Quick price check |
| `!alerts` | View active alerts |
| `!stats` | Trade performance |
| `!scan NVDA` | Run scanner |
| `!setup TSLA` | Setup analysis |
| `!brief` | Market brief (SPY, QQQ, IWM, DIA) |
| `!queue` | Task queue status |
| `@SEF Trading Bot how is NVDA?` | Natural language (Haiku-powered) |

You can also type commands without the `!` prefix — `price SPY` works too.

## Next Steps

1. **Connect Clawdbot** — Route Clawdbot's output through the Discord webhook for unified notifications.
2. **Morning brief cron** — Schedule `POST /discord/command {"command": "brief"}` at 9:25 AM ET daily.
