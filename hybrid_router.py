"""
Hybrid Router — Haiku-Powered Message Parser
=============================================
Uses Claude Haiku ($0.25/M input) to classify incoming messages
and route them to the right handler. Cheap parsing, smart routing.

Flow:
  User message → Haiku classifies → Quick task? Handle inline : Queue for heavy processing

Cost: ~$0.0003 per message classification

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from task_queue import Task, TaskType, TaskPriority, QUICK_TASKS

# Anthropic API for Haiku
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HAIKU_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

CLASSIFICATION_PROMPT = """You are a message classifier for a trading system. Classify the user's message into exactly one task type and extract parameters.

Available task types:
- alert_lookup: User wants to see their alerts (e.g., "show alerts", "what alerts do I have", "alerts for SPY")
- trade_stats: User wants trade performance stats (e.g., "how am I doing", "win rate", "trade stats")
- price_check: User wants a quick price (e.g., "price of AAPL", "where is SPY", "TSLA price")
- watchlist: User wants to see watchlist (e.g., "show watchlist", "my symbols")
- scanner_run: User wants to run scanner on a symbol (e.g., "scan MSFT", "analyze NVDA setup")
- setup_analysis: User wants detailed setup analysis (e.g., "full setup for SPY", "entry and targets for QQQ")
- alert_check: User wants to check all alerts against current prices (e.g., "check alerts", "any alerts hit?")
- full_analysis: User wants deep AI analysis (e.g., "deep dive on AAPL", "what do you think about META long term")
- trade_plan: User wants a trade plan generated (e.g., "give me a trade plan for TSLA", "plan a swing trade on AMD")
- market_brief: User wants market overview (e.g., "morning brief", "market update", "what's happening today")
- custom_query: Anything else that doesn't fit above categories

Respond in JSON only:
{
  "task_type": "one_of_the_types_above",
  "symbol": "TICKER or null",
  "params": {},
  "confidence": 0.0-1.0,
  "quick_response": "If this is a simple greeting or acknowledgment, provide a brief response here, otherwise null"
}"""


class HybridRouter:
    """Routes messages using Haiku for classification"""

    def __init__(self):
        self._client = None
        self._classification_cache: Dict[str, Dict] = {}  # Simple LRU-ish cache

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # =========================================================================
    # HAIKU CLASSIFICATION
    # =========================================================================

    async def classify_message(self, text: str) -> Dict:
        """Use Haiku to classify a message into a task type"""
        # Check AI kill switch
        try:
            import unified_server
            if getattr(unified_server, 'AI_KILL_SWITCH', False):
                return self._keyword_classify(text)
        except Exception:
            pass

        if not ANTHROPIC_API_KEY:
            # Fallback: keyword-based classification
            return self._keyword_classify(text)

        try:
            client = await self._get_client()

            response = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": HAIKU_MODEL,
                    "max_tokens": 200,
                    "system": CLASSIFICATION_PROMPT,
                    "messages": [
                        {"role": "user", "content": text}
                    ]
                }
            )

            if response.status_code != 200:
                print(f"⚠️ Haiku API error: {response.status_code}")
                return self._keyword_classify(text)

            result = response.json()
            content = result.get("content", [{}])[0].get("text", "{}")

            # Parse JSON response
            classification = json.loads(content)
            print(f"🧠 Haiku classified: {text[:50]}... → {classification.get('task_type')}")
            return classification

        except json.JSONDecodeError:
            print("⚠️ Haiku returned non-JSON, falling back to keyword classification")
            return self._keyword_classify(text)
        except Exception as e:
            print(f"⚠️ Haiku classification error: {e}")
            return self._keyword_classify(text)

    def _keyword_classify(self, text: str) -> Dict:
        """Fallback keyword-based classification (zero cost)"""
        text_lower = text.lower().strip()

        # Greetings
        if text_lower in ["hi", "hello", "hey", "yo", "sup"]:
            return {
                "task_type": "custom_query",
                "symbol": None,
                "params": {},
                "confidence": 1.0,
                "quick_response": "Hey Rob! What do you need? Try /alerts, /stats, or /scan SYMBOL"
            }

        # Extract symbol if present (look for uppercase tickers)
        symbol = self._extract_symbol(text)

        # Keyword matching
        if any(w in text_lower for w in ["alert", "alerts"]):
            if any(w in text_lower for w in ["check", "hit", "trigger"]):
                return {"task_type": TaskType.ALERT_CHECK, "symbol": symbol, "params": {}, "confidence": 0.8, "quick_response": None}
            return {"task_type": TaskType.ALERT_LOOKUP, "symbol": symbol, "params": {}, "confidence": 0.8, "quick_response": None}

        if any(w in text_lower for w in ["stat", "stats", "performance", "win rate", "how am i"]):
            return {"task_type": TaskType.TRADE_STATS, "symbol": None, "params": {}, "confidence": 0.8, "quick_response": None}

        if any(w in text_lower for w in ["price", "where is", "quote"]):
            return {"task_type": TaskType.PRICE_CHECK, "symbol": symbol, "params": {}, "confidence": 0.8, "quick_response": None}

        if any(w in text_lower for w in ["watchlist", "symbols", "my list"]):
            return {"task_type": TaskType.WATCHLIST, "symbol": None, "params": {}, "confidence": 0.8, "quick_response": None}

        if any(w in text_lower for w in ["scan", "scanner", "analyze"]):
            return {"task_type": TaskType.SCANNER_RUN, "symbol": symbol, "params": {}, "confidence": 0.7, "quick_response": None}

        if any(w in text_lower for w in ["setup", "entry", "target", "levels"]):
            return {"task_type": TaskType.SETUP_ANALYSIS, "symbol": symbol, "params": {}, "confidence": 0.7, "quick_response": None}

        if any(w in text_lower for w in ["deep", "dive", "think", "opinion"]):
            return {"task_type": TaskType.FULL_ANALYSIS, "symbol": symbol, "params": {}, "confidence": 0.6, "quick_response": None}

        if any(w in text_lower for w in ["plan", "trade plan", "swing"]):
            return {"task_type": TaskType.TRADE_PLAN, "symbol": symbol, "params": {}, "confidence": 0.7, "quick_response": None}

        if any(w in text_lower for w in ["brief", "morning", "market", "update", "overview"]):
            return {"task_type": TaskType.MARKET_BRIEF, "symbol": None, "params": {}, "confidence": 0.7, "quick_response": None}

        # Default: if there's a symbol, do a setup analysis
        if symbol:
            return {"task_type": TaskType.SETUP_ANALYSIS, "symbol": symbol, "params": {}, "confidence": 0.5, "quick_response": None}

        # Truly unknown
        return {"task_type": TaskType.CUSTOM_QUERY, "symbol": None, "params": {"raw_text": text}, "confidence": 0.3, "quick_response": None}

    def _extract_symbol(self, text: str) -> Optional[str]:
        """Extract a stock ticker from text"""
        # Common tickers to recognize
        known_tickers = {
            "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META",
            "TSLA", "NVDA", "AMD", "NFLX", "DIS", "BA", "JPM", "GS", "V",
            "MA", "UNH", "JNJ", "PFE", "XOM", "CVX", "WMT", "HD", "NKE",
            "COST", "CRM", "ADBE", "INTC", "PYPL", "SQ", "SHOP", "UBER",
            "ABNB", "COIN", "SNOW", "PLTR", "SOFI", "RIVN", "LCID",
            "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "SLV", "TLT",
            "SMCI", "ARM", "AVGO", "LLY", "PANW", "CRWD", "DDOG",
            "NET", "ZS", "MSTR", "MU", "QCOM", "TXN", "LRCX", "AMAT",
        }

        # Common English words to NEVER treat as tickers
        common_words = {
            "I", "A", "AM", "PM", "THE", "FOR", "AND", "OR", "AT", "TO",
            "IN", "ON", "IS", "IT", "MY", "ME", "DO", "IF", "UP", "NO",
            "SO", "HI", "OK", "OF", "AN", "AS", "BE", "BY", "HE", "WE",
            "ALL", "ANY", "ARE", "BUT", "CAN", "DAY", "DID", "GET", "GOT",
            "HAS", "HAD", "HER", "HIM", "HIS", "HOW", "ITS", "LET", "MAY",
            "NEW", "NOT", "NOW", "OLD", "OUR", "OUT", "OWN", "RAN", "RUN",
            "SAY", "SHE", "THE", "TOO", "TWO", "USE", "WAY", "WHO", "WHY",
            "YES", "YET", "YOU", "WHAT", "WHEN", "WITH", "WILL", "SHOW",
            "GIVE", "HAVE", "JUST", "LIKE", "LONG", "LOOK", "MAKE", "MUCH",
            "NEED", "PLAN", "SCAN", "SOME", "TELL", "THAT", "THEM", "THEN",
            "THIS", "VERY", "WANT", "WERE", "YOUR", "FROM", "BEEN", "ALSO",
            "BACK", "BEST", "BOTH", "CAME", "COME", "DEEP", "DONE", "DOWN",
            "EACH", "EVEN", "FIND", "FULL", "GOOD", "HARD", "HELP", "HERE",
            "HIGH", "KEEP", "KNOW", "LAST", "LEFT", "MADE", "MANY", "MORE",
            "MOST", "MOVE", "MUST", "NAME", "ONLY", "OPEN", "OVER", "RATE",
            "READ", "REAL", "SAID", "SAME", "TAKE", "THAN", "THEY", "TIME",
            "TURN", "USED", "WELL", "WENT", "WORK", "YEAR", "DOES", "LOSE",
            "LOSS", "WINS", "STOP", "EXIT", "SELL", "HOLD", "WAIT", "ONCE",
            "CHECK", "PRICE", "BRIEF", "ALERT", "TRADE", "STATS", "WHERE", "HIT",
            "ABOUT", "THINK", "TODAY", "THOSE", "THEIR", "COULD", "WOULD",
            "STILL", "AFTER", "WHICH", "THESE", "OTHER", "FIRST", "BEING",
        }

        words = text.upper().split()

        # First pass: look for known tickers only
        for word in words:
            clean = word.strip(".,!?()[]{}$#")
            if clean in known_tickers:
                return clean

        # Second pass: look for ticker-like words (2-5 uppercase, not common words)
        for word in words:
            clean = word.strip(".,!?()[]{}$#")
            if (clean.isalpha() and 2 <= len(clean) <= 5
                    and clean == clean.upper()
                    and clean not in common_words):
                return clean

        return None

    # =========================================================================
    # ROUTING
    # =========================================================================

    async def route_message(self, text: str, chat_id: str = "", user_id: str = "") -> Tuple[str, Optional[Task]]:
        """
        Classify and route a message.
        Returns: (response_text, task_or_none)
        - If quick task: returns response directly, no task queued
        - If heavy task: returns acknowledgment, Task object to queue
        """
        classification = await self.classify_message(text)

        task_type = classification.get("task_type", TaskType.CUSTOM_QUERY)
        symbol = classification.get("symbol")
        params = classification.get("params", {})
        quick_response = classification.get("quick_response")

        # Quick response (greetings, etc.)
        if quick_response:
            return quick_response, None

        # Build task
        task = Task(
            task_type=task_type,
            payload={
                "symbol": symbol,
                "raw_text": text,
                **params
            },
            priority=self._get_priority(task_type),
            source="telegram",
            user_id=user_id,
            chat_id=chat_id
        )

        # Quick tasks — return indicator that handler should process inline
        if task_type in QUICK_TASKS:
            return "__QUICK__", task

        # Heavy tasks — acknowledge and queue
        ack_messages = {
            TaskType.SCANNER_RUN: f"🔍 Running scanner{f' on {symbol}' if symbol else ''}... I'll send results shortly.",
            TaskType.SETUP_ANALYSIS: f"📊 Analyzing setup{f' for {symbol}' if symbol else ''}... Stand by.",
            TaskType.ALERT_CHECK: "🔔 Checking all alerts against current prices...",
            TaskType.FULL_ANALYSIS: f"🧠 Running deep analysis{f' on {symbol}' if symbol else ''}... This may take a minute.",
            TaskType.TRADE_PLAN: f"📋 Generating trade plan{f' for {symbol}' if symbol else ''}... Working on it.",
            TaskType.MARKET_BRIEF: "📡 Compiling market brief... One moment.",
            TaskType.CUSTOM_QUERY: "🤔 Processing your request... I'll get back to you.",
        }

        ack = ack_messages.get(task_type, "⏳ Got it, processing...")
        return ack, task

    def _get_priority(self, task_type: str) -> int:
        """Assign priority based on task type"""
        priority_map = {
            TaskType.ALERT_CHECK: TaskPriority.URGENT,
            TaskType.PRICE_CHECK: TaskPriority.URGENT,
            TaskType.ALERT_LOOKUP: TaskPriority.HIGH,
            TaskType.SCANNER_RUN: TaskPriority.HIGH,
            TaskType.SETUP_ANALYSIS: TaskPriority.HIGH,
            TaskType.TRADE_PLAN: TaskPriority.NORMAL,
            TaskType.TRADE_STATS: TaskPriority.NORMAL,
            TaskType.FULL_ANALYSIS: TaskPriority.NORMAL,
            TaskType.MARKET_BRIEF: TaskPriority.LOW,
            TaskType.CUSTOM_QUERY: TaskPriority.LOW,
        }
        return priority_map.get(task_type, TaskPriority.NORMAL)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_router: Optional[HybridRouter] = None

def get_router() -> HybridRouter:
    """Get the global router instance"""
    global _router
    if _router is None:
        _router = HybridRouter()
    return _router
