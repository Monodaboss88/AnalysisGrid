"""
Notification Service — Firebase Cloud Messaging
=================================================
Multi-device push notification service.

Architecture:
  - Stores FCM tokens per user with device type (web/android/ios)
  - Sends via firebase-admin messaging SDK
  - Supports individual, multi-device, and topic-based sends
  - App-ready: native apps just register their FCM token — same backend

Integration:
  - trade_monitor.on_close(callback) fires notify_trade_close()
  - Future: scanner signals, options flow, journal reminders

Usage:
  svc = get_notification_service()
  svc.register_token(user_id, token, device_type='web')
  svc.send_to_user(user_id, title, body, data={})
  svc.notify_trade_close(event)
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("notifications")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [NOTIFY] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, messaging, firestore
    _firebase_available = True
except ImportError:
    _firebase_available = False
    logger.warning("firebase-admin not installed — notifications disabled")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DeviceToken:
    """Represents a single device registration"""
    token: str
    device_type: str  # 'web', 'android', 'ios'
    user_id: str
    registered_at: str = ""
    last_used: str = ""
    active: bool = True

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = datetime.now(timezone.utc).isoformat()
        if not self.last_used:
            self.last_used = self.registered_at


@dataclass
class NotificationPrefs:
    """Per-user notification preferences"""
    trade_closes: bool = True
    scanner_signals: bool = False
    options_flow: bool = False
    journal_reminders: bool = False
    quiet_hours_start: str = ""   # "22:00" 
    quiet_hours_end: str = ""     # "08:00"


# =============================================================================
# NOTIFICATION SERVICE
# =============================================================================

class NotificationService:
    """Firebase Cloud Messaging notification service"""

    def __init__(self):
        self._db = None
        self._tokens: Dict[str, List[DeviceToken]] = {}  # user_id → tokens
        self._prefs: Dict[str, NotificationPrefs] = {}    # user_id → prefs
        self._send_count = 0
        self._error_count = 0
        self._last_sent = None
        self._init_firebase()

    def _init_firebase(self):
        """Initialize Firebase Admin SDK (reuse existing app if present)"""
        if not _firebase_available:
            logger.warning("Firebase Admin SDK not available")
            return

        try:
            from firebase_init import init_firebase_app
            app = init_firebase_app()
            if not app:
                logger.warning("⚠️ Firebase not initialized — FCM sending disabled")
                return

            self._db = firestore.client()
            self._load_tokens_from_firestore()
        except Exception as e:
            logger.error("Firebase init error: %s", e)

    # ========================================================================
    # TOKEN MANAGEMENT
    # ========================================================================

    def register_token(self, user_id: str, token: str, device_type: str = "web") -> bool:
        """Register an FCM token for a user/device"""
        if not token or not user_id:
            return False

        # Deduplicate — don't store same token twice
        existing = self._tokens.get(user_id, [])
        for dt in existing:
            if dt.token == token:
                dt.active = True
                dt.last_used = datetime.now(timezone.utc).isoformat()
                dt.device_type = device_type
                self._save_token_to_firestore(user_id, dt)
                logger.info("🔄 Updated token for %s (%s)", user_id[:8], device_type)
                return True

        # New token
        device = DeviceToken(
            token=token,
            device_type=device_type,
            user_id=user_id
        )
        if user_id not in self._tokens:
            self._tokens[user_id] = []
        self._tokens[user_id].append(device)
        self._save_token_to_firestore(user_id, device)
        logger.info("✅ Registered %s token for %s", device_type, user_id[:8])
        return True

    def unregister_token(self, user_id: str, token: str) -> bool:
        """Remove a specific FCM token"""
        if user_id not in self._tokens:
            return False

        before = len(self._tokens[user_id])
        self._tokens[user_id] = [dt for dt in self._tokens[user_id] if dt.token != token]
        removed = before - len(self._tokens[user_id])

        if removed > 0:
            self._remove_token_from_firestore(user_id, token)
            logger.info("🗑️ Removed token for %s", user_id[:8])
        return removed > 0

    def get_user_tokens(self, user_id: str) -> List[DeviceToken]:
        """Get all active tokens for a user"""
        return [dt for dt in self._tokens.get(user_id, []) if dt.active]

    def get_token_count(self) -> int:
        """Total active tokens across all users"""
        return sum(len(self.get_user_tokens(uid)) for uid in self._tokens)

    # ========================================================================
    # PREFERENCES
    # ========================================================================

    def get_prefs(self, user_id: str) -> dict:
        """Get user notification preferences"""
        prefs = self._prefs.get(user_id, NotificationPrefs())
        return asdict(prefs)

    def update_prefs(self, user_id: str, updates: dict) -> dict:
        """Update user notification preferences"""
        if user_id not in self._prefs:
            self._prefs[user_id] = NotificationPrefs()

        prefs = self._prefs[user_id]
        for key, val in updates.items():
            if hasattr(prefs, key):
                setattr(prefs, key, val)

        self._save_prefs_to_firestore(user_id, prefs)
        return asdict(prefs)

    # ========================================================================
    # SEND NOTIFICATIONS
    # ========================================================================

    def send_to_user(self, user_id: str, title: str, body: str,
                     data: Optional[Dict[str, str]] = None,
                     image: Optional[str] = None,
                     category: str = "general") -> dict:
        """Send push notification to all of a user's devices"""
        if not _firebase_available:
            return {"sent": 0, "errors": ["firebase-admin not available"]}

        tokens = self.get_user_tokens(user_id)
        if not tokens:
            logger.debug("No tokens for user %s", user_id[:8])
            return {"sent": 0, "errors": ["no tokens registered"]}

        results = {"sent": 0, "failed": 0, "errors": []}

        for dt in tokens:
            try:
                msg = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body,
                        image=image
                    ),
                    data=self._prepare_data(data, category),
                    token=dt.token,
                    webpush=messaging.WebpushConfig(
                        notification=messaging.WebpushNotification(
                            icon="/icons/icon-192.svg",
                            badge="/icons/badge-72.svg",
                            tag=category,
                            renotify=True,
                            actions=[
                                messaging.WebpushNotificationAction(
                                    action="open_desk",
                                    title="Open Desk"
                                )
                            ]
                        ),
                        fcm_options=messaging.WebpushFCMOptions(
                            link="https://analysis-grid.web.app/desk.html"
                        )
                    )
                )

                messaging.send(msg)
                results["sent"] += 1
                self._send_count += 1
                self._last_sent = datetime.now(timezone.utc).isoformat()

                # Update last_used
                dt.last_used = self._last_sent

            except messaging.UnregisteredError:
                # Token is no longer valid — mark inactive
                dt.active = False
                results["failed"] += 1
                results["errors"].append(f"token expired ({dt.device_type})")
                logger.info("🗑️ Token expired for %s (%s)", user_id[:8], dt.device_type)

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(str(e))
                self._error_count += 1
                logger.error("FCM send error: %s", e)

        return results

    def send_to_all(self, title: str, body: str,
                    data: Optional[Dict[str, str]] = None,
                    category: str = "general") -> dict:
        """Broadcast to all registered users"""
        total_sent = 0
        total_failed = 0
        for user_id in list(self._tokens.keys()):
            r = self.send_to_user(user_id, title, body, data, category=category)
            total_sent += r["sent"]
            total_failed += r["failed"]
        return {"sent": total_sent, "failed": total_failed}

    # ========================================================================
    # TRADE MONITOR INTEGRATION
    # ========================================================================

    def notify_trade_close(self, event: dict):
        """
        Called by trade_monitor.on_close() callback.
        event has: symbol, direction, result, pnl, entry, exit_price,
                   r_multiple, user_id, trade_id, timestamp
        """
        user_id = event.get("user_id", "")
        if not user_id:
            logger.warning("Trade close event has no user_id — skipping notification")
            return

        # Check user prefs
        prefs = self._prefs.get(user_id, NotificationPrefs())
        if not prefs.trade_closes:
            return

        symbol = event.get("symbol", "???")
        direction = event.get("direction", "")
        result = event.get("result", "")
        pnl = event.get("pnl", 0)
        r_mult = event.get("r_multiple", 0)
        exit_price = event.get("exit_price", 0)

        # Build notification
        if result == "WIN":
            icon = "🟢"
            title = f"{icon} {symbol} Target Hit!"
            body = f"{direction} closed @ ${exit_price:.2f} — +${pnl:.2f} ({r_mult:+.1f}R)"
        elif result == "LOSS":
            icon = "🔴"
            title = f"{icon} {symbol} Stop Hit"
            body = f"{direction} closed @ ${exit_price:.2f} — -${abs(pnl):.2f} ({r_mult:+.1f}R)"
        else:
            icon = "⚪"
            title = f"{symbol} Trade Closed"
            body = f"{direction} @ ${exit_price:.2f} — ${pnl:+.2f}"

        data = {
            "type": "trade_close",
            "symbol": symbol,
            "trade_id": event.get("trade_id", ""),
            "result": result,
            "pnl": str(pnl)
        }

        self.send_to_user(user_id, title, body, data, category="trade_close")
        logger.info("📤 Notified %s: %s %s %s", user_id[:8], result, direction, symbol)

    # ========================================================================
    # SCANNER SIGNAL INTEGRATION (future)
    # ========================================================================

    def notify_scanner_signal(self, user_id: str, symbol: str, signal: str,
                               confidence: int, timeframe: str):
        """Send scanner signal notification"""
        prefs = self._prefs.get(user_id, NotificationPrefs())
        if not prefs.scanner_signals:
            return

        colors = {"RED": "🔴", "GREEN": "🟢", "YELLOW": "🟡"}
        icon = colors.get(signal, "⚪")

        title = f"{icon} {symbol} — {signal} Signal"
        body = f"{confidence}% confidence on {timeframe} timeframe"

        self.send_to_user(user_id, title, body,
                         data={"type": "scanner_signal", "symbol": symbol, "signal": signal},
                         category="scanner_signal")

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(self) -> dict:
        """Get notification service status"""
        return {
            "active": _firebase_available and self._db is not None,
            "total_users": len(self._tokens),
            "total_tokens": self.get_token_count(),
            "tokens_by_type": self._count_by_type(),
            "sent_count": self._send_count,
            "error_count": self._error_count,
            "last_sent": self._last_sent
        }

    def _count_by_type(self) -> dict:
        counts = {"web": 0, "android": 0, "ios": 0}
        for tokens in self._tokens.values():
            for dt in tokens:
                if dt.active and dt.device_type in counts:
                    counts[dt.device_type] += 1
        return counts

    # ========================================================================
    # FIRESTORE PERSISTENCE
    # ========================================================================

    def _load_tokens_from_firestore(self):
        """Load all registered tokens from Firestore"""
        if not self._db:
            return

        try:
            docs = self._db.collection("fcm_tokens").stream()
            count = 0
            for doc in docs:
                data = doc.to_dict()
                user_id = data.get("user_id", doc.id)
                token = DeviceToken(
                    token=data.get("token", ""),
                    device_type=data.get("device_type", "web"),
                    user_id=user_id,
                    registered_at=data.get("registered_at", ""),
                    last_used=data.get("last_used", ""),
                    active=data.get("active", True)
                )
                if token.token and token.active:
                    if user_id not in self._tokens:
                        self._tokens[user_id] = []
                    self._tokens[user_id].append(token)
                    count += 1
            logger.info("📥 Loaded %d FCM tokens from Firestore", count)
        except Exception as e:
            logger.error("Failed to load tokens: %s", e)

    def _save_token_to_firestore(self, user_id: str, dt: DeviceToken):
        """Persist a token to Firestore"""
        if not self._db:
            return
        try:
            doc_id = f"{user_id}_{self._hash_token(dt.token)}"
            self._db.collection("fcm_tokens").document(doc_id).set({
                "user_id": user_id,
                "token": dt.token,
                "device_type": dt.device_type,
                "registered_at": dt.registered_at,
                "last_used": dt.last_used,
                "active": dt.active
            })
        except Exception as e:
            logger.error("Failed to save token: %s", e)

    def _remove_token_from_firestore(self, user_id: str, token: str):
        """Remove a token from Firestore"""
        if not self._db:
            return
        try:
            doc_id = f"{user_id}_{self._hash_token(token)}"
            self._db.collection("fcm_tokens").document(doc_id).delete()
        except Exception as e:
            logger.error("Failed to remove token: %s", e)

    def _save_prefs_to_firestore(self, user_id: str, prefs: NotificationPrefs):
        """Persist notification preferences"""
        if not self._db:
            return
        try:
            self._db.collection("notification_prefs").document(user_id).set(asdict(prefs))
        except Exception as e:
            logger.error("Failed to save prefs: %s", e)

    def _load_prefs_from_firestore(self, user_id: str) -> NotificationPrefs:
        """Load user prefs from Firestore"""
        if not self._db:
            return NotificationPrefs()
        try:
            doc = self._db.collection("notification_prefs").document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                prefs = NotificationPrefs(**{k: v for k, v in data.items() if hasattr(NotificationPrefs, k)})
                self._prefs[user_id] = prefs
                return prefs
        except Exception:
            pass
        return NotificationPrefs()

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _prepare_data(data: Optional[Dict], category: str) -> Dict[str, str]:
        """Ensure all data values are strings (FCM requirement)"""
        result = {"category": category, "timestamp": datetime.now(timezone.utc).isoformat()}
        if data:
            for k, v in data.items():
                result[k] = str(v)
        return result

    @staticmethod
    def _hash_token(token: str) -> str:
        """Short hash for document ID"""
        import hashlib
        return hashlib.sha256(token.encode()).hexdigest()[:12]


# =============================================================================
# SINGLETON
# =============================================================================

_instance: Optional[NotificationService] = None

def get_notification_service() -> NotificationService:
    global _instance
    if _instance is None:
        _instance = NotificationService()
    return _instance
