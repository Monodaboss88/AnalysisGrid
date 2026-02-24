"""
Firebase Authentication Middleware
===================================
Handles user auth, subscription tiers, usage limits,
and enterprise multi-tenancy (orgId + role via custom claims).
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
import sqlite3

logger = logging.getLogger(__name__)

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
    firebase_available = True
except ImportError:
    firebase_available = False
    print("⚠️ firebase-admin not installed. Run: pip install firebase-admin")

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# =============================================================================
# ENTERPRISE ROLES & ORG CONTEXT
# =============================================================================

ROLES = {
    "superadmin": 100,   # Platform owner — can see all orgs
    "admin":       80,   # Org admin — manages users, modules, configs
    "manager":     60,   # Desk manager — read-only on all org traders
    "trader":      40,   # Full platform, scoped to own data
    "viewer":      20,   # Read-only dashboards & reports
}

DEFAULT_ORG_MODULES = [
    "quick_scanner", "deep_analysis", "charts", "trades",
    "capital_engine", "seasons", "sustainability", "research_builder",
    "catalyst_scanner", "buffett_scanner", "options_flow", "war_room",
    "regime_scanner", "trading_cards", "journal", "backtest_lab",
    "premarket_checklist", "session_notes",
]


class OrgContext:
    """Parsed from Firebase custom claims on every request."""
    __slots__ = ("uid", "email", "name", "org_id", "role", "role_level",
                 "subscription_tier", "limits")

    def __init__(self, uid: str, email: str = "", name: str = "",
                 org_id: str = "personal", role: str = "trader",
                 subscription_tier: str = "free"):
        self.uid = uid
        self.email = email
        self.name = name
        self.org_id = org_id
        self.role = role
        self.role_level = ROLES.get(role, 0)
        self.subscription_tier = subscription_tier
        self.limits = SUBSCRIPTION_TIERS.get(subscription_tier,
                                              SUBSCRIPTION_TIERS["free"])

    def is_at_least(self, min_role: str) -> bool:
        return self.role_level >= ROLES.get(min_role, 0)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid, "email": self.email, "name": self.name,
            "org_id": self.org_id, "role": self.role,
            "subscription_tier": self.subscription_tier,
            "limits": self.limits,
        }


# =============================================================================
# AUDIT LOGGER
# =============================================================================

_audit_db_ready = False

def _init_audit_db():
    global _audit_db_ready
    if _audit_db_ready:
        return
    try:
        conn = sqlite3.connect('audit.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                org_id TEXT NOT NULL,
                uid TEXT NOT NULL,
                email TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                detail TEXT,
                ip TEXT
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_log(org_id, timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_audit_uid ON audit_log(uid, timestamp)')
        conn.commit()
        conn.close()
        _audit_db_ready = True
    except Exception as e:
        logger.error(f"Audit DB init failed: {e}")


def log_audit(ctx: OrgContext, action: str, resource: str = "",
              detail: str = "", ip: str = ""):
    """Write one audit row. Fire-and-forget — never raises."""
    try:
        _init_audit_db()
        conn = sqlite3.connect('audit.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO audit_log (timestamp, org_id, uid, email, action, resource, detail, ip)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.utcnow().isoformat(), ctx.org_id, ctx.uid,
              ctx.email, action, resource, detail, ip))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"Audit write failed: {e}")


def query_audit(org_id: str, uid: str = None,
                start: str = None, end: str = None,
                limit: int = 200) -> List[dict]:
    """Query audit log — always scoped to one org."""
    _init_audit_db()
    conn = sqlite3.connect('audit.db')
    c = conn.cursor()
    sql = "SELECT id, timestamp, uid, email, action, resource, detail, ip FROM audit_log WHERE org_id = ?"
    params: list = [org_id]
    if uid:
        sql += " AND uid = ?"
        params.append(uid)
    if start:
        sql += " AND timestamp >= ?"
        params.append(start)
    if end:
        sql += " AND timestamp <= ?"
        params.append(end)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = c.execute(sql, params).fetchall()
    conn.close()
    return [
        {"id": r[0], "timestamp": r[1], "uid": r[2], "email": r[3],
         "action": r[4], "resource": r[5], "detail": r[6], "ip": r[7]}
        for r in rows
    ]


# =============================================================================
# CONFIGURATION
# =============================================================================

# Subscription tiers and limits
SUBSCRIPTION_TIERS = {
    "free": {
        "name": "Free Trial",
        "ai_calls_per_day": 5,
        "scans_per_day": 20,
        "watchlists": 1,
        "journal_entries": 10,
        "features": ["basic_scan", "single_timeframe"]
    },
    "basic": {
        "name": "Basic",
        "price": 19.99,
        "ai_calls_per_day": 50,
        "scans_per_day": 200,
        "watchlists": 5,
        "journal_entries": 100,
        "features": ["basic_scan", "single_timeframe", "ai_commentary", "journal"]
    },
    "pro": {
        "name": "Pro Trader",
        "price": 49.99,
        "ai_calls_per_day": 200,
        "scans_per_day": 1000,
        "watchlists": 20,
        "journal_entries": -1,  # Unlimited
        "features": ["basic_scan", "mtf_analysis", "ai_commentary", "journal", "alerts", "range_watcher"]
    },
    "premium": {
        "name": "Premium",
        "price": 99.99,
        "ai_calls_per_day": -1,  # Unlimited
        "scans_per_day": -1,
        "watchlists": -1,
        "journal_entries": -1,
        "features": ["all"]
    }
}


# =============================================================================
# DATABASE FOR USER TRACKING
# =============================================================================

def init_user_db():
    """Initialize SQLite database for user tracking"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            uid TEXT PRIMARY KEY,
            email TEXT,
            display_name TEXT,
            subscription_tier TEXT DEFAULT 'free',
            subscription_expires TEXT,
            stripe_customer_id TEXT,
            created_at TEXT,
            last_login TEXT
        )
    ''')
    
    # Usage tracking table
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT,
            date TEXT,
            ai_calls INTEGER DEFAULT 0,
            scans INTEGER DEFAULT 0,
            UNIQUE(uid, date)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize on import
init_user_db()


# =============================================================================
# FIREBASE INITIALIZATION
# =============================================================================

firebase_app = None

def init_firebase(service_account_path: str = None, service_account_json: str = None):
    """Initialize Firebase Admin SDK"""
    global firebase_app
    
    if not firebase_available:
        print("❌ Firebase Admin SDK not available")
        return False
    
    if firebase_app:
        return True
    
    try:
        if service_account_json:
            # From environment variable (for Railway)
            cred_dict = json.loads(service_account_json)
            cred = credentials.Certificate(cred_dict)
            firebase_app = firebase_admin.initialize_app(cred)
        elif service_account_path and os.path.exists(service_account_path):
            # From file
            cred = credentials.Certificate(service_account_path)
            firebase_app = firebase_admin.initialize_app(cred)
        else:
            # Use shared initializer (handles service account + ADC)
            from firebase_init import init_firebase_app
            firebase_app = init_firebase_app()
            if not firebase_app:
                print("⚠️ No Firebase credentials found")
                return False
        
        print("✅ Firebase Admin initialized")
        return True
    
    except Exception as e:
        print(f"❌ Firebase init error: {e}")
        return False


# =============================================================================
# USER MANAGEMENT
# =============================================================================

class UserManager:
    """Manage user subscriptions and usage"""
    
    @staticmethod
    def get_or_create_user(uid: str, email: str = "", name: str = "") -> Dict:
        """Get user from DB or create new one"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute('SELECT * FROM users WHERE uid = ?', (uid,))
        row = c.fetchone()
        
        if row:
            user = {
                "uid": row[0],
                "email": row[1],
                "display_name": row[2],
                "subscription_tier": row[3],
                "subscription_expires": row[4],
                "stripe_customer_id": row[5],
                "created_at": row[6],
                "last_login": row[7]
            }
            # Update last login
            c.execute('UPDATE users SET last_login = ? WHERE uid = ?', 
                     (datetime.now().isoformat(), uid))
        else:
            # Create new user with free tier
            now = datetime.now().isoformat()
            c.execute('''
                INSERT INTO users (uid, email, display_name, subscription_tier, created_at, last_login)
                VALUES (?, ?, ?, 'free', ?, ?)
            ''', (uid, email, name, now, now))
            
            user = {
                "uid": uid,
                "email": email,
                "display_name": name,
                "subscription_tier": "free",
                "subscription_expires": None,
                "stripe_customer_id": None,
                "created_at": now,
                "last_login": now
            }
        
        conn.commit()
        conn.close()
        return user
    
    @staticmethod
    def update_subscription(uid: str, tier: str, expires: str = None, stripe_id: str = None):
        """Update user subscription"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute('''
            UPDATE users 
            SET subscription_tier = ?, subscription_expires = ?, stripe_customer_id = ?
            WHERE uid = ?
        ''', (tier, expires, stripe_id, uid))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_usage(uid: str) -> Dict:
        """Get today's usage for a user"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute('SELECT ai_calls, scans FROM usage WHERE uid = ? AND date = ?', (uid, today))
        row = c.fetchone()
        
        conn.close()
        
        if row:
            return {"ai_calls": row[0], "scans": row[1]}
        return {"ai_calls": 0, "scans": 0}
    
    @staticmethod
    def increment_usage(uid: str, usage_type: str):
        """Increment usage counter"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Insert or update
        c.execute('''
            INSERT INTO usage (uid, date, ai_calls, scans)
            VALUES (?, ?, 0, 0)
            ON CONFLICT(uid, date) DO NOTHING
        ''', (uid, today))
        
        if usage_type == "ai":
            c.execute('UPDATE usage SET ai_calls = ai_calls + 1 WHERE uid = ? AND date = ?', (uid, today))
        elif usage_type == "scan":
            c.execute('UPDATE usage SET scans = scans + 1 WHERE uid = ? AND date = ?', (uid, today))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def check_limit(uid: str, tier: str, usage_type: str) -> tuple[bool, str]:
        """Check if user is within limits. Returns (allowed, message)"""
        limits = SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])
        usage = UserManager.get_usage(uid)
        
        if usage_type == "ai":
            limit = limits["ai_calls_per_day"]
            current = usage["ai_calls"]
            if limit != -1 and current >= limit:
                return False, f"AI call limit reached ({current}/{limit}). Upgrade for more!"
        
        elif usage_type == "scan":
            limit = limits["scans_per_day"]
            current = usage["scans"]
            if limit != -1 and current >= limit:
                return False, f"Scan limit reached ({current}/{limit}). Upgrade for more!"
        
        return True, "OK"


# =============================================================================
# AUTHENTICATION DEPENDENCY
# =============================================================================

security = HTTPBearer(auto_error=False)

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[OrgContext]:
    """
    Verify Firebase ID token and return OrgContext.
    Reads custom claims: orgId, role.
    Returns None if no auth provided (for public endpoints).
    """
    if not credentials:
        return None
    
    if not firebase_available or not firebase_app:
        # Dev mode - accept any token, return superadmin context
        return OrgContext(
            uid="dev-user", email="dev@test.com", name="Dev User",
            org_id="dev-org", role="superadmin", subscription_tier="premium"
        )
    
    try:
        token = credentials.credentials
        decoded = firebase_auth.verify_id_token(token)
        
        uid   = decoded['uid']
        email = decoded.get('email', '')
        name  = decoded.get('name', '')

        # ── Enterprise claims ──
        org_id = decoded.get('orgId', 'personal')
        role   = decoded.get('role', 'trader')
        if role not in ROLES:
            role = 'trader'
        
        # Get/create user in our DB (subscription tracking)
        user = UserManager.get_or_create_user(uid, email, name)
        tier = user.get("subscription_tier", "free")
        
        ctx = OrgContext(uid=uid, email=email, name=name,
                         org_id=org_id, role=role, subscription_tier=tier)

        # Audit the request (lightweight — just the endpoint path)
        ip = request.client.host if request.client else ""
        log_audit(ctx, "api_call", resource=str(request.url.path), ip=ip)
        
        return ctx
    
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication: {str(e)}")


async def require_auth(user: OrgContext = Depends(get_current_user)) -> OrgContext:
    """Require authentication - raises 401 if not authenticated"""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def require_admin(user: OrgContext = Depends(require_auth)) -> OrgContext:
    """Require admin or superadmin role"""
    if not user.is_at_least("admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def require_manager(user: OrgContext = Depends(require_auth)) -> OrgContext:
    """Require manager, admin, or superadmin role"""
    if not user.is_at_least("manager"):
        raise HTTPException(status_code=403, detail="Manager access required")
    return user


async def require_subscription(user: OrgContext = Depends(require_auth)) -> OrgContext:
    """Require paid subscription"""
    if user.subscription_tier == "free":
        raise HTTPException(status_code=403, detail="Paid subscription required")
    return user


def check_feature(feature: str):
    """Decorator to check if user has access to a feature"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: OrgContext = Depends(require_auth), **kwargs):
            features = user.limits.get("features", [])
            
            if "all" not in features and feature not in features:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Feature '{feature}' requires upgrade. Current tier: {user.subscription_tier}"
                )
            
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# USAGE TRACKING HELPERS
# =============================================================================

async def track_ai_usage(user: OrgContext = Depends(require_auth)):
    """Track AI call and check limits"""
    allowed, msg = UserManager.check_limit(user.uid, user.subscription_tier, "ai")
    if not allowed:
        raise HTTPException(status_code=429, detail=msg)
    
    UserManager.increment_usage(user.uid, "ai")
    return user


async def track_scan_usage(user: OrgContext = Depends(require_auth)):
    """Track scan and check limits"""
    allowed, msg = UserManager.check_limit(user.uid, user.subscription_tier, "scan")
    if not allowed:
        raise HTTPException(status_code=429, detail=msg)
    
    UserManager.increment_usage(user.uid, "scan")
    return user


# =============================================================================
# CLAIMS MANAGEMENT (for admin endpoints)
# =============================================================================

def set_user_claims(uid: str, org_id: str, role: str) -> bool:
    """Set custom claims on a Firebase user (orgId + role).
    Must be called from an admin endpoint."""
    if not firebase_available or not firebase_app:
        logger.warning("Firebase not available — cannot set claims")
        return False
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}. Must be one of {list(ROLES.keys())}")
    try:
        firebase_auth.set_custom_user_claims(uid, {"orgId": org_id, "role": role})
        logger.info(f"Set claims for {uid}: orgId={org_id}, role={role}")
        return True
    except Exception as e:
        logger.error(f"Failed to set claims for {uid}: {e}")
        return False


def get_user_claims(uid: str) -> dict:
    """Read current custom claims for a user."""
    if not firebase_available or not firebase_app:
        return {}
    try:
        user = firebase_auth.get_user(uid)
        return user.custom_claims or {}
    except Exception as e:
        logger.error(f"Failed to get claims for {uid}: {e}")
        return {}


def list_org_users(org_id: str) -> List[dict]:
    """List all Firebase users with orgId == org_id (via custom claims).
    Note: iterates all users — intended for admin use, not hot path."""
    if not firebase_available or not firebase_app:
        return []
    users = []
    try:
        page = firebase_auth.list_users()
        for u in page.iterate_all():
            claims = u.custom_claims or {}
            if claims.get("orgId") == org_id:
                users.append({
                    "uid": u.uid,
                    "email": u.email,
                    "name": u.display_name or "",
                    "role": claims.get("role", "trader"),
                    "org_id": org_id,
                    "disabled": u.disabled,
                    "last_sign_in": u.user_metadata.last_sign_in_timestamp,
                })
    except Exception as e:
        logger.error(f"Failed to list users for org {org_id}: {e}")
    return users
