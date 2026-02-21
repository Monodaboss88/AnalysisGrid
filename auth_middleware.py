"""
Firebase Authentication Middleware
===================================
Handles user auth, subscription tiers, and usage limits
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps
import sqlite3

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
) -> Optional[Dict]:
    """
    Verify Firebase ID token and return user info.
    Returns None if no auth provided (for public endpoints).
    """
    if not credentials:
        return None
    
    if not firebase_available or not firebase_app:
        # Dev mode - accept any token
        return {
            "uid": "dev-user",
            "email": "dev@test.com",
            "subscription_tier": "premium",
            "limits": SUBSCRIPTION_TIERS["premium"]
        }
    
    try:
        token = credentials.credentials
        decoded = firebase_auth.verify_id_token(token)
        
        uid = decoded['uid']
        email = decoded.get('email', '')
        name = decoded.get('name', '')
        
        # Get/create user in our DB
        user = UserManager.get_or_create_user(uid, email, name)
        user["limits"] = SUBSCRIPTION_TIERS.get(user["subscription_tier"], SUBSCRIPTION_TIERS["free"])
        
        return user
    
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication: {str(e)}")


async def require_auth(user: Dict = Depends(get_current_user)) -> Dict:
    """Require authentication - raises 401 if not authenticated"""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def require_subscription(user: Dict = Depends(require_auth)) -> Dict:
    """Require paid subscription"""
    if user["subscription_tier"] == "free":
        raise HTTPException(status_code=403, detail="Paid subscription required")
    return user


def check_feature(feature: str):
    """Decorator to check if user has access to a feature"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: Dict = Depends(require_auth), **kwargs):
            limits = user.get("limits", SUBSCRIPTION_TIERS["free"])
            features = limits.get("features", [])
            
            if "all" not in features and feature not in features:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Feature '{feature}' requires upgrade. Current tier: {user['subscription_tier']}"
                )
            
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# USAGE TRACKING HELPERS
# =============================================================================

async def track_ai_usage(user: Dict = Depends(require_auth)):
    """Track AI call and check limits"""
    allowed, msg = UserManager.check_limit(user["uid"], user["subscription_tier"], "ai")
    if not allowed:
        raise HTTPException(status_code=429, detail=msg)
    
    UserManager.increment_usage(user["uid"], "ai")
    return user


async def track_scan_usage(user: Dict = Depends(require_auth)):
    """Track scan and check limits"""
    allowed, msg = UserManager.check_limit(user["uid"], user["subscription_tier"], "scan")
    if not allowed:
        raise HTTPException(status_code=429, detail=msg)
    
    UserManager.increment_usage(user["uid"], "scan")
    return user
