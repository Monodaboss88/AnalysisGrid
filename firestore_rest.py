"""
Firestore REST API Client
==========================
Reads Firestore data using Firebase Auth REST API + Firestore REST API.
No service account key needed - uses email/password bot user.
"""

import os
import json
import time
import httpx
from typing import Dict, List, Optional
from datetime import datetime

# Firebase project config (from web app)
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "AIzaSyB4oBeoFzFV2oS5ZxEvQ4TIRRVJFC3nl-w")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "analysis-grid")

# Bot user credentials (set these on Railway)
FIREBASE_BOT_EMAIL = os.environ.get("FIREBASE_BOT_EMAIL", "")
FIREBASE_BOT_PASSWORD = os.environ.get("FIREBASE_BOT_PASSWORD", "")

# Firebase Auth REST API
AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"

# Firestore REST API base
FIRESTORE_BASE = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents"

# Token cache
_token_cache = {
    "id_token": None,
    "expires_at": 0,
    "uid": None
}


def _sign_in() -> Optional[str]:
    """Sign in bot user via Firebase Auth REST API, return ID token"""
    global _token_cache
    
    # Return cached token if still valid (with 5 min buffer)
    if _token_cache["id_token"] and time.time() < _token_cache["expires_at"] - 300:
        return _token_cache["id_token"]
    
    if not FIREBASE_BOT_EMAIL or not FIREBASE_BOT_PASSWORD:
        print("❌ FIREBASE_BOT_EMAIL or FIREBASE_BOT_PASSWORD not set")
        return None
    
    try:
        resp = httpx.post(AUTH_URL, json={
            "email": FIREBASE_BOT_EMAIL,
            "password": FIREBASE_BOT_PASSWORD,
            "returnSecureToken": True
        }, headers={"Referer": "https://analysis-grid.web.app"}, timeout=10)
        
        if resp.status_code != 200:
            error = resp.json().get("error", {}).get("message", "Unknown error")
            print(f"❌ Firebase Auth error: {error}")
            return None
        
        data = resp.json()
        _token_cache["id_token"] = data["idToken"]
        _token_cache["expires_at"] = time.time() + int(data.get("expiresIn", 3600))
        _token_cache["uid"] = data.get("localId", "")
        print(f"✅ Firebase bot signed in as {FIREBASE_BOT_EMAIL} (uid: {_token_cache['uid']})")
        return data["idToken"]
        
    except Exception as e:
        print(f"❌ Firebase sign-in failed: {e}")
        return None


def _parse_firestore_value(value: Dict) -> any:
    """Convert Firestore REST API value format to Python"""
    if "stringValue" in value:
        return value["stringValue"]
    elif "integerValue" in value:
        return int(value["integerValue"])
    elif "doubleValue" in value:
        return float(value["doubleValue"])
    elif "booleanValue" in value:
        return value["booleanValue"]
    elif "timestampValue" in value:
        return value["timestampValue"]
    elif "nullValue" in value:
        return None
    elif "arrayValue" in value:
        return [_parse_firestore_value(v) for v in value["arrayValue"].get("values", [])]
    elif "mapValue" in value:
        return {k: _parse_firestore_value(v) for k, v in value["mapValue"].get("fields", {}).items()}
    return str(value)


def _parse_document(doc: Dict) -> Dict:
    """Parse a Firestore document into a plain dict"""
    fields = doc.get("fields", {})
    result = {}
    for key, value in fields.items():
        result[key] = _parse_firestore_value(value)
    # Extract doc ID from name path
    name = doc.get("name", "")
    result["_doc_id"] = name.split("/")[-1] if name else ""
    result["_path"] = name
    return result


def get_all_user_ids() -> List[str]:
    """List all user document IDs in the users collection"""
    token = _sign_in()
    if not token:
        return []
    
    try:
        headers = {"Authorization": f"Bearer {token}", "Referer": "https://analysis-grid.web.app"}
        resp = httpx.get(
            f"{FIRESTORE_BASE}/users",
            headers=headers,
            timeout=10
        )
        
        if resp.status_code != 200:
            print(f"❌ Firestore list users error: {resp.status_code} {resp.text[:200]}")
            return []
        
        data = resp.json()
        documents = data.get("documents", [])
        user_ids = []
        for doc in documents:
            name = doc.get("name", "")
            uid = name.split("/")[-1]
            user_ids.append(uid)
        return user_ids
        
    except Exception as e:
        print(f"❌ Firestore list users failed: {e}")
        return []


def get_user_alerts(user_id: str, symbol: str = None) -> List[Dict]:
    """Get alerts for a specific user, optionally filtered by symbol"""
    token = _sign_in()
    if not token:
        return []
    
    try:
        headers = {"Authorization": f"Bearer {token}", "Referer": "https://analysis-grid.web.app"}
        resp = httpx.get(
            f"{FIRESTORE_BASE}/users/{user_id}/alerts",
            headers=headers,
            timeout=10
        )
        
        if resp.status_code != 200:
            # 403 = permission denied (bot can't read this user), skip silently
            if resp.status_code != 403:
                print(f"⚠️ Firestore alerts error for {user_id}: {resp.status_code}")
            return []
        
        data = resp.json()
        documents = data.get("documents", [])
        alerts = []
        
        for doc in documents:
            alert = _parse_document(doc)
            if symbol:
                alert_symbol = str(alert.get("symbol", "")).upper()
                if alert_symbol != symbol.upper():
                    continue
            alerts.append(alert)
        
        return alerts
        
    except Exception as e:
        print(f"❌ Firestore get alerts failed: {e}")
        return []


def search_all_alerts(symbol: str = None) -> List[Dict]:
    """Search alerts across ALL users, optionally filtered by symbol.
    Uses Firestore runQuery (collection group query via REST API)."""
    token = _sign_in()
    if not token:
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Use structured query to search across all users' alerts
        # First try: iterate known users
        user_ids = get_all_user_ids()
        all_alerts = []
        
        for uid in user_ids:
            alerts = get_user_alerts(uid, symbol)
            for alert in alerts:
                alert["_user_id"] = uid
            all_alerts.extend(alerts)
        
        return all_alerts
        
    except Exception as e:
        print(f"❌ Firestore search all alerts failed: {e}")
        return []


def get_bot_uid() -> Optional[str]:
    """Get the bot user's Firebase UID"""
    _sign_in()
    return _token_cache.get("uid")


def is_available() -> bool:
    """Check if REST client can authenticate"""
    return bool(FIREBASE_BOT_EMAIL and FIREBASE_BOT_PASSWORD)


def get_status() -> Dict:
    """Get status info for debugging"""
    return {
        "available": is_available(),
        "bot_email": FIREBASE_BOT_EMAIL or "(not set)",
        "project_id": FIREBASE_PROJECT_ID,
        "has_cached_token": bool(_token_cache["id_token"]),
        "token_expires": datetime.fromtimestamp(_token_cache["expires_at"]).isoformat() if _token_cache["expires_at"] else None,
        "bot_uid": _token_cache.get("uid")
    }
