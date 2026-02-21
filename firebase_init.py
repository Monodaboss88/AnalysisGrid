"""
Firebase Initializer — Shared bootstrap for firebase-admin
============================================================
Handles two credential types:
  1. Service account key JSON (FIREBASE_SERVICE_ACCOUNT env var)
  2. Authorized user / ADC credentials (GOOGLE_CREDENTIALS_JSON env var)

The org policy on this project blocks service account key creation,
so we support ADC (Application Default Credentials) as a fallback.
"""

import os
import json
import tempfile

_initialized = False
_app = None


def init_firebase_app():
    """
    Initialize firebase_admin once. Safe to call multiple times.
    Returns the Firebase app or None.
    """
    global _initialized, _app

    if _initialized:
        return _app

    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        print("⚠️ firebase-admin not installed")
        _initialized = True
        return None

    # Already initialized?
    try:
        _app = firebase_admin.get_app()
        _initialized = True
        return _app
    except ValueError:
        pass

    # Collect all possible credential sources
    sa_json = os.getenv('FIREBASE_SERVICE_ACCOUNT', '')
    gc_json = os.getenv('GOOGLE_CREDENTIALS_JSON', '')
    cred_json = sa_json or gc_json

    print(f"🔍 Firebase init: FIREBASE_SERVICE_ACCOUNT={'set' if sa_json else 'empty'} "
          f"GOOGLE_CREDENTIALS_JSON={'set' if gc_json else 'empty'}")

    if not cred_json:
        print("⚠️ No Firebase credentials found — Firestore/FCM/Auth disabled")
        _initialized = True
        return None

    try:
        cred_dict = json.loads(cred_json)
        cred_type = cred_dict.get('type', 'unknown')
        print(f"🔍 Credential type: {cred_type}")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse credential JSON: {e}")
        _initialized = True
        return None

    # ── Method 1: Service account key (classic) ──
    if cred_type == 'service_account':
        try:
            cred = credentials.Certificate(cred_dict)
            _app = firebase_admin.initialize_app(cred)
            _initialized = True
            print("✅ Firebase initialized (service account key)")
            return _app
        except Exception as e:
            print(f"⚠️ Service account init failed: {e}")

    # ── Method 2: ADC / authorized_user credentials ──
    if cred_type == 'authorized_user':
        try:
            cred_dict = json.loads(cred_json)
            if cred_dict.get('type') == 'authorized_user':
                # Write to temp file so google-auth can pick it up
                tmp = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', delete=False, prefix='firebase_adc_'
                )
                json.dump(cred_dict, tmp)
                tmp.close()
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = tmp.name

                # Initialize with Application Default Credentials
                _app = firebase_admin.initialize_app(options={
                    'projectId': cred_dict.get('quota_project_id', 'analysis-grid')
                })
                _initialized = True
                print("✅ Firebase initialized (ADC / authorized_user)")
                return _app
        except Exception as e:
            print(f"⚠️ ADC init failed: {e}")

    # ── Method 3: GOOGLE_APPLICATION_CREDENTIALS file path ──
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        try:
            _app = firebase_admin.initialize_app(options={
                'projectId': 'analysis-grid'
            })
            _initialized = True
            print("✅ Firebase initialized (GOOGLE_APPLICATION_CREDENTIALS file)")
            return _app
        except Exception as e:
            print(f"⚠️ GAC file init failed: {e}")

    print("⚠️ No Firebase credentials found — Firestore/FCM/Auth disabled")
    _initialized = True
    return None
