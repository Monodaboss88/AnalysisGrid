"""
Admin / Enterprise Endpoints
=============================
Org management, user claims, module visibility, audit log,
white-label branding, AI control, session policy, cross-trader reporting,
and audit export.
"""

import logging
import csv
import io
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auth_middleware import (
    OrgContext, require_auth, require_admin, require_manager,
    set_user_claims, get_user_claims, list_org_users,
    log_audit, query_audit, ROLES, DEFAULT_ORG_MODULES,
)

logger = logging.getLogger(__name__)
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── Firestore helpers (org config) ─────────────────────────────────

def _get_db():
    """Lazy Firestore client."""
    try:
        from firebase_init import init_firebase_app
        init_firebase_app()
        from firebase_admin import firestore
        return firestore.client()
    except Exception:
        return None


def _get_org_config(org_id: str) -> dict:
    """Read /orgs/{orgId}/config doc. Returns defaults if missing."""
    db = _get_db()
    if not db:
        return {"modules": DEFAULT_ORG_MODULES, "name": org_id, "theme": "dark"}
    try:
        doc = db.collection("orgs").document(org_id).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        logger.debug(f"Org config read failed: {e}")
    return {"modules": DEFAULT_ORG_MODULES, "name": org_id, "theme": "dark"}


def _set_org_config(org_id: str, data: dict):
    """Write /orgs/{orgId}/config doc."""
    db = _get_db()
    if not db:
        raise HTTPException(status_code=503, detail="Firestore unavailable")
    try:
        db.collection("orgs").document(org_id).set(data, merge=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save org config: {e}")


# ═════════════════════════════════════════════════════════════════════
# ORG CONFIG
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/org/config")
async def get_org_config(user: OrgContext = Depends(require_manager)):
    """Get org configuration (modules, name, theme)."""
    config = _get_org_config(user.org_id)
    return {"org_id": user.org_id, **config}


class OrgConfigUpdate(BaseModel):
    name: Optional[str] = None
    modules: Optional[List[str]] = None
    theme: Optional[str] = None
    ai_enabled: Optional[bool] = None


@admin_router.put("/org/config")
async def update_org_config(body: OrgConfigUpdate,
                            user: OrgContext = Depends(require_admin)):
    """Update org configuration. Admin only."""
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow().isoformat()
    updates["updated_by"] = user.uid
    _set_org_config(user.org_id, updates)
    log_audit(user, "org_config_update", resource=user.org_id,
              detail=str(list(updates.keys())))
    return {"status": "ok", "updated": list(updates.keys())}


@admin_router.get("/org/modules")
async def get_org_modules(user: OrgContext = Depends(require_auth)):
    """Get enabled modules for the current user's org.
    Any authenticated user can read this (controls their UI)."""
    config = _get_org_config(user.org_id)
    return {
        "org_id": user.org_id,
        "modules": config.get("modules", DEFAULT_ORG_MODULES),
        "all_modules": DEFAULT_ORG_MODULES,
    }


# ═════════════════════════════════════════════════════════════════════
# USER MANAGEMENT
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/users")
async def list_users(user: OrgContext = Depends(require_admin)):
    """List all users in the admin's org."""
    users = list_org_users(user.org_id)
    log_audit(user, "list_users", resource=user.org_id)
    return {"org_id": user.org_id, "users": users, "count": len(users)}


class SetRoleRequest(BaseModel):
    uid: str
    role: str


@admin_router.post("/users/set-role")
async def set_user_role(body: SetRoleRequest,
                        user: OrgContext = Depends(require_admin)):
    """Set a user's role within the admin's org."""
    if body.role not in ROLES:
        raise HTTPException(status_code=400,
                            detail=f"Invalid role. Choose from: {list(ROLES.keys())}")
    # Prevent non-superadmins from creating superadmins
    if body.role == "superadmin" and user.role != "superadmin":
        raise HTTPException(status_code=403, detail="Only superadmins can grant superadmin")
    # Verify target user is in same org (or caller is superadmin)
    if user.role != "superadmin":
        target_claims = get_user_claims(body.uid)
        if target_claims.get("orgId") != user.org_id:
            raise HTTPException(status_code=403,
                                detail="Cannot modify users outside your organization")

    ok = set_user_claims(body.uid, user.org_id, body.role)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to set claims")

    log_audit(user, "set_role", resource=body.uid, detail=f"role={body.role}")
    return {"status": "ok", "uid": body.uid, "role": body.role, "org_id": user.org_id}


class InviteUserRequest(BaseModel):
    uid: str
    role: str = "trader"


@admin_router.post("/users/invite")
async def invite_user_to_org(body: InviteUserRequest,
                             user: OrgContext = Depends(require_admin)):
    """Add an existing Firebase user to this org with a role."""
    if body.role not in ROLES:
        raise HTTPException(status_code=400,
                            detail=f"Invalid role. Choose from: {list(ROLES.keys())}")
    if body.role in ("superadmin", "admin") and user.role != "superadmin":
        if body.role == "superadmin":
            raise HTTPException(status_code=403, detail="Only superadmins can grant superadmin")

    ok = set_user_claims(body.uid, user.org_id, body.role)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to set claims")

    log_audit(user, "invite_user", resource=body.uid,
              detail=f"org={user.org_id}, role={body.role}")
    return {"status": "ok", "uid": body.uid, "org_id": user.org_id, "role": body.role}


class RemoveUserRequest(BaseModel):
    uid: str


@admin_router.post("/users/remove")
async def remove_user_from_org(body: RemoveUserRequest,
                               user: OrgContext = Depends(require_admin)):
    """Remove a user from this org (sets them to personal/trader)."""
    if user.role != "superadmin":
        target_claims = get_user_claims(body.uid)
        if target_claims.get("orgId") != user.org_id:
            raise HTTPException(status_code=403,
                                detail="Cannot modify users outside your organization")

    ok = set_user_claims(body.uid, "personal", "trader")
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to remove user from org")

    log_audit(user, "remove_user", resource=body.uid, detail=f"from org={user.org_id}")
    return {"status": "ok", "uid": body.uid, "removed_from": user.org_id}


# ═════════════════════════════════════════════════════════════════════
# MY CONTEXT (any user)
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/me")
async def get_my_context(user: OrgContext = Depends(require_auth)):
    """Return the current user's org context, role, and enabled modules."""
    config = _get_org_config(user.org_id)
    return {
        **user.to_dict(),
        "modules": config.get("modules", DEFAULT_ORG_MODULES),
        "org_name": config.get("name", user.org_id),
    }


# ═════════════════════════════════════════════════════════════════════
# AUDIT LOG
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/audit")
async def get_audit_log(
    uid: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 200,
    user: OrgContext = Depends(require_manager),
):
    """Query audit log — scoped to the caller's org.
    Managers and admins can see all org activity.
    Optional filters: uid, start/end dates (ISO), limit."""
    rows = query_audit(user.org_id, uid=uid, start=start, end=end, limit=limit)
    return {"org_id": user.org_id, "entries": rows, "count": len(rows)}


# ═════════════════════════════════════════════════════════════════════
# SUPERADMIN: ORG CREATION
# ═════════════════════════════════════════════════════════════════════

class CreateOrgRequest(BaseModel):
    org_id: str
    name: str
    admin_uid: str
    modules: Optional[List[str]] = None


@admin_router.post("/orgs/create")
async def create_org(body: CreateOrgRequest,
                     user: OrgContext = Depends(require_auth)):
    """Create a new organization and set the first admin.
    Superadmin only."""
    if user.role != "superadmin":
        raise HTTPException(status_code=403, detail="Superadmin required")

    config = {
        "name": body.name,
        "modules": body.modules or DEFAULT_ORG_MODULES,
        "theme": "dark",
        "created_at": datetime.utcnow().isoformat(),
        "created_by": user.uid,
    }
    _set_org_config(body.org_id, config)
    set_user_claims(body.admin_uid, body.org_id, "admin")

    log_audit(user, "create_org", resource=body.org_id,
              detail=f"admin={body.admin_uid}")
    return {"status": "ok", "org_id": body.org_id, "admin_uid": body.admin_uid}


# ═════════════════════════════════════════════════════════════════════
# SSO CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

class SSOConfigUpdate(BaseModel):
    provider: Optional[str] = None       # "google", "microsoft", "saml", "oidc"
    domains: Optional[List[str]] = None  # e.g. ["acme.com", "acme.org"]
    auto_provision: Optional[bool] = None  # auto-add matching domain users
    default_role: Optional[str] = None   # role for auto-provisioned users
    enforce_sso: Optional[bool] = None   # block email/password for org domains
    oidc_issuer: Optional[str] = None    # OIDC issuer URL (if provider=oidc)
    oidc_client_id: Optional[str] = None
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None
    saml_certificate: Optional[str] = None


@admin_router.get("/sso/config")
async def get_sso_config(user: OrgContext = Depends(require_admin)):
    """Get SSO configuration for the org."""
    config = _get_org_config(user.org_id)
    sso = config.get("sso", {})
    return {"org_id": user.org_id, "sso": sso}


@admin_router.put("/sso/config")
async def update_sso_config(body: SSOConfigUpdate,
                            user: OrgContext = Depends(require_admin)):
    """Update SSO configuration. Admin only."""
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Validate provider
    valid_providers = ["google", "microsoft", "saml", "oidc", "none"]
    if "provider" in updates and updates["provider"] not in valid_providers:
        raise HTTPException(status_code=400,
                            detail=f"Invalid provider. Choose: {valid_providers}")

    # Validate default_role
    if "default_role" in updates and updates["default_role"] not in ROLES:
        raise HTTPException(status_code=400,
                            detail=f"Invalid role. Choose: {list(ROLES.keys())}")

    # Normalize domains to lowercase
    if "domains" in updates:
        updates["domains"] = [d.lower().strip() for d in updates["domains"] if d.strip()]

    # Merge into sso sub-object
    config = _get_org_config(user.org_id)
    sso = config.get("sso", {})
    sso.update(updates)
    sso["updated_at"] = datetime.utcnow().isoformat()
    sso["updated_by"] = user.uid

    _set_org_config(user.org_id, {"sso": sso})
    log_audit(user, "sso_config_update", resource=user.org_id,
              detail=str(list(updates.keys())))
    return {"status": "ok", "sso": sso}


@admin_router.get("/sso/domain-lookup")
async def domain_lookup(domain: str):
    """Public endpoint — given an email domain, return the org's SSO config.
    Used by the login page to detect SSO orgs. No auth required."""
    domain = domain.lower().strip()
    if not domain:
        raise HTTPException(status_code=400, detail="Domain required")

    db = _get_db()
    if not db:
        return {"sso": None, "org_id": None}

    # Search all orgs for matching domain
    try:
        orgs = db.collection("orgs").stream()
        for doc in orgs:
            data = doc.to_dict()
            sso = data.get("sso", {})
            domains = sso.get("domains", [])
            if domain in domains:
                # Return safe subset (no secrets)
                return {
                    "org_id": doc.id,
                    "org_name": data.get("name", doc.id),
                    "sso": {
                        "provider": sso.get("provider"),
                        "enforce_sso": sso.get("enforce_sso", False),
                        "auto_provision": sso.get("auto_provision", False),
                    }
                }
    except Exception as e:
        logger.warning(f"Domain lookup error: {e}")

    return {"sso": None, "org_id": None}


@admin_router.post("/sso/auto-provision")
async def auto_provision_user(request: Request):
    """Called after SSO login — auto-assign user to org if domain matches.
    Requires a valid Firebase token (user just signed in via SSO)."""
    from auth_middleware import require_auth as _ra
    user = await _ra(request)

    email = user.email
    if not email or '@' not in email:
        return {"status": "skipped", "reason": "no email"}

    domain = email.split('@')[1].lower()

    # Already in a non-personal org? Skip
    if user.org_id != "personal":
        return {"status": "skipped", "reason": "already_in_org", "org_id": user.org_id}

    db = _get_db()
    if not db:
        return {"status": "skipped", "reason": "no_db"}

    # Find org with matching domain + auto_provision
    try:
        orgs = db.collection("orgs").stream()
        for doc in orgs:
            data = doc.to_dict()
            sso = data.get("sso", {})
            domains = sso.get("domains", [])
            if domain in domains and sso.get("auto_provision", False):
                org_id = doc.id
                role = sso.get("default_role", "trader")
                ok = set_user_claims(user.uid, org_id, role)
                if ok:
                    log_audit(user, "sso_auto_provision", resource=org_id,
                              detail=f"email={email}, role={role}")
                    return {"status": "provisioned", "org_id": org_id, "role": role}
                else:
                    return {"status": "error", "reason": "claims_failed"}
    except Exception as e:
        logger.warning(f"Auto-provision error: {e}")

    return {"status": "skipped", "reason": "no_matching_org"}


# ═════════════════════════════════════════════════════════════════════
# DATA MIGRATION (Phase 6 — personal → org)
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/migration/status")
async def migration_status(uid: Optional[str] = None,
                           user: OrgContext = Depends(require_admin)):
    """Check migration status for a user (or self)."""
    from firestore_store import get_firestore
    fs = get_firestore()
    target_uid = uid or user.uid
    status = fs.get_migration_status(target_uid, user.org_id)
    return {"uid": target_uid, "org_id": user.org_id, "status": status}


class MigrateRequest(BaseModel):
    uid: Optional[str] = None
    collections: Optional[List[str]] = None


@admin_router.post("/migration/run")
async def run_migration(body: MigrateRequest,
                        user: OrgContext = Depends(require_admin)):
    """Migrate a user's personal data into the org.
    Admin migrates a specific user, or self if uid not provided."""
    from firestore_store import get_firestore
    fs = get_firestore()
    target_uid = body.uid or user.uid

    if user.org_id == "personal":
        raise HTTPException(status_code=400,
                            detail="Cannot migrate to personal org")

    result = fs.migrate_user_to_org(
        target_uid, user.org_id,
        collections=body.collections
    )
    log_audit(user, "data_migration", resource=target_uid,
              detail=f"org={user.org_id}, result={result}")
    return {"status": "ok", "uid": target_uid, "org_id": user.org_id,
            "migrated": result}


@admin_router.post("/migration/run-all")
async def run_migration_all(user: OrgContext = Depends(require_admin)):
    """Migrate ALL users in the org at once."""
    from firestore_store import get_firestore
    fs = get_firestore()

    if user.org_id == "personal":
        raise HTTPException(status_code=400,
                            detail="Cannot migrate to personal org")

    users = list_org_users(user.org_id)
    results = {}
    for u in users:
        uid = u.get("uid")
        if uid:
            results[uid] = fs.migrate_user_to_org(uid, user.org_id)

    log_audit(user, "bulk_migration", resource=user.org_id,
              detail=f"users={len(results)}")
    return {"status": "ok", "org_id": user.org_id,
            "users_migrated": len(results), "details": results}


# ═════════════════════════════════════════════════════════════════════
# WHITE-LABEL BRANDING
# ═════════════════════════════════════════════════════════════════════

class BrandingUpdate(BaseModel):
    app_name: Optional[str] = None          # e.g. "Apex Trading"
    logo_url: Optional[str] = None          # URL to logo image
    logo_mark: Optional[str] = None         # 2-char mark shown in header circle
    accent_color: Optional[str] = None      # primary accent hex e.g. "#6366f1"
    accent_rgb: Optional[str] = None        # CSS rgb values e.g. "99,102,241"
    bg_color: Optional[str] = None          # background hex
    card_color: Optional[str] = None        # card bg hex
    signal_text: Optional[str] = None       # e.g. "APEX ACTIVE" in signal bar
    module_labels: Optional[Dict[str, str]] = None  # rename modules {"buffett_scanner":"Value Scanner"}


@admin_router.get("/branding")
async def get_branding(user: OrgContext = Depends(require_auth)):
    """Get branding config for the org. Any authed user can read (needed for desk.html)."""
    config = _get_org_config(user.org_id)
    return {"org_id": user.org_id, "branding": config.get("branding", {})}


@admin_router.put("/branding")
async def update_branding(body: BrandingUpdate,
                          user: OrgContext = Depends(require_admin)):
    """Update white-label branding. Admin only."""
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    config = _get_org_config(user.org_id)
    branding = config.get("branding", {})
    branding.update(updates)
    branding["updated_at"] = datetime.utcnow().isoformat()

    _set_org_config(user.org_id, {"branding": branding})
    log_audit(user, "branding_update", resource=user.org_id,
              detail=str(list(updates.keys())))
    return {"status": "ok", "branding": branding}


@admin_router.get("/branding/public")
async def get_branding_public(org_id: str = "personal"):
    """Public endpoint — fetch branding by org_id. No auth (used at load time)."""
    if not org_id or org_id == "personal":
        return {"branding": {}}
    config = _get_org_config(org_id)
    return {"branding": config.get("branding", {})}


# ═════════════════════════════════════════════════════════════════════
# AI ORG-SCOPED CONTROL
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/ai/status")
async def get_ai_status(user: OrgContext = Depends(require_auth)):
    """Get AI enabled/disabled status for the org."""
    config = _get_org_config(user.org_id)
    return {
        "org_id": user.org_id,
        "ai_enabled": config.get("ai_enabled", True),
    }


class AIControlUpdate(BaseModel):
    ai_enabled: bool


@admin_router.put("/ai/control")
async def update_ai_control(body: AIControlUpdate,
                            user: OrgContext = Depends(require_admin)):
    """Enable/disable AI features for the entire org. Admin only."""
    _set_org_config(user.org_id, {"ai_enabled": body.ai_enabled})
    log_audit(user, "ai_control_update", resource=user.org_id,
              detail=f"ai_enabled={body.ai_enabled}")
    return {"status": "ok", "ai_enabled": body.ai_enabled}


# ═════════════════════════════════════════════════════════════════════
# SESSION POLICY
# ═════════════════════════════════════════════════════════════════════

class SessionPolicyUpdate(BaseModel):
    idle_timeout_minutes: Optional[int] = None   # 0 = no timeout
    max_session_hours: Optional[int] = None      # 0 = unlimited
    require_reauth_on_return: Optional[bool] = None


@admin_router.get("/session-policy")
async def get_session_policy(user: OrgContext = Depends(require_auth)):
    """Get session policy for the org."""
    config = _get_org_config(user.org_id)
    defaults = {"idle_timeout_minutes": 0, "max_session_hours": 0,
                "require_reauth_on_return": False}
    policy = config.get("session_policy", defaults)
    return {"org_id": user.org_id, "session_policy": policy}


@admin_router.put("/session-policy")
async def update_session_policy(body: SessionPolicyUpdate,
                                user: OrgContext = Depends(require_admin)):
    """Update session timeout policy. Admin only."""
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    config = _get_org_config(user.org_id)
    policy = config.get("session_policy", {})
    policy.update(updates)
    _set_org_config(user.org_id, {"session_policy": policy})
    log_audit(user, "session_policy_update", resource=user.org_id,
              detail=str(updates))
    return {"status": "ok", "session_policy": policy}


# ═════════════════════════════════════════════════════════════════════
# DESK VIEW — CROSS-TRADER AGGREGATION (manager+)
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/desk-view/summary")
async def desk_view_summary(user: OrgContext = Depends(require_manager)):
    """Aggregate summary for the manager desk view:
    total org trades, total PnL, active alerts, and per-trader stats."""
    from firestore_store import get_firestore
    fs = get_firestore()

    trades = fs.get_org_trades(user.org_id) or []
    alerts = fs.get_org_alerts(user.org_id) or []
    users = list_org_users(user.org_id)

    # Aggregate PnL
    total_pnl = 0
    wins = 0
    losses = 0
    per_trader: Dict[str, dict] = {}
    for t in trades:
        uid = t.get("uid", "unknown")
        pnl = float(t.get("pnl", 0) or 0)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

        if uid not in per_trader:
            per_trader[uid] = {"uid": uid, "trades": 0, "pnl": 0, "wins": 0, "losses": 0}
        per_trader[uid]["trades"] += 1
        per_trader[uid]["pnl"] += pnl
        if pnl > 0:
            per_trader[uid]["wins"] += 1
        elif pnl < 0:
            per_trader[uid]["losses"] += 1

    # Attach name/email from user list
    user_map = {u["uid"]: u for u in users}
    trader_stats = []
    for uid, stats in per_trader.items():
        info = user_map.get(uid, {})
        stats["email"] = info.get("email", "")
        stats["name"] = info.get("name", "")
        stats["pnl"] = round(stats["pnl"], 2)
        trader_stats.append(stats)

    trader_stats.sort(key=lambda x: x["pnl"], reverse=True)

    return {
        "org_id": user.org_id,
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
        "active_alerts": len(alerts),
        "traders": trader_stats,
        "member_count": len(users),
    }


@admin_router.get("/desk-view/alerts")
async def desk_view_alerts(user: OrgContext = Depends(require_manager)):
    """All active alerts across the org."""
    from firestore_store import get_firestore
    fs = get_firestore()
    alerts = fs.get_org_alerts(user.org_id) or []

    users = list_org_users(user.org_id)
    user_map = {u["uid"]: u.get("email", "") for u in users}
    for a in alerts:
        a["trader_email"] = user_map.get(a.get("uid", ""), "")

    return {"org_id": user.org_id, "alerts": alerts, "count": len(alerts)}


@admin_router.get("/desk-view/trades")
async def desk_view_trades(
    limit: int = 100,
    user: OrgContext = Depends(require_manager)
):
    """Recent trades across the org with trader attribution."""
    from firestore_store import get_firestore
    fs = get_firestore()
    trades = fs.get_org_trades(user.org_id) or []

    users = list_org_users(user.org_id)
    user_map = {u["uid"]: u.get("email", "") for u in users}
    for t in trades:
        t["trader_email"] = user_map.get(t.get("uid", ""), "")

    # Sort by most recent
    trades.sort(key=lambda x: x.get("opened_at", x.get("timestamp", "")), reverse=True)
    return {"org_id": user.org_id, "trades": trades[:limit], "total": len(trades)}


# ═════════════════════════════════════════════════════════════════════
# AUDIT EXPORT
# ═════════════════════════════════════════════════════════════════════

@admin_router.get("/audit/export")
async def export_audit_log(
    format: str = "csv",
    uid: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 5000,
    user: OrgContext = Depends(require_admin),
):
    """Export audit log as CSV or JSON. Admin only."""
    rows = query_audit(user.org_id, uid=uid, start=start, end=end, limit=limit)
    log_audit(user, "audit_export", resource=user.org_id,
              detail=f"format={format}, rows={len(rows)}")

    if format == "json":
        return {"org_id": user.org_id, "entries": rows, "count": len(rows)}

    # CSV export
    output = io.StringIO()
    writer = csv.DictWriter(output,
                            fieldnames=["id", "timestamp", "uid", "email",
                                        "action", "resource", "detail", "ip"])
    writer.writeheader()
    writer.writerows(rows)
    csv_content = output.getvalue()

    from fastapi.responses import Response
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=audit_{user.org_id}_{datetime.utcnow().strftime('%Y%m%d')}.csv"}
    )
