"""
Admin / Enterprise Endpoints
=============================
Org management, user claims, module visibility, and audit log.
All endpoints require admin role (via custom claims).
"""

import logging
from typing import Optional, List
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
