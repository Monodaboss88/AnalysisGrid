"""
Stripe Payment Integration
===========================
Handle subscriptions, webhooks, and billing
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict

try:
    import stripe
    stripe_available = True
except ImportError:
    stripe_available = False
    print("⚠️ stripe not installed. Run: pip install stripe")

from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel

from auth_middleware import UserManager, SUBSCRIPTION_TIERS


# =============================================================================
# CONFIGURATION
# =============================================================================

# Initialize Stripe
stripe_api_key = os.getenv('STRIPE_SECRET_KEY', '')
stripe_webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')

if stripe_available and stripe_api_key:
    stripe.api_key = stripe_api_key

# Price IDs from your Stripe Dashboard
STRIPE_PRICES = {
    "basic": os.getenv('STRIPE_PRICE_BASIC', 'price_basic_monthly'),
    "pro": os.getenv('STRIPE_PRICE_PRO', 'price_pro_monthly'),
    "premium": os.getenv('STRIPE_PRICE_PREMIUM', 'price_premium_monthly')
}


# =============================================================================
# ROUTER
# =============================================================================

payment_router = APIRouter(prefix="/api/payments", tags=["Payments"])


# =============================================================================
# MODELS
# =============================================================================

class CreateCheckoutRequest(BaseModel):
    """Request to create Stripe checkout session"""
    tier: str  # basic, pro, premium
    success_url: str
    cancel_url: str


class CustomerPortalRequest(BaseModel):
    """Request for customer portal"""
    return_url: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@payment_router.get("/plans")
async def get_plans():
    """Get available subscription plans"""
    plans = []
    for tier_id, tier in SUBSCRIPTION_TIERS.items():
        if tier_id == "free":
            continue
        plans.append({
            "id": tier_id,
            "name": tier["name"],
            "price": tier.get("price", 0),
            "ai_calls": tier["ai_calls_per_day"],
            "scans": tier["scans_per_day"],
            "features": tier["features"]
        })
    return {"plans": plans}


@payment_router.post("/create-checkout")
async def create_checkout_session(request: CreateCheckoutRequest, uid: str = Header(...)):
    """Create Stripe checkout session for subscription"""
    if not stripe_available or not stripe_api_key:
        raise HTTPException(status_code=503, detail="Payment system not configured")
    
    if request.tier not in STRIPE_PRICES:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {request.tier}")
    
    try:
        # Create or get Stripe customer
        # Get user from DB
        from auth_middleware import UserManager
        import sqlite3
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT email, stripe_customer_id FROM users WHERE uid = ?', (uid,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        
        email, customer_id = row
        
        # Create Stripe customer if needed
        if not customer_id:
            customer = stripe.Customer.create(
                email=email,
                metadata={"firebase_uid": uid}
            )
            customer_id = customer.id
            
            # Save to DB
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('UPDATE users SET stripe_customer_id = ? WHERE uid = ?', (customer_id, uid))
            conn.commit()
            conn.close()
        
        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': STRIPE_PRICES[request.tier],
                'quantity': 1
            }],
            mode='subscription',
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            metadata={
                "firebase_uid": uid,
                "tier": request.tier
            }
        )
        
        return {"checkout_url": session.url, "session_id": session.id}
    
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@payment_router.post("/customer-portal")
async def create_portal_session(request: CustomerPortalRequest, uid: str = Header(...)):
    """Create Stripe customer portal session for managing subscription"""
    if not stripe_available or not stripe_api_key:
        raise HTTPException(status_code=503, detail="Payment system not configured")
    
    try:
        import sqlite3
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT stripe_customer_id FROM users WHERE uid = ?', (uid,))
        row = c.fetchone()
        conn.close()
        
        if not row or not row[0]:
            raise HTTPException(status_code=404, detail="No subscription found")
        
        session = stripe.billing_portal.Session.create(
            customer=row[0],
            return_url=request.return_url
        )
        
        return {"portal_url": session.url}
    
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@payment_router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks for subscription events"""
    if not stripe_available:
        raise HTTPException(status_code=503, detail="Stripe not available")
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        if stripe_webhook_secret:
            event = stripe.Webhook.construct_event(
                payload, sig_header, stripe_webhook_secret
            )
        else:
            # Dev mode - parse without verification
            event = json.loads(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")
    
    # Handle events
    event_type = event.get('type', '')
    data = event.get('data', {}).get('object', {})
    
    if event_type == 'checkout.session.completed':
        # Subscription started
        uid = data.get('metadata', {}).get('firebase_uid')
        tier = data.get('metadata', {}).get('tier', 'basic')
        
        if uid:
            # Set subscription for 1 month
            expires = (datetime.now() + timedelta(days=30)).isoformat()
            UserManager.update_subscription(uid, tier, expires, data.get('customer'))
            print(f"✅ Subscription activated: {uid} -> {tier}")
    
    elif event_type == 'customer.subscription.updated':
        # Subscription changed
        customer_id = data.get('customer')
        status = data.get('status')
        
        # Find user by customer ID
        import sqlite3
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT uid FROM users WHERE stripe_customer_id = ?', (customer_id,))
        row = c.fetchone()
        
        if row:
            uid = row[0]
            if status == 'active':
                # Get tier from price
                price_id = data.get('items', {}).get('data', [{}])[0].get('price', {}).get('id')
                tier = 'basic'
                for t, p in STRIPE_PRICES.items():
                    if p == price_id:
                        tier = t
                        break
                
                expires = datetime.fromtimestamp(data.get('current_period_end', 0)).isoformat()
                UserManager.update_subscription(uid, tier, expires)
            elif status in ['canceled', 'unpaid', 'past_due']:
                UserManager.update_subscription(uid, 'free', None)
        
        conn.close()
    
    elif event_type == 'customer.subscription.deleted':
        # Subscription canceled
        customer_id = data.get('customer')
        
        import sqlite3
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT uid FROM users WHERE stripe_customer_id = ?', (customer_id,))
        row = c.fetchone()
        
        if row:
            UserManager.update_subscription(row[0], 'free', None)
            print(f"⚠️ Subscription canceled: {row[0]}")
        
        conn.close()
    
    return {"status": "ok"}


@payment_router.get("/subscription")
async def get_subscription_status(uid: str = Header(...)):
    """Get current subscription status for a user"""
    import sqlite3
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT subscription_tier, subscription_expires FROM users WHERE uid = ?', (uid,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return {"tier": "free", "expires": None, "limits": SUBSCRIPTION_TIERS["free"]}
    
    tier = row[0] or "free"
    expires = row[1]
    
    # Check if expired
    if expires:
        try:
            exp_date = datetime.fromisoformat(expires)
            if exp_date < datetime.now():
                tier = "free"
        except:
            pass
    
    return {
        "tier": tier,
        "expires": expires,
        "limits": SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])
    }
