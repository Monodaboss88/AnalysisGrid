"""
Authorize.net Payment Integration
==================================
Handle subscriptions and recurring billing with Authorize.net
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict

# Authorize.net SDK
try:
    from authorizenet import apicontractsv1
    from authorizenet.apicontrollers import (
        createCustomerProfileController,
        createSubscriptionController,
        cancelSubscriptionController,
        getSubscriptionController,
        createTransactionController
    )
    authorizenet_available = True
except ImportError:
    authorizenet_available = False
    print("⚠️ authorizenet not installed. Run: pip install authorizenet")

from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel

from auth_middleware import UserManager, SUBSCRIPTION_TIERS
import sqlite3


# =============================================================================
# CONFIGURATION
# =============================================================================

# Authorize.net credentials (set in Railway environment)
AUTHORIZE_API_LOGIN_ID = os.getenv('AUTHORIZE_API_LOGIN_ID', '')
AUTHORIZE_TRANSACTION_KEY = os.getenv('AUTHORIZE_TRANSACTION_KEY', '')
AUTHORIZE_SANDBOX = os.getenv('AUTHORIZE_SANDBOX', 'true').lower() == 'true'

# Subscription amounts (in dollars)
SUBSCRIPTION_AMOUNTS = {
    "basic": 19.99,
    "pro": 49.99,
    "premium": 99.99
}


# =============================================================================
# ROUTER
# =============================================================================

payment_router = APIRouter(prefix="/api/payments", tags=["Payments"])


# =============================================================================
# MODELS
# =============================================================================

class CreateSubscriptionRequest(BaseModel):
    """Request to create subscription"""
    tier: str  # basic, pro, premium
    card_number: str
    expiration_date: str  # MMYY format
    card_code: str  # CVV
    first_name: str
    last_name: str
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""


class CancelSubscriptionRequest(BaseModel):
    """Request to cancel subscription"""
    reason: str = ""


# =============================================================================
# AUTHORIZE.NET HELPERS
# =============================================================================

def get_merchant_auth():
    """Get merchant authentication object"""
    merchant_auth = apicontractsv1.merchantAuthenticationType()
    merchant_auth.name = AUTHORIZE_API_LOGIN_ID
    merchant_auth.transactionKey = AUTHORIZE_TRANSACTION_KEY
    return merchant_auth


def save_subscription_to_db(uid: str, subscription_id: str, tier: str):
    """Save Authorize.net subscription ID to user record"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Add column if not exists
    try:
        c.execute('ALTER TABLE users ADD COLUMN authorize_subscription_id TEXT')
    except:
        pass
    
    expires = (datetime.now() + timedelta(days=30)).isoformat()
    c.execute('''
        UPDATE users 
        SET subscription_tier = ?, subscription_expires = ?, authorize_subscription_id = ?
        WHERE uid = ?
    ''', (tier, expires, subscription_id, uid))
    
    conn.commit()
    conn.close()


def get_subscription_id(uid: str) -> Optional[str]:
    """Get Authorize.net subscription ID for user"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        c.execute('SELECT authorize_subscription_id FROM users WHERE uid = ?', (uid,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    except:
        conn.close()
        return None


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
            "price": SUBSCRIPTION_AMOUNTS.get(tier_id, 0),
            "ai_calls": tier["ai_calls_per_day"],
            "scans": tier["scans_per_day"],
            "features": tier["features"]
        })
    return {"plans": plans}


@payment_router.post("/subscribe")
async def create_subscription(request: CreateSubscriptionRequest, uid: str = Header(...)):
    """Create a subscription with Authorize.net"""
    if not authorizenet_available:
        raise HTTPException(status_code=503, detail="Payment system not available")
    
    if not AUTHORIZE_API_LOGIN_ID or not AUTHORIZE_TRANSACTION_KEY:
        raise HTTPException(status_code=503, detail="Payment credentials not configured")
    
    if request.tier not in SUBSCRIPTION_AMOUNTS:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {request.tier}")
    
    try:
        # Get user email
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT email FROM users WHERE uid = ?', (uid,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        
        email = row[0]
        amount = SUBSCRIPTION_AMOUNTS[request.tier]
        
        # Create subscription request
        merchant_auth = get_merchant_auth()
        
        # Payment schedule - monthly
        payment_schedule = apicontractsv1.paymentScheduleType()
        payment_schedule.interval = apicontractsv1.paymentScheduleTypeInterval()
        payment_schedule.interval.length = 1
        payment_schedule.interval.unit = apicontractsv1.ARBSubscriptionUnitEnum.months
        payment_schedule.startDate = datetime.now().strftime('%Y-%m-%d')
        payment_schedule.totalOccurrences = 9999  # Ongoing
        
        # Credit card
        credit_card = apicontractsv1.creditCardType()
        credit_card.cardNumber = request.card_number.replace(' ', '').replace('-', '')
        credit_card.expirationDate = request.expiration_date
        credit_card.cardCode = request.card_code
        
        payment = apicontractsv1.paymentType()
        payment.creditCard = credit_card
        
        # Billing info
        bill_to = apicontractsv1.nameAndAddressType()
        bill_to.firstName = request.first_name
        bill_to.lastName = request.last_name
        if request.address:
            bill_to.address = request.address
        if request.city:
            bill_to.city = request.city
        if request.state:
            bill_to.state = request.state
        if request.zip_code:
            bill_to.zip = request.zip_code
        
        # Create subscription
        subscription = apicontractsv1.ARBSubscriptionType()
        subscription.name = f"SEF {request.tier.title()} - {email}"
        subscription.paymentSchedule = payment_schedule
        subscription.amount = str(amount)
        subscription.payment = payment
        subscription.billTo = bill_to
        
        # Create request
        create_request = apicontractsv1.ARBCreateSubscriptionRequest()
        create_request.merchantAuthentication = merchant_auth
        create_request.subscription = subscription
        
        # Execute
        controller = createSubscriptionController(create_request)
        if AUTHORIZE_SANDBOX:
            controller.setenvironment('https://apitest.authorize.net/xml/v1/request.api')
        else:
            controller.setenvironment('https://api.authorize.net/xml/v1/request.api')
        
        controller.execute()
        response = controller.getresponse()
        
        if response.messages.resultCode == "Ok":
            subscription_id = str(response.subscriptionId)
            
            # Save to database
            save_subscription_to_db(uid, subscription_id, request.tier)
            
            return {
                "status": "success",
                "subscription_id": subscription_id,
                "tier": request.tier,
                "amount": amount,
                "message": f"Subscription activated! You now have {request.tier.title()} access."
            }
        else:
            error_msg = response.messages.message[0].text if response.messages.message else "Payment failed"
            raise HTTPException(status_code=400, detail=error_msg)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Payment error: {str(e)}")


@payment_router.post("/cancel")
async def cancel_subscription(request: CancelSubscriptionRequest, uid: str = Header(...)):
    """Cancel a subscription"""
    if not authorizenet_available:
        raise HTTPException(status_code=503, detail="Payment system not available")
    
    subscription_id = get_subscription_id(uid)
    if not subscription_id:
        raise HTTPException(status_code=404, detail="No active subscription found")
    
    try:
        merchant_auth = get_merchant_auth()
        
        cancel_request = apicontractsv1.ARBCancelSubscriptionRequest()
        cancel_request.merchantAuthentication = merchant_auth
        cancel_request.subscriptionId = subscription_id
        
        controller = cancelSubscriptionController(cancel_request)
        if AUTHORIZE_SANDBOX:
            controller.setenvironment('https://apitest.authorize.net/xml/v1/request.api')
        else:
            controller.setenvironment('https://api.authorize.net/xml/v1/request.api')
        
        controller.execute()
        response = controller.getresponse()
        
        if response.messages.resultCode == "Ok":
            # Update database
            UserManager.update_subscription(uid, 'free', None)
            
            return {
                "status": "cancelled",
                "message": "Subscription cancelled. You've been moved to the free tier."
            }
        else:
            error_msg = response.messages.message[0].text if response.messages.message else "Cancel failed"
            raise HTTPException(status_code=400, detail=error_msg)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cancel error: {str(e)}")


@payment_router.get("/subscription")
async def get_subscription_status(uid: str = Header(...)):
    """Get current subscription status"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        c.execute('SELECT subscription_tier, subscription_expires, authorize_subscription_id FROM users WHERE uid = ?', (uid,))
        row = c.fetchone()
    except:
        c.execute('SELECT subscription_tier, subscription_expires FROM users WHERE uid = ?', (uid,))
        row = c.fetchone()
        row = (row[0], row[1], None) if row else None
    
    conn.close()
    
    if not row:
        return {"tier": "free", "expires": None, "limits": SUBSCRIPTION_TIERS["free"]}
    
    tier = row[0] or "free"
    expires = row[1]
    subscription_id = row[2] if len(row) > 2 else None
    
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
        "has_subscription": subscription_id is not None,
        "limits": SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])
    }


@payment_router.post("/webhook")
async def authorize_webhook(request: Request):
    """
    Handle Authorize.net webhooks for subscription events
    Configure in Authorize.net Dashboard → Account → Webhooks
    """
    try:
        payload = await request.json()
        event_type = payload.get('eventType', '')
        
        if event_type == 'net.authorize.customer.subscription.expiring':
            # Subscription about to expire - could send email notification
            subscription_id = payload.get('payload', {}).get('id')
            print(f"⚠️ Subscription {subscription_id} expiring soon")
        
        elif event_type == 'net.authorize.customer.subscription.suspended':
            # Payment failed - downgrade to free
            subscription_id = payload.get('payload', {}).get('id')
            
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT uid FROM users WHERE authorize_subscription_id = ?', (subscription_id,))
            row = c.fetchone()
            
            if row:
                UserManager.update_subscription(row[0], 'free', None)
                print(f"⚠️ Subscription {subscription_id} suspended - user downgraded")
            
            conn.close()
        
        elif event_type == 'net.authorize.customer.subscription.cancelled':
            subscription_id = payload.get('payload', {}).get('id')
            
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT uid FROM users WHERE authorize_subscription_id = ?', (subscription_id,))
            row = c.fetchone()
            
            if row:
                UserManager.update_subscription(row[0], 'free', None)
                print(f"⚠️ Subscription {subscription_id} cancelled")
            
            conn.close()
        
        return {"status": "ok"}
    
    except Exception as e:
        print(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}


# =============================================================================
# ONE-TIME PAYMENT (Alternative to subscription)
# =============================================================================

@payment_router.post("/charge")
async def one_time_charge(request: CreateSubscriptionRequest, months: int = 1, uid: str = Header(...)):
    """
    Process a one-time charge for X months of access
    Useful if user prefers not to have recurring billing
    """
    if not authorizenet_available:
        raise HTTPException(status_code=503, detail="Payment system not available")
    
    if request.tier not in SUBSCRIPTION_AMOUNTS:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {request.tier}")
    
    try:
        amount = SUBSCRIPTION_AMOUNTS[request.tier] * months
        
        merchant_auth = get_merchant_auth()
        
        # Credit card
        credit_card = apicontractsv1.creditCardType()
        credit_card.cardNumber = request.card_number.replace(' ', '').replace('-', '')
        credit_card.expirationDate = request.expiration_date
        credit_card.cardCode = request.card_code
        
        payment = apicontractsv1.paymentType()
        payment.creditCard = credit_card
        
        # Transaction
        transaction = apicontractsv1.transactionRequestType()
        transaction.transactionType = "authCaptureTransaction"
        transaction.amount = str(amount)
        transaction.payment = payment
        
        # Create request
        charge_request = apicontractsv1.createTransactionRequest()
        charge_request.merchantAuthentication = merchant_auth
        charge_request.transactionRequest = transaction
        
        controller = createTransactionController(charge_request)
        if AUTHORIZE_SANDBOX:
            controller.setenvironment('https://apitest.authorize.net/xml/v1/request.api')
        else:
            controller.setenvironment('https://api.authorize.net/xml/v1/request.api')
        
        controller.execute()
        response = controller.getresponse()
        
        if response.messages.resultCode == "Ok":
            if hasattr(response.transactionResponse, 'transId'):
                # Set subscription for X months
                expires = (datetime.now() + timedelta(days=30 * months)).isoformat()
                
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('''
                    UPDATE users 
                    SET subscription_tier = ?, subscription_expires = ?
                    WHERE uid = ?
                ''', (request.tier, expires, uid))
                conn.commit()
                conn.close()
                
                return {
                    "status": "success",
                    "transaction_id": str(response.transactionResponse.transId),
                    "tier": request.tier,
                    "months": months,
                    "amount": amount,
                    "expires": expires,
                    "message": f"Payment successful! {months} month(s) of {request.tier.title()} access activated."
                }
        
        error_msg = "Payment declined"
        if hasattr(response, 'transactionResponse') and hasattr(response.transactionResponse, 'errors'):
            error_msg = response.transactionResponse.errors.error[0].errorText
        elif response.messages.message:
            error_msg = response.messages.message[0].text
        
        raise HTTPException(status_code=400, detail=error_msg)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Payment error: {str(e)}")
