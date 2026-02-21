/**
 * Analysis Grid — Push Notification Module
 * ==========================================
 * Include after config.js and Firebase SDK on any page.
 * Handles: permission request, FCM token registration, foreground messages.
 *
 * Usage:
 *   await AGNotifications.init();     // request permission + register
 *   AGNotifications.isEnabled()       // check status
 *   AGNotifications.disable()         // unregister token
 */

const AGNotifications = (() => {
    // Firebase Messaging VAPID key — get this from Firebase Console > Cloud Messaging > Web Push certificates
    // This is the public key; the private key stays in Firebase.
    // If not set yet, the script will still work but won't be able to get a token.
    let _vapidKey = '';  // Will be set from Firebase Console

    let _messaging = null;
    let _currentToken = null;
    let _userId = 'anonymous';
    let _initialized = false;

    /**
     * Initialize notifications — call once on page load
     */
    async function init(userId) {
        if (_initialized) return _currentToken;
        if (userId) _userId = userId;

        // Check browser support
        if (!('Notification' in window)) {
            console.warn('[Notify] Browser does not support notifications');
            return null;
        }

        if (!('serviceWorker' in navigator)) {
            console.warn('[Notify] Service workers not supported');
            return null;
        }

        try {
            // Register service worker
            const reg = await navigator.serviceWorker.register('/firebase-messaging-sw.js');
            console.log('[Notify] Service worker registered');

            // Initialize Firebase Messaging
            if (typeof firebase !== 'undefined' && firebase.messaging) {
                _messaging = firebase.messaging();

                // Handle foreground messages
                _messaging.onMessage((payload) => {
                    console.log('[Notify] Foreground message:', payload);
                    _showForegroundToast(payload);
                });

                _initialized = true;
                console.log('[Notify] Module initialized');

                // Auto-restore token if permission was previously granted
                if (Notification.permission === 'granted') {
                    await _getAndRegisterToken();
                }

                return _currentToken;
            } else {
                console.warn('[Notify] Firebase Messaging SDK not loaded');
            }
        } catch (err) {
            console.error('[Notify] Init error:', err);
        }
        return null;
    }

    /**
     * Request permission and register token
     */
    async function enable() {
        if (!_initialized) await init(_userId);

        const permission = await Notification.requestPermission();
        if (permission !== 'granted') {
            console.warn('[Notify] Permission denied');
            return false;
        }

        return await _getAndRegisterToken();
    }

    /**
     * Disable notifications — unregister token
     */
    async function disable() {
        if (_currentToken) {
            try {
                // Unregister from backend
                await fetch(`${API_BASE}/api/notifications/unregister`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: _userId, token: _currentToken })
                });

                // Delete token from Firebase
                if (_messaging) {
                    await _messaging.deleteToken();
                }

                _currentToken = null;
                localStorage.removeItem('ag_fcm_token');
                console.log('[Notify] Disabled');
                return true;
            } catch (err) {
                console.error('[Notify] Disable error:', err);
            }
        }
        return false;
    }

    /**
     * Check if notifications are enabled
     */
    function isEnabled() {
        return Notification.permission === 'granted' && _currentToken !== null;
    }

    /**
     * Get notification status details
     */
    function getStatus() {
        return {
            supported: 'Notification' in window,
            permission: Notification.permission,
            token: _currentToken ? '...' + _currentToken.slice(-8) : null,
            enabled: isEnabled(),
            userId: _userId
        };
    }

    /**
     * Update user ID (call after Firebase Auth sign-in)
     */
    function setUserId(uid) {
        const oldId = _userId;
        _userId = uid || 'anonymous';

        // Re-register token with new user ID
        if (_currentToken && _userId !== oldId) {
            _registerTokenWithBackend(_currentToken);
        }
    }

    /**
     * Send a test notification
     */
    async function sendTest() {
        try {
            const res = await fetch(`${API_BASE}/api/notifications/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: _userId })
            });
            return await res.json();
        } catch (err) {
            console.error('[Notify] Test send error:', err);
            return { error: err.message };
        }
    }

    // ========================================================================
    // INTERNAL HELPERS
    // ========================================================================

    async function _getAndRegisterToken() {
        if (!_messaging) return false;

        try {
            const reg = await navigator.serviceWorker.getRegistration();
            const tokenConfig = { serviceWorkerRegistration: reg };

            // Add VAPID key if configured
            if (_vapidKey) {
                tokenConfig.vapidKey = _vapidKey;
            }

            const token = await _messaging.getToken(tokenConfig);
            if (token) {
                _currentToken = token;
                localStorage.setItem('ag_fcm_token', token);
                await _registerTokenWithBackend(token);
                console.log('[Notify] Token obtained:', token.slice(-8));
                return true;
            }
        } catch (err) {
            console.error('[Notify] Token error:', err);
        }
        return false;
    }

    async function _registerTokenWithBackend(token) {
        try {
            await fetch(`${API_BASE}/api/notifications/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    token: token,
                    user_id: _userId,
                    device_type: _detectDeviceType()
                })
            });
        } catch (err) {
            console.error('[Notify] Backend register error:', err);
        }
    }

    function _detectDeviceType() {
        const ua = navigator.userAgent.toLowerCase();
        if (/android/.test(ua)) return 'android';
        if (/iphone|ipad|ipod/.test(ua)) return 'ios';
        return 'web';
    }

    /**
     * Show an in-app toast for foreground messages
     * (Background messages are handled by the service worker)
     */
    function _showForegroundToast(payload) {
        const notification = payload.notification || {};
        const data = payload.data || {};
        const title = notification.title || 'Analysis Grid';
        const body = notification.body || '';

        // Create toast element
        const toast = document.createElement('div');
        toast.className = 'ag-notification-toast';
        toast.innerHTML = `
            <div class="ag-toast-header">
                <span class="ag-toast-icon">${_getIcon(data.type)}</span>
                <strong>${title}</strong>
                <button onclick="this.parentElement.parentElement.remove()" class="ag-toast-close">&times;</button>
            </div>
            <div class="ag-toast-body">${body}</div>
        `;

        // Style
        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            zIndex: '99999',
            background: 'linear-gradient(135deg, #1a1a2e, #16213e)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            borderRadius: '12px',
            padding: '14px 18px',
            minWidth: '280px',
            maxWidth: '380px',
            color: '#e2e8f0',
            fontFamily: 'system-ui, sans-serif',
            fontSize: '14px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
            animation: 'slideInRight 0.3s ease-out',
            cursor: 'pointer'
        });

        toast.addEventListener('click', (e) => {
            if (e.target.tagName !== 'BUTTON') {
                toast.remove();
                // Navigate based on type
                if (data.type === 'trade_close') {
                    window.location.hash = '#trades';
                }
            }
        });

        document.body.appendChild(toast);

        // Auto-remove after 8 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100%)';
                toast.style.transition = 'all 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }
        }, 8000);
    }

    function _getIcon(type) {
        switch (type) {
            case 'trade_close': return '📊';
            case 'scanner_signal': return '🎯';
            case 'options_flow': return '📡';
            default: return '🔔';
        }
    }

    // Inject toast animation CSS
    if (typeof document !== 'undefined') {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            .ag-toast-header {
                display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
            }
            .ag-toast-header strong { flex: 1; }
            .ag-toast-close {
                background: none; border: none; color: #94a3b8;
                font-size: 20px; cursor: pointer; padding: 0; line-height: 1;
            }
            .ag-toast-close:hover { color: #e2e8f0; }
            .ag-toast-body { color: #94a3b8; font-size: 13px; }
            .ag-toast-icon { font-size: 18px; }
        `;
        document.head.appendChild(style);
    }

    // Public API
    return { init, enable, disable, isEnabled, getStatus, setUserId, sendTest };
})();
