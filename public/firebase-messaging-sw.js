// Firebase Cloud Messaging Service Worker
// ==========================================
// Handles background push notifications + PWA caching
// Must be served from root: /firebase-messaging-sw.js

importScripts('https://www.gstatic.com/firebasejs/9.22.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/9.22.0/firebase-messaging-compat.js');

// Firebase config (same as app)
firebase.initializeApp({
    apiKey: "AIzaSyB4oBeoFzFV2oS5ZxEvQ4TIRRVJFC3nl-w",
    authDomain: "analysis-grid.firebaseapp.com",
    projectId: "analysis-grid",
    storageBucket: "analysis-grid.firebasestorage.app",
    messagingSenderId: "633207767657",
    appId: "1:633207767657:web:e623871eca352a268b58f8"
});

const messaging = firebase.messaging();

// Handle background messages (when app/tab is not focused)
messaging.onBackgroundMessage((payload) => {
    console.log('[SW] Background message:', payload);

    const data = payload.data || {};
    const notification = payload.notification || {};

    const title = notification.title || 'Analysis Grid';
    const options = {
        body: notification.body || '',
        icon: '/icons/icon-192.svg',
        badge: '/icons/badge-72.svg',
        tag: data.category || 'general',
        renotify: true,
        vibrate: [200, 100, 200],
        data: {
            url: data.url || '/desk.html',
            type: data.type || 'general',
            symbol: data.symbol || '',
            trade_id: data.trade_id || ''
        },
        actions: getActions(data.type)
    };

    self.registration.showNotification(title, options);
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    const data = event.notification.data || {};
    let url = '/desk.html';

    // Route based on notification type
    if (data.type === 'trade_close') {
        url = '/desk.html#trades';
    } else if (data.type === 'scanner_signal') {
        url = `/desk.html#scan&symbol=${data.symbol}`;
    } else if (data.type === 'options_flow') {
        url = '/options.html#flow';
    }

    // Handle action buttons
    if (event.action === 'open_desk') {
        url = '/desk.html';
    } else if (event.action === 'view_trade') {
        url = '/desk.html#trades';
    } else if (event.action === 'view_chart') {
        url = `/desk.html#scan&symbol=${data.symbol}`;
    }

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then((windowClients) => {
            // Focus existing tab if open
            for (const client of windowClients) {
                if (client.url.includes('analysis-grid.web.app') && 'focus' in client) {
                    client.navigate(url);
                    return client.focus();
                }
            }
            // Otherwise open new tab
            return clients.openWindow(url);
        })
    );
});

function getActions(type) {
    switch (type) {
        case 'trade_close':
            return [
                { action: 'view_trade', title: '📊 View Trade' },
                { action: 'open_desk', title: '🖥️ Open Desk' }
            ];
        case 'scanner_signal':
            return [
                { action: 'view_chart', title: '📈 View Chart' },
                { action: 'open_desk', title: '🖥️ Open Desk' }
            ];
        default:
            return [
                { action: 'open_desk', title: '🖥️ Open Desk' }
            ];
    }
}


// =============================================================================
// PWA CACHING — App Shell Strategy
// =============================================================================

const CACHE_NAME = 'ag-v1';
const APP_SHELL = [
    '/desk.html',
    '/config.js',
    '/notifications.js',
    '/manifest.json',
    '/icons/icon-192.svg'
];

// Install — pre-cache app shell
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(APP_SHELL).catch(() => {
                console.log('[SW] Some app shell files not cached (ok on first deploy)');
            });
        })
    );
    self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
            );
        })
    );
    self.clients.claim();
});

// Fetch — network-first with cache fallback for HTML pages
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Skip non-GET and API/external requests
    if (event.request.method !== 'GET') return;
    if (url.pathname.startsWith('/api/')) return;
    if (!url.origin.includes('analysis-grid.web.app') && !url.hostname === 'localhost') return;

    event.respondWith(
        fetch(event.request)
            .then((response) => {
                // Cache successful responses
                if (response.ok && url.pathname.match(/\.(html|js|css|png|svg)$/)) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
                }
                return response;
            })
            .catch(() => {
                // Offline fallback
                return caches.match(event.request);
            })
    );
});
