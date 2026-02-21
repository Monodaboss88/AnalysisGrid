// Shared configuration for all frontend pages
// Include this script before any page-specific JS

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? ''
    : 'https://analysisgrid-production.up.railway.app';

// Alias — some pages use BACKEND instead of API_BASE
const BACKEND = API_BASE;
