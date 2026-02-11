"""
UI + SERVER Integration Patch: Fib Data → Trade Journal
=========================================================
COMPLETE wiring guide. The fib data currently DIES in two places:

PROBLEM 1 (SERVER): The MTF AI endpoint (/api/analyze/live/mtf/{symbol}/ai)
   calculates fibs (lines 2894-2948) and feeds them to GPT as text,
   but the RETURN JSON (lines 3304-3326) has ZERO fib fields.
   → The UI never receives fib data.

PROBLEM 2 (UI): logAITrade() reads window.lastAIPlan which comes from
   the MTF AI response above → has no fib data → writes none.

PROBLEM 3 (UI): quickLogTrade() and logTrade() have no fib fields at all.

FIX ORDER:
   1. Server: MTF AI endpoint must RETURN fib data in JSON  ← CRITICAL
   2. UI: logAITrade() must READ fib data from response
   3. UI: quickLogTrade() and logTrade() get empty defaults

APPLY THESE CHANGES TO:
   - unified_server.py (PATCH 0 — the critical server fix)
   - unified_ui_v2.html (PATCHES 1-4)
"""

# ==================================================================
# PATCH 0: SERVER — MTF AI ENDPOINT RETURN (THE CRITICAL FIX)
# ==================================================================
# File: unified_server.py
# Location: /api/analyze/live/mtf/{symbol}/ai endpoint
# The endpoint already calculates swing_high, swing_low, fib_236-786,
# fib_position, and confluences — but NONE of these are in the return dict.
#
# OPTION A: Quick fix — add calculated vars to the existing return
# OPTION B: Full fix — replace inline calc with fib_retracement.analyze_fibs()
#
# OPTION A (minimal change — apply to existing code):

PATCH_0A_LOCATION = "unified_server.py, lines 3304-3326"

PATCH_0A_OLD = """
        return {
            "symbol": symbol.upper(),
            "ai_commentary": response.choices[0].message.content.strip(),
            "high_prob": result.high_prob,
            "low_prob": result.low_prob,
            "confluence": result.confluence_pct,
            "dominant_signal": result.dominant_signal,
            "trade_timeframe": config["label"],
            "leading_direction": leading_direction,
            "leading_reason": leading_reason,
            "extension_override": extension_override,
            "extension_snap_prob": extension_snap_prob if extension_override else None,
            "bull_score": result.weighted_bull,
            "bear_score": result.weighted_bear,
            "rvol": rvol,
            "volume_trend": volume_trend,
            "vah": vah,
            "poc": poc,
            "val": val,
            "vwap": vwap,
            "rsi": rsi,
            "current_price": current_price
        }
"""

PATCH_0A_NEW = """
        # Determine fib zone from price position
        fib_zone = ""
        if fib_236 > 0:
            if current_price >= fib_236:
                fib_zone = "ABOVE_236"
            elif current_price >= fib_382:
                fib_zone = "PULLBACK_SHALLOW"
            elif current_price >= fib_500:
                fib_zone = "PULLBACK_ENTRY"
            elif current_price >= fib_618:
                fib_zone = "GOLDEN_ZONE"
            elif current_price >= fib_786:
                fib_zone = "DEEP_RETRACE"
            else:
                fib_zone = "BELOW_786"

        # Build confluences list for response
        fib_confluences = []
        if vah > 0 and fib_382 > 0:
            if abs(vah - fib_382) / vah < 0.015:
                fib_confluences.append(f"VAH ≈ Fib 38.2% at ${vah:.2f}")
            if abs(poc - fib_500) / max(poc, 1) < 0.015:
                fib_confluences.append(f"POC ≈ Fib 50% at ${poc:.2f}")
            if abs(val - fib_618) / max(val, 1) < 0.015:
                fib_confluences.append(f"VAL ≈ Fib 61.8% at ${val:.2f}")

        return {
            "symbol": symbol.upper(),
            "ai_commentary": response.choices[0].message.content.strip(),
            "high_prob": result.high_prob,
            "low_prob": result.low_prob,
            "confluence": result.confluence_pct,
            "confluence_pct": result.confluence_pct,
            "dominant_signal": result.dominant_signal,
            "trade_timeframe": config["label"],
            "leading_direction": leading_direction,
            "leading_reason": leading_reason,
            "extension_override": extension_override,
            "extension_snap_prob": extension_snap_prob if extension_override else None,
            "bull_score": result.weighted_bull,
            "bear_score": result.weighted_bear,
            "rvol": rvol,
            "volume_trend": volume_trend,
            "vah": vah,
            "poc": poc,
            "val": val,
            "vwap": vwap,
            "rsi": rsi,
            "current_price": current_price,
            # ═══════════════════════════════════════════
            # V2: FIBONACCI DATA (was calculated but never returned!)
            # ═══════════════════════════════════════════
            "fib_levels": {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "trend": "UPTREND",  # MTF AI uses simple high-to-low
                "fib_236": round(fib_236, 2),
                "fib_382": round(fib_382, 2),
                "fib_500": round(fib_500, 2),
                "fib_618": round(fib_618, 2),
                "fib_786": round(fib_786, 2),
            },
            "fib_position": fib_position if 'fib_position' in dir() else "",
            "fib_zone": fib_zone,
            "fib_quality": "MODERATE",  # Quick fix — proper scoring needs fib_retracement.py
            "fib_confluence": fib_confluences,
        }
"""

# OPTION B (full fix — replace inline fib calc with module):
PATCH_0B_NOTE = """
FULL FIX: Replace lines 2894-2948 (inline fib calc) with:

    from fib_retracement import analyze_fibs

    # Replace the try/except block that calculates fibs
    fib_result = None
    fib_text = ""
    try:
        df_15d = scanner._get_candles(symbol.upper(), "D", 15)
        if df_15d is not None and len(df_15d) >= 5:
            fib_result = analyze_fibs(
                df_15d, symbol.upper(),
                vah=vah, poc=poc, val=val,
                current_price=current_price
            )
            if fib_result:
                fib_text = f'''
═══════════════════════════════════════════
🔍 FIBONACCI RETRACEMENT ({fib_result.swing_quality} quality)
═══════════════════════════════════════════
Swing High: ${fib_result.swing_high.price:.2f} ({fib_result.swing_high.date[:10]})
Swing Low: ${fib_result.swing_low.price:.2f} ({fib_result.swing_low.date[:10]})
Trend: {fib_result.trend}

Fib 23.6%: ${fib_result.levels[0].price:.2f} | Fib 38.2%: ${fib_result.levels[1].price:.2f}
Fib 50%: ${fib_result.levels[2].price:.2f} | Fib 61.8%: ${fib_result.levels[3].price:.2f}
Fib 78.6%: ${fib_result.levels[4].price:.2f}

🔍 PRICE POSITION: {fib_result.price_position}
{"🎯 VP+FIB CONFLUENCE: " + "; ".join(fib_result.confluences) if fib_result.confluences else ""}
'''
    except Exception as e:
        print(f"Fib calculation error: {e}")

Then in the return dict, replace the fib fields with:

    "fib_levels": fib_result.to_dict() if fib_result else {},
    "fib_position": fib_result.price_position if fib_result else "",
    "fib_zone": fib_result.price_zone if fib_result else "",
    "fib_quality": fib_result.swing_quality if fib_result else "",
    "fib_confluence": fib_result.confluences if fib_result else [],
"""


# ==================================================================
# PATCH 0C: SERVER — ALSO FIX /api/analyze/live/{symbol} ENDPOINT
# ==================================================================
# The quickAnalyze() button calls this endpoint. It DOES have inline
# fib calc (lines 2514-2700) and includes fib_levels in the response.
# But it's missing fib_zone and fib_quality.
# Apply the same fib_retracement.py integration here.

PATCH_0C_NOTE = """
File: unified_server.py, /api/analyze/live/{symbol} endpoint (line ~2326)

This endpoint already returns fib_levels and fib_position in its response.
It does NOT return fib_zone or fib_quality.

After the fib_position calculation (around line 2641), add:

    # Derive zone code from fib_position text
    if "GOLDEN ZONE" in response.get("fib_position", ""):
        response["fib_zone"] = "GOLDEN_ZONE"
    elif "pullback entry" in response.get("fib_position", "").lower():
        response["fib_zone"] = "PULLBACK_ENTRY"
    elif "shallow pullback" in response.get("fib_position", "").lower():
        response["fib_zone"] = "PULLBACK_SHALLOW"
    elif "deep" in response.get("fib_position", "").lower():
        response["fib_zone"] = "DEEP_RETRACE"
    elif "trend may be broken" in response.get("fib_position", "").lower():
        response["fib_zone"] = "TREND_BROKEN"
    elif "Above swing" in response.get("fib_position", ""):
        response["fib_zone"] = "ABOVE_SWING"
    elif "Below swing" in response.get("fib_position", ""):
        response["fib_zone"] = "BELOW_SWING"
    else:
        response["fib_zone"] = ""

    response["fib_quality"] = "MODERATE"  # Quick fix

Or better — replace the entire inline fib block with:

    from fib_retracement import analyze_fibs
    fib_result = analyze_fibs(df_fib, symbol, vah=vah, poc=poc, val=val)
    if fib_result:
        response["fib_levels"] = fib_result.to_dict()
        response["fib_position"] = fib_result.price_position
        response["fib_zone"] = fib_result.price_zone
        response["fib_quality"] = fib_result.swing_quality
        response["fib_confluence"] = fib_result.confluences
"""


# ==================================================================
# PATCH 1: UI — logAITrade() (line ~3564)
# ==================================================================
# This is the "Log Trade" button after clicking "Get AI Trade Plan".
# It reads window.lastAIPlan which NOW has fib data (after PATCH 0).

PATCH_1_LOCATION = "unified_ui_v2.html, line ~3574 (inside logAITrade)"

PATCH_1_OLD = """
                const tradeData = {
                    symbol: plan.symbol.toUpperCase(),
                    direction: parsed.bias,
                    timeframe: plan.trade_timeframe || 'SWING',
                    entry: parsed.entry,
                    stop: parsed.stop,
                    target: parsed.target1,
                    target2: parsed.target2,
                    signal: plan.dominant_signal || '',
                    confidence: plan.confluence_pct || plan.confluence || 0,
                    ai_commentary: plan.ai_commentary || '',
                    vah: plan.vah || 0,
                    poc: plan.poc || 0,
                    val: plan.val || 0,
                    rsi: plan.rsi || 0,
                    notes: `AI Trade Plan - ${plan.trade_timeframe || 'SWING'}`,
                    status: 'pending',
                    pnl: 0,
                    created_at: new Date().toISOString()
                };
"""

PATCH_1_NEW = """
                // V2: Extract fib data from API response (now returned by server)
                const fibLevels = plan.fib_levels || {};
                const fibConfluence = plan.fib_confluence || [];

                const tradeData = {
                    symbol: plan.symbol.toUpperCase(),
                    direction: parsed.bias,
                    timeframe: plan.trade_timeframe || 'SWING',
                    entry: parsed.entry,
                    stop: parsed.stop,
                    target: parsed.target1,
                    target2: parsed.target2,
                    signal: plan.dominant_signal || '',
                    confidence: plan.confluence_pct || plan.confluence || 0,
                    ai_commentary: plan.ai_commentary || '',
                    vah: plan.vah || 0,
                    poc: plan.poc || 0,
                    val: plan.val || 0,
                    rsi: plan.rsi || 0,
                    // V2: Fibonacci context (from server PATCH 0)
                    fib_swing_high: fibLevels.swing_high || 0,
                    fib_swing_low: fibLevels.swing_low || 0,
                    fib_trend: fibLevels.trend || '',
                    fib_236: fibLevels.fib_236 || 0,
                    fib_382: fibLevels.fib_382 || 0,
                    fib_500: fibLevels.fib_500 || 0,
                    fib_618: fibLevels.fib_618 || 0,
                    fib_786: fibLevels.fib_786 || 0,
                    fib_position: plan.fib_position || '',
                    fib_zone: plan.fib_zone || _deriveFibZone(parsed.entry, fibLevels),
                    fib_confluence: Array.isArray(fibConfluence) ? fibConfluence.join('; ') : (fibConfluence || ''),
                    fib_quality: plan.fib_quality || '',
                    notes: `AI Trade Plan - ${plan.trade_timeframe || 'SWING'}`,
                    status: 'pending',
                    pnl: 0,
                    created_at: new Date().toISOString()
                };
"""


# ==================================================================
# PATCH 2: UI — HELPER FUNCTION (add near top of <script> block)
# ==================================================================
# Fallback zone derivation when the server doesn't return fib_zone.

HELPER_FUNCTION = """
        // V2: Derive fib zone from entry price and fib levels (fallback)
        function _deriveFibZone(entryPrice, fibLevels) {
            if (!entryPrice || !fibLevels || !fibLevels.fib_236) return '';

            const sh = fibLevels.swing_high || 0;
            const sl = fibLevels.swing_low || 0;
            const f236 = fibLevels.fib_236 || 0;
            const f382 = fibLevels.fib_382 || 0;
            const f500 = fibLevels.fib_500 || 0;
            const f618 = fibLevels.fib_618 || 0;
            const f786 = fibLevels.fib_786 || 0;

            // Fibs are measured down from high (standard retracement)
            if (entryPrice >= sh) return 'ABOVE_SWING';
            if (entryPrice >= f236) return 'ABOVE_236';
            if (entryPrice >= f382) return 'PULLBACK_SHALLOW';
            if (entryPrice >= f500) return 'PULLBACK_ENTRY';
            if (entryPrice >= f618) return 'GOLDEN_ZONE';
            if (entryPrice >= f786) return 'DEEP_RETRACE';
            if (entryPrice >= sl) return 'TREND_BROKEN';
            return 'BELOW_SWING';
        }
"""


# ==================================================================
# PATCH 3: UI — quickLogTrade() (line ~5159)
# ==================================================================
# Batch scan quick log — no fib data available. Add empty defaults.

PATCH_3_LOCATION = "unified_ui_v2.html, line ~5166 (inside quickLogTrade)"

PATCH_3_OLD = """
            const data = {
                symbol: symbol.toUpperCase(),
                timeframe: '1HR',
                direction: signal?.includes('LONG') ? 'LONG' : 'SHORT',
                entry: entry || 0,
                stop: stop || 0,
                target: 0,
                notes: 'Logged from AI analysis',
                status: 'pending',
                pnl: 0,
                created_at: new Date().toISOString()
            };
"""

PATCH_3_NEW = """
            const data = {
                symbol: symbol.toUpperCase(),
                timeframe: '1HR',
                direction: signal?.includes('LONG') ? 'LONG' : 'SHORT',
                entry: entry || 0,
                stop: stop || 0,
                target: 0,
                // V2: Empty fib defaults (batch scan has no individual fib data)
                fib_swing_high: 0, fib_swing_low: 0, fib_trend: '',
                fib_236: 0, fib_382: 0, fib_500: 0, fib_618: 0, fib_786: 0,
                fib_position: '', fib_zone: '', fib_confluence: '', fib_quality: '',
                notes: 'Logged from batch scan (no fib context)',
                status: 'pending',
                pnl: 0,
                created_at: new Date().toISOString()
            };
"""


# ==================================================================
# PATCH 4: UI — logTrade() manual form (line ~5608)
# ==================================================================
# Manual entry form — add empty defaults for schema consistency.

PATCH_4_LOCATION = "unified_ui_v2.html, line ~5608 (inside logTrade)"

PATCH_4_OLD = """
        async function logTrade() {
            const data = {
                symbol: document.getElementById('trade-symbol').value.toUpperCase(),
                timeframe: '1HR',
                direction: document.getElementById('trade-direction').value,
                entry: parseFloat(document.getElementById('trade-entry').value),
                stop: parseFloat(document.getElementById('trade-stop').value),
                target: parseFloat(document.getElementById('trade-target').value),
                notes: document.getElementById('trade-notes').value,
                status: 'pending',
                pnl: 0,
                created_at: new Date().toISOString()
            };
"""

PATCH_4_NEW = """
        async function logTrade() {
            const data = {
                symbol: document.getElementById('trade-symbol').value.toUpperCase(),
                timeframe: '1HR',
                direction: document.getElementById('trade-direction').value,
                entry: parseFloat(document.getElementById('trade-entry').value),
                stop: parseFloat(document.getElementById('trade-stop').value),
                target: parseFloat(document.getElementById('trade-target').value),
                notes: document.getElementById('trade-notes').value,
                // V2: Empty fib defaults for manual entries
                fib_swing_high: 0, fib_swing_low: 0, fib_trend: '',
                fib_236: 0, fib_382: 0, fib_500: 0, fib_618: 0, fib_786: 0,
                fib_position: '', fib_zone: 'MANUAL_ENTRY', fib_confluence: '', fib_quality: '',
                status: 'pending',
                pnl: 0,
                created_at: new Date().toISOString()
            };
"""


# ==================================================================
# COMPLETE DATA FLOW (after all patches applied)
# ==================================================================

DATA_FLOW = """
╔══════════════════════════════════════════════════════════════════════╗
║                    COMPLETE FIB DATA FLOW                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  USER CLICKS "ANALYZE"                                             ║
║       │                                                            ║
║       ▼                                                            ║
║  quickAnalyze() → /api/analyze/live/{symbol}                       ║
║       │            Server calculates fibs                          ║
║       │            Returns: fib_levels, fib_position, fib_zone ✅  ║
║       ▼                                                            ║
║  User sees results card with VP levels, scores, etc.               ║
║  User clicks "🤖 Get AI Trade Plan"                                ║
║       │                                                            ║
║       ▼                                                            ║
║  getAICommentary() → /api/analyze/live/mtf/{symbol}/ai             ║
║       │               Server calculates fibs (lines 2894-2948)     ║
║       │               Feeds fibs to GPT in prompt text             ║
║       │               PATCH 0: NOW returns fib data in JSON ✅     ║
║       ▼                                                            ║
║  window.lastAIPlan = response (now HAS fib_levels, fib_zone...)    ║
║  User sees AI trade plan with Log Trade button                     ║
║  User clicks "📋 Log Trade"                                        ║
║       │                                                            ║
║       ▼                                                            ║
║  logAITrade() reads window.lastAIPlan                              ║
║       │  PATCH 1: NOW extracts fib data from plan ✅               ║
║       │  Includes: fib_swing_high/low, fib_236-786, fib_zone,     ║
║       │            fib_position, fib_confluence, fib_quality       ║
║       ▼                                                            ║
║  Firestore or /api/journal/log                                     ║
║       │  trade_journal.py stores all 12 fib + 4 context columns    ║
║       ▼                                                            ║
║  Trade closed → _calculate_fib_stats() answers:                    ║
║       • Do Golden Zone entries outperform? (win rate by zone)      ║
║       • Does VP+Fib confluence give an edge? (with vs without)     ║
║       • Does swing quality matter? (STRONG vs WEAK)                ║
║       ▼                                                            ║
║  /api/journal/fib-report → Human-readable performance report       ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                    CHANGE SUMMARY                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  PATCH 0 (SERVER - CRITICAL)                                       ║
║  unified_server.py: MTF AI endpoint return JSON                    ║
║  → Add fib_levels, fib_position, fib_zone, fib_quality,           ║
║    fib_confluence to the response dict (lines 3304-3326)           ║
║  → Without this, ALL UI patches are useless                        ║
║                                                                    ║
║  PATCH 0C (SERVER)                                                 ║
║  unified_server.py: /api/analyze/live/{symbol}                     ║
║  → Add fib_zone and fib_quality to response                       ║
║                                                                    ║
║  PATCH 1 (UI - MAIN PATH)                                         ║
║  unified_ui_v2.html: logAITrade() tradeData object                 ║
║  → Read fib fields from plan and include in trade log              ║
║                                                                    ║
║  PATCH 2 (UI - HELPER)                                             ║
║  unified_ui_v2.html: Add _deriveFibZone() function                 ║
║  → Fallback zone classification from price + fib levels            ║
║                                                                    ║
║  PATCH 3 (UI - BATCH)                                              ║
║  unified_ui_v2.html: quickLogTrade() data object                   ║
║  → Add empty fib defaults for schema consistency                   ║
║                                                                    ║
║  PATCH 4 (UI - MANUAL)                                             ║
║  unified_ui_v2.html: logTrade() data object                        ║
║  → Add empty fib defaults with zone='MANUAL_ENTRY'                 ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(DATA_FLOW)
