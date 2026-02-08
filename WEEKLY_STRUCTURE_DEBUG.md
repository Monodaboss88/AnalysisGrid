# Weekly Structure Feature - Debug Notes

## What We're Trying to Accomplish

Add a **Weekly Macro Structure** section to the Range Structure UI that shows:
- Weekly HH/HL/LH/LL counts (from actual weekly candles)
- Weekly trend classification (STRONG_UPTREND, UPTREND, NEUTRAL, DOWNTREND, STRONG_DOWNTREND)
- Weekly close position (0-100% where the week closed in its range)
- Weekly close signal (BULLISH_REVERSAL, BEARISH_REVERSAL, STRONG_CLOSE, WEAK_CLOSE, etc.)

This gives macro context beyond just daily period comparisons.

---

## Current Status

### ✅ Backend - WORKING
The API endpoint `/api/range/analyze/{symbol}` now returns `weekly_structure`:

```bash
curl -s "https://analysisgrid-production.up.railway.app/api/range/analyze/META"
```

Returns:
```json
{
  "weekly_structure": {
    "trend": "NEUTRAL",
    "hh_count": 3,
    "hl_count": 2, 
    "lh_count": 4,
    "ll_count": 5,
    "weekly_close_position": 0.2,
    "weekly_close_signal": "STRONG_BEAR_CLOSE",
    "last_week_structure": "LH+LL"
  }
}
```

### ❌ Frontend - NOT DISPLAYING

The UI code to display weekly structure was added to `public/index.html` around line 3907.
It should appear between the daily periods table and the Key Levels section.

**But the user reports it's not showing up in the browser.**

---

## Files Modified

1. **finnhub_scanner.py** - Added `calculate_range_structure()` method in TechnicalCalculator class (~line 356)
2. **chart_input_analyzer.py** - Added `RangeContext` dataclass (~line 102)
3. **rangewatcher/range_watcher_endpoints.py** - Added `fetch_weekly_structure()` function and integrated into `/api/range/analyze/{symbol}` endpoint
4. **public/index.html** - Added weekly structure display UI (~line 3907)

---

## The Problem

The frontend code exists in `public/index.html` at line 3907:
```javascript
${data.weekly_structure ? `
<div style="margin-top: 1.5rem; padding: 1.25rem; background: linear-gradient(135deg, rgba(138,43,226,0.1), rgba(100,100,255,0.08)); ...">
    <div>📊 Weekly Macro Structure</div>
    ...
</div>
` : ''}
```

**Possible issues:**
1. Browser cache not cleared - need hard refresh (Ctrl+Shift+R)
2. Firebase hosting not updated - need `firebase deploy --only hosting`
3. JavaScript error preventing render - check browser console (F12)
4. The conditional `${data.weekly_structure ? ...}` might be failing
5. The template literal syntax might have an issue

---

## Debug Steps

1. **Verify Firebase deployed latest:**
   ```bash
   firebase deploy --only hosting
   ```

2. **Hard refresh browser:** Ctrl+Shift+R

3. **Check browser console for errors:** F12 → Console tab

4. **Verify API returns weekly_structure:**
   ```bash
   curl "https://analysisgrid-production.up.railway.app/api/range/analyze/META" | jq .weekly_structure
   ```

5. **Check if the HTML is in the deployed version:**
   - View page source in browser
   - Search for "Weekly Macro Structure"

---

## Expected UI Location

```
┌─────────────────────────────────────┐
│ RANGE STRUCTURE                     │
│ Current Price: $661.46              │
├─────────────────────────────────────┤
│ Period Table (3D, 6D, 9D, etc.)     │
├─────────────────────────────────────┤
│ 📊 WEEKLY MACRO STRUCTURE  ← HERE   │
│ [HH: 3] [HL: 2] [LH: 4] [LL: 5]    │
│ Weekly Close: 20% STRONG_BEAR_CLOSE │
├─────────────────────────────────────┤
│ KEY LEVELS                          │
│ Resistance / Support                │
└─────────────────────────────────────┘
```

---

## Contact

The relevant code sections:
- Backend: `rangewatcher/range_watcher_endpoints.py` lines 41-115 (`fetch_weekly_structure`)
- Frontend: `public/index.html` lines 3907-4000 (weekly structure UI)
