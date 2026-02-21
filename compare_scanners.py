"""
Compare V1 vs V2 Alpha Scanner results side-by-side.
"""
import json, time, sys, os

# ── Run V1 ──
print("=" * 60)
print("  V1 ALPHA SCANNER")
print("=" * 60)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_scanner import run_alpha_scan as v1_scan

start = time.time()
v1 = v1_scan("mag7", max_results=5)
v1_time = time.time() - start

mc = v1["steps"].get("market_context", {})
us = v1["steps"].get("universe_scan", {})
print(f"Market: {mc.get('verdict', '?')}")
print(f"  Green: {mc.get('green_count',0)}/3 | Above SMA: {mc.get('above_sma_count',0)}/3")
print(f"Scanned: {us.get('scanned',0)} → {us.get('passed',0)} passed filter")

sf = v1["steps"].get("squeeze_filter", {})
if sf:
    print(f"Squeeze: {sf.get('firing',0)} firing, {sf.get('active',0)} active, {sf.get('forming',0)} forming")

of = v1["steps"].get("odds_filter", {})
if of:
    print(f"Odds filter: {of.get('passed',0)} passed (filtered {of.get('filtered_out',0)})")

print(f"\nResults: {len(v1['results'])} candidates")
for i, r in enumerate(v1.get("results", []), 1):
    print(f"\n--- #{i} {r['symbol']} ---")
    print(f"  Alpha Score: {r.get('alpha_score', '?')}")
    print(f"  Duration: {r.get('duration_label','?')} ({r.get('duration_tier','?')})")
    print(f"  Setup Type: {r.get('setup_type','?')}")
    print(f"  Scan Score: {r.get('scan_score','?')} | RSI: {r.get('rsi','?')} | RVOL: {r.get('rvol','?')}")
    sq = r.get("squeeze", {})
    print(f"  Squeeze: {sq.get('squeeze_status','NONE')} (score {sq.get('squeeze_score',0)})")
    odds = r.get("odds", {})
    print(f"  Call Hit 3D: {odds.get('call_hit_3d',0)}% | Win 1D: {odds.get('call_win_1d',0)}%")
    print(f"  Regime: {odds.get('regime','?')} | Z-Score: {odds.get('zscore',0)}")
    struct = r.get("structure", {})
    print(f"  Structure: {struct.get('pattern','?')} (score {struct.get('structure_score',0)})")
    wr = r.get("war_room", {})
    print(f"  Fade Conv: {wr.get('fade_conviction','?')} | Thin Top: {wr.get('thin_top_pct','?')}")

if not v1["results"]:
    verdict = v1.get("meta", {}).get("verdict", "?")
    print(f"  → {verdict}")

print(f"\nV1 Duration: {v1_time:.1f}s")

# Save V1 full output
with open("v1_scan_result.json", "w") as f:
    json.dump(v1, f, indent=2, default=str)


# ── Run V2 ──
print("\n" + "=" * 60)
print("  V2 ALPHA SCANNER (scanner_refactor)")
print("=" * 60)

# Import V2 from scanner_refactor
refactor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner_refactor")
sys.path.insert(0, refactor_dir)

# Need to import under a different name to avoid collision
import importlib
v2_mod = importlib.import_module("alpha_scanner")
# If the module was already loaded (V1), force reload from scanner_refactor path
if hasattr(v2_mod, '_fallback_scan'):
    # Already V2
    v2_scan_fn = v2_mod.run_alpha_scan
else:
    # Force reload from refactor path
    import importlib
    v2_mod = importlib.reload(v2_mod)
    v2_scan_fn = v2_mod.run_alpha_scan

start = time.time()
v2 = v2_scan_fn("mag7", max_results=5)
v2_time = time.time() - start

regime = v2["steps"].get("market_regime", {})
print(f"Regime: {regime.get('regime_label', '?')} (score {regime.get('regime_score', '?')})")
print(f"  Long ×{regime.get('long_multiplier', '?')} | Short ×{regime.get('short_multiplier', '?')}")
print(f"  Fallback mode: {v2['steps'].get('fallback_used', False)}")

bs = v2["steps"].get("broad_scan", {})
print(f"Broad scan: {bs.get('scanned',0)} scanned → {bs.get('returned',0)} returned")
if bs.get("top_raw_scores"):
    print("  Top raw scores:")
    for t in bs["top_raw_scores"][:5]:
        print(f"    {t['symbol']}: raw={t['raw_score']} (bull={t['bull']}, bear={t['bear']})")

de = v2["steps"].get("deep_enrich", {})
if de:
    print(f"Deep enrich: {de.get('enriched',0)} symbols | Squeezes: {de.get('squeeze_found',0)} | High odds: {de.get('high_odds',0)}")

sc = v2["steps"].get("scoring", {})
if sc:
    print(f"Setup types: {sc.get('setup_types',{})}")
    print(f"Longs: {sc.get('long_candidates',0)} | Shorts: {sc.get('short_candidates',0)}")

print(f"\nResults: {len(v2['results'])} candidates")
for i, r in enumerate(v2.get("results", []), 1):
    direction = r.get("direction", "?")
    emoji = "🟢" if direction == "LONG" else "🔴"
    print(f"\n--- #{i} {emoji} {r['symbol']} ({direction}) ---")
    print(f"  Alpha Score: {r.get('alpha_score', '?')}")
    print(f"  Setup Type: {r.get('setup_type','?')}")
    print(f"  Duration: {r.get('duration_label','?')} ({r.get('duration_tier','?')})")
    print(f"  Bull: {r.get('bull_score','?')} | Bear: {r.get('bear_score','?')} | Signal: {r.get('signal','?')}")
    print(f"  RSI: {r.get('rsi','?')} | RVOL: {r.get('rvol','?')}")

    bd = r.get("score_breakdown", {})
    if bd:
        print(f"  Score Breakdown:")
        for dim in ["v2_signal", "squeeze", "odds", "structure", "war_room", "extension"]:
            print(f"    {dim}: {bd.get(dim, '?')}")
        print(f"    regime_mult: {bd.get('regime_multiplier','?')} | raw_alpha: {bd.get('raw_alpha','?')}")

    sq = r.get("squeeze_detail", {})
    print(f"  Squeeze: {sq.get('tier','NONE')} (score {sq.get('score',0)}) release={sq.get('release_firing',False)}")
    odds = r.get("odds", {})
    print(f"  Call Hit 3D: {odds.get('call_hit_3d',0)}% | Win 1D: {odds.get('call_win_1d',0)}%")
    print(f"  Regime: {odds.get('regime','?')} | Z-Score: {odds.get('zscore',0)}")
    wr = r.get("war_room", {})
    print(f"  Fade Conv: {wr.get('fade_conviction','?')} | Thin Top: {wr.get('thin_top_pct','?')}")
    ext = r.get("extension", {})
    if ext:
        print(f"  Extension: score={ext.get('extension_score',0)} snap_back={ext.get('snap_back_probability',0)}%")

if not v2["results"]:
    verdict = v2.get("meta", {}).get("verdict", "?")
    print(f"  → {verdict}")

print(f"\nV2 Duration: {v2_time:.1f}s")

# Save V2 full output
with open("v2_scan_result.json", "w") as f:
    json.dump(v2, f, indent=2, default=str)


# ── Side-by-side comparison ──
print("\n" + "=" * 60)
print("  SIDE-BY-SIDE COMPARISON")
print("=" * 60)

v1_syms = {r["symbol"] for r in v1.get("results", [])}
v2_syms = {r["symbol"] for r in v2.get("results", [])}

print(f"\nV1 picks: {sorted(v1_syms) if v1_syms else 'NONE'}")
print(f"V2 picks: {sorted(v2_syms) if v2_syms else 'NONE'}")
print(f"Overlap:  {sorted(v1_syms & v2_syms) if (v1_syms & v2_syms) else 'NONE'}")
print(f"V1 only:  {sorted(v1_syms - v2_syms) if (v1_syms - v2_syms) else 'NONE'}")
print(f"V2 only:  {sorted(v2_syms - v1_syms) if (v2_syms - v1_syms) else 'NONE'}")

print(f"\nSpeed: V1={v1_time:.1f}s vs V2={v2_time:.1f}s")

# Compare shared picks
shared = v1_syms & v2_syms
if shared:
    print("\nShared picks comparison:")
    for sym in sorted(shared):
        r1 = next(r for r in v1["results"] if r["symbol"] == sym)
        r2 = next(r for r in v2["results"] if r["symbol"] == sym)
        print(f"  {sym}: V1 alpha={r1.get('alpha_score','?')} vs V2 alpha={r2.get('alpha_score','?')}")

print(f"\nFull results saved to v1_scan_result.json and v2_scan_result.json")
