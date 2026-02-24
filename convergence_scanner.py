"""
Convergence Edge Scanner
========================
Meta-scanner that pulls directional signals from all 8 Alpha scanners,
normalizes them into a unified score, and surfaces:
  • Convergence Alerts  – most scanners agree → high-conviction entry
  • Divergence Alerts   – scanners conflict  → fade / stay-out warning

Drop this file alongside alpha_scanner.py.  It imports nothing from that
module directly — it consumes the *return dicts* each scanner already
produces, so you call it from your endpoint layer the same way you call
the others.

Author : Claude (Convergence Edge concept by Rob @ SEF Intel)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Weights ────────────────────────────────────────────────────────
# Two preset profiles.  The endpoint lets the caller pick one or send
# custom weights.

WEIGHT_PROFILES = {
    #                       simple  mtf_raw  mtf_ai  signal_quick  options_flow  war_room  buffett  sustainability
    "daytrade": {
        "simple":        0.15,
        "mtf_raw":       0.20,
        "mtf_ai":        0.20,
        "signal_quick":  0.15,
        "options_flow":  0.15,
        "war_room":      0.10,
        "buffett":       0.025,
        "sustainability":0.025,
    },
    "swing": {
        "simple":        0.10,
        "mtf_raw":       0.15,
        "mtf_ai":        0.15,
        "signal_quick":  0.10,
        "options_flow":  0.15,
        "war_room":      0.10,
        "buffett":       0.15,
        "sustainability":0.10,
    },
    "equal": {k: 0.125 for k in [
        "simple", "mtf_raw", "mtf_ai", "signal_quick",
        "options_flow", "war_room", "buffett", "sustainability"]},
}


# ── Normalizer helpers ─────────────────────────────────────────────
# Each function takes the raw dict a scanner already returns and
# extracts a directional score in the range  -1.0 (max bear) … +1.0 (max bull).
# A secondary "confidence" float 0‑1 tells us how much data the
# scanner actually had (so a scanner that returned nulls gets down-weighted).

@dataclass
class ScannerVote:
    """One scanner's normalized opinion."""
    scanner: str
    direction: float        # -1.0 … +1.0
    confidence: float       # 0.0 … 1.0
    label: str              # human-readable e.g. "Strong Bull"
    raw_summary: str        # one-liner from the raw data

    @property
    def weighted(self) -> float:
        return self.direction * self.confidence


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _safe(d: dict, key: str, default=None):
    """Nested-safe dict get  ('a.b.c' → d['a']['b']['c'])."""
    keys = key.split(".")
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            return default
    return cur


def _direction_label(score: float) -> str:
    if score >=  0.6: return "Strong Bull"
    if score >=  0.2: return "Lean Bull"
    if score >= -0.2: return "Neutral"
    if score >= -0.6: return "Lean Bear"
    return "Strong Bear"


# ── Per-scanner normalizers ────────────────────────────────────────

def _norm_simple(data: Optional[dict]) -> ScannerVote:
    """Simple Scanner → direction from bull_score vs bear_score + signal."""
    if not data:
        return ScannerVote("simple", 0, 0, "No Data", "—")

    bull  = _safe(data, "bull_score", 0) or 0
    bear  = _safe(data, "bear_score", 0) or 0
    total = bull + bear
    if total == 0:
        direction = 0.0
        conf = 0.1
    else:
        direction = _clamp((bull - bear) / total)
        conf = min(total / 10, 1.0)        # 10-point scale → full conf

    signal = str(_safe(data, "signal", "")).upper()
    if "STRONG" in signal and "BULL" in signal:
        direction = max(direction, 0.7)
    elif "STRONG" in signal and "BEAR" in signal:
        direction = min(direction, -0.7)

    rsi = _safe(data, "rsi")
    summary = f"Signal={_safe(data,'signal','?')}  Bull={bull} Bear={bear}"
    if rsi is not None:
        summary += f"  RSI={rsi}"

    return ScannerVote("simple", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


def _norm_mtf_raw(data: Optional[dict]) -> ScannerVote:
    """MTF Raw → dominant direction + weighted score."""
    if not data:
        return ScannerVote("mtf_raw", 0, 0, "No Data", "—")

    dom = str(_safe(data, "dominant_direction", "")).upper()
    w_score = _safe(data, "weighted_score", 0) or 0

    # weighted_score is typically on a -10…+10 or similar scale
    direction = _clamp(w_score / 10)

    # reinforce with dominant direction text
    if "BULL" in dom:
        direction = max(direction, 0.1)
    elif "BEAR" in dom:
        direction = min(direction, -0.1)

    # count how many timeframes reported
    tfs = ["30m", "1h", "2h", "4h"]
    reported = sum(1 for tf in tfs if _safe(data, f"{tf}_signal") is not None)
    conf = reported / len(tfs)

    summary = f"Dom={dom}  WtScore={w_score}  TFs={reported}/{len(tfs)}"
    return ScannerVote("mtf_raw", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


def _norm_mtf_ai(data: Optional[dict]) -> ScannerVote:
    """MTF AI (Claude trade plan) → conviction + direction from entry bias."""
    if not data:
        return ScannerVote("mtf_ai", 0, 0, "No Data", "—")

    conviction = str(_safe(data, "conviction", "")).upper()
    bias       = str(_safe(data, "bias", _safe(data, "direction", ""))).upper()
    rr         = _safe(data, "risk_reward", _safe(data, "rr", 0)) or 0

    # direction from bias
    if "BULL" in bias or "LONG" in bias or "CALL" in bias:
        direction = 0.5
    elif "BEAR" in bias or "SHORT" in bias or "PUT" in bias:
        direction = -0.5
    else:
        direction = 0.0

    # amplify by conviction grade
    conv_map = {"A+": 1.0, "A": 0.9, "B+": 0.8, "B": 0.7, "C+": 0.6,
                "C": 0.5, "D": 0.3, "F": 0.1,
                "HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    conf = conv_map.get(conviction, 0.5)

    # R:R amplifier — higher R:R increases magnitude
    if rr and float(rr) > 0:
        direction *= min(float(rr) / 2, 1.5)  # cap at 1.5×
        direction = _clamp(direction)

    summary = f"Bias={bias}  Conviction={conviction}  R:R={rr}"
    return ScannerVote("mtf_ai", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


def _norm_signal_quick(data: Optional[dict]) -> ScannerVote:
    """Signal Quick → call/put hit-rate differential."""
    if not data:
        return ScannerVote("signal_quick", 0, 0, "No Data", "—")

    call_1d = _safe(data, "call_hit_1d", 0) or 0
    put_1d  = _safe(data, "put_hit_1d", 0)  or 0
    call_3d = _safe(data, "call_hit_3d", 0) or 0
    put_3d  = _safe(data, "put_hit_3d", 0)  or 0

    # blend 1D and 3D (equal weight)
    call_avg = (call_1d + call_3d) / 2
    put_avg  = (put_1d  + put_3d)  / 2
    total    = call_avg + put_avg

    if total == 0:
        direction = 0.0
        conf = 0.1
    else:
        direction = _clamp((call_avg - put_avg) / total * 2)   # ×2 to spread
        conf = min(total / 100, 1.0)       # hit rates near 100% → high conf

    straddle = _safe(data, "straddle_rate")
    summary = f"Call1D={call_1d}% Put1D={put_1d}%  Call3D={call_3d}% Put3D={put_3d}%"
    if straddle is not None:
        summary += f"  Straddle={straddle}%"

    return ScannerVote("signal_quick", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


def _norm_options_flow(data: Optional[dict]) -> ScannerVote:
    """Options Flow → sentiment + P/C ratio + unusual activity.
    Handles both generic keys and real options_flow_scanner camelCase keys."""
    if not data:
        return ScannerVote("options_flow", 0, 0, "No Data", "—")

    sentiment = str(_safe(data, "sentiment", "")).upper()
    # Real scanner uses pcVolumeRatio / pcOIRatio, fallback to pc_ratio
    pc_ratio  = (_safe(data, "pcVolumeRatio") or _safe(data, "pc_ratio", 1.0)) or 1.0
    unusual_count = _safe(data, "unusualCount", 0) or 0
    unusual   = unusual_count > 0 or _safe(data, "unusual_activity", False)

    # P/C < 0.7 is bullish, > 1.3 is bearish
    pc_dir = _clamp((1.0 - pc_ratio) * 2)  # invert: low PC → positive

    # sentiment override
    if "BULL" in sentiment:
        sent_dir = 0.4
    elif "BEAR" in sentiment:
        sent_dir = -0.4
    else:
        sent_dir = 0.0

    direction = _clamp((pc_dir + sent_dir) / 2 * 1.5)
    conf = 0.7
    if unusual:
        conf = 0.9     # unusual flow → higher confidence in reading

    summary = f"Sentiment={sentiment}  P/C={pc_ratio}  Unusual={unusual_count or 'no'}"
    # Real scanner uses avgIV, fallback to iv
    iv = _safe(data, "avgIV") or _safe(data, "iv")
    if iv is not None:
        summary += f"  IV={iv}"

    return ScannerVote("options_flow", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


def _norm_war_room(data: Optional[dict]) -> ScannerVote:
    """War Room → regime + exhaustion + fade conviction."""
    if not data:
        return ScannerVote("war_room", 0, 0, "No Data", "—")

    regime     = str(_safe(data, "regime", "")).upper()
    exhaustion = _safe(data, "exhaustion", 0) or 0     # 0-100 scale
    fade_conv  = _safe(data, "fade_conviction", 0) or 0

    # regime direction
    if "BULL" in regime or "UPTREND" in regime or "MARKUP" in regime:
        direction = 0.5
    elif "BEAR" in regime or "DOWNTREND" in regime or "MARKDOWN" in regime:
        direction = -0.5
    elif "RANGE" in regime or "CHOP" in regime:
        direction = 0.0
    else:
        direction = 0.0

    # if exhaustion is high, dampen or flip direction
    if exhaustion > 70:
        direction *= (1 - (exhaustion - 70) / 60)  # fade toward 0 then flip
    if fade_conv > 70:
        direction *= -0.5   # active fade signal

    conf = 0.6 + min(exhaustion / 200, 0.2) + min(fade_conv / 200, 0.2)

    summary = f"Regime={regime}  Exhaustion={exhaustion}  FadeConv={fade_conv}"
    return ScannerVote("war_room", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


def _norm_buffett(data: Optional[dict]) -> ScannerVote:
    """Buffett → value grade + blood score (fear) + range position.
    Handles both generic keys and real buffett_scanner camelCase keys."""
    if not data:
        return ScannerVote("buffett", 0, 0, "No Data", "—")

    grade_map = {"A+": 1.0, "A": 0.85, "B+": 0.7, "B": 0.55,
                 "C+": 0.4, "C": 0.25, "D": 0.1, "F": -0.2}
    # Real scanner uses "grade", fallback to "value_grade"
    grade_raw = str(_safe(data, "grade", _safe(data, "value_grade", "C"))).upper()
    grade_val = grade_map.get(grade_raw, 0.25)

    # Real scanner uses "bloodScore" (0-100), fallback to "blood_score"
    blood    = (_safe(data, "bloodScore") or _safe(data, "blood_score", 0)) or 0
    # Real scanner uses "rangePosition" (0-1.0), fallback to "range_position" (0-100)
    range_pos_raw = _safe(data, "rangePosition", _safe(data, "range_position", 50))
    range_pos = range_pos_raw * 100 if range_pos_raw is not None and range_pos_raw <= 1.0 else (range_pos_raw or 50)

    # value direction: high grade + high blood + low range = buy
    dir_grade = grade_val - 0.5         # center around 0
    dir_blood = _clamp((blood - 50) / 50)   # >50 is fearful → bullish contrarian
    dir_range = _clamp((50 - range_pos) / 50)  # low in range → upside room

    direction = _clamp((dir_grade * 0.4 + dir_blood * 0.3 + dir_range * 0.3) * 2)
    conf = 0.5 if grade_raw in grade_map else 0.3

    rev = _safe(data, "revenueGrowth") or _safe(data, "revenue_growth")
    summary = f"Grade={grade_raw}  Blood={blood}  RangePos={range_pos:.0f}"
    if rev is not None:
        summary += f"  RevGrowth={rev}%"

    return ScannerVote("buffett", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


def _norm_sustainability(data: Optional[dict]) -> ScannerVote:
    """Sustainability → RS score + cycle phase + insider activity.
    Handles both generic keys and real RunSustainabilityAnalyzer output."""
    if not data:
        return ScannerVote("sustainability", 0, 0, "No Data", "—")

    # Real analyzer uses overall_score (0-100), fallback to rs_score
    rs_score = _safe(data, "overall_score", _safe(data, "rs_score", 50)) or 50

    # Real analyzer nests cycle_position.estimated_cycle_phase
    phase = str(
        _safe(data, "cycle_position.estimated_cycle_phase",
              _safe(data, "cycle_phase", ""))
    ).upper()

    # Real analyzer nests smart_money.insider_net_signal
    insider = str(
        _safe(data, "smart_money.insider_net_signal",
              _safe(data, "insider_activity", ""))
    ).upper()

    # RS > 60 bullish, < 40 bearish
    direction = _clamp((rs_score - 50) / 50)

    # cycle phase modifier
    phase_map = {"EARLY": 0.3, "MID": 0.15, "LATE": -0.1, "EXTENDED": -0.3,
                 "ACCUMULATION": 0.3, "DISTRIBUTION": -0.3}
    direction += phase_map.get(phase, 0)
    direction = _clamp(direction)

    # insider buying/selling
    if "BUY" in insider:
        direction = _clamp(direction + 0.15)
    elif "SELL" in insider:
        direction = _clamp(direction - 0.15)

    conf = 0.5
    if rs_score > 70 or rs_score < 30:
        conf = 0.7     # extreme RS → more conviction

    summary = f"Score={rs_score}  Phase={phase}  Insider={insider}"
    return ScannerVote("sustainability", _clamp(direction), _clamp(conf, 0, 1),
                       _direction_label(direction), summary)


# ── Registry ───────────────────────────────────────────────────────

NORMALIZERS = {
    "simple":         _norm_simple,
    "mtf_raw":        _norm_mtf_raw,
    "mtf_ai":         _norm_mtf_ai,
    "signal_quick":   _norm_signal_quick,
    "options_flow":   _norm_options_flow,
    "war_room":       _norm_war_room,
    "buffett":        _norm_buffett,
    "sustainability": _norm_sustainability,
}


# ── Core engine ────────────────────────────────────────────────────

@dataclass
class ConvergenceResult:
    """Full result for one ticker."""
    ticker: str
    convergence_score: float        # 0-100 (how aligned are all scanners)
    direction_score: float          # -1 … +1 (net direction)
    direction_label: str
    alert_type: str                 # "CONVERGENCE" | "DIVERGENCE" | "MIXED"
    conviction: str                 # A+ through F
    votes: List[Dict[str, Any]]     # per-scanner breakdown
    bulls: List[str]                # scanner names voting bull
    bears: List[str]                # scanner names voting bear
    neutrals: List[str]
    agreement_pct: float            # % of scanners agreeing with majority
    summary: str

    def to_dict(self) -> dict:
        return {
            "ticker":             self.ticker,
            "convergence_score":  round(self.convergence_score, 1),
            "direction_score":    round(self.direction_score, 3),
            "direction_label":    self.direction_label,
            "alert_type":         self.alert_type,
            "conviction":         self.conviction,
            "agreement_pct":      round(self.agreement_pct, 1),
            "bulls":              self.bulls,
            "bears":              self.bears,
            "neutrals":           self.neutrals,
            "votes":              self.votes,
            "summary":            self.summary,
        }


def score_convergence(
    ticker: str,
    scanner_outputs: Dict[str, Optional[dict]],
    profile: str = "equal",
    custom_weights: Optional[Dict[str, float]] = None,
) -> ConvergenceResult:
    """
    Main entry point.

    Parameters
    ----------
    ticker : str
        The symbol being scored.
    scanner_outputs : dict
        Keys are scanner names (e.g. "simple", "mtf_raw", …),
        values are the raw dicts each scanner already returns.
        Missing / None values are handled gracefully.
    profile : str
        One of "daytrade", "swing", "equal".
    custom_weights : dict, optional
        Overrides the profile weights if provided.

    Returns
    -------
    ConvergenceResult
    """
    weights = custom_weights or WEIGHT_PROFILES.get(profile, WEIGHT_PROFILES["equal"])

    # ---- 1. Normalize every scanner --------------------------------
    votes: List[ScannerVote] = []
    for name, normalizer in NORMALIZERS.items():
        raw = scanner_outputs.get(name)
        vote = normalizer(raw)
        votes.append(vote)

    # ---- 2. Weighted direction score --------------------------------
    weighted_sum = 0.0
    weight_total = 0.0
    for v in votes:
        w = weights.get(v.scanner, 0.125)
        weighted_sum += v.direction * v.confidence * w
        weight_total += w * v.confidence

    direction_score = _clamp(weighted_sum / weight_total if weight_total else 0)

    # ---- 3. Agreement / convergence metrics -------------------------
    bulls    = [v.scanner for v in votes if v.direction >  0.15]
    bears    = [v.scanner for v in votes if v.direction < -0.15]
    neutrals = [v.scanner for v in votes if -0.15 <= v.direction <= 0.15]

    majority_count = max(len(bulls), len(bears), len(neutrals))
    active_voters  = len([v for v in votes if v.confidence > 0])
    agreement_pct  = (majority_count / active_voters * 100) if active_voters else 0

    # Convergence score: how tightly clustered are the directional votes?
    if len(votes) > 1:
        directions = [v.direction for v in votes if v.confidence > 0]
        mean_dir   = sum(directions) / len(directions) if directions else 0
        variance   = sum((d - mean_dir)**2 for d in directions) / len(directions) if directions else 0
        std_dev    = math.sqrt(variance)
        # Low std_dev → high convergence.  Max std_dev ~1.0 (all over the place)
        convergence_score = _clamp((1.0 - std_dev) * 100, 0, 100)
    else:
        convergence_score = 50.0

    # ---- 4. Classify alert type ------------------------------------
    if convergence_score >= 70 and abs(direction_score) >= 0.25:
        alert_type = "CONVERGENCE"
    elif convergence_score < 40 and len(bulls) >= 2 and len(bears) >= 2:
        alert_type = "DIVERGENCE"
    else:
        alert_type = "MIXED"

    # ---- 5. Conviction letter grade --------------------------------
    raw_conv = convergence_score * abs(direction_score)  # 0-100 scale
    if   raw_conv >= 80: conviction = "A+"
    elif raw_conv >= 65: conviction = "A"
    elif raw_conv >= 50: conviction = "B+"
    elif raw_conv >= 40: conviction = "B"
    elif raw_conv >= 30: conviction = "C+"
    elif raw_conv >= 20: conviction = "C"
    elif raw_conv >= 10: conviction = "D"
    else:                conviction = "F"

    # ---- 6. Build human-readable summary ---------------------------
    dir_word = "BULL" if direction_score > 0.15 else ("BEAR" if direction_score < -0.15 else "NEUTRAL")
    summary = (
        f"{ticker} → {alert_type} alert | {dir_word} bias "
        f"(score {direction_score:+.2f}) | "
        f"Convergence {convergence_score:.0f}/100 | "
        f"Conviction {conviction} | "
        f"Agreement {agreement_pct:.0f}% "
        f"({len(bulls)}B / {len(bears)}S / {len(neutrals)}N)"
    )

    # ---- 7. Serialize votes ----------------------------------------
    vote_dicts = [
        {
            "scanner":    v.scanner,
            "direction":  round(v.direction, 3),
            "confidence": round(v.confidence, 2),
            "label":      v.label,
            "raw":        v.raw_summary,
        }
        for v in votes
    ]

    return ConvergenceResult(
        ticker=ticker,
        convergence_score=convergence_score,
        direction_score=direction_score,
        direction_label=_direction_label(direction_score),
        alert_type=alert_type,
        conviction=conviction,
        votes=vote_dicts,
        bulls=bulls,
        bears=bears,
        neutrals=neutrals,
        agreement_pct=agreement_pct,
        summary=summary,
    )


# ── Batch helper (scan a whole watchlist) ──────────────────────────

def score_watchlist(
    watchlist: List[str],
    fetch_fn,
    profile: str = "equal",
    custom_weights: Optional[Dict[str, float]] = None,
    sort_by: str = "convergence",        # "convergence" | "direction" | "conviction"
    min_convergence: float = 0,
    alert_filter: Optional[str] = None,  # "CONVERGENCE" | "DIVERGENCE" | None
) -> List[dict]:
    """
    Score every ticker in a watchlist.

    Parameters
    ----------
    watchlist : list[str]
        Ticker symbols.
    fetch_fn : callable(ticker) → dict
        A function that, given a ticker, returns a dict keyed by scanner
        name with each scanner's raw output.  This is where you plug in
        your existing scanner calls.

        Example:
            def fetch_all(ticker):
                return {
                    "simple":         run_simple_scanner(ticker),
                    "mtf_raw":        run_mtf_raw(ticker),
                    "mtf_ai":         run_mtf_ai(ticker),
                    "signal_quick":   run_signal_quick(ticker),
                    "options_flow":   run_options_flow(ticker),
                    "war_room":       run_war_room(ticker),
                    "buffett":        run_buffett(ticker),
                    "sustainability": run_sustainability(ticker),
                }
    profile : str
    custom_weights : dict, optional
    sort_by : str
    min_convergence : float
    alert_filter : str, optional

    Returns
    -------
    list[dict]  — sorted list of ConvergenceResult.to_dict()
    """
    results: List[ConvergenceResult] = []
    for ticker in watchlist:
        scanner_data = fetch_fn(ticker)
        result = score_convergence(ticker, scanner_data, profile, custom_weights)
        if result.convergence_score >= min_convergence:
            if alert_filter is None or result.alert_type == alert_filter:
                results.append(result)

    # sort
    if sort_by == "direction":
        results.sort(key=lambda r: abs(r.direction_score), reverse=True)
    elif sort_by == "conviction":
        grade_order = {"A+": 0, "A": 1, "B+": 2, "B": 3, "C+": 4, "C": 5, "D": 6, "F": 7}
        results.sort(key=lambda r: grade_order.get(r.conviction, 99))
    else:  # convergence (default)
        results.sort(key=lambda r: r.convergence_score, reverse=True)

    return [r.to_dict() for r in results]

# ── Step 8 Integration: Score from Alpha Pipeline Candidate ────────
#
# This avoids re-fetching scanners that the alpha pipeline already ran.
# Maps the enrichment dicts (c["squeeze"], c["odds"], c["war_room"],
# c["structure"]) into the format the normalizers expect.
# The 3 scanners NOT in the alpha pipeline (options_flow, buffett,
# sustainability) can be passed in via `extra_data` if fetched separately.

def _reshape_simple_from_candidate(c: dict) -> dict:
    """Reshape alpha candidate root-level fields into simple scanner format."""
    return {
        "bull_score":  c.get("scan_score", 0) * 0.1,  # 0-100 → ~0-10
        "bear_score":  max(0, 10 - c.get("scan_score", 0) * 0.1),
        "signal":      c.get("scanner_signal", c.get("direction", "")),
        "rsi":         c.get("rsi", 50),
    }


def _reshape_signal_quick_from_odds(odds: dict) -> dict:
    """Reshape alpha _check_odds output into signal_quick normalizer format."""
    return {
        "call_hit_1d": odds.get("call_hit_1d", 0),
        "call_hit_3d": odds.get("call_hit_3d", 0),
        "put_hit_1d":  0,   # alpha pipeline is bullish-only, no put stats stored
        "put_hit_3d":  0,
        "straddle_rate": odds.get("straddle_rate", 0),
    }


def _reshape_war_room(wr: dict) -> dict:
    """War room data is already in the right shape — pass through."""
    return wr


def _reshape_structure_as_sustainability(struct: dict) -> dict:
    """
    Map structure data to sustainability normalizer format as a proxy.
    range_position_52w → rs_score (higher range pos ≈ relative strength),
    pattern → cycle_phase.
    """
    pattern = struct.get("pattern", "")
    if "UPTREND" in pattern or "HH+HL" in pattern:
        phase = "MID"
    elif "HIGHER LOWS" in pattern:
        phase = "EARLY"
    elif "HIGHER HIGHS" in pattern:
        phase = "MID"
    else:
        phase = ""

    return {
        "rs_score":        struct.get("range_position_52w", 50),
        "cycle_phase":     phase,
        "insider_activity": "",
    }


def score_from_alpha_candidate(
    candidate: dict,
    extra_data: Optional[Dict[str, dict]] = None,
    profile: str = "equal",
) -> ConvergenceResult:
    """
    Step 8 entry point — score convergence from an alpha pipeline candidate.

    The alpha pipeline already collected:
      - Root: scan_score, rsi, rvol, direction, scanner_signal
      - c["squeeze"]:   squeeze_score, squeeze_status, has_squeeze
      - c["odds"]:      call_hit_3d, call_win_1d, straddle_rate, regime, zscore, ...
      - c["war_room"]:  exhaustion, fade_conviction, thin_top_pct, ...
      - c["structure"]: bullish_structure, structure_score, pattern, range_position_52w

    extra_data can supply the 3 missing scanners:
      - "options_flow": raw options_flow_scanner output
      - "buffett":      raw buffett_scanner output
      - "sustainability": raw sustainability output
    (If not provided, those votes will be Neutral/low-confidence)

    Parameters
    ----------
    candidate : dict
        The enriched alpha candidate dict (after Steps 3-6).
    extra_data : dict, optional
        Keys: "options_flow", "buffett", "sustainability"
    profile : str
        Weight profile — "daytrade", "swing", "equal"

    Returns
    -------
    ConvergenceResult
    """
    extra = extra_data or {}
    ticker = candidate.get("symbol", "???")

    # Build the scanner_outputs dict from what we already have
    scanner_outputs = {
        "simple":         _reshape_simple_from_candidate(candidate),
        "mtf_raw":        None,  # not in alpha pipeline — skip
        "mtf_ai":         None,  # not in alpha pipeline — skip
        "signal_quick":   _reshape_signal_quick_from_odds(candidate.get("odds", {})),
        "options_flow":   extra.get("options_flow"),
        "war_room":       _reshape_war_room(candidate.get("war_room", {})),
        "buffett":        extra.get("buffett"),
        "sustainability": extra.get("sustainability",
                                    _reshape_structure_as_sustainability(candidate.get("structure", {}))),
    }

    # Use adjusted weights for pipeline mode — zero out MTF since we don't have it
    pipeline_weights = dict(WEIGHT_PROFILES.get(profile, WEIGHT_PROFILES["equal"]))
    # Redistribute MTF weight to the scanners we DO have
    mtf_total = pipeline_weights.pop("mtf_raw", 0) + pipeline_weights.pop("mtf_ai", 0)
    have_keys = [k for k in pipeline_weights if scanner_outputs.get(k) is not None]
    if have_keys:
        bonus = mtf_total / len(have_keys)
        for k in have_keys:
            pipeline_weights[k] = pipeline_weights.get(k, 0) + bonus

    return score_convergence(
        ticker=ticker,
        scanner_outputs=scanner_outputs,
        profile=profile,
        custom_weights=pipeline_weights,
    )