"""
Automated Stock Scanner
=======================
Background loop that scans stocks every 30 minutes during market hours
and posts actionable setups to Discord.

Scan Tiers:
- Every 30 min: Squeeze detection + Dual setup scan on top watchlist symbols
- Every 60 min: Capitulation/euphoria detection on broader list
- Results posted as rich Discord embeds via webhook

Author: SEF Trading Systems
"""

import asyncio
import os
import traceback
from datetime import datetime, time, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Eastern Time helper (handles EST/EDT automatically if zoneinfo available)
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    # Fallback: assume EST (-5) if zoneinfo not available
    ET = timezone(timedelta(hours=-5))

def _now_et() -> datetime:
    """Get current time in Eastern Time"""
    return datetime.now(ET)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCAN_INTERVAL_MINUTES = 30
MARKET_OPEN = time(9, 30)    # ET
MARKET_CLOSE = time(16, 0)   # ET
PRE_MARKET_START = time(8, 0)  # Start scanning pre-market

# Default symbols if watchlist unavailable (centralized)
from universe import AUTO_SCANNER_DEFAULTS as DEFAULT_SYMBOLS

# =============================================================================
# SCANNER STATE
# =============================================================================

@dataclass
class ScanResult:
    """Aggregated scan results for a single cycle"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    squeeze_setups: List[Dict] = field(default_factory=list)
    dual_setups: List[Dict] = field(default_factory=list)
    capitulation_signals: List[Dict] = field(default_factory=list)
    euphoria_signals: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    scan_count: int = 0


class AutoScanner:
    """
    Background scanner that runs on a schedule and posts results to Discord.
    
    Uses existing scanner modules:
    - squeeze_detector_v2.scan_for_squeezes_v2()
    - dual_setup_generator_v2.scan_symbols()
    - capitulation_detector_v2.scan_both()
    """

    def __init__(self, watchlist_mgr=None, discord_client=None):
        self.watchlist_mgr = watchlist_mgr
        self.discord_client = discord_client
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_result: Optional[ScanResult] = None
        self._scan_history: List[ScanResult] = []

        # Import scanners lazily
        self._squeeze_scanner = None
        self._dual_scanner = None
        self._cap_scanner = None

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Start the background scan loop"""
        if self._running:
            print("⚠️ Auto-scanner already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._scan_loop())
        print(f"🔄 Auto-scanner started — scanning every {SCAN_INTERVAL_MINUTES} min during market hours")

    def stop(self):
        """Stop the background scan loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        print("⏹️ Auto-scanner stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_result(self) -> Optional[ScanResult]:
        return self._last_result

    @property
    def status(self) -> Dict:
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "interval_minutes": SCAN_INTERVAL_MINUTES,
            "last_scan": self._last_result.timestamp.isoformat() if self._last_result else None,
            "last_squeeze_count": len(self._last_result.squeeze_setups) if self._last_result else 0,
            "last_dual_count": len(self._last_result.dual_setups) if self._last_result else 0,
            "last_cap_count": len(self._last_result.capitulation_signals) if self._last_result else 0,
            "last_euph_count": len(self._last_result.euphoria_signals) if self._last_result else 0,
        }

    # =========================================================================
    # SCAN LOOP
    # =========================================================================

    async def _scan_loop(self):
        """Main loop: scan every N minutes during market hours"""
        print(f"🔄 Auto-scanner loop started, interval={SCAN_INTERVAL_MINUTES}m")

        # Small initial delay to let server finish starting
        await asyncio.sleep(10)

        while self._running:
            try:
                if self._is_scan_window():
                    self._cycle_count += 1
                    print(f"\n{'='*60}")
                    print(f"  🔍 AUTO-SCAN CYCLE #{self._cycle_count} — {_now_et().strftime('%I:%M %p ET')}")
                    print(f"{'='*60}")

                    result = await self._run_scan_cycle()
                    self._last_result = result
                    self._scan_history.append(result)

                    # Keep only last 20 results
                    if len(self._scan_history) > 20:
                        self._scan_history = self._scan_history[-20:]

                    # Circuit breaker: skip posting if scan is mostly failures
                    if self._should_skip_post(result):
                        print(f"  ⚠️ Circuit breaker: too many errors/empty results — skipping Discord post")
                    else:
                        # Post to Discord
                        await self._post_results(result)

                    print(f"  ✅ Scan cycle #{self._cycle_count} complete — "
                          f"Squeezes: {len(result.squeeze_setups)}, "
                          f"Setups: {len(result.dual_setups)}, "
                          f"Capitulations: {len(result.capitulation_signals)}")
                else:
                    # Outside market hours — check less frequently
                    pass

                # Wait for next interval
                await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)

            except asyncio.CancelledError:
                print("🔄 Auto-scanner loop cancelled")
                break
            except Exception as e:
                print(f"❌ Auto-scanner error: {e}")
                traceback.print_exc()
                # Don't crash — wait and retry
                await asyncio.sleep(60)

    def _is_scan_window(self) -> bool:
        """Check if we're in the scanning window (pre-market through close)"""
        now = _now_et()
        current_time = now.time()
        weekday = now.weekday()

        # Skip weekends
        if weekday >= 5:
            return False

        # Scan from pre-market through close
        return PRE_MARKET_START <= current_time <= MARKET_CLOSE

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _should_skip_post(self, result: ScanResult) -> bool:
        """
        Circuit breaker: skip Discord post if scan results indicate data source failure.

        Triggers when:
        - More than 50% of scan types errored out
        - All scan types returned zero results AND there are errors
        - 3+ consecutive cycles with zero results (data source likely down)
        """
        # Count how many scan types errored
        total_scan_types = 3  # squeeze, dual, capitulation
        error_count = len(result.errors)

        if error_count >= 2:
            return True

        # All empty + errors = data source probably down
        total_results = (
            len(result.squeeze_setups) + len(result.dual_setups)
            + len(result.capitulation_signals) + len(result.euphoria_signals)
        )
        if total_results == 0 and error_count > 0:
            return True

        # Check for consecutive empty cycles (3+)
        if len(self._scan_history) >= 3:
            recent = self._scan_history[-3:]
            all_empty = all(
                (len(r.squeeze_setups) + len(r.dual_setups)
                 + len(r.capitulation_signals) + len(r.euphoria_signals)) == 0
                for r in recent
            )
            if all_empty:
                return True

        return False

    # =========================================================================
    # SCANNER EXECUTION
    # =========================================================================

    def _get_scan_symbols(self) -> List[str]:
        """Get symbols to scan from watchlist manager or defaults"""
        symbols = []

        if self.watchlist_mgr:
            try:
                from watchlist_manager import quick_scan_list
                # Get mega cap tech + top S&P names
                mega = quick_scan_list(self.watchlist_mgr, "mega")
                nasdaq = quick_scan_list(self.watchlist_mgr, "nasdaq")
                dow = quick_scan_list(self.watchlist_mgr, "dow")

                # Merge and deduplicate, cap at 50
                seen = set()
                for sym in mega + nasdaq + dow:
                    if sym not in seen:
                        symbols.append(sym)
                        seen.add(sym)
                    if len(symbols) >= 50:
                        break

            except Exception as e:
                print(f"⚠️ Watchlist error, using defaults: {e}")

        if not symbols:
            symbols = DEFAULT_SYMBOLS.copy()

        return symbols

    async def _run_scan_cycle(self) -> ScanResult:
        """Execute all scan types for this cycle"""
        result = ScanResult()
        symbols = self._get_scan_symbols()
        result.scan_count = len(symbols)

        print(f"  📋 Scanning {len(symbols)} symbols...")

        # Run scans in thread pool (they use yfinance which blocks)
        loop = asyncio.get_event_loop()

        # === SQUEEZE SCAN (every cycle) ===
        try:
            print(f"  🔧 Running squeeze detection...")
            squeezes = await loop.run_in_executor(
                None, self._scan_squeezes, symbols
            )
            result.squeeze_setups = squeezes
            print(f"    → Found {len(squeezes)} squeeze setups")
        except Exception as e:
            err = f"Squeeze scan error: {e}"
            result.errors.append(err)
            print(f"  ❌ {err}")

        # === DUAL SETUP SCAN (every cycle) ===
        try:
            print(f"  🔧 Running dual setup analysis...")
            setups = await loop.run_in_executor(
                None, self._scan_dual_setups, symbols[:30]  # Top 30 to manage time
            )
            result.dual_setups = setups
            print(f"    → Found {len(setups)} quality setups")
        except Exception as e:
            err = f"Dual setup scan error: {e}"
            result.errors.append(err)
            print(f"  ❌ {err}")

        # === CAPITULATION / EUPHORIA (every other cycle) ===
        if self._cycle_count % 2 == 0:
            try:
                print(f"  🔧 Running capitulation/euphoria detection...")
                cap_results = await loop.run_in_executor(
                    None, self._scan_capitulation, symbols[:40]
                )
                result.capitulation_signals = cap_results.get("capitulation", [])
                result.euphoria_signals = cap_results.get("euphoria", [])
                print(f"    → Capitulations: {len(result.capitulation_signals)}, "
                      f"Euphoria: {len(result.euphoria_signals)}")
            except Exception as e:
                err = f"Capitulation scan error: {e}"
                result.errors.append(err)
                print(f"  ❌ {err}")

        return result

    # =========================================================================
    # INDIVIDUAL SCANNERS (run in thread pool)
    # =========================================================================

    def _scan_squeezes(self, symbols: List[str]) -> List[Dict]:
        """Scan for squeeze setups using squeeze_detector_v2"""
        try:
            from squeeze_detector_v2 import scan_for_squeezes_v2
        except ImportError:
            print("  ⚠️ squeeze_detector_v2 not available")
            return []

        results = scan_for_squeezes_v2(symbols, min_tier="ACTIVE")

        # Convert to dicts for Discord formatting
        setups = []
        for m in results[:15]:  # Top 15
            setups.append({
                "symbol": m.symbol,
                "score": m.score,
                "tier": m.tier,
                "grade": getattr(m, 'quality_grade', '?'),
                "direction": getattr(m, 'direction_bias', 'N/A'),
                "setup_type": getattr(m, 'setup_type', 'N/A'),
                "entry_trigger": getattr(m, 'entry_trigger', ''),
                "ttm_squeeze": getattr(m, 'ttm_squeeze', False),
                "squeeze_duration": getattr(m, 'squeeze_duration', 0),
            })

        return setups

    def _scan_dual_setups(self, symbols: List[str]) -> List[Dict]:
        """Scan for dual (long+short) setups using dual_setup_generator_v2"""
        try:
            from dual_setup_generator_v2 import scan_symbols
        except ImportError:
            print("  ⚠️ dual_setup_generator_v2 not available")
            return []

        results = scan_symbols(symbols, min_grade="B")

        setups = []
        for r in results[:10]:  # Top 10
            preferred = (r.long_setup if r.preferred_direction == 'LONG'
                        else r.short_setup)
            setups.append({
                "symbol": r.symbol,
                "direction": r.preferred_direction,
                "grade": preferred.grade,
                "probability": preferred.probability if preferred.probability else 0,
                "quality_score": preferred.quality_score,
                "entry_zone": f"${preferred.entry_low:.2f} - ${preferred.entry_high:.2f}" if preferred.entry_low and preferred.entry_high else "N/A",
                "stop": f"${preferred.stop:.2f}" if preferred.stop else "N/A",
                "target_1": f"${preferred.target_1:.2f}" if preferred.target_1 else "N/A",
                "target_2": f"${preferred.target_2:.2f}" if preferred.target_2 else "N/A",
                "rr_ratio": f"{preferred.risk_reward:.1f}:1" if preferred.risk_reward else "N/A",
            })

        return setups

    def _scan_capitulation(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Scan for capitulation and euphoria extremes"""
        try:
            from capitulation_detector_v2 import scan_both
        except ImportError:
            print("  ⚠️ capitulation_detector_v2 not available")
            return {"capitulation": [], "euphoria": []}

        raw = scan_both(symbols, min_level="DEVELOPING")

        output = {"capitulation": [], "euphoria": []}

        for item in raw.get("capitulation", [])[:8]:
            level_val = getattr(item, 'capitulation_level', None)
            output["capitulation"].append({
                "symbol": getattr(item, 'symbol', str(item)),
                "score": getattr(item, 'capitulation_score', 0),
                "level": level_val.value if level_val else 'UNKNOWN',
                "description": getattr(item, 'description', ''),
            })

        for item in raw.get("euphoria", [])[:8]:
            level_val = getattr(item, 'euphoria_level', None)
            output["euphoria"].append({
                "symbol": getattr(item, 'symbol', str(item)),
                "score": getattr(item, 'euphoria_score', 0),
                "level": level_val.value if level_val else 'UNKNOWN',
                "description": getattr(item, 'description', ''),
            })

        return output

    # =========================================================================
    # DISCORD POSTING
    # =========================================================================

    async def _post_results(self, result: ScanResult):
        """Post scan results to Discord as rich embeds"""
        if not self.discord_client:
            print("  ⚠️ No Discord client — skipping post")
            return

        has_results = (result.squeeze_setups or result.dual_setups
                       or result.capitulation_signals or result.euphoria_signals)

        if not has_results:
            # Only post "no results" message every 4th cycle to reduce noise
            if self._cycle_count % 4 == 0:
                await self.discord_client.send_message(
                    content=f"📡 **Auto-Scan #{self._cycle_count}** — No actionable setups found "
                            f"({result.scan_count} symbols scanned at "
                            f"{_now_et().strftime('%I:%M %p ET')})"
                )
            return

        # === HEADER ===
        await self.discord_client.send_message(
            content=f"{'='*40}\n"
                    f"📡 **AUTO-SCAN #{self._cycle_count}** — "
                    f"{_now_et().strftime('%I:%M %p ET')} | "
                    f"{result.scan_count} symbols\n"
                    f"{'='*40}"
        )

        # === SQUEEZE SETUPS ===
        if result.squeeze_setups:
            await self._post_squeeze_embed(result.squeeze_setups)

        # === DUAL SETUPS ===
        if result.dual_setups:
            await self._post_dual_embed(result.dual_setups)

        # === CAPITULATION / EUPHORIA ===
        if result.capitulation_signals or result.euphoria_signals:
            await self._post_extremes_embed(
                result.capitulation_signals, result.euphoria_signals
            )

        # === ERRORS (if any) ===
        if result.errors:
            error_text = "\n".join(f"• {e}" for e in result.errors)
            await self.discord_client.send_message(
                content=f"⚠️ **Scan Issues:**\n{error_text}"
            )

    async def _post_squeeze_embed(self, squeezes: List[Dict]):
        """Post squeeze setups as a Discord embed"""
        description = ""
        for i, s in enumerate(squeezes[:10], 1):
            tier_emoji = {
                "TEXTBOOK": "🔥",
                "PRIME": "⚡",
                "ACTIVE": "🟢",
                "FORMING": "🟡"
            }.get(s["tier"], "⚪")

            direction_emoji = "⬆️" if "bull" in str(s.get("direction", "")).lower() else \
                              "⬇️" if "bear" in str(s.get("direction", "")).lower() else "↔️"

            description += (
                f"{i}. {tier_emoji} **{s['symbol']}** — "
                f"Score: **{s['score']}**/100 | {s['tier']} ({s['grade']})\n"
                f"   {direction_emoji} {s.get('setup_type', 'N/A')}"
            )
            if s.get("entry_trigger"):
                description += f" | Entry: {s['entry_trigger']}"
            description += "\n"

        ttm_count = sum(1 for s in squeezes if s.get("ttm_squeeze"))

        embed = {
            "title": f"🔧 Squeeze Setups ({len(squeezes)})",
            "description": description,
            "color": 0xFFC800,  # Gold
            "fields": [
                {"name": "TTM Squeeze Active", "value": f"{ttm_count} / {len(squeezes)}", "inline": True},
                {"name": "Top Score", "value": f"{squeezes[0]['score']}/100" if squeezes else "N/A", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Auto-Scanner | Squeeze Detector V2"}
        }

        await self.discord_client.send_message(embeds=[embed])

    async def _post_dual_embed(self, setups: List[Dict]):
        """Post dual setup results as a Discord embed"""
        description = ""
        for i, s in enumerate(setups[:8], 1):
            dir_emoji = "🟢" if s["direction"] == "LONG" else "🔴"
            grade_emoji = "🔥" if s["grade"] in ["A+", "A"] else "✅" if s["grade"] == "B" else "⚪"

            description += (
                f"{i}. {dir_emoji} **{s['symbol']}** {s['direction']} — "
                f"Grade: **{s['grade']}** {grade_emoji} | "
                f"Prob: {s['probability']}%\n"
                f"   Entry: {s['entry_zone']} | Stop: {s['stop']} | "
                f"T1: {s['target_1']} | R:R {s['rr_ratio']}\n"
            )

        embed = {
            "title": f"📊 Trade Setups ({len(setups)})",
            "description": description,
            "color": 0x00FF88,  # Green
            "fields": [
                {"name": "Best Grade", "value": setups[0]["grade"] if setups else "N/A", "inline": True},
                {"name": "Preferred", "value": f"{sum(1 for s in setups if s['direction']=='LONG')} Long / "
                                                f"{sum(1 for s in setups if s['direction']=='SHORT')} Short",
                 "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Auto-Scanner | Dual Setup Generator V2"}
        }

        await self.discord_client.send_message(embeds=[embed])

    async def _post_extremes_embed(self, capitulations: List[Dict], euphorias: List[Dict]):
        """Post capitulation/euphoria extremes as a Discord embed"""
        description = ""

        if capitulations:
            description += "**🟢 Capitulation (Potential Bottoms):**\n"
            for c in capitulations[:5]:
                description += f"  • **{c['symbol']}** — Score: {c['score']}/100 | {c['level']}\n"

        if euphorias:
            if capitulations:
                description += "\n"
            description += "**🔴 Euphoria (Potential Tops):**\n"
            for e in euphorias[:5]:
                description += f"  • **{e['symbol']}** — Score: {e['score']}/100 | {e['level']}\n"

        embed = {
            "title": f"⚡ Market Extremes ({len(capitulations)} caps / {len(euphorias)} euphs)",
            "description": description or "No extremes detected.",
            "color": 0xFF8800,  # Orange
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SEF Auto-Scanner | Capitulation Detector V2"}
        }

        await self.discord_client.send_message(embeds=[embed])

    # =========================================================================
    # MANUAL TRIGGER
    # =========================================================================

    async def run_now(self) -> ScanResult:
        """Manually trigger a scan cycle (regardless of market hours)"""
        self._cycle_count += 1
        print(f"\n🔍 MANUAL SCAN #{self._cycle_count} triggered")

        result = await self._run_scan_cycle()
        self._last_result = result
        self._scan_history.append(result)

        await self._post_results(result)
        return result


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_auto_scanner: Optional[AutoScanner] = None


def get_auto_scanner() -> Optional[AutoScanner]:
    """Get the global auto-scanner instance"""
    return _auto_scanner


def setup_auto_scanner(watchlist_mgr=None, discord_client=None, auto_start: bool = True) -> AutoScanner:
    """Initialize and optionally start the auto-scanner"""
    global _auto_scanner

    _auto_scanner = AutoScanner(
        watchlist_mgr=watchlist_mgr,
        discord_client=discord_client
    )

    if auto_start:
        _auto_scanner.start()

    return _auto_scanner
