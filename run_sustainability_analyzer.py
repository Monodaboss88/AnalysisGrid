"""
Run Sustainability Analyzer
============================
Analyzes whether a stock's 100%+ run is sustainable or exhausted.

Covers:
1. EPS Growth vs Price Growth (Multiple Expansion Detection)
2. Revenue Growth & Acceleration (Demand Visibility)
3. Margin Trends (Pricing Power / Cost Leverage)
4. Analyst Estimate Revisions (Forward Visibility Proxy)
5. Institutional & Insider Activity (Smart Money Signals)
6. Cycle Position Detection (Early / Mid / Late)
7. Valuation Stretch (P/E vs Historical, PEG Ratio)
8. Historical Precedent Scoring (How long 100%+ runs typically last)

Author: Rob's Trading Systems
Version: 1.0.0
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import traceback

# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────

@dataclass
class MultipleExpansion:
    """EPS growth vs price growth analysis"""
    eps_growth_pct: Optional[float]
    price_growth_pct: float
    multiple_expansion_pct: Optional[float]
    current_pe: Optional[float]
    avg_pe_5yr: Optional[float]
    pe_percentile: Optional[float]  # where current P/E sits in 5yr range
    peg_ratio: Optional[float]
    signal: str  # GREEN, YELLOW, RED
    detail: str

@dataclass
class RevenueHealth:
    """Revenue growth, acceleration, and visibility"""
    revenue_growth_yoy: Optional[float]
    revenue_growth_qoq: Optional[float]
    revenue_acceleration: Optional[float]  # is growth speeding up or slowing
    gross_margin_current: Optional[float]
    gross_margin_trend: Optional[str]  # EXPANDING, STABLE, CONTRACTING
    operating_margin_current: Optional[float]
    operating_margin_trend: Optional[str]
    quarters_of_growth: int
    signal: str
    detail: str

@dataclass
class AnalystSentiment:
    """Forward estimate revisions and consensus"""
    current_year_est: Optional[float]
    next_year_est: Optional[float]
    est_revision_trend: Optional[str]  # UP, FLAT, DOWN
    num_analysts: Optional[int]
    target_price: Optional[float]
    target_upside_pct: Optional[float]
    recommendation: Optional[str]
    signal: str
    detail: str

@dataclass
class SmartMoney:
    """Institutional and insider activity"""
    institutional_pct: Optional[float]
    institutional_change: Optional[str]
    insider_buys_6mo: int
    insider_sells_6mo: int
    insider_net_signal: str
    short_pct_float: Optional[float]
    short_signal: str
    signal: str
    detail: str

@dataclass
class CyclePosition:
    """Where we are in the business/stock cycle"""
    run_duration_months: float
    return_from_52w_low: float
    distance_from_52w_high_pct: float
    revenue_growth_trajectory: Optional[str]  # ACCELERATING, PEAK, DECELERATING
    margin_trajectory: Optional[str]
    estimated_cycle_phase: str  # EARLY, MID, LATE, EXTENDED
    historical_avg_run_months: float
    pct_through_typical_run: float
    signal: str
    detail: str

@dataclass
class SustainabilityScore:
    """Overall sustainability assessment"""
    symbol: str
    company_name: str
    current_price: float
    market_cap: float
    market_cap_tier: str  # MEGA, LARGE, MID, SMALL
    annual_return_pct: float
    ytd_return_pct: float
    
    # Component scores (0-100)
    multiple_expansion: MultipleExpansion
    revenue_health: RevenueHealth
    analyst_sentiment: AnalystSentiment
    smart_money: SmartMoney
    cycle_position: CyclePosition
    
    # Overall
    overall_score: int  # 0-100
    overall_grade: str  # A+, A, B+, B, C+, C, D, F
    sustainability_verdict: str  # STRONG RUN, CAUTION, LATE STAGE, OVEREXTENDED
    recommended_action: str
    key_risks: List[str]
    key_strengths: List[str]


# ─────────────────────────────────────────────────────────────
# Core Analyzer
# ─────────────────────────────────────────────────────────────

class RunSustainabilityAnalyzer:
    """Analyzes whether a stock's big run is sustainable"""
    
    # Historical benchmarks for run duration
    RUN_BENCHMARKS = {
        "MEGA": {"avg_months": 20, "max_typical": 30},    # >$200B
        "LARGE": {"avg_months": 16, "max_typical": 24},   # $50B-$200B
        "MID": {"avg_months": 12, "max_typical": 18},     # $10B-$50B
        "SMALL": {"avg_months": 9, "max_typical": 15},    # $2B-$10B
        "MICRO": {"avg_months": 6, "max_typical": 12},    # <$2B
    }
    
    def __init__(self):
        self.cache = {}
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Full sustainability analysis for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            
            # Get historical data
            hist_1y = ticker.history(period="1y")
            hist_2y = ticker.history(period="2y")
            hist_5y = ticker.history(period="5y")
            
            if hist_1y.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Basic info
            current_price = hist_1y['Close'].iloc[-1] if not hist_1y.empty else info.get('currentPrice', 0)
            market_cap = info.get('marketCap', 0) or 0
            market_cap_tier = self._get_cap_tier(market_cap)
            company_name = info.get('shortName', info.get('longName', symbol))
            
            # Calculate returns
            annual_return = self._calc_annual_return(hist_1y)
            ytd_return = self._calc_ytd_return(hist_1y)
            
            # Get financials
            quarterly_financials = None
            annual_financials = None
            quarterly_earnings = None
            try:
                quarterly_financials = ticker.quarterly_financials
                annual_financials = ticker.financials
                quarterly_earnings = ticker.quarterly_earnings
            except:
                pass
            
            # Run all analyses
            multiple_exp = self._analyze_multiple_expansion(info, hist_1y, hist_5y, quarterly_earnings)
            rev_health = self._analyze_revenue_health(info, quarterly_financials, annual_financials)
            analyst_sent = self._analyze_analyst_sentiment(info, ticker)
            smart = self._analyze_smart_money(info, ticker)
            cycle = self._analyze_cycle_position(info, hist_1y, hist_2y, market_cap_tier, rev_health)
            
            # Calculate overall score
            scores = []
            weights = []
            
            score_map = {"GREEN": 85, "YELLOW": 55, "RED": 20}
            
            scores.append(score_map.get(multiple_exp.signal, 50))
            weights.append(25)
            
            scores.append(score_map.get(rev_health.signal, 50))
            weights.append(25)
            
            scores.append(score_map.get(analyst_sent.signal, 50))
            weights.append(15)
            
            scores.append(score_map.get(smart.signal, 50))
            weights.append(15)
            
            scores.append(score_map.get(cycle.signal, 50))
            weights.append(20)
            
            overall_score = int(np.average(scores, weights=weights))
            overall_grade = self._score_to_grade(overall_score)
            verdict = self._score_to_verdict(overall_score)
            action = self._get_recommended_action(overall_score, cycle.estimated_cycle_phase, multiple_exp.signal)
            
            # Compile risks and strengths
            risks = self._identify_risks(multiple_exp, rev_health, analyst_sent, smart, cycle)
            strengths = self._identify_strengths(multiple_exp, rev_health, analyst_sent, smart, cycle)
            
            result = SustainabilityScore(
                symbol=symbol.upper(),
                company_name=company_name,
                current_price=round(current_price, 2),
                market_cap=market_cap,
                market_cap_tier=market_cap_tier,
                annual_return_pct=round(annual_return, 1),
                ytd_return_pct=round(ytd_return, 1),
                multiple_expansion=multiple_exp,
                revenue_health=rev_health,
                analyst_sentiment=analyst_sent,
                smart_money=smart,
                cycle_position=cycle,
                overall_score=overall_score,
                overall_grade=overall_grade,
                sustainability_verdict=verdict,
                recommended_action=action,
                key_risks=risks,
                key_strengths=strengths,
            )
            
            return self._to_dict(result)
            
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e), "symbol": symbol}
    
    def scan_multiple(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple symbols and rank by sustainability"""
        results = []
        for sym in symbols:
            r = self.analyze(sym)
            if "error" not in r:
                results.append(r)
        
        # Sort by overall score descending
        results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        return results
    
    # ─────────────────────────────────────────────────────────
    # Component Analyzers
    # ─────────────────────────────────────────────────────────
    
    def _analyze_multiple_expansion(self, info, hist_1y, hist_5y, quarterly_earnings) -> MultipleExpansion:
        """Check if run is earnings-driven or multiple-expansion-driven"""
        price_growth = self._calc_annual_return(hist_1y)
        
        # EPS growth
        eps_growth = None
        try:
            if quarterly_earnings is not None and not quarterly_earnings.empty and len(quarterly_earnings) >= 4:
                # Compare recent 4Q sum vs prior 4Q sum  
                recent_eps = quarterly_earnings['Reported EPS'].iloc[:4].sum()
                prior_eps = quarterly_earnings['Reported EPS'].iloc[4:8].sum() if len(quarterly_earnings) >= 8 else None
                if prior_eps and prior_eps > 0:
                    eps_growth = ((recent_eps - prior_eps) / abs(prior_eps)) * 100
        except:
            pass
        
        # Fallback: use trailing vs forward EPS
        if eps_growth is None:
            trailing_eps = info.get('trailingEps')
            # Try to approximate prior year EPS from P/E and price history
            if trailing_eps and trailing_eps > 0 and not hist_1y.empty:
                price_1y_ago = hist_1y['Close'].iloc[0]
                pe_1y_ago_est = price_1y_ago / trailing_eps if trailing_eps > 0 else None
        
        # Multiple expansion calculation
        multiple_expansion_pct = None
        if eps_growth is not None:
            multiple_expansion_pct = price_growth - eps_growth
        
        # Current P/E
        current_pe = info.get('trailingPE') or info.get('forwardPE')
        
        # 5-year P/E range estimate
        avg_pe_5yr = None
        pe_percentile = None
        forward_pe = info.get('forwardPE')
        trailing_pe = info.get('trailingPE')
        
        if trailing_pe and trailing_pe > 0:
            # Rough estimate using price history and current EPS
            trailing_eps = info.get('trailingEps', 0)
            if trailing_eps and trailing_eps > 0 and not hist_5y.empty:
                historical_pes = hist_5y['Close'] / trailing_eps
                # This is approximate since EPS changes, but gives directional signal
                avg_pe_5yr = round(float(historical_pes.median()), 1)
                pe_percentile = round(float((historical_pes < trailing_pe).mean() * 100), 0)
        
        # PEG ratio
        peg = info.get('pegRatio')
        
        # Signal determination
        if eps_growth is not None and multiple_expansion_pct is not None:
            if multiple_expansion_pct < 10:
                signal = "GREEN"
                detail = f"Earnings-driven run. EPS grew {eps_growth:.0f}% vs price {price_growth:.0f}%. Stock is getting cheaper on fundamentals."
            elif multiple_expansion_pct < 40:
                signal = "YELLOW"
                detail = f"Mixed run. EPS grew {eps_growth:.0f}%, price grew {price_growth:.0f}%. {multiple_expansion_pct:.0f}% was multiple expansion."
            else:
                signal = "RED"
                detail = f"Multiple expansion warning. EPS grew {eps_growth:.0f}% but price grew {price_growth:.0f}%. {multiple_expansion_pct:.0f}% is P/E stretch."
        elif peg is not None:
            if peg < 1.0:
                signal = "GREEN"
                detail = f"PEG ratio {peg:.1f} suggests growth justifies valuation."
            elif peg < 2.0:
                signal = "YELLOW"
                detail = f"PEG ratio {peg:.1f} — fairly valued for growth rate."
            else:
                signal = "RED"
                detail = f"PEG ratio {peg:.1f} — expensive relative to growth."
        else:
            signal = "YELLOW"
            detail = "Insufficient earnings data for precise multiple expansion calc. Using available metrics."
        
        return MultipleExpansion(
            eps_growth_pct=round(eps_growth, 1) if eps_growth else None,
            price_growth_pct=round(price_growth, 1),
            multiple_expansion_pct=round(multiple_expansion_pct, 1) if multiple_expansion_pct else None,
            current_pe=round(current_pe, 1) if current_pe else None,
            avg_pe_5yr=avg_pe_5yr,
            pe_percentile=pe_percentile,
            peg_ratio=round(peg, 2) if peg else None,
            signal=signal,
            detail=detail,
        )
    
    def _analyze_revenue_health(self, info, quarterly_fin, annual_fin) -> RevenueHealth:
        """Analyze revenue growth, acceleration, and margin trends"""
        rev_growth_yoy = info.get('revenueGrowth')
        if rev_growth_yoy is not None:
            rev_growth_yoy = round(rev_growth_yoy * 100, 1)
        
        rev_growth_qoq = None
        rev_acceleration = None
        quarters_of_growth = 0
        
        gross_margin = info.get('grossMargins')
        if gross_margin:
            gross_margin = round(gross_margin * 100, 1)
        
        operating_margin = info.get('operatingMargins')
        if operating_margin:
            operating_margin = round(operating_margin * 100, 1)
        
        gross_margin_trend = "UNKNOWN"
        operating_margin_trend = "UNKNOWN"
        
        # Analyze quarterly financials for trends
        try:
            if quarterly_fin is not None and not quarterly_fin.empty:
                rev_row = None
                for label in ['Total Revenue', 'Revenue', 'Operating Revenue']:
                    if label in quarterly_fin.index:
                        rev_row = quarterly_fin.loc[label]
                        break
                
                if rev_row is not None and len(rev_row) >= 2:
                    # QoQ growth
                    latest_rev = rev_row.iloc[0]
                    prior_rev = rev_row.iloc[1]
                    if prior_rev and prior_rev > 0:
                        rev_growth_qoq = round(((latest_rev - prior_rev) / prior_rev) * 100, 1)
                    
                    # Count consecutive quarters of growth
                    for i in range(len(rev_row) - 1):
                        if rev_row.iloc[i] > rev_row.iloc[i+1]:
                            quarters_of_growth += 1
                        else:
                            break
                    
                    # Revenue acceleration (compare recent growth rate vs prior)
                    if len(rev_row) >= 4:
                        recent_growth = (rev_row.iloc[0] - rev_row.iloc[1]) / abs(rev_row.iloc[1]) if rev_row.iloc[1] != 0 else 0
                        prior_growth = (rev_row.iloc[2] - rev_row.iloc[3]) / abs(rev_row.iloc[3]) if rev_row.iloc[3] != 0 else 0
                        rev_acceleration = round((recent_growth - prior_growth) * 100, 1)
                
                # Margin trends from quarterly data
                gp_row = None
                for label in ['Gross Profit']:
                    if label in quarterly_fin.index:
                        gp_row = quarterly_fin.loc[label]
                        break
                
                if gp_row is not None and rev_row is not None and len(gp_row) >= 4:
                    recent_margin = (gp_row.iloc[0] / rev_row.iloc[0]) if rev_row.iloc[0] != 0 else 0
                    older_margin = (gp_row.iloc[3] / rev_row.iloc[3]) if rev_row.iloc[3] != 0 else 0
                    margin_delta = (recent_margin - older_margin) * 100
                    if margin_delta > 1:
                        gross_margin_trend = "EXPANDING"
                    elif margin_delta < -1:
                        gross_margin_trend = "CONTRACTING"
                    else:
                        gross_margin_trend = "STABLE"
                
                op_row = None
                for label in ['Operating Income', 'EBIT']:
                    if label in quarterly_fin.index:
                        op_row = quarterly_fin.loc[label]
                        break
                
                if op_row is not None and rev_row is not None and len(op_row) >= 4:
                    recent_op_margin = (op_row.iloc[0] / rev_row.iloc[0]) if rev_row.iloc[0] != 0 else 0
                    older_op_margin = (op_row.iloc[3] / rev_row.iloc[3]) if rev_row.iloc[3] != 0 else 0
                    op_margin_delta = (recent_op_margin - older_op_margin) * 100
                    if op_margin_delta > 1:
                        operating_margin_trend = "EXPANDING"
                    elif op_margin_delta < -1:
                        operating_margin_trend = "CONTRACTING"
                    else:
                        operating_margin_trend = "STABLE"
        except Exception as e:
            pass
        
        # Signal
        signal = "YELLOW"
        detail = ""
        
        if rev_growth_yoy is not None:
            if rev_growth_yoy > 20 and gross_margin_trend in ["EXPANDING", "STABLE"]:
                signal = "GREEN"
                detail = f"Revenue growing {rev_growth_yoy}% YoY with {gross_margin_trend.lower()} margins. "
            elif rev_growth_yoy > 10:
                signal = "YELLOW" if gross_margin_trend != "CONTRACTING" else "RED"
                detail = f"Revenue growing {rev_growth_yoy}% YoY. Margins {gross_margin_trend.lower()}. "
            elif rev_growth_yoy > 0:
                signal = "YELLOW"
                detail = f"Revenue growing just {rev_growth_yoy}% YoY — slowing. "
            else:
                signal = "RED"
                detail = f"Revenue declining {rev_growth_yoy}% YoY — demand problem. "
        
        if rev_acceleration is not None:
            if rev_acceleration > 0:
                detail += f"Growth accelerating (+{rev_acceleration}pp). "
            else:
                detail += f"Growth decelerating ({rev_acceleration}pp). "
                if signal == "GREEN" and rev_acceleration < -5:
                    signal = "YELLOW"
        
        if quarters_of_growth >= 4:
            detail += f"{quarters_of_growth} consecutive quarters of revenue growth."
        
        if not detail:
            detail = "Limited financial data available for revenue analysis."
        
        return RevenueHealth(
            revenue_growth_yoy=rev_growth_yoy,
            revenue_growth_qoq=rev_growth_qoq,
            revenue_acceleration=rev_acceleration,
            gross_margin_current=gross_margin,
            gross_margin_trend=gross_margin_trend,
            operating_margin_current=operating_margin,
            operating_margin_trend=operating_margin_trend,
            quarters_of_growth=quarters_of_growth,
            signal=signal,
            detail=detail.strip(),
        )
    
    def _analyze_analyst_sentiment(self, info, ticker) -> AnalystSentiment:
        """Analyze analyst estimates, revisions, and targets"""
        current_est = info.get('forwardEps')
        next_year_est = None
        num_analysts = info.get('numberOfAnalystOpinions')
        target_price = info.get('targetMeanPrice')
        current_price = info.get('currentPrice', 0)
        recommendation = info.get('recommendationKey', '').upper()
        
        # Target upside
        target_upside = None
        if target_price and current_price and current_price > 0:
            target_upside = round(((target_price - current_price) / current_price) * 100, 1)
        
        # Estimate revision trend from earnings history
        est_revision_trend = None
        try:
            earnings_est = ticker.earnings_estimate
            if earnings_est is not None and not earnings_est.empty:
                pass
        except:
            pass
        
        # Use recommendation + target as proxy for estimate direction
        if recommendation in ['STRONG_BUY', 'BUY']:
            est_revision_trend = "UP"
        elif recommendation in ['HOLD']:
            est_revision_trend = "FLAT"
        elif recommendation in ['SELL', 'STRONG_SELL', 'UNDERPERFORM']:
            est_revision_trend = "DOWN"
        
        # Signal
        signal = "YELLOW"
        detail = ""
        
        if target_upside is not None:
            if target_upside > 15 and recommendation in ['STRONG_BUY', 'BUY']:
                signal = "GREEN"
                detail = f"Analysts see {target_upside}% upside. Consensus: {recommendation}. "
            elif target_upside > 0:
                signal = "YELLOW"
                detail = f"Analysts see {target_upside}% upside. Consensus: {recommendation}. "
            else:
                signal = "RED"
                detail = f"Analysts see {target_upside}% downside from here. Price has overshot consensus. "
        
        if num_analysts:
            detail += f"Covered by {num_analysts} analysts."
        
        if not detail:
            detail = "Limited analyst coverage data available."
        
        return AnalystSentiment(
            current_year_est=round(current_est, 2) if current_est else None,
            next_year_est=next_year_est,
            est_revision_trend=est_revision_trend,
            num_analysts=num_analysts,
            target_price=round(target_price, 2) if target_price else None,
            target_upside_pct=target_upside,
            recommendation=recommendation if recommendation else None,
            signal=signal,
            detail=detail.strip(),
        )
    
    def _analyze_smart_money(self, info, ticker) -> SmartMoney:
        """Analyze institutional ownership, insider trades, short interest"""
        institutional_pct = info.get('heldPercentInstitutions')
        if institutional_pct:
            institutional_pct = round(institutional_pct * 100, 1)
        
        short_pct = info.get('shortPercentOfFloat')
        if short_pct:
            short_pct = round(short_pct * 100, 1)
        
        # Insider transactions
        insider_buys = 0
        insider_sells = 0
        insider_change = "UNKNOWN"
        try:
            insider_txns = ticker.insider_transactions
            if insider_txns is not None and not insider_txns.empty:
                for _, txn in insider_txns.iterrows():
                    text = str(txn.get('Text', '')).lower()
                    if 'purchase' in text or 'buy' in text or 'acquisition' in text:
                        insider_buys += 1
                    elif 'sale' in text or 'sell' in text:
                        insider_sells += 1
        except:
            pass
        
        # Insider signal
        if insider_buys > insider_sells:
            insider_signal = "BULLISH"
        elif insider_sells > insider_buys * 2:
            insider_signal = "BEARISH"
        elif insider_sells > insider_buys:
            insider_signal = "CAUTIOUS"
        else:
            insider_signal = "NEUTRAL"
        
        # Institutional change signal
        inst_change = "UNKNOWN"
        
        # Short interest signal
        short_signal = "NEUTRAL"
        if short_pct:
            if short_pct > 10:
                short_signal = "HIGH_SHORT"
            elif short_pct > 5:
                short_signal = "MODERATE_SHORT"
            elif short_pct < 2:
                short_signal = "LOW_SHORT"
        
        # Overall signal
        signal = "YELLOW"
        detail = ""
        
        if institutional_pct:
            if institutional_pct > 70:
                detail += f"Strong institutional backing ({institutional_pct}% held). "
            elif institutional_pct > 40:
                detail += f"Moderate institutional ownership ({institutional_pct}%). "
            else:
                detail += f"Low institutional ownership ({institutional_pct}%) — higher volatility risk. "
        
        if insider_signal == "BULLISH":
            detail += f"Insiders buying ({insider_buys} buys vs {insider_sells} sells). "
            signal = "GREEN"
        elif insider_signal == "BEARISH":
            detail += f"Heavy insider selling ({insider_sells} sells vs {insider_buys} buys) — caution. "
            signal = "RED"
        elif insider_sells > 0 or insider_buys > 0:
            detail += f"Insider activity: {insider_buys} buys, {insider_sells} sells. "
        
        if short_pct:
            if short_pct > 10:
                detail += f"Short interest elevated at {short_pct}% of float."
            elif short_pct < 3:
                detail += f"Short interest low at {short_pct}% — bears not fighting this."
                if signal != "RED":
                    signal = "GREEN"
        
        if not detail:
            detail = "Limited smart money data available."
        
        return SmartMoney(
            institutional_pct=institutional_pct,
            institutional_change=inst_change,
            insider_buys_6mo=insider_buys,
            insider_sells_6mo=insider_sells,
            insider_net_signal=insider_signal,
            short_pct_float=short_pct,
            short_signal=short_signal,
            signal=signal,
            detail=detail.strip(),
        )
    
    def _analyze_cycle_position(self, info, hist_1y, hist_2y, cap_tier, rev_health) -> CyclePosition:
        """Determine where in the run cycle we are"""
        # Run duration — find the bottom of the run
        if not hist_2y.empty:
            low_idx = hist_2y['Close'].idxmin()
            low_date = pd.Timestamp(low_idx)
            now = pd.Timestamp(datetime.now())
            # Handle tz-aware vs tz-naive comparison
            if low_date.tzinfo is not None and now.tzinfo is None:
                now = now.tz_localize(low_date.tzinfo)
            elif low_date.tzinfo is None and now.tzinfo is not None:
                low_date = low_date.tz_localize(now.tzinfo)
            run_duration_months = max((now - low_date).days / 30.44, 0)
        else:
            run_duration_months = 0
        
        # Distance from 52w high/low
        if not hist_1y.empty:
            high_52w = hist_1y['Close'].max()
            low_52w = hist_1y['Close'].min()
            current = hist_1y['Close'].iloc[-1]
            return_from_low = ((current - low_52w) / low_52w) * 100 if low_52w > 0 else 0
            dist_from_high = ((current - high_52w) / high_52w) * 100 if high_52w > 0 else 0
        else:
            return_from_low = 0
            dist_from_high = 0
            current = 0
        
        # Revenue growth trajectory
        rev_trajectory = "UNKNOWN"
        if rev_health.revenue_acceleration is not None:
            if rev_health.revenue_acceleration > 3:
                rev_trajectory = "ACCELERATING"
            elif rev_health.revenue_acceleration > -3:
                rev_trajectory = "PEAK"
            else:
                rev_trajectory = "DECELERATING"
        
        margin_trajectory = rev_health.gross_margin_trend
        
        # Cycle phase estimation
        benchmarks = self.RUN_BENCHMARKS.get(cap_tier, self.RUN_BENCHMARKS["LARGE"])
        avg_run = benchmarks["avg_months"]
        max_run = benchmarks["max_typical"]
        
        pct_through = (run_duration_months / avg_run) * 100 if avg_run > 0 else 0
        
        if pct_through < 40 and rev_trajectory in ["ACCELERATING", "UNKNOWN"]:
            cycle_phase = "EARLY"
        elif pct_through < 70 and rev_trajectory != "DECELERATING":
            cycle_phase = "MID"
        elif pct_through < 100:
            cycle_phase = "LATE"
        else:
            cycle_phase = "EXTENDED"
        
        # Override with fundamental signals
        if rev_trajectory == "ACCELERATING" and rev_health.gross_margin_trend == "EXPANDING":
            if cycle_phase == "LATE":
                cycle_phase = "MID"  # fundamentals still strong
        elif rev_trajectory == "DECELERATING" and rev_health.gross_margin_trend == "CONTRACTING":
            if cycle_phase in ["EARLY", "MID"]:
                cycle_phase = "LATE"  # fundamentals weakening despite time
        
        # Signal
        signal_map = {"EARLY": "GREEN", "MID": "YELLOW", "LATE": "RED", "EXTENDED": "RED"}
        signal = signal_map.get(cycle_phase, "YELLOW")
        
        # Adjust: if mid-cycle but fundamentals strong, keep green
        if cycle_phase == "MID" and rev_trajectory == "ACCELERATING":
            signal = "GREEN"
        
        detail = (
            f"Run duration: {run_duration_months:.0f} months "
            f"(typical for {cap_tier}-cap: {avg_run} months). "
            f"{pct_through:.0f}% through typical run. "
            f"Cycle phase: {cycle_phase}. "
        )
        
        if rev_trajectory != "UNKNOWN":
            detail += f"Revenue {rev_trajectory.lower()}. "
        
        if dist_from_high > -3:
            detail += "Trading near 52-week highs."
        elif dist_from_high > -10:
            detail += f"Pulled back {abs(dist_from_high):.1f}% from highs."
        else:
            detail += f"Down {abs(dist_from_high):.1f}% from highs — correction territory."
        
        return CyclePosition(
            run_duration_months=round(run_duration_months, 1),
            return_from_52w_low=round(return_from_low, 1),
            distance_from_52w_high_pct=round(dist_from_high, 1),
            revenue_growth_trajectory=rev_trajectory,
            margin_trajectory=margin_trajectory,
            estimated_cycle_phase=cycle_phase,
            historical_avg_run_months=avg_run,
            pct_through_typical_run=round(pct_through, 0),
            signal=signal,
            detail=detail.strip(),
        )
    
    # ─────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────
    
    def _calc_annual_return(self, hist) -> float:
        if hist.empty or len(hist) < 2:
            return 0
        return ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
    
    def _calc_ytd_return(self, hist) -> float:
        if hist.empty:
            return 0
        current_year = datetime.now().year
        ytd = hist[hist.index.year == current_year]
        if ytd.empty or len(ytd) < 2:
            return 0
        return ((ytd['Close'].iloc[-1] - ytd['Close'].iloc[0]) / ytd['Close'].iloc[0]) * 100
    
    def _get_cap_tier(self, market_cap: float) -> str:
        if market_cap >= 200e9:
            return "MEGA"
        elif market_cap >= 50e9:
            return "LARGE"
        elif market_cap >= 10e9:
            return "MID"
        elif market_cap >= 2e9:
            return "SMALL"
        else:
            return "MICRO"
    
    def _score_to_grade(self, score: int) -> str:
        if score >= 90: return "A+"
        if score >= 85: return "A"
        if score >= 78: return "B+"
        if score >= 70: return "B"
        if score >= 62: return "C+"
        if score >= 55: return "C"
        if score >= 45: return "D"
        return "F"
    
    def _score_to_verdict(self, score: int) -> str:
        if score >= 80: return "STRONG RUN — Fundamentals Support Continuation"
        if score >= 65: return "HEALTHY — Monitor For Deceleration"
        if score >= 50: return "CAUTION — Mixed Signals, Tighten Stops"
        if score >= 35: return "LATE STAGE — Consider Taking Profits"
        return "OVEREXTENDED — High Risk of Mean Reversion"
    
    def _get_recommended_action(self, score, cycle_phase, multiple_signal) -> str:
        if score >= 80 and cycle_phase in ["EARLY", "MID"]:
            return "HOLD / ADD on pullbacks. Earnings-driven run with room to continue."
        elif score >= 65:
            return "HOLD position. Trail stops at 20-day MA. Trim 20% if P/E exceeds 2x 5yr avg."
        elif score >= 50:
            return "TRIM 25-30% to lock gains. Hold rest with tight trailing stop."
        elif score >= 35:
            return "TRIM 50%+. Late-cycle signals appearing. Protect capital."
        else:
            return "EXIT or maintain only small position. Mean reversion risk is high."
    
    def _identify_risks(self, mult, rev, analyst, smart, cycle) -> List[str]:
        risks = []
        if mult.signal == "RED":
            risks.append(f"Multiple expansion: {mult.multiple_expansion_pct or 'high'}% of gains from P/E stretch, not earnings")
        if rev.signal == "RED":
            risks.append(f"Revenue weakness: {rev.detail[:80]}")
        if rev.revenue_acceleration is not None and rev.revenue_acceleration < -5:
            risks.append(f"Revenue growth decelerating ({rev.revenue_acceleration}pp)")
        if rev.gross_margin_trend == "CONTRACTING":
            risks.append("Gross margins contracting — pricing power fading")
        if analyst.signal == "RED":
            risks.append(f"Stock has overshot analyst targets ({analyst.target_upside_pct}% downside to consensus)")
        if smart.insider_net_signal == "BEARISH":
            risks.append(f"Heavy insider selling ({smart.insider_sells_6mo} sells vs {smart.insider_buys_6mo} buys)")
        if smart.short_pct_float and smart.short_pct_float > 8:
            risks.append(f"Elevated short interest ({smart.short_pct_float}% of float)")
        if cycle.estimated_cycle_phase in ["LATE", "EXTENDED"]:
            risks.append(f"Run is {cycle.pct_through_typical_run:.0f}% through typical duration for {cycle.signal}-cap stocks")
        if cycle.distance_from_52w_high_pct > -2:
            risks.append("Trading at/near 52-week highs — limited upside buffer")
        return risks[:6]  # cap at 6
    
    def _identify_strengths(self, mult, rev, analyst, smart, cycle) -> List[str]:
        strengths = []
        if mult.signal == "GREEN":
            strengths.append("Run is earnings-driven — stock may actually be cheaper than it looks")
        if mult.peg_ratio and mult.peg_ratio < 1.0:
            strengths.append(f"PEG ratio {mult.peg_ratio} — growth outpacing valuation")
        if rev.revenue_growth_yoy and rev.revenue_growth_yoy > 20:
            strengths.append(f"Strong revenue growth at {rev.revenue_growth_yoy}% YoY")
        if rev.revenue_acceleration and rev.revenue_acceleration > 0:
            strengths.append("Revenue growth is accelerating")
        if rev.gross_margin_trend == "EXPANDING":
            strengths.append("Gross margins expanding — pricing power or efficiency improving")
        if analyst.target_upside_pct and analyst.target_upside_pct > 10:
            strengths.append(f"Analysts still see {analyst.target_upside_pct}% upside")
        if smart.insider_net_signal == "BULLISH":
            strengths.append("Insiders are buying their own stock")
        if smart.short_pct_float and smart.short_pct_float < 3:
            strengths.append("Low short interest — bears aren't fighting this")
        if cycle.estimated_cycle_phase == "EARLY":
            strengths.append(f"Only {cycle.pct_through_typical_run:.0f}% through typical run — early innings")
        if rev.quarters_of_growth >= 4:
            strengths.append(f"{rev.quarters_of_growth} consecutive quarters of revenue growth")
        return strengths[:6]
    
    def _to_dict(self, obj) -> Dict:
        """Convert dataclass tree to dict, sanitizing NaN/Inf values"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                result[field_name] = self._to_dict(value)
            return result
        elif isinstance(obj, list):
            return [self._to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = RunSustainabilityAnalyzer()
    result = analyzer.analyze("NVDA")
    print(json.dumps(result, indent=2, default=str))
