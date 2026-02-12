"""
Options Greeks Calculator using Black-Scholes Model
===================================================
Calculates option Greeks for single and multi-leg strategies.
Integrates with existing trade rule engine.

Author: Strategic Edge Flow
Version: 1.0.0
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timedelta


@dataclass
class OptionGreeks:
    """Greeks for a single option"""
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float   # Per 1% IV change
    rho: float
    
    # Additional metrics
    implied_vol: float
    time_value: float
    intrinsic_value: float
    
    
@dataclass
class OptionLeg:
    """Single leg of an options strategy"""
    option_type: str  # 'call' or 'put'
    strike: float
    expiration_days: int
    premium: float
    quantity: int  # Positive for long, negative for short
    implied_vol: float


@dataclass
class StrategyAnalysis:
    """Complete strategy with risk metrics"""
    strategy_name: str
    legs: List[OptionLeg]
    
    # Greeks (portfolio level)
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    
    # Risk metrics
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    
    # Cost
    net_debit: float  # Positive = you pay
    net_credit: float  # Positive = you receive
    
    # Warnings
    warnings: List[str]


class GreeksCalculator:
    """Black-Scholes Greeks calculator"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        
    def calculate_greeks(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,  # In years
        volatility: float,       # Annual IV as decimal (e.g., 0.50 for 50%)
        option_type: str = 'call',
        dividend_yield: float = 0.0
    ) -> OptionGreeks:
        """Calculate all Greeks for a single option"""
        
        # Time to expiry in years
        T = time_to_expiry
        if T <= 0:
            T = 0.0001  # Avoid division by zero
            
        S = spot_price
        K = strike
        r = self.risk_free_rate
        q = dividend_yield
        sigma = volatility
        
        # d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            intrinsic = max(S - K, 0)
        else:  # put
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            intrinsic = max(K - S, 0)
            
        # Gamma (same for calls and puts)
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        
        # Vega (per 1% change in IV) - same for calls and puts
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta (per day)
        if option_type.lower() == 'call':
            theta = (
                -(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)
            ) / 365
        else:
            theta = (
                -(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)
            ) / 365
            
        # Rho (per 1% change in interest rate)
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        # Calculate option price for time value
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            
        time_value = price - intrinsic
        
        return OptionGreeks(
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 2),
            vega=round(vega, 2),
            rho=round(rho, 2),
            implied_vol=volatility,
            time_value=round(time_value, 2),
            intrinsic_value=round(intrinsic, 2)
        )
        
    def calculate_expected_move(
        self,
        spot_price: float,
        implied_vol: float,
        days_to_expiry: int
    ) -> Dict[str, float]:
        """Calculate expected move based on IV"""
        
        time_fraction = days_to_expiry / 365
        
        # 1 standard deviation move
        one_sd = spot_price * implied_vol * np.sqrt(time_fraction)
        
        return {
            "one_sd_dollars": round(one_sd, 2),
            "one_sd_percent": round((one_sd / spot_price) * 100, 2),
            "upper_range": round(spot_price + one_sd, 2),
            "lower_range": round(spot_price - one_sd, 2),
            "two_sd_upper": round(spot_price + (2 * one_sd), 2),
            "two_sd_lower": round(spot_price - (2 * one_sd), 2)
        }


class OptionsStrategyAnalyzer:
    """Analyze multi-leg options strategies"""
    
    def __init__(self):
        self.greeks_calc = GreeksCalculator()
        
    def analyze_hedged_long(
        self,
        spot_price: float,
        call_strike: float,
        call_dte: int,
        call_premium: float,
        call_iv: float,
        put_strike: float,
        put_dte: int,
        put_premium: float,
        put_iv: float
    ) -> StrategyAnalysis:
        """Analyze the hedged long (call + protective put) strategy"""
        
        # Calculate greeks for each leg
        call_greeks = self.greeks_calc.calculate_greeks(
            spot_price, call_strike, call_dte / 365, call_iv, 'call'
        )
        put_greeks = self.greeks_calc.calculate_greeks(
            spot_price, put_strike, put_dte / 365, put_iv, 'put'
        )
        
        # Portfolio Greeks
        total_delta = call_greeks.delta + put_greeks.delta
        total_gamma = call_greeks.gamma + put_greeks.gamma
        total_theta = call_greeks.theta + put_greeks.theta
        total_vega = call_greeks.vega + put_greeks.vega
        
        # Cost
        net_debit = call_premium + put_premium
        
        # Max profit/loss
        max_loss = net_debit  # Both expire worthless
        max_profit = float('inf')  # Unlimited upside on call
        
        # Breakeven
        breakeven_up = call_strike + net_debit
        # Lower breakeven harder to calculate with different DTEs
        
        # Warnings
        warnings = []
        
        # Check IV levels
        if call_iv > 0.80 or put_iv > 0.80:
            warnings.append(f"⚠️ HIGH IV: {max(call_iv, put_iv):.1%} - IV crush risk is severe")
            
        # Check DTE asymmetry
        dte_diff = abs(call_dte - put_dte)
        if dte_diff > 15:
            warnings.append(f"⚠️ DTE GAP: {dte_diff} days between expirations - complex management")
            
        # Check theta burn
        daily_theta = total_theta
        if daily_theta < -30:
            days_to_zero = net_debit / abs(daily_theta)
            warnings.append(f"⚠️ THETA BURN: ${abs(daily_theta):.0f}/day - position worthless in {days_to_zero:.0f} days if flat")
            
        # Check hedge effectiveness
        strike_gap_pct = abs(call_strike - put_strike) / spot_price * 100
        if strike_gap_pct < 5:
            warnings.append(f"⚠️ NARROW HEDGE: {strike_gap_pct:.1f}% gap - limited protection")
        elif strike_gap_pct > 15:
            warnings.append(f"⚠️ WIDE HEDGE: {strike_gap_pct:.1f}% gap - expensive protection")
            
        # Check delta balance
        if abs(total_delta) < 0.20:
            warnings.append(f"⚠️ LOW DELTA: {total_delta:.2f} - position is too neutral for directional bet")
            
        return StrategyAnalysis(
            strategy_name="Hedged Long (Call + Put)",
            legs=[
                OptionLeg('call', call_strike, call_dte, call_premium, 1, call_iv),
                OptionLeg('put', put_strike, put_dte, put_premium, 1, put_iv)
            ],
            total_delta=round(total_delta, 3),
            total_gamma=round(total_gamma, 5),
            total_theta=round(total_theta, 2),
            total_vega=round(total_vega, 2),
            max_profit=max_profit,
            max_loss=round(max_loss, 2),
            breakeven_points=[round(breakeven_up, 2)],
            probability_of_profit=round((0.5 + total_delta) * 100, 1),  # Rough estimate
            net_debit=round(net_debit, 2),
            net_credit=0,
            warnings=warnings
        )


# Example usage
if __name__ == "__main__":
    analyzer = OptionsStrategyAnalyzer()
    
    # Analyze the hedged long example from the image
    result = analyzer.analyze_hedged_long(
        spot_price=390,
        call_strike=385,
        call_dte=21,
        call_premium=385,
        call_iv=1.146,  # 114.6%
        put_strike=370,
        put_dte=10,
        put_premium=370,
        put_iv=1.146
    )
    
    print(f"Strategy: {result.strategy_name}")
    print(f"Net Cost: ${result.net_debit}")
    print(f"Max Loss: ${result.max_loss}")
    print(f"\nPortfolio Greeks:")
    print(f"  Delta: {result.total_delta} (directional exposure)")
    print(f"  Theta: ${result.total_theta}/day (time decay)")
    print(f"  Vega: ${result.total_vega} per 1% IV change")
    print(f"\nBreakeven: ${result.breakeven_points[0]}")
    print(f"Probability of Profit: {result.probability_of_profit}%")
    print(f"\nWarnings:")
    for w in result.warnings:
        print(f"  {w}")
