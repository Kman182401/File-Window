"""
Post-Cost Performance Metrics
CAGR/MaxDD ratio and other risk-adjusted metrics with transaction costs
"""
import numpy as np
import pandas as pd

def equity_curve(returns):
    """Build equity curve from returns."""
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return np.array(eq)

def cagr(equity, bars_per_year):
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        equity: Equity curve array
        bars_per_year: Number of trading bars per year (e.g., 78*252 for 5-min)
    """
    if len(equity) < 2:
        return 0.0
    
    total_return = equity[-1] / equity[0]
    years = len(equity) / bars_per_year
    
    if years <= 0:
        return 0.0
    
    return (total_return ** (1/years) - 1)

def max_drawdown(equity):
    """Calculate maximum drawdown (returns negative number)."""
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak == 0, 1, peak) - 1.0
    return dd.min()

def sharpe_ratio(returns, ann_factor=252):
    """Calculate Sharpe ratio with annualization."""
    r = np.asarray(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(ann_factor)

def calmar_ratio(equity, bars_per_year):
    """Calculate Calmar ratio (CAGR/MaxDD)."""
    c = cagr(equity, bars_per_year)
    dd = max_drawdown(equity)
    
    if dd >= 0:  # No drawdown
        return c if c > 0 else 0.0
    
    return c / abs(dd)

def calculate_post_cost_metrics(raw_returns, trade_flags, cost_per_trade=0.002, bars_per_year=78*252):
    """
    Calculate all metrics after transaction costs.
    
    Args:
        raw_returns: Array of raw returns
        trade_flags: Binary array indicating trades (1=trade, 0=no trade)
        cost_per_trade: Round-trip cost as fraction (0.002 = 0.2%)
        bars_per_year: Annual bar count for CAGR calculation
    
    Returns:
        Dict with all metrics
    """
    # Apply costs
    post_cost_returns = raw_returns - (trade_flags * cost_per_trade)
    
    # Build equity curve
    eq = equity_curve(post_cost_returns)
    
    # Calculate metrics
    metrics = {
        'sharpe': sharpe_ratio(post_cost_returns, ann_factor=np.sqrt(bars_per_year)),
        'cagr': cagr(eq, bars_per_year),
        'max_dd': max_drawdown(eq),
        'calmar': calmar_ratio(eq, bars_per_year),
        'total_trades': int(trade_flags.sum()),
        'turnover': trade_flags.mean(),
        'final_equity': eq[-1],
        'total_return': (eq[-1] / eq[0] - 1) if len(eq) > 1 else 0.0
    }
    
    return metrics

def mcc(tp, tn, fp, fn):
    """Matthews Correlation Coefficient for classification quality."""
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else num / den