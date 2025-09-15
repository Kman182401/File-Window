"""
Triple-Barrier Labeling with Sample Weights
Event-driven labels for ML trading with anti-overfitting controls
"""
import numpy as np
import pandas as pd

def triple_barrier_labels(close: pd.Series, pt: float = 0.008, sl: float = 0.006, max_h: int = 20):
    """
    Vectorized triple-barrier labeling with sample weights.
    
    Args:
        close: Price series (pandas Series with datetime index)
        pt: Profit target as fraction (0.008 = 0.8%)
        sl: Stop loss as fraction (0.006 = 0.6%)
        max_h: Max holding period in bars (default: 20)
    
    Returns:
        (labels, t_end, sample_weights) as pandas Series
    """
    
    n = len(close)
    v = close.values
    up = (1 + pt) * v  # Upper barrier
    dn = (1 - sl) * v  # Lower barrier
    
    y = np.zeros(n, dtype=np.int8)
    t_end = np.arange(n)
    
    # Vectorized loop for efficiency on small CPU
    for i in range(n - 1):
        end = min(i + 1 + max_h, n)
        seg = v[i + 1:end]
        
        if seg.size == 0:
            continue
            
        # Find first touch of barriers
        u = np.where(seg >= up[i])[0]
        d = np.where(seg <= dn[i])[0]
        
        if u.size and (not d.size or u[0] < d[0]):
            y[i] = 1  # Long wins
            t_end[i] = i + 1 + u[0]
        elif d.size:
            y[i] = -1  # Short wins
            t_end[i] = i + 1 + d[0]
        else:
            y[i] = 0  # Neutral (max_h reached)
            t_end[i] = end - 1
    
    # Sample weights proportional to realized move
    ret = (v[t_end] - v) / v
    w = np.abs(ret)
    w[np.isnan(w)] = 0.0
    
    idx = close.index
    return (
        pd.Series(y, index=idx, name="label"),
        pd.Series(t_end, index=idx, name="t_end"),
        pd.Series(w, index=idx, name="sample_weight")
    )