"""
Centralized Trading Configuration
Shared parameters for anti-overfitting modules
"""

# Triple-Barrier Parameters
TB_MAX_H = 20  # Max holding period in bars
TB_PT = 0.008  # Profit target (0.8%)
TB_SL = 0.006  # Stop loss (0.6%)

# Cross-Validation Parameters
N_FOLDS = 3  # Number of CV folds (keep low for CPU)
EMBARGO = TB_MAX_H  # Embargo size matches TB horizon

# Position Sizing Parameters
VOL_FLOOR = 1e-3  # Minimum volatility floor (prevents flip-flop)
MAX_POSITION_SIZE = 1  # Max contracts per trade

# Cost Parameters
ROUND_TRIP_COST = 0.002  # 0.2% round-trip cost

# Market Data Parameters
BARS_PER_YEAR = 78 * 252  # 5-minute bars
LEAN_TICKERS = ["ES1!", "NQ1!"]  # Start with minimal symbols