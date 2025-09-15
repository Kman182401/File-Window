"""
Purged Walk-Forward Cross-Validation with Embargo
Prevents leakage in time series ML with configurable embargo
"""
import numpy as np

def purged_walk_forward_indices(n: int, n_folds: int = 3, embargo: int = 20):
    """
    Generate train/test indices for purged walk-forward validation.
    
    Args:
        n: Total number of samples
        n_folds: Number of folds (default: 3 for CPU efficiency)
        embargo: Gap between train and test (default: 20, should match TB horizon)
    
    Yields:
        (train_indices, test_indices) as numpy arrays
    """
    
    fold_size = n // n_folds
    
    for k in range(n_folds):
        # Test window
        te_start = k * fold_size
        te_end = n if k == n_folds - 1 else (k + 1) * fold_size
        test_idx = np.arange(te_start, te_end)
        
        # Train windows with embargo gap
        train_before = np.arange(0, max(0, te_start - embargo))
        train_after = np.arange(min(n, te_end + embargo), n)
        train_idx = np.concatenate([train_before, train_after])
        
        # Skip if train set too small
        if len(train_idx) < 10:
            continue
            
        yield train_idx, test_idx