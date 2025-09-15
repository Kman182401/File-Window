"""
Market Data Validation and Sanitization Module

Provides comprehensive validation for all market data inputs to prevent
invalid data from corrupting the trading system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketDataValidator:
    """Validates and sanitizes market data inputs"""
    
    def __init__(self):
        self.validation_stats = {
            'total_rows_processed': 0,
            'rows_cleaned': 0,
            'rows_rejected': 0,
            'validation_errors': []
        }
    
    def validate_ohlcv(self, df: pd.DataFrame, 
                      symbol: str = "Unknown") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate OHLCV data frame
        
        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        if df is None or df.empty:
            raise ValueError(f"Empty dataframe received for {symbol}")
        
        original_len = len(df)
        report = {'symbol': symbol, 'issues': []}
        
        # Required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 1. Remove rows with any negative prices
        negative_mask = (df['open'] < 0) | (df['high'] < 0) | \
                       (df['low'] < 0) | (df['close'] < 0)
        if negative_mask.any():
            report['issues'].append(f"Removed {negative_mask.sum()} rows with negative prices")
            df = df[~negative_mask]
        
        # 2. Remove rows with NaN in price columns
        price_cols = ['open', 'high', 'low', 'close']
        nan_mask = df[price_cols].isnull().any(axis=1)
        if nan_mask.any():
            report['issues'].append(f"Removed {nan_mask.sum()} rows with NaN prices")
            df = df[~nan_mask]
        
        # 3. Fix negative/NaN volumes
        if (df['volume'] < 0).any():
            report['issues'].append(f"Fixed {(df['volume'] < 0).sum()} negative volumes")
            df['volume'] = df['volume'].clip(lower=0)
        
        if df['volume'].isnull().any():
            report['issues'].append(f"Fixed {df['volume'].isnull().sum()} NaN volumes")
            df['volume'] = df['volume'].fillna(0)
        
        # 4. Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            report['issues'].append(f"Removed {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            df = df[~invalid_ohlc]
        
        # 5. Remove extreme outliers (>10 std from mean)
        for col in price_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            outlier_mask = np.abs(df[col] - mean_val) > (10 * std_val)
            if outlier_mask.any():
                report['issues'].append(f"Removed {outlier_mask.sum()} extreme outliers in {col}")
                df = df[~outlier_mask]
        
        # 6. Check for duplicate timestamps if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            duplicates = df.index.duplicated()
            if duplicates.any():
                report['issues'].append(f"Removed {duplicates.sum()} duplicate timestamps")
                df = df[~duplicates]
        
        # Update stats
        rows_removed = original_len - len(df)
        self.validation_stats['total_rows_processed'] += original_len
        self.validation_stats['rows_cleaned'] += len(df)
        self.validation_stats['rows_rejected'] += rows_removed
        
        report['original_rows'] = original_len
        report['clean_rows'] = len(df)
        report['removed_rows'] = rows_removed
        report['removal_rate'] = rows_removed / original_len if original_len > 0 else 0
        
        if rows_removed > 0:
            logger.warning(f"Data validation for {symbol}: removed {rows_removed}/{original_len} rows")
        
        return df, report
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Check for valid characters
        if not symbol.replace('!', '').replace('_', '').isalnum():
            return False
        
        # Check reasonable length
        if len(symbol) < 1 or len(symbol) > 10:
            return False
        
        return True
    
    def validate_order_params(self, 
                             symbol: str,
                             quantity: float,
                             order_type: str,
                             price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate order parameters
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate symbol
        if not self.validate_symbol(symbol):
            return False, f"Invalid symbol: {symbol}"
        
        # Validate quantity
        if not isinstance(quantity, (int, float)):
            return False, f"Invalid quantity type: {type(quantity)}"
        
        if quantity == 0:
            return False, "Quantity cannot be zero"
        
        if abs(quantity) > 1000:  # Sanity check
            return False, f"Quantity too large: {quantity}"
        
        # Validate order type
        valid_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
        if order_type.upper() not in valid_types:
            return False, f"Invalid order type: {order_type}"
        
        # Validate price for limit orders
        if order_type.upper() in ['LIMIT', 'STOP_LIMIT']:
            if price is None:
                return False, f"Price required for {order_type} order"
            
            if not isinstance(price, (int, float)):
                return False, f"Invalid price type: {type(price)}"
            
            if price <= 0:
                return False, f"Price must be positive: {price}"
        
        return True, "Valid"
    
    def validate_feature_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean feature dictionary
        """
        cleaned = {}
        
        for key, value in features.items():
            # Skip None values
            if value is None:
                continue
            
            # Convert to numeric if possible
            if isinstance(value, (int, float)):
                # Check for inf/nan
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = float(value)
            elif isinstance(value, str):
                try:
                    cleaned[key] = float(value)
                except ValueError:
                    # Keep as string if can't convert
                    cleaned[key] = value
            else:
                # Convert other types to string
                cleaned[key] = str(value)
        
        return cleaned
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""
        return {
            'total_rows_processed': self.validation_stats['total_rows_processed'],
            'rows_cleaned': self.validation_stats['rows_cleaned'],
            'rows_rejected': self.validation_stats['rows_rejected'],
            'rejection_rate': (
                self.validation_stats['rows_rejected'] / 
                self.validation_stats['total_rows_processed']
                if self.validation_stats['total_rows_processed'] > 0 else 0
            ),
            'error_count': len(self.validation_stats['validation_errors'])
        }


def validate_market_data_batch(data_batch: List[pd.DataFrame], 
                              symbols: List[str]) -> List[pd.DataFrame]:
    """
    Validate a batch of market data frames
    
    Args:
        data_batch: List of dataframes to validate
        symbols: List of corresponding symbols
    
    Returns:
        List of validated dataframes
    """
    validator = MarketDataValidator()
    validated_batch = []
    
    for df, symbol in zip(data_batch, symbols):
        try:
            clean_df, report = validator.validate_ohlcv(df, symbol)
            validated_batch.append(clean_df)
            
            if report['issues']:
                logger.info(f"Validation report for {symbol}: {report}")
        
        except Exception as e:
            logger.error(f"Failed to validate data for {symbol}: {e}")
            # Return empty dataframe on validation failure
            validated_batch.append(pd.DataFrame())
    
    # Log summary
    summary = validator.get_validation_summary()
    logger.info(f"Batch validation complete: {summary}")
    
    return validated_batch