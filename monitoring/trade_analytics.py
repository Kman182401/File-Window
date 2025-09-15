"""
Trading Performance Analytics Module

Comprehensive analytics for trading performance including PnL tracking,
risk metrics, trade analysis, and performance attribution.

Author: AI Trading System
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Container for trade information"""
    trade_id: str
    ticker: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: int = 0
    side: str = ""  # 'long' or 'short'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: str = "open"  # 'open', 'closed', 'cancelled'
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingMetrics:
    """Comprehensive trading performance metrics"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # PnL metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    value_at_risk: float = 0.0
    conditional_value_at_risk: float = 0.0
    
    # Trade statistics
    avg_trade_duration: float = 0.0
    avg_winning_duration: float = 0.0
    avg_losing_duration: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    
    # Efficiency metrics
    total_commission: float = 0.0
    total_slippage: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    
    # Time-based metrics
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    monthly_pnl: Dict[str, float] = field(default_factory=dict)
    hourly_distribution: Dict[int, float] = field(default_factory=dict)


class TradeAnalytics:
    """
    Advanced trade analytics and performance tracking
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize trade analytics
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [0.0]
        self.timestamps: List[datetime] = [datetime.now()]
        self.metrics = TradingMetrics()
        
        # Performance by strategy
        self.strategy_performance: Dict[str, TradingMetrics] = defaultdict(TradingMetrics)
        
        # Performance by ticker
        self.ticker_performance: Dict[str, TradingMetrics] = defaultdict(TradingMetrics)
        
        logger.info("TradeAnalytics initialized")
    
    def add_trade(self, trade: Trade):
        """Add a completed trade for analysis"""
        self.trades.append(trade)
        
        if trade.status == "closed":
            self._update_metrics(trade)
            self._update_equity_curve(trade)
    
    def close_trade(self, trade_id: str, exit_time: datetime, 
                   exit_price: float) -> Optional[Trade]:
        """
        Close an open trade
        
        Args:
            trade_id: ID of the trade to close
            exit_time: Exit timestamp
            exit_price: Exit price
            
        Returns:
            Updated trade or None if not found
        """
        for trade in self.trades:
            if trade.trade_id == trade_id and trade.status == "open":
                trade.exit_time = exit_time
                trade.exit_price = exit_price
                trade.status = "closed"
                
                # Calculate PnL
                if trade.side == "long":
                    trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                else:  # short
                    trade.pnl = (trade.entry_price - exit_price) * trade.quantity
                
                trade.pnl -= (trade.commission + trade.slippage)
                trade.pnl_percent = (trade.pnl / (trade.entry_price * abs(trade.quantity))) * 100
                
                self._update_metrics(trade)
                self._update_equity_curve(trade)
                
                return trade
        
        return None
    
    def _update_metrics(self, trade: Trade):
        """Update metrics with a new trade"""
        if trade.status != "closed":
            return
        
        # Update counts
        self.metrics.total_trades += 1
        
        if trade.pnl > 0:
            self.metrics.winning_trades += 1
            self.metrics.gross_profit += trade.pnl
        else:
            self.metrics.losing_trades += 1
            self.metrics.gross_loss += abs(trade.pnl)
        
        # Update PnL
        self.metrics.total_pnl += trade.pnl
        self.metrics.total_commission += trade.commission
        self.metrics.total_slippage += trade.slippage
        
        # Update win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        # Update averages
        if self.metrics.winning_trades > 0:
            self.metrics.avg_win = self.metrics.gross_profit / self.metrics.winning_trades
        
        if self.metrics.losing_trades > 0:
            self.metrics.avg_loss = self.metrics.gross_loss / self.metrics.losing_trades
        
        # Update profit factor
        if self.metrics.gross_loss > 0:
            self.metrics.profit_factor = self.metrics.gross_profit / self.metrics.gross_loss
        
        # Update largest win/loss
        if trade.pnl > self.metrics.largest_win:
            self.metrics.largest_win = trade.pnl
        if trade.pnl < self.metrics.largest_loss:
            self.metrics.largest_loss = trade.pnl
        
        # Update consecutive wins/losses
        if len(self.trades) > 1:
            prev_trade = self.trades[-2]
            if prev_trade.status == "closed":
                if trade.pnl > 0 and prev_trade.pnl > 0:
                    self.metrics.consecutive_wins += 1
                    self.metrics.consecutive_losses = 0
                elif trade.pnl < 0 and prev_trade.pnl < 0:
                    self.metrics.consecutive_losses += 1
                    self.metrics.consecutive_wins = 0
                else:
                    self.metrics.consecutive_wins = 1 if trade.pnl > 0 else 0
                    self.metrics.consecutive_losses = 1 if trade.pnl < 0 else 0
        
        # Update time-based metrics
        if trade.exit_time:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
            
            # Update average durations
            all_durations = [
                (t.exit_time - t.entry_time).total_seconds() / 3600
                for t in self.trades
                if t.status == "closed" and t.exit_time
            ]
            
            if all_durations:
                self.metrics.avg_trade_duration = np.mean(all_durations)
            
            # Update winning/losing durations
            if trade.pnl > 0:
                winning_durations = [
                    (t.exit_time - t.entry_time).total_seconds() / 3600
                    for t in self.trades
                    if t.status == "closed" and t.exit_time and t.pnl > 0
                ]
                if winning_durations:
                    self.metrics.avg_winning_duration = np.mean(winning_durations)
            else:
                losing_durations = [
                    (t.exit_time - t.entry_time).total_seconds() / 3600
                    for t in self.trades
                    if t.status == "closed" and t.exit_time and t.pnl <= 0
                ]
                if losing_durations:
                    self.metrics.avg_losing_duration = np.mean(losing_durations)
            
            # Update daily PnL
            date_key = trade.exit_time.strftime("%Y-%m-%d")
            self.metrics.daily_pnl[date_key] = self.metrics.daily_pnl.get(date_key, 0) + trade.pnl
            
            # Update monthly PnL
            month_key = trade.exit_time.strftime("%Y-%m")
            self.metrics.monthly_pnl[month_key] = self.metrics.monthly_pnl.get(month_key, 0) + trade.pnl
            
            # Update hourly distribution
            hour = trade.exit_time.hour
            self.metrics.hourly_distribution[hour] = self.metrics.hourly_distribution.get(hour, 0) + trade.pnl
        
        # Update expectancy
        if self.metrics.total_trades > 0:
            self.metrics.expectancy = (
                self.metrics.win_rate * self.metrics.avg_win - 
                (1 - self.metrics.win_rate) * self.metrics.avg_loss
            )
        
        # Update Kelly Criterion
        if self.metrics.avg_loss > 0:
            win_loss_ratio = self.metrics.avg_win / self.metrics.avg_loss
            self.metrics.kelly_criterion = (
                self.metrics.win_rate - (1 - self.metrics.win_rate) / win_loss_ratio
            )
        
        # Update strategy-specific metrics
        if trade.strategy:
            self._update_strategy_metrics(trade.strategy, trade)
        
        # Update ticker-specific metrics
        self._update_ticker_metrics(trade.ticker, trade)
    
    def _update_strategy_metrics(self, strategy: str, trade: Trade):
        """Update metrics for a specific strategy"""
        metrics = self.strategy_performance[strategy]
        
        metrics.total_trades += 1
        metrics.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            metrics.winning_trades += 1
            metrics.gross_profit += trade.pnl
        else:
            metrics.losing_trades += 1
            metrics.gross_loss += abs(trade.pnl)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
    
    def _update_ticker_metrics(self, ticker: str, trade: Trade):
        """Update metrics for a specific ticker"""
        metrics = self.ticker_performance[ticker]
        
        metrics.total_trades += 1
        metrics.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            metrics.winning_trades += 1
            metrics.gross_profit += trade.pnl
        else:
            metrics.losing_trades += 1
            metrics.gross_loss += abs(trade.pnl)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
    
    def _update_equity_curve(self, trade: Trade):
        """Update equity curve with new trade"""
        if trade.status == "closed" and trade.exit_time:
            new_equity = self.equity_curve[-1] + trade.pnl
            self.equity_curve.append(new_equity)
            self.timestamps.append(trade.exit_time)
            
            # Update drawdown
            self._calculate_drawdown()
            
            # Update risk metrics
            self._calculate_risk_metrics()
    
    def _calculate_drawdown(self):
        """Calculate maximum drawdown and duration"""
        if len(self.equity_curve) < 2:
            return
        
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        
        self.metrics.max_drawdown = np.min(drawdown)
        
        # Calculate max drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        if drawdown_periods:
            self.metrics.max_drawdown_duration = max(drawdown_periods)
    
    def _calculate_risk_metrics(self):
        """Calculate risk-adjusted performance metrics"""
        if len(self.equity_curve) < 2:
            return
        
        returns = np.diff(self.equity_curve)
        
        if len(returns) == 0:
            return
        
        # Sharpe Ratio
        if np.std(returns) > 0:
            daily_rf = self.risk_free_rate / 252
            self.metrics.sharpe_ratio = (
                (np.mean(returns) - daily_rf) / np.std(returns) * np.sqrt(252)
            )
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            self.metrics.sortino_ratio = (
                np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
            )
        
        # Calmar Ratio
        if self.metrics.max_drawdown < 0:
            annual_return = self.metrics.total_pnl / len(self.metrics.daily_pnl) * 252
            self.metrics.calmar_ratio = annual_return / abs(self.metrics.max_drawdown)
        
        # Value at Risk (95% confidence)
        if len(returns) > 20:
            self.metrics.value_at_risk = np.percentile(returns, 5)
            
            # Conditional Value at Risk (Expected Shortfall)
            var_threshold = self.metrics.value_at_risk
            tail_losses = returns[returns <= var_threshold]
            if len(tail_losses) > 0:
                self.metrics.conditional_value_at_risk = np.mean(tail_losses)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'overview': {
                'total_trades': self.metrics.total_trades,
                'win_rate': f"{self.metrics.win_rate * 100:.1f}%",
                'total_pnl': self.metrics.total_pnl,
                'profit_factor': self.metrics.profit_factor,
                'expectancy': self.metrics.expectancy
            },
            'risk_metrics': {
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'sortino_ratio': self.metrics.sortino_ratio,
                'calmar_ratio': self.metrics.calmar_ratio,
                'max_drawdown': f"{self.metrics.max_drawdown * 100:.1f}%",
                'value_at_risk': self.metrics.value_at_risk
            },
            'trade_statistics': {
                'avg_win': self.metrics.avg_win,
                'avg_loss': self.metrics.avg_loss,
                'largest_win': self.metrics.largest_win,
                'largest_loss': self.metrics.largest_loss,
                'avg_duration_hours': self.metrics.avg_trade_duration
            },
            'efficiency': {
                'total_commission': self.metrics.total_commission,
                'total_slippage': self.metrics.total_slippage,
                'kelly_criterion': f"{self.metrics.kelly_criterion * 100:.1f}%"
            }
        }
    
    def get_strategy_comparison(self) -> pd.DataFrame:
        """Compare performance across strategies"""
        data = []
        for strategy, metrics in self.strategy_performance.items():
            data.append({
                'Strategy': strategy,
                'Trades': metrics.total_trades,
                'Win Rate': metrics.win_rate,
                'Total PnL': metrics.total_pnl,
                'Avg Win': metrics.avg_win,
                'Avg Loss': metrics.avg_loss
            })
        
        return pd.DataFrame(data).sort_values('Total PnL', ascending=False)
    
    def get_ticker_performance(self) -> pd.DataFrame:
        """Get performance breakdown by ticker"""
        data = []
        for ticker, metrics in self.ticker_performance.items():
            data.append({
                'Ticker': ticker,
                'Trades': metrics.total_trades,
                'Win Rate': metrics.win_rate,
                'Total PnL': metrics.total_pnl,
                'Profit Factor': metrics.profit_factor
            })
        
        return pd.DataFrame(data).sort_values('Total PnL', ascending=False)
    
    def export_trades(self, filepath: str):
        """Export trades to CSV"""
        if not self.trades:
            return
        
        df = pd.DataFrame([
            {
                'trade_id': t.trade_id,
                'ticker': t.ticker,
                'side': t.side,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'status': t.status,
                'strategy': t.strategy
            }
            for t in self.trades
        ])
        
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.trades)} trades to {filepath}")


if __name__ == "__main__":
    # Self-test
    print("Trade Analytics Self-Test")
    print("=" * 50)
    
    analytics = TradeAnalytics()
    
    # Add sample trades
    print("\nAdding sample trades...")
    
    trades = [
        Trade("T001", "ES1!", datetime.now() - timedelta(hours=5), 
              entry_price=4500, quantity=1, side="long", strategy="momentum"),
        Trade("T002", "NQ1!", datetime.now() - timedelta(hours=4),
              entry_price=15000, quantity=1, side="short", strategy="mean_reversion"),
        Trade("T003", "ES1!", datetime.now() - timedelta(hours=3),
              entry_price=4510, quantity=2, side="long", strategy="momentum")
    ]
    
    # Close trades with results
    for i, trade in enumerate(trades):
        analytics.add_trade(trade)
        
        # Simulate closing trades
        exit_price = trade.entry_price * (1.01 if i % 2 == 0 else 0.995)
        analytics.close_trade(
            trade.trade_id,
            datetime.now() - timedelta(hours=2-i),
            exit_price
        )
    
    # Get performance summary
    print("\nPerformance Summary:")
    summary = analytics.get_performance_summary()
    for category, metrics in summary.items():
        print(f"\n{category}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    print("\nSelf-test complete!")