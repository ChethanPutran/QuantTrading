"""
Trading simulator for realistic execution simulation.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional


class BaseExecutionEngine(ABC):
    """Executes trades in simulation or live."""

    @abstractmethod
    def step(self, action: int) -> dict:
        pass


class TradingSimulator(BaseExecutionEngine):
    """
    Simulate realistic trading with costs and constraints.
    
    Model:
    - Position changes based on actions
    - Transaction costs (spread, fees)
    - PnL computation
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        initial_price: float = 100.0,
        max_position: float = 10.0,
        transaction_cost: float = 0.001,  # 0.1% spread
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize trading simulator.
        
        Args:
            initial_balance: Starting cash balance
            initial_price: Initial stock price
            max_position: Maximum allowed position size
            transaction_cost: Transaction cost as fraction of position value
            slippage: Slippage as fraction of position value
        """
        self.initial_balance = initial_balance
        self.initial_price = initial_price
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Current state
        self.balance = initial_balance
        self.position = 0.0  # Number of shares held
        self.entry_price = 0.0
        self.current_price = initial_price
        
        # History
        self.price_history = [initial_price]
        self.position_history = [0.0]
        self.balance_history = [initial_balance]
        self.pnl_history = [0.0]
        self.timestamp = 0

    def step(
        self,
        action: int,
        current_price: float = None,
        order_size: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute one step of trading.
        
        Args:
            action: -1 (sell), 0 (hold), 1 (buy)
            current_price: Current market price (if None, use last price)
            order_size: Fraction of position to trade
        
        Returns:
            Dictionary with execution details
        """
        if current_price is None:
            current_price = self.current_price
        
        self.current_price = current_price
        self.timestamp += 1
        
        # Apply slippage
        execution_price = current_price * (1 + self.slippage * np.sign(action))
        
        # Execute trade
        if action == 1:  # Buy
            # How many shares can we afford?
            num_shares = (self.balance / execution_price) * order_size
            
            if num_shares > 0:
                cost = num_shares * execution_price
                trans_fee = cost * self.transaction_cost
                total_cost = cost + trans_fee
                
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.position += num_shares
                    self.entry_price = execution_price
                    cost_incurred = trans_fee
                else:
                    # Not enough balance, buy what we can
                    available = self.balance / (execution_price * (1 + self.transaction_cost))
                    if available > 0:
                        cost = available * execution_price
                        trans_fee = cost * self.transaction_cost
                        self.balance -= (cost + trans_fee)
                        self.position += available
                        self.entry_price = execution_price
                        cost_incurred = trans_fee
                        num_shares = available
                    else:
                        cost_incurred = 0.0
                        num_shares = 0.0
            else:
                cost_incurred = 0.0
                num_shares = 0.0
        
        elif action == -1:  # Sell
            if self.position > 0:
                num_shares = self.position * order_size
                revenue = num_shares * execution_price
                trans_fee = revenue * self.transaction_cost
                net_revenue = revenue - trans_fee
                
                self.balance += net_revenue
                self.position -= num_shares
                cost_incurred = -trans_fee
            else:
                cost_incurred = 0.0
                num_shares = 0.0
        
        else:  # Hold
            cost_incurred = 0.0
            num_shares = 0.0
        
        # Compute unrealized PnL
        if self.position > 0:
            unrealized_pnl = self.position * (self.current_price - self.entry_price)
        else:
            unrealized_pnl = 0.0
        
        # Total portfolio value
        portfolio_value = self.balance + self.position * self.current_price
        pnl = portfolio_value - self.initial_balance
        
        # Record history
        self.price_history.append(self.current_price)
        self.position_history.append(self.position)
        self.balance_history.append(self.balance)
        self.pnl_history.append(pnl)
        
        # Return execution details
        return {
            'action': action,
            'execution_price': execution_price,
            'shares_traded': num_shares,
            'transaction_cost': cost_incurred,
            'new_position': self.position,
            'balance': self.balance,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': pnl,
            'portfolio_value': portfolio_value
        }
    
    def update_price(self, new_price: float) -> None:
        """Update current price without trading."""
        self.current_price = new_price
    
    def set_price(self, price: float) -> None:
        """Set current price."""
        self.current_price = price
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        portfolio_value = self.balance + self.position * self.current_price
        pnl = portfolio_value - self.initial_balance
        
        return {
            'timestamp': self.timestamp,
            'balance': self.balance,
            'position': self.position,
            'price': self.current_price,
            'portfolio_value': portfolio_value,
            'pnl': pnl,
            'entry_price': self.entry_price
        }
    
    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_price = self.initial_price
        self.timestamp = 0
        
        self.price_history = [self.initial_price]
        self.position_history = [0.0]
        self.balance_history = [self.initial_balance]
        self.pnl_history = [0.0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        pnl_array = np.array(self.pnl_history[1:])  # Exclude initial
        trades = sum(1 for i in range(1, len(self.position_history)) 
                    if self.position_history[i] != self.position_history[i-1])
        
        return {
            'n_steps': self.timestamp,
            'n_trades': trades,
            'final_pnl': self.pnl_history[-1],
            'max_pnl': np.max(pnl_array) if len(pnl_array) > 0 else 0,
            'min_pnl': np.min(pnl_array) if len(pnl_array) > 0 else 0,
            'sharpe_ratio': self._compute_sharpe_ratio(),
            'max_drawdown': self._compute_max_drawdown(),
            'win_rate': self._compute_win_rate()
        }
    
    def _compute_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Compute Sharpe ratio."""
        pnl = np.asarray(self.pnl_history, dtype=float)
        balances = np.asarray(self.balance_history[:-1], dtype=float)

        if len(pnl) < 2 or len(balances) == 0:
            return 0.0

        safe_balances = np.where(np.abs(balances) > 1e-12, balances, np.nan)
        returns = np.diff(pnl) / safe_balances
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        
        if return_std == 0:
            return 0.0
        
        return (return_mean - risk_free_rate) / return_std * np.sqrt(252)
    
    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown."""
        portfolio_values = np.array(self.balance_history) + \
                          np.array(self.position_history) * np.array(self.price_history)
        
        if len(portfolio_values) == 0:
            return 0.0

        running_max = np.maximum.accumulate(portfolio_values)
        safe_running_max = np.where(np.abs(running_max) > 1e-12, running_max, np.nan)
        drawdown = (portfolio_values - running_max) / safe_running_max
        drawdown = drawdown[np.isfinite(drawdown)]
        
        return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _compute_win_rate(self) -> float:
        """Compute fraction of profitable trades."""
        if len(self.pnl_history) < 2:
            return 0.0
        
        returns = np.diff(self.pnl_history)
        n_positive = np.sum(returns > 0)
        
        return float(n_positive) / len(returns) if len(returns) > 0 else 0.0