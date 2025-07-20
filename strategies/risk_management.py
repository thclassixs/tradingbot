import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class RiskParameters:
    max_risk_percent: float
    position_size: float
    stop_loss: float
    take_profit: float
    risk_reward: float

class RiskManagement:
    def __init__(self, account_balance: float, max_risk_percent: float = 2.0):
        self.account_balance = account_balance
        self.max_risk_percent = max_risk_percent
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              pip_value: float) -> RiskParameters:
        """Calculate optimal position size based on risk parameters"""
        risk_amount = self.account_balance * (self.max_risk_percent / 100)
        stop_distance = abs(entry_price - stop_loss)
        
        # Calculate position size
        position_size = risk_amount / (stop_distance * pip_value)
        
        # Calculate take profit based on RR ratio
        take_profit = entry_price + (stop_distance * 2)  # 1:2 RR ratio
        
        return RiskParameters(
            max_risk_percent=self.max_risk_percent,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=2.0
        )
    
    def adjust_for_market_conditions(self, base_size: float, 
                                   volatility: float, 
                                   session_multiplier: float) -> float:
        """Adjust position size based on market conditions"""
        adjusted_size = base_size * session_multiplier
        
        if volatility > 1.5:  # High volatility
            adjusted_size *= 0.75
        elif volatility < 0.5:  # Low volatility
            adjusted_size *= 1.25
            
        return adjusted_size
    
    def calculate_dynamic_exits(self, df: pd.DataFrame, entry_price: float, 
                              direction: str) -> Tuple[float, float]:
        """Calculate dynamic exit points based on market structure"""
        atr = df['atr'].iloc[-1]
        
        if direction == 'long':
            stop_loss = entry_price - (atr * 1.5)
            take_profit = entry_price + (atr * 2.5)
        else:
            stop_loss = entry_price + (atr * 1.5)
            take_profit = entry_price - (atr * 2.5)
            
        return stop_loss, take_profit
    
    def context_aware_position_sizing(self, base_size: float, market_context: str, session: str) -> float:
        """Adjust position size based on market context and session"""
        multiplier = 1.0
        if market_context == "Uptrend":
            multiplier *= 1.1
        elif market_context == "Downtrend":
            multiplier *= 0.9
        if session == "London":
            multiplier *= 1.2
        elif session == "New York":
            multiplier *= 1.1
        return base_size * multiplier
    
    def adjust_risk_for_symbol(self, symbol: str, base_size: float, config) -> float:
        """Adjust risk for specific symbol using config"""
        symbol_cfg = config.get_symbol_config(symbol)
        return min(base_size, symbol_cfg.max_lot)

    def check_daily_risk_limit(self, trades_today: float, max_daily_risk: float) -> bool:
        """Check if daily risk limit is exceeded"""
        return trades_today < max_daily_risk

    async def validate_signal(self, signal) -> bool:
        """
        Validate a trading signal against risk rules.
        This is a placeholder and should be expanded with actual risk logic.
        """
        # Example: Check if stop loss is too close or too far
        if signal.stop_loss is None or abs(signal.entry_price - signal.stop_loss) < 0.001:  # Example threshold
            # Implement actual logic here (e.g., check against ATR, fixed pips)
            return False  # Signal rejected

        # Example: Ensure position size is within limits (requires symbol config)
        # This would require passing the Config object or getting symbol info
        # For now, let's assume it passes

        return True  # Signal is valid based on risk rules
