import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from config import Config

@dataclass
class RiskParameters:
    max_risk_percent: float
    position_size: float
    stop_loss: float
    take_profit: float
    risk_reward: float

class RiskManagement:
    def __init__(self, account_balance: float, mt5_handler, max_risk_percent: float = 2.0):
        self.account_balance = account_balance
        self.max_risk_percent = max_risk_percent
        self.mt5_handler = mt5_handler # Store the MT5 handler instance
        self.config = Config()
        
    async def calculate_position_size(self, signal) -> float:
        """Calculate optimal position size based on risk parameters and current tier."""
        symbol_info = await self.mt5_handler.get_symbol_info(signal.symbol)
        if not symbol_info:
            return 0.0

        # --- FIX: Get pip_value from config, not symbol_info ---
        symbol_config = self.config.get_symbol_config(signal.symbol)
        pip_value = symbol_config.pip_value

        # Determine risk percentage for the current tier
        base_risk_percent = self.config.RISK_CONFIG['base_risk_percent']
        tier_multiplier = self.config.RISK_CONFIG['tier_multipliers'][signal.risk_tier]
        risk_percent = base_risk_percent * tier_multiplier

        # Calculate risk amount
        risk_amount = self.account_balance * (risk_percent / 100)
        
        # Calculate stop distance in price terms
        stop_distance_price = abs(signal.entry_price - signal.stop_loss)
        
        # Calculate position size
        if stop_distance_price > 0 and pip_value > 0:
            position_size = risk_amount / (stop_distance_price * pip_value)
        else:
            position_size = symbol_config.min_lot

        # Clamp to min/max lot and broker rules
        position_size = max(symbol_info['min_lot'], min(position_size, symbol_info['max_lot'], self.config.RISK_CONFIG['max_lot']))
        
        # Adjust for lot step
        lot_step = symbol_info['lot_step']
        position_size = round(position_size / lot_step) * lot_step
        
        return position_size

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
    
    async def calculate_dynamic_exits(self, df: pd.DataFrame, entry_price: float,
                                    direction: str, symbol: str) -> Tuple[float, float]:
        """Calculate dynamic exit points based on market structure and broker rules."""
        # Get symbol info to find the minimum stop level
        symbol_info = await self.mt5_handler.get_symbol_info(symbol)
        if not symbol_info:
            # Fallback if symbol info is not available
            return entry_price * 0.99, entry_price * 1.01

        # Calculate the minimum distance required by the broker
        stops_level = symbol_info.get('stops_level', 10) # Default to 10 if not present
        point = symbol_info['point']
        min_stop_distance = stops_level * point

        # Calculate the stop distance based on ATR
        atr = df['atr'].iloc[-1]
        atr_stop_distance = self.config.ATR_SL_MULTIPLIER * atr

        # Use the larger of the two distances to ensure we meet the broker's requirement
        final_stop_distance = max(min_stop_distance, atr_stop_distance)

        if direction.upper() == 'BUY':
            stop_loss = entry_price - final_stop_distance
            take_profit = entry_price + (final_stop_distance * self.config.ATR_TP_MULTIPLIER) # Maintain R:R
        else: # SELL
            stop_loss = entry_price + final_stop_distance
            take_profit = entry_price - (final_stop_distance * self.config.ATR_TP_MULTIPLIER) # Maintain R:R

        return round(stop_loss, symbol_info['digits']), round(take_profit, symbol_info['digits'])
    
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
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    async def validate_signal(self, signal) -> bool:
        """
        Validate a trading signal against risk rules.
        """
        if signal.stop_loss is None or abs(signal.entry_price - signal.stop_loss) < 0.0001:
            return False

        return True