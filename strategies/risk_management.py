import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from config import Config
from .market_structure import MarketStructure

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
        self.mt5_handler = mt5_handler
        self.config = Config()
        self.market_structure = MarketStructure()

    async def calculate_position_size(self, signal) -> float:
        symbol_info = await self.mt5_handler.get_symbol_info(signal.symbol)
        if not symbol_info: return 0.0

        symbol_config = self.config.get_symbol_config(signal.symbol)
        risk_amount = self.account_balance * (self.config.RISK_CONFIG['base_risk_percent'] / 100)
        stop_distance_price = abs(signal.entry_price - signal.stop_loss)

        if stop_distance_price <= 0: return symbol_config.min_lot
        
        # Correctly calculate value per pip
        value_per_pip = symbol_info['trade_contract_size'] * symbol_info['point']
        pip_value = value_per_pip / signal.entry_price if "USD" in signal.symbol else value_per_pip
        
        position_size = risk_amount / (stop_distance_price / symbol_info['point'] * pip_value)
        
        position_size = max(symbol_config.min_lot, min(position_size, symbol_config.max_lot))
        return round(position_size / symbol_info['lot_step']) * symbol_info['lot_step']

    async def calculate_dynamic_exits(self, df: pd.DataFrame, entry_price: float,
                                    direction: str, symbol: str) -> Tuple[float, float]:
        symbol_info = await self.mt5_handler.get_symbol_info(symbol)
        if not symbol_info:
            sl = entry_price * (1 - 0.01) if direction.upper() == "BUY" else entry_price * (1 + 0.01)
            tp = entry_price * (1 + 0.02) if direction.upper() == "BUY" else entry_price * (1 - 0.02)
            return sl, tp

        atr = self.calculate_atr(df).iloc[-1]
        stop_distance = self.config.ATR_SL_MULTIPLIER * atr
        
        highs, lows = self.market_structure.identify_swing_points(df)
        
        if direction.upper() == 'BUY':
            stop_loss = entry_price - stop_distance
            recent_highs = [df['high'].iloc[h] for h in highs if df['high'].iloc[h] > entry_price]
            take_profit = min(recent_highs) if recent_highs else entry_price + (stop_distance * self.config.ATR_TP_MULTIPLIER)
        else:
            stop_loss = entry_price + stop_distance
            recent_lows = [df['low'].iloc[l] for l in lows if df['low'].iloc[l] < entry_price]
            take_profit = max(recent_lows) if recent_lows else entry_price - (stop_distance * self.config.ATR_TP_MULTIPLIER)

        return round(stop_loss, symbol_info['digits']), round(take_profit, symbol_info['digits'])
    
    # *** FIX: Reverted parameter name to match the calling code and Config file ***
    def check_daily_risk_limit(self, trades_today: int, max_daily_risk: float) -> bool:
        """Check if daily risk limit (by number of trades) is exceeded."""
        return trades_today < max_daily_risk
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([df['high'] - df['low'], 
                        abs(df['high'] - df['close'].shift()), 
                        abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    async def validate_signal(self, signal) -> bool:
        if not all([signal.stop_loss, signal.take_profit]) or signal.entry_price == signal.stop_loss:
            return False

        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        
        if risk <= 0 or (reward / risk) < 1.0: # Enforce minimum 1R
            return False

        return True