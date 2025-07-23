import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum

class TradingMode(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class MarketSession(Enum):
    LONDON = "london"
    NEW_YORK = "new_york"
    ASIAN = "asian"
    OVERLAP = "overlap"

@dataclass
class SymbolConfig:
    """Configuration for individual trading symbols"""
    name: str
    mt5_symbol: str
    min_lot: float
    max_lot: float
    pip_value: float # For risk calculation
    spread_threshold: int

class Config:
    """Main configuration class for the SMC Trading Bot"""
    
    VERSION = "2.0.0-SMC"
    
    # --- MT5 & Telegram (from .env) ---
    MT5_CONFIG = {
        "login": int(os.getenv("MT5_LOGIN", "248184948")),
        "password": os.getenv("MT5_PASSWORD", "Classixs12340&"),
        "server": os.getenv("MT5_SERVER", "Exness-MT5Trial"),
        "timeout": 60000,
    }
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "5122288334:AAFQbaRRFhgkuH3BKuOyZ27mfALKr7l3AOg")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1766888445")
    
    # --- Trading Symbols ---
    SYMBOLS = {
        "GOLD": SymbolConfig(
            name="GOLD",
            mt5_symbol="XAUUSDm",
            min_lot=0.01,
            max_lot=1.0,
            pip_value=0.1,
            spread_threshold=30,
        ),
        "BITCOIN": SymbolConfig(
            name="BITCOIN",
            mt5_symbol="BTCUSDm",
            min_lot=0.01,
            max_lot=0.5,
            pip_value=1.0,
            spread_threshold=50,
        )
    }
    DEFAULT_SYMBOL = "GOLD"
    MULTI_SYMBOL_MODE = True
    
    # --- Multi-Timeframe (MTF) Strategy ---
    # The new logic uses a specific 3-timeframe structure.
    # [0] = Directional Bias, [1] = Setup/POI, [2] = Entry Confirmation
    MTF_TIMEFRAMES = ["H1", "M15", "M5"]
    
    # --- Risk Management ---
    MAX_DAILY_RISK = 10  # Max number of trades per day
    MAX_DRAWDOWN_PERCENT = 10.0
    
    RISK_CONFIG = {
      "base_risk_percent": 0.5, # 0.5% risk per trade
      "max_lot": 1.0,
    }
    
    # ATR for Stop Loss calculation. TP is now based on Market Structure.
    ATR_PERIODS = 14
    ATR_SL_MULTIPLIER = 1.5
    # ATR_TP_MULTIPLIER is now only a fallback if no liquidity target is found
    ATR_TP_MULTIPLIER = 2.0 
    
    # --- Signal Generation & Confluence ---
    SIGNAL_THRESHOLDS = {
        "min_confidence": 0.7, # Minimum confidence score to execute
        "signal_cooldown": 300 # 5 minutes cooldown per symbol after a signal
    }
    
    CONFLUENCE_WEIGHTS = {
        "poi_reaction": 0.4,
        "ltf_choch": 0.4,
        "volume_confirmation": 0.1,
        "pattern_confirmation": 0.1,
    }

    # --- Session Timing (UTC) ---
    TRADING_SESSIONS = {
        MarketSession.ASIAN: (0, 8),
        MarketSession.LONDON: (8, 16),
        MarketSession.NEW_YORK: (13, 21),
        MarketSession.OVERLAP: (13, 16)
    }
    
    @classmethod
    def get_symbol_config(cls, symbol_name: str) -> SymbolConfig:
        return cls.SYMBOLS.get(symbol_name.upper())

    @classmethod
    def is_trading_session_active(cls, current_hour: int) -> Tuple[bool, MarketSession]:
        for session, (start, end) in cls.TRADING_SESSIONS.items():
            if start <= current_hour < end:
                return True, session
        return False, None