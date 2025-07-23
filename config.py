import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
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
    lot_step: float
    pip_value: float
    spread_threshold: int
    volatility_filter: float
    session_multiplier: Dict[MarketSession, float]
    
    strategy_name: str = "default_smc"
    min_confidence: float = 0.70
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.5

class Config:
    """Main configuration class for advanced trading bot"""
    
    VERSION = "1.2.2"
    ENVIRONMENT = "production"
    
    MT5_CONFIG = {
        "login": int(os.getenv("MT5_LOGIN", "248184948")),
        "password": os.getenv("MT5_PASSWORD", "Classixs12340&"),
        "server": os.getenv("MT5_SERVER", "Exness-MT5Trial"),
        "timeout": 60000,
        "retry_count": 3,
        "magic": 234000
    }
    
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "5122288334:AAFQbaRRFhgkuH3BKuOyZ27mfALKr7l3AOg")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1766888445")
    
    SYMBOLS = {
        "GOLD": SymbolConfig(
            name="GOLD",
            mt5_symbol="XAUUSDm",
            min_lot=0.01,
            max_lot=10.0,
            lot_step=0.01,
            pip_value=0.1,
            spread_threshold=30,
            volatility_filter=0.5,
            session_multiplier={
                MarketSession.LONDON: 1.2, MarketSession.NEW_YORK: 1.0,
                MarketSession.ASIAN: 0.8, MarketSession.OVERLAP: 1.5
            },
            strategy_name="gold_reversal",
            min_confidence=0.75,
            atr_sl_multiplier=1.2,
            atr_tp_multiplier=2.0,
        ),
        "BITCOIN": SymbolConfig(
            name="BITCOIN",
            mt5_symbol="BTCUSDm",
            min_lot=0.01,
            max_lot=5.0,
            lot_step=0.01,
            pip_value=1.0,
            spread_threshold=50,
            volatility_filter=1.0,
            session_multiplier={
                MarketSession.LONDON: 1.1, MarketSession.NEW_YORK: 1.2,
                MarketSession.ASIAN: 0.9, MarketSession.OVERLAP: 1.3
            },
            strategy_name="btc_breakout",
            min_confidence=0.65,
            atr_sl_multiplier=2.0,
            atr_tp_multiplier=3.0,
        )
    }
    
    DEFAULT_SYMBOL = "GOLD"
    MULTI_SYMBOL_MODE = True
    
    PRIMARY_TIMEFRAME = "M5"
    TIMEFRAMES = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60
    }
    MTF_TIMEFRAMES = ["M5", "M15", "H1"]
    
    # --- FIX: RESTORED MISSING RISK MANAGEMENT ATTRIBUTES ---
    MAX_RISK_PERCENT = 2.0
    MAX_DAILY_RISK = 10.0
    MAX_DRAWDOWN = 10.0
    RISK_CONFIG = {
      "base_risk_percent": 0.25,
      "tier_multipliers": { "LOW": 1.0, "MED": 2.0, "HIGH": 4.0 },
      "max_lot": 0.50,
    }
    # --------------------------------------------------------

    ADVANCED_FEATURES = {
        "auto_lot_boost": True,
        "breakeven_trailing": True,
        "hedge_logic": True,
    }

    AUTO_LOT_BOOST = {
        "enabled": True,
        "win_streak_threshold": 3,
        "boost_multiplier": 1.5,
        "reset_on_loss": True,
    }

    BREAKEVEN_TRAILING = {
        "enabled": True,
        "breakeven_pips": 10,
        "trailing_step_pips": 5,
    }

    HEDGE_LOGIC = {
        "enabled": True,
        "drawdown_pips_threshold": 50,
        "hedge_cooldown_minutes": 30,
    }

    CAPITAL_CONTROLS = {
      "max_daily_drawdown_percent": 5,
      "max_daily_trades": 25,
      "cooldown_after_loss_seconds": 60
    }

    SIGNAL_THRESHOLDS = {
        "signal_cooldown": 300
    }

    EXECUTION = {
        "max_slippage": 20
    }
    
    TRADING_SESSIONS = {
        MarketSession.LONDON: (8, 17),
        MarketSession.NEW_YORK: (14, 23),
        MarketSession.ASIAN: (0, 9),
        MarketSession.OVERLAP: (14, 17)
    }
    
    MONITORING = {"telegram_alerts": True}
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    for directory in [DATA_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    DEBUG = {"close_positions_on_shutdown": False}
    
    @classmethod
    def validate_config(cls):
        return True
    
    @classmethod
    def get_symbol_config(cls, symbol_name: str) -> SymbolConfig:
        return cls.SYMBOLS.get(symbol_name.upper(), cls.SYMBOLS[cls.DEFAULT_SYMBOL])
    
    @classmethod
    def is_trading_session_active(cls, current_hour: int) -> Tuple[bool, Optional[MarketSession]]:
        if not (0 <= current_hour <= 23):
            raise ValueError(f"Invalid hour: {current_hour}")
        
        for session, (start, end) in cls.TRADING_SESSIONS.items():
            if start <= current_hour < end:
                return True, session
        return False, None