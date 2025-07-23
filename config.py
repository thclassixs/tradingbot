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
    lot_step: float
    pip_value: float
    spread_threshold: int
    volatility_filter: float
    session_multiplier: Dict[MarketSession, float]

class Config:
    """Main configuration class for advanced trading bot"""
    
    # Version and environment settings
    VERSION = "1.1.0"  # Incremented version
    ENVIRONMENT = "production"  # or "development"
    
    # MT5 settings - SECURITY FIX: Use environment variables
    MT5_CONFIG = {
        "login": int(os.getenv("MT5_LOGIN", "248184948")),
        "password": os.getenv("MT5_PASSWORD", "Classixs12340&"),
        "server": os.getenv("MT5_SERVER", "Exness-MT5Trial"),
        "timeout": 60000,
        "retry_count": 3
    }
    
    # TELEGRAM SETTINGS - SECURITY FIX: Use environment variables
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "5122288334:AAFQbaRRFhgkuH3BKuOyZ27mfALKr7l3AOg")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1766888445")
    
    # TRADING SYMBOLS
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
                MarketSession.LONDON: 1.2,
                MarketSession.NEW_YORK: 1.0,
                MarketSession.ASIAN: 0.8,
                MarketSession.OVERLAP: 1.5
            }
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
                MarketSession.LONDON: 1.1,
                MarketSession.NEW_YORK: 1.2,
                MarketSession.ASIAN: 0.9,
                MarketSession.OVERLAP: 1.3
            }
        )
    }
    
    # Default symbol for single-symbol mode
    DEFAULT_SYMBOL = "GOLD"
    MULTI_SYMBOL_MODE = True
    
    # TIMEFRAME SETTINGS
    PRIMARY_TIMEFRAME = "M5"
    TIMEFRAMES = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60
    }
    
    # Multi-timeframe analysis
    MTF_TIMEFRAMES = ["M5", "M15", "H1"]
    HTF_CONFIRMATION = True
    
    # RISK MANAGEMENT
    MAX_RISK_PERCENT = 2.0
    MAX_DAILY_RISK = 24.0
    MAX_DRAWDOWN = 10.0
    
    POSITION_SIZING = {
        TradingMode.CONSERVATIVE: 0.5,
        TradingMode.BALANCED: 1.0,
        TradingMode.AGGRESSIVE: 1.5
    }
    
    CURRENT_MODE = TradingMode.AGGRESSIVE
    
    # ATR-based risk management
    ATR_PERIODS = 14
    ATR_SL_MULTIPLIER = 1.5
    ATR_TP_MULTIPLIER = 2.5
    
    # Dynamic risk adjustment
    VOLATILITY_ADJUSTMENT = True
    HIGH_VOLATILITY_THRESHOLD = 2.0
    LOW_VOLATILITY_THRESHOLD = 0.5

    # --- NEW: ENHANCED RISK CONFIG ---
    RISK_CONFIG = {
      "lot_mode": "dynamic",  # or "fixed"
      "base_risk_percent": 0.25,
      "tier_multipliers": {
        "LOW": 1.0,
        "MED": 2.0,
        "HIGH": 4.0
      },
      "max_lot": 0.50,
      "max_risk_per_trade_percent": 2.0,
      "reset_model": "step_down",  # or "full_reset"
      "streak_map": {
        0: "LOW",
        1: "MED",
        2: "HIGH"
      }
    }

    CAPITAL_CONTROLS = {
      "max_daily_drawdown_percent": 5,
      "max_daily_trades": 25,
      "cooldown_after_loss_seconds": 60
    }

    EQUITY_BANDS = {
      "step_usd": 5   # recompute base lot every $5 change in equity
    }

    TELEGRAM_COMMANDS = {
        "/reset_risk": "reset_streak_to_low",
        "/status": "report_current_state"
    }
    
    # MARKET STRUCTURE SETTINGS
    SWING_DETECTION = {
        "lookback_periods": 5,
        "min_swing_size": 10,
        "confirmation_periods": 2
    }
    
    SUPPORT_RESISTANCE = {
        "touch_threshold": 3,
        "proximity_pips": 5,
        "strength_periods": 100,
        "max_levels": 10
    }
    
    ORDER_BLOCKS = {
        "min_size_pips": 15,
        "max_age_bars": 50,
        "retest_sensitivity": 0.8
    }
    
    FAIR_VALUE_GAPS = {
        "min_gap_pips": 8,
        "max_age_bars": 20,
        "fill_threshold": 0.5
    }
    
    # VOLUME ANALYSIS SETTINGS
    VOLUME_DELTA = {
        "calculation_method": "tick_volume",
        "smoothing_periods": 5,
        "divergence_threshold": 0.3,
        "exhaustion_multiplier": 2.0
    }
    
    VOLUME_PROFILE = {
        "profile_periods": 100,
        "poc_sensitivity": 0.1,
        "value_area_percent": 70
    }
    
    BUY_SELL_PRESSURE = {
        "calculation_window": 20,
        "imbalance_threshold": 0.6,
        "confirmation_periods": 3
    }
    
    # PATTERN ANALYSIS SETTINGS
    CANDLESTICK_PATTERNS = {
        "enable_single_candle": True,
        "enable_multi_candle": True,
        "volume_confirmation": True,
        "min_body_size": 0.3,
        "doji_threshold": 0.1
    }
    
    PATTERN_SCORING = {
        "volume_weight": 0.3,
        "structure_weight": 0.4,
        "confluence_weight": 0.3,
        "min_score": 0.6
    }
    
    PATTERN_CATEGORIES = {
        "REVERSAL": ["hammer", "doji", "engulfing", "harami", "morning_star", "evening_star"],
        "CONTINUATION": ["flag", "pennant", "rising_three", "falling_three"],
        "INDECISION": ["spinning_top", "long_doji", "rickshaw_man"],
        "BREAKOUT": ["marubozu", "belt_hold", "three_white_soldiers", "three_black_crows"]
    }
    
    # ML MODEL SETTINGS
    ML_CONFIG = {
        "model_type": "RandomForest",
        "retrain_hours": 24,
        "min_accuracy": 0.6,
        "feature_importance_threshold": 0.01,
        "cross_validation_folds": 5
    }
    
    FEATURE_ENGINEERING = {
        "lookback_periods": [3, 5, 10, 20],
        "technical_indicators": True,
        "market_structure_features": True,
        "volume_features": True,
        "pattern_features": True,
        "time_features": True
    }
    
    # SIGNAL GENERATION
    SIGNAL_THRESHOLDS = {
        "min_confidence": 0.65,
        "confluence_required": True,
        "min_confluence_score": 0.7,
        "signal_cooldown": 300
    }
    
    CONFLUENCE_WEIGHTS = {
        "ml_prediction": 0.5,
        "market_structure": 0.25,
        "volume_analysis": 0.20,
        "pattern_analysis": 0.15
    }
    
    # EXECUTION SETTINGS
    EXECUTION = {
        "max_slippage": 20,
        "order_retry_attempts": 3,
        "execution_timeout": 30,
        "partial_fills_allowed": True
    }
    
    # MOROCCO-SPECIFIC SETTINGS - FIXED
    MOROCCO_CONFIG = {
        "timezone": "Africa/Casablanca",
        "local_trading_hours": [(9, 18), (20, 23)],
        "ramadan_adjustments": True,
        "weekend_trading": True,
    }
    
    # Session timing (UTC+1 for Morocco)
    TRADING_SESSIONS = {
        MarketSession.LONDON: (8, 17),
        MarketSession.NEW_YORK: (14, 23),
        MarketSession.ASIAN: (0, 9),
        MarketSession.OVERLAP: (14, 17)
    }
    
    # MONITORING & ALERTS
    MONITORING = {
        "telegram_alerts": True,
        "performance_reporting": True,
        "daily_summary": True,
        "error_notifications": True
    }
    
    ALERT_TRIGGERS = {
        "new_signal": True,
        "trade_execution": True,
        "stop_loss_hit": True,
        "take_profit_hit": True,
        "daily_loss_limit": True,
        "system_errors": True
    }
    
    # DATA & STORAGE
    DATA_STORAGE = {
        "save_raw_data": True,
        "save_patterns": True,
        "save_signals": True,
        "data_retention_days": 90,
        "backup_enabled": True
    }
    
    DATABASE_CONFIG = {
        "type": "sqlite",
        "path": "data/trading_data.db",
        "backup_frequency": "daily"
    }
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # ADVANCED FEATURES
    ADVANCED_FEATURES = {
        "multi_symbol_correlation": True,
        "market_regime_detection": True,
        "adaptive_parameters": True,
        "news_sentiment_analysis": True,
        "options_flow_analysis": False,
        "session_analysis": True,
        "liquidity_zone_detection": True,
        "trend_context_analysis": True,
        "pattern_confluence": True,
        "volume_exhaustion_detection": True,
        "absorption_detection": True,
        "tick_by_tick_analysis": True,
        "multi_timeframe_patterns": True,
        "correlation_analysis": True
    }
    
    # PERFORMANCE OPTIMIZATION
    PERFORMANCE = {
        "parallel_processing": True,
        "max_workers": 4,
        "memory_optimization": True,
        "cache_indicators": True,
        "cache_duration": 300
    }
    
    # DEVELOPMENT & DEBUG
    DEBUG = {
        "debug_mode": False,
        "verbose_logging": True,
        "save_debug_data": False,
        "backtesting_mode": False,
        "paper_trading": False
    }
    
    # Validation methods
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Validate risk settings
        if cls.MAX_RISK_PERCENT <= 0 or cls.MAX_RISK_PERCENT > 10:
            errors.append("MAX_RISK_PERCENT should be between 0 and 10")
        
        # Validate symbols
        for symbol_name, symbol_config in cls.SYMBOLS.items():
            if symbol_config.min_lot <= 0:
                errors.append(f"Invalid min_lot for {symbol_name}")
        
        # Validate thresholds
        if cls.SIGNAL_THRESHOLDS["min_confidence"] < 0.5:
            errors.append("Minimum confidence should be at least 0.5")
        
        # Validate MT5 credentials
        if not cls.MT5_CONFIG["login"] or not cls.MT5_CONFIG["password"]:
            errors.append("MT5 credentials not properly configured")
        
        # Validate trading hours - NEW VALIDATION
        for start, end in cls.MOROCCO_CONFIG["local_trading_hours"]:
            if not (0 <= start <= 23) or not (0 <= end <= 23):
                errors.append(f"Invalid trading hours: ({start}, {end}). Hours must be between 0-23")
        
        # Validate trading sessions
        for session, (start, end) in cls.TRADING_SESSIONS.items():
            if not (0 <= start <= 23) or not (0 <= end <= 23):
                errors.append(f"Invalid session hours for {session.value}: ({start}, {end})")
        
        if errors:
            raise ValueError(f"Configuration errors: {errors}")
        
        return True
    
    @classmethod
    def get_symbol_config(cls, symbol_name: str) -> SymbolConfig:
        """Get configuration for specific symbol"""
        return cls.SYMBOLS.get(symbol_name.upper(), cls.SYMBOLS[cls.DEFAULT_SYMBOL])
    
    @classmethod
    def is_trading_session_active(cls, current_hour: int) -> Tuple[bool, MarketSession]:
        """Check if current hour is in active trading session"""
        # Validate current_hour parameter
        if not (0 <= current_hour <= 23):
            raise ValueError(f"Invalid hour: {current_hour}. Hour must be between 0-23")
        
        for session, (start, end) in cls.TRADING_SESSIONS.items():
            if start <= current_hour <= end:
                return True, session
        return False, None
    
    @classmethod
    def is_local_trading_time(cls, current_hour: int) -> bool:
        """Check if current hour is within local Morocco trading hours"""
        if not (0 <= current_hour <= 23):
            raise ValueError(f"Invalid hour: {current_hour}. Hour must be between 0-23")
        
        for start, end in cls.MOROCCO_CONFIG["local_trading_hours"]:
            if start <= current_hour <= end:
                return True
        return False
    
    @classmethod
    def load_encrypted_credentials(cls):
        """Load encrypted credentials from environment variables"""
        return {
            "login": cls.MT5_CONFIG["login"],
            "password": cls.MT5_CONFIG["password"],
            "server": cls.MT5_CONFIG["server"]
        }
    
    @classmethod
    def get_trading_mode(cls) -> TradingMode:
        """Get current trading mode with validation"""
        if not isinstance(cls.CURRENT_MODE, TradingMode):
            return TradingMode.CONSERVATIVE
        return cls.CURRENT_MODE