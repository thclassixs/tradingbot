import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
import traceback
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    SYSTEM = "SYSTEM"
    TRADING = "TRADING"
    SIGNALS = "SIGNALS"
    EXECUTION = "EXECUTION"
    RISK = "RISK"
    MARKET_DATA = "MARKET_DATA"
    ML_MODEL = "ML_MODEL"
    TELEGRAM = "TELEGRAM"
    PERFORMANCE = "PERFORMANCE"
    VOLUME_DELTA = "VOLUME_DELTA"
    PATTERN = "PATTERN"
    STRUCTURE = "STRUCTURE"
    SESSION = "SESSION"
    NEWS = "NEWS"
    CORRELATION = "CORRELATION"

@dataclass
class TradingEvent:
    """Structure for trading-specific log events"""
    timestamp: datetime
    category: LogCategory
    event_type: str
    symbol: Optional[str] = None
    signal: Optional[str] = None
    confidence: Optional[float] = None
    price: Optional[float] = None
    lot_size: Optional[float] = None
    pnl: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

class TradingLogger:
    """Advanced logging system for trading bot"""

    _session_started = False  # Class variable to ensure the banner prints only once

    def __init__(self, name: str = "TradingBot", logs_dir: str = "logs"):
        """
        Initializes the logger. The session start banner will only be printed
        once per application run.
        """
        self.name = name
        self.logs_dir = logs_dir
        self.trading_events: list = []
        self.performance_metrics: Dict[str, Any] = {}
        self.error_count: Dict[str, int] = {}
        
        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup the different loggers
        self.main_logger = self._setup_main_logger()
        self.trading_logger = self._setup_trading_logger()
        self.error_logger = self._setup_error_logger()
        self.performance_logger = self._setup_performance_logger()
        
        # Log the session start banner only on the first instantiation
        if not TradingLogger._session_started:
            self._log_session_start()
            TradingLogger._session_started = True
    
    def _setup_main_logger(self) -> logging.Logger:
        """Sets up the main application logger for console and file output."""
        logger = logging.getLogger(f"{self.name}.main")
        logger.setLevel(logging.INFO)  # Set the lowest level to capture all messages
        
        # Prevent duplicate handlers if this method is called multiple times
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # --- Console Handler (for clean, high-level output) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s', # Simplified format for console
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        # Only show WARNINGS and above on the console to keep it clean
        console_handler.setLevel(logging.WARNING)
        
        # --- File Handler (for detailed, persistent logs) ---
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(self.logs_dir, 'trading_bot.log'),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO) # Log everything to the file
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        
        return logger
    
    def _setup_trading_logger(self) -> logging.Logger:
        """Setup trading-specific logger"""
        logger = logging.getLogger(f"{self.name}.trading")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        handler = TimedRotatingFileHandler(
            filename=os.path.join(self.logs_dir, 'trading_events.log'),
            when='midnight', interval=1, backupCount=90, encoding='utf-8'
        )
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
    
    def _setup_error_logger(self) -> logging.Logger:
        """Setup error tracking logger"""
        logger = logging.getLogger(f"{self.name}.errors")
        logger.setLevel(logging.WARNING)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        handler = RotatingFileHandler(
            filename=os.path.join(self.logs_dir, 'errors.log'),
            maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s\n%(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance monitoring logger"""
        logger = logging.getLogger(f"{self.name}.performance")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        handler = TimedRotatingFileHandler(
            filename=os.path.join(self.logs_dir, 'performance.log'),
            when='midnight', interval=1, backupCount=30, encoding='utf-8'
        )
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
    
    def _log_session_start(self):
        """Log session start information to the file, and print a minimal version to console."""
        banner = "=" * 60
        start_message = (
            f"\n{banner}\n"
            f"🚀 TRADING BOT SESSION STARTED\n"
            f"⏰ Timestamp: {datetime.now()}\n"
            f"📁 Logs Directory: {os.path.abspath(self.logs_dir)}\n"
            f"{banner}"
        )
        # Log the full banner to the file
        self.main_logger.info(start_message)
        # Print a single, clean line to the console
        print(f"Trading Bot v{Config.VERSION} Started. Logging to '{os.path.abspath(self.logs_dir)}'")

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        self._log(logging.ERROR, message, exception=exception, **kwargs)
        error_key = message[:50]
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
    
    def critical(self, message: str, exception: Exception = None, **kwargs):
        self._log(logging.CRITICAL, message, exception=exception, **kwargs)
    
    def _log(self, level: int, message: str, exception: Exception = None, **kwargs):
        """Internal logging method."""
        log_message = message
        if kwargs:
            details = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            log_message = f"{message} | {details}"
        
        if level >= logging.ERROR and exception:
            self.error_logger.log(level, log_message, exc_info=exception)
        else:
            self.main_logger.log(level, log_message)

# (The rest of the file, including ColoredFormatter, remains the same)
# ...
class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname: <8}{self.COLORS['RESET']}" # Padded
        return super().format(record)

# Need to add Config import for the version number
from config import Config
