"""
Advanced Logging System for Trading Bot
Handles logging, error tracking, and performance monitoring
"""

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
    
    def __init__(self, name: str = "TradingBot", logs_dir: str = "logs"):
        self.name = name
        self.logs_dir = logs_dir
        self.trading_events: list = []
        self.performance_metrics: Dict[str, Any] = {}
        self.error_count: Dict[str, int] = {}
        
        # Create logs directory
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup loggers
        self.main_logger = self._setup_main_logger()
        self.trading_logger = self._setup_trading_logger()
        self.error_logger = self._setup_error_logger()
        self.performance_logger = self._setup_performance_logger()
        
        # Start session
        self._log_session_start()
    
    def _setup_main_logger(self) -> logging.Logger:
        """Setup main application logger"""
        logger = logging.getLogger(f"{self.name}.main")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(category)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
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
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        
        return logger
    
    def _setup_trading_logger(self) -> logging.Logger:
        """Setup trading-specific logger"""
        logger = logging.getLogger(f"{self.name}.trading")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Trading events file
        handler = TimedRotatingFileHandler(
            filename=os.path.join(self.logs_dir, 'trading_events.log'),
            when='midnight',
            interval=1,
            backupCount=90,
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
        return logger
    
    def _setup_error_logger(self) -> logging.Logger:
        """Setup error tracking logger"""
        logger = logging.getLogger(f"{self.name}.errors")
        logger.setLevel(logging.WARNING)
        logger.handlers.clear()
        
        # Error file with larger size limit
        handler = RotatingFileHandler(
            filename=os.path.join(self.logs_dir, 'errors.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
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
        logger.handlers.clear()
        
        # Performance metrics file
        handler = TimedRotatingFileHandler(
            filename=os.path.join(self.logs_dir, 'performance.log'),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
        return logger
    
    def _log_session_start(self):
        """Log session start information"""
        self.info("=" * 60, LogCategory.SYSTEM)
        self.info("🚀 TRADING BOT SESSION STARTED", LogCategory.SYSTEM)
        self.info(f"⏰ Timestamp: {datetime.now()}", LogCategory.SYSTEM)
        self.info(f"📁 Logs Directory: {self.logs_dir}", LogCategory.SYSTEM)
        self.info("=" * 60, LogCategory.SYSTEM)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, category, **kwargs)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, category, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, category, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, exception: Exception = None, **kwargs):
        """Log error message"""
        self._log(LogLevel.ERROR, message, category, exception=exception, **kwargs)
        
        # Track error count
        error_key = f"{category.value}:{message[:50]}"
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, exception: Exception = None, **kwargs):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, category, exception=exception, **kwargs)
    
    def _log(self, level: LogLevel, message: str, category: LogCategory, exception: Exception = None, **kwargs):
        """Internal logging method"""
        # Prepare log message
        log_message = message
        if kwargs:
            details = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            log_message = f"{message} | {details}"
        
        # Add category to logger context
        logger_dict = {
            'category': category.value
        }
        
        # Log to appropriate logger
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            if exception:
                self.error_logger.log(
                    getattr(logging, level.value),
                    log_message,
                    exc_info=exception,
                    extra=logger_dict
                )
            else:
                self.error_logger.log(
                    getattr(logging, level.value),
                    log_message,
                    extra=logger_dict
                )
        else:
            self.main_logger.log(
                getattr(logging, level.value),
                log_message,
                extra=logger_dict
            )
    
    def log_trading_event(self, event: TradingEvent):
        """Log trading-specific event"""
        self.trading_events.append(event)
        
        # Create detailed message
        message_parts = [
            f"Event: {event.event_type}",
            f"Category: {event.category.value}"
        ]
        
        if event.symbol:
            message_parts.append(f"Symbol: {event.symbol}")
        if event.signal:
            message_parts.append(f"Signal: {event.signal}")
        if event.confidence:
            message_parts.append(f"Confidence: {event.confidence:.2f}%")
        if event.price:
            message_parts.append(f"Price: {event.price}")
        if event.lot_size:
            message_parts.append(f"Lot: {event.lot_size}")
        if event.pnl:
            message_parts.append(f"PnL: {event.pnl}")
        
        message = " | ".join(message_parts)
        
        if event.details:
            details_str = json.dumps(event.details, default=str)
            message += f" | Details: {details_str}"
        
        self.trading_logger.info(message)
        
        # Also log to main logger with appropriate emoji
        emoji_map = {
            "SIGNAL_GENERATED": "🎯",
            "TRADE_EXECUTED": "✅",
            "TRADE_CLOSED": "🏁",
            "STOP_LOSS": "🛑",
            "TAKE_PROFIT": "💰",
            "MODEL_RETRAINED": "🧠",
            "ERROR": "❌"
        }
        
        emoji = emoji_map.get(event.event_type, "📊")
        self.info(f"{emoji} {message}", event.category)
    
    def log_signal(self, symbol: str, signal: str, confidence: float, price: float, **kwargs):
        """Log trading signal"""
        event = TradingEvent(
            timestamp=datetime.now(),
            category=LogCategory.SIGNALS,
            event_type="SIGNAL_GENERATED",
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=price,
            details=kwargs
        )
        self.log_trading_event(event)
    
    def log_execution(self, symbol: str, signal: str, lot_size: float, price: float, result_code: int, **kwargs):
        """Log trade execution"""
        event_type = "TRADE_EXECUTED" if result_code == 10009 else "EXECUTION_FAILED"
        event = TradingEvent(
            timestamp=datetime.now(),
            category=LogCategory.EXECUTION,
            event_type=event_type,
            symbol=symbol,
            signal=signal,
            lot_size=lot_size,
            price=price,
            details={"result_code": result_code, **kwargs}
        )
        self.log_trading_event(event)
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.performance_metrics.update(metrics)
        metrics_str = json.dumps(metrics, default=str, indent=2)
        self.performance_logger.info(f"Performance Update:\n{metrics_str}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary statistics"""
        now = datetime.now()
        session_start = min([event.timestamp for event in self.trading_events]) if self.trading_events else now
        
        # Count events by type
        event_counts = {}
        for event in self.trading_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        # Calculate PnL
        total_pnl = sum([event.pnl for event in self.trading_events if event.pnl])
        
        return {
            "session_duration": str(now - session_start),
            "total_events": len(self.trading_events),
            "event_breakdown": event_counts,
            "total_pnl": total_pnl,
            "error_count": sum(self.error_count.values()),
            "top_errors": dict(list(self.error_count.items())[:5])
        }
    
    def export_session_data(self, filename: Optional[str] = None) -> str:
        """Export session data to JSON file"""
        if not filename:
            filename = f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.logs_dir, filename)
        
        export_data = {
            "session_summary": self.get_session_summary(),
            "trading_events": [asdict(event) for event in self.trading_events],
            "performance_metrics": self.performance_metrics,
            "error_statistics": self.error_count
        }
        
        # Convert datetime objects to strings
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=json_serializer, ensure_ascii=False)
        
        self.info(f"📁 Session data exported to: {filepath}", LogCategory.SYSTEM)
        return filepath
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for filename in os.listdir(self.logs_dir):
            filepath = os.path.join(self.logs_dir, filename)
            if os.path.isfile(filepath):
                file_date = datetime.fromtimestamp(os.path.getctime(filepath))
                if file_date < cutoff_date:
                    os.remove(filepath)
                    self.info(f"🗑️ Removed old log file: {filename}", LogCategory.SYSTEM)

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
    
    CATEGORY_COLORS = {
        'SYSTEM': '\033[96m',      # Light Cyan
        'TRADING': '\033[92m',     # Light Green
        'SIGNALS': '\033[93m',     # Light Yellow
        'EXECUTION': '\033[91m',   # Light Red
        'RISK': '\033[95m',        # Light Magenta
        'MARKET_DATA': '\033[94m', # Light Blue
        'ML_MODEL': '\033[90m',    # Dark Gray
        'TELEGRAM': '\033[97m',    # White
        'PERFORMANCE': '\033[35m'   # Purple
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Add color to category if present
        if hasattr(record, 'category'):
            category_color = self.CATEGORY_COLORS.get(record.category, '')
            record.category = f"{category_color}{record.category}{self.COLORS['RESET']}"
        else:
            record.category = ''
        
        return super().format(record)