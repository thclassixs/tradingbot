"""
Main Trading Bot
"""
import asyncio
import signal
import sys
from datetime import datetime
import pandas as pd

# --- Corrected and Organized Imports ---
from config import Config
from components.mt5_handler import MT5Handler
from strategies.market_structure import MarketStructure
from strategies.volume_analysis import VolumeAnalysis
from strategies.pattern_analysis import PatternAnalysis
from strategies.session_analysis import SessionAnalysis
from strategies.risk_management import RiskManagement
from strategies.signal_generator import SignalGenerator
from utils.logger import TradingLogger
# Import TradeSignal from its new central location in helpers
from utils.helpers import TelegramNotifier, TradeSignal


class TradingBot:
    """
    The main class for the trading bot. It orchestrates all other components,
    manages the main trading loop, and handles state.
    """

    def __init__(self):
        """Initializes the bot's configuration, logger, and state variables."""
        self.config = Config()
        self.logger = TradingLogger("TradingBot")
        
        # Core components will be initialized in the async `initialize` method
        self.mt5_handler = None
        self.signal_generator = None
        self.risk_manager = None
        
        # Analysis modules
        self.market_structure = None
        self.volume_analyzer = None
        self.pattern_analyzer = None
        self.session_analyzer = None
        
        # Utilities
        self.telegram_notifier = None
        
        # Bot state
        self.is_running = False
        self.is_initialized = False
        self.last_signal_time = {}
        
        # Performance tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_time = None
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Sets up handlers for SIGINT and SIGTERM to ensure graceful shutdown."""
        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}, shutting down gracefully...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize(self) -> bool:
        """
        Asynchronously initializes all components of the bot in the correct order.
        Establishes connections and prepares the bot for the main trading loop.
        """
        try:
            self.logger.info("Starting bot initialization...")

            # 1. Validate configuration first
            Config.validate_config()
            self.logger.info("Configuration validated successfully")
            
            # 2. Initialize MT5 Handler and connect to the server
            self.mt5_handler = MT5Handler()
            if not await self.mt5_handler.initialize():
                self.logger.error("Failed to initialize MT5 handler. Bot cannot start.")
                return False
            
            # 3. Now that we are connected, get account info and initialize Risk Management
            account_info = await self.mt5_handler.get_account_info()
            account_balance = account_info.get("balance", 0.0)
            self.risk_manager = RiskManagement(account_balance=account_balance, max_risk_percent=Config.MAX_RISK_PERCENT)
            self.logger.info(f"Risk Manager initialized with balance: {account_balance}")

            # 4. Initialize all analysis modules
            self.market_structure = MarketStructure()
            self.volume_analyzer = VolumeAnalysis()
            self.pattern_analyzer = PatternAnalysis()
            self.session_analyzer = SessionAnalysis()
            
            # 5. Initialize Signal Generator, passing all required components
            self.signal_generator = SignalGenerator(
                market_structure=self.market_structure,
                volume_analysis=self.volume_analyzer,
                pattern_analysis=self.pattern_analyzer,
                session_analysis=self.session_analyzer,
                risk_management=self.risk_manager
            )
            
            # 6. Initialize Telegram Notifier if enabled
            if Config.MONITORING["telegram_alerts"]:
                self.telegram_notifier = TelegramNotifier(
                    bot_token=Config.TELEGRAM_TOKEN,
                    chat_id=Config.TELEGRAM_CHAT_ID
                )
                await self.telegram_notifier.initialize()
            
            # 7. Run any initialization methods within the components themselves
            await self._initialize_components()
            
            # 8. Create necessary data directories
            import os
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            os.makedirs(Config.LOGS_DIR, exist_ok=True)
            
            self.is_initialized = True
            self.start_time = datetime.now()
            
            await self._send_notification("🤖 Trading Bot Initialized Successfully")
            self.logger.info("Trading bot initialization completed.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"A critical error occurred during initialization: {e}", exc_info=True)
            print(f"Initialization failed: {e}")
            await self._send_notification(f"❌ Bot Initialization Failed: {str(e)}")
            return False

    async def _initialize_components(self):
        """Helper method to run the async initialize method on each component if it exists."""
        components = [
            self.market_structure,
            self.volume_analyzer,
            self.pattern_analyzer,
            self.session_analyzer,
            # SignalGenerator and RiskManager don't have their own `initialize` methods currently
        ]
        
        for component in components:
            if hasattr(component, 'initialize') and asyncio.iscoroutinefunction(component.initialize):
                await component.initialize()
                self.logger.info(f"Initialized {component.__class__.__name__}")

    async def run(self):
        """The main execution loop of the bot."""
        if not self.is_initialized:
            self.logger.error("Bot not initialized. Cannot run. Call initialize() first.")
            return
        
        try:
            self.is_running = True
            self.logger.info("🚀 Trading Bot Started and entering main loop.")
            await self._send_notification("🚀 Trading Bot Started")
            
            while self.is_running:
                try:
                    # Check if global conditions allow trading
                    if not await self._should_trade():
                        await asyncio.sleep(60)  # Wait a minute before checking again
                        continue
                    
                    # Process the main trading logic for all symbols
                    await self._process_trading_cycle()

                    # Send a periodic status update to Telegram
                    summary_data = await self.get_performance_summary()
                    await self.telegram_notifier.send_status_update(
                        status="Monitoring markets...",
                        details=summary_data
                    )
                    
                    # Wait for the configured cooldown period before the next cycle
                    await asyncio.sleep(Config.SIGNAL_THRESHOLDS["signal_cooldown"])
                    
                except asyncio.CancelledError:
                    self.logger.info("Main trading loop has been cancelled.")
                    break
                except Exception as e:
                    self.logger.error(f"An error occurred in the main trading loop: {e}", exc_info=True)
                    await self._send_notification(f"⚠️ Trading Loop Error: {str(e)}")
                    await asyncio.sleep(30)  # Brief pause after an error
                    
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Shutting down.")
        except Exception as e:
            self.logger.critical(f"A critical error occurred in the run method: {e}", exc_info=True)
            await self._send_notification(f"🚨 Critical Error: {str(e)}")
        finally:
            await self.shutdown()

    async def _should_trade(self) -> bool:
        """Checks global conditions (connection, session, risk) to see if bot should trade."""
        try:
            if not self.mt5_handler.is_connected():
                self.logger.warning("MT5 not connected. Attempting to reconnect...")
                if not await self.mt5_handler.reconnect():
                    return False
            
            current_hour = datetime.now().hour
            is_active, _ = Config.is_trading_session_active(current_hour)
            if not is_active:
                self.logger.info("Outside of active trading sessions. Pausing.")
                return False
            
            if not self.risk_manager.check_daily_risk_limit(
                trades_today=self.daily_trades,
                max_daily_risk=Config.MAX_DAILY_RISK
            ):
                self.logger.warning("Daily risk limit has been reached. Stopping trades for the day.")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in _should_trade check: {e}", exc_info=True)
            return False

    async def _process_trading_cycle(self):
        """Processes one full trading cycle across all configured symbols."""
        symbols_to_trade = list(Config.SYMBOLS.keys()) if Config.MULTI_SYMBOL_MODE else [Config.DEFAULT_SYMBOL]
        
        for symbol in symbols_to_trade:
            await self._process_symbol(symbol)

    async def _process_symbol(self, symbol: str):
        """Processes the trading logic for a single symbol."""
        try:
            if not await self.mt5_handler.check_market_conditions(symbol):
                self.logger.info(f"Market for {symbol} is closed or not tradable. Skipping.")
                return

            if not self._check_signal_cooldown(symbol):
                return
            
            market_data = await self.mt5_handler.get_market_data(symbol)
            if not market_data:
                self.logger.warning(f"Could not retrieve market data for {symbol}.")
                return

            df = pd.DataFrame(market_data)
            dfs = {Config.PRIMARY_TIMEFRAME: df}
            
            signal = self.signal_generator.generate_signal(dfs, Config.PRIMARY_TIMEFRAME)
            
            if signal:
                self.logger.info(f"Signal generated for {symbol}: {signal.direction} with confidence {signal.confidence:.2f}")
                if await self._validate_and_execute_signal(signal):
                    self.last_signal_time[symbol] = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)

    def _check_signal_cooldown(self, symbol: str) -> bool:
        """Checks if the cooldown period for a symbol has passed."""
        last_time = self.last_signal_time.get(symbol)
        if not last_time:
            return True
        
        time_since = (datetime.now() - last_time).total_seconds()
        return time_since >= Config.SIGNAL_THRESHOLDS["signal_cooldown"]

    async def _validate_and_execute_signal(self, signal: TradeSignal) -> bool:
        """Validates a signal against risk rules and executes it if valid."""
        try:
            if not await self.risk_manager.validate_signal(signal):
                self.logger.info(f"Signal for {signal.symbol} rejected by risk manager.")
                return False
            
            execution_result = await self.mt5_handler.execute_trade(signal)
            
            if execution_result and execution_result.get("success"):
                self.daily_trades += 1
                await self._send_notification(
                    f"✅ Trade Executed: {signal.symbol} {signal.direction} "
                    f"@ {signal.entry_price} (Confidence: {signal.confidence:.2f})"
                )
                self.logger.info(f"Trade executed successfully. Ticket: {execution_result.get('ticket')}")
                return True
            else:
                error_msg = execution_result.get('error', 'Unknown error')
                await self._send_notification(
                    f"❌ Trade Execution Failed: {signal.symbol} - {error_msg}"
                )
                self.logger.error(f"Trade execution failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating/executing signal for {signal.symbol}: {e}", exc_info=True)
            return False

    async def _send_notification(self, message: str):
        """Sends a notification via Telegram if configured."""
        try:
            if self.telegram_notifier and Config.MONITORING["telegram_alerts"]:
                await self.telegram_notifier.send_message(message)
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}", exc_info=True)

    async def get_performance_summary(self) -> dict:
        """Gathers current performance and state metrics for status updates."""
        try:
            account_info = await self.mt5_handler.get_account_info()
            uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            
            return {
                "uptime_hours": round(uptime_hours, 2),
                "daily_trades": self.daily_trades,
                "account_balance": account_info.get("balance", 0),
                "account_equity": account_info.get("equity", 0),
                "open_positions": len(await self.mt5_handler.get_positions()),
                "mt5_connected": self.mt5_handler.is_connected(),
                "trading_mode": Config.CURRENT_MODE.value
            }
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}", exc_info=True)
            return {}

    async def shutdown(self):
        """Performs a graceful shutdown of the bot."""
        if not self.is_running:
            return
            
        try:
            self.logger.info("Shutting down trading bot...")
            self.is_running = False
            
            await self._send_notification("🛑 Trading Bot Shutting Down")
            
            if self.mt5_handler:
                await self.mt5_handler.disconnect()
            
            summary = await self.get_performance_summary()
            summary_text = (
                f"📊 Final Performance Summary:\n"
                f"Uptime: {summary.get('uptime_hours', 0):.2f}h | "
                f"Trades: {summary.get('daily_trades', 0)} | "
                f"Balance: ${summary.get('account_balance', 0):.2f}"
            )
            await self._send_notification(summary_text)
            
            self.logger.info("Trading bot shutdown completed.")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)

# Main execution block
async def main():
    """The main entry point for the trading bot application."""
    bot = TradingBot()
    
    if await bot.initialize():
        await bot.run()
    else:
        print("Failed to initialize trading bot. Please check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        sys.exit(1)
