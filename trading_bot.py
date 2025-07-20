"""
Main Trading Bot
"""
import asyncio
import signal
import sys
from datetime import datetime
from config import Config
from components.mt5_handler import MT5Handler
from strategies.signal_generator import SignalGenerator
from strategies.signal_generator import TradeSignal
from strategies.risk_management import RiskManagement
from strategies.market_structure import MarketStructure
from strategies.volume_analysis import VolumeAnalysis
from strategies.pattern_analysis import PatternAnalysis
from strategies.session_analysis import SessionAnalysis
from utils.logger import TradingLogger
from utils.helpers import TelegramNotifier
import pandas as pd
class TradingBot:

    def __init__(self):
        self.config = Config()
        self.logger = TradingLogger("TradingBot")
        
        # Core components
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
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}, shutting down gracefully...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize(self) -> bool:

        try:
            self.logger.info("Starting bot initialization...")
            
            # Validate configuration
            Config.validate_config()
            self.logger.info("Configuration validated successfully")
            
            # Initialize core components
            self.mt5_handler = MT5Handler()
            if not await self.mt5_handler.initialize():
                self.logger.error("Failed to initialize MT5 handler")
                return False
            
            # Initialize analysis modules
            self.market_structure = MarketStructure()
            self.volume_analyzer = VolumeAnalysis()
            self.pattern_analyzer = PatternAnalysis()
            self.session_analyzer = SessionAnalysis()
            
            # Initialize signal generator with all analyzers
            self.signal_generator = SignalGenerator(
                market_structure = self.market_structure,
                volume_analysis = self.volume_analyzer,
                pattern_analysis = self.pattern_analyzer,
                session_analysis = self.session_analyzer
            )
            
            # Initialize risk manager
            account_info = await self.mt5_handler.get_account_info() # Get account info
            account_balance = account_info.get("balance", 0.0) # Extract balance
            self.risk_manager = RiskManagement(account_balance=account_balance, max_risk_percent=Config.MAX_RISK_PERCENT)
                        
            # Initialize telegram notifier
            if Config.MONITORING["telegram_alerts"]:
                self.telegram_notifier = TelegramNotifier(
                    bot_token=Config.TELEGRAM_TOKEN,
                    chat_id=Config.TELEGRAM_CHAT_ID
                )
                await self.telegram_notifier.initialize()
            
            # Initialize all components
            await self._initialize_components()
            
            # Create necessary directories
            import os
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            os.makedirs(Config.LOGS_DIR, exist_ok=True)
            
            self.is_initialized = True
            self.start_time = datetime.now()
            
            await self._send_notification("🤖 Trading Bot Initialized Successfully")
            self.logger.info("Trading bot initialization completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            print(f"Initialization failed: {e}")  # Add this line for direct console output
            await self._send_notification(f"❌ Bot Initialization Failed: {str(e)}")
            return False

    async def _initialize_components(self):
        """Initialize all analysis components"""
        components = [
            self.market_structure,
            self.volume_analyzer,
            self.pattern_analyzer,
            self.session_analyzer,
            self.signal_generator,
            self.risk_manager
        ]
        
        for component in components:
            if hasattr(component, 'initialize'):
                await component.initialize()
                self.logger.info(f"Initialized {component.__class__.__name__}")

    async def run(self):
        """Main bot execution loop"""
        if not self.is_initialized:
            self.logger.error("Bot not initialized. Call initialize() first.")
            return
        
        try:
            self.is_running = True
            self.logger.info("🚀 Trading Bot Started")
            await self._send_notification("🚀 Trading Bot Started")
            
            while self.is_running:
                try:
                    # Check if trading is allowed
                    if not await self._should_trade():
                        await asyncio.sleep(60)  # Check every minute
                        continue
                    
                    # Process trading cycle
                    await self._process_trading_cycle()
                    
                    # Sleep for configured cooldown
                    await asyncio.sleep(Config.SIGNAL_THRESHOLDS["signal_cooldown"])
                    
                except asyncio.CancelledError:
                    self.logger.info("Trading loop cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    await self._send_notification(f"⚠️ Trading Loop Error: {str(e)}")
                    await asyncio.sleep(30)  # Brief pause on error
                    
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.critical(f"Critical error in main loop: {e}")
            await self._send_notification(f"🚨 Critical Error: {str(e)}")
        finally:
            await self.shutdown()

    async def _should_trade(self) -> bool:
        """Check if trading conditions are met"""
        try:
            # Check if MT5 is connected
            if not self.mt5_handler.is_connected():
                self.logger.warning("MT5 not connected, attempting reconnection...")
                if not await self.mt5_handler.reconnect():
                    return False
            
            # Check trading sessions
            current_hour = datetime.now().hour
            is_active, session = Config.is_trading_session_active(current_hour)
            
            if not is_active:
                return False
            
            # Check risk limits
            if not self.risk_manager.check_daily_risk_limit(
                trades_today=self.daily_trades,
                max_daily_risk=Config.MAX_DAILY_RISK
            ):
                self.logger.warning("Daily risk limits reached")
                return False
            
            # Check market conditions
            if not await self.mt5_handler.check_market_conditions():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking trading conditions: {e}")
            return False

    async def _process_trading_cycle(self):
        """Process one complete trading cycle"""
        try:
            symbols = list(Config.SYMBOLS.keys()) if Config.MULTI_SYMBOL_MODE else [Config.DEFAULT_SYMBOL]
            
            for symbol in symbols:
                await self._process_symbol(symbol)
                
        except Exception as e:
            self.logger.error(f"Error processing trading cycle: {e}")
            raise

    async def _process_symbol(self, symbol: str):
        """Process trading for a specific symbol"""
        try:
            # Check signal cooldown
            if not self._check_signal_cooldown(symbol):
                return
            
            # Get market data
            market_data = await self.mt5_handler.get_market_data(symbol)
            if not market_data:
                self.logger.warning(f"No market data for {symbol}")
                return

            # Generate signals
            df = pd.DataFrame(market_data)
            signals = await self.signal_generator.generate_signals(symbol, df)
            if not signals:
                return
            
            # Process each signal
            for signal in signals:
                if await self._validate_and_execute_signal(signal):
                    self.last_signal_time[symbol] = datetime.now()
                    break  # Execute only one signal per symbol per cycle
                    
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")

    def _check_signal_cooldown(self, symbol: str) -> bool:
        """Check if signal cooldown period has passed"""
        if symbol not in self.last_signal_time:
            return True
        
        time_since_last = (datetime.now() - self.last_signal_time[symbol]).total_seconds()
        return time_since_last >= Config.SIGNAL_THRESHOLDS["signal_cooldown"]

    async def _validate_and_execute_signal(self, signal) -> bool:
        """Validate and execute a trading signal"""
        try:
            # Risk validation
            if not await self.risk_manager.validate_signal(signal):
                self.logger.info(f"Signal rejected by risk manager: {signal.symbol}")
                return False
            
            # Execute trade
            execution_result = await self.mt5_handler.execute_trade(signal)
            
            if execution_result.success:
                self.daily_trades += 1
                await self._send_notification(
                    f"✅ Trade Executed: {signal.symbol} {signal.direction} "
                    f"@ {signal.entry_price} (Confidence: {signal.confidence:.2f})"
                )
                self.logger.info(f"Trade executed successfully: {execution_result.ticket}")
                return True
            else:
                await self._send_notification(
                    f"❌ Trade Execution Failed: {signal.symbol} - {execution_result.error}"
                )
                self.logger.error(f"Trade execution failed: {execution_result.error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating/executing signal: {e}")
            return False

    async def _send_notification(self, message: str):
        """Send notification via configured channels"""
        try:
            if self.telegram_notifier and Config.MONITORING["telegram_alerts"]:
                await self.telegram_notifier.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")

    async def get_performance_summary(self) -> dict:
        """Get current performance summary"""
        try:
            account_info = await self.mt5_handler.get_account_info()
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            
            return {
                "uptime_hours": round(uptime, 2),
                "daily_trades": self.daily_trades,
                "account_balance": account_info.get("balance", 0),
                "account_equity": account_info.get("equity", 0),
                "free_margin": account_info.get("margin_free", 0),
                "open_positions": len(await self.mt5_handler.get_positions()),
                "is_connected": self.mt5_handler.is_connected(),
                "trading_mode": Config.CURRENT_MODE.value
            }
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

    async def shutdown(self):
        """Graceful shutdown of the trading bot"""
        try:
            self.logger.info("Shutting down trading bot...")
            self.is_running = False
            
            # Send shutdown notification
            await self._send_notification("🛑 Trading Bot Shutting Down")
            
            # Close all positions if configured
            if Config.DEBUG.get("close_positions_on_shutdown", False):
                await self.mt5_handler.close_all_positions()
            
            # Disconnect MT5
            if self.mt5_handler:
                await self.mt5_handler.disconnect()
            
            # Final performance summary
            summary = await self.get_performance_summary()
            summary_text = (
                f"📊 Final Performance Summary:\n"
                f"Uptime: {summary.get('uptime_hours', 0):.2f}h\n"
                f"Daily Trades: {summary.get('daily_trades', 0)}\n"
                f"Account Balance: ${summary.get('account_balance', 0):.2f}"
            )
            await self._send_notification(summary_text)
            
            self.logger.info("Trading bot shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def manual_trade(self, symbol: str, direction: str, lot_size: float = None):
        """Execute a manual trade (for testing/emergency)"""
        try:
            if not self.is_initialized:
                return {"success": False, "error": "Bot not initialized"}

            market_data = await self.mt5_handler.get_market_data(symbol)
            current_price = market_data["close"][-1]
            
            signal = TradeSignal(
                symbol=symbol,
                direction=direction.upper(),
                confidence=1.0,
                entry_price=current_price,
                stop_loss=current_price * 0.99 if direction.upper() == "BUY" else current_price * 1.01,
                take_profit=current_price * 1.02 if direction.upper() == "BUY" else current_price * 0.98,
                timeframe="MANUAL",
                reasons=["Manual override"]
            )
            
            return await self._validate_and_execute_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error in manual trade: {e}")
            return {"success": False, "error": str(e)}

    async def health_check(self):
        """Quick health check"""
        return {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "mt5_connected": self.mt5_handler.is_connected(),
            "current_time": datetime.now().isoformat(),
            "should_trade": await self._should_trade(),
            "active_symbols": list(Config.SYMBOLS.keys()) if Config.MULTI_SYMBOL_MODE else [Config.DEFAULT_SYMBOL]
        }

# Main execution
async def main():
    """Main entry point"""
    bot = TradingBot()
    
    if await bot.initialize():
        await bot.run()
    else:
        print("Failed to initialize trading bot")
        sys.exit(1)

if __name__ == "__main__":
    # Create event loop and run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)