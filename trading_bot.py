"""
Trading Bot - Final Version
"""
import asyncio
import sys
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5
from config import Config
from components.mt5_handler import MT5Handler
from strategies.signal_generator import SignalGenerator
from strategies.risk_management import RiskManagement
from strategies.market_structure import MarketStructure
from strategies.volume_analysis import VolumeAnalysis
from strategies.pattern_analysis import PatternAnalysis
from strategies.session_analysis import SessionAnalysis
from utils.logger import TradingLogger
from utils.helpers import TelegramNotifier, TradeSignal

class TradingBot:

    def __init__(self):
        self.config = Config()
        self.logger = TradingLogger("TradingBot")
        
        self.mt5_handler = None
        self.signal_generator = None
        self.risk_manager = None
        self.session_analyzer = None
        self.telegram_notifier = None
        
        self.is_running = False
        self.is_initialized = False
        self.last_signal_time = {}
        self.daily_trades = 0

    def _safe_dataframe_preparation(self, market_data, symbol):
        """Safely prepares a DataFrame for analysis."""
        try:
            df = pd.DataFrame(market_data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            
            numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna().sort_index()
            
            if len(df) < 200: # Ensure enough data for all indicator periods
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows, need 200.")
                return None
            return df
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame for {symbol}: {e}", exc_info=True)
            return None

    async def initialize(self) -> bool:
        """Initializes all bot components."""
        try:
            self.logger.info("Starting bot initialization...")
            
            # *** FIX: Removed call to the non-existent Config.validate_config() ***
            
            self.mt5_handler = MT5Handler()
            if not await self.mt5_handler.initialize():
                self.logger.critical("Failed to initialize MT5 handler. Exiting.")
                return False
            
            account_info = await self.mt5_handler.get_account_info()
            account_balance = account_info.get("balance", 0.0)
            
            # Initialize all components
            self.risk_manager = RiskManagement(
                account_balance=account_balance,
                mt5_handler=self.mt5_handler
            )
            market_structure = MarketStructure()
            volume_analysis = VolumeAnalysis()
            pattern_analysis = PatternAnalysis()
            self.session_analyzer = SessionAnalysis()
            
            self.signal_generator = SignalGenerator(
                market_structure=market_structure,
                volume_analysis=volume_analysis,
                pattern_analysis=pattern_analysis,
                session_analysis=self.session_analyzer,
                risk_management=self.risk_manager
            )

            if self.config.TELEGRAM_TOKEN and self.config.TELEGRAM_CHAT_ID:
                self.telegram_notifier = TelegramNotifier(
                    bot_token=self.config.TELEGRAM_TOKEN,
                    chat_id=self.config.TELEGRAM_CHAT_ID
                )
                await self.telegram_notifier.initialize()
            
            self.is_initialized = True
            self.logger.info("Trading bot initialization completed.")
            return True
            
        except Exception as e:
            self.logger.critical(f"Initialization failed: {e}", exc_info=True)
            return False

    async def run(self):
        """Main bot execution loop."""
        if not self.is_initialized:
            self.logger.error("Bot not initialized. Aborting run.")
            return
        
        self.is_running = True
        await self._send_notification("🚀 Trading Bot Started & Monitoring Markets...")
        
        while self.is_running:
            try:
                if await self._should_trade():
                    await self._process_trading_cycle()
                
                await asyncio.sleep(self.config.SIGNAL_THRESHOLDS["signal_cooldown"])
                
            except asyncio.CancelledError:
                self.logger.info("Trading loop gracefully cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Critical error in trading loop: {e}", exc_info=True)
                await self._send_notification(f"⚠️ Critical Loop Error: {str(e)}. Pausing for 60s.")
                await asyncio.sleep(60)
        
        await self.shutdown()

    async def _should_trade(self) -> bool:
        """Checks if all conditions for trading are met."""
        if not self.mt5_handler.is_connected():
            self.logger.warning("MT5 disconnected. Attempting to reconnect...")
            return await self.mt5_handler.reconnect()
        
        current_hour = datetime.now().hour
        is_active, _ = self.config.is_trading_session_active(current_hour)
        if not is_active:
            return False
        
        if self.daily_trades >= self.config.MAX_DAILY_RISK:
            self.logger.warning(f"Max daily trades ({self.config.MAX_DAILY_RISK}) reached. Pausing until next day.")
            return False
        
        return True

    async def _process_trading_cycle(self):
        """Processes all configured symbols."""
        symbols = list(self.config.SYMBOLS.keys()) if self.config.MULTI_SYMBOL_MODE else [self.config.DEFAULT_SYMBOL]
        tasks = [self._process_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)

    async def _process_symbol(self, symbol: str):
        """Fetches data and generates signals for a single symbol."""
        try:
            if not await self.mt5_handler.check_market_conditions(symbol):
                return

            if not self._check_signal_cooldown(symbol):
                return
            
            timeframes = self.config.MTF_TIMEFRAMES
            dfs = {}
            for tf_str in timeframes:
                tf_id = getattr(mt5, f'TIMEFRAME_{tf_str}')
                market_data = await self.mt5_handler.get_market_data(symbol, timeframe=tf_id, count=200)
                if not market_data:
                    self.logger.warning(f"Could not fetch market data for {symbol} on {tf_str}.")
                    return
                
                df = self._safe_dataframe_preparation(market_data, symbol)
                if df is None: return
                dfs[tf_str] = df
            
            if len(dfs) != len(timeframes): return

            signal = await self.signal_generator.generate_signal(dfs, symbol)
            
            if signal and await self._validate_and_execute_signal(signal):
                self.last_signal_time[symbol] = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)

    def _check_signal_cooldown(self, symbol: str) -> bool:
        """Ensures the bot does not spam trades on the same symbol."""
        last_time = self.last_signal_time.get(symbol)
        if not last_time: return True
        return (datetime.now() - last_time).total_seconds() >= self.config.SIGNAL_THRESHOLDS["signal_cooldown"]

    async def _validate_and_execute_signal(self, signal: TradeSignal) -> bool:
        """Validates risk and executes a trade signal."""
        try:
            execution_result = await self.mt5_handler.execute_trade(signal)
            
            if execution_result.get('success'):
                self.daily_trades += 1
                self.logger.info(f"Trade executed successfully: {execution_result}")
                await self._send_notification(f"✅ TRADE EXECUTED: {signal.direction} {signal.symbol} @ {signal.entry_price:.4f}")
                return True
            else:
                error_msg = execution_result.get('error', 'Unknown execution error')
                self.logger.error(f"Trade execution failed: {error_msg}")
                await self._send_notification(f"❌ TRADE FAILED: {signal.symbol} - {error_msg}")
                return False
        except Exception as e:
            self.logger.error(f"Error validating/executing signal: {e}", exc_info=True)
            return False

    async def _send_notification(self, message: str):
        """Sends a message to the configured Telegram chat."""
        if self.telegram_notifier:
            await self.telegram_notifier.send_message(message)

    async def shutdown(self):
        """Gracefully shuts down the bot."""
        self.logger.info("Shutting down trading bot...")
        self.is_running = False
        await self._send_notification("🛑 Trading Bot Shutting Down.")
        if self.mt5_handler:
            await self.mt5_handler.disconnect()
        self.logger.info("Trading bot shutdown completed.")

async def main():
    """Main entry point for the bot application."""
    bot = TradingBot()
    if await bot.initialize():
        await bot.run()
    else:
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nBot stopped by user.")