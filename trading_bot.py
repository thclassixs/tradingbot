"""
Trading Bot - Debug and Fix for All Identified Issues
"""
import asyncio
import signal
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
        
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}, shutting down gracefully...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _safe_dataframe_preparation(self, market_data, symbol):
        df = pd.DataFrame(market_data)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna().sort_index()
        
        if len(df) < 50: # Increased minimum for better analysis
            self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
            return None
        return df

    async def initialize(self) -> bool:
        try:
            self.logger.info("Starting bot initialization...")
            
            Config.validate_config()
            self.logger.info("Configuration validated successfully")
            
            self.mt5_handler = MT5Handler()
            if not await self.mt5_handler.initialize():
                self.logger.error("Failed to initialize MT5 handler")
                return False
            
            account_info = await self.mt5_handler.get_account_info()
            account_balance = account_info.get("balance", 0.0)
            
            self.risk_manager = RiskManagement(
                account_balance=account_balance,
                mt5_handler=self.mt5_handler, 
                max_risk_percent=Config.MAX_RISK_PERCENT
            )
            
            self.market_structure = MarketStructure()
            self.volume_analyzer = VolumeAnalysis()
            self.pattern_analyzer = PatternAnalysis()
            self.session_analyzer = SessionAnalysis()
            
            self.signal_generator = SignalGenerator(
                market_structure=self.market_structure,
                volume_analysis=self.volume_analyzer,
                pattern_analysis=self.pattern_analyzer,
                session_analysis=self.session_analyzer,
                risk_management=self.risk_manager
            )

            if Config.MONITORING["telegram_alerts"]:
                self.telegram_notifier = TelegramNotifier(
                    bot_token=Config.TELEGRAM_TOKEN,
                    chat_id=Config.TELEGRAM_CHAT_ID
                )
                await self.telegram_notifier.initialize()
            
            self.is_initialized = True
            self.start_time = datetime.now()
            self.logger.info("Trading bot initialization completed")
            return True
            
        except Exception as e:
            self.logger.critical(f"Initialization failed: {e}", exc_info=True)
            await self._send_notification(f"❌ Bot Initialization Failed: {str(e)}")
            return False

    async def run(self):
        if not self.is_initialized:
            self.logger.error("Bot not initialized. Call initialize() first.")
            return
        
        self.is_running = True
        self.logger.info("🚀 Trading Bot Started")
        
        while self.is_running:
            try:
                if await self._should_trade():
                    await self._process_trading_cycle()
                
                await asyncio.sleep(Config.SIGNAL_THRESHOLDS["signal_cooldown"])
                
            except asyncio.CancelledError:
                self.logger.info("Trading loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}", exc_info=True)
                await self._send_notification(f"⚠️ Trading Loop Error: {str(e)}")
                await asyncio.sleep(30)
        
        await self.shutdown()

    async def _should_trade(self) -> bool:
        if not self.mt5_handler.is_connected():
            self.logger.warning("MT5 not connected, attempting reconnection...")
            if not await self.mt5_handler.reconnect():
                return False
        
        current_hour = datetime.now().hour
        is_active, _ = Config.is_trading_session_active(current_hour)
        if not is_active:
            return False
        
        # *** FIX: Corrected the keyword argument to match the function definition ***
        if not self.risk_manager.check_daily_risk_limit(
            trades_today=self.daily_trades, 
            max_daily_risk=Config.MAX_DAILY_RISK
        ):
            self.logger.warning("Daily risk limits reached. Pausing trading for today.")
            return False
        
        return await self.mt5_handler.check_market_conditions()

    async def _process_trading_cycle(self):
        symbols = list(Config.SYMBOLS.keys()) if Config.MULTI_SYMBOL_MODE else [Config.DEFAULT_SYMBOL]
        tasks = [self._process_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)

    async def _process_symbol(self, symbol: str):
        try:
            if not await self.mt5_handler.check_market_conditions(symbol):
                return

            if not self._check_signal_cooldown(symbol):
                return
            
            timeframes = ['H1', 'M15', 'M5']
            dfs = {}
            for tf_str in timeframes:
                tf_id = getattr(mt5, f'TIMEFRAME_{tf_str}')
                market_data = await self.mt5_handler.get_market_data(symbol, timeframe=tf_id, count=500)
                if not market_data:
                    self.logger.warning(f"No market data for {symbol} on {tf_str}")
                    return
                
                df = self._safe_dataframe_preparation(market_data, symbol)
                if df is None: return
                dfs[tf_str] = df
            
            if len(dfs) != len(timeframes): return # Ensure all data was fetched

            signal = await self.signal_generator.generate_signal(dfs, symbol)
            
            if signal and await self._validate_and_execute_signal(signal):
                self.last_signal_time[symbol] = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)

    def _check_signal_cooldown(self, symbol: str) -> bool:
        last_time = self.last_signal_time.get(symbol)
        if not last_time: return True
        return (datetime.now() - last_time).total_seconds() >= Config.SIGNAL_THRESHOLDS["signal_cooldown"]

    async def _validate_and_execute_signal(self, signal: TradeSignal) -> bool:
        try:
            if not await self.risk_manager.validate_signal(signal):
                self.logger.warning(f"Signal rejected by risk manager: {signal.symbol} - {signal.reasons}")
                return False
            
            execution_result = await self.mt5_handler.execute_trade(signal)
            
            if execution_result.get('success'):
                self.daily_trades += 1
                self.logger.info(f"Trade executed successfully: {execution_result}")
                if self.telegram_notifier:
                    await self.telegram_notifier.send_trade_alert(signal)
                return True
            else:
                error_msg = execution_result.get('error', 'Unknown execution error')
                self.logger.error(f"Trade execution failed: {error_msg}")
                await self._send_notification(f"❌ Trade Failed: {signal.symbol} - {error_msg}")
                return False
        except Exception as e:
            self.logger.error(f"Error validating/executing signal: {e}", exc_info=True)
            return False

    async def _send_notification(self, message: str):
        if self.telegram_notifier:
            await self.telegram_notifier.send_message(message)

    async def shutdown(self):
        self.logger.info("Shutting down trading bot...")
        self.is_running = False
        await self._send_notification("🛑 Trading Bot Shutting Down")
        if self.mt5_handler:
            await self.mt5_handler.disconnect()
        self.logger.info("Trading bot shutdown completed.")

async def main():
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