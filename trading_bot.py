import asyncio
import signal
import sys , os
from datetime import datetime
import pandas as pd
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
        self.hedged_positions = set()
        self.win_streak = 0

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

    def _safe_dataframe_preparation(self, market_data, symbol):
        """Safely prepare DataFrame with proper data types and error handling"""
        try:
            df = pd.DataFrame(market_data)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.set_index('time')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            numeric_columns = ['open', 'high', 'low', 'close', 'tick_volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'volume' not in df.columns and 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']

            df = df.dropna()
            df = df.sort_index()

            if len(df) < 20:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                return None
            return df
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame for {symbol}: {e}")
            return None

    def _fix_pandas_interval_issues(self, df):
        """Fix common pandas interval comparison issues"""
        try:
            for col in df.columns:
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    if isinstance(df[col].cat.categories, pd.IntervalIndex):
                        df[col] = df[col].apply(lambda x: x.mid if pd.notna(x) else x)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            self.logger.error(f"Error fixing pandas intervals: {e}")
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
                risk_management=self.risk_manager,
                htf_timeframe=Config.MTF_TIMEFRAMES[-1] if Config.MTF_TIMEFRAMES else 'H1'
            )

            if Config.MONITORING["telegram_alerts"]:
                self.telegram_notifier = TelegramNotifier(
                    bot_token=Config.TELEGRAM_TOKEN,
                    chat_id=Config.TELEGRAM_CHAT_ID
                )
                await self.telegram_notifier.initialize()

            os.makedirs(Config.DATA_DIR, exist_ok=True)
            os.makedirs(Config.LOGS_DIR, exist_ok=True)

            self.is_initialized = True
            self.start_time = datetime.now()
            self.logger.info("Trading bot initialization completed")
            await self._send_notification("✅ **Trading Bot Initialized and Running**")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            print(f"Initialization failed: {e}")
            await self._send_notification(f"❌ Bot Initialization Failed: {str(e)}")
            return False

    async def run(self):
        """Main bot execution loop"""
        if not self.is_initialized:
            self.logger.error("Bot not initialized. Call initialize() first.")
            return

        try:
            self.is_running = True
            self.logger.info("🚀 Trading Bot Started")

            while self.is_running:
                try:
                    if not await self._should_trade():
                        await asyncio.sleep(60)
                        continue

                    # --- NEW: Concurrently manage open trades ---
                    await asyncio.gather(
                        self._process_trading_cycle(),
                        self._manage_open_trades()
                    )

                    await asyncio.sleep(Config.SIGNAL_THRESHOLDS["signal_cooldown"])
                except asyncio.CancelledError:
                    self.logger.info("Trading loop cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    await self._send_notification(f"⚠️ **Trading Loop Error:**\n`{str(e)}`")
                    await asyncio.sleep(30)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            await self.shutdown()

    async def _should_trade(self) -> bool:
        """Check if trading conditions are met"""
        try:
            if not self.mt5_handler.is_connected():
                self.logger.warning("MT5 not connected, attempting reconnection...")
                if not await self.mt5_handler.reconnect():
                    return False

            current_hour = datetime.now().hour
            is_active, _ = Config.is_trading_session_active(current_hour)
            if not is_active:
                self.logger.info("Outside of active trading sessions. Pausing.")
                return False
            
            if self.daily_trades >= Config.CAPITAL_CONTROLS["max_daily_trades"]:
                 self.logger.warning("Daily trade limit reached. Pausing until next session.")
                 return False

            if not await self.mt5_handler.check_market_conditions():
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking trading conditions: {e}")
            return False

    async def _process_trading_cycle(self):
        """Process one complete trading cycle"""
        symbols = list(Config.SYMBOLS.keys())
        tasks = [self._process_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)

    async def _process_symbol(self, symbol: str):
        """Process trading for a specific symbol"""
        try:
            if not await self.mt5_handler.check_market_conditions(symbol):
                return

            if not self._check_signal_cooldown(symbol):
                return

            dfs = {}
            for tf_name in Config.MTF_TIMEFRAMES:
                tf_code = Config.TIMEFRAMES.get(tf_name)
                if tf_code:
                    market_data = await self.mt5_handler.get_market_data(symbol, timeframe=tf_code)
                    if market_data:
                        df = self._safe_dataframe_preparation(market_data, symbol)
                        if df is not None:
                            dfs[tf_name] = self._fix_pandas_interval_issues(df)

            if Config.PRIMARY_TIMEFRAME not in dfs:
                self.logger.warning(f"Could not fetch primary timeframe data for {symbol}")
                return

            signal = await self.signal_generator.generate_signal(dfs, Config.PRIMARY_TIMEFRAME, symbol)

            if not signal:
                return

            if await self._validate_and_execute_signal(signal):
                self.last_signal_time[symbol] = datetime.now()
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

    def _check_signal_cooldown(self, symbol: str) -> bool:
        """Check if signal cooldown period has passed"""
        if symbol not in self.last_signal_time:
            return True
        time_since_last = (datetime.now() - self.last_signal_time[symbol]).total_seconds()
        return time_since_last >= Config.SIGNAL_THRESHOLDS["signal_cooldown"]

    async def _validate_and_execute_signal(self, signal: TradeSignal) -> bool:
        """Validate and execute a trading signal"""
        try:
            if not await self.risk_manager.validate_signal(signal):
                self.logger.info(f"Signal rejected by risk manager: {signal.symbol}")
                return False

            execution_result = await self.mt5_handler.execute_trade(signal, win_streak=self.win_streak)

            if execution_result.get('success'):
                self.daily_trades += 1
                self.logger.info(f"Trade executed successfully: {execution_result.get('ticket')}")
                if self.telegram_notifier:
                    await self.telegram_notifier.send_trade_alert(signal)
                
                # --- Win Streak Logic ---
                if signal.profit > 0:
                    self.win_streak += 1
                elif self.config.AUTO_LOT_BOOST["reset_on_loss"]:
                    self.win_streak = 0
                return True
            else:
                error_msg = execution_result.get('error', 'Unknown execution error')
                await self._send_notification(f"❌ **Trade Execution Failed:**\n`{signal.symbol} - {error_msg}`")
                self.logger.error(f"Trade execution failed: {error_msg}")
                return False
        except Exception as e:
            self.logger.error(f"Error validating/executing signal: {e}")
            return False

    async def _manage_open_trades(self):
        """Manage all open positions for breakeven, trailing, and hedging."""
        if not any([self.config.BREAKEVEN_TRAILING["enabled"], self.config.HEDGE_LOGIC["enabled"]]):
            return

        open_positions = await self.mt5_handler.get_positions()
        for pos in open_positions:
            if self.config.BREAKEVEN_TRAILING["enabled"]:
                await self._apply_breakeven_trailing(pos)
            
            if self.config.HEDGE_LOGIC["enabled"] and pos['ticket'] not in self.hedged_positions:
                await self._apply_hedge_logic(pos)

    async def _apply_breakeven_trailing(self, position: dict):
        """Apply breakeven and trailing stop loss logic."""
        pips_in_profit = 0
        point = await self.mt5_handler.get_symbol_info(position['symbol'])['point']
        
        if position['type'] == 'BUY':
            pips_in_profit = (position['price_current'] - position['price_open']) / point
        else: # SELL
            pips_in_profit = (position['price_open'] - position['price_current']) / point

        # Move to Breakeven
        if pips_in_profit >= self.config.BREAKEVEN_TRAILING["breakeven_pips"] and position['sl'] != position['price_open']:
            await self.mt5_handler.modify_position(position['ticket'], new_sl=position['price_open'])
            await self._send_notification(f"🔒 Moved SL to Breakeven for trade {position['ticket']}.")

        # Trailing Stop
        if pips_in_profit >= self.config.BREAKEVEN_TRAILING["breakeven_pips"] + self.config.BREAKEVEN_TRAILING["trailing_step_pips"]:
            new_sl = 0
            if position['type'] == 'BUY':
                new_sl = position['price_current'] - (self.config.BREAKEVEN_TRAILING["trailing_step_pips"] * point)
                if new_sl > position['sl']:
                    await self.mt5_handler.modify_position(position['ticket'], new_sl=new_sl)
            else: # SELL
                new_sl = position['price_current'] + (self.config.BREAKEVEN_TRAILING["trailing_step_pips"] * point)
                if new_sl < position['sl']:
                    await self.mt5_handler.modify_position(position['ticket'], new_sl=new_sl)

    async def _apply_hedge_logic(self, position: dict):
        """Apply hedging logic if a trade is in significant drawdown."""
        pips_in_drawdown = 0
        point = await self.mt5_handler.get_symbol_info(position['symbol'])['point']

        if position['type'] == 'BUY':
            pips_in_drawdown = (position['price_open'] - position['price_current']) / point
        else: # SELL
            pips_in_drawdown = (position['price_current'] - position['price_open']) / point

        if pips_in_drawdown >= self.config.HEDGE_LOGIC["drawdown_pips_threshold"]:
            hedge_direction = "SELL" if position['type'] == 'BUY' else "BUY"
            hedge_signal = TradeSignal(
                symbol=position['symbol'],
                direction=hedge_direction,
                confidence=1.0, # Hedge is a forced action
                entry_price=position['price_current'],
                stop_loss=0, # No SL/TP for hedge trades initially
                take_profit=0,
                timeframe="HEDGE",
                reasons=[f"Hedging position {position['ticket']} due to significant drawdown."]
            )
            
            hedge_result = await self.mt5_handler.execute_trade(hedge_signal)
            if hedge_result.get('success'):
                self.hedged_positions.add(position['ticket'])
                self.hedged_positions.add(hedge_result['ticket']) # Add the new hedge trade as well
                await self._send_notification(f"🛡️ Hedged position {position['ticket']} with new trade {hedge_result['ticket']}.")

    async def _send_notification(self, message: str):
        """Send notification via configured channels"""
        if self.telegram_notifier:
            try:
                await self.telegram_notifier.send_message(message, parse_mode="Markdown")
            except Exception as e:
                self.logger.error(f"Failed to send Telegram notification: {e}")

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
                "open_positions": len(await self.mt5_handler.get_positions()),
                "is_connected": self.mt5_handler.is_connected(),
                "trading_mode": Config.CURRENT_MODE.value
            }
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

    async def shutdown(self):
        """Graceful shutdown of the trading bot"""
        self.logger.info("Shutting down trading bot...")
        self.is_running = False

        await self._send_notification("🛑 **Trading Bot Shutting Down**")

        if self.mt5_handler:
            if Config.DEBUG.get("close_positions_on_shutdown", False):
                await self.mt5_handler.close_all_positions()
            await self.mt5_handler.disconnect()

        self.logger.info("Trading bot shutdown completed")

async def main():
    bot = TradingBot()
    if await bot.initialize():
        await bot.run()
    else:
        print("Failed to initialize trading bot. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        sys.exit(1)