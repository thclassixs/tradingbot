import asyncio
import pandas as pd
from datetime import datetime
import pytz
import MetaTrader5 as mt5

# Import necessary components from your trading bot
from config import Config
from components.mt5_handler import MT5Handler
from strategies.signal_generator import SignalGenerator
from strategies.risk_management import RiskManagement
from strategies.market_data import MarketStructure
from strategies.volume_analysis import VolumeAnalysis
from strategies.pattern_analysis import PatternAnalysis
from strategies.session_analysis import SessionAnalysis
from utils.logger import TradingLogger
from utils.helpers import TradeSignal

class Backtester:
    """
    A class to backtest the trading bot's strategy on historical data.
    """

    def __init__(self, start_date: str, end_date: str, symbol: str, timeframe: str, initial_balance: float = 100.0):
        """
        Initializes the Backtester.
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.open_trades = []
        self.trade_history = []
        self.logger = TradingLogger("Backtester")
        self.config = Config()
        self.mt5_handler = MT5Handler()
        self.risk_manager = RiskManagement(account_balance=self.initial_balance, mt5_handler=self.mt5_handler, max_risk_percent=self.config.RISK_MANAGEMENT["MAX_RISK_PERCENT"])
        self.market_data = MarketStructure()
        self.volume_analyzer = VolumeAnalysis()
        self.pattern_analyzer = PatternAnalysis()
        self.session_analyzer = SessionAnalysis()
        self.signal_generator = SignalGenerator(
            market_data=self.market_data,
            volume_analysis=self.volume_analyzer,
            pattern_analysis=self.pattern_analyzer,
            session_analysis=self.session_analyzer,
            risk_management=self.risk_manager,
        )

    async def run(self):
        """
        Runs the backtest.
        """
        self.logger.info("Starting backtest...")
        self.logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        self.logger.info(f"Symbol: {self.symbol}, Timeframe: {self.timeframe}")
        self.logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")

        historical_data = await self._load_historical_data()
        if historical_data.empty:
            self.logger.error("Could not load historical data. Aborting backtest.")
            return

        for i in range(1, len(historical_data)):
            current_candle = historical_data.iloc[i]
            self._update_open_trades(current_candle)

            df_slice = historical_data.iloc[: i + 1]

            # Generate a signal
            signal = await self.signal_generator.generate_signal(
                dfs={self.timeframe: df_slice},
                timeframe=self.timeframe,
                symbol=self.symbol,
            )

            if signal:
                await self._execute_trade(signal, historical_data.index[i])

        self._close_all_trades(historical_data.iloc[-1])
        self.generate_performance_report()

    async def _load_historical_data(self) -> pd.DataFrame:
        """
        Loads historical data from MT5 for the specified date range.
        """
        await self.mt5_handler.initialize()
        if not self.mt5_handler.is_connected():
            self.logger.error("Could not connect to MT5.")
            return pd.DataFrame()

        timezone = pytz.timezone("Etc/UTC")
        utc_from = timezone.localize(self.start_date)
        utc_to = timezone.localize(self.end_date)

        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
        }
        mt5_timeframe = timeframe_map.get(self.timeframe, mt5.TIMEFRAME_H1)

        rates = mt5.copy_rates_range(self.symbol, mt5_timeframe, utc_from, utc_to)

        if rates is None or len(rates) == 0:
            self.logger.error(f"No historical data found for {self.symbol} in the specified range.")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    async def _execute_trade(self, signal: TradeSignal, timestamp: datetime):
        """
        Simulates the execution of a trade.
        """
        entry_price = signal.entry_price
        lot_size = await self.risk_manager.calculate_position_size(signal)

        if lot_size > 0:
            trade = {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry_price": entry_price,
                "lot_size": lot_size,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "entry_time": timestamp,
                "status": "open",
            }
            self.open_trades.append(trade)
            self.logger.info(f"Opened {signal.direction} trade for {signal.symbol} at {entry_price:.5f}")

    def _update_open_trades(self, candle: pd.Series):
        """
        Updates the status of open trades based on the current candle.
        """
        for trade in self.open_trades[:]:
            if trade["status"] == "open":
                symbol_config = self.config.get_symbol_config(trade["symbol"])
                pip_value = symbol_config.pip_value if symbol_config else 0.1

                if trade["direction"] == "BUY":
                    if candle["low"] <= trade["stop_loss"]:
                        self._close_trade(trade, trade["stop_loss"], candle.name, "Stop Loss")
                    elif candle["high"] >= trade["take_profit"]:
                        self._close_trade(trade, trade["take_profit"], candle.name, "Take Profit")
                elif trade["direction"] == "SELL":
                    if candle["high"] >= trade["stop_loss"]:
                        self._close_trade(trade, trade["stop_loss"], candle.name, "Stop Loss")
                    elif candle["low"] <= trade["take_profit"]:
                        self._close_trade(trade, trade["take_profit"], candle.name, "Take Profit")

        open_pnl = 0
        for t in self.open_trades:
            symbol_config = self.config.get_symbol_config(t["symbol"])
            pip_value = symbol_config.pip_value if symbol_config else 0.1
            if t['direction'] == 'BUY':
                t['pnl'] = (candle['close'] - t['entry_price']) * t['lot_size'] * pip_value
            else:
                t['pnl'] = (t['entry_price'] - candle['close']) * t['lot_size'] * pip_value
            open_pnl += t.get('pnl', 0)
        self.equity = self.balance + open_pnl

    def _close_trade(self, trade: dict, close_price: float, close_time: datetime, reason: str):
        """
        Closes a trade and records it in the trade history.
        """
        symbol_config = self.config.get_symbol_config(trade["symbol"])
        pip_value = symbol_config.pip_value if symbol_config else 0.1

        if trade["direction"] == "BUY":
            pnl = (close_price - trade["entry_price"]) * trade["lot_size"] * pip_value
        else: # SELL
            pnl = (trade["entry_price"] - close_price) * trade["lot_size"] * pip_value

        trade["status"] = "closed"
        trade["close_price"] = close_price
        trade["close_time"] = close_time
        trade["pnl"] = pnl
        trade["reason"] = reason

        self.balance += pnl
        self.trade_history.append(trade)
        if trade in self.open_trades:
            self.open_trades.remove(trade)

        self.logger.info(f"Closed trade for {trade['symbol']}. P&L: ${pnl:,.2f}. Reason: {reason}")

    def _close_all_trades(self, last_candle: pd.Series):
        """
        Closes all remaining open trades.
        """
        for trade in self.open_trades[:]:
            self._close_trade(trade, last_candle["close"], last_candle.name, "End of Backtest")

    def generate_performance_report(self):
        """
        Generates and prints a detailed performance report.
        """
        if not self.trade_history:
            print("\n--- No trades were executed during the backtest ---")
            return

        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t["pnl"] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum([t["pnl"] for t in self.trade_history])
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0

        total_wins = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else "inf"

        max_drawdown = self._calculate_max_drawdown()

        print("\n--- Backtest Performance Report ---")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print("-" * 35)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print("-" * 35)
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Average P&L per Trade: ${average_pnl:,.2f}")
        print(f"Profit Factor: {profit_factor if isinstance(profit_factor, str) else round(profit_factor, 2)}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print("-" * 35)

    def _calculate_max_drawdown(self):
        """
        Calculates the maximum drawdown of the strategy.
        """
        equity_curve = [self.initial_balance]
        running_balance = self.initial_balance
        for trade in self.trade_history:
            running_balance += trade['pnl']
            equity_curve.append(running_balance)

        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown * 100

if __name__ == "__main__":
    # --- Configuration ---
    START_DATE = "2025-06-01"
    END_DATE = "2025-06-30"
    SYMBOL = "XAUUSDm"
    TIMEFRAME = "M30"  # M1, M5, M15, M30, H1
    INITIAL_BALANCE = 100.0

    async def main():
        backtester = Backtester(
            start_date=START_DATE,
            end_date=END_DATE,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            initial_balance=INITIAL_BALANCE,
        )
        await backtester.run()

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")