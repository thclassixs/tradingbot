import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from config import Config
from utils.logger import TradingLogger

class MT5Handler:
    """Enhanced MT5 Handler with proper error handling and async support"""

    def __init__(self):
        self.logger = TradingLogger("MT5Handler")
        self.is_initialized = False
        self.connection_retries = 0
        self.max_retries = 3
        self.symbol_map = {
            'GOLD': 'XAUUSDm',
            'BITCOIN': 'BTCUSDm',
            'XAU': 'XAUUSDm',
            'BTC': 'BTCUSDm',
            'XAUUSD': 'XAUUSDm',
            'BTCUSD': 'BTCUSDm'
        }
        self.timeframe_map = {
            1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30, 60: mt5.TIMEFRAME_H1, 240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1,
        }

    async def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error}")
                return False

            if Config.MT5_CONFIG["login"] and Config.MT5_CONFIG["password"]:
                login_result = mt5.login(
                    login=Config.MT5_CONFIG["login"],
                    password=Config.MT5_CONFIG["password"],
                    server=Config.MT5_CONFIG["server"]
                )
                if not login_result:
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False

            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False

            self.is_initialized = True
            self.logger.info(f"MT5 initialized successfully. Account: {account_info.login}")
            await self._validate_symbols()
            return True
        except Exception as e:
            self.logger.error(f"Exception during MT5 initialization: {e}")
            return False

    def _get_correct_symbol(self, symbol: str) -> str:
        """Get the correct symbol name for the broker"""
        if mt5.symbol_info(symbol) is not None:
            return symbol
        mapped_symbol = self.symbol_map.get(symbol.upper())
        if mapped_symbol and mt5.symbol_info(mapped_symbol) is not None:
            self.logger.info(f"Using mapped symbol: {symbol} -> {mapped_symbol}")
            return mapped_symbol
        return symbol

    async def _validate_symbols(self):
        """Validate that our target symbols are available"""
        target_symbols = ['XAUUSDm', 'BTCUSDm']
        for symbol in target_symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"Symbol {symbol} not available")
            else:
                if not mt5.symbol_select(symbol, True):
                    self.logger.warning(f"Could not select symbol {symbol}")
                else:
                    self.logger.info(f"Symbol {symbol} validated and selected")

    def is_connected(self) -> bool:
        """Check if MT5 is connected"""
        try:
            if not self.is_initialized: return False
            return mt5.account_info() is not None
        except Exception:
            return False

    async def reconnect(self) -> bool:
        """Attempt to reconnect to MT5"""
        try:
            if self.connection_retries >= self.max_retries:
                self.logger.error("Max reconnection attempts reached")
                return False
            self.connection_retries += 1
            self.logger.info(f"Reconnecting to MT5 (attempt {self.connection_retries})")
            mt5.shutdown()
            await asyncio.sleep(2)
            success = await self.initialize()
            if success:
                self.connection_retries = 0
                self.logger.info("Reconnection successful")
            return success
        except Exception as e:
            self.logger.error(f"Error during reconnection: {e}")
            return False

    async def get_market_data(self, symbol: str, timeframe: int = 5, count: int = 500) -> Optional[Dict]:
        """Get market data for a symbol"""
        try:
            if not self.is_connected():
                self.logger.error("MT5 not connected")
                return None
            correct_symbol = self._get_correct_symbol(symbol)
            mt5_timeframe = self.timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
            if not mt5.symbol_select(correct_symbol, True):
                self.logger.error(f"Could not select symbol {correct_symbol}")
                return None
            await asyncio.sleep(0.1)
            rates = mt5.copy_rates_from_pos(correct_symbol, mt5_timeframe, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to get rates for {correct_symbol}: {mt5.last_error()}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return {
                'symbol': correct_symbol, 'time': df['time'].values, 'open': df['open'].values,
                'high': df['high'].values, 'low': df['low'].values, 'close': df['close'].values,
                'tick_volume': df['tick_volume'].values, 'real_volume': df.get('real_volume', np.zeros(len(df))).values
            }
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            correct_symbol = self._get_correct_symbol(symbol)
            if not mt5.symbol_select(correct_symbol, True): return None
            symbol_info = mt5.symbol_info(correct_symbol)
            if symbol_info is None: return None
            return {
                'symbol': symbol_info.name, 'bid': symbol_info.bid, 'ask': symbol_info.ask,
                'point': symbol_info.point, 'digits': symbol_info.digits,
                'stops_level': symbol_info.trade_stops_level, 'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max, 'lot_step': symbol_info.volume_step,
                'trade_mode': symbol_info.trade_mode, 'time': symbol_info.time
            }
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    async def execute_trade(self, signal, win_streak: int = 0) -> Dict:
        """Execute a trading signal"""
        try:
            if not self.is_connected():
                return {"success": False, "error": "MT5 not connected"}
            correct_symbol = self._get_correct_symbol(signal.symbol)
            symbol_info = await self.get_symbol_info(signal.symbol)
            if not symbol_info:
                return {"success": False, "error": f"Could not get info for {correct_symbol}"}
            
            lot_size = self._calculate_lot_size(signal, symbol_info, win_streak)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": correct_symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": symbol_info['ask'] if signal.direction == "BUY" else symbol_info['bid'],
                "sl": signal.stop_loss, "tp": signal.take_profit,
                "deviation": Config.EXECUTION.get("max_slippage", 20),
                "magic": Config.MT5_CONFIG.get("magic", 234000),
                "comment": f"Classixs-{signal.direction}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"{result.retcode} - {result.comment}"}
            return {"success": True, "ticket": result.order}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _calculate_lot_size(self, signal, symbol_info: Dict, win_streak: int = 0) -> float:
        """Calculate appropriate lot size for the trade"""
        try:
            base_lot = Config.get_symbol_config(signal.symbol).min_lot
            account_info = mt5.account_info()
            if account_info and hasattr(signal, 'risk_percentage'):
                risk_amount = account_info.equity * (signal.risk_percentage / 100)
                sl_distance = abs(signal.entry_price - signal.stop_loss)
                if sl_distance > 0:
                    pip_value = symbol_info['point'] * 10
                    base_lot = risk_amount / (sl_distance / pip_value)
            
            if Config.AUTO_LOT_BOOST["enabled"] and win_streak >= Config.AUTO_LOT_BOOST["win_streak_threshold"]:
                base_lot *= Config.AUTO_LOT_BOOST["boost_multiplier"]
                self.logger.info(f"Auto-Lot Boost Applied! Streak: {win_streak}")

            lot_size = max(symbol_info['min_lot'], min(base_lot, symbol_info['max_lot']))
            lot_step = symbol_info['lot_step']
            return round(lot_size / lot_step) * lot_step
        except Exception as e:
            self.logger.error(f"Error calculating lot size: {e}")
            return symbol_info['min_lot']

    async def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None: return []
            return [
                {'ticket': p.ticket, 'symbol': p.symbol, 'type': 'BUY' if p.type == 0 else 'SELL',
                 'volume': p.volume, 'price_open': p.price_open, 'price_current': p.price_current,
                 'sl': p.sl, 'tp': p.tp, 'profit': p.profit}
                for p in positions
            ]
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
            
    async def get_deal_by_ticket(self, ticket: int) -> Optional[Dict]:
        """Fetches a historical deal by its position/order ticket."""
        try:
            deals = mt5.history_deals_get(position=ticket)
            if deals and len(deals) > 0:
                deal = deals[-1]
                return {
                    'ticket': deal.order, 'profit': deal.profit, 'volume': deal.volume,
                    'price': deal.price, 'type': deal.type, 'entry': deal.entry
                }
            return None
        except Exception as e:
            self.logger.error(f"Error fetching deal {ticket}: {e}")
            return None

    async def modify_position(self, ticket: int, new_sl: float = None, new_tp: float = None) -> bool:
        """Modify the stop loss or take profit of an open position."""
        try:
            request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket}
            if new_sl is not None: request['sl'] = new_sl
            if new_tp is not None: request['tp'] = new_tp
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {ticket} modified successfully.")
                return True
            else:
                self.logger.error(f"Failed to modify position {ticket}: {result.comment}")
                return False
        except Exception as e:
            self.logger.error(f"Error modifying position {ticket}: {e}")
            return False

    async def close_position(self, ticket: int) -> Dict:
        """Close a specific position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position: return {"success": False, "error": "Position not found"}
            pos = position[0]
            symbol_info = await self.get_symbol_info(pos.symbol)
            if not symbol_info: return {"success": False, "error": "Symbol info not found"}

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol, "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": symbol_info['bid'] if pos.type == 0 else symbol_info['ask'],
                "deviation": 20, "magic": 234000, "comment": "TradingBot-Close",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"{result.retcode} - {result.comment}"}
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
            
            return {
                'login': account_info.login, 'server': account_info.server,
                'balance': account_info.balance, 'equity': account_info.equity,
                'margin': account_info.margin, 'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level, 'currency': account_info.currency,
                'company': account_info.company, 'trade_allowed': getattr(account_info, 'trade_allowed', 1),
                'trade_expert': getattr(account_info, 'trade_expert', 1)
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}

    # --- THIS METHOD WAS MISSING ---
    async def check_market_conditions(self, symbol: str = None) -> bool:
        """Check if market conditions are suitable for trading"""
        try:
            account_info = await self.get_account_info()
            if not account_info or not account_info.get('trade_allowed'):
                self.logger.warning("Trading not allowed on this account")
                return False
            
            if not account_info.get('trade_expert'):
                self.logger.warning("Expert trading not allowed on this account")
                return False
            
            if symbol:
                symbol_info = await self.get_symbol_info(symbol)
                
                if not symbol_info:
                    self.logger.warning(f"Symbol {symbol} not available")
                    return False
                
                if symbol_info['trade_mode'] == mt5.SYMBOL_TRADE_MODE_DISABLED:
                    self.logger.warning(f"Trading disabled for {symbol}")
                    return False

                last_quote_time = datetime.fromtimestamp(symbol_info['time'])
                time_since_last_quote = datetime.now() - last_quote_time
                
                if time_since_last_quote.total_seconds() > 300:
                    self.logger.info(f"Market for {symbol} appears stale. Last quote was at {last_quote_time}.")
                    return False

            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {e}")
            return False

    async def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.logger.info("MT5 disconnected")