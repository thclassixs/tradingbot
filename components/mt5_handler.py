import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Change relative imports to absolute imports
from config import Config
from utils.logger import TradingLogger

class MT5Handler:
    """Enhanced MT5 Handler with proper error handling and async support"""
    
    def __init__(self):
        """Initialize MT5 Handler"""
        self.logger = TradingLogger("MT5Handler")
        self.is_initialized = False
        self.connection_retries = 0
        self.max_retries = 3
        
    async def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error}")
                return False
            
            # Login if credentials are provided
            if Config.MT5_CONFIG["login"] and Config.MT5_CONFIG["password"]:
                login_result = mt5.login(
                    login=Config.MT5_CONFIG["login"],
                    password=Config.MT5_CONFIG["password"],
                    server=Config.MT5_CONFIG["server"]
                )
                
                if not login_result:
                    error = mt5.last_error()
                    self.logger.error(f"MT5 login failed: {error}")
                    return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            self.is_initialized = True
            self.logger.info(f"MT5 initialized successfully. Account: {account_info.login}")
            return True
            
        except Exception as e:
            self.logger.error(f"Exception during MT5 initialization: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if MT5 is connected"""
        try:
            if not self.is_initialized:
                return False
            
            account_info = mt5.account_info()
            return account_info is not None
            
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
            
            # Shutdown and reinitialize
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
    
    async def get_market_data(self, symbol: str, timeframe=mt5.TIMEFRAME_M5, count: int = 100) -> Optional[Dict]:
        """Get market data for a symbol"""
        try:
            if not self.is_connected():
                self.logger.error("MT5 not connected")
                return None
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get rates for {symbol}: {error}")
                return None
            
            if len(rates) == 0:
                self.logger.warning(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Return as dictionary with numpy arrays
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'time': df['time'].values,
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'tick_volume': df['tick_volume'].values,
                'spread': df.get('spread', np.zeros(len(df))).values,
                'real_volume': df.get('real_volume', np.zeros(len(df))).values
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return None
            
            return {
                'symbol': symbol_info.name,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'trade_mode': symbol_info.trade_mode,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'margin_required': getattr(symbol_info, 'margin_initial', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    async def execute_trade(self, signal) -> Dict:
        """Execute a trading signal"""
        try:
            if not self.is_connected():
                return {"success": False, "error": "MT5 not connected"}
            
            # Get symbol info
            symbol_info = await self.get_symbol_info(signal.symbol)
            if not symbol_info:
                return {"success": False, "error": f"Could not get info for {signal.symbol}"}
            
            # Prepare trade request
            action = mt5.TRADE_ACTION_DEAL
            order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
            price = symbol_info['ask'] if signal.direction == "BUY" else symbol_info['bid']
            
            # Calculate lot size
            lot_size = self._calculate_lot_size(signal, symbol_info)
            
            request = {
                "action": action,
                "symbol": signal.symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": signal.stop_loss if hasattr(signal, 'stop_loss') else 0,
                "tp": signal.take_profit if hasattr(signal, 'take_profit') else 0,
                "deviation": Config.MT5_CONFIG.get("deviation", 20),
                "magic": Config.MT5_CONFIG.get("magic", 234000),
                "comment": f"TradingBot-{signal.signal_type}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Trade failed: {result.retcode} - {result.comment}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg, "retcode": result.retcode}
            
            self.logger.info(f"Trade executed successfully: {result.order}")
            return {
                "success": True,
                "ticket": result.order,
                "volume": result.volume,
                "price": result.price,
                "request_id": result.request_id
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_lot_size(self, signal, symbol_info: Dict) -> float:
        """Calculate appropriate lot size for the trade"""
        try:
            # Get configured lot size
            base_lot = signal.lot_size if hasattr(signal, 'lot_size') else Config.get_symbol_config(signal.symbol).min_lot
            
            # Apply risk management
            if hasattr(signal, 'risk_percentage') and signal.risk_percentage:
                account_info = mt5.account_info()
                if account_info:
                    risk_amount = account_info.equity * (signal.risk_percentage / 100)
                    
                    # Calculate lot size based on stop loss distance
                    if hasattr(signal, 'stop_loss') and signal.stop_loss:
                        sl_distance = abs(signal.entry_price - signal.stop_loss)
                        if sl_distance > 0:
                            pip_value = symbol_info['point'] * 10  # For most forex pairs
                            calculated_lot = risk_amount / (sl_distance / pip_value)
                            base_lot = min(calculated_lot, symbol_info['max_lot'])
            
            # Ensure lot size is within limits
            lot_size = max(symbol_info['min_lot'], min(base_lot, symbol_info['max_lot']))
            
            # Round to lot step
            lot_step = symbol_info['lot_step']
            lot_size = round(lot_size / lot_step) * lot_step
            
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Error calculating lot size: {e}")
            return symbol_info['min_lot']
    
    async def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'time': datetime.fromtimestamp(pos.time),
                    'comment': pos.comment
                })
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def close_position(self, ticket: int) -> Dict:
        """Close a specific position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"success": False, "error": "Position not found"}
            
            pos = position[0]
            symbol_info = await self.get_symbol_info(pos.symbol)
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = symbol_info['bid'] if pos.type == mt5.POSITION_TYPE_BUY else symbol_info['ask']
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": Config.MT5_CONFIG.get("deviation", 20),
                "magic": Config.MT5_CONFIG.get("magic", 234000),
                "comment": "TradingBot-Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Close failed: {result.retcode} - {result.comment}"
                return {"success": False, "error": error_msg}
            
            return {"success": True, "ticket": result.order}
            
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_all_positions(self) -> Dict:
        """Close all open positions"""
        try:
            positions = await self.get_positions()
            results = {"closed": 0, "failed": 0, "errors": []}
            
            for pos in positions:
                result = await self.close_position(pos['ticket'])
                if result['success']:
                    results['closed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Ticket {pos['ticket']}: {result['error']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return {"closed": 0, "failed": 0, "errors": [str(e)]}
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
            
            return {
                'login': account_info.login,
                'server': account_info.server,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency,
                'company': account_info.company
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    async def check_market_conditions(self) -> bool:
        """Check if market conditions are suitable for trading"""
        try:
            # Check if market is open
            account_info = mt5.account_info()
            if not account_info:
                return False
            
            # Check if trading is allowed
            if account_info.trade_allowed == 0:
                self.logger.warning("Trading not allowed on this account")
                return False
            
            # Add more market condition checks here
            # For example: spread checks, volatility checks, etc.
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MT5"""
        try:
            mt5.shutdown()
            self.is_initialized = False
            self.logger.info("MT5 disconnected")
        except Exception as e:
            self.logger.error(f"Error disconnecting from MT5: {e}")