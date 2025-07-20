import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Tuple, Dict, List
import logging
from telegram import Bot
from telegram.constants import ParseMode

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def is_session_active(current_time: datetime) -> Tuple[bool, str]:
    """Check if current time is in active trading session"""
    hour = current_time.hour
    
    # Morocco time (UTC+1)
    london_session = (9, 18)  # 9:00-18:00 local time
    ny_session = (15, 24)     # 15:00-24:00 local time
    
    if london_session[0] <= hour < london_session[1]:
        return True, "London"
    elif ny_session[0] <= hour < ny_session[1]:
        return True, "New York"
    return False, "Inactive"

def detect_high_volatility(df: pd.DataFrame, window: int = 20) -> bool:
    """Detect high volatility periods"""
    atr = calculate_atr(df)
    current_atr = atr.iloc[-1]
    avg_atr = atr.rolling(window).mean().iloc[-1]
    
    return current_atr > (avg_atr * 1.5)

def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
    """Detect swing highs and lows"""
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        if df['high'].iloc[i] > df['high'].iloc[i-lookback:i+lookback+1].max():
            highs.append(i)
        if df['low'].iloc[i] < df['low'].iloc[i-lookback:i+lookback+1].min():
            lows.append(i)
    return highs, lows

def dynamic_support_resistance(df: pd.DataFrame, window: int = 50) -> Dict[str, List[float]]:
    """Dynamic support/resistance using swing points"""
    highs, lows = detect_swing_points(df)
    support = [df['low'].iloc[i] for i in lows]
    resistance = [df['high'].iloc[i] for i in highs]
    return {'support': sorted(set(support)), 'resistance': sorted(set(resistance))}

def session_aware_volatility(df: pd.DataFrame, session: str) -> float:
    """Calculate volatility for specific session"""
    if session == "London":
        mask = (df.index.hour >= 9) & (df.index.hour < 18)
    elif session == "New York":
        mask = (df.index.hour >= 15) & (df.index.hour < 24)
    else:
        mask = [True] * len(df)
    session_df = df[mask]
    return calculate_atr(session_df).iloc[-1] if not session_df.empty else 0

def calculate_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Calculate correlation between two assets"""
    return df1['close'].corr(df2['close'])

def news_impact_analysis(news_events: List[Dict], df: pd.DataFrame) -> Dict[str, float]:
    """Analyze news impact on volatility"""
    impact = {}
    for event in news_events:
        event_time = event['time']
        window = df[(df.index >= event_time - pd.Timedelta(minutes=30)) & (df.index <= event_time + pd.Timedelta(minutes=30))]
        if not window.empty:
            impact[event['name']] = window['high'].max() - window['low'].min()
    return impact

class TelegramNotifier:
    """Telegram notification handler for trading bot (async, python-telegram-bot)"""

    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token from BotFather
            chat_id: Chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        # Optionally test connection here
        return await self.test_connection()

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram asynchronously
        
        Args:
            message: Message text to send
            parse_mode: Telegram parse mode (HTML or Markdown)
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML if parse_mode == "HTML" else ParseMode.MARKDOWN
            )
            self.logger.info("Telegram message sent successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
            return False

    async def send_trade_alert(self, symbol: str, action: str, price: float, quantity: float, reason: str) -> bool:
        """
        Send a formatted trade alert
        
        Args:
            symbol: Trading symbol
            action: Trade action (BUY/SELL)
            price: Entry price
            quantity: Position size
            reason: Trading reason/signal
            
        Returns:
            bool: True if sent successfully
        """
        emoji = "🟢" if action.upper() == "BUY" else "🔴"
        message = f"""
{emoji} <b>TRADE ALERT</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action.upper()}
<b>Price:</b> {price:.5f}
<b>Quantity:</b> {quantity}
<b>Reason:</b> {reason}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return await self.send_message(message.strip())

    async def send_error_alert(self, error_message: str, component: str = "Trading Bot") -> bool:
        """
        Send an error alert
        
        Args:
            error_message: Error message
            component: Component where error occurred
            
        Returns:
            bool: True if sent successfully
        """
        message = f"""
⚠️ <b>ERROR ALERT</b> ⚠️

<b>Component:</b> {component}
<b>Error:</b> {error_message}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return await self.send_message(message.strip())

    async def send_status_update(self, status: str, details: Dict = None) -> bool:
        """
        Send a status update
        
        Args:
            status: Status message
            details: Optional details dictionary
            
        Returns:
            bool: True if sent successfully
        """
        message = f"""
ℹ️ <b>STATUS UPDATE</b> ℹ️

<b>Status:</b> {status}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if details:
            message += "\n\n<b>Details:</b>"
            for key, value in details.items():
                message += f"\n• <b>{key}:</b> {value}"
        
        return await self.send_message(message.strip())

    async def send_daily_summary(self, summary_data: Dict) -> bool:
        """
        Send daily trading summary
        
        Args:
            summary_data: Dictionary with summary data
            
        Returns:
            bool: True if sent successfully
        """
        message = f"""
📊 <b>DAILY SUMMARY</b> 📊

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}
<b>Total Trades:</b> {summary_data.get('total_trades', 0)}
<b>Profitable Trades:</b> {summary_data.get('profitable_trades', 0)}
<b>Win Rate:</b> {summary_data.get('win_rate', 0):.2f}%
<b>Total P&L:</b> {summary_data.get('total_pnl', 0):.2f}
<b>Best Trade:</b> {summary_data.get('best_trade', 0):.2f}
<b>Worst Trade:</b> {summary_data.get('worst_trade', 0):.2f}
        """
        
        return await self.send_message(message.strip())

    async def test_connection(self) -> bool:
        """
        Test the Telegram connection
        
        Returns:
            bool: True if connection is working
        """
        test_message = "🤖 Trading Bot Connection Test - " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return await self.send_message(test_message)