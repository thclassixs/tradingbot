import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging  # <-- This was the missing import
from telegram import Bot
from telegram.constants import ParseMode

@dataclass
class TradeSignal:
    """Data class to hold all information about a trading signal."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    reasons: List[str]

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def is_session_active(current_time: datetime) -> Tuple[bool, str]:
    """Check if the current time falls within active trading sessions."""
    hour = current_time.hour
    
    # Session times defined in local time (e.g., Morocco UTC+1)
    london_session = (9, 18)
    ny_session = (15, 24)
    
    if london_session[0] <= hour < london_session[1]:
        return True, "London"
    elif ny_session[0] <= hour < ny_session[1]:
        return True, "New York"
    return False, "Inactive"

# --- Telegram Notifier Class ---

class TelegramNotifier:
    """Handles all notifications sent to a Telegram chat."""

    def __init__(self, bot_token: str, chat_id: str):
        """
        Initializes the Telegram notifier.
        
        Args:
            bot_token: Your Telegram bot token from BotFather.
            chat_id: The ID of the chat where messages will be sent.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        # Initialize a logger specific to this class
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Tests the connection to Telegram upon startup."""
        return await self.test_connection()

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Sends a message to the configured Telegram chat.
        
        Returns:
            True if the message was sent successfully, False otherwise.
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML if parse_mode == "HTML" else ParseMode.MARKDOWN
            )
            self.logger.info("Telegram message sent successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}", exc_info=True)
            return False

    async def send_trade_alert(self, signal: TradeSignal) -> bool:
        """Sends a beautifully formatted alert for an executed trade."""
        emoji = "🟢" if signal.direction.upper() == "BUY" else "🔴"
        message = f"""
{emoji} <b>TRADE ALERT</b> {emoji}

<b>Symbol:</b> {signal.symbol}
<b>Action:</b> {signal.direction.upper()}
<b>Entry Price:</b> {signal.entry_price:.5f}
<b>Confidence:</b> {signal.confidence:.2%}
<b>Stop Loss:</b> {signal.stop_loss:.5f}
<b>Take Profit:</b> {signal.take_profit:.5f}

<b>Reason:</b>
- {"\n- ".join(signal.reasons)}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return await self.send_message(message.strip())

    async def send_status_update(self, status: str, details: Dict = None) -> bool:
        """Sends a periodic status update of the bot's health and performance."""
        message = f"""
ℹ️ <b>BOT STATUS UPDATE</b> ℹ️

<b>Status:</b> {status}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if details:
            message += "\n\n<b>PERFORMANCE:</b>"
            for key, value in details.items():
                # Format the key to be more readable
                formatted_key = key.replace('_', ' ').title()
                message += f"\n• <b>{formatted_key}:</b> {value}"
        
        return await self.send_message(message.strip())

    async def test_connection(self) -> bool:
        """Sends a test message to confirm the Telegram connection is working."""
        test_message = f"🤖 Trading Bot connection test successful at {datetime.now().strftime('%H:%M:%S')}"
        return await self.send_message(test_message)

