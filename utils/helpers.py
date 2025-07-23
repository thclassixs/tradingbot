import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging
from telegram import Bot
from telegram.request import HTTPXRequest
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

class TelegramNotifier:
    """Handles all notifications sent to a Telegram chat."""

    def __init__(self, bot_token: str, chat_id: str):
        if not bot_token or not chat_id:
            raise ValueError("Telegram bot_token and chat_id are required.")
        # Increased timeouts for more reliability
        request = HTTPXRequest(connect_timeout=15.0, read_timeout=15.0)
        self.bot = Bot(token=bot_token, request=request)
        self.chat_id = chat_id
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Tests the connection to Telegram upon startup."""
        return await self.test_connection()

    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Sends a message to the configured Telegram chat."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            self.logger.info("Telegram message sent successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}", exc_info=True)
            return False

    async def send_trade_alert(self, signal: TradeSignal) -> bool:
        """Sends a beautifully formatted alert for an executed trade."""
        emoji = "🟢" if signal.direction.upper() == "BUY" else "🔴"
        message = f"""
{emoji} *TRADE ALERT* {emoji}

*Symbol:* `{signal.symbol}`
*Action:* `{signal.direction.upper()}`
*Entry Price:* `{signal.entry_price:.5f}`
*Confidence:* `{signal.confidence:.2%}`
*Stop Loss:* `{signal.stop_loss:.5f}`
*Take Profit:* `{signal.take_profit:.5f}`
        """
        return await self.send_message(message.strip())

    async def send_status_update(self, status: str, details: Dict = None) -> bool:
        """Sends a periodic status update of the bot's health and performance."""
        message = f"""
ℹ️ *BOT STATUS UPDATE* ℹ️

*Status:* {status}
*Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
        """

        if details:
            message += "\n\n*PERFORMANCE:*"
            for key, value in details.items():
                formatted_key = key.replace('_', ' ').title()
                message += f"\n• *{formatted_key}:* `{value}`"

        return await self.send_message(message.strip())

    async def test_connection(self) -> bool:
        """Sends a test message to confirm the Telegram connection is working."""
        try:
            bot_info = await self.bot.get_me()
            self.logger.info(f"Successfully connected to Telegram bot: {bot_info.username}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Telegram: {e}")
            return False