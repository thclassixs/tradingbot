"""
Session Analysis Module
Analyzes trading sessions and their characteristics, including common liquidity traps.
"""

import asyncio
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple, Optional
from config import Config, MarketSession
from utils.logger import TradingLogger

class SessionAnalysis:
    """Analyzes market sessions and anticipates time-based liquidity events."""
    
    def __init__(self):
        self.logger = TradingLogger("SessionAnalysis")
        self.current_session = None
        self.session_strength = {}
        
    async def initialize(self):
        """Initialize session analysis component."""
        try:
            self.logger.info("Initializing Session Analysis...")
            await self._update_current_session()
            self.logger.info("Session Analysis initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Session Analysis: {e}")
            raise
    
    async def _update_current_session(self):
        """Update current trading session information."""
        try:
            current_hour = datetime.now().hour
            
            is_active, session = Config.is_trading_session_active(current_hour)
            self.current_session = session if is_active else None
            
        except Exception as e:
            self.logger.error(f"Error updating current session: {e}")
            self.current_session = None
    
    async def detect_judas_swing(self, df: pd.DataFrame, session: MarketSession) -> Optional[str]:
        """
        Detects the "Judas Swing" (a fakeout move) at the start of a session.
        Returns the direction of the likely true move.
        """
        if session != MarketSession.LONDON:
            return None # Judas Swing is most common at London open

        session_start_hour = Config.TRADING_SESSIONS[MarketSession.LONDON][0]
        
        # Filter for Asian session (typically preceding London) and early London
        asian_session_df = df[df.index.hour < session_start_hour]
        london_open_df = df[df.index.hour >= session_start_hour]

        if asian_session_df.empty or london_open_df.empty:
            return None

        asian_high = asian_session_df['high'].max()
        asian_low = asian_session_df['low'].min()

        # Check for a sweep of Asian session liquidity
        sweep_high = london_open_df['high'].iloc[0] > asian_high
        sweep_low = london_open_df['low'].iloc[0] < asian_low

        if sweep_high and not sweep_low:
            # Liquidity was taken from the top, expect a move down.
            self.logger.info("Judas Swing detected: Asian high swept. Anticipating bearish move.")
            return "SELL"
        
        if sweep_low and not sweep_high:
            # Liquidity was taken from the bottom, expect a move up.
            self.logger.info("Judas Swing detected: Asian low swept. Anticipating bullish move.")
            return "BUY"
            
        return None

    def get_current_session(self) -> Optional[MarketSession]:
        """Get current market session."""
        return self.current_session