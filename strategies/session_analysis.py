"""
Session Analysis Module
Analyzes trading sessions and their characteristics
"""

import asyncio
from datetime import datetime, time
from typing import Dict, Tuple, Optional
from config import Config, MarketSession
from utils.logger import TradingLogger

class SessionAnalysis:
    """Analyzes market sessions and trading conditions"""
    
    def __init__(self):
        self.logger = TradingLogger("SessionAnalysis")
        self.current_session = None
        self.session_strength = {}
        
    async def initialize(self):
        """Initialize session analysis component"""
        try:
            self.logger.info("Initializing Session Analysis...")
            await self._update_current_session()
            self.logger.info("Session Analysis initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Session Analysis: {e}")
            raise
    
    async def _update_current_session(self):
        """Update current trading session information"""
        try:
            current_hour = datetime.now().hour
            
            # Validate hour before processing
            if not (0 <= current_hour <= 23):
                self.logger.error(f"Invalid current hour: {current_hour}")
                return
            
            is_active, session = Config.is_trading_session_active(current_hour)
            self.current_session = session if is_active else None
            
            # Check local trading hours
            local_active = Config.is_local_trading_time(current_hour)
            
            self.logger.info(f"Current hour: {current_hour}, Session: {session}, Local active: {local_active}")
            
        except Exception as e:
            self.logger.error(f"Error updating current session: {e}")
            self.current_session = None
    
    async def get_session_strength(self, symbol: str) -> float:
        """Get strength multiplier for current session"""
        try:
            if not self.current_session:
                await self._update_current_session()
            
            if not self.current_session:
                return 1.0  # Default multiplier
            
            symbol_config = Config.get_symbol_config(symbol)
            return symbol_config.session_multiplier.get(self.current_session, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error getting session strength: {e}")
            return 1.0
    
    async def is_high_activity_session(self) -> bool:
        """Check if current session is high activity"""
        try:
            if not self.current_session:
                await self._update_current_session()
            
            high_activity_sessions = [MarketSession.LONDON, MarketSession.NEW_YORK, MarketSession.OVERLAP]
            return self.current_session in high_activity_sessions
            
        except Exception as e:
            self.logger.error(f"Error checking session activity: {e}")
            return False
    
    def get_current_session(self) -> Optional[MarketSession]:
        """Get current market session"""
        return self.current_session