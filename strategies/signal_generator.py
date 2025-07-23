import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import the analysis and risk management modules
from .market_structure import MarketStructure
from .volume_analysis import VolumeAnalysis
from .pattern_analysis import PatternAnalysis
from .session_analysis import SessionAnalysis
from .risk_management import RiskManagement
# Import TradeSignal from its new, central location
from utils.helpers import TradeSignal


class SignalGenerator:
    """
    Generates high-confidence trading signals by finding confluence
    among various analytical methods.
    """
    def __init__(self, market_structure: MarketStructure, volume_analysis: VolumeAnalysis, 
                 pattern_analysis: PatternAnalysis, session_analysis: SessionAnalysis, 
                 risk_management: RiskManagement, htf_timeframe: str = 'H1', 
                 fvg_sensitivity: float = 0.002, order_block_sensitivity: float = 0.005):
        """Initializes the SignalGenerator with all required analysis components."""
        self.market_structure = market_structure
        self.volume_analysis = volume_analysis
        self.pattern_analysis = pattern_analysis
        self.session_analysis = session_analysis
        self.risk_manager = risk_management
        
        # Configuration for signal evaluation
        self.min_confluence_factors = 2  # MODIFIED
        self.htf_timeframe = 'H1'
        self.setup_timeframe = 'M15'
        self.entry_timeframe = 'M5'

    async def generate_signal(self, dfs: Dict[str, pd.DataFrame], symbol: str) -> Optional[TradeSignal]:
        """
        The main method to generate a trading signal. It orchestrates all analyses
        and evaluates their combined output.
        """
        h1_df = dfs[self.htf_timeframe]
        m15_df = dfs[self.setup_timeframe]
        m5_df = dfs[self.entry_timeframe]

        # 1. Higher-Timeframe (H1) Directional Bias
        h1_trend = self.market_structure.analyze_trend_context(h1_df)

        # 2. Setup Timeframe (M15) Analysis
        m15_order_blocks = self.market_structure.detect_order_blocks(m15_df)
        m15_fvgs = self.market_structure.identify_fair_value_gaps(m15_df)
        
        # 3. Entry Timeframe (M5) Analysis
        m5_breaks = self.market_structure.detect_market_structure_break(m5_df)

        # 4. Evaluate Confluence and Generate Signal
        signal = await self._evaluate_confluence(
            h1_trend=h1_trend,
            m15_order_blocks=m15_order_blocks,
            m15_fvgs=m15_fvgs,
            m5_breaks=m5_breaks,
            m5_df=m5_df,
            symbol=symbol
        )
        
        return signal

    async def _evaluate_confluence(self, h1_trend: str, m15_order_blocks: List, 
                           m15_fvgs: List, m5_breaks: List, 
                           m5_df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """
        The core logic engine. Evaluates all analysis inputs to find a
        high-probability trade setup.
        """
        reasons = []
        confidence_factors = 0
        
        current_price = m5_df['close'].iloc[-1]

        # Determine trade direction based on H1 trend
        if h1_trend == "Uptrend":
            direction = "BUY"
            reasons.append(f"H1 trend is bullish.")
            confidence_factors += 1
        elif h1_trend == "Downtrend":
            direction = "SELL"
            reasons.append(f"H1 trend is bearish.")
            confidence_factors += 1
        else:
            return None # No clear trend on H1

        # Check for M15 Order Block or FVG
        poi = None # Point of Interest
        if direction == "BUY":
            # Look for bullish order block or FVG
            for ob in reversed(m15_order_blocks):
                if ob.type == 'bullish' and current_price >= ob.low and current_price <= ob.high:
                    poi = ob
                    reasons.append(f"Price entered a bullish M15 Order Block.")
                    confidence_factors += 1
                    break
            if not poi:
                for fvg in reversed(m15_fvgs):
                    if fvg['type'] == 'bullish' and current_price >= fvg['bottom'] and current_price <= fvg['top']:
                        poi = fvg
                        reasons.append(f"Price entered a bullish M15 Fair Value Gap.")
                        confidence_factors += 1
                        break
        else: # SELL
            # Look for bearish order block or FVG
            for ob in reversed(m15_order_blocks):
                if ob.type == 'bearish' and current_price >= ob.low and current_price <= ob.high:
                    poi = ob
                    reasons.append(f"Price entered a bearish M15 Order Block.")
                    confidence_factors += 1
                    break
            if not poi:
                for fvg in reversed(m15_fvgs):
                    if fvg['type'] == 'bearish' and current_price >= fvg['bottom'] and current_price <= fvg['top']:
                        poi = fvg
                        reasons.append(f"Price entered a bearish M15 Fair Value Gap.")
                        confidence_factors += 1
                        break
        
        if not poi:
            return None # No point of interest found on M15

        # Check for M5 Change of Character (CHoCH)
        choch = None
        if m5_breaks:
            last_break = m5_breaks[-1]
            if (direction == "BUY" and last_break['direction'] == 'bullish') or \
               (direction == "SELL" and last_break['direction'] == 'bearish'):
                choch = last_break
                reasons.append(f"M5 Change of Character confirmed entry.")
                confidence_factors += 1

        if not choch:
            return None # No entry confirmation on M5

        # --- Final Decision ---
        
        if confidence_factors < self.min_confluence_factors:
            return None

        confidence = min(1.0, 0.5 + (confidence_factors * 0.15)) # MODIFIED
        
        if 'atr' not in m5_df.columns:
             m5_df['atr'] = self.risk_manager.calculate_atr(m5_df) 
        stop_loss, take_profit = await self.risk_manager.calculate_dynamic_exits(m5_df, current_price, direction, symbol)


        return TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=self.entry_timeframe,
            reasons=reasons
        )

    def higher_timeframe_confirmation(self, df: pd.DataFrame, htf_df: pd.DataFrame) -> bool:
        """Confirms if the short-term signal aligns with the long-term trend."""
        htf_trend = self.market_structure.analyze_trend_context(htf_df)
        
        # Determine the direction of the most recent candle on the lower timeframe
        ltf_direction = "Uptrend" if df['close'].iloc[-1] > df['open'].iloc[-1] else "Downtrend"

        # Check for alignment
        if ltf_direction == "Uptrend" and htf_trend == "Uptrend":
            return True
        if ltf_direction == "Downtrend" and htf_trend == "Downtrend":
            return True
        
        return False