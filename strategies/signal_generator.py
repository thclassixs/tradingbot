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
        self.min_confluence_factors = 3  # Minimum number of confirming factors for a valid signal
        self.htf_timeframe = htf_timeframe
        self.fvg_sensitivity = fvg_sensitivity
        self.order_block_sensitivity = order_block_sensitivity

    async def generate_signal(self, dfs: Dict[str, pd.DataFrame], timeframe: str, symbol: str) -> Optional[TradeSignal]:
        """
        The main method to generate a trading signal. It orchestrates all analyses
        and evaluates their combined output.
        """
        df = dfs[timeframe]
        htf_df = dfs.get(self.htf_timeframe)

        # 1. Higher-Timeframe Confirmation
        if htf_df is not None and not htf_df.empty:
            mtf_confirmed = self.higher_timeframe_confirmation(df, htf_df)
        else:
            # If HTF data is not available, we can't confirm, but can still proceed
            mtf_confirmed = False 

        # 2. Perform Primary Analyses on the main timeframe
        df_with_delta = self.volume_analysis.calculate_volume_delta(df.copy())
        
        structure_breaks = self.market_structure.detect_market_structure_break(df_with_delta)
        volume_profile = self.volume_analysis.analyze_volume_profile(df_with_delta)
        patterns = self.pattern_analysis.analyze_patterns(df_with_delta, timeframe)
        order_blocks = self.market_structure.detect_order_blocks(df_with_delta)
        fvgs = self.market_structure.identify_fair_value_gaps(df_with_delta)
        
        # 3. Score the analyses
        pattern_score = self.pattern_analysis.pattern_confluence(patterns)
        volume_confirmed = any(p.volume_confirmed for p in patterns)

        # 4. Evaluate Confluence and Generate Signal
        signal = await self._evaluate_confluence(
            df=df_with_delta,
            structure_breaks=structure_breaks,
            volume_profile=volume_profile,
            patterns=patterns,
            order_blocks=order_blocks,
            fvgs=fvgs,
            mtf_confirmed=mtf_confirmed,
            pattern_score=pattern_score,
            volume_confirmed=volume_confirmed,
            timeframe=timeframe,
            symbol=symbol
        )
        
        return signal

    async def _evaluate_confluence(self, df: pd.DataFrame, structure_breaks: List, 
                           volume_profile: Dict, patterns: List, order_blocks: List, 
                           fvgs: List, mtf_confirmed: bool, pattern_score: float, 
                           volume_confirmed: bool, timeframe: str, symbol: str) -> Optional[TradeSignal]:
        """
        The core logic engine. Evaluates all analysis inputs to find a
        high-probability trade setup.
        """
        # A recent break of market structure is our primary trigger for a potential signal
        if not structure_breaks:
            return None

        last_break = structure_breaks[-1]
        direction = "BUY" if last_break['type'] == 'bullish' else "SELL"
        
        reasons = [f"Primary Signal: {last_break['type']} break of structure."]
        confidence_factors = 1
        
        current_price = df['close'].iloc[-1]

        # --- Confluence Checks ---

        # 1. Candlestick Pattern Confirmation
        if pattern_score > 0.6 and volume_confirmed:
            reasons.append(f"Confirmed by strong candlestick patterns (Score: {pattern_score:.2f}) with volume.")
            confidence_factors += 1

        # 2. Higher-Timeframe Alignment
        if mtf_confirmed:
            reasons.append("Direction aligns with the higher-timeframe trend.")
            confidence_factors += 1
        
        # 3. Order Block Confluence
        for ob in reversed(order_blocks):
            is_relevant_ob = (direction == "BUY" and ob.type == 'bullish') or \
                             (direction == "SELL" and ob.type == 'bearish')
            # Check if price is near a relevant order block
            if is_relevant_ob and abs(current_price - ob.high) < (current_price * self.order_block_sensitivity):
                reasons.append(f"Price is reacting to a significant {ob.type} order block.")
                confidence_factors += 1
                break

        # 4. Fair Value Gap (FVG) Confluence
        for fvg in reversed(fvgs):
            # Calculate the middle of the FVG
            fvg_midpoint = (fvg['gap_size'] / 2) + min(df['low'].iloc[fvg['end_idx']], df['high'].iloc[fvg['start_idx']])
            if abs(current_price - fvg_midpoint) < (current_price * self.fvg_sensitivity):
                reasons.append("Price is mitigating a Fair Value Gap, suggesting a potential reversal or continuation.")
                confidence_factors += 1
                break

        # 5. Volume Profile Confluence
        if volume_profile:
            poc_interval = max(volume_profile, key=volume_profile.get, default=None)
            if poc_interval:
                poc = poc_interval.mid
                # For a buy, we want to be below the Point of Control (buying at a discount)
                if direction == "BUY" and current_price < poc:
                    reasons.append("Price is below the Point of Control (POC), indicating potential for upward movement.")
                    confidence_factors += 1
                # For a sell, we want to be above the Point of Control (selling at a premium)
                elif direction == "SELL" and current_price > poc:
                    reasons.append("Price is above the Point of Control (POC), indicating potential for downward movement.")
                    confidence_factors += 1

        # --- Final Decision ---
        
        # If we don't have enough confirming factors, it's not a high-probability trade.
        if confidence_factors < self.min_confluence_factors:
            return None

        # Calculate a confidence score (0.0 to 1.0)
        # We start with a base of 0.5 and add points for each confirming factor.
        confidence = min(1.0, 0.5 + (confidence_factors * 0.1))
        
        # Get dynamic stop-loss and take-profit from the risk manager
        # We need to add an 'atr' column to the dataframe first
        if 'atr' not in df.columns:
             df['atr'] = self.risk_manager.calculate_atr(df) 
        stop_loss, take_profit = await self.risk_manager.calculate_dynamic_exits(df, current_price, direction, symbol)


        return TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=timeframe,
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