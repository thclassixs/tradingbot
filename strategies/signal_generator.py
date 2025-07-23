import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

# Import the analysis and risk management modules
from .market_structure import MarketStructure, OrderBlock
from .volume_analysis import VolumeAnalysis
from .pattern_analysis import PatternAnalysis
from .session_analysis import SessionAnalysis
from .risk_management import RiskManagement
# Import TradeSignal and MarketSession for type hinting
from utils.helpers import TradeSignal
from config import MarketSession, Config


class SignalGenerator:
    """
    Generates trading signals based on a confluence of smart money concepts
    and multi-timeframe analysis.
    """
    def __init__(self, market_structure: MarketStructure, volume_analysis: VolumeAnalysis, 
                 pattern_analysis: PatternAnalysis, session_analysis: SessionAnalysis, 
                 risk_management: RiskManagement):
        """Initializes the SignalGenerator with all required analysis components."""
        self.market_structure = market_structure
        self.volume_analysis = volume_analysis
        self.pattern_analysis = pattern_analysis
        self.session_analysis = session_analysis
        self.risk_manager = risk_management
        self.logger = logging.getLogger(__name__)
        self.config = Config()

    async def generate_signal(self, dfs: Dict[str, pd.DataFrame], symbol: str) -> Optional[TradeSignal]:
        """
        Main method to generate a trading signal using multi-timeframe analysis.
        - HTF: Directional Bias
        - STF: Setup and Point of Interest (POI)
        - LTF: Entry Confirmation
        """
        try:
            htf_str, stf_str, ltf_str = self.config.MTF_TIMEFRAMES
            df_htf, df_stf, df_ltf = dfs[htf_str], dfs[stf_str], dfs[ltf_str]
        except KeyError as e:
            self.logger.error(f"Missing required timeframe in dfs dictionary: {e}")
            return None

        # 1. Session and Trend Analysis on Higher Timeframe (HTF)
        await self.session_analysis._update_current_session()
        current_session = self.session_analysis.get_current_session()
        
        htf_trend = self.market_structure.analyze_trend_context(df_htf)

        if current_session == MarketSession.LONDON:
            judas_direction = await self.session_analysis.detect_judas_swing(df_htf, current_session)
            if judas_direction:
                new_trend = "Uptrend" if judas_direction == "BUY" else "Downtrend"
                self.logger.info(f"Judas Swing detected. Overriding HTF trend from {htf_trend} to {new_trend}.")
                htf_trend = new_trend

        if htf_trend == "Range":
            return None

        # 2. Identify Point of Interest (POI) on Setup Timeframe (STF)
        poi = self._find_poi_on_setup_tf(df_stf, htf_trend)
        if not poi:
            return None

        # 3. Find Entry Confirmation on Lower Timeframe (LTF)
        signal = await self._find_entry_on_ltf(df_ltf, htf_trend, poi, symbol)
        
        return signal

    def _find_poi_on_setup_tf(self, df: pd.DataFrame, trend: str) -> Optional[Dict]:
        """Identifies the most relevant Point of Interest (Order Block or FVG) on the setup timeframe."""
        order_blocks = self.market_structure.detect_order_blocks(df)
        fvgs = self.market_structure.identify_fair_value_gaps(df)
        current_price = df['close'].iloc[-1]
        
        standardized_pois = []
        for ob in order_blocks:
            price_level = ob.low if ob.type == 'bullish' else ob.high
            standardized_pois.append({'type': 'OrderBlock', 'price_level': price_level, 'data': ob})
        for fvg in fvgs:
            price_level = fvg['low'] if fvg['type'] == 'bullish' else fvg['high']
            standardized_pois.append({'type': 'FVG', 'price_level': price_level, 'data': fvg})
        
        if trend == "Uptrend":
            potential_pois = [p for p in standardized_pois if ((isinstance(p['data'], OrderBlock) and p['data'].type == 'bullish') or (isinstance(p['data'], dict) and p['data']['type'] == 'bullish')) and p['price_level'] < current_price]
            all_pois = sorted(potential_pois, key=lambda p: p['price_level'], reverse=True)
        elif trend == "Downtrend":
            potential_pois = [p for p in standardized_pois if ((isinstance(p['data'], OrderBlock) and p['data'].type == 'bearish') or (isinstance(p['data'], dict) and p['data']['type'] == 'bearish')) and p['price_level'] > current_price]
            all_pois = sorted(potential_pois, key=lambda p: p['price_level'])
        else:
            return None

        return all_pois[0] if all_pois else None

    async def _find_entry_on_ltf(self, df: pd.DataFrame, trend: str, poi: Dict, symbol: str) -> Optional[TradeSignal]:
        """Looks for entry confirmation (CHoCH) on the lower timeframe and builds a confluence score."""
        structure_events = self.market_structure.detect_market_structure(df)
        last_event = structure_events[-1] if structure_events else None

        if not last_event or last_event['type'] != 'CHoCH':
            return None

        reasons = []
        confidence_score = 0.0
        direction = ""

        # --- Primary Confirmation: LTF CHoCH aligning with HTF trend ---
        if trend == "Uptrend" and last_event['direction'] == 'bullish':
            direction = "BUY"
            confidence_score += self.config.CONFLUENCE_WEIGHTS['ltf_choch']
            reasons.append(f"Bullish CHoCH on {self.config.MTF_TIMEFRAMES[2]} aligning with {trend} bias.")
        elif trend == "Downtrend" and last_event['direction'] == 'bearish':
            direction = "SELL"
            confidence_score += self.config.CONFLUENCE_WEIGHTS['ltf_choch']
            reasons.append(f"Bearish CHoCH on {self.config.MTF_TIMEFRAMES[2]} aligning with {trend} bias.")
        else:
            return None

        # --- Confluence Checks ---
        current_price = df['close'].iloc[-1]
        poi_price = poi['price_level']
        if abs(current_price - poi_price) < (current_price * 0.002):
            confidence_score += self.config.CONFLUENCE_WEIGHTS['poi_reaction']
            reasons.append(f"Price reacting near {self.config.MTF_TIMEFRAMES[1]} {poi['type']}.")

        # Volume Delta Confirmation
        vol_df = self.volume_analysis.calculate_volume_delta(df.copy())
        if (vol_df['volume_delta'].iloc[-1] > 0 and direction == "BUY") or \
           (vol_df['volume_delta'].iloc[-1] < 0 and direction == "SELL"):
            confidence_score += self.config.CONFLUENCE_WEIGHTS['volume_confirmation']
            reasons.append("Volume delta confirms entry direction.")

        # Candlestick Pattern Confirmation
        pattern_score = self.pattern_analysis.get_pattern_confluence_score(df, at_index=len(df)-1)
        if pattern_score > 0.6:
            confidence_score += self.config.CONFLUENCE_WEIGHTS['pattern_confirmation']
            reasons.append(f"Candlestick pattern score: {pattern_score:.2f}.")

        # --- Final Decision ---
        if confidence_score < self.config.SIGNAL_THRESHOLDS['min_confidence']:
            self.logger.info(f"Signal for {symbol} did not meet min confidence: {confidence_score:.2f}")
            return None

        # --- Exit Calculation & Final Validation ---
        stop_loss, take_profit = await self.risk_manager.calculate_dynamic_exits(df, current_price, direction, symbol)
        
        signal = TradeSignal(
            symbol=symbol, direction=direction, confidence=confidence_score,
            entry_price=current_price, stop_loss=stop_loss, take_profit=take_profit,
            timeframe=self.config.MTF_TIMEFRAMES[2], reasons=reasons
        )
        
        if await self.risk_manager.validate_signal(signal):
            return signal
        
        self.logger.warning(f"Signal for {symbol} rejected by risk manager (R:R or invalid exits).")
        return None