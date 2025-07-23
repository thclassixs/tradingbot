import pandas as pd
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
from config import MarketSession


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
        
        self.min_confluence_factors = 2

    async def generate_signal(self, dfs: Dict[str, pd.DataFrame], symbol: str) -> Optional[TradeSignal]:
        """
        Main method to generate a trading signal using multi-timeframe analysis.
        - 1H: Directional Bias
        - 15M: Setup and Point of Interest (POI)
        - 5M: Entry Confirmation
        """
        if not all(tf in dfs for tf in ['H1', 'M15', 'M5']):
            self.logger.warning("Missing one or more required timeframes (H1, M15, M5).")
            return None

        df_1h, df_15m, df_5m = dfs['H1'], dfs['M15'], dfs['M5']
        
        # --- Session Analysis Integration ---
        await self.session_analysis._update_current_session()
        current_session = self.session_analysis.get_current_session()
        
        # 1. Higher-Timeframe (1H) for Directional Bias
        htf_trend = self.market_structure.analyze_trend_context(df_1h)

        # Override trend based on Judas Swing during London Open
        if current_session == MarketSession.LONDON:
            judas_direction = await self.session_analysis.detect_judas_swing(df_1h, current_session)
            if judas_direction:
                htf_trend = "Uptrend" if judas_direction == "BUY" else "Downtrend"
                self.logger.info(f"Judas Swing detected. Overriding HTF trend to {htf_trend}.")

        if htf_trend == "Range":
            return None

        # 2. Medium-Timeframe (15M) for Setup and POI
        poi = self._find_poi_on_15m(df_15m, htf_trend)
        if not poi:
            return None

        # 3. Lower-Timeframe (5M) for Entry Confirmation
        signal = await self._find_entry_on_5m(df_5m, htf_trend, poi, symbol)
        
        return signal

    def _find_poi_on_15m(self, df: pd.DataFrame, trend: str) -> Optional[Dict]:
        """Identifies a Point of Interest (Order Block or FVG) on the 15M chart."""
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

        if all_pois:
            return all_pois[0]
        
        return None

    async def _find_entry_on_5m(self, df: pd.DataFrame, trend: str, poi: Dict, symbol: str) -> Optional[TradeSignal]:
        """Looks for a Change of Character (CHoCH) on the 5M chart for entry confirmation."""
        
        structure_events = self.market_structure.detect_market_structure(df)
        last_event = structure_events[-1] if structure_events else None

        if not last_event or last_event['type'] != 'CHoCH':
            return None

        reasons = []
        confidence_factors = 0
        
        if trend == "Uptrend" and last_event['direction'] == 'bullish':
            reasons.append("Bullish CHoCH on 5M in alignment with HTF bias.")
            confidence_factors += 1
            direction = "BUY"
            
        elif trend == "Downtrend" and last_event['direction'] == 'bearish':
            reasons.append("Bearish CHoCH on 5M in alignment with HTF bias.")
            confidence_factors += 1
            direction = "SELL"
        else:
            return None

        current_price = df['close'].iloc[-1]
        poi_price = poi['price_level']
        if abs(current_price - poi_price) < (current_price * 0.002):
             reasons.append(f"Price reacting near 15M {poi['type']} at {poi_price:.5f}.")
             confidence_factors += 1
        
        if confidence_factors < self.min_confluence_factors:
            return None
            
        confidence = min(1.0, 0.6 + (confidence_factors * 0.20))
        
        # Use the fully upgraded risk manager for exit calculations
        stop_loss, take_profit = await self.risk_manager.calculate_dynamic_exits(df, current_price, direction, symbol)

        # Final validation of the generated signal
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe='M5',
            reasons=reasons
        )
        
        if await self.risk_manager.validate_signal(signal):
            return signal
        else:
            self.logger.warning(f"Signal for {symbol} rejected by risk manager due to poor R:R or invalid exits.")
            return None