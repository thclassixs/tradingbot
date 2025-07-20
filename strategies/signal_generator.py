import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TradeSignal:
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    reasons: List[str]

class SignalGenerator:
    def __init__(self, market_structure, volume_analysis, pattern_analysis, session_analysis, 
                 htf_timeframe: str = 'H1', fvg_sensitivity: float = 0.002, order_block_sensitivity: float = 0.005):
        self.market_structure = market_structure
        self.volume_analysis = volume_analysis
        self.pattern_analysis = pattern_analysis
        self.session_analysis = session_analysis
        self.min_confluence = 4  # Increased due to more factors
        self.htf_timeframe = htf_timeframe
        self.fvg_sensitivity = fvg_sensitivity
        self.order_block_sensitivity = order_block_sensitivity

    def generate_signal(self, dfs: Dict[str, pd.DataFrame], timeframe: str) -> Optional[TradeSignal]:
        """Generate trading signal based on all analyses, including multi-timeframe."""
        df = dfs[timeframe]
        htf_df = dfs.get(self.htf_timeframe)

        if htf_df is None:
            # Handle cases where higher timeframe data might not be available
            # For simplicity, we can default to no confirmation, but you might want to log this
            mtf_confirmed = False 
        else:
            mtf_confirmed = self.higher_timeframe_confirmation(df, htf_df)

        df_with_delta = self.volume_analysis.calculate_volume_delta(df.copy())
        
        structure_breaks = self.market_structure.detect_market_structure_break(df_with_delta)
        volume_profile = self.volume_analysis.analyze_volume_profile(df_with_delta)
        patterns = self.pattern_analysis.analyze_patterns(df_with_delta, timeframe)
        session_stats = self.session_analysis.get_current_session()
        order_blocks = self.market_structure.detect_order_blocks(df_with_delta)
        fvgs = self.market_structure.identify_fair_value_gaps(df_with_delta)
        
        pattern_score = self.pattern_analysis.pattern_confluence(patterns)
        volume_confirmed = any([p.volume_confirmed for p in patterns])

        signal = self._evaluate_confluence(
            df_with_delta, structure_breaks, volume_profile, patterns, session_stats,
            order_blocks, fvgs, mtf_confirmed=mtf_confirmed, 
            pattern_score=pattern_score, volume_confirmed=volume_confirmed
        )
        return signal

    def _evaluate_confluence(self, df: pd.DataFrame, structure_breaks, 
                           volume_profile, patterns, session_stats,
                           order_blocks, fvgs, mtf_confirmed=True, 
                           pattern_score=0.0, volume_confirmed=False) -> Optional[TradeSignal]:
        """Evaluate confluence of all analysis factors."""
        reasons = []
        current_price = df['close'].iloc[-1]
        confidence_factors = 0

        # Base conditions
        if not (structure_breaks and pattern_score > 0.6 and volume_confirmed):
            return None

        direction = "BUY" if structure_breaks[-1]['type'] == 'bullish' else "SELL"
        reasons.append("Structure break, pattern confluence, and volume confirmed.")
        confidence_factors += 3

        # Higher-Timeframe Confirmation
        if mtf_confirmed:
            reasons.append("Signal aligns with higher-timeframe trend.")
            confidence_factors += 1
        
        # Order Block Confluence
        for ob in order_blocks:
            is_relevant_ob = (direction == "BUY" and ob.type == 'bullish') or \
                             (direction == "SELL" and ob.type == 'bearish')
            if is_relevant_ob and abs(current_price - ob.high) < (current_price * self.order_block_sensitivity):
                reasons.append(f"Price near significant {ob.type} order block.")
                confidence_factors += 1
                break

        # FVG Confluence
        for fvg in fvgs:
            fvg_midpoint = (fvg['gap_size'] / 2) + min(df['low'].iloc[fvg['end_idx']], df['high'].iloc[fvg['start_idx']])
            if abs(current_price - fvg_midpoint) < (current_price * self.fvg_sensitivity):
                reasons.append(f"Price near FVG midpoint, potential rebalance area.")
                confidence_factors += 1
                break

        # Volume Profile Confluence
        poc = max(volume_profile, key=volume_profile.get, default=0)
        if poc != 0:
            value_area_high = np.percentile(list(volume_profile.keys()), 85)
            value_area_low = np.percentile(list(volume_profile.keys()), 15)
            if direction == "BUY" and (current_price > value_area_low and current_price < poc):
                reasons.append("Price in lower value area, potential for mean reversion.")
                confidence_factors += 1
            elif direction == "SELL" and (current_price < value_area_high and current_price > poc):
                reasons.append("Price in upper value area, potential for mean reversion.")
                confidence_factors += 1

        # Final check for minimum confluence
        if confidence_factors < self.min_confluence:
            return None

        confidence = min(1.0, (pattern_score + (0.05 * confidence_factors))) * 100
        entry_price = current_price
        stop_loss = entry_price - 50 if direction == "BUY" else entry_price + 50
        take_profit = entry_price + 100 if direction == "BUY" else entry_price - 100

        # Adjust TP towards FVG
        if fvgs:
            if direction == "BUY" and fvgs[-1]['type'] == 'bullish':
                take_profit = min(take_profit, fvgs[-1]['gap_size'] + df['high'].iloc[fvgs[-1]['start_idx']])
            elif direction == "SELL" and fvgs[-1]['type'] == 'bearish':
                take_profit = max(take_profit, df['low'].iloc[fvgs[-1]['start_idx']] - fvgs[-1]['gap_size'])

        return TradeSignal(
            symbol="BTCUSDm",
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe="M5",
            reasons=reasons
        )

    def higher_timeframe_confirmation(self, df: pd.DataFrame, htf_df: pd.DataFrame) -> bool:
        """Confirm signal with higher timeframe trend."""
        htf_trend = self.market_structure.analyze_trend_context(htf_df)
        ltf_direction = "Uptrend" if df['close'].iloc[-1] > df['open'].iloc[-1] else "Downtrend"

        if ltf_direction == "Uptrend" and htf_trend == "Uptrend":
            return True
        if ltf_direction == "Downtrend" and htf_trend == "Downtrend":
            return True
        
        return False