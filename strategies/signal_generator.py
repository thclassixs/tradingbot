import pandas as pd
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
    def __init__(self, market_structure, volume_analysis, pattern_analysis, session_analysis):
        self.market_structure = market_structure
        self.volume_analysis = volume_analysis
        self.pattern_analysis = pattern_analysis
        self.session_analysis = session_analysis
        self.min_confluence = 3
        
    def generate_signal(self, df: pd.DataFrame, timeframe: str) -> Optional[TradeSignal]:
        """Generate trading signal based on all analyses"""
        # Get individual analysis results
        structure_breaks = self.market_structure.detect_market_structure_break(df)
        volume_profile = self.volume_analysis.analyze_volume_delta_profile(df)
        patterns = self.pattern_analysis.analyze_patterns(df, timeframe)
        session_stats = self.session_analysis.analyze_session_characteristics(df)

        # Multi-timeframe confirmation (example)
        mtf_confirmed = True  # Placeholder for multi-timeframe logic

        # Pattern confluence scoring
        pattern_score = self.pattern_analysis.pattern_confluence(patterns)
        volume_confirmed = any([p.volume_confirmed for p in patterns])

        # Combine analyses for confluence
        signal = self._evaluate_confluence(
            df, structure_breaks, volume_profile, patterns, session_stats,
            mtf_confirmed=mtf_confirmed, pattern_score=pattern_score, volume_confirmed=volume_confirmed
        )

        return signal

    def _evaluate_confluence(self, df: pd.DataFrame, structure_breaks, 
                           volume_profile, patterns, session_stats,
                           mtf_confirmed=True, pattern_score=0.0, volume_confirmed=False) -> Optional[TradeSignal]:
        """Evaluate confluence of different analysis factors"""
        reasons = []
        current_price = df['close'].iloc[-1]

        # Example logic: require structure break, pattern score, and volume confirmation
        if structure_breaks and pattern_score > 0.6 and volume_confirmed and mtf_confirmed:
            direction = "BUY" if structure_breaks[-1]['type'] == 'bullish' else "SELL"
            confidence = min(1.0, pattern_score + 0.2) * 100
            entry_price = current_price
            stop_loss = entry_price - 50 if direction == "BUY" else entry_price + 50
            take_profit = entry_price + 100 if direction == "BUY" else entry_price - 100
            reasons.append("Structure break, pattern confluence, volume confirmed")
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
        # Return None if insufficient confluence found
        return None

    def generate_multi_symbol_signal(self, dfs: Dict[str, pd.DataFrame], timeframes: Dict[str, str]) -> Dict[str, Optional[TradeSignal]]:
        """Generate signals for multiple symbols"""
        signals = {}
        for symbol, df in dfs.items():
            tf = timeframes.get(symbol, "M5")
            signals[symbol] = self.generate_signal(df, tf)
        return signals

    def higher_timeframe_confirmation(self, df: pd.DataFrame, htf_df: pd.DataFrame) -> bool:
        """Confirm signal with higher timeframe (placeholder)"""
        # Example: check if trend matches
        return True
