import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PatternSignal:
    """Represents a detected candlestick pattern and its context."""
    pattern_type: str
    strength: float
    context: str  # e.g., 'Uptrend', 'Downtrend', 'at_resistance'
    volume_confirmed: bool
    timeframe: str
    price_level: float

class PatternAnalysis:
    """
    Analyzes candlestick patterns for confluence within a smart money framework.
    This module is not for generating primary signals, but for confirming entries at POIs.
    """
    def __init__(self):
        self.patterns = {
            'reversal': {
                'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING',
                'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLHARAMI'
            },
            'indecision': {
                'CDLDOJI', 'CDLSPINNINGTOP'
            }
        }

    def get_pattern_confluence_score(self, df: pd.DataFrame, at_index: int, window: int = 5) -> float:
        """
        Calculates a confluence score based on recent patterns near a specific index.
        This is the primary method to be used by the SignalGenerator.
        """
        signals = []
        # Analyze a small window of candles around the point of interest
        start_index = max(0, at_index - window)
        end_index = min(len(df), at_index + 1)
        
        for pattern_type, patterns in self.patterns.items():
            for pattern_code in patterns:
                if hasattr(talib, pattern_code):
                    pattern_func = getattr(talib, pattern_code)
                    result = pattern_func(df['open'], df['high'], df['low'], df['close'])
                    
                    # Check for a valid pattern signal within our target window
                    for i in range(start_index, end_index):
                        if result.iloc[i] != 0:
                            strength = self._calculate_pattern_strength(df, i, result.iloc[i])
                            if strength > 0.6:  # Only consider strong patterns
                                signals.append(strength)
        
        if not signals:
            return 0.0
        
        # Return the strength of the strongest recent pattern as the score
        return max(signals)

    def _calculate_pattern_strength(self, df: pd.DataFrame, idx: int, pattern_result) -> float:
        """Calculate pattern strength based on candle size and volume."""
        try:
            if idx < 1 or idx >= len(df): return 0.0
            
            base_strength = abs(float(pattern_result)) / 200.0  # Normalize TA-Lib's output

            candle_range = df['high'].iloc[idx] - df['low'].iloc[idx]
            body_size = abs(df['close'].iloc[idx] - df['open'].iloc[idx])
            
            if candle_range > 0:
                body_ratio = body_size / candle_range
                base_strength = (base_strength + body_ratio) / 2

            # Volume Confirmation
            avg_volume = df['tick_volume'].iloc[max(0, idx - 20):idx].mean()
            current_volume = df['tick_volume'].iloc[idx]
            
            if current_volume > avg_volume * 1.5:
                base_strength += 0.1 # Add a bonus for high volume
            
            return min(base_strength, 1.0)
        except (IndexError, ValueError, TypeError) as e:
            # logger.error(f"Error calculating pattern strength at index {idx}: {e}")
            return 0.0