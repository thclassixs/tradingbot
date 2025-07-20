import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PatternSignal:
    pattern_type: str
    strength: float
    context: str
    volume_confirmed: bool
    timeframe: str
    price_level: float

class PatternAnalysis:
    def __init__(self):
        self.patterns = {
            'reversal': {
                'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING',
                'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLHARAMI'
            },
            'continuation': {
                'CDLMARUBOZU', 'CDL3WHITESOLDIERS', 'CDL3BLACKCROWS',
                'CDLRISEFALL3METHODS'
            },
            'indecision': {
                'CDLDOJI', 'CDLSPINNINGTOP', 'CDLLONGLEGGEDDOJI'
            }
        }
    
    def analyze_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Analyze candlestick patterns with volume confirmation"""
        signals = []
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                if hasattr(talib, pattern):
                    pattern_func = getattr(talib, pattern)
                    result = pattern_func(df['open'], df['high'], df['low'], df['close'])
                    
                    # Find where pattern occurs
                    pattern_locations = np.where(result != 0)[0]
                    
                    for idx in pattern_locations:
                        strength = self._calculate_pattern_strength(df, idx)
                        volume_confirmed = self._check_volume_confirmation(df, idx)
                        context = self._determine_market_context(df, idx)
                        
                        if strength > 0.6:  # Minimum strength threshold
                            signals.append(PatternSignal(
                                pattern_type=pattern,
                                strength=strength,
                                context=context,
                                volume_confirmed=volume_confirmed,
                                timeframe=timeframe,
                                price_level=df['close'].iloc[idx]
                            ))
        
        return signals
    
    def detect_multi_bar_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Detect complex multi-bar patterns (e.g. 3-bar reversals)"""
        signals = []
        # Example: 3-bar reversal
        for i in range(2, len(df)):
            if df['close'].iloc[i-2] < df['close'].iloc[i-1] < df['close'].iloc[i]:
                strength = self._calculate_pattern_strength(df, i)
                volume_confirmed = self._check_volume_confirmation(df, i)
                context = self._determine_market_context(df, i)
                if strength > 0.7:
                    signals.append(PatternSignal(
                        pattern_type="3_bar_bullish",
                        strength=strength,
                        context=context,
                        volume_confirmed=volume_confirmed,
                        timeframe=timeframe,
                        price_level=df['close'].iloc[i]
                    ))
        return signals

    def detect_custom_moroccan_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Detect custom Moroccan market patterns (placeholder)"""
        signals = []
        # Example: Add custom logic here
        # signals.append(PatternSignal(...))
        return signals

    def multi_timeframe_pattern_analysis(self, dfs: Dict[str, pd.DataFrame]) -> List[PatternSignal]:
        """Analyze patterns across multiple timeframes (placeholder)"""
        mtf_signals = []
        for tf, df in dfs.items():
            mtf_signals.extend(self.analyze_patterns(df, tf))
        return mtf_signals

    def pattern_confluence(self, signals: List[PatternSignal]) -> float:
        """Score confluence of multiple patterns"""
        score = sum([s.strength for s in signals if s.volume_confirmed]) / max(len(signals), 1)
        return score

    def classify_pattern_context(self, signal: PatternSignal) -> str:
        """Classify pattern as reversal or continuation based on context"""
        if signal.context == "Uptrend" and signal.pattern_type in ['CDLHAMMER', 'CDLENGULFING']:
            return "Reversal"
        elif signal.context == "Downtrend" and signal.pattern_type in ['CDLMARUBOZU', 'CDL3BLACKCROWS']:
            return "Continuation"
        return "Indecision"

    def _calculate_pattern_strength(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate pattern strength based on size and location"""
        # Implementation details...
        return 0.8
    
    def _check_volume_confirmation(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if pattern is confirmed by volume"""
        if idx < 1 or idx >= len(df):
            return False
            
        avg_volume = df['tick_volume'].iloc[idx-1:idx+1].mean()
        current_volume = df['tick_volume'].iloc[idx]
        
        return current_volume > (avg_volume * 1.5)
    
    def _determine_market_context(self, df: pd.DataFrame, idx: int) -> str:
        """Determine market context for pattern"""
        if idx < 20:
            return "Unknown"
            
        sma20 = df['close'].rolling(20).mean()
        current_price = df['close'].iloc[idx]
        
        if current_price > sma20.iloc[idx]:
            return "Uptrend"
        return "Downtrend"
