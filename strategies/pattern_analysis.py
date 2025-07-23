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
        self.multi_bar_patterns = {
            'CDL3INSIDE',       # Three Inside Up/Down
            'CDL3OUTSIDE',      # Three Outside Up/Down
            'CDL3LINESTRIKE',   # Three-Line Strike
            'CDLADVANCEBLOCK',  # Advance Block
            'CDLXSIDEGAP3METHODS' # Upside/Downside Gap Three Methods
        }

    def analyze_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Analyze candlestick patterns with volume confirmation"""
        signals = []

        try:
            for pattern_type, patterns in self.patterns.items():
                for pattern in patterns:
                    if hasattr(talib, pattern):
                        pattern_func = getattr(talib, pattern)
                        result = pattern_func(df['open'], df['high'], df['low'], df['close'])

                        # Fix: Use proper indexing to avoid interval comparison issues
                        pattern_locations = np.where(result != 0)[0]

                        for idx in pattern_locations[-10:]:  # Only check last 10 patterns
                            if idx < len(df) and idx >= 0:  # Ensure valid index
                                try:
                                    # --- FIX: Use .iloc for position-based indexing ---
                                    pattern_value = result.iloc[idx] if hasattr(result, 'iloc') else result[idx]
                                    strength = self._calculate_pattern_strength(df, idx, pattern_value)
                                    volume_confirmed = self._check_volume_confirmation(df, idx)
                                    context = self._determine_market_context(df, idx)

                                    if strength > 0.6:
                                        signals.append(PatternSignal(
                                            pattern_type=pattern,
                                            strength=strength,
                                            context=context,
                                            volume_confirmed=volume_confirmed,
                                            timeframe=timeframe,
                                            price_level=float(df['close'].iloc[idx])
                                        ))
                                except Exception as e:
                                    print(f"Error processing pattern {pattern} at index {idx}: {e}")
                                    continue
        except Exception as e:
            print(f"Error in analyze_patterns: {e}")

        return signals
    def multi_timeframe_pattern_analysis(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[PatternSignal]]:
        """Analyze patterns across multiple timeframes."""
        mtf_signals = {}
        for tf, df in dfs.items():
            try:
                mtf_signals[tf] = self.analyze_patterns(df, tf)
            except Exception as e:
                print(f"Error analyzing timeframe {tf}: {e}")
                mtf_signals[tf] = []
        return mtf_signals

    def detect_multi_bar_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Detect complex multi-bar patterns using TA-Lib."""
        signals = []

        try:
            for pattern in self.multi_bar_patterns:
                if hasattr(talib, pattern):
                    pattern_func = getattr(talib, pattern)
                    result = pattern_func(df['open'], df['high'], df['low'], df['close'])

                    pattern_locations = np.where(result != 0)[0]

                    for idx in pattern_locations[-10:]:  # Only check last 10 patterns
                        if idx < len(df) and idx >= 0:  # Ensure valid index
                            try:
                                # --- FIX: Use .iloc for position-based indexing ---
                                pattern_value = result.iloc[idx] if hasattr(result, 'iloc') else result[idx]
                                strength = self._calculate_pattern_strength(df, idx, pattern_value)
                                volume_confirmed = self._check_volume_confirmation(df, idx)
                                context = self._determine_market_context(df, idx)

                                if strength > 0.7:
                                    signals.append(PatternSignal(
                                        pattern_type=pattern,
                                        strength=strength,
                                        context=context,
                                        volume_confirmed=volume_confirmed,
                                        timeframe=timeframe,
                                        price_level=float(df['close'].iloc[idx])
                                    ))
                            except Exception as e:
                                print(f"Error processing multi-bar pattern {pattern} at index {idx}: {e}")
                                continue
        except Exception as e:
            print(f"Error in detect_multi_bar_patterns: {e}")

        return signals

    def detect_custom_moroccan_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Detect custom Moroccan market patterns"""
        signals = []

        try:
            # Example: Detect Morning Gap pattern common in emerging markets
            if len(df) > 5:
                for i in range(5, len(df)):
                    # Morning gap up with volume
                    if (df['open'].iloc[i] > df['close'].iloc[i-1] * 1.02 and
                        df['tick_volume'].iloc[i] > df['tick_volume'].iloc[i-5:i].mean() * 1.5):

                        signals.append(PatternSignal(
                            pattern_type="MORNING_GAP",
                            strength=0.75,
                            context=self._determine_market_context(df, i),
                            volume_confirmed=True,
                            timeframe=timeframe,
                            price_level=float(df['close'].iloc[i])
                        ))
        except Exception as e:
            print(f"Error in detect_custom_moroccan_patterns: {e}")

        return signals

    def pattern_confluence(self, signals: List[PatternSignal]) -> float:
        """Score confluence of multiple patterns"""
        if not signals:
            return 0.0

        try:
            volume_confirmed_signals = [s for s in signals if s.volume_confirmed]
            if not volume_confirmed_signals:
                return 0.0

            score = sum([s.strength for s in volume_confirmed_signals]) / len(volume_confirmed_signals)
            return min(score, 1.0)
        except Exception as e:
            print(f"Error in pattern_confluence: {e}")
            return 0.0

    def classify_pattern_context(self, signal: PatternSignal) -> str:
        """Classify pattern as reversal or continuation based on context"""
        try:
            if signal.context == "Uptrend" and signal.pattern_type in ['CDLHAMMER', 'CDLENGULFING']:
                return "Reversal"
            elif signal.context == "Downtrend" and signal.pattern_type in ['CDLMARUBOZU', 'CDL3BLACKCROWS']:
                return "Continuation"
            return "Indecision"
        except Exception as e:
            print(f"Error in classify_pattern_context: {e}")
            return "Unknown"

    def _calculate_pattern_strength(self, df: pd.DataFrame, idx: int, pattern_result) -> float:
        """Calculate pattern strength based on candle size and pattern result."""
        try:
            if not isinstance(idx, int) or idx < 0 or idx >= len(df):
                return 0.0

            try:
                pattern_value = float(pattern_result)
            except (TypeError, ValueError):
                return 0.0

            strength = abs(pattern_value) / 200.0

            candle_high = float(df['high'].iloc[idx])
            candle_low = float(df['low'].iloc[idx])
            candle_close = float(df['close'].iloc[idx])
            candle_open = float(df['open'].iloc[idx])

            candle_range = candle_high - candle_low
            body_size = abs(candle_close - candle_open)

            if candle_range > 0:
                body_ratio = body_size / candle_range
                strength = (strength + body_ratio) / 2

            return min(strength, 1.0)

        except Exception as e:
            print(f"Error calculating pattern strength at index {idx}: {e}")
            return 0.0

    def _check_volume_confirmation(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if pattern is confirmed by volume"""
        try:
            if not isinstance(idx, int) or idx < 1 or idx >= len(df):
                return False

            # Ensure we have volume data
            if 'tick_volume' not in df.columns:
                return False

            start_idx = max(0, idx - 20)
            avg_volume = float(df['tick_volume'].iloc[start_idx:idx].mean())
            current_volume = float(df['tick_volume'].iloc[idx])

            return current_volume > (avg_volume * 1.5)

        except Exception as e:
            print(f"Error checking volume confirmation at index {idx}: {e}")
            return False

    def _determine_market_context(self, df: pd.DataFrame, idx: int) -> str:
        """Determine market context for pattern"""
        try:
            if not isinstance(idx, int) or idx < 20 or idx >= len(df):
                return "Unknown"

            sma20 = df['close'].rolling(20).mean()
            current_price = float(df['close'].iloc[idx])
            sma20_value = float(sma20.iloc[idx])

            if current_price > sma20_value:
                return "Uptrend"
            return "Downtrend"

        except Exception as e:
            print(f"Error determining market context at index {idx}: {e}")
            return "Unknown"

    def get_trading_signal(self, df: pd.DataFrame, timeframe: str = "1H") -> Dict:
        """Main method to get trading signal from pattern analysis"""
        try:
            # Get all pattern signals
            pattern_signals = self.analyze_patterns(df, timeframe)
            multi_bar_signals = self.detect_multi_bar_patterns(df, timeframe)
            custom_signals = self.detect_custom_moroccan_patterns(df, timeframe)

            all_signals = pattern_signals + multi_bar_signals + custom_signals

            if not all_signals:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'patterns': [],
                    'reasoning': 'No significant patterns detected'
                }

            # Calculate confluence
            confluence_score = self.pattern_confluence(all_signals)

            # Determine overall signal
            bullish_patterns = [s for s in all_signals if s.pattern_type in
                              ['CDLHAMMER', 'CDLMORNINGSTAR', 'CDLENGULFING'] and s.context != "Downtrend"]
            bearish_patterns = [s for s in all_signals if s.pattern_type in
                              ['CDLEVENINGSTAR', 'CDL3BLACKCROWS'] and s.context != "Uptrend"]

            if len(bullish_patterns) > len(bearish_patterns) and confluence_score > 0.6:
                signal = 'BUY'
            elif len(bearish_patterns) > len(bullish_patterns) and confluence_score > 0.6:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'signal': signal,
                'confidence': confluence_score,
                'patterns': [{'type': s.pattern_type, 'strength': s.strength,
                            'volume_confirmed': s.volume_confirmed} for s in all_signals],
                'reasoning': f"Detected {len(all_signals)} patterns with confluence score {confluence_score:.2f}"
            }

        except Exception as e:
            print(f"Error in get_trading_signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'patterns': [],
                'reasoning': f'Error in pattern analysis: {str(e)}'
            }