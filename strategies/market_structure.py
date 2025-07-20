import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class StructurePoint:
    index: int
    price: float
    type: str  # 'high' or 'low'
    strength: float
    confirmed: bool

@dataclass
class OrderBlock:
    start_idx: int
    end_idx: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    strength: float

class MarketStructure:
    def __init__(self, min_swing_length: int = 5, swing_strength_threshold: float = 0.001, min_break_size: float = 0.005):
        self.min_swing_length = min_swing_length
        self.swing_strength_threshold = swing_strength_threshold # New parameter for swing point filtering
        self.min_break_size = min_break_size # New parameter for market structure break filtering
        self.structure_points: List[StructurePoint] = []
        self.order_blocks: List[OrderBlock] = []
        
    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Identify swing highs and lows with a strength threshold.
        A swing point is only considered if its price movement from the lowest low
        (for a high) or highest high (for a low) within its window exceeds the threshold.
        """
        highs, lows = [], []
        
        for i in range(self.min_swing_length, len(df) - self.min_swing_length):
            # Swing high detection
            if (df['high'].iloc[i] == df['high'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].max() and
                df['high'].iloc[i] - df['low'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].min() > df['close'].iloc[i] * self.swing_strength_threshold):
                highs.append(i)
                
            # Swing low detection
            if (df['low'].iloc[i] == df['low'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].min() and
                df['high'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].max() - df['low'].iloc[i] > df['close'].iloc[i] * self.swing_strength_threshold):
                lows.append(i)
        
        return highs, lows
    
    def detect_market_structure_break(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect breaks in market structure, filtering by a minimum break size.
        """
        breaks = []
        highs, lows = self.identify_swing_points(df)
        
        # Sort swing points by index to process chronologically
        all_swing_points = sorted([(idx, df['high'].iloc[idx], 'high') for idx in highs] + 
                                  [(idx, df['low'].iloc[idx], 'low') for idx in lows])

        if len(all_swing_points) < 2:
            return breaks

        for i in range(1, len(all_swing_points)):
            prev_idx, prev_price, prev_type = all_swing_points[i-1]
            curr_idx, curr_price, curr_type = all_swing_points[i]

            # Consider only actual "breaks" that move beyond previous significant points
            if prev_type == 'high' and curr_type == 'low':
                # Check for bearish market structure break
                # If current low is significantly lower than previous swing low, it's a bearish break
                if df['low'].iloc[curr_idx] < df['low'].iloc[prev_idx] - self.min_break_size: # Simplified for demonstration
                     breaks.append({
                        'index': curr_idx,
                        'type': 'bearish',
                        'price': df['low'].iloc[curr_idx]
                    })
            elif prev_type == 'low' and curr_type == 'high':
                # Check for bullish market structure break
                # If current high is significantly higher than previous swing high, it's a bullish break
                if df['high'].iloc[curr_idx] > df['high'].iloc[prev_idx] + self.min_break_size: # Simplified for demonstration
                    breaks.append({
                        'index': curr_idx,
                        'type': 'bullish',
                        'price': df['high'].iloc[curr_idx]
                    })
        
        # Re-evaluating original logic with min_break_size
        # The original logic checks for lower lows after highs (bearish) and higher highs after lows (bullish)
        # We need to apply the min_break_size to these checks.
        
        for i in range(1, len(highs)):
            # Bearish break: current low is below the low of the *previous swing high*
            # and the magnitude of the break is significant
            if df['low'].iloc[highs[i]] < df['low'].iloc[highs[i-1]] - self.min_break_size:
                breaks.append({
                    'index': highs[i],
                    'type': 'bearish',
                    'price': df['low'].iloc[highs[i]]
                })
                
        for i in range(1, len(lows)):
            # Bullish break: current high is above the high of the *previous swing low*
            # and the magnitude of the break is significant
            if df['high'].iloc[lows[i]] > df['high'].iloc[lows[i-1]] + self.min_break_size:
                breaks.append({
                    'index': lows[i],
                    'type': 'bullish',
                    'price': df['high'].iloc[lows[i]]
                })

        return breaks
    
    def identify_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Identify fair value gaps in price action"""
        fvgs = []
        
        for i in range(1, len(df)-1):
            if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                fvgs.append({
                    'type': 'bullish',
                    'start_idx': i-1,
                    'end_idx': i+1,
                    'gap_size': df['low'].iloc[i+1] - df['high'].iloc[i-1]
                })
            elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                fvgs.append({
                    'type': 'bearish',
                    'start_idx': i-1,
                    'end_idx': i+1,
                    'gap_size': df['low'].iloc[i-1] - df['high'].iloc[i+1]
                })
                
        return fvgs
    
    def calculate_dynamic_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Calculate dynamic support and resistance levels"""
        levels = {'support': [], 'resistance': []}
        
        for i in range(window, len(df)):
            section = df.iloc[i-window:i]
            # Suppress FutureWarning by adding observed=False
            price_clusters = pd.qcut(section['close'], q=5, duplicates='drop', observed=False)
            freq = price_clusters.value_counts()
            
            # Identify significant levels
            for level in freq.index:
                if freq[level] > window * 0.2:  # 20% threshold
                    if level.left > df['close'].iloc[i]:
                        levels['resistance'].append(level.left)
                    else:
                        levels['support'].append(level.right)
        
        return levels
    
    def detect_liquidity_zones(self, df: pd.DataFrame, window: int = 20) -> List[Dict]:
        """Detect liquidity zones where price consolidates and volume is high"""
        zones = []
        for i in range(window, len(df)):
            section = df.iloc[i-window:i]
            price_std = section['close'].std()
            avg_volume = section['tick_volume'].mean()
            if price_std < section['close'].mean() * 0.002 and avg_volume > section['tick_volume'].mean() * 1.5:
                zones.append({
                    'start_idx': i-window,
                    'end_idx': i,
                    'avg_price': section['close'].mean(),
                    'avg_volume': avg_volume
                })
        return zones

    def analyze_trend_context(self, df: pd.DataFrame, window: int = 50) -> str:
        """Analyze overall trend context (uptrend, downtrend, range)"""
        sma = df['close'].rolling(window).mean()
        if df['close'].iloc[-1] > sma.iloc[-1]:
            return "Uptrend"
        elif df['close'].iloc[-1] < sma.iloc[-1]:
            return "Downtrend"
        else:
            return "Range"

    def enhanced_swing_point_detection(self, df: pd.DataFrame) -> List[StructurePoint]:
        """Detect swing highs/lows with strength scoring"""
        points = []
        highs, lows = self.identify_swing_points(df)
        for idx in highs:
            strength = (df['high'].iloc[idx] - df['low'].iloc[idx]) / df['close'].iloc[idx]
            points.append(StructurePoint(idx, df['high'].iloc[idx], 'high', strength, True))
        for idx in lows:
            strength = (df['high'].iloc[idx] - df['low'].iloc[idx]) / df['close'].iloc[idx]
            points.append(StructurePoint(idx, df['low'].iloc[idx], 'low', strength, True))
        return points

    def dynamic_support_resistance_levels(self, df: pd.DataFrame, window: int = 50) -> Dict[str, List[float]]:
        """Dynamic S/R levels based on swing points and clustering"""
        highs, lows = self.identify_swing_points(df)
        support = [df['low'].iloc[i] for i in lows]
        resistance = [df['high'].iloc[i] for i in highs]
        # Cluster levels to reduce noise
        support = sorted(set([round(s, 2) for s in support]))
        resistance = sorted(set([round(r, 2) for r in resistance]))
        return {'support': support, 'resistance': resistance}

    def detect_order_blocks(self, df: pd.DataFrame, min_size: float = 15) -> List[OrderBlock]:
        """Detect institutional order blocks (basic version)"""
        blocks = []
        highs, lows = self.identify_swing_points(df)
        for i in range(1, len(highs)):
            size = abs(df['high'].iloc[highs[i]] - df['low'].iloc[highs[i-1]])
            if size >= min_size:
                blocks.append(OrderBlock(
                    start_idx=highs[i-1],
                    end_idx=highs[i],
                    high=df['high'].iloc[highs[i]],
                    low=df['low'].iloc[highs[i-1]],
                    type='bullish' if df['close'].iloc[highs[i]] > df['open'].iloc[highs[i]] else 'bearish',
                    strength=size
                ))
        return blocks

    def multi_timeframe_swing_points(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[List[int], List[int]]]:
        """Detect swing points across multiple timeframes (placeholder)"""
        mtf_swings = {}
        for tf, df in dfs.items():
            mtf_swings[tf] = self.identify_swing_points(df)
        return mtf_swings