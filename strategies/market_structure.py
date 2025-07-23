import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN

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

@dataclass
class SupportResistanceZone:
    level: float
    strength: int
    type: str  # 'support' or 'resistance'

class MarketStructure:
    def __init__(self, min_swing_length: int = 5, swing_strength_threshold: float = 0.001, min_break_size: float = 0.005, dbscan_eps: float = 0.01, dbscan_min_samples: int = 2):
        self.min_swing_length = min_swing_length
        self.swing_strength_threshold = swing_strength_threshold
        self.min_break_size = min_break_size
        self.structure_points: List[StructurePoint] = []
        self.order_blocks: List[OrderBlock] = []
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Identify swing highs and lows with a strength threshold.
        """
        highs, lows = [], []
        
        for i in range(self.min_swing_length, len(df) - self.min_swing_length):
            is_high = df['high'].iloc[i] == df['high'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].max()
            is_low = df['low'].iloc[i] == df['low'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].min()

            if is_high:
                highs.append(i)
            if is_low:
                lows.append(i)
        
        return highs, lows

    def multi_timeframe_swing_points(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[List[int], List[int]]]:
        """Detect swing points across multiple timeframes."""
        mtf_swings = {}
        for tf, df in dfs.items():
            mtf_swings[tf] = self.identify_swing_points(df)
        return mtf_swings

    def enhanced_dynamic_support_resistance(self, df: pd.DataFrame) -> List[SupportResistanceZone]:
        """
        Calculates dynamic support and resistance levels using clustering.
        """
        highs, lows = self.identify_swing_points(df)
        
        support_prices = df['low'].iloc[lows].values.reshape(-1, 1)
        resistance_prices = df['high'].iloc[highs].values.reshape(-1, 1)

        zones = []

        if len(support_prices) > self.dbscan_min_samples:
            db_support = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(support_prices)
            for label in set(db_support.labels_):
                if label != -1:
                    cluster_prices = support_prices[db_support.labels_ == label]
                    zones.append(SupportResistanceZone(
                        level=cluster_prices.mean(),
                        strength=len(cluster_prices),
                        type='support'
                    ))

        if len(resistance_prices) > self.dbscan_min_samples:
            db_resistance = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(resistance_prices)
            for label in set(db_resistance.labels_):
                if label != -1:
                    cluster_prices = resistance_prices[db_resistance.labels_ == label]
                    zones.append(SupportResistanceZone(
                        level=cluster_prices.mean(),
                        strength=len(cluster_prices),
                        type='resistance'
                    ))

        return zones

    def detect_market_structure_break(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect breaks in market structure (BOS and CHoCH).
        """
        breaks = []
        highs, lows = self.identify_swing_points(df)
        
        swing_points = sorted([(idx, df['high'].iloc[idx], 'high') for idx in highs] + 
                                  [(idx, df['low'].iloc[idx], 'low') for idx in lows])

        if len(swing_points) < 2:
            return breaks

        # Simplified BOS/CHoCH detection
        last_high = None
        last_low = None
        for i in range(len(df)):
            if i in highs:
                if last_high and df['high'].iloc[i] > last_high[1]:
                    breaks.append({'index': i, 'type': 'BOS', 'price': df['high'].iloc[i], 'direction': 'bullish'})
                last_high = (i, df['high'].iloc[i])
            if i in lows:
                if last_low and df['low'].iloc[i] < last_low[1]:
                     breaks.append({'index': i, 'type': 'BOS', 'price': df['low'].iloc[i], 'direction': 'bearish'})
                last_low = (i, df['low'].iloc[i])
        
        return breaks

    def identify_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Identify fair value gaps in price action"""
        fvgs = []
        
        for i in range(1, len(df)-1):
            # Bullish FVG
            if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                fvgs.append({
                    'type': 'bullish',
                    'start_idx': i-1,
                    'end_idx': i+1,
                    'top': df['low'].iloc[i+1],
                    'bottom': df['high'].iloc[i-1],
                    'gap_size': df['low'].iloc[i+1] - df['high'].iloc[i-1]
                })
            # Bearish FVG
            elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                fvgs.append({
                    'type': 'bearish',
                    'start_idx': i-1,
                    'end_idx': i+1,
                    'top': df['low'].iloc[i-1],
                    'bottom': df['high'].iloc[i+1],
                    'gap_size': df['low'].iloc[i-1] - df['high'].iloc[i+1]
                })
                
        return fvgs
    
    def detect_liquidity_zones(self, df: pd.DataFrame, window: int = 20) -> List[Dict]:
        """Detect liquidity zones where price consolidates and volume is high"""
        zones = []
        for i in range(window, len(df)):
            section = df.iloc[i-window:i]
            price_std = section['close'].std()
            avg_volume = section['tick_volume'].mean()
            if price_std < section['close'].mean() * 0.002 and avg_volume > df['tick_volume'].mean() * 1.5: # MODIFIED
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

    def detect_order_blocks(self, df: pd.DataFrame, min_size: float = 15) -> List[OrderBlock]:
        """Detect institutional order blocks (basic version)"""
        blocks = []
        highs, lows = self.identify_swing_points(df)
        
        # Bullish Order Blocks
        for low in lows:
            if low > 0 and df['close'].iloc[low-1] < df['open'].iloc[low-1]: # Previous candle is bearish
                blocks.append(OrderBlock(
                    start_idx=low-1,
                    end_idx=low-1,
                    high=df['high'].iloc[low-1],
                    low=df['low'].iloc[low-1],
                    type='bullish',
                    strength=df['high'].iloc[low-1] - df['low'].iloc[low-1]
                ))

        # Bearish Order Blocks
        for high in highs:
            if high > 0 and df['close'].iloc[high-1] > df['open'].iloc[high-1]: # Previous candle is bullish
                blocks.append(OrderBlock(
                    start_idx=high-1,
                    end_idx=high-1,
                    high=df['high'].iloc[high-1],
                    low=df['low'].iloc[high-1],
                    type='bearish',
                    strength=df['high'].iloc[high-1] - df['low'].iloc[high-1]
                ))
        return blocks