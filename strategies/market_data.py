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
            strength_check_high = df['high'].iloc[i] - df['low'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].min() > df['close'].iloc[i] * self.swing_strength_threshold

            if is_high and strength_check_high:
                highs.append(i)

            is_low = df['low'].iloc[i] == df['low'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].min()
            strength_check_low = df['high'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].max() - df['low'].iloc[i] > df['close'].iloc[i] * self.swing_strength_threshold

            if is_low and strength_check_low:
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

    def detect_market_data_break(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect breaks in market structure, filtering by a minimum break size.
        """
        breaks = []
        highs, lows = self.identify_swing_points(df)

        all_swing_points = sorted([(idx, df['high'].iloc[idx], 'high') for idx in highs] +
                                  [(idx, df['low'].iloc[idx], 'low') for idx in lows])

        if len(all_swing_points) < 2:
            return breaks

        # Simplified break detection
        last_high = None
        last_low = None

        for idx, price, type in all_swing_points:
            if type == 'high':
                if last_low is not None and price > last_low[1] + self.min_break_size:
                     if last_high is None or price > last_high[1]:
                        breaks.append({'index': idx, 'type': 'bullish', 'price': price})
                last_high = (idx, price)
            elif type == 'low':
                if last_high is not None and price < last_high[1] - self.min_break_size:
                    if last_low is None or price < last_low[1]:
                        breaks.append({'index': idx, 'type': 'bearish', 'price': price})
                last_low = (idx, price)


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
                    'gap_size': df['low'].iloc[i+1] - df['high'].iloc[i-1]
                })
            # Bearish FVG
            elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                fvgs.append({
                    'type': 'bearish',
                    'start_idx': i-1,
                    'end_idx': i+1,
                    'gap_size': df['low'].iloc[i-1] - df['high'].iloc[i+1]
                })

        return fvgs

    def detect_liquidity_zones(self, df: pd.DataFrame, window: int = 20) -> List[Dict]:
        """Detect liquidity zones where price consolidates and volume is high"""
        zones = []
        # Calculate rolling standard deviation of price and rolling mean of volume
        price_std = df['close'].rolling(window=window).std()
        avg_volume = df['tick_volume'].rolling(window=window).mean()

        # Identify zones where price is consolidating (low std dev) and volume is high
        # The constant 0.002 is a sensitivity factor for consolidation
        consolidation_threshold = df['close'].rolling(window=window).mean() * 0.002
        # The constant 1.5 is a sensitivity factor for volume spikes
        volume_threshold = avg_volume * 1.5

        # Vectorized check for zones
        is_liquidity_zone = (price_std < consolidation_threshold) & (df['tick_volume'] > volume_threshold)

        zone_indices = np.where(is_liquidity_zone)[0]

        for i in zone_indices:
             if i > window:
                zones.append({
                    'start_idx': i-window,
                    'end_idx': i,
                    'avg_price': df['close'].iloc[i-window:i].mean(),
                    'avg_volume': df['tick_volume'].iloc[i-window:i].mean()
                })
        return zones

    def analyze_trend_context(self, df: pd.DataFrame, window: int = 50) -> str:
        """Analyze overall trend context (uptrend, downtrend, range)"""
        if len(df) < window:
            return "Range" # Not enough data for trend analysis

        sma = df['close'].rolling(window).mean()
        if df['close'].iloc[-1] > sma.iloc[-1] * 1.001: # Add a small buffer to avoid noise
            return "Uptrend"
        elif df['close'].iloc[-1] < sma.iloc[-1] * 0.999: # Add a small buffer
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

    def detect_order_blocks(self, df: pd.DataFrame, min_size_pips: float = 15) -> List[OrderBlock]:
        """Detect institutional order blocks (basic version)"""
        blocks = []

        # Find the last down-candle before a strong up-move (Bullish OB)
        # and the last up-candle before a strong down-move (Bearish OB)
        for i in range(1, len(df)-1):
            # Potential Bullish OB: A down candle followed by a strong up candle
            if df['close'].iloc[i-1] < df['open'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i]:
                move_size = df['close'].iloc[i] - df['open'].iloc[i]
                if move_size / df['close'].iloc[i] > (min_size_pips * 0.0001): # Assuming pips for forex
                    blocks.append(OrderBlock(
                        start_idx=i-1, end_idx=i-1,
                        high=df['high'].iloc[i-1], low=df['low'].iloc[i-1],
                        type='bullish', strength=move_size
                    ))
            # Potential Bearish OB: An up candle followed by a strong down candle
            if df['close'].iloc[i-1] > df['open'].iloc[i-1] and df['close'].iloc[i] < df['open'].iloc[i]:
                move_size = df['open'].iloc[i] - df['close'].iloc[i]
                if move_size / df['close'].iloc[i] > (min_size_pips * 0.0001):
                    blocks.append(OrderBlock(
                        start_idx=i-1, end_idx=i-1,
                        high=df['high'].iloc[i-1], low=df['low'].iloc[i-1],
                        type='bearish', strength=move_size
                    ))
        return blocks