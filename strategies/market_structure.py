import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from config import Config

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
    mitigated: bool = False

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
        self.config = Config()

    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Identify swing highs and lows with a strength threshold.
        """
        highs, lows = [], []
        
        for i in range(self.min_swing_length, len(df) - self.min_swing_length):
            is_high = df['high'].iloc[i] == df['high'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].max()
            is_significant_high = (df['high'].iloc[i] - df['low'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].min()) > (df['close'].iloc[i] * self.swing_strength_threshold)

            if is_high and is_significant_high:
                highs.append(i)

            is_low = df['low'].iloc[i] == df['low'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].min()
            is_significant_low = (df['high'].iloc[i-self.min_swing_length:i+self.min_swing_length+1].max() - df['low'].iloc[i]) > (df['close'].iloc[i] * self.swing_strength_threshold)
                
            if is_low and is_significant_low:
                lows.append(i)
        
        return highs, lows

    def detect_market_structure(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detects Break of Structure (BOS) and Change of Character (CHoCH).
        """
        events = []
        highs, lows = self.identify_swing_points(df)
        
        swing_highs = sorted([{'index': i, 'price': df['high'].iloc[i]} for i in highs], key=lambda x: x['index'])
        swing_lows = sorted([{'index': i, 'price': df['low'].iloc[i]} for i in lows], key=lambda x: x['index'])

        # Simplified trend detection
        last_swing_high = swing_highs[-1] if swing_highs else None
        last_swing_low = swing_lows[-1] if swing_lows else None

        if not last_swing_high or not last_swing_low:
            return []

        # Bullish trend if last swing high is higher than previous, and last swing low is higher than previous
        is_uptrend = (len(swing_highs) > 1 and len(swing_lows) > 1 and 
                      swing_highs[-1]['price'] > swing_highs[-2]['price'] and 
                      swing_lows[-1]['price'] > swing_lows[-2]['price'])
        
        # Bearish trend
        is_downtrend = (len(swing_highs) > 1 and len(swing_lows) > 1 and 
                        swing_highs[-1]['price'] < swing_highs[-2]['price'] and 
                        swing_lows[-1]['price'] < swing_lows[-2]['price'])

        for i in range(1, len(df)):
            if is_uptrend:
                # BOS: Closing above the last swing high
                if df['close'].iloc[i] > last_swing_high['price'] and df['close'].iloc[i-1] <= last_swing_high['price']:
                    events.append({'type': 'BOS', 'direction': 'bullish', 'index': i, 'price': df['close'].iloc[i]})
                # CHoCH: Closing below the last swing low
                if df['close'].iloc[i] < last_swing_low['price'] and df['close'].iloc[i-1] >= last_swing_low['price']:
                    events.append({'type': 'CHoCH', 'direction': 'bearish', 'index': i, 'price': df['close'].iloc[i]})

            elif is_downtrend:
                # BOS: Closing below the last swing low
                if df['close'].iloc[i] < last_swing_low['price'] and df['close'].iloc[i-1] >= last_swing_low['price']:
                    events.append({'type': 'BOS', 'direction': 'bearish', 'index': i, 'price': df['close'].iloc[i]})
                # CHoCH: Closing above the last swing high
                if df['close'].iloc[i] > last_swing_high['price'] and df['close'].iloc[i-1] <= last_swing_high['price']:
                    events.append({'type': 'CHoCH', 'direction': 'bullish', 'index': i, 'price': df['close'].iloc[i]})
        
        return events


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
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i+1],
                    'gap_size': df['low'].iloc[i+1] - df['high'].iloc[i-1]
                })
            # Bearish FVG
            elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                fvgs.append({
                    'type': 'bearish',
                    'start_idx': i-1,
                    'end_idx': i+1,
                    'high': df['high'].iloc[i+1],
                    'low': df['low'].iloc[i-1],
                    'gap_size': df['low'].iloc[i-1] - df['high'].iloc[i+1]
                })
                
        return fvgs

    def detect_liquidity_zones(self, df: pd.DataFrame, lookback: int = 20, consolidation_threshold: float = 0.002, volume_multiplier: float = 1.5) -> List[Dict]:
        """Detects liquidity zones based on price consolidation and high volume."""
        zones = []
        rolling_avg_vol = df['tick_volume'].rolling(window=lookback).mean()

        for i in range(lookback, len(df)):
            price_range = df.iloc[i-lookback:i]
            price_std_dev = price_range['close'].std()
            price_mean = price_range['close'].mean()

            # Check for consolidation (low price volatility)
            if price_std_dev < price_mean * consolidation_threshold:
                # Check for high volume
                if df['tick_volume'].iloc[i] > rolling_avg_vol.iloc[i] * volume_multiplier:
                    zones.append({
                        'start_idx': i-lookback,
                        'end_idx': i,
                        'avg_price': price_mean,
                        'volume': df['tick_volume'].iloc[i],
                        'type': 'consolidation_high_volume'
                    })
        return zones
    
    def detect_inducement(self, df: pd.DataFrame, highs: List[int], lows: List[int]) -> List[Dict]:
        """
        Detects inducement points (minor highs/lows) before a major swing.
        """
        inducement_points = []

        # For an uptrend, look for a minor low before a new high is made
        for i in range(1, len(highs)):
            prev_high_idx = highs[i-1]
            curr_high_idx = highs[i]
            
            # Find the lowest point between the two highs
            intermediate_lows = [l for l in lows if prev_high_idx < l < curr_high_idx]
            if intermediate_lows:
                inducement_low = min(intermediate_lows, key=lambda l: df['low'].iloc[l])
                inducement_points.append({'type': 'inducement_low', 'index': inducement_low, 'price': df['low'].iloc[inducement_low]})

        # For a downtrend, look for a minor high before a new low is made
        for i in range(1, len(lows)):
            prev_low_idx = lows[i-1]
            curr_low_idx = lows[i]
            
            # Find the highest point between the two lows
            intermediate_highs = [h for h in highs if prev_low_idx < h < curr_low_idx]
            if intermediate_highs:
                inducement_high = max(intermediate_highs, key=lambda h: df['high'].iloc[h])
                inducement_points.append({'type': 'inducement_high', 'index': inducement_high, 'price': df['high'].iloc[inducement_high]})
        
        return inducement_points

    def analyze_trend_context(self, df: pd.DataFrame, window: int = 50) -> str:
        """Analyze overall trend context (uptrend, downtrend, range)"""
        sma = df['close'].rolling(window).mean()
        if df['close'].iloc[-1] > sma.iloc[-1] and sma.iloc[-1] > sma.iloc[-2]:
            return "Uptrend"
        elif df['close'].iloc[-1] < sma.iloc[-1] and sma.iloc[-1] < sma.iloc[-2]:
            return "Downtrend"
        else:
            return "Range"

    def detect_order_blocks(self, df: pd.DataFrame, min_size_pips: int = 15) -> List[OrderBlock]:
        """Detect institutional order blocks"""
        blocks = []
        
        # Get pip value from config
        # Assuming a default symbol if not specified, or you can pass it as an argument
        symbol_config = self.config.get_symbol_config(self.config.DEFAULT_SYMBOL)
        pip_value = symbol_config.pip_value

        for i in range(1, len(df)):
            # Bullish Order Block: A down candle before a strong up move
            if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                move_size = df['high'].iloc[i] - df['low'].iloc[i-1]
                if move_size > min_size_pips * pip_value:
                    blocks.append(OrderBlock(
                        start_idx=i-1,
                        end_idx=i-1,
                        high=df['high'].iloc[i-1],
                        low=df['low'].iloc[i-1],
                        type='bullish',
                        strength=move_size
                    ))
            
            # Bearish Order Block: An up candle before a strong down move
            if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                move_size = df['high'].iloc[i-1] - df['low'].iloc[i]
                if move_size > min_size_pips * pip_value:
                    blocks.append(OrderBlock(
                        start_idx=i-1,
                        end_idx=i-1,
                        high=df['high'].iloc[i-1],
                        low=df['low'].iloc[i-1],
                        type='bearish',
                        strength=move_size
                    ))
        return blocks