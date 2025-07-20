import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VolumeDelta:
    timestamp: pd.Timestamp
    delta: float
    cumulative: float
    strength: float
    imbalance: bool
    exhaustion: bool
    momentum: float

class VolumeAnalysis:
    def __init__(self, delta_threshold: float = 0.6, exhaustion_multiplier: float = 2.0):
        self.delta_threshold = delta_threshold
        self.exhaustion_multiplier = exhaustion_multiplier
        
    def calculate_volume_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate buying vs selling pressure (Delta per Candle)"""
        df['volume_delta'] = df['tick_volume'] * (2 * (df['close'] > df['open']) - 1)
        df['cumulative_delta'] = df['volume_delta'].cumsum()
        df['delta_strength'] = df['volume_delta'].abs() / df['tick_volume']
        
        return df
    
    def calculate_delta_momentum(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate the momentum of the volume delta"""
        return df['volume_delta'].rolling(window=window).mean()

    def detect_volume_exhaustion(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect volume exhaustion signals"""
        avg_volume = df['tick_volume'].rolling(window=window).mean()
        return df['tick_volume'] > (avg_volume * self.exhaustion_multiplier)

    def detect_volume_imbalance(self, df: pd.DataFrame, window: int = 20) -> List[dict]:
        """Detect areas of volume imbalance"""
        imbalances = []
        vol_mean = df['tick_volume'].rolling(window=window).mean()
        vol_std = df['tick_volume'].rolling(window=window).std()
        
        for i in range(window, len(df)):
            if df['tick_volume'].iloc[i] > vol_mean.iloc[i] + 2 * vol_std.iloc[i]:
                imbalances.append({
                    'index': i,
                    'type': 'high',
                    'volume': df['tick_volume'].iloc[i]
                })
                
        return imbalances
    
    def analyze_volume_profile(self, df: pd.DataFrame, price_levels: int = 50) -> Dict[float, float]:
        """Create volume profile analysis"""
        price_bins = pd.qcut(df['close'], q=price_levels, duplicates='drop')
        volume_profile = df.groupby(price_bins, observed=False)['tick_volume'].sum()
        
        return dict(volume_profile)
    
    def detect_absorption(self, df: pd.DataFrame, window: int = 5) -> List[dict]:
        """Detect volume absorption patterns"""
        absorption_zones = []
        
        for i in range(window, len(df)):
            price_range = df['high'].iloc[i] - df['low'].iloc[i]
            avg_range = (df['high'].iloc[i-window:i] - df['low'].iloc[i-window:i]).mean()
            
            if (price_range < avg_range * 0.5 and 
                df['tick_volume'].iloc[i] > df['tick_volume'].iloc[i-window:i].mean() * 2):
                absorption_zones.append({
                    'index': i,
                    'volume': df['tick_volume'].iloc[i],
                    'price': df['close'].iloc[i]
                })
                
        return absorption_zones
    
    def analyze_volume_delta_profile(self, df: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
        """Enhanced volume delta analysis with microstructure features"""
        if 'volume_delta' not in df.columns:
            df = self.calculate_volume_delta(df)

        results = {}
        
        results['buy_pressure'] = df['volume_delta'].clip(lower=0)
        results['sell_pressure'] = df['volume_delta'].clip(upper=0).abs()
        
        price_volume = df.groupby(pd.qcut(df['close'], q=50, duplicates='drop'), observed=False)['tick_volume'].sum()
        results['volume_profile'] = price_volume
        
        results['delta_momentum'] = self.calculate_delta_momentum(df, window)
        
        results['volume_exhaustion'] = self.detect_volume_exhaustion(df, window)
        
        return results
    
    def delta_divergence(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect price vs volume delta divergence"""
        price_change = df['close'].pct_change(window)
        delta_change = df['volume_delta'].pct_change(window)
        return price_change * delta_change < 0

    def cumulative_delta_by_session(self, df: pd.DataFrame, session_hours: Tuple[int, int]) -> float:
        """Calculate cumulative delta for a session"""
        mask = (df.index.hour >= session_hours[0]) & (df.index.hour < session_hours[1])
        session_df = df[mask]
        if 'volume_delta' not in session_df.columns:
            session_df = self.calculate_volume_delta(session_df)
        return session_df['volume_delta'].sum() if not session_df.empty else 0

    def volume_at_price_distribution(self, df: pd.DataFrame, session_hours: Tuple[int, int], bins: int = 20) -> Dict[float, float]:
        """Volume at price for a session"""
        mask = (df.index.hour >= session_hours[0]) & (df.index.hour < session_hours[1])
        session_df = df[mask]
        price_bins = pd.cut(session_df['close'], bins, duplicates='drop')
        volume_profile = session_df.groupby(price_bins, observed=False)['tick_volume'].sum()
        return dict(volume_profile)