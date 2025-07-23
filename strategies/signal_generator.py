import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import Config
from .market_structure import MarketStructure
from .volume_analysis import VolumeAnalysis
from .pattern_analysis import PatternAnalysis
from .session_analysis import SessionAnalysis
from .risk_management import RiskManagement
from utils.helpers import TradeSignal
from utils.logger import TradingLogger

class SignalGenerator:
    """
    Generates high-confidence trading signals by finding confluence
    among various analytical methods.
    """
    def __init__(self, market_structure: MarketStructure, volume_analysis: VolumeAnalysis,
                 pattern_analysis: PatternAnalysis, session_analysis: SessionAnalysis,
                 risk_management: RiskManagement, htf_timeframe: str = 'H1',
                 fvg_sensitivity: float = 0.002, order_block_sensitivity: float = 0.005):
        """Initializes the SignalGenerator with all required analysis components."""
        self.market_structure = market_structure
        self.volume_analysis = volume_analysis
        self.pattern_analysis = pattern_analysis
        self.session_analysis = session_analysis
        self.risk_manager = risk_management
        self.logger = TradingLogger(self.__class__.__name__)

        # Configuration for signal evaluation
        self.min_confluence_factors = 2  # Adjusted for strategy-specific logic
        self.htf_timeframe = htf_timeframe
        self.fvg_sensitivity = fvg_sensitivity
        self.order_block_sensitivity = order_block_sensitivity

    async def generate_signal(self, dfs: Dict[str, pd.DataFrame], timeframe: str, symbol: str) -> Optional[TradeSignal]:
        """
        Main method to generate a signal. Routes to the correct strategy
        based on the symbol's configuration.
        """
        symbol_config = Config.get_symbol_config(symbol)
        
        if symbol_config.strategy_name == "gold_reversal":
            return await self._generate_gold_reversal_signal(dfs, timeframe, symbol, symbol_config)
        elif symbol_config.strategy_name == "btc_breakout":
            return await self._generate_btc_breakout_signal(dfs, timeframe, symbol, symbol_config)
        else:
            return await self._generate_default_signal(dfs, timeframe, symbol, symbol_config)

    async def _analyze_market_data(self, dfs: Dict[str, pd.DataFrame], timeframe: str):
        """Helper to perform common analysis on market data."""
        df = dfs[timeframe]
        htf_df = dfs.get(self.htf_timeframe)

        mtf_confirmed = False
        if htf_df is not None and not htf_df.empty:
            mtf_confirmed = self.higher_timeframe_confirmation(df, htf_df)

        df_with_delta = self.volume_analysis.calculate_volume_delta(df.copy())
        
        analysis = {
            "df": df_with_delta,
            "mtf_confirmed": mtf_confirmed,
            "structure_breaks": self.market_structure.detect_market_structure_break(df_with_delta),
            "patterns": self.pattern_analysis.analyze_patterns(df_with_delta, timeframe),
            "order_blocks": self.market_structure.detect_order_blocks(df_with_delta),
            "fvgs": self.market_structure.identify_fair_value_gaps(df_with_delta)
        }
        analysis["pattern_score"] = self.pattern_analysis.pattern_confluence(analysis["patterns"])
        analysis["volume_confirmed"] = any(p.volume_confirmed for p in analysis["patterns"])
        
        return analysis

    async def _generate_gold_reversal_signal(self, dfs, timeframe, symbol, config):
        """Generates a signal for GOLD based on mean-reversion and liquidity sweeps."""
        self.logger.info(f"[{symbol}] Running GOLD reversal strategy...")
        analysis = await self._analyze_market_data(dfs, timeframe)
        df = analysis["df"]
        
        if not analysis["fvgs"]:
            return None

        last_fvg = analysis["fvgs"][-1]
        direction = "BUY" if last_fvg['type'] == 'bullish' else "SELL"
        
        reasons = [f"Primary Signal: Reacting to a {last_fvg['type']} Fair Value Gap."]
        confidence_factors = 1
        current_price = df['close'].iloc[-1]
        
        # Confluence: Is FVG near a recent liquidity sweep (break of structure)?
        if analysis["structure_breaks"]:
            last_break = analysis["structure_breaks"][-1]
            if (direction == "BUY" and last_break['type'] == 'bullish') or \
               (direction == "SELL" and last_break['type'] == 'bearish'):
                reasons.append("FVG is confirmed by a recent market structure break.")
                confidence_factors += 1

        # Confluence: Higher-Timeframe Alignment
        if analysis["mtf_confirmed"]:
            reasons.append("Direction aligns with the higher-timeframe trend.")
            confidence_factors += 1

        # Confluence: Reversal candlestick patterns
        if analysis["pattern_score"] > 0.6 and analysis["volume_confirmed"]:
            reasons.append("Confirmed by strong reversal candlestick patterns.")
            confidence_factors += 1

        return await self._finalize_signal(df, symbol, direction, reasons, confidence_factors, config, timeframe)

    async def _generate_btc_breakout_signal(self, dfs, timeframe, symbol, config):
        """Generates a signal for BTC based on breakout logic."""
        self.logger.info(f"[{symbol}] Running BTC breakout strategy...")
        analysis = await self._analyze_market_data(dfs, timeframe)
        df = analysis["df"]

        if not analysis["structure_breaks"]:
            return None

        last_break = analysis["structure_breaks"][-1]
        direction = "BUY" if last_break['type'] == 'bullish' else "SELL"
        
        reasons = [f"Primary Signal: {last_break['type']} break of structure (Breakout)."]
        confidence_factors = 1
        
        # Confluence: Volume confirmation on the breakout candle
        if analysis["volume_confirmed"]:
            reasons.append("Breakout confirmed by high volume delta.")
            confidence_factors += 1

        # Confluence: Higher-Timeframe Alignment
        if analysis["mtf_confirmed"]:
            reasons.append("Direction aligns with the higher-timeframe trend.")
            confidence_factors += 1

        return await self._finalize_signal(df, symbol, direction, reasons, confidence_factors, config, timeframe)

    async def _generate_default_signal(self, dfs, timeframe, symbol, config):
        """Runs the standard Smart Money Concepts confluence strategy."""
        self.logger.info(f"[{symbol}] Running default SMC strategy...")
        analysis = await self._analyze_market_data(dfs, timeframe)
        df = analysis["df"]

        if not analysis["structure_breaks"]:
            return None

        last_break = analysis["structure_breaks"][-1]
        direction = "BUY" if last_break['type'] == 'bullish' else "SELL"
        
        reasons = [f"Primary Signal: {last_break['type']} break of structure."]
        confidence_factors = 1

        # Add other confluence checks from original logic (Order Block, FVG, etc.)
        if analysis["mtf_confirmed"]:
            reasons.append("Aligned with HTF trend.")
            confidence_factors += 1
        
        # Simplified check for FVG
        if any(fvg['type'] == ('bullish' if direction == 'BUY' else 'bearish') for fvg in analysis['fvgs']):
             reasons.append("Price is near a relevant FVG.")
             confidence_factors += 1

        return await self._finalize_signal(df, symbol, direction, reasons, confidence_factors, config, timeframe)

    async def _finalize_signal(self, df, symbol, direction, reasons, confidence_factors, config, timeframe):
        """Common final step to create and return a TradeSignal object."""
        if confidence_factors < self.min_confluence_factors:
            return None

        confidence = min(1.0, 0.5 + (confidence_factors * 0.15))
        
        if confidence < config.min_confidence:
            self.logger.info(f"[{symbol}] Signal confidence {confidence:.2f} is below threshold {config.min_confidence}.")
            return None

        current_price = df['close'].iloc[-1]
        if 'atr' not in df.columns:
            df['atr'] = self.risk_manager.calculate_atr(df)
        
        stop_loss, take_profit = await self.risk_manager.calculate_dynamic_exits(df, current_price, direction, symbol)

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=timeframe,
            reasons=reasons
        )

    def higher_timeframe_confirmation(self, df: pd.DataFrame, htf_df: pd.DataFrame) -> bool:
        """Confirms if the short-term signal aligns with the long-term trend."""
        htf_trend = self.market_structure.analyze_trend_context(htf_df)
        ltf_direction = "Uptrend" if df['close'].iloc[-1] > df['open'].iloc[-1] else "Downtrend"

        if (ltf_direction == "Uptrend" and htf_trend == "Uptrend") or \
           (ltf_direction == "Downtrend" and htf_trend == "Downtrend"):
            return True
        return False