import numpy as np

class GannCalculator:
    """
    Calculates Take Profit and Stop Loss levels based on Gann Square Root Theory.
    """

    def calculate_gann_tp(self, entry_price: float, harmonic_increment: float = 0.125) -> float:
        """
        Calculates the Take Profit level using the Gann Square Root method.
        """
        if entry_price <= 0:
            return 0.0
        sqrt_price = np.sqrt(entry_price)
        projected_sqrt = sqrt_price + harmonic_increment
        tp_level = projected_sqrt ** 2
        return tp_level

    def calculate_gann_sl(self, entry_price: float, harmonic_decrement: float = 0.125) -> float:
        """
        Calculates the Stop Loss level using the Gann Square Root method.
        """
        if entry_price <= 0:
            return 0.0
        sqrt_price = np.sqrt(entry_price)
        projected_sqrt = sqrt_price - harmonic_decrement
        sl_level = projected_sqrt ** 2
        return sl_level