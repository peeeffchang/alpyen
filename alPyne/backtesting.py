import enum
from typing import List, Dict

from . import datacontainer


class PathGenerationType(enum.Enum):
    """
    Enum class for path generation type.
    """
    ReturnShuffling = 1
    ReturnResampling = 2


class MetricType(enum.Enum):
    """
    Enum class for performance metric.
    """
    PoorMansSharpeRatio = 1  # Assuming zero risk free rate
    MaximumDrawDown = 2
    Return = 3


class SignalInfo:
    """
    class for signal info.
    """
    def __init__(self,
                 input_names: List[str],
                 params: List[float]) -> None:
        self._input_names = input_names
        self._params = params

    def get_input_names(self) -> List[str]:
        return self._input_names

    def get_params(self) -> List[float]:
        return self._params


class StrategyInfo:
    """
    class for strategy info.
    """
    def __init__(self,
                 input_names: List[str],
                 params: List[float],
                 combo_names: List[str]) -> None:
        self._input_names = input_names
        self._params = params
        self._combo_names = combo_names

    def get_input_names(self) -> List[str]:
        return self._input_names

    def get_params(self) -> List[float]:
        return self._params

    def get_combo_names(self) -> List[str]:
        return self._combo_names


class Backtester:
    """
    Class for backtester.
    """
    def __init__(self,
                 aggregated_input: List[datacontainer.PriceTimeSeries],
                 names: List[str],
                 signal_info: Dict[str, SignalInfo],
                 strategy_info: Dict[str, StrategyInfo],
                 number_simulation: int = 1) -> None:
        """
        Initialize backtester.

        Parameters
        ----------
        aggregated_input: List[PriceTimeSeries]
            All input data used for backtesting.
        names: List[str]
            Time of the realization of the data point.
        signal_info: Dict[str, SignalInfo]
            Information for building signals.
        strategy_info: Dict[str, StrategyInfo]
            Information for building strategies.
        number_simulation: int
            Number of simulation paths.
        """
        self._aggregated_input = aggregated_input
        self._names = names
        self._signal_info = signal_info
        self._strategy_info = strategy_info
        self._number_simulation = number_simulation

    def run_backtest(self) -> None:
        pass
