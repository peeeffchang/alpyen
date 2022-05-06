from arch import arch_model
import enum
from eventkit import Event
from itertools import accumulate
import numpy as np
import random
from typing import List, Dict, Union

from . import datacontainer
from . import signal
from . import strategy
from . import utils


class PathGenerationType(enum.Enum):
    """
    Enum class for path generation type.
    """
    ReturnShuffling = 1
    ReturnResampling = 2
    GARCH_1_0_1 = 3


class MetricType(enum.Enum):
    """
    Enum class for performance metric.
    """
    PoorMansSharpeRatio = 1  # Assuming zero risk free rate
    MaximumDrawDown = 2
    Return = 3


class Backtester:
    """
    Class for backtester.
    """
    def __init__(self,
                 aggregated_input: List[datacontainer.PriceTimeSeries],
                 names: List[str],
                 signal_info: Dict[str, utils.SignalInfo],
                 strategy_info: Dict[str, utils.StrategyInfo],
                 number_simulation: int = 1) -> None:
        """
        Initialize backtester.

        Parameters
        ----------
        aggregated_input: List[PriceTimeSeries]
            All input data used for backtesting.
        names: List[str]
            Data event names.
        signal_info: Dict[str, utils.SignalInfo]
            Information for building signals.
        strategy_info: Dict[str, utils.StrategyInfo]
            Information for building strategies.
        number_simulation: int
            Number of simulation paths.
        """
        self._aggregated_input = aggregated_input
        self._names = names
        self._signal_info = signal_info
        self._strategy_info = strategy_info
        self._number_simulation = number_simulation
        self._results: Dict[str, Dict[str, List[float]]] = {}
        self._fitted_path_generation_model: Dict[str, ] = {}

    def run_backtest(self,
                     path_type: PathGenerationType = PathGenerationType.ReturnShuffling) -> None:
        """
        Run backtest.

        Parameters
        ----------
        path_type: PathGenerationType
            Path generation type.
        """
        # Build dictionaries to store simulation results
        simulation_results_dict: Dict[str, Dict[str, List[float]]] = {}
        for key in self._strategy_info.keys():
            simulation_results_dict[key] = {}

        # Fit model for path simulation
        for j in range(len(self._aggregated_input)):
            self._fitted_path_generation_model[self._aggregated_input[j].get_name()] = \
                self._fit_path_generation_model(self._aggregated_input[j].get_price(), path_type)

        for sim_i in range(self._number_simulation):
            # Create incoming data
            data_event_dict: Dict[str, Event] = {}
            for name in self._names:
                data_event_dict[name] = Event(name)
            # In case there are traded contracts that are not in any signal
            for k, v in self._strategy_info.items():
                for i in range(len(v.get_contract_names())):
                    if v.get_contract_names()[i] not in data_event_dict:
                        data_event_dict[v.get_contract_names()[i]] = Event(v.get_contract_names()[i])

            # Create signals
            signal_dict = self.create_signal_dict(data_event_dict)

            # Create strategies
            strategy_dict = self.create_strategy_dict(signal_dict, data_event_dict)

            # Read and fire data; the first trial is always the actual history
            price_dict: Dict[str, List[float]] = {}
            for j in range(len(self._aggregated_input)):
                if sim_i == 0:
                    price_dict[self._aggregated_input[j].get_name()] = self._aggregated_input[j].get_price()
                else:
                    price_dict[self._aggregated_input[j].get_name()] = \
                        self.generate_simulated_path(self._aggregated_input[j].get_price(),
                                                     self._fitted_path_generation_model[
                                                         self._aggregated_input[j].get_name()
                                                     ],
                                                     path_type)

            for t in range(len(price_dict[next(iter(price_dict))])):
                for k, v in data_event_dict.items():
                    data_name = k
                    timestamp = self._aggregated_input[0].get_timestamp()[t]
                    price = price_dict[data_name][t]
                    v.emit(datacontainer.TimeDouble(data_name, timestamp, price))

            # Save results
            for k, v in strategy_dict.items():
                simulation_results_dict[k][k + str(sim_i)] = v.get_mtm_history()
        # Analyze results and print out
        metrics_to_calculate = [MetricType.PoorMansSharpeRatio,
                                MetricType.MaximumDrawDown,
                                MetricType.Return]
        for key in self._strategy_info.keys():
            metrics = self.calculate_metrics(simulation_results_dict[key], metrics_to_calculate)
            self._results[key] = metrics

    def get_results(self):
        return self._results

    def create_signal_dict(self,
                           data_event_dict: Dict[str, Event]) -> Dict[str, signal.SignalBase]:
        """
        Create signal dictionary.

        Parameters
        ----------
        data_event_dict: Dict[str, Event]
            Data event dictionary.
        """
        output_dict: Dict[str, signal.SignalBase] = {}
        signal_info_dict: Dict[str, utils.SignalInfo] = self._signal_info
        for k, v in signal_info_dict.items():
            price_event_list: List[Event] = []
            for name_i in v.get_input_names():
                price_event_list.append(data_event_dict.get(name_i))
            # Create signal
            output_dict[k] = signal.SignalBase(v.get_signal_signature(),
                                               price_event_list,
                                               v.get_warmup_length(),
                                               **v.get_custom_params())
        return output_dict

    def create_strategy_dict(self,
                             signal_dict: Dict[str, signal.SignalBase],
                             data_event_dict: Dict[str, Event]) -> Dict[str, strategy.StrategyBase]:
        """
        Create strategy dictionary.

        Parameters
        ----------
        signal_dict: Dict[str, signal.SignalBase]
            Signal dictionary.
        data_event_dict: Dict[str, Event]
            Data event dictionary.
        """
        output_dict: Dict[str, strategy.StrategyBase] = {}
        strategy_info_dict: Dict[str, utils.StrategyInfo] = self._strategy_info
        for k, v in strategy_info_dict.items():
            price_event_list: List[Event] = []
            for i in range(len(v.get_contract_names())):
                price_event_list.append(data_event_dict.get(v.get_contract_names()[i]))
            # Create TradeCombos
            trade_combos = strategy.TradeCombos(price_event_list, v.get_combo_definition(), v.get_order_types())
            # Create strategy
            signal_event_list: List[Event] = []
            for i in range(len(v.get_input_names())):
                signal_event_list.append(signal_dict.get(v.get_input_names()[i]).get_signal_event())
            output_dict[k] = strategy.StrategyBase(v.get_strategy_signature(), signal_event_list,
                                                   trade_combos, v.get_warmup_length(), **v.get_custom_params())
        return output_dict

    def _fit_path_generation_model(self,
                                   original_data: List[float],
                                   path_type: PathGenerationType = PathGenerationType.ReturnShuffling):
        """
        Fit the model that is used for path generation

        Parameters
        ----------
        original_data: List[float]
            Original price data array.
        path_type: PathGenerationType
            Path generation type.

        Returns
        -------
            Union[None, arch.univariate.base.ARCHModelResult]
                Fitted model for path generation.
        """
        if path_type == PathGenerationType.ReturnShuffling:
            return None
        elif path_type == PathGenerationType.ReturnResampling:
            return None
        elif path_type == PathGenerationType.GARCH_1_0_1:
            return self._fit_garch_1_0_1_model(original_data)
        else:
            raise ValueError('Backtester.generate_simulated_path: Unknown generation type.')

    def _fit_garch_1_0_1_model(self, original_data: List[float]):
        """
        Fit the model that is used for path generation

        Parameters
        ----------
        original_data: List[float]
            Original price data array.

        Returns
        -------
            arch.univariate.base.ARCHModelResult
                Fitted model for path generation.
        """
        # Calculate return from price
        return_list = [p_t / p_tm1 - 1 for p_t, p_tm1 in zip(original_data[1:], original_data[:-1])]

        # Fitting
        scaled_return_list = [100 * r for r in return_list]
        scaled_return_array = np.reshape(scaled_return_list, [len(scaled_return_list), 1])
        arch_model_ = arch_model(scaled_return_array, vol='GARCH', p=1, o=0, q=1, dist='normal')
        arch_model_fitted = arch_model_.fit(first_obs=0)
        return arch_model_fitted

    def generate_simulated_path(self,
                                original_data: List[float],
                                fitted_model,
                                path_type: PathGenerationType = PathGenerationType.ReturnShuffling) -> List[float]:
        """
        Generate simulated path from original price data.

        Parameters
        ----------
        original_data: List[float]
            Original price data array.
        fitted_model
            Fitted model for path generation.
        path_type: PathGenerationType
            Path generation type.

        Returns
        -------
            List[float]
                Simulated price data.
        """
        if path_type == PathGenerationType.ReturnShuffling:
            return self.generate_shuffled_return_path(original_data)
        elif path_type == PathGenerationType.ReturnResampling:
            return self.generate_resampled_return_path(original_data)
        elif path_type == PathGenerationType.GARCH_1_0_1:
            return self.generate_garch_1_0_1_return_path(original_data, fitted_model)
        else:
            raise ValueError('Backtester.generate_simulated_path: Unknown generation type.')

    def generate_shuffled_return_path(self,
                                      original_data: List[float]) -> List[float]:
        """
        Generate shuffled path from original price data.

        Parameters
        ----------
        original_data: List[float]
            Original price data array.

        Returns
        -------
            List[float]
                Simulated price data.
        """
        # Calculate return from price
        return_list = [p_t / p_tm1 - 1 for p_t, p_tm1 in zip(original_data[1:], original_data[:-1])]

        # Shuffle
        random.shuffle(return_list)

        # Create new price series
        price_t0 = original_data[0]
        return_list = [price_t0] + return_list
        new_data = [*accumulate(return_list, lambda a, b: a * (1.0 + b))]
        return new_data

    def generate_resampled_return_path(self,
                                       original_data: List[float]) -> List[float]:
        """
        Generate resampled path from original price data.

        Parameters
        ----------
        original_data: List[float]
            Original price data array.

        Returns
        -------
            List[float]
                Simulated price data.
        """
        # Calculate return from price
        return_list = [p_t / p_tm1 - 1 for p_t, p_tm1 in zip(original_data[1:], original_data[:-1])]

        # Sample
        random_indices = list(range(0, len(return_list), 1))
        random.shuffle(random_indices)
        return_randomized = [return_list[i] for i in random_indices]

        # Create new price series
        price_t0 = original_data[0]
        return_randomized = [price_t0] + return_randomized
        new_data = [*accumulate(return_randomized, lambda a, b: a * (1.0 + b))]
        return new_data

    def generate_garch_1_0_1_return_path(self,
                                         original_data: List[float],
                                         fitted_model) -> List[float]:
        """
        Generate GARCH(1,0,1) path from original price data.

        Parameters
        ----------
        original_data: List[float]
            Original price data array.
        fitted_model: arch.univariate.base.ARCHModelResult
            Fitted Garch model.
        Returns
        -------
            List[float]
                Simulated price data.
        """
        # Forecast
        garch_forecasts = fitted_model.forecast(start=0, method='simulation', simulations=2)
        simulated_return = np.reshape(garch_forecasts.simulations.values[:, 0, :], [len(original_data) - 1]) / 100.0

        # Create new price series
        price_t0 = original_data[0]
        return_list = [price_t0] + simulated_return.tolist()
        new_data = [*accumulate(return_list, lambda a, b: a * (1.0 + b))]
        return new_data

    def calculate_metrics(self,
                          simulation_results: Dict[str, List[float]],
                          metrics_to_calculate: List[MetricType]) -> Dict[str, List[float]]:
        """
        Calculate a set of specifiec metrics.

        Parameters
        ----------
        simulation_results: Dict[str, List[float]]
            Simulation results.
        metrics_to_calculate: List[MetricType]
            Entities to calculate

        Returns
        -------
            Dict[str, List[float]]
                Metrics.
        """
        output = {}

        for i in range(len(metrics_to_calculate)):
            metric_storage = []
            for k, v in simulation_results.items():
                return_list = [p_t / p_tm1 - 1 for p_t, p_tm1 in zip(v[1:], v[:-1])]
                price_list = v
                if metrics_to_calculate[i] == MetricType.PoorMansSharpeRatio:
                    metric_storage.append(self.calculate_poor_mans_sharpe_ratio(return_list))
                elif metrics_to_calculate[i] == MetricType.MaximumDrawDown:
                    metric_storage.append(self.calculate_mdd(price_list))
                elif metrics_to_calculate[i] == MetricType.Return:
                    metric_storage.append(price_list[-1] / price_list[0] - 1.0)
                else:
                    raise ValueError('Backtester.calculate_metrics: Unknown metrics type.')
            output[str(metrics_to_calculate[i])] = metric_storage
        return output

    def calculate_poor_mans_sharpe_ratio(self,
                                         return_list: List[float]) -> float:
        """
        Calculate 'poor man's Sharpe ratio' (i.e. assuming risk-free rate is zero)

        Parameters
        ----------
        return_list: List[float]
            Return series.

        Returns
        -------
            float
                Poor man's Sharpe ratio.
        """
        if len(return_list) == 0:
            return 0.0
        else:
            average = sum(return_list) / len(return_list)
            deviation_squared = [(x - average)**2 for x in return_list]
            return average / (sum(deviation_squared) / (len(deviation_squared) - 1))**0.5

    def calculate_mdd(self,
                      price_list: List[float]) -> float:
        """
        Calculate maximum drawdown

        Parameters
        ----------
        price_list: List[float]
            Price series.

        Returns
        -------
            float
                Maximum drawdown.
        """
        mdd = 0.0
        peak = price_list[0]
        for p_i in price_list:
            if p_i > peak:
                peak = p_i
            dd = (peak - p_i) / peak
            if dd > mdd:
                mdd = dd
        return mdd
