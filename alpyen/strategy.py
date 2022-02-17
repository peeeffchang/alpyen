from abc import ABC, abstractmethod
from collections import deque
from datetime import timedelta
from datetime import datetime
import enum
from eventkit import Event
import math
import pandas as pd
from typing import List, Deque, Dict

from . import brokerinterface
from . import datacontainer
from . import utils


class WeightingScheme(enum.Enum):
    """
    Enum class for portfolio weighting scheme
    """
    Equal = 1


class TradeCombos:
    """
    Class for trade combos, which is a basket of contracts traded by a strategy.
    """

    def __init__(self,
                 contract_array: List[Event],
                 combo_definition: Dict[str, List[float]] = None) -> None:
        """
        Initialize trade combo.

        Parameters
        ----------
        contract_array: List[Event]
            List of event subscriptions on contracts to be traded.
        combo_definition: Dict[str, List[float]]
            A dictionary, with keys being combo names and values being weights to be traded.
        """
        if combo_definition is not None:
            for key, value in combo_definition.items():
                if len(contract_array) != len(value):
                    raise ValueError('TradeCombo.__init__: Different numbers of contracts and weights.')
            self._combo_definition = combo_definition
        else:
            self._combo_definition = None
        self._contract_array = contract_array

    def get_combo_def(self) -> Dict[str, List[float]]:
        if self._combo_definition is not None:
            return self._combo_definition
        else:
            return None

    def get_contract_array(self) -> List[Event]:
        return self._contract_array


class StrategyBase(ABC):
    """
    Base class for strategy.
    """
    CASH_ACCOUNT_NAME = 'cash_account'

    _strategy_classes_registry = {}

    _strategy_signature = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._strategy_signature is None:
            raise KeyError('StrategyBase: Missing signature for ' + str(cls))
        elif cls._strategy_signature in cls._strategy_classes_registry:
            raise KeyError('StrategyBase: Conflict in signature ' + cls._strategy_signature)
        else:
            cls._strategy_classes_registry[cls._strategy_signature] = cls

    @classmethod
    def get_class_registry(cls):
        return cls._strategy_classes_registry

    def __new__(cls,
                signature_str: str,
                input_signal_array: List[Event],
                trade_combo: TradeCombos,
                warmup_length: int,
                initial_capital: float = 100.0,
                order_manager: brokerinterface.OrderManagerBase = None,
                **kwargs):
        """
        Create derived strategy object.

        Parameters
        ----------
        signature_str: str
            Unique signature of the strategy.
        input_signal_array: List[Event]
            List of event subscription that is required for trade decision.
        trade_combo: TradeCombos
            TradeCombos object that defines the combos to be traded. Note that if combo definition is missing,
            all trades from this strategy will be under a generic combo.
        warmup_length: int
            Number of data points to 'burn'.
        initial_capital: float
            Initial capital.
        order_manager: OrderManager
            OrderManager object for live trading.
        """
        if signature_str not in cls.get_class_registry():
            raise ValueError('StrategyBase.__new__: ' + signature_str + ' is not a valid key.')

        my_strategy_obj = super().__new__(cls.get_class_registry()[signature_str])

        # TBD: Use pandas df to store some of the member fields
        my_strategy_obj._warmup_length = warmup_length
        my_strategy_obj._mtm = initial_capital
        if order_manager is None:
            my_strategy_obj._is_live_trading = False
        else:
            my_strategy_obj._is_live_trading = True
            my_strategy_obj._order_manager = order_manager

        my_strategy_obj._strategy_active = True  # A strategy is active by default
        my_strategy_obj._initialize_signal_time_storage(input_signal_array)
        contract_array = trade_combo.get_contract_array()
        my_strategy_obj._contract_array = contract_array
        if not trade_combo.get_combo_def() is None:
            my_strategy_obj._combo_def = trade_combo.get_combo_def()
        else:
            my_strategy_obj._combo_def = None
        my_strategy_obj._initialize_contract_time_storage(contract_array)
        my_strategy_obj._input_signal_array = input_signal_array

        position_temp = {security.name(): 0.0 for security in contract_array}
        # Remember to add cash account
        position_temp[my_strategy_obj.CASH_ACCOUNT_NAME] = initial_capital
        my_strategy_obj._contract_positions = position_temp

        my_strategy_obj._combo_positions = {combo_name: 0.0 for combo_name in my_strategy_obj._combo_def.keys()}

        my_strategy_obj._combo_order = {combo_name: 0.0 for combo_name in my_strategy_obj._combo_def.keys()}
        my_strategy_obj._average_entry_price = {security.name(): 0.0 for security in contract_array}
        my_strategy_obj._mtm_price = {security.name(): 0.0 for security in contract_array}

        my_strategy_obj._mtm_history = [initial_capital]

        return my_strategy_obj

    def __init__(self,
                 signature_str: str,
                 input_signal_array: List[Event],
                 trade_combo: TradeCombos,
                 warmup_length: int,
                 initial_capital: float = 100.0,
                 order_manager: brokerinterface.OrderManagerBase = None,
                 **kwargs) -> None:
        pass

    def _initialize_signal_time_storage(self, input_signal_array: List[Event]) -> None:
        """
        Initialize storage.

        Parameters
        ----------
        input_signal_array: List[Event]
            List of signal subscription that is required for ordering.
        """
        signal_storage = {}
        signal_time_storage = {}
        signal_slot_storage = []
        for input_signal in input_signal_array:
            signal_storage[input_signal.name()] = deque([])
            signal_time_storage[input_signal.name()] = deque([])

            this_signal_slot = SignalSlot(input_signal.name(), [self])
            signal_slot_storage.append(this_signal_slot)
            # Connect event to slot
            input_signal += this_signal_slot.on_event
        self._signal_storage = signal_storage
        self._signal_time_storage = signal_time_storage
        self._signal_slot_storage = signal_slot_storage

    def _initialize_contract_time_storage(self, contract_array: List[Event]) -> None:
        """
        Initialize storage.

        Parameters
        ----------
        contract_array: List[Event]
            List of contracts.
        """
        contract_time_storage = {}
        mtm_price_slot_storage = []
        for input_event in contract_array:
            contract_time_storage[input_event.name()] = deque([])

            this_mtm_price_slot = MTMPriceSlot(input_event.name(), [self])
            mtm_price_slot_storage.append(this_mtm_price_slot)
            # Connect event to slot
            input_event += this_mtm_price_slot.on_event
        self._contract_time_storage = contract_time_storage
        self._mtm_price_slot_storage = mtm_price_slot_storage

    def get_strategy_active(self) -> bool:
        return self._strategy_active

    def get_strategy_name(self) -> str:
        return self._strategy_name

    def set_strategy_active(self, activity: bool) -> None:
        self._strategy_active = activity

    def check_all_signals_received(self, signal_name: str) -> bool:
        output = True
        num_signal = len(self._input_signal_array)
        if num_signal == 1:
            return output  # If there is only one incoming signal stream, no need to check
        for i in range(num_signal):
            event_i = self._input_signal_array[i]
            if len(self.get_signal_time_by_name(event_i.name())) == 0:
                return False

            time_diff = self.get_signal_time_by_name(signal_name)[-1] - self.get_signal_time_by_name(event_i.name())[-1]
            if time_diff > timedelta(microseconds=1):
                return False
        return output

    def update_data(self, new_data: datacontainer.TimeDouble) -> None:
        """
        Update signal storage.

        Parameters
        ----------
        new_data: TimeDouble
            Incoming new data.
        """
        # Extend storage
        self._signal_storage[new_data.get_name()].append(new_data.get_value())
        self._signal_time_storage[new_data.get_name()].append(new_data.get_time())
        # Remove oldest if warming up is complete
        if len(self._signal_storage[new_data.get_name()]) > self._warmup_length:
            self._signal_storage[new_data.get_name()].popleft()
            self._signal_time_storage[new_data.get_name()].popleft()

    def get_signal_time_by_name(self, signal_name: str) -> Deque:
        return self._signal_time_storage[signal_name]

    def get_signal_by_name(self, signal_name: str) -> Deque:
        return self._signal_storage[signal_name]

    def check_all_contract_data_received(self, data_name: str) -> bool:
        output = True
        num_data = len(self._contract_array)
        if num_data == 1:
            return output  # If there is only one incoming data stream, no need to check
        for i in range(num_data):
            event_name_i = self._contract_array[i].name()
            if len(self.get_contract_time_by_name(event_name_i)) == 0:
                return False

            time_diff = self.get_contract_time_by_name(data_name)[-1] - self.get_contract_time_by_name(event_name_i)[-1]
            if time_diff > timedelta(microseconds=1):
                return False
        return output

    def update_mtm(self, new_data: datacontainer.TimeDouble) -> None:
        """
        Update data storage used for MTM calculation.

        Parameters
        ----------
        new_data: TimeDouble
            Incoming new data.
        """
        self._mtm_price[new_data.get_name()] = new_data.get_value()
        self.get_contract_time_by_name(new_data.get_name()).append(new_data.get_time())
        if len(self.get_contract_time_by_name(new_data.get_name())) > 1:
            self.get_contract_time_by_name(new_data.get_name()).popleft()
        # Update PM combo mtm price
        if self._is_live_trading:
            # Build a dataframe
            combo_mtm = pd.DataFrame(columns=['strategy_name', 'combo_name', 'combo_mtm_price'])
            for combo_name in self._combo_def.keys():
                combo_mtm = combo_mtm.append({'strategy_name': self._strategy_name,
                                              'combo_name': combo_name,
                                              'combo_mtm_price': self.get_combo_mtm_price(combo_name)},
                                             ignore_index=True)
            # Update
            self._order_manager.get_portfolio_manager().update_combo_mtm_price(combo_mtm)
        pass

    def get_mtm_history(self) -> List[float]:
        return self._mtm_history

    def get_mtm_price_by_name(self, data_name: str) -> float:
        """Return the MTM price of a contract."""
        return self._mtm_price[data_name]

    def get_combo_mtm_price(self, combo_name: str) -> float:
        """
        Retrieve combo mtm price.


        Parameters
        ----------
        combo_name: str
            Combo name.
        """
        combo_def = self._combo_def[combo_name]
        return sum(i[0] * i[1] for i in zip(combo_def, self._mtm_price.values()))

    def get_contract_time_by_name(self, data_name: str) -> Deque:
        return self._contract_time_storage[data_name]

    def get_signal_containing_string(self,
                                     input_string: str) -> Deque:
        """
        Retrieve the signals that contain a certain string in the name.

        Parameters
        ----------
        input_string: str
            String that is supposed to be part of the name.

        Returns
        -------
            Deque
                The signal data that is stored.
        """
        for key, value in self._signal_storage.items():
            if input_string in key:
                return value
        # If not found
        raise ValueError('StrategyBase.get_signal_containing_string: Signal not found.')

    def send_order(self) -> None:
        """
        Send pending orders.
        """
        if self._is_live_trading:
            self.send_order_live()
        else:
            self.send_order_backtest()

    def send_order_backtest(self) -> None:
        """
        Send pending orders and use latest price to calculate average cost.
        """
        if self._combo_order is None:
            return
        for combo_name, amount in self._combo_order.items():
            if combo_name in self._combo_def:
                combo_def = self._combo_def[combo_name]
                self._combo_positions[combo_name] += amount
                for weight_i, contract_i in zip(combo_def, self._contract_array):
                    # Adjust cash account
                    name_i = contract_i.name()
                    self._contract_positions[self.CASH_ACCOUNT_NAME] -= amount * weight_i * self._mtm_price[name_i]
                    # Calculate average cost
                    self._average_entry_price[name_i] = self._calculate_new_average_entry_price(
                        self._average_entry_price[name_i],
                        self._mtm_price[name_i],
                        self._contract_positions[name_i],
                        amount * weight_i)
                    # Modify current position
                    self._contract_positions[name_i] += amount * weight_i
            else:
                # Some strategies do not have predefined combos;
                # for them we simply place orders for individual contract.
                contract_name = combo_name
                # Adjust cash account
                self._contract_positions[self.CASH_ACCOUNT_NAME] -= amount * self._mtm_price[contract_name]
                # Calculate average cost
                self._average_entry_price[contract_name] = self._calculate_new_average_entry_price(
                    self._average_entry_price[contract_name],
                    self._mtm_price[contract_name],
                    self._contract_positions[contract_name],
                    amount)
                # Modify current position
                self._contract_positions[contract_name] += amount
            # Reset pending order
            self._combo_order[combo_name] = 0.0

    def send_order_live(self) -> None:
        """
        Send pending orders to broker.
        """
        if self._combo_order is None:
            return
        contract_array = [self._order_manager.get_event_contract_dict()[i.name()] for i in self._contract_array]
        for combo_name, amount in self._combo_order.items():
            if combo_name in self._combo_def:
                combo_def = self._combo_def[combo_name]
                self._order_manager.register_combo_level_info(self._strategy_name,
                                                              contract_array,
                                                              combo_def,
                                                              combo_name)
                time_stamp = str(datetime.now())
                for i in range(len(combo_def)):
                    self._order_manager.place_order(self._strategy_name,
                                                    combo_name,
                                                    time_stamp,
                                                    i,
                                                    contract_array[i],
                                                    amount
                                                    )
                self._combo_positions[combo_name] += amount
            else:
                # Some strategies do not have predefined combos;
                # for them we simply place orders for individual contract.
                raise ValueError('StrategyBase.send_order_live: Non-combo ordering not implemented yet.')

    def set_combo_order(self, combo_order: Dict[str, float]) -> None:
        """Set combo order manually (used exclusively for liquidation)"""
        self._combo_order = combo_order

    def _calculate_new_average_entry_price(self,
                                           old_average_entry_price: float,
                                           new_entry_price: float,
                                           existing_position: float,
                                           new_pending_order: float) -> float:
        """
        Calculate new average entry price.

        Parameters
        ----------
        old_average_entry_price: float
            Existing average entry price.
        new_entry_price: float
            New entry price.
        existing_position: float
            Existing position.
        new_pending_order: float
            New pending order.

        Returns
        -------
            float
                The new average entry price.
        """
        if math.copysign(1, existing_position) != math.copysign(1, existing_position + new_pending_order):
            # If going from net long to net short or vice versa
            return new_entry_price
        elif abs(existing_position + new_pending_order) > 0.0:
            return (existing_position * old_average_entry_price + new_pending_order * new_entry_price) / (
                existing_position + new_pending_order)
        else:
            return new_entry_price

    def calculate_mtm(self) -> None:
        """
        Mark to market.
        """
        mtm = 0.0
        for key, value in self._mtm_price.items():
            mtm += value * self._contract_positions[key]
        mtm += self._contract_positions[self.CASH_ACCOUNT_NAME]
        self._update_mtm_history(mtm)

    def _update_mtm_history(self, new_mtm: float) -> None:
        """
        Update MTM history.

        Parameters
        ----------
        new_mtm: float
            Latest MTM.
        """
        self._mtm_history.append(new_mtm)

    def has_pending_order(self) -> bool:
        """
        Check if there is any pending order.
        """
        for key, value in self._combo_order.items():
            if abs(value) > 0.0:
                return True
        return False

    def has_specific_pending_order(self, combo_name: str) -> bool:
        """
        Check if a specific contract has any pending order.
        """
        if abs(self._combo_order[combo_name]) > 0.0:
            return True
        else:
            return False

    def is_currently_long(self, combo_name: str) -> bool:
        return self._combo_positions[combo_name] > 0.0

    def is_currently_short(self, combo_name: str) -> bool:
        return self._combo_positions[combo_name] <= 0.0

    def is_currently_flat(self, combo_name: str) -> bool:
        return abs(self._combo_positions[combo_name]) <= utils.EPSILON

    def get_warmup_length(self) -> int:
        return self._warmup_length

    def get_is_live_trading(self) -> bool:
        return self._is_live_trading

    @abstractmethod
    def make_order_decision(self) -> Dict[str, float]:
        """
        Virtual method to be implemented by the derived class.

        Returns
        -------
            Dict[str, float]
                A dictionary of combo name to trade, and amount to trade.
        """
        pass


class MACrossingStrategy(StrategyBase):
    """
    MA Crossing Strategy
    """

    _strategy_signature = 'MACrossing'

    def __init__(self,
                 signature_str: str,
                 input_signal_array: List[Event],
                 trade_combo: TradeCombos,
                 warmup_length: int,
                 initial_capital: float = 100.0,
                 order_manager: brokerinterface.OrderManagerBase = None) -> None:
        """
        Initialize MA Crossing Strategy.

        Parameters
        ----------
        signature_str: str
            Unique signature of the strategy.
        input_signal_array: List[Event]
            List of event subscription that is required for trade decision.
        trade_combo: TradeCombos
            TradeCombos object that defines the combos to be traded. Note that if combo definition is missing,
            all trades from this strategy will be under a generic combo.
        warmup_length: int
            Number of data points to 'burn'.
        initial_capital: float
            Initial capital.
        order_manager: OrderManager
            OrderManager object for live trading.
        """
        if len(trade_combo.get_combo_def()) != 1:
            raise ValueError('MACrossingStrategy.__init__: Too many combos specified.')
        self._strategy_name = input_signal_array[0].name() + self._strategy_signature
        self._long_signal_name, self._short_signal_name = self._deduce_signal_name(input_signal_array)

    def _deduce_signal_name(self, input_signal_array: List[Event]) -> (str, str):
        """
        Deduce the name of the signal with longer (shorter) average window.

        Parameters
        ----------
        input_signal_array: List[Event]
            List of event subscription that is required for trade decision.

        Returns
        -------
            str, str
                The long and short window signal names respectively.
        """
        name_1 = input_signal_array[0].name()
        name_2 = input_signal_array[1].name()
        delimiter = '_'
        substr_1 = name_1.split(delimiter)[-1]
        substr_2 = name_2.split(delimiter)[-1]
        if not (isinstance(int(substr_1), int) and isinstance(int(substr_2), int)):
            raise TypeError('MACrossingStrategy._deduce_signal_name: Signal name in wrong format.')
        if int(substr_1) > int(substr_2):
            return name_1, name_2
        else:
            return name_2, name_1

    def make_order_decision(self) -> Dict[str, float]:
        combo_name = next(iter(self._combo_def))
        current_price = self.get_combo_mtm_price(combo_name)
        if not self.has_specific_pending_order(combo_name):
            if self._signal_storage[self._long_signal_name][-1] >= self._signal_storage[self._short_signal_name][-1] \
                    and (self.is_currently_long(combo_name) or self.is_currently_flat(combo_name)):
                return {combo_name: -self._mtm_history[-1] / current_price - self._combo_positions[combo_name]}
            elif self._signal_storage[self._long_signal_name][-1] < self._signal_storage[self._short_signal_name][-1] \
                    and (self.is_currently_short(combo_name) or self.is_currently_flat(combo_name)):
                return {combo_name: self._mtm_history[-1] / current_price - self._combo_positions[combo_name]}
            else:
                return {combo_name: 0.0}
        else:
            return {combo_name: 0.0}


class VAAStrategy(StrategyBase):
    """
    VAAStrategy
    https://indexswingtrader.blogspot.com/2017/07/breadth-momentum-and-vigilant-asset.html
    Given a top selection T and a breadth protection threshold B, for each month:
    - Compute 13612W momentum for each asset
    - Pick the best performing assets in the "risk-on" universe as top T
    - Pick the best asset in the "risk-off" universe as safety asset for "cash"
    - Compute the number of assets with non-positive momentum in the "risk-on" universe (b)
    - Compute b/B and round down to multiples of 1/T as "cash fraction" CF for "easy trading"
    - Replace CF of top T by "cash" asset as selected in step 3
    """

    _strategy_signature = 'VAA'

    def __init__(self,
                 signature_str: str,
                 input_signal_array: List[Event],
                 trade_combo: TradeCombos,
                 warmup_length: int,
                 initial_capital: float = 100.0,
                 order_manager: brokerinterface.OrderManagerBase = None,
                 **kwargs) -> None:
        """
        Initialize VAA rotation Strategy.

        Parameters
        ----------
        signature_str: str
            Unique signature of the strategy.
        input_signal_array: List[Event]
            List of event subscription that is required for trade decision.
        trade_combo: TradeCombos
            TradeCombos object that defines the combos to be traded. Note that if combo definition is missing,
            all trades from this strategy will be under a generic combo.
        warmup_length: int
            Number of data points to 'burn'.
        initial_capital: float
            Initial capital.
        order_manager: OrderManager
            OrderManager object for live trading.

        Other Parameters
        ----------------
        risk_on_size: int
            Number of risk-on assets within the input signals.
        num_assets_to_hold: int
            Portfolio size in terms of number of holding.
        breadth_protection_threshold: int
            Threshold for protection.
        weighting_scheme: WeightingScheme
            Portfolio weighting scheme.
        """
        if len(trade_combo.get_contract_array()) < kwargs['num_assets_to_hold']:
            raise ValueError('VAAStrategy.__init__: num_assets_to_hold must be smaller than universe size.')
        if len(input_signal_array) != len(trade_combo.get_contract_array()):
            raise ValueError('VAAStrategy.__init__: input_signal_array and trade combo contracts '
                             'should be of the same size.')
        all_input_names = [x.name() for x in input_signal_array]
        self._strategy_name = ''.join(all_input_names) + "_" + self._strategy_signature
        self._num_assets_to_hold = kwargs['num_assets_to_hold']
        self._weighting_scheme = kwargs['weighting_scheme']
        self._breadth_protection_threshold = kwargs['breadth_protection_threshold']
        self._risk_on_names, self._risk_off_names = self.initialize_names(input_signal_array, kwargs['risk_on_size'])

    def initialize_names(self,
                         input_signal_array: List[Event],
                         risk_on_size: int) -> (List[str], List[str]):
        """
        Initialize the risk-on and risk-off names.

        Parameters
        ----------
        input_signal_array: List[Event]
            List of event subscription that is required for trade decision.
        risk_on_size: int
            Number of risk-on assets within the input signals.

        Returns
        -------
            List[str], List[str]
                The risk-on and risk-off name lists, respectively.
        """
        output_on = []
        output_off = []
        for i in range(len(input_signal_array)):
            output_on.append(input_signal_array[i].name()) if i < risk_on_size else \
                output_off.append(input_signal_array[i].name())
        return output_on, output_off

    def get_risk_on_off_signals(self, is_on: bool) -> Dict[str, Deque[float]]:
        """
        Retrieve risk-on or risk-off signals.

        Parameters
        ----------
        is_on: bool
            Whether we are interested in risk-on or risk-off.

        Returns
        -------
            Dict[str, Deque[float]]
                A dictionary containing signals on the selection of names.
        """
        output = {}
        names = self._risk_on_names if is_on else self._risk_off_names
        for name in names:
            output[name] = self.get_signal_by_name(name)
        return output

    def make_order_decision(self) -> Dict[str, float]:
        # Identify top performing assets from the risk-on assets
        on_signals = self.get_risk_on_off_signals(True)
        top_risk_on_performers = sorted(
            on_signals.items(), key=lambda e: e[1][-1], reverse=True)[:self._num_assets_to_hold]
        # Identify top performing assets from the risk-off assets
        off_signals = self.get_risk_on_off_signals(False)
        top_risk_off_performers = sorted(
            off_signals.items(), key=lambda e: e[1][-1], reverse=True)[0]
        # Compute the number of assets with non-positive momentum in the "risk-on" universe
        max_signals = map(max, on_signals.values())
        num_non_positive = sum(i < 0.0 for i in max_signals)

        rounded_cash_fraction = max(0.0, min(1.0, (1.0 / float(self._num_assets_to_hold))
                                             * math.floor(float(num_non_positive) * float(self._num_assets_to_hold)
                                                          / float(self._breadth_protection_threshold))))

        # Rebalance - Buy new holdings
        output = self.calculate_weight(top_risk_on_performers, top_risk_off_performers, rounded_cash_fraction)

        # Rebalance - Sell old holdings
        for pos_k, pos_v in self._contract_positions.items():
            if (abs(pos_v) > 0.0) and (pos_k != self.CASH_ACCOUNT_NAME):
                if pos_k not in output:
                    output[pos_k] = -pos_v
                else:
                    output[pos_k] -= pos_v
        return output

    def calculate_weight(self,
                         top_performers: List[str],
                         top_risk_off_performer: str,
                         rounded_cash_fraction: float) -> Dict[str, float]:
        """
        Calculate the weight to trade.

        Parameters
        ----------
        top_performers: List[str]
            Top risk-on performers' performances.
        top_risk_off_performer: str
            Top risk-off performer's performance.
        rounded_cash_fraction: float
            Rounded fraction to be held in cash.

        Returns
        -------
            Dict[str, float]
                Trade decision.
        """
        num_risk_on_to_buy = round(1.0 - rounded_cash_fraction) / (1.0 / float(self._num_assets_to_hold))
        output = {}
        num_risk_on_bought = 0
        for t in top_performers:
            if num_risk_on_bought < num_risk_on_to_buy:
                contract_name = t[0].split('_')[0]
                if self._weighting_scheme == WeightingScheme.Equal:
                    asset_weight = 1.0 / self._num_assets_to_hold
                else:
                    raise ValueError('VAAStrategy.calculate_weight: unknown weighting scheme '
                                     + str(self._weighting_scheme))
                output[contract_name] = asset_weight
                num_risk_on_bought += 1
            else:
                contract_name = top_risk_off_performer[0].split('_')[0]
                output[contract_name] = rounded_cash_fraction\
                    * self.get_mtm_history()[-1] / self.get_mtm_price_by_name(contract_name)
                break
        return output


class SlotBase:
    """
    Base class for slot, to be derived for Strategy.
    """

    def __init__(self,
                 data_name: str,
                 parent_strategies: List[StrategyBase]) -> None:
        """
        Initialize base slot.

        Parameters
        ----------
        data_name: str
            Name of the input data.
        parent_strategies: List[StrategyBase]
            Strategies that listens to this data.
        """
        self._data_name = data_name
        self._parent_strategies = parent_strategies

    @abstractmethod
    def on_event(self, new_data: datacontainer.TimeDouble) -> None:
        """
        Perform the action upon getting an event.
        """
        pass


class SignalSlot(SlotBase):
    """
    Class for signal slot.
    """

    def __init__(self,
                 data_name: str,
                 parent_strategies: List[StrategyBase]) -> None:
        super().__init__(data_name, parent_strategies)

    def on_event(self, new_signal: datacontainer.TimeDouble) -> None:
        """
        Perform the action upon getting an event.

        When there is an event (arrival of signal) we want to
        - Update signal storage
        - Make trade decision if warming up is complete
        - Send order

        Parameters
        ----------
        new_signal: TimeDouble
            Incoming new signal.
        """
        for _parent in self._parent_strategies:
            # 1. Update signal storage
            _parent.update_data(new_signal)

            # 2. If warming up is complete and all data arrived, make trade decision
            if len(_parent.get_signal_by_name(new_signal.get_name())) == _parent.get_warmup_length()\
                    and _parent.check_all_signals_received(new_signal.get_name()):
                if _parent.get_strategy_active():
                    _parent._combo_order = _parent.make_order_decision()
                    # 3. Send order
                    if _parent.get_is_live_trading():
                        _parent.send_order()


class MTMPriceSlot(SlotBase):
    """
    Class for marking to market price slot.
    """

    def __init__(self,
                 data_name: str,
                 parent_strategies: List[StrategyBase]) -> None:
        super().__init__(data_name, parent_strategies)

    def on_event(self, new_data: datacontainer.TimeDouble) -> None:
        """
        Perform the action upon getting an event.

        When there is an event (arrival of contract prices) we want to
        - Update mtm price storage
        - Mark to market

        Parameters
        ----------
        new_data: TimeDouble
            Incoming new data.
        """
        for _parent in self._parent_strategies:
            # 1. Update mtm price storage
            _parent.update_mtm(new_data)
            if _parent.check_all_contract_data_received(new_data.get_name()):
                # 2. Send order
                if not _parent.get_is_live_trading():
                    _parent.send_order()
                # 3. Mark to market
                _parent.calculate_mtm()
