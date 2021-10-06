from abc import ABC, abstractmethod
from collections import deque
from datetime import timedelta
from eventkit import Event
import math
from typing import List, Deque, Dict

from . import brokerinterface
from . import datacontainer
from . import utils


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

    def __init__(self,
                 input_signal_array: List[Event],
                 trade_combo: TradeCombos,
                 warmup_length: int,
                 strategy_name: str,
                 initial_capital: float = 100.0,
                 order_manager: brokerinterface.OrderManager = None) -> None:
        """
        Initialize strategy base.

        Parameters
        ----------
        input_signal_array: List[Event]
            List of event subscription that is required for trade decision.
        trade_combo: TradeCombos
            TradeCombos object that defines the combos to be traded. Note that if combo definition is missing,
            all trades from this strategy will be under a generic combo.
        warmup_length: int
            Number of data points to 'burn'
        strategy_name: str
            Name of the strategy
        initial_capital: float
            Initial capital.
        order_manager: OrderManager
            OrderManager object for live trading.
        """
        # TBD: Use pandas df to store some of the member fields
        self._strategy_name = strategy_name
        self._warmup_length = warmup_length
        self._mtm = initial_capital
        if order_manager is None:
            self._is_live_trading = False
        else:
            self._is_live_trading = True
            self._order_manager = order_manager

        self._strategy_active = True  # A strategy is active by default
        self._initialize_signal_time_storage(input_signal_array)
        contract_array = trade_combo.get_contract_array()
        self._contract_array = contract_array
        if not trade_combo.get_combo_def() is None:
            self._combo_def = trade_combo.get_combo_def()
        else:
            self._combo_def = None
        self._initialize_contract_time_storage(contract_array)
        self._input_signal_array = input_signal_array

        position_temp = {security.name(): 0.0 for security in contract_array}
        # Remember to add cash account
        position_temp[self.CASH_ACCOUNT_NAME] = initial_capital
        self._contract_positions = position_temp

        self._combo_positions = {combo_name: 0.0 for combo_name in self._combo_def.keys()}

        self._combo_order = {combo_name: 0.0 for combo_name in self._combo_def.keys()}
        self._average_entry_price = {security.name(): 0.0 for security in contract_array}
        self._mtm_price = {security.name(): 0.0 for security in contract_array}

        self._mtm_history = [initial_capital]

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

    def get_mtm_history(self) -> List[float]:
        return self._mtm_history

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
        for combo_name, amount in self._combo_order.items():
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
            # Reset pending order
            self._combo_order[combo_name] = 0.0

    def send_order_live(self) -> None:
        """
        Send pending orders to broker.
        """
        pass

    #         self._order_manager.place_order(self._strategy_name,
    #                                        self._contract_array,
    #                                        )
    #                     weight_array: List[float],
    #                     combo_unit: float,
    #                     combo_name: str

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

    def __init__(self,
                 input_signal_array: List[Event],
                 trade_combo: TradeCombos,
                 warmup_length: int,
                 order_manager: brokerinterface.OrderManager = None) -> None:
        """
        Initialize MA Crossing Strategy.

        Parameters
        ----------
        input_signal_array: List[Event]
            List of event subscription that is required for trade decision.
        trade_combo: TradeCombos
            TradeCombos object that defines the combos to be traded. Note that if combo definition is missing,
            all trades from this strategy will be under a generic combo.
        warmup_length: int
            Number of data points to 'burn'
        order_manager: OrderManager
            OrderManager object for live trading.
        """
        if len(trade_combo.get_combo_def()) != 1:
            raise ValueError('MACrossingStrategy.__init__: Too many combos specified.')
        super().__init__(input_signal_array, trade_combo, warmup_length, input_signal_array[0].name()
                         + "_MACrossing", order_manager=order_manager)
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