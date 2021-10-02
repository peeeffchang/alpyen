from abc import ABC, abstractmethod
from collections import deque
from datetime import timedelta
from eventkit import Event
from typing import List, Deque, Dict

from . import datacontainer


class SignalBase(ABC):
    """
    Base class for signal.
    """

    def __init__(self,
                 input_data_array: List[Event],
                 warmup_length: int,
                 signal_name: str) -> None:
        """
        Initialize signal base.

        Parameters
        ----------
        input_data_array: List[Event]
            List of event subscription that is required for signal calculation.
        warmup_length: int
            Number of data points to 'burn'
        signal_name: str
            Name of the signal.
        """
        self._input_data_array = input_data_array
        self._signal_name = signal_name
        self._warmup_length = warmup_length
        self._initialize_data_time_storage(input_data_array)
        self._signal_event = Event(signal_name)

    def _initialize_data_time_storage(self, input_data_array: List[Event]) -> None:
        """
        Initialize storage.

        Parameters
        ----------
        input_data_array: List[Event]
            List of event subscription that is required for signal calculation.
        """
        data_storage = {}
        time_storage = {}
        data_slot_storage = []
        for input_data in input_data_array:
            data_storage[input_data.name()] = deque([])
            time_storage[input_data.name()] = deque([])

            this_slot = DataSlot(input_data.name(), [self])
            data_slot_storage.append(this_slot)
            # Connect event to slot
            input_data += this_slot.on_event
        self._data_storage = data_storage
        self._time_storage = time_storage
        self._data_slot_storage = data_slot_storage

    def check_all_received(self, data_name: str) -> bool:
        output = True
        num_data = len(self._input_data_array)
        if num_data == 1:
            return output  # If there is only one incoming data stream, no need to check
        for i in range(num_data):
            event_i = self._input_data_array[i]
            if len(self.get_time_by_name(event_i.name())) == 0:
                return False

            time_diff = self.get_time_by_name(data_name)[-1] - self.get_time_by_name(event_i.name())[-1]
            if time_diff > timedelta(microseconds=1):
                return False
        return output

    def update_data(self, new_data: datacontainer.TimeDouble) -> None:
        """
        Update data storage.

        Parameters
        ----------
        new_data: TimeDouble
            Incoming new data.
        """
        # Extend storage
        self._data_storage[new_data.get_name()].append(new_data.get_value())
        self._time_storage[new_data.get_name()].append(new_data.get_time())
        # Remove oldest if warming up is complete
        if len(self._data_storage[new_data.get_name()]) > self._warmup_length:
            self._data_storage[new_data.get_name()].popleft()
            self._time_storage[new_data.get_name()].popleft()

    def get_data_by_name(self, data_name: str) -> Deque:
        return self._data_storage[data_name]

    def get_time_by_name(self, data_name: str) -> Deque:
        return self._time_storage[data_name]

    def get_warmup_length(self) -> int:
        return self._warmup_length

    def get_signal_name(self) -> str:
        return self._signal_name

    def get_signal_event(self) -> Event:
        return self._signal_event

    @abstractmethod
    def calculate_signal(self) -> float:
        """
        Virtual method to be implemented by the derived class.
        """
        pass


class MASignal(SignalBase):
    """
    Moving average signal.
    """

    def __init__(self,
                 input_data_array: List[Event],
                 warmup_length: int) -> None:
        super().__init__(input_data_array, warmup_length, input_data_array[0].name() + "_MA_" + str(warmup_length))

    def calculate_signal(self) -> float:
        """
        Compute the moving average.
        """
        prices = self.get_data_by_name(self._input_data_array[0].name())
        return sum(prices) / len(prices)


class WMomSignal(SignalBase):
    """
    Weighted momentum signal.
    """
    def __init__(self,
                 input_data_array: List[Event],
                 warmup_length: int) -> None:
        super().__init__(input_data_array, warmup_length, input_data_array[0].name() + "_WM")
        # Constants
        self._trading_days_in_month = 21
        self._normalization_factor = 4.0
        self._pivot_months = [1, 3, 6, 12]

    def calculate_signal(self) -> float:
        """
        Compute the weighted momentum.
        """
        prices = self.get_data_by_name(self._input_data_array[0].name())
        weighted_momentum = 0.0
        for _month in self._pivot_months:
            p0 = prices[-1]
            pt = prices[len(prices) - (_month * self._trading_days_in_month) - 1]
            weighted_momentum += (12.0 / _month) * (p0 / pt - 1.0) / self._normalization_factor
        return weighted_momentum


class DataSlot:
    """
    Class for data slot.
    """

    def __init__(self,
                 data_name: str,
                 parent_signals: List[SignalBase]) -> None:
        """
        Initialize data slot.

        Parameters
        ----------
        data_name: str
            Name of the input data.
        parent_signals: List[SignalBase]
            Signals that listens to this data.
        """
        self._data_name = data_name
        self._parent_signals = parent_signals
        signal_events = {}
        for _parent in self._parent_signals:
            signal_events[_parent.get_signal_name()] = Event(name=_parent.get_signal_name())
        self._signal_events = signal_events

    def get_signal_events(self) -> Dict[str, Event]:
        return self._signal_events

    def on_event(self, new_data: datacontainer.TimeDouble) -> None:
        """
        Perform the action upon getting an event.

        When there is an event (arrival of data) we want to
        - Update data storage
        - Calculate signal if warming up is complete

        Parameters
        ----------
        new_data: TimeDouble
            Incoming new data.
        """
        for _parent in self._parent_signals:
            # 1. Update data storage
            _parent.update_data(new_data)

            # 2. If warming up is complete and all data arrived, calculate and publish signal
            if len(_parent.get_data_by_name(new_data.get_name())) == \
                    _parent.get_warmup_length() and _parent.check_all_received(new_data.get_name()):
                calculated_signal = _parent.calculate_signal()
                latest_timestamp = _parent.get_time_by_name(new_data.get_name())[-1]
                self._signal_events[_parent.get_signal_name()].emit(
                    datacontainer.TimeDouble(_parent.get_signal_name(), latest_timestamp, calculated_signal))
