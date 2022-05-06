from datetime import datetime, timedelta
import enum
from eventkit import Event
from typing import List, Dict


EPSILON = 1.e-10


class ContractType(enum.Enum):
    """
    Enum class for contract type.
    """
    Stock = 1
    Option = 2
    Future = 3
    FX = 4
    Index = 5


class PriceBidAskType(enum.Enum):
    """
    Enum class for price bid-ask type.
    """
    Bid = 1
    Ask = 2
    Mid = 3


class PriceOHLCType(enum.Enum):
    """
    Enum class for price OHLC type.
    """
    Open = 1
    High = 2
    Low = 3
    Close = 4


class OrderType(enum.Enum):
    """
    Enum class for price order type.
    """
    Market = 1
    Limit = 2
    StopLoss = 3


class SignalInfo:
    """
    class for signal info.
    """
    def __init__(self,
                 signal_signature: str,
                 input_names: List[str],
                 contract_types: List[ContractType],
                 price_ohlc_types: List[PriceOHLCType],
                 warmup_length: int,
                 custom_params: Dict) -> None:
        """
        Initialize signal info

        Parameters
        ----------
        signal_signature: str
            Unique signature of the signal.
        input_names: List[str]
            List of inputs the signal is listening to.
        contract_types: List[ContractType]
            List of contract types.
        price_ohlc_types: List[PriceOHLCType]
            List of price OHLC types.
        warmup_length: int
            Warm-up length.
        custom_params: Dict
            Other signal specific parameters.
        """
        self._input_names = input_names
        self._contract_types = contract_types
        self._price_ohlc_types = price_ohlc_types
        self._warmup_length = warmup_length
        self._custom_params = custom_params
        self._signal_signature = signal_signature

    def get_input_names(self) -> List[str]:
        return self._input_names

    def get_contract_types(self) -> List[ContractType]:
        return self._contract_types

    def get_price_ohlc_types(self) -> List[PriceOHLCType]:
        return self._price_ohlc_types

    def get_warmup_length(self) -> int:
        return self._warmup_length

    def get_custom_params(self) -> Dict:
        return self._custom_params

    def get_signal_signature(self) -> str:
        return self._signal_signature


class StrategyInfo:
    """
    class for strategy info.
    """
    def __init__(self,
                 strategy_signature: str,
                 input_names: List[str],
                 warmup_length: int,
                 custom_params: Dict,
                 contract_names: List[str],
                 contract_types: List[ContractType] = None,
                 combo_definition: Dict[str, List[float]] = None,
                 order_types: Dict[str, List[OrderType]] = None) -> None:
        """
        Initialize strategy info

        Parameters
        ----------
        strategy_signature: str
            Unique signature of the signal.
        input_names: List[str]
            List of inputs the strategy is listening to.
        warmup_length: int
            Number of data points to 'burn'
        custom_params: Dict
            Other strategy specific parameters.
        contract_names: List[str]
            Contract names for TradeCombos creation.
        contract_types: List[ContractType]
            Contract types.
        combo_definition: Dict[str, List[float]]
            A dictionary, with keys being combo names and values being weights to be traded.
        order_types: Dict[str, List[OrderType]]
            A dictionary, with keys being combo names and values being order type to use.
        """
        # Check input integrity
        if contract_types is not None:
            assert len(contract_names) == len(contract_types),\
                'Contract names and contract types have different lengths.'
        for k, v in combo_definition.items():
            assert len(v) == len(contract_names), 'Contract names and weight for ' + k + ' have different lengths.'
        if order_types is not None:
            for k, v in order_types.items():
                assert len(v) == len(contract_names), \
                    'Contract names and order type for ' + k + ' have different lengths.'

        self._input_names = input_names
        self._warmup_length = warmup_length
        self._custom_params = custom_params
        self._strategy_signature = strategy_signature
        self._contract_names = contract_names
        self._contract_types = contract_types
        self._combo_definition = combo_definition
        self._order_types = order_types

    def get_input_names(self) -> List[str]:
        return self._input_names

    def get_warmup_length(self) -> int:
        return self._warmup_length

    def get_custom_params(self) -> Dict:
        return self._custom_params

    def get_contract_names(self) -> List[str]:
        return self._contract_names

    def get_contract_types(self) -> List[ContractType]:
        return self._contract_types

    def get_combo_definition(self) -> Dict[str, List[float]]:
        return self._combo_definition

    def get_order_types(self) -> Dict[str, List[OrderType]]:
        return self._order_types

    def get_strategy_signature(self) -> str:
        return self._strategy_signature


def closest_end_time(duration: timedelta, reference_time: datetime) -> datetime:
    """
    Calculate the closest 'integer' future time to act as bar end time, dropping microsecond.

    Do not work well with weird intervals (such as 90m, 45m), because in such cases, this function does not know that,
    e.g. 12:00 is a solution but 13:00 is not.

    Examples: (assuming now = 14:23:13)
    - closest_end_time(5s)      = 14:23:15
    - closest_end_time(10s)     = 14:23:20
    - closest_end_time(15s)     = 14:23:15
    - closest_end_time(20s)     = 14:23:20
    - closest_end_time(60s)     = 14:24:00
    - closest_end_time(120s)    = 14:24:00
    - closest_end_time(300s)    = 14:25:00
    - closest_end_time(600s)    = 14:30:00
    - closest_end_time(1800s)   = 14:30:00
    - closest_end_time(3600s)   = 15:00:00
    - closest_end_time(14400s)  = 16:00:00

    Parameters
    ----------
    duration: timedelta
        Time span of a single bar.
    reference_time: datetime
        The 'current moment' that acts as the reference point.

    Returns
    -------
        datetime
            Future time.
    """
    if duration < timedelta(seconds=60):
        wait_time_in_second = duration.seconds - \
                              reference_time.second % duration.seconds
    elif timedelta(seconds=60 * 60) > duration >= timedelta(seconds=60):
        wait_time_in_second = duration.seconds - \
                              (reference_time.minute * 60 + reference_time.second) % duration.seconds
    elif timedelta(seconds=24 * 60 * 60) > duration >= timedelta(seconds=60 * 60):
        wait_time_in_second = duration.seconds - \
                              (reference_time.hour * 60 * 60 +
                               reference_time.minute * 60 +
                               reference_time.second) % duration.seconds
    else:
        wait_time_in_second = 0.0

    # Add buffer if wait time is too close
    buffer_time = 0.1

    if wait_time_in_second < buffer_time:
        reference_time_nontruncated = closest_end_time(duration, reference_time +
                                                       timedelta(seconds=wait_time_in_second + buffer_time + EPSILON))
    else:
        reference_time_nontruncated = reference_time + timedelta(seconds=wait_time_in_second)

    return reference_time_nontruncated.replace(microsecond=0)


class Bar:
    """Representation of an OHLC bar."""
    def __init__(self,
                 open_: float = None,
                 high_: float = None,
                 low_: float = None,
                 close_: float = None,
                 bid_ask: PriceBidAskType = PriceBidAskType.Mid,
                 duration: timedelta = timedelta(seconds=5),
                 end_time: datetime = datetime.now()) -> None:
        """
        Initialize bar.

        Parameters
        ----------
        open_: float
            Open value.
        high_: float
            High value.
        low_: float
            Low value.
        close_: float
            Close value.
        bid_ask: utils.PriceBidAskType
            Bid or ask.
        duration: timedelta
            Duration of each bar.
        end_time: datetime
            Time of bar conclusion.
        """
        self._duration = duration
        self._end_time = end_time
        self._open = open_
        self._high = high_
        self._low = low_
        self._close = close_
        self._bid_ask = bid_ask

    def update_open(self, new_value: float) -> None:
        self._open = new_value

    def update_high(self, new_value: float) -> None:
        self._high = new_value

    def update_low(self, new_value: float) -> None:
        self._low = new_value

    def update_close(self, new_value: float) -> None:
        self._close = new_value

    def update_end_time(self, new_value: datetime) -> None:
        self._end_time = new_value

    def get_end_time(self) -> datetime:
        return self._end_time

    def get_bid_ask(self) -> PriceBidAskType:
        return self._bid_ask

    def get_duration(self) -> timedelta:
        return self._duration

    def get_open(self) -> float:
        return self._open

    def get_high(self) -> float:
        return self._high

    def get_low(self) -> float:
        return self._low

    def get_close(self) -> float:
        return self._close


class TickToBarAggregator:
    """Aggregate tick data to bar."""
    def __init__(self,
                 data_name: str,
                 open_: float = None,
                 high_: float = None,
                 low_: float = None,
                 close_: float = None,
                 bid_ask: PriceBidAskType = PriceBidAskType.Mid,
                 duration: timedelta = timedelta(seconds=5),
                 end_time: datetime = datetime.now()) -> None:
        """
        Initialize bar.

        Parameters
        ----------
        open_: float
            Open value.
        high_: float
            High value.
        low_: float
            Low value.
        close_: float
            Close value.
        bid_ask: utils.PriceBidAskType
            Bid or ask.
        duration: timedelta
            Duration of each bar.
        end_time: datetime
            Time of bar conclusion.
        """
        self._current_bar = Bar(open_, high_, low_, close_, bid_ask, duration, end_time)
        self._ready_bar = Bar(None, None, None, None, bid_ask, duration, end_time)
        self._bar_publication_event = Event('BarPublicationEvent_' + data_name)

    def get_bar_event(self) -> Event:
        return self._bar_publication_event

    def update_bar(self,
                   new_value: float,
                   bid_ask: PriceBidAskType,
                   incoming_timestamp: datetime = None) -> None:
        """Update with new incoming value."""
        if bid_ask != self._current_bar.get_bid_ask() or \
                incoming_timestamp < (self._current_bar.get_end_time() - self._current_bar.get_duration()):
            pass
        if incoming_timestamp is None:
            raise TypeError('TickToBarAggregator.update_bar: incoming_timestamp is None.')
        # Check if we have to reset bar
        if incoming_timestamp > self._current_bar.get_end_time():
            if self._current_bar.get_open() is not None:
                self.copy_to_ready_bar()
                self.publish_ready_bar()
            while incoming_timestamp > self._current_bar.get_end_time():
                self.reset_current_bar()

        self._current_bar.update_close(new_value)
        if self._current_bar.get_open() is None:
            self._current_bar.update_open(new_value)
            self._current_bar.update_high(new_value)
            self._current_bar.update_low(new_value)
        else:
            if new_value > self._current_bar.get_high():
                self._current_bar.update_high(new_value)
            elif new_value < self._current_bar.get_low():
                self._current_bar.update_low(new_value)

    def reset_current_bar(self) -> None:
        self._current_bar.update_end_time(self._current_bar.get_end_time() + self._current_bar.get_duration())
        self._current_bar.update_open(None)
        self._current_bar.update_high(None)
        self._current_bar.update_low(None)
        self._current_bar.update_close(None)

    def copy_to_ready_bar(self) -> None:
        self._ready_bar.update_open(self._current_bar.get_open())
        self._ready_bar.update_high(self._current_bar.get_high())
        self._ready_bar.update_low(self._current_bar.get_low())
        self._ready_bar.update_close(self._current_bar.get_close())
        self._ready_bar.update_end_time(self._current_bar.get_end_time())

    def publish_ready_bar(self) -> None:
        self._bar_publication_event.emit(self._ready_bar)
