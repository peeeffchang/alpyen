# This should be the only file that accesses broker api
from abc import abstractmethod
from datetime import date
import enum
from eventkit import Event
import ib_insync as ibi  # For Interactive Brokers (IB)
from typing import List, Dict, Optional

from . import datacontainer
from . import signal


class DataSlot:
    """
    Class for data slot.
    """

    def __init__(self,
                 data_name: str,
                 parent_signals: List[signal.SignalBase]) -> None:
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


class BrokerAPIBase:
    """Base class for broker API handle."""
    def __init__(self, broker_api_handle) -> None:
        self._handle = broker_api_handle

    def get_handle(self):
        return self._handle


class BrokerEventRelayBase:
    """
    Base class for broker event relay.
    """

    def __init__(self,
                 listener: DataSlot,
                 data_name: str,
                 field_name: str = 'close',
                 ) -> None:
        """
        Initialize broker event relay.

        Parameters
        ----------
        listener: DataSlot
            Listener data slot.
        data_name: str
            Name of the input data.
        field_name: str
            Data field name (open, high, low, close, volume, etc.).
        """
        self._relay_event = Event()
        self._field_name = field_name
        self._relay_event += listener.on_event
        self._data_name = data_name


class ContractType(enum.Enum):
    """
    Enum class for contract type.
    """
    Stock = 1
    Option = 2
    Future = 3
    FX = 4
    Index = 5


class BrokerContractBase:
    """Base class for contract."""
    def __init__(self,
                 type: ContractType,
                 symbol: str,
                 strike: Optional[float],
                 expiry: Optional[date]) -> None:
        """
        Initialize broker contract.

        Parameters
        ----------
        type: ContractType
            Contract type.
        symbol: str
            Ticker symbol.
        strike: Optional[float]
            Strike (optional).
        expiry: Optional[date]
            Expiry (optional).
        """
        self._type = self._type_translation(type)
        self._symbol = symbol
        if strike is not None:
            self._strike = strike
        if expiry is not None:
            self._expiry = expiry
        self._contract = self._create_contract()

    @abstractmethod
    def _create_contract(self):
        pass

    @abstractmethod
    def _type_translation(self, type: ContractType) -> str:
        pass

    def get_contract(self):
        return self._contract


class IBBrokerAPI(BrokerAPIBase):
    """Class for IB API handle."""
    def __init__(self) -> None:
        ibi.util.startLoop()
        super().__init__(ibi.IB())

    def connect(self,
                address: str = '127.0.0.1',
                port: int = 4002,
                client_id: int = 1) -> None:
        self.get_handle().connect(address, port, clientId=client_id)

    class IBBrokerEventRelay(BrokerEventRelayBase):
        """IB event relay"""
        def __init__(self,
                     listener: DataSlot,
                     data_name: str,
                     field_name: str = 'close',
                     ) -> None:
            """
            Initialize IB event relay.

            Parameters
            ----------
            listener: DataSlot
                Listener data slot.
            data_name: str
                Name of the input data.
            field_name: str
                Data field name (open, high, low, close, volume, etc.).
            """
            super().__init__(listener, data_name, field_name)

        # TBD: Add different relay member functions (open, high, low, close, volume)
        def live_bar(self,
                     bars: ibi.RealTimeBarList,
                     has_new_bar: bool) -> None:
            """
            Translate IB real time bar event into price update.

            Parameters
            ----------
            bars: ibi.RealTimeBarList
                IB RealTimeBarList.
            has_new_bar: bool
                Whether there is new bar.
            """
            if has_new_bar:
                if self._field_name == 'close':
                    field = bars[-1].close
                else:
                    raise TypeError('IBBrokerEventRelay.live_bar: Unsupported data field type.')
                relay_data = datacontainer.TimeDouble(self._data_name, bars[-1].time, field)
                self._relay_event.emit(relay_data)

    class IBBrokerContract(BrokerContractBase):
        """Class for IB contracts."""
        def __init__(self,
                     type: str,
                     symbol: str,
                     strike: Optional[float],
                     expiry: Optional[date]) -> None:
            super().__init__(type, symbol, strike, expiry)

        def _create_contract(self):
            return ibi.contract(symbol=self._symbol,
                                secType=self._type,
                                lastTradeDateOrContractMonth='' if self._expiry is None
                                else self._expiry.strftime('%Y%m%d'),
                                strike=0.0 if self._strike is None else self._strike)


        def _type_translation(self, type: ContractType) -> str:
            if type == ContractType.Stock:
                return 'STK'
            elif type == ContractType.Option:
                return 'OPT'
            elif type == ContractType.FX:
                return 'CASH'
            elif type == ContractType.Future:
                return 'FUT'
            elif type == ContractType.Index:
                return 'IND'
            else:
                raise ValueError('IBBrokerContract._type_translation: Type not implemented.')
