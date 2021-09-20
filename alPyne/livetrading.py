import ib_insync as ibi
from eventkit import Event

from . import strategyutils
from . import datacontainer


class EventContext:
    """
    Class for event context.
    TBD: Make it a singleton
    """

    def __init__(self, broker_handle: ibi.IB, data_handle: ibi.IB) -> None:
        """
        Initialize event context.

        Parameters
        ----------
        broker_handle: ibi.IB
            Broker API handle.
        data_handle: ibi.IB
            Data source handle.
        """
        self._broker_handle = broker_handle
        self._data_handle = data_handle

    def get_broker(self) -> ibi.IB:
        return self._broker_handle

    def get_data(self) -> ibi.IB:
        return self._data_handle


class BrokerEventRelay:
    """
    Class for broker event relay.
    TBD: Allow for different brokers (create derived classes)
    TBD: Add different relay member functions (open, high, low, close, volume)
    """

    def __init__(self,
                 listener: strategyutils.DataSlot,
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

    def ib_live_bar(self,
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
                raise TypeError('BrokerEventRelay.ib_live_bar: Unsupported data field type.')
            relay_data = datacontainer.TimeDouble(self._data_name, bars[-1].time, field)
            self._relay_event.emit(relay_data)
