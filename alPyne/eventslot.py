from abc import abstractmethod
from eventkit import Event
from typing import List, Dict

from . import datacontainer
from . import signal
from . import strategy


class SlotBase:
    """
    Base class for slot, to be derived for Strategy.
    """

    def __init__(self,
                 data_name: str,
                 parent_strategies: List[strategy.StrategyBase]) -> None:
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
                 parent_strategies: List[strategy.StrategyBase]) -> None:
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
                 parent_strategies: List[strategy.StrategyBase]) -> None:
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
