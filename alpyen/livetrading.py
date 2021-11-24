from eventkit import Event
from typing import Dict, List

from . import brokerinterface
from . import signal
from . import strategy
from . import utils


class LiveTrader:
    """
    Class for live trader.
    TBD: Make it a singleton
    """

    def __init__(self,
                 broker_api: brokerinterface.BrokerAPIBase,
                 data_api: brokerinterface.BrokerAPIBase,
                 signal_info: Dict[str, utils.SignalInfo],
                 strategy_info: Dict[str, utils.StrategyInfo],
                 ) -> None:
        """
        Initialize live trader.

        Parameters
        ----------
        broker_api: brokerinterface.BrokerAPIBase
            Broker API.
        data_api: brokerinterface.BrokerAPIBase
            Data source API.
        signal_info: Dict[str, utils.SignalInfo]
            Information for building signals.
        strategy_info: Dict[str, utils.StrategyInfo]
            Information for building strategies.
        """
        self._broker = broker_api
        self._data_source = data_api
        self._signal_info = signal_info
        self._strategy_info = strategy_info

    def get_broker(self) -> brokerinterface.BrokerAPIBase:
        return self._broker

    def get_data_source(self) -> brokerinterface.BrokerAPIBase:
        return self._data_source

    def start_trading(self) -> None:
        """
        Start live trading.
        """
        self.get_broker().get_handle().connect()

        # Create contracts
        contract_dict = self.create_contract_dict()

        # Create broker relays
        relay_dict = self.create_relay_dict()

        # Create signals
        data_event_dict: Dict[str, Event] = {}
        for key, value in relay_dict.items():
            data_event_dict[key] = value.get_event()
        signal_dict = self.create_signal_dict(data_event_dict)

        # Subscribe to data
        for name, contract in contract_dict.items():
            # TBD: Allow for bid or ask
            live_bar = self._data_source.request_live_bars(contract, brokerinterface.PriceBidAskType.Mid)
            live_bar.updateEvent += relay_dict[name].live_bar

        # Create strategies
        portfolio_manager = brokerinterface.PortfolioManagerBase(self.get_broker())
        order_manager = brokerinterface.OrderManagerBase(self.get_broker(), portfolio_manager, contract_dict)
        strategy_dict = self.create_strategy_dict(signal_dict, data_event_dict, order_manager)

    def create_contract_dict(self) -> Dict[str, brokerinterface.BrokerContractBase]:
        """Create contract dictionary."""
        output_dict: Dict[str, brokerinterface.BrokerContractBase] = {}
        for k_signal, v_signal in self._signal_info.items():
            for contract_type, input_name in zip(v_signal.get_contract_types(), v_signal.get_input_names()):
                if input_name not in output_dict:
                    output_dict[input_name] = brokerinterface.BrokerContractBase(contract_type, input_name)
        return output_dict

    def create_relay_dict(self) -> Dict[str, brokerinterface.BrokerEventRelayBase]:
        """Create relay dictionary."""
        output_dict: Dict[str, brokerinterface.BrokerEventRelayBase] = {}
        for k_signal, v_signal in self._signal_info.items():
            for price_type, input_name in zip(v_signal.get_price_ohlc_types(), v_signal.get_input_names()):
                if input_name not in output_dict:
                    output_dict[input_name] = brokerinterface.BrokerEventRelayBase(input_name, price_type)
        return output_dict

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
                             data_event_dict: Dict[str, Event],
                             order_manager: brokerinterface.OrderManagerBase) -> Dict[str, strategy.StrategyBase]:
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
            trade_combos = strategy.TradeCombos(price_event_list, v.get_combo_definition())
            # Create strategy
            signal_event_list: List[Event] = []
            for i in range(len(v.get_input_names())):
                signal_event_list.append(signal_dict.get(v.get_input_names()[i]).get_signal_event())
            output_dict[k] = strategy.StrategyBase(v.get_strategy_signature(), signal_event_list,
                                                   trade_combos, v.get_warmup_length(), **v.get_custom_params(),
                                                   order_manager=order_manager)
        return output_dict
