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
                 broker_api_signature: str,
                 signal_info: Dict[str, utils.SignalInfo],
                 strategy_info: Dict[str, utils.StrategyInfo],
                 ) -> None:
        """
        Initialize live trader.

        Parameters
        ----------
        broker_api_signature: str
            Broker API signature.
        signal_info: Dict[str, utils.SignalInfo]
            Information for building signals.
        strategy_info: Dict[str, utils.StrategyInfo]
            Information for building strategies.
        """
        self._broker = brokerinterface.BrokerAPIBase(broker_api_signature)
        self._signal_info = signal_info
        self._strategy_info = strategy_info
        self._portfolio_manager = {}
        self._order_manager = {}
        self._strategy_dict = {}
        self.contract_dict = {}
        self._is_trading: bool = True

    def get_portfolio_manager(self) -> brokerinterface.PortfolioManagerBase:
        return self._portfolio_manager

    def get_broker(self) -> brokerinterface.BrokerAPIBase:
        return self._broker

    def disconnect(self) -> None:
        self.get_broker().disconnect()

    def get_is_trading(self) -> bool:
        return self._is_trading

    def stop_trading(self) -> None:
        self._is_trading = False

    async def start_trading(self) -> None:
        """
        Start live trading.
        """
        self.get_broker().connect(is_async=False)

        # Create contracts
        self.contract_dict = self.create_contract_dict()

        # Create broker relays
        relay_dict = self.create_relay_dict()

        # Create signals
        data_event_dict: Dict[str, Event] = {}
        for key, value in relay_dict.items():
            data_event_dict[key] = value.get_event()
        signal_dict = self.create_signal_dict(data_event_dict)

        # Subscribe to data
        for name, contract in self.contract_dict.items():
            # TBD: Allow for bid or ask
            live_bar = self.get_broker().request_live_bars(contract, utils.PriceBidAskType.Mid)
            live_bar.updateEvent += relay_dict[name].live_bar

        # Create strategies
        self._portfolio_manager = brokerinterface.PortfolioManagerBase(self.get_broker().get_class_signature(),
                                                                       self.get_broker())
        self._order_manager = brokerinterface.OrderManagerBase(self.get_broker().get_class_signature(),
                                                               self.get_broker(), self._portfolio_manager,
                                                               self.contract_dict)
        self._strategy_dict = self.create_strategy_dict(signal_dict, data_event_dict, self._order_manager)

        while self.get_is_trading():
            self.get_broker().get_handle().sleep(10)

    def create_contract_dict(self) -> Dict[str, brokerinterface.BrokerContractBase]:
        """Create contract dictionary."""
        output_dict: Dict[str, brokerinterface.BrokerContractBase] = {}
        for k_signal, v_signal in self._signal_info.items():
            for contract_type, input_name in zip(v_signal.get_contract_types(), v_signal.get_input_names()):
                if input_name not in output_dict:
                    output_dict[input_name] = brokerinterface.BrokerContractBase(
                        self.get_broker().get_class_signature(), contract_type, input_name)

        # In case there are traded contracts that are not in any signal
        strategy_info_dict: Dict[str, utils.StrategyInfo] = self._strategy_info
        for k, v in strategy_info_dict.items():
            for i in range(len(v.get_contract_names())):
                if v.get_contract_names()[i] not in output_dict:
                    output_dict[v.get_contract_names()[i]] = brokerinterface.BrokerContractBase(
                        self.get_broker().get_class_signature(), v.get_contract_types()[i], v.get_contract_names()[i])
        return output_dict

    def create_relay_dict(self) -> Dict[str, brokerinterface.BrokerEventRelayBase]:
        """Create relay dictionary."""
        output_dict: Dict[str, brokerinterface.BrokerEventRelayBase] = {}
        for k_signal, v_signal in self._signal_info.items():
            for price_type, input_name in zip(v_signal.get_price_ohlc_types(), v_signal.get_input_names()):
                if input_name not in output_dict:
                    output_dict[input_name] = brokerinterface.BrokerEventRelayBase(
                        self.get_broker().get_class_signature(), input_name, price_type)

        # In case there are traded contracts that are not in any signal
        strategy_info_dict: Dict[str, utils.StrategyInfo] = self._strategy_info
        for k, v in strategy_info_dict.items():
            for i in range(len(v.get_contract_names())):
                if v.get_contract_names()[i] not in output_dict:
                    # OK to default to close price for mtm
                    output_dict[v.get_contract_names()[i]] = brokerinterface.BrokerEventRelayBase(
                        self.get_broker().get_class_signature(), v.get_contract_names()[i], utils.PriceOHLCType.Close)
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
        order_manager: brokerinterface.OrderManagerBase
            Order manager object.
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

    def switch_all_strategies_on_off(self, activity: bool) -> None:
        """
        Turn on / off all strategies.

        Parameters
        ----------
        activity: bool
            True to turn on; False to turn off.
        """
        for value in self._strategy_dict.values():
            value.set_strategy_active(activity)

    def switch_strategy_on_off(self, strategy_name: str, activity: bool) -> None:
        """
        Turn on / off a specific strategy.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy.
        activity: bool
            True to turn on; False to turn off.
        """
        self._strategy_dict[strategy_name].set_strategy_active(activity)

    def liquidate_all(self) -> None:
        """
        Liquidate all holdings (only those holdings portfolio manager is aware of).

        Note that:
        - All strategies would be switched off
        - There is no guarantee that the book would be clean after running this function
            - Some positions can be outside alpyen
            - Some positions can be dangling order when this function is triggered
        """
        current_holdings = self._portfolio_manager.contract_info_df

        # Switch off all strategies
        self.switch_all_strategies_on_off(False)

        # Go through the holdings and do unwind trades
        for _, holding in current_holdings.iterrows():
            # Retrieve contract
            contract_to_trade = self.contract_dict[holding['symbol'].iloc[0]]
            trade_object = self._order_manager.order_wrapper(contract_to_trade, -holding['position'].iloc[0])

        # Clear all records in portfolio manager
        self._portfolio_manager.portfolio_info_df = self._portfolio_manager.portfolio_info_df[0:0]
        self._portfolio_manager.contract_info_df = self._portfolio_manager.contract_info_df[0:0]
