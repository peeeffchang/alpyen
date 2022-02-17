import asyncio
from contextlib import contextmanager
import datetime
from eventkit import Event
from typing import Dict, List
import yaml

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
                 path_to_control_file: str
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
        path_to_control_file: str
            File path to the control file.
        """
        self._broker = brokerinterface.BrokerAPIBase(broker_api_signature)
        self._signal_info = signal_info
        self._strategy_info = strategy_info
        self._strategy_dict = {}
        self.contract_dict = {}
        self._is_trading: bool = True
        self._path_to_control_file = path_to_control_file
        self._run_until: datetime.datetime = None

        # Create contracts
        self.contract_dict = self.create_contract_dict()

        self._portfolio_manager = brokerinterface.PortfolioManagerBase(self.get_broker().get_class_signature(),
                                                                       self.get_broker())
        self._order_manager = brokerinterface.OrderManagerBase(self.get_broker().get_class_signature(),
                                                               self.get_broker(), self._portfolio_manager,
                                                               self.contract_dict)

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

    @contextmanager
    def setup(self) -> None:
        self.get_broker().connect()
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
        strategy_dict = self.create_strategy_dict(signal_dict, data_event_dict, self._order_manager)
        self._strategy_dict = strategy_dict

        yield signal_dict, strategy_dict

    def start_trading(self) -> None:
        """
        Start live trading.
        """
        with self.setup() as (signal_dict, strategy_dict):
            event_loop = asyncio.get_event_loop()

            task = event_loop.create_task(self.async_sleep())
            event_loop.run_until_complete(task)

    async def async_sleep(self):
        while self.get_is_trading():
            # Check for new instruction
            with open(self._path_to_control_file, "r") as stream:
                try:
                    instruction_info = yaml.safe_load(stream)
                    self.carry_out_instruction(instruction_info)
                except yaml.YAMLError:
                    pass
            await asyncio.sleep(5)

    def carry_out_instruction(self, instruction_info) -> None:
        """Carry out the instruction as specified by the control file."""
        # Activity
        if instruction_info['activity'] is not None:
            if instruction_info['activity']['is_active'] is not None:
                if not instruction_info['activity']['is_active']:
                    # is_active has top priority
                    self._is_trading = False
            elif self._run_until is not None and self._run_until < datetime.datetime.now():
                # Time to shut down
                self._is_trading = False

            if ((instruction_info['activity']['run_until'] is not None)
                    and (instruction_info['activity']['seconds_to_shut_down'] is not None)):
                raise ValueError('LiveTrader.carry_out_instruction: run_until and seconds_to_shut_down '
                                 'cannot be specified simultaneously.')
            elif instruction_info['activity']['run_until'] is not None:
                self._run_until = datetime.strptime(instruction_info['activity']['run_until'], '%m/%d/%y %H:%M:%S')
            elif ((instruction_info['activity']['seconds_to_shut_down'] is not None)
                    and (self._run_until is None)):
                # We only consider seconds_to_shut_down at start up, otherwise we run into infinite loop
                datetime_now = datetime.datetime.now()
                self._run_until = datetime_now\
                    + datetime.timedelta(0, int(instruction_info['activity']['seconds_to_shut_down']))

            if instruction_info['activity']['liquidate'] is not None:
                for strategy_name in instruction_info['activity']['liquidate']:
                    if strategy_name == 'all':
                        self.switch_off_and_liquidate_all_strategies()
                    else:
                        self.switch_off_and_liquidate_strategy(strategy_name)

        # Monitoring
        if instruction_info['monitoring'] is not None:
            if instruction_info['monitoring']['by_strategy'] is not None:
                if instruction_info['monitoring']['by_strategy'] == 'print':
                    print(self.get_portfolio_manager().get_portfolio_info())
                elif instruction_info['monitoring']['by_strategy'] == 'csv':
                    by_strategy_output_file = 'alpyen_portfolio_by_strategy.csv'
                    with open(by_strategy_output_file, 'a+') as f_bs:
                        f_bs.write('\n')
                        self.get_portfolio_manager().get_portfolio_info().to_csv(f_bs,
                                                                                 index=False,
                                                                                 mode='a',
                                                                                 sep='|')
            if instruction_info['monitoring']['account'] is not None:
                if instruction_info['monitoring']['account']:
                    account_output_file = 'alpyen_broker_account.csv'
                    with open(account_output_file, 'a+') as f_a:
                        f_a.write('\n')
                        self.get_broker().get_account_info().to_csv(f_a,
                                                                    index=False,
                                                                    mode='a',
                                                                    sep='|')
            if instruction_info['monitoring']['portfolio'] is not None:
                if instruction_info['monitoring']['portfolio']:
                    portfolio_output_file = 'alpyen_portfolio.csv'
                    with open(portfolio_output_file, 'a+') as f_p:
                        f_p.write('\n')
                        self.get_broker().get_portfolio_info().to_csv(f_p,
                                                                      index=False,
                                                                      mode='a',
                                                                      sep='|')

        # Strategies
        if instruction_info['strategies'] is not None:
            for strategy_name, dict_for_strategy in instruction_info['strategies'].items():
                for strategy_attribute, value in dict_for_strategy.items():
                    if strategy_attribute == 'on_off':
                        self.switch_strategy_on_off(strategy_name, value)

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
            my_signal = signal.SignalBase(v.get_signal_signature(),
                                          price_event_list,
                                          v.get_warmup_length(),
                                          **v.get_custom_params())
            output_dict[my_signal.get_signal_name()] = my_signal
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
            my_strategy = strategy.StrategyBase(v.get_strategy_signature(), signal_event_list,
                                                trade_combos, v.get_warmup_length(), **v.get_custom_params(),
                                                order_manager=order_manager)
            output_dict[my_strategy.get_strategy_name()] = my_strategy
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

    def switch_off_and_liquidate_all_strategies(self) -> None:
        """
        Liquidate all holdings (only those holdings portfolio manager is aware of).

        Note that:
        - All strategies would be switched off
        - There is no guarantee that the book would be clean after running this function
            - Some positions can be outside alpyen
            - Some positions can be dangling order when this function is triggered
        """
        for strategy_name in self._strategy_dict.keys():
            self.switch_off_and_liquidate_strategy(strategy_name)

    def switch_off_and_liquidate_strategy(self, strategy_name: str) -> None:
        """
        Liquidate all holdings of the strategy.

        Note that:
        - The strategy would be switched off
        - There is no guarantee that the book would be clean after running this function
            - Some positions can be dangling order when this function is triggered
        """
        all_portfolio_holdings = self._portfolio_manager.portfolio_info_df
        strategy_portfolio_holdings = all_portfolio_holdings.loc[all_portfolio_holdings['strategy_name'] == strategy_name]

        # Switch off strategy
        self.switch_strategy_on_off(strategy_name, False)

        # Go through the holdings and do unwind trades
        for _, holding in strategy_portfolio_holdings.iterrows():
            combo_to_unwind = holding['combo_name']
            combo_units_to_unwind = -holding['combo_position']
            self._strategy_dict[strategy_name].set_combo_order({combo_to_unwind: combo_units_to_unwind})
            self._strategy_dict[strategy_name].send_order()
