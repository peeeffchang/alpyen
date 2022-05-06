# This should be the only file that accesses broker api
from abc import abstractmethod
from datetime import date, datetime, timedelta
from eventkit import Event
import gemini  # For Gemini
import ib_insync as ibi  # For Interactive Brokers (IB)
import pandas as pd
from typing import Optional, Dict, List

from . import datacontainer
from . import utils


class BrokerEventRelayBase:
    """
    Base class for broker event relay.
    """
    _broker_relay_classes_registry = {}

    _broker_relay_signature = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._broker_relay_signature is None:
            raise KeyError('BrokerEventRelayBase: Missing signature for ' + str(cls))
        elif cls._broker_relay_signature in cls._broker_relay_classes_registry:
            raise KeyError('BrokerEventRelayBase: Conflict in signature ' + cls._broker_relay_signature)
        else:
            cls._broker_relay_classes_registry[cls._broker_relay_signature] = cls

    @classmethod
    def get_class_registry(cls):
        return cls._broker_relay_classes_registry

    def __new__(cls,
                signature_str: str,
                data_name: str,
                ohlc: utils.PriceOHLCType = utils.PriceOHLCType.Close,
                **kwargs
                ):
        """
        Initialize broker event relay.

        Parameters
        ----------
        signature_str: str
            Unique signature of the relay class.
        data_name: str
            Name of the input data.
        ohlc: utils.PriceOHLCType
            OHLC name (open, high, low, close, volume, etc.).
        """
        if signature_str not in cls.get_class_registry():
            raise ValueError('BrokerEventRelayBase.__new__: ' + signature_str + ' is not a valid key.')

        my_relay_obj = super().__new__(cls.get_class_registry()[signature_str])

        my_relay_obj._relay_event = Event(data_name)
        my_relay_obj._ohlc = ohlc
        my_relay_obj._data_name = data_name

        return my_relay_obj

    def __init__(self,
                 signature_str: str,
                 data_name: str,
                 ohlc: utils.PriceOHLCType = utils.PriceOHLCType.Close,
                 **kwargs) -> None:
        pass

    def get_event(self) -> Event:
        return self._relay_event


class BrokerContractBase:
    """Base class for contract."""
    _broker_contract_classes_registry = {}

    _broker_contract_signature = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._broker_contract_signature is None:
            raise KeyError('BrokerContractBase: Missing signature for ' + str(cls))
        elif cls._broker_contract_signature in cls._broker_contract_classes_registry:
            raise KeyError('BrokerContractBase: Conflict in signature ' + cls._broker_contract_signature)
        else:
            cls._broker_contract_classes_registry[cls._broker_contract_signature] = cls

    @classmethod
    def get_class_registry(cls):
        return cls._broker_contract_classes_registry

    def __new__(cls,
                signature_str: str,
                type_: utils.ContractType,
                symbol: str,
                strike: Optional[float] = None,
                expiry: Optional[date] = None):
        """
        Initialize broker contract.

        Parameters
        ----------
        signature_str: str
            Unique signature of the contract class.
        type_: ContractType
            Contract type.
        symbol: str
            Ticker symbol.
        strike: Optional[float]
            Strike (optional).
        expiry: Optional[date]
            Expiry (optional).
        """
        if signature_str not in cls.get_class_registry():
            raise ValueError('BrokerContractBase.__new__: ' + signature_str + ' is not a valid key.')

        my_contract_obj = super().__new__(cls.get_class_registry()[signature_str])

        my_contract_obj._type = my_contract_obj.type_translation(type_)
        my_contract_obj._symbol = symbol
        my_contract_obj._strike = strike
        my_contract_obj._expiry = expiry
        my_contract_obj._contract = my_contract_obj.create_contract()

        return my_contract_obj

    def __init__(self,
                 signature_str: str,
                 type_: utils.ContractType,
                 symbol: str,
                 strike: Optional[float] = None,
                 expiry: Optional[date] = None) -> None:
        pass

    def create_contract(self):
        """By default, use the symbol itself as contract."""
        return self._symbol

    @abstractmethod
    def type_translation(self, type_: utils.ContractType) -> str:
        pass

    def get_contract(self):
        return self._contract

    def get_symbol(self) -> str:
        return self._symbol


class BrokerAPIBase:
    """Base class for broker API handle."""
    _broker_api_classes_registry = {}

    _broker_api_signature = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._broker_api_signature is None:
            raise KeyError('BrokerAPIBase: Missing signature for ' + str(cls))
        elif cls._broker_api_signature in cls._broker_api_classes_registry:
            raise KeyError('BrokerAPIBase: Conflict in signature ' + cls._broker_api_signature)
        else:
            cls._broker_api_classes_registry[cls._broker_api_signature] = cls

    @classmethod
    def get_class_registry(cls):
        return cls._broker_api_classes_registry

    def get_class_signature(self):
        return self._broker_api_signature

    def __new__(cls,
                signature_str: str,
                bar_duration: int = 5,
                **kwargs):
        if signature_str not in cls.get_class_registry():
            raise ValueError('BrokerAPIBase.__new__: ' + signature_str + ' is not a valid key.')

        my_api_obj = super().__new__(cls.get_class_registry()[signature_str])
        my_api_obj._bar_duration = bar_duration
        return my_api_obj

    def __init__(self,
                 signature_str: str) -> None:
        pass

    def get_handle(self):
        return self._handle

    @abstractmethod
    def request_live_bars(self,
                          contract: BrokerContractBase,
                          price_type: utils.PriceBidAskType) -> Event:
        pass

    @abstractmethod
    def connect(self, **kwargs) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def get_account_info(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_portfolio_info(self) -> pd.DataFrame:
        pass


class IBBrokerAPI(BrokerAPIBase):
    """Class for IB API handle."""

    _broker_api_signature = 'IB'

    def __init__(self, signature_str: str, **kwargs) -> None:
        ibi.util.startLoop()
        self._handle = ibi.IB()

    def connect(self,
                address: str = '127.0.0.1',
                port: int = 4002,
                client_id: int = 1) -> None:
        self.get_handle().connect(address, port, clientId=client_id)

    async def async_connect(self,
                            address: str = '127.0.0.1',
                            port: int = 4002,
                            client_id: int = 1) -> None:
        await self.get_handle().connectAsync(address, port, clientId=client_id)

    def disconnect(self):
        self.get_handle().disconnect()

    def get_account_info(self) -> pd.DataFrame:
        output_df = pd.DataFrame(columns=['Net Value', 'Margin Requirement', 'Buying Power'])
        account_value_list = self.get_handle().accountSummary()
        temp_dict = {}
        for account_value_item in account_value_list:
            if account_value_item.tag == 'NetLiquidation':
                temp_dict['Net Value'] = [account_value_item.value]
            elif account_value_item.tag == 'MaintMarginReq':
                temp_dict['Margin Requirement'] = [account_value_item.value]
            elif account_value_item.tag == 'BuyingPower':
                temp_dict['Buying Power'] = [account_value_item.value]
        output_df = pd.concat([output_df, pd.DataFrame.from_dict(temp_dict)])
        return output_df

    def get_portfolio_info(self) -> pd.DataFrame:
        output_df = pd.DataFrame(columns=['Security', 'Amount', 'Avg Cost',
                                          'Mkt Price', 'Realized PnL', 'Unrealized PnL'])
        my_portfolio = self.get_handle().portfolio()
        for portfolio_item in my_portfolio:
            temp_dict = {
                'Security': [portfolio_item.contract.symbol],
                'Amount': [portfolio_item.position],
                'Avg Cost': [portfolio_item.averageCost],
                'Mkt Price': [portfolio_item.marketPrice],
                'Realized PnL': [portfolio_item.realizedPNL],
                'Unrealized PnL': [portfolio_item.unrealizedPNL]
            }
            output_df = pd.concat([output_df, pd.DataFrame.from_dict(temp_dict)])
        return output_df

    class IBBrokerEventRelay(BrokerEventRelayBase):
        """IB event relay"""
        _broker_relay_signature = 'IB'

        def __init__(self,
                     signature_str: str,
                     data_name: str,
                     ohlc: utils.PriceOHLCType = utils.PriceOHLCType.Close
                     ) -> None:
            """
            Initialize IB event relay.

            Parameters
            ----------
            signature_str: str
                Unique signature of the relay class.
            data_name: str
                Name of the input data.
            ohlc: utils.PriceOHLCType
                OHLC name (open, high, low, close, volume, etc.).
            """
            pass

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
                if self._ohlc == utils.PriceOHLCType.Close:
                    field = bars[-1].close
                else:
                    raise TypeError('IBBrokerEventRelay.live_bar: Unsupported data field type.')
                relay_data = datacontainer.TimeDouble(self._data_name, bars[-1].time, field)
                self._relay_event.emit(relay_data)

    class IBBrokerContract(BrokerContractBase):
        """Class for IB contracts."""
        _broker_contract_signature = 'IB'

        def __init__(self,
                     signature_str: str,
                     type_: utils.ContractType,
                     symbol: str,
                     strike: Optional[float] = None,
                     expiry: Optional[date] = None) -> None:
            pass

        def create_contract(self):
            if self._type == self.type_translation(utils.ContractType.FX):
                return ibi.contract.Forex(self._symbol)
            else:
                return ibi.contract.Contract(symbol=self._symbol,
                                             secType=self._type,
                                             lastTradeDateOrContractMonth='' if self._expiry is None
                                             else self._expiry.strftime('%Y%m%d'),
                                             strike=0.0 if self._strike is None else self._strike)

        def type_translation(self, type_: utils.ContractType) -> str:
            if type_ == utils.ContractType.Stock:
                return 'STK'
            elif type_ == utils.ContractType.Option:
                return 'OPT'
            elif type_ == utils.ContractType.FX:
                return 'CASH'
            elif type_ == utils.ContractType.Future:
                return 'FUT'
            elif type_ == utils.ContractType.Index:
                return 'IND'
            else:
                raise ValueError('IBBrokerContract.type_translation: Type not implemented.')

    def request_live_bars(self,
                          contract: IBBrokerContract,
                          price_type: utils.PriceBidAskType) -> Event:
        """
        Request live price data bars.

        Parameters
        ----------
        contract: IBBrokerContract
            IB contract.
        price_type: utils.PriceBidAskType
            Price type.

        Returns
        -------
            Event
                Bar event.
        """
        ib_price_type_dict: Dict[utils.PriceBidAskType, str] = {utils.PriceBidAskType.Bid: 'BID',
                                                                utils.PriceBidAskType.Ask: 'ASK',
                                                                utils.PriceBidAskType.Mid: 'MIDPOINT'}
        return self.get_handle().reqRealTimeBars(contract.get_contract(),
                                                 5,
                                                 ib_price_type_dict[price_type],
                                                 False).updateEvent


class GeminiBrokerAPI(BrokerAPIBase):
    """Class for Gemini API handle."""

    _broker_api_signature = 'Gemini'

    def __init__(self, signature_str: str, **kwargs) -> None:
        """
        Initialize GeminiBrokerAPI.

        Parameters
        ----------
        signature_str: str
            Unique signature of the broker API.

        Other Parameters
        ----------------
        sandbox: bool
            Whether we are in sandbox mode.
        sandbox: bool
            Whether paper trading is intended.
        public_key: str
            Gemini public key for private client.
        private_key: str
            Gemini private key for private client.
        """
        default_kwargs = {'sandbox': True}
        kwargs = {**default_kwargs, **kwargs}

        sandbox = kwargs['sandbox']
        if kwargs['public_key'] is None:
            raise ValueError('GeminiBrokerAPI.__init__: Missing public key.')
        if kwargs['private_key'] is None:
            raise ValueError('GeminiBrokerAPI.__init__: Missing private key.')

        self._public_client = gemini.PublicClient(sandbox=sandbox)
        self._private_client = gemini.PrivateClient(kwargs['public_key'],
                                                    kwargs['private_key'],
                                                    sandbox=sandbox)
        self._is_sandbox = sandbox
        self._market_data_web_sockets = {}
        self._order_event_web_socket = self.OrderEventWSEventEmitting(kwargs['public_key'],
                                                                      kwargs['private_key'],
                                                                      sandbox=sandbox)
        self._order_event_web_socket.start()

        self._handle = self.GeminiHandle(self._public_client, self._private_client, self._order_event_web_socket)

        self._tick_aggregators = {}

    def get_is_sandbox(self) -> bool:
        return self._is_sandbox

    def connect(self) -> None:
        pass

    def disconnect(self):
        for web_socket in self._market_data_web_sockets.values():
            web_socket.close()
        self._order_event_web_socket.close()

    class GeminiHandle:
        def __init__(self,
                     public_client,
                     private_client,
                     order_event_web_socket: 'OrderEventWSEventEmitting') -> None:
            self._public_client = public_client
            self._private_client = private_client
            self._order_event_web_socket = order_event_web_socket

        def get_public_client(self):
            return self._public_client

        def get_private_client(self):
            return self._private_client

        def get_order_event_web_socket(self):
            return self._order_event_web_socket

    def get_account_info(self) -> pd.DataFrame:
        # These information are not available for Gemini.
        output_df = pd.DataFrame(columns=['Net Value', 'Margin Requirement', 'Buying Power'])
        return output_df

    def get_portfolio_info(self) -> pd.DataFrame:
        output_df = pd.DataFrame(columns=['Security', 'Amount', 'Avg Cost',
                                          'Mkt Price', 'Realized PnL', 'Unrealized PnL'])
        my_portfolio = self.get_handle().get_private_client().get_balance()
        for portfolio_item in my_portfolio:
            temp_dict = {
                'Security': [portfolio_item['currency']],
                'Amount': [portfolio_item['amount']],
                'Avg Cost': [None],
                'Mkt Price': [None],
                'Realized PnL': [None],
                'Unrealized PnL': [None]
            }
            output_df = pd.concat([output_df, pd.DataFrame.from_dict(temp_dict)])
        return output_df

    class MarketDataWSEventEmitting(gemini.MarketDataWS):
        """
        An event-emitting version of gemini.MarketDataWS upon getting update.
        The on_message method is overridden.
        """

        def __init__(self, product_id, gemini_public_client, sandbox=False):
            super().__init__(product_id, sandbox=sandbox)
            self._product_id = product_id
            self.tick_event = Event('GeminiTickEvent_' + product_id)
            self._public_client = gemini_public_client

        def on_message(self, msg):
            gemini_price_type_dict: Dict[str, utils.PriceBidAskType] = {'bid': utils.PriceBidAskType.Bid,
                                                                        'ask': utils.PriceBidAskType.Ask}
            # There's no native mid so we have to proxy it using get_ticker of PublicClient.
            # get_ticker returns latest activities.
            recent_activity = self._public_client.get_ticker(self._product_id)
            self.tick_event.emit((float(recent_activity['bid']) +
                                  float(recent_activity['ask'])) / 2.0,
                                 utils.PriceBidAskType.Mid,
                                 datetime.fromtimestamp(recent_activity['volume']['timestamp'] / 1000.0))

            # Bid and ask
            if msg['socket_sequence'] >= 1:
                if len(msg['events']) == 1:
                    event = msg['events'][0]
                    if event['type'] == 'trade':
                        self.tick_event.emit(event['price'],
                                             gemini_price_type_dict[event['makerSide']],
                                             datetime.fromtimestamp(msg['timestampms'] / 1000.0))
                else:
                    # TBD: Handle other events
                    pass

        def get_event(self) -> Event:
            return self.tick_event

    class OrderEventWSEventEmitting(gemini.OrderEventsWS):
        """
        An event-emitting version of gemini.OrderEventsWS upon getting update.
        The on_message method is overridden.
        """

        def __init__(self, PUBLIC_API_KEY, PRIVATE_API_KEY, sandbox=False):
            super().__init__(PUBLIC_API_KEY, PRIVATE_API_KEY, sandbox=sandbox)
            self.order_event = Event('GeminiOrderEvent')

        def on_message(self, msg):
            if isinstance(msg, list):
                if len(msg) > 0:
                    for item in msg:
                        if item['type'] == 'fill':
                            self.order_event.emit(int(item['order_id']),
                                                  item['side'] == 'buy',
                                                  float(item['avg_execution_price']),
                                                  float(item['fill']['amount']),
                                                  item['symbol'],
                                                  'CryptoFX',
                                                  'Gemini',
                                                  item['symbol'][-3:])
                        else:
                            pass

        def get_event(self) -> Event:
            return self.order_event

    class GeminiBrokerEventRelay(BrokerEventRelayBase):
        """Gemini event relay"""
        _broker_relay_signature = 'Gemini'

        def __init__(self,
                     signature_str: str,
                     data_name: str,
                     ohlc: utils.PriceOHLCType = utils.PriceOHLCType.Close
                     ) -> None:
            """
            Initialize Gemini event relay.

            Parameters
            ----------
            signature_str: str
                Unique signature of the relay class.
            data_name: str
                Name of the input data.
            ohlc: utils.PriceOHLCType
                OHLC name (open, high, low, close, volume, etc.).
            """
            pass

        # TBD: Add different relay member functions (open, high, low, close, volume)
        def live_bar(self,
                     bars: utils.Bar) -> None:
            """
            Translate Bar event into price update.

            Parameters
            ----------
            bars: Bar
                Bar object.
            """
            if self._ohlc == utils.PriceOHLCType.Close:
                field = bars.get_close()
            else:
                raise TypeError('GeminiBrokerEventRelay.live_bar: Unsupported data field type.')
            relay_data = datacontainer.TimeDouble(self._data_name, bars.get_end_time(), field)
            self._relay_event.emit(relay_data)

    class GeminiBrokerContract(BrokerContractBase):
        """Class for Gemini contracts."""
        _broker_contract_signature = 'Gemini'

        def __init__(self,
                     signature_str: str,
                     type_: utils.ContractType,
                     symbol: str,
                     strike: Optional[float] = None,
                     expiry: Optional[date] = None) -> None:
            pass

        def type_translation(self, type_: utils.ContractType) -> str:
            pass

    def request_live_bars(self,
                          contract: GeminiBrokerContract,
                          price_type: utils.PriceBidAskType) -> Event:
        """
        Request live price data bars.

        Parameters
        ----------
        contract: GeminiBrokerContract
            Gemini contract.
        price_type: utils.PriceBidAskType
            Price type.

        Returns
        -------
            Event
                Bar event.
        """
        # Subscribe from broker
        web_socket = self.MarketDataWSEventEmitting(contract.get_contract(),
                                                    gemini.PublicClient(),
                                                    self.get_is_sandbox())
        self._market_data_web_sockets[contract.get_contract()] = web_socket
        web_socket.start()

        bar_duration = timedelta(seconds=self._bar_duration)
        end_time = utils.closest_end_time(bar_duration, datetime.now())
        # Create TickToBarAggregator object that listens to web socket
        aggregator = utils.TickToBarAggregator(contract.get_contract(),
                                               None, None, None, None,
                                               price_type, bar_duration, end_time)
        self._tick_aggregators[contract.get_contract()] = aggregator

        tick_event = web_socket.get_event()
        tick_event += aggregator.update_bar

        return aggregator.get_bar_event()


class PortfolioManagerBase:
    """
    Base class for portfolio manager.

    This class is responsible for:
    * Keeping an up-to-date record of security holdings
    * Keeping an up-to-date record of strategy-based holding
    * Reconciling security holdings with broker account record
    * Allow OrderManager to report new transaction
    * Provide holding info for strategies to decide on whether to trade
    """

    _broker_portfolio_manager_classes_registry = {}

    _broker_portfolio_manager_signature = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._broker_portfolio_manager_signature is None:
            raise KeyError('PortfolioManagerBase: Missing signature for ' + str(cls))
        elif cls._broker_portfolio_manager_signature in cls._broker_portfolio_manager_classes_registry:
            raise KeyError('PortfolioManagerBase: Conflict in signature ' + cls._broker_portfolio_manager_signature)
        else:
            cls._broker_portfolio_manager_classes_registry[cls._broker_portfolio_manager_signature] = cls

    @classmethod
    def get_class_registry(cls):
        return cls._broker_portfolio_manager_classes_registry

    def __new__(cls,
                signature_str: str,
                broker_api: BrokerAPIBase):
        """
        Initialize portfolio manager.

        Parameters
        ----------
        signature_str: str
            Unique signature of the portfolio manager class.
        broker_api: BrokerAPIBase
            Broker API.
        """
        if signature_str not in cls.get_class_registry():
            raise ValueError('PortfolioManagerBase.__new__: ' + signature_str + ' is not a valid key.')

        my_portfolio_manager_obj = super().__new__(cls.get_class_registry()[signature_str])

        my_portfolio_manager_obj._broker_handle = broker_api.get_handle()

        # Create dataframes for storing information
        my_portfolio_manager_obj.portfolio_info_df = pd.DataFrame(columns=['strategy_name', 'combo_name',
                                                                           'combo_position', 'combo_entry_price',
                                                                           'combo_mtm_price', 'unrealized_pnl',
                                                                           'realized_pnl'])
        my_portfolio_manager_obj.contract_info_df = pd.DataFrame(columns=['symbol', 'type', 'exchange',
                                                                          'currency', 'position'])

        return my_portfolio_manager_obj

    def __init__(self,
                 signature_str: str,
                 broker_api: BrokerAPIBase) -> None:
        pass

    def update_combo_mtm_price(self, combo_mtm: pd.DataFrame):
        """
        Update the combo MTM price.

        Parameters
        ----------
        combo_mtm: pd.DataFrame
            A dataframe with columns ['strategy_name', 'combo_name', 'combo_mtm_price'].
        """
        if len(self.portfolio_info_df) == 0:
            return
        temp_df = pd.merge(self.portfolio_info_df, combo_mtm,
                           how="left", on=['strategy_name', 'combo_name'])
        temp_df['combo_mtm_price_y'].fillna(temp_df['combo_mtm_price_x'], inplace=True)
        self.portfolio_info_df['combo_mtm_price'] = temp_df['combo_mtm_price_y'].values
        self.portfolio_info_df['unrealized_pnl'] = ((self.portfolio_info_df['combo_mtm_price']
                                                    - self.portfolio_info_df['combo_entry_price'])
                                                    * self.portfolio_info_df['combo_position'])

    def get_combo_position(self,
                           strategy_name: str,
                           combo_name: str) -> float:
        """
        Return the current position of a specific combo.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.

        Returns
        -------
            float
                The combo position.
        """
        is_existing = ((self.portfolio_info_df['strategy_name'] == strategy_name) &
                       (self.portfolio_info_df['combo_name'] == combo_name)).any()
        if is_existing:
            return (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                               (self.portfolio_info_df['combo_name'] == combo_name),
                                               'combo_position'].iloc[0])
        else:
            return 0.0

    def get_portfolio_info(self) -> pd.DataFrame:
        return self.portfolio_info_df

    def register_contract_trade(self,
                                symbol: str,
                                type_: str,
                                exchange: str,
                                currency: str,
                                position: float) -> None:
        """
        Register completed contract trade from order manager

        symbol: str
            Ticker symbol.
        type_: str
            Contract type (stock, future, etc.).
        exchange: str
            Exchange traded on.
        currency: str
            Denomination currency.
        position: float
            Amount traded
        """
        if ((self.contract_info_df['symbol'] == symbol) &
                (self.contract_info_df['type'] == type_) &
                (self.contract_info_df['exchange'] == exchange) &
                (self.contract_info_df['currency'] == currency)).any():
            self.contract_info_df.loc[(self.contract_info_df['symbol'] == symbol) &
                                      (self.contract_info_df['type'] == type_) &
                                      (self.contract_info_df['exchange'] == exchange) &
                                      (self.contract_info_df['currency'] == currency), 'position'] += position
        else:
            new_row = {'symbol': [symbol],
                       'type': [type_],
                       'exchange': [exchange],
                       'currency': [currency],
                       'position': [position]}
            self.contract_info_df = pd.concat([self.contract_info_df, pd.DataFrame.from_dict(new_row)])

    def register_combo_trade(self,
                             strategy_name: str,
                             combo_name: str,
                             combo_unit: float,
                             combo_def: List[float],
                             combo_entry_price: float) -> None:
        """
        Register completed combo trade from order manager.

        strategy_name: str
            Name of the strategy placing the order.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        combo_unit: float
            Number of combo units traded.
        combo_def: List[float]
            Weight that defines the combo.
        combo_entry_price: float
            Entry price.
        """
        is_existing = ((self.portfolio_info_df['strategy_name'] == strategy_name) &
                       (self.portfolio_info_df['combo_name'] == combo_name)).any()
        if is_existing:
            existing_entry_price = (self.portfolio_info_df.loc[
                (self.portfolio_info_df['strategy_name'] == strategy_name) &
                (self.portfolio_info_df['combo_name'] == combo_name),
                'combo_entry_price'].iloc[0])
            existing_position = (self.portfolio_info_df.loc[
                (self.portfolio_info_df['strategy_name'] == strategy_name) &
                (self.portfolio_info_df['combo_name'] == combo_name),
                'combo_position'].iloc[0])
            # Update entry price
            if existing_position * combo_unit < 0.0:
                # If new order and existing holding are of opposite sign
                if abs(combo_unit) > abs(existing_position):
                    (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                                (self.portfolio_info_df['combo_name'] == combo_name),
                                                'combo_entry_price']) = combo_entry_price
                else:
                    # Do not update entry price, because old position not completely consumed.
                    pass
            else:
                # If new order and existing holding of same sign: Calculate weighted entry price
                new_weighted_entry_price = ((abs(existing_position) * existing_entry_price
                                             + abs(combo_unit) * combo_entry_price)
                                            / abs(existing_position + combo_unit))
                (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                            (self.portfolio_info_df['combo_name'] == combo_name),
                                            'combo_entry_price']) = new_weighted_entry_price
            # Update realized PnL
            position_closed = 0.0
            if existing_position * combo_unit < 0.0:
                # If new order and existing holding are of opposite sign
                position_closed = min(abs(existing_position), abs(combo_unit))
            existing_pnl = (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                                       (self.portfolio_info_df['combo_name'] == combo_name),
                                                       'realized_pnl']).iloc[0]

            existing_holding_direction = (existing_position / abs(existing_position)
                                          if abs(existing_position) > utils.EPSILON else 0.0)

            new_realized = existing_pnl + existing_holding_direction * position_closed\
                * (combo_entry_price - existing_entry_price)
            (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                        (self.portfolio_info_df['combo_name'] == combo_name),
                                        'realized_pnl']) = new_realized

            old_unrealized = (self.portfolio_info_df.loc[
                (self.portfolio_info_df['strategy_name'] == strategy_name) &
                (self.portfolio_info_df['combo_name'] == combo_name),
                'unrealized_pnl'
            ])
            (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                        (self.portfolio_info_df['combo_name'] == combo_name),
                                        'unrealized_pnl']) = old_unrealized - new_realized

            # Update combo holding
            (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                        (self.portfolio_info_df['combo_name'] == combo_name),
                                        'combo_position']) = existing_position + combo_unit
        else:
            new_row = {'strategy_name': [strategy_name],
                       'combo_name': [combo_name],
                       'combo_position': [combo_unit],
                       'combo_entry_price': [combo_entry_price],
                       'combo_mtm_price': [combo_entry_price],
                       'realized_pnl': [0.0]}
            self.portfolio_info_df = pd.concat([self.portfolio_info_df, pd.DataFrame.from_dict(new_row)])


class IBPortfolioManager(PortfolioManagerBase):
    """IB portfolio manager."""
    _broker_portfolio_manager_signature = 'IB'

    def __init__(self,
                 signature_str: str,
                 broker_api: IBBrokerAPI) -> None:
        """
        Initialize portfolio manager.

        Parameters
        ----------
        signature_str: str
            Unique signature of the portfolio manager class.
        broker_api: IBBrokerAPI
            IB Broker API.
        """
        pass


class GeminiPortfolioManager(PortfolioManagerBase):
    """Gemini portfolio manager."""
    _broker_portfolio_manager_signature = 'Gemini'

    def __init__(self,
                 signature_str: str,
                 broker_api: GeminiBrokerAPI) -> None:
        """
        Initialize portfolio manager.

        Parameters
        ----------
        signature_str: str
            Unique signature of the portfolio manager class.
        broker_api: IBBrokerAPI
            Gemini Broker API.
        """
        pass


class OrderManagerBase:
    """
    Base class for order manager.

    This class is responsible for:
    * Receiving orders from strategies
    * Sending orders to broker
    * Receiving confirmation from broker
    * Registering confirmed trades with PortfolioManager
    * Keeping track of dangling (i.e. unconfirmed) orders
    """

    _broker_order_manager_classes_registry = {}

    _broker_order_manager_signature = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._broker_order_manager_signature is None:
            raise KeyError('OrderManagerBase: Missing signature for ' + str(cls))
        elif cls._broker_order_manager_signature in cls._broker_order_manager_classes_registry:
            raise KeyError('OrderManagerBase: Conflict in signature ' + cls._broker_order_manager_signature)
        else:
            cls._broker_order_manager_classes_registry[cls._broker_order_manager_signature] = cls

    @classmethod
    def get_class_registry(cls):
        return cls._broker_order_manager_classes_registry

    def __new__(cls,
                signature_str: str,
                broker_api: BrokerAPIBase,
                portfolio_manager: PortfolioManagerBase,
                event_contract_dict: Dict[str, BrokerContractBase]):
        """
        Initialize order manager.

        Parameters
        ----------
        signature_str: str
            Unique signature of the order manager class.
        broker_api: BrokerAPIBase
            Broker API.
        portfolio_manager: PortfolioManagerBase
            Portfolio manager.
        event_contract_dict: Dict[str, BrokerContractBase]
            A dictionary allowing for mapping to contracts.
        """
        if signature_str not in cls.get_class_registry():
            raise ValueError('OrderManagerBase.__new__: ' + signature_str + ' is not a valid key.')

        my_order_manager_obj = super().__new__(cls.get_class_registry()[signature_str])

        my_order_manager_obj._broker_handle = broker_api.get_handle()
        my_order_manager_obj._portfolio_manager = portfolio_manager

        # Create dataframes for storing information
        my_order_manager_obj.order_info_df = pd.DataFrame(columns=['strategy_name', 'combo_name', 'contract_index',
                                                                   'combo_unit', 'dangling_order', 'entry_price',
                                                                   'order_id', 'time_stamp'])

        my_order_manager_obj._strategy_contracts = {}  # A dictionary { strategy_name: contract_array }
        my_order_manager_obj._combo_weight = {}  # A dictionary { (strategy_name, combo_name): combo_weight }

        my_order_manager_obj._event_contract_dict = event_contract_dict

        return my_order_manager_obj

    def __init__(self,
                 signature_str: str,
                 broker_api: IBBrokerAPI,
                 portfolio_manager: IBPortfolioManager,
                 event_contract_dict: Dict[str, IBBrokerAPI.IBBrokerContract]) -> None:
        pass

    def get_event_contract_dict(self):
        return self._event_contract_dict

    def get_portfolio_manager(self):
        return self._portfolio_manager

    @abstractmethod
    def place_order(self,
                    strategy_name: str,
                    combo_name: str,
                    time_stamp: str,
                    contract_index: int,
                    contract: BrokerContractBase,
                    unit: float,
                    order_type: utils.OrderType = utils.OrderType.Market,
                    limit_price: float = None
                    ) -> None:
        """
        Place order that is requested by strategy.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy placing the order.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        time_stamp: str
            Timestamp.
        contract_index: int
            Index of the contract as found in contract array.
        contract: BrokerContractBase
            Contract to be traded.
        unit: float
            Number of unit to trade.
        order_type: utils.OrderType
            Order type.
        limit_price: float
            Limit price.
        """
        pass

    def register_combo_level_info(self,
                                  strategy_name: str,
                                  contract_array: List[BrokerContractBase],
                                  weight_array: List[float],
                                  combo_name: str) -> None:
        """
        Register combo level information that does not show up in place_order.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy placing the order.
        contract_array: List[BrokerContractBase]
            List of contracts to be traded.
        weight_array: List[float]
            Weights of the contracts to be traded.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        """
        self._strategy_contracts[strategy_name] = contract_array
        self._combo_weight[(strategy_name, combo_name)] = weight_array

    @abstractmethod
    def update_order_status(self,
                            **kwargs) -> None:
        pass

    def update_order_status_base(self,
                                 order_id: int,
                                 is_buy: bool,
                                 average_fill_price: float,
                                 filled_amount: float,
                                 contract_symbol: str,
                                 contract_type: str,
                                 exchange: str,
                                 currency: str
                                 ) -> None:
        """
        Perform the actual updating logic.

        Parameters
        ----------
        order_id: int
            Order id.
        is_buy: bool
            Whether it is a buy order.
        average_fill_price: float
            Average execution price.
        filled_amount: float
            Filled amount.
        contract_symbol: str
            Contract name.
        contract_type: str
            Type of contract.
        exchange: str
            Name of the exchange on which the trade is executed.
        currency: str
            Denomination currency.
        """
        if abs(filled_amount) > utils.EPSILON:
            # Register leg with PortfolioManager
            if order_id not in self.order_info_df['order_id'].to_list():
                # Probably receiving order status update after it is already completely filled; do nothing
                return
            combo_name = self.order_info_df.loc[self.order_info_df['order_id'] == order_id, 'combo_name'].iloc[0]
            time_stamp = self.order_info_df.loc[self.order_info_df['order_id'] == order_id, 'time_stamp'].iloc[0]
            if sum(abs(self.order_info_df.loc[(self.order_info_df['combo_name'] == combo_name) &
                                              (self.order_info_df['time_stamp'] == time_stamp), 'dangling_order'])) \
                    < utils.EPSILON:
                # All legs in the combo already filled.
                return

            # Update dangling order status.
            order_direction = 1.0 if is_buy else -1.0
            self.order_info_df.loc[self.order_info_df['order_id'] == order_id, 'dangling_order']\
                -= order_direction * filled_amount

            # Record leg entry price
            self.order_info_df.loc[self.order_info_df['order_id'] == order_id, 'entry_price'] = average_fill_price

            # Communicate update to portfolio manager.
            self._portfolio_manager.register_contract_trade(symbol=contract_symbol,
                                                            type_=str(contract_type),
                                                            exchange=exchange,
                                                            currency=currency,
                                                            position=filled_amount)

            # Check if all legs in the combo are filled, if so update portfolio manager accordingly.
            if sum(abs(self.order_info_df.loc[(self.order_info_df['combo_name'] == combo_name) &
                                              (self.order_info_df['time_stamp'] == time_stamp), 'dangling_order'])) \
                    < utils.EPSILON:
                # Calculate combo entry price.
                leg_entry_prices = (self.order_info_df.loc[(self.order_info_df['combo_name'] == combo_name) &
                                    (self.order_info_df['time_stamp'] == time_stamp),
                                                           ['contract_index', 'entry_price']])
                leg_entry_prices.sort_values(['contract_index'], ascending=True, inplace=True)
                strategy_name = self.order_info_df.loc[
                    self.order_info_df['combo_name'] == combo_name, 'strategy_name'].iloc[0]
                combo_def = [self._combo_weight[(strategy_name, combo_name)][index]
                             for index in leg_entry_prices['contract_index']]
                combo_entry_price = sum([x * y for x, y in zip(leg_entry_prices['entry_price'].tolist(), combo_def)])

                # Register with portfolio manager
                combo_unit = (self.order_info_df.loc[(self.order_info_df['combo_name'] == combo_name) &
                                                     (self.order_info_df['time_stamp'] == time_stamp),
                              'combo_unit'].iloc[0])
                self._portfolio_manager.register_combo_trade(
                    strategy_name=strategy_name,
                    combo_name=combo_name,
                    combo_unit=combo_unit,
                    combo_def=combo_def,
                    combo_entry_price=combo_entry_price)

                # Clean up dataframe by deleting rows already filled
                self.order_info_df = self.order_info_df[~(self.order_info_df['combo_name'] == combo_name) |
                                                        ~(self.order_info_df['time_stamp'] == time_stamp)]

    @abstractmethod
    def order_wrapper(self,
                      contract: BrokerContractBase,
                      order_amount: float,
                      order_type: utils.OrderType = utils.OrderType.Market,
                      limit_price: float = None):
        """
        Wrapper for the broker's order function.

        Parameters
        ----------
        contract: GeminiBrokerAPI.GeminiBrokerContract
            Contract to be traded.
        order_amount: float
            Amount to trade.
        order_type: utils.OrderType
            Order type.
        limit_price: float
            Limit price for limit order.
        """
        pass


class IBOrderManager(OrderManagerBase):
    """IB order manager."""

    _broker_order_manager_signature = 'IB'

    def __init__(self,
                 signature_str: str,
                 broker_api: IBBrokerAPI,
                 portfolio_manager: IBPortfolioManager,
                 event_contract_dict: Dict[str, IBBrokerAPI.IBBrokerContract]) -> None:
        """
        Initialize order manager.

        Parameters
        ----------
        signature_str: str
            Unique signature of the order manager class.
        broker_api: IBBrokerAPI
            IB Broker API.
        portfolio_manager: IBPortfolioManager
            IB portfolio manager.
        event_contract_dict: Dict[str, IBBrokerAPI.IBBrokerContract]
            A dictionary allowing for mapping to contracts.
        """
        pass

    def order_wrapper(self,
                      contract: IBBrokerAPI.IBBrokerContract,
                      order_amount: float,
                      order_type: utils.OrderType = utils.OrderType.Market,
                      limit_price: float = None):
        """
        Wrapper for the broker's order function.

        Parameters
        ----------
        contract: IBBrokerAPI.IBBrokerContract
            Contract to be traded.
        order_amount: float
            Amount to trade.
        order_type: utils.OrderType
            Order type.
        limit_price: float
            Limit price for limit order.
        """
        buy_sell: str = 'BUY' if order_amount > utils.EPSILON else 'SELL'
        if order_type == utils.OrderType.Market:
            ib_order = ibi.order.MarketOrder(buy_sell, abs(order_amount))
            trade_object = self._broker_handle.placeOrder(contract, ib_order)
            return trade_object
        else:
            # TBD: Support other order types
            raise ValueError('IBOrderManager.order_wrapper: Order type ' + str(order_type) + ' is not supported.')

    def place_order(self,
                    strategy_name: str,
                    combo_name: str,
                    time_stamp: str,
                    contract_index: int,
                    contract: IBBrokerAPI.IBBrokerContract,
                    unit: float,
                    order_type: utils.OrderType = utils.OrderType.Market,
                    limit_price: float = None
                    ) -> None:
        """
        Place order that is requested by strategy.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy placing the order.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        time_stamp: str
            Timestamp.
        contract_index: int
            Index of the contract as found in contract array.
        contract: IBBrokerAPI.IBBrokerContract
            Contract to be traded.
        unit: float
            Number of unit to trade.
        order_type: utils.OrderType
            Order type.
        limit_price: float
            Limit price.
        """
        order_notional = unit * self._combo_weight[(strategy_name, combo_name)][contract_index]
        if abs(order_notional) > utils.EPSILON:
            # Update order status
            trade_object = self.order_wrapper(contract.get_contract(), order_notional, order_type, limit_price)
            trade_object.statusEvent += self.update_order_status

            new_row = {'strategy_name': [strategy_name],
                       'combo_name': [combo_name],
                       'contract_index': [contract_index],
                       'combo_unit': [unit],
                       'dangling_order': [order_notional],
                       'order_id': [trade_object.orderStatus.orderId],
                       'time_stamp': [time_stamp]}
            self.order_info_df = pd.concat([self.order_info_df, pd.DataFrame.from_dict(new_row)])

    def update_order_status(self,
                            trade: ibi.order.Trade) -> None:
        """
        Update order status.

        Parameters
        ----------
        trade: ibi.order.Trade
            IB Trade object.
        """
        self.update_order_status_base(trade.orderStatus.orderId,
                                      trade.order.action == 'BUY',
                                      trade.orderStatus.avgFillPrice,
                                      trade.orderStatus.filled,
                                      trade.contract.symbol,
                                      trade.contract.secType,
                                      trade.contract.exchange,
                                      trade.contract.currency)


class GeminiOrderManager(OrderManagerBase):
    """Gemini order manager."""

    _broker_order_manager_signature = 'Gemini'

    def __init__(self,
                 signature_str: str,
                 broker_api: GeminiBrokerAPI,
                 portfolio_manager: GeminiPortfolioManager,
                 event_contract_dict: Dict[str, GeminiBrokerAPI.GeminiBrokerContract]) -> None:
        """
        Initialize order manager.

        Parameters
        ----------
        signature_str: str
            Unique signature of the order manager class.
        broker_api: GeminiBrokerAPI
            Gemini Broker API.
        portfolio_manager: GeminiPortfolioManager
            Gemini portfolio manager.
        event_contract_dict: Dict[str, GeminiBrokerAPI.GeminiBrokerContract]
            A dictionary allowing for mapping to contracts.
        """
        order_event = broker_api.get_handle().get_order_event_web_socket().get_event()
        order_event += self.update_order_status

    def order_wrapper(self,
                      contract: GeminiBrokerAPI.GeminiBrokerContract,
                      order_amount: float,
                      order_type: utils.OrderType = utils.OrderType.Market,
                      limit_price: float = None):
        """
        Wrapper for the broker's order function.

        Parameters
        ----------
        contract: GeminiBrokerAPI.GeminiBrokerContract
            Contract to be traded.
        order_amount: float
            Amount to trade.
        order_type: utils.OrderType
            Order type.
        limit_price: float
            Limit price for limit order.
        """
        buy_sell: str = 'buy' if order_amount > utils.EPSILON else 'sell'

        new_order = self._broker_handle.get_private_client().new_order(contract.get_contract(),
                                                                       str(format(abs(order_amount), '.10f')),
                                                                       str(format(abs(limit_price), '.2f')),
                                                                       buy_sell,
                                                                       options=[])
        return new_order

    def place_order(self,
                    strategy_name: str,
                    combo_name: str,
                    time_stamp: str,
                    contract_index: int,
                    contract: GeminiBrokerAPI.GeminiBrokerContract,
                    unit: float,
                    order_type: utils.OrderType = utils.OrderType.Market,
                    limit_price: float = None
                    ) -> None:
        """
        Place order that is requested by strategy.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy placing the order.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        time_stamp: str
            Timestamp.
        contract_index: int
            Index of the contract as found in contract array.
        contract: GeminiBrokerAPI.GeminiBrokerContract
            Contract to be traded.
        unit: float
            Number of unit to trade.
        order_type: utils.OrderType
            Order type.
        limit_price: float
            Limit price.
        """
        order_notional = unit * self._combo_weight[(strategy_name, combo_name)][contract_index]
        if abs(order_notional) > utils.EPSILON:
            # Update order status
            trade_object = self.order_wrapper(contract, order_notional, order_type, limit_price)

            new_row = {'strategy_name': [strategy_name],
                       'combo_name': [combo_name],
                       'contract_index': [contract_index],
                       'combo_unit': [unit],
                       'dangling_order': [order_notional],
                       'order_id': [int(trade_object['order_id'])],
                       'time_stamp': [time_stamp]}
            self.order_info_df = pd.concat([self.order_info_df, pd.DataFrame.from_dict(new_row)])

    def update_order_status(self,
                            order_id: int,
                            is_buy: bool,
                            average_fill_price: float,
                            filled_amount: float,
                            contract_symbol: str,
                            contract_type: str,
                            exchange: str,
                            currency: str
                            ) -> None:
        """
        Perform the actual updating logic.

        Parameters
        ----------
        order_id: int
            Order id.
        is_buy: bool
            Whether it is a buy order.
        average_fill_price: float
            Average execution price.
        filled_amount: float
            Filled amount.
        contract_symbol: str
            Contract name.
        contract_type: str
            Type of contract.
        exchange: str
            Name of the exchange on which the trade is executed.
        currency: str
            Denomination currency.
        """
        self.update_order_status_base(order_id,
                                      is_buy,
                                      average_fill_price,
                                      filled_amount,
                                      contract_symbol,
                                      contract_type,
                                      exchange,
                                      currency)
