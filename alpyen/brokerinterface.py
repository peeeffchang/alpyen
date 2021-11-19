# This should be the only file that accesses broker api
from abc import abstractmethod
from datetime import date
import enum
from eventkit import Event
import ib_insync as ibi  # For Interactive Brokers (IB)
import pandas as pd
from typing import Optional, Dict, List

from . import datacontainer
from . import utils


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
                 data_name: str,
                 field_name: str = 'close',
                 ) -> None:
        """
        Initialize broker event relay.

        Parameters
        ----------
        data_name: str
            Name of the input data.
        field_name: str
            Data field name (open, high, low, close, volume, etc.).
        """
        self._relay_event = Event(data_name)
        self._field_name = field_name
        self._data_name = data_name

    def get_event(self) -> Event:
        return self._relay_event


class ContractType(enum.Enum):
    """
    Enum class for contract type.
    """
    Stock = 1
    Option = 2
    Future = 3
    FX = 4
    Index = 5


class PriceDataType(enum.Enum):
    """
    Enum class for price data type.
    """
    Bid = 1
    Ask = 2
    Mid = 3


class BrokerContractBase:
    """Base class for contract."""

    def __init__(self,
                 type_: ContractType,
                 symbol: str,
                 strike: Optional[float],
                 expiry: Optional[date]) -> None:
        """
        Initialize broker contract.

        Parameters
        ----------
        type_: ContractType
            Contract type.
        symbol: str
            Ticker symbol.
        strike: Optional[float]
            Strike (optional).
        expiry: Optional[date]
            Expiry (optional).
        """
        self._type = self._type_translation(type_)
        self._symbol = symbol
        self._strike = strike
        self._expiry = expiry
        self._contract = self._create_contract()

    @abstractmethod
    def _create_contract(self):
        pass

    @abstractmethod
    def _type_translation(self, type_: ContractType) -> str:
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

    def disconnect(self):
        self.get_handle().disconnect()

    class IBBrokerEventRelay(BrokerEventRelayBase):
        """IB event relay"""

        def __init__(self,
                     data_name: str,
                     field_name: str
                     ) -> None:
            """
            Initialize IB event relay.

            Parameters
            ----------
            data_name: str
                Name of the input data.
            field_name: str
                Data field name (open, high, low, close, volume, etc.).
            """
            super().__init__(data_name, field_name)

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
                     type_: ContractType,
                     symbol: str,
                     strike: Optional[float] = None,
                     expiry: Optional[date] = None) -> None:
            super().__init__(type_, symbol, strike, expiry)

        def _create_contract(self):
            if self._type == self._type_translation(ContractType.FX):
                return ibi.contract.Forex(self._symbol)
            else:
                return ibi.contract.Contract(symbol=self._symbol,
                                             secType=self._type,
                                             lastTradeDateOrContractMonth='' if self._expiry is None
                                             else self._expiry.strftime('%Y%m%d'),
                                             strike=0.0 if self._strike is None else self._strike)

        def _type_translation(self, type_: ContractType) -> str:
            if type_ == ContractType.Stock:
                return 'STK'
            elif type_ == ContractType.Option:
                return 'OPT'
            elif type_ == ContractType.FX:
                return 'CASH'
            elif type_ == ContractType.Future:
                return 'FUT'
            elif type_ == ContractType.Index:
                return 'IND'
            else:
                raise ValueError('IBBrokerContract._type_translation: Type not implemented.')

    def request_live_bars(self,
                          contract: IBBrokerContract,
                          price_type: PriceDataType):
        """
        Request live price data bars.

        Parameters
        ----------
        contract: IBBrokerContract
            IB contract.
        price_type: PriceDataType
            Price type.
        """
        ib_price_type_dict: Dict[PriceDataType, str] = {PriceDataType.Bid: 'BID',
                                                        PriceDataType.Ask: 'ASK',
                                                        PriceDataType.Mid: 'MIDPOINT'}
        return self.get_handle().reqRealTimeBars(contract.get_contract(), 5, ib_price_type_dict[price_type], False)


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

    def __init__(self,
                 broker_api: BrokerAPIBase) -> None:
        """
        Initialize portfolio manager.

        Parameters
        ----------
        broker_api: BrokerAPIBase
            Broker API.
        """
        self._broker_handle = broker_api.get_handle()

        # Create dataframes for storing information
        self.portfolio_info_df = pd.DataFrame(columns=['strategy_name', 'combo_name',
                                                       'combo_position', 'combo_entry_price',
                                                       'combo_mtm_price', 'unrealized_pnl'])
        self.contract_info_df = pd.DataFrame(columns=['symbol', 'type', 'exchange', 'currency', 'position'])

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
        self.portfolio_info_df['combo_mtm_price'] = temp_df['combo_mtm_price_y'].fillna(temp_df['combo_mtm_price_x'])
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
            return (self.portfolio_info_df[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                           (self.portfolio_info_df['combo_name'] == combo_name)]
                                          ['combo_position'])
        else:
            return 0.0

    def get_portfolio_info(self) -> pd.DataFrame:
        return self.portfolio_info_df

    @abstractmethod
    def portfolio_update(self, **kwargs) -> None:
        pass

    @abstractmethod
    def register_contract_trade(self, **kwargs) -> None:
        """
        Register completed contract trade from order manager
        """
        pass

    @abstractmethod
    def register_combo_trade(self, **kwargs) -> None:
        """
        Register completed combo trade from order manager
        """
        pass


class IBPortfolioManager(PortfolioManagerBase):
    """IB portfolio manager."""

    def __init__(self,
                 broker_api: IBBrokerAPI) -> None:
        """
        Initialize portfolio manager.

        Parameters
        ----------
        broker_api: IBBrokerAPI
            IB Broker API.
        """
        super().__init__(broker_api)

    def portfolio_update(self, ) -> None:
        pass

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
            self.contract_info_df[(self.contract_info_df['symbol'] == symbol) &
                                  (self.contract_info_df['type'] == type_) &
                                  (self.contract_info_df['exchange'] == exchange) &
                                  (self.contract_info_df['currency'] == currency)]['position'] += position
        else:
            self.contract_info_df = self.contract_info_df.append({'symbol': symbol,
                                                                  'type': type_,
                                                                  'exchange': exchange,
                                                                  'currency': currency,
                                                                  'position': position},
                                                                 ignore_index=True)

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
                                                       'realized_pnl'])
            existing_holding_direction = existing_position / abs(existing_position)
            (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                        (self.portfolio_info_df['combo_name'] == combo_name),
                                        'realized_pnl']) = existing_pnl + existing_holding_direction * position_closed\
                * (combo_entry_price - existing_entry_price)

            # Update combo holding
            (self.portfolio_info_df.loc[(self.portfolio_info_df['strategy_name'] == strategy_name) &
                                        (self.portfolio_info_df['combo_name'] == combo_name),
                                        'combo_position']) = existing_position + combo_unit
        else:
            self.portfolio_info_df = self.portfolio_info_df.append({'strategy_name': strategy_name,
                                                                    'combo_name': combo_name,
                                                                    'combo_position': combo_unit,
                                                                    'combo_entry_price': combo_entry_price,
                                                                    'combo_mtm_price': combo_entry_price,
                                                                    'realized_pnl': 0.0},
                                                                   ignore_index=True)


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

    def __init__(self,
                 broker_api: BrokerAPIBase,
                 portfolio_manager: PortfolioManagerBase,
                 event_contract_dict: Dict[str, BrokerContractBase]) -> None:
        """
        Initialize order manager.

        Parameters
        ----------
        broker_api: BrokerAPIBase
            Broker API.
        portfolio_manager: PortfolioManagerBase
            Portfolio manager.
        event_contract_dict: Dict[str, BrokerContractBase]
            A dictionary allowing for mapping to contracts.
        """
        self._broker_handle = broker_api.get_handle()
        self._portfolio_manager = portfolio_manager

        # Create dataframes for storing information
        self.order_info_df = pd.DataFrame(columns=['strategy_name', 'combo_name', 'contract_index',
                                                   'combo_unit', 'dangling_order', 'entry_price',
                                                   'order_id', 'time_stamp'])

        self._strategy_contracts: Dict[
            str, List[BrokerContractBase]] = {}  # A dictionary { strategy_name: contract_array }
        self._combo_weight: Dict[
            (str, str), List[float]] = {}  # A dictionary { (strategy_name, combo_name): combo_weight }

        self._event_contract_dict = event_contract_dict

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
                    unit: float
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
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def update_order_status(self,
                            **kwargs) -> None:
        pass


class IBOrderManager(OrderManagerBase):
    """IB order manager."""

    def __init__(self,
                 broker_api: IBBrokerAPI,
                 portfolio_manager: IBPortfolioManager,
                 event_contract_dict: Dict[str, IBBrokerAPI.IBBrokerContract]) -> None:
        """
        Initialize order manager.

        Parameters
        ----------
        broker_api: IBBrokerAPI
            IB Broker API.
        portfolio_manager: IBPortfolioManager
            IB portfolio manager.
        event_contract_dict: Dict[str, IBBrokerAPI.IBBrokerContract]
            A dictionary allowing for mapping to contracts.
        """
        super().__init__(broker_api, portfolio_manager, event_contract_dict)

    def register_combo_level_info(self,
                                  strategy_name: str,
                                  contract_array: List[IBBrokerAPI.IBBrokerContract],
                                  weight_array: List[float],
                                  combo_name: str) -> None:
        """
        Register combo level information that does not show up in place_order.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy placing the order.
        contract_array: List[IBBrokerAPI.IBBrokerContract]
            List of contracts to be traded.
        weight_array: List[float]
            Weights of the contracts to be traded.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        """
        self._strategy_contracts[strategy_name] = contract_array
        self._combo_weight[(strategy_name, combo_name)] = weight_array

    def place_order(self,
                    strategy_name: str,
                    combo_name: str,
                    time_stamp: str,
                    contract_index: int,
                    contract: IBBrokerAPI.IBBrokerContract,
                    unit: float
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
        """
        order_notional = unit * self._combo_weight[(strategy_name, combo_name)][contract_index]
        if abs(order_notional) > utils.EPSILON:
            # Update order status
            buy_sell: str = 'BUY' if order_notional > utils.EPSILON else 'SELL'
            # TBD: Other order types
            # TBD: Need re-writing with brokerinterface
            ib_order = ibi.order.MarketOrder(buy_sell, order_notional)
            trade_object = self._broker_handle.placeOrder(contract.get_contract(), ib_order)
            trade_object.statusEvent += self.update_order_status

            self.order_info_df = self.order_info_df.append({'strategy_name': strategy_name,
                                                            'combo_name': combo_name,
                                                            'contract_index': contract_index,
                                                            'combo_unit': unit,
                                                            'dangling_order': order_notional,
                                                            'order_id': trade_object.orderStatus.orderId,
                                                            'time_stamp': time_stamp},
                                                           ignore_index=True)

    def update_order_status(self,
                            trade: ibi.order.Trade) -> None:
        """
        Update order status.

        Parameters
        ----------
        trade: ibi.order.Trade
            IB Trade object.
        """
        order_id = trade.orderStatus.orderId
        average_fill_price = trade.orderStatus.avgFillPrice
        filled_amount = trade.orderStatus.filled

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
            self.order_info_df.loc[self.order_info_df['order_id'] == order_id, 'dangling_order'] -= filled_amount

            # Record leg entry price
            self.order_info_df.loc[self.order_info_df['order_id'] == order_id, 'entry_price'] = average_fill_price

            # Communicate update to portfolio manager.
            self._portfolio_manager.register_contract_trade(symbol=trade.contract.symbol,
                                                            type_=trade.contract.secType,
                                                            exchange=trade.contract.exchange,
                                                            currency=trade.contract.currency,
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
