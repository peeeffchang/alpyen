# This should be the only file that accesses broker api
from abc import abstractmethod
from datetime import date
import enum
from eventkit import Event
import ib_insync as ibi  # For Interactive Brokers (IB)
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
                     field_name: str = 'close',
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
                 event_contract_dict: Dict[str, BrokerContractBase]) -> None:
        """
        Initialize order manager.

        Parameters
        ----------
        broker_api: brokerinterface.BrokerAPIBase
            Broker API.
        event_contract_dict: Dict[str, BrokerContractBase]
            A dictionary allowing for mapping to contracts.
        """
        self._broker_handle = broker_api.get_handle()
        self._dangling_orders: Dict[
            (str, str), List[float]] = {}  # A dictionary { (strategy_name, combo_name): weight_array }
        self._entry_prices: Dict[
            (str, str), List[float]] = {}  # A dictionary { (strategy_name, combo_name): entry_price }
        self._strategy_contracts: Dict[
            str, List[BrokerContractBase]] = {}  # A dictionary { strategy_name: contract_array }
        self._combo_unit: Dict[(str, str), float] = {}  # A dictionary { (strategy_name, combo_name): combo_unit }
        self._combo_weight: Dict[
            (str, str), List[float]] = {}  # A dictionary { (strategy_name, combo_name): combo_weight }
        self._order_register: Dict[int, (str, str)] = {}  # A dictionary { order_id: (strategy_name, combo_name) }
        self._outstanding_order_id: List[int] = []
        self._event_contract_dict = event_contract_dict

    def get_event_contract_dict(self):
        return self._event_contract_dict

    @abstractmethod
    def place_order(self,
                    strategy_name: str,
                    contract_array: List[BrokerContractBase],
                    weight_array: List[float],
                    combo_unit: float,
                    combo_name: str
                    ) -> None:
        """
        Place order that is requested by strategy.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy placing the order.
        contract_array: List[BrokerContractBase]
            List of contracts to be traded.
        weight_array: List[float]
            Weights of the contracts to be traded.
        combo_unit: float
            Number of unit to trade, i.e. a multiplier for the weight_array.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        """
        pass


class IBOrderManager(OrderManagerBase):
    """IB order manager."""
    def __init__(self,
                 broker_api: IBBrokerAPI,
                 event_contract_dict: Dict[str, IBBrokerAPI.IBBrokerContract]) -> None:
        """
        Initialize order manager.

        Parameters
        ----------
        broker_api: IBBrokerAPI
            IB Broker API.
        event_contract_dict: Dict[str, IBBrokerAPI.IBBrokerContract]
            A dictionary allowing for mapping to contracts.
        """
        super().__init__(broker_api, event_contract_dict)

    def place_order(self,
                    strategy_name: str,
                    contract_array: List[IBBrokerAPI.IBBrokerContract],
                    weight_array: List[float],
                    combo_unit: float,
                    combo_name: str
                    ) -> None:
        """
        Place order that is requested by strategy.

        Parameters
        ----------
        strategy_name: str
            Name of the strategy placing the order.
        contract_array: List[IBBrokerAPI.IBBrokerContract]
            List of contracts to be traded.
        weight_array: List[float]
            Weights of the contracts to be traded.
        combo_unit: float
            Number of unit to trade, i.e. a multiplier for the weight_array.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        """
        if len(contract_array) != len(weight_array):
            raise ValueError('IBOrderManager.place_order: Different numbers of contracts and weights.')
        # Update dangling order status
        self._dangling_orders[(strategy_name, combo_name)] = weight_array
        self._entry_prices[(strategy_name, combo_name)] = [0.0] * len(weight_array)
        self._strategy_contracts[strategy_name] = contract_array
        self._combo_unit[(strategy_name, combo_name)] = combo_unit
        self._combo_weight[(strategy_name, combo_name)] = weight_array

        for contract, weight in zip(contract_array, weight_array):
            if abs(weight) > utils.EPSILON:
                this_order_id = self._broker_handle.getReqId()
                self._order_register[this_order_id] = (strategy_name, combo_name)

                order_notional = combo_unit * weight
                buy_sell: str = 'BUY' if order_notional > utils.EPSILON else 'SELL'
                # TBD: Other order types
                # TBD: Need re-writing with brokerinterface
                ib_order = self._broker_handle.MarketOrder(buy_sell, order_notional)
                self._broker_handle.placeOrder(contract, ib_order)
                self._outstanding_order_id.append(this_order_id)
