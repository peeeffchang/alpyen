from eventkit import Event
from typing import List, Dict

from . import brokerinterface
from . import utils


class TradeCombos:
    """
    Class for trade combos, which is a basket of contracts traded by a strategy.
    """

    def __init__(self,
                 contract_array: List[Event],
                 combo_definition: Dict[str, List[float]] = None) -> None:
        """
        Initialize trade combo.

        Parameters
        ----------
        contract_array: List[Event]
            List of event subscriptions on contracts to be traded.
        combo_definition: Dict[str, List[float]]
            A dictionary, with keys being combo names and values being weights to be traded.
        """
        if combo_definition is not None:
            for key, value in combo_definition.items():
                if len(contract_array) != len(value):
                    raise ValueError('TradeCombo.__init__: Different numbers of contracts and weights.')
            self._combo_definition = combo_definition
        else:
            self._combo_definition = None
        self._contract_array = contract_array

    def get_combo_def(self) -> Dict[str, List[float]]:
        if self._combo_definition is not None:
            return self._combo_definition
        else:
            return None

    def get_contract_array(self) -> List[Event]:
        return self._contract_array


class OrderManager:
    """
    Class for order manager.
    """

    def __init__(self, broker_api: brokerinterface.BrokerAPIBase) -> None:
        """
        Initialize order manager.

        Parameters
        ----------
        broker_api: brokerinterface.BrokerAPIBase
            Broker API.
        """
        self._broker_handle = broker_api.get_handle()
        self._dangling_orders: Dict[
            (str, str), List[float]] = {}  # A dictionary { (strategy_name, combo_name): weight_array }
        self._entry_prices: Dict[
            (str, str), List[float]] = {}  # A dictionary { (strategy_name, combo_name): entry_price }
        self._strategy_contracts: Dict[str, List[Event]] = {}  # A dictionary { strategy_name: contract_array }
        self._combo_unit: Dict[(str, str), float] = {}  # A dictionary { (strategy_name, combo_name): combo_unit }
        self._combo_weight: Dict[
            (str, str), List[float]] = {}  # A dictionary { (strategy_name, combo_name): combo_weight }
        self._order_register: Dict[int, (str, str)] = {}  # A dictionary { order_id: (strategy_name, combo_name) }
        self._outstanding_order_id: List[int] = []

    def place_order(self,
                    strategy_name: str,
                    contract_array: List[Event],
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
        contract_array: List[Event]
            List of event subscriptions on contracts to be traded.
        weight_array: List[float]
            Weights of the contracts to be traded.
        combo_unit: float
            Number of unit to trade, i.e. a multiplier for the weight_array.
        combo_name: str
            Name of the combo. Each combo in a strategy should have a unique name.
        """
        if len(contract_array) != len(weight_array):
            raise ValueError('OrderManager.place_order: Different numbers of contracts and weights.')

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
                # TBD: Other order types, other contract types
                # TBD: Need re-writing with brokerinterface
                ib_order = self._broker_handle.MarketOrder(buy_sell, order_notional)
                ib_contract = self._broker_handle.Forex(contract.name())
                self._broker_handle.placeOrder(ib_contract, ib_order)
                self._outstanding_order_id.append(this_order_id)
