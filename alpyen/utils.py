import enum
from typing import List, Dict


EPSILON = 1.e-10


class ContractType(enum.Enum):
    """
    Enum class for contract type.
    """
    Stock = 1
    Option = 2
    Future = 3
    FX = 4
    Index = 5


class PriceBidAskType(enum.Enum):
    """
    Enum class for price bid-ask type.
    """
    Bid = 1
    Ask = 2
    Mid = 3


class PriceOHLCType(enum.Enum):
    """
    Enum class for price OHLC type.
    """
    Open = 1
    High = 2
    Low = 3
    Close = 4


class SignalInfo:
    """
    class for signal info.
    """
    def __init__(self,
                 signal_signature: str,
                 input_names: List[str],
                 contract_types: List[ContractType],
                 price_ohlc_types: List[PriceOHLCType],
                 warmup_length: int,
                 custom_params: Dict) -> None:
        """
        Initialize signal info

        Parameters
        ----------
        signal_signature: str
            Unique signature of the signal.
        input_names: List[str]
            List of inputs the signal is listening to.
        contract_types: List[ContractType]
            List of contract types.
        price_ohlc_types: List[PriceOHLCType]
            List of price OHLC types.
        warmup_length: int
            Warm-up length.
        custom_params: Dict
            Other signal specific parameters.
        """
        self._input_names = input_names
        self._contract_types = contract_types
        self._price_ohlc_types = price_ohlc_types
        self._warmup_length = warmup_length
        self._custom_params = custom_params
        self._signal_signature = signal_signature

    def get_input_names(self) -> List[str]:
        return self._input_names

    def get_contract_types(self) -> List[ContractType]:
        return self._contract_types

    def get_price_ohlc_types(self) -> List[PriceOHLCType]:
        return self._price_ohlc_types

    def get_warmup_length(self) -> int:
        return self._warmup_length

    def get_custom_params(self) -> Dict:
        return self._custom_params

    def get_signal_signature(self) -> str:
        return self._signal_signature


class StrategyInfo:
    """
    class for strategy info.
    """
    def __init__(self,
                 strategy_signature: str,
                 input_names: List[str],
                 warmup_length: int,
                 custom_params: Dict,
                 contract_names: List[str],
                 contract_types: List[ContractType] = None,
                 combo_definition: Dict[str, List[float]] = None) -> None:
        """
        Initialize strategy info

        Parameters
        ----------
        strategy_signature: str
            Unique signature of the signal.
        input_names: List[str]
            List of inputs the strategy is listening to.
        warmup_length: int
            Number of data points to 'burn'
        custom_params: Dict
            Other strategy specific parameters.
        contract_names: List[str]
            Contract names for TradeCombos creation.
        contract_types: List[ContractType]
            Contract types.
        combo_definition: Dict[str, List[float]]
            Weight dictionay for TradeCombos creation.
        """
        # Check input integrity
        if contract_types is not None:
            assert len(contract_names) == len(contract_types),\
                'Contract names and contract types have different lengths.'
        for k, v in combo_definition.items():
            assert len(v) == len(contract_names), 'Contract names and definition for ' + k + ' have different lengths.'

        self._input_names = input_names
        self._warmup_length = warmup_length
        self._custom_params = custom_params
        self._strategy_signature = strategy_signature
        self._contract_names = contract_names
        self._contract_types = contract_types
        self._combo_definition = combo_definition

    def get_input_names(self) -> List[str]:
        return self._input_names

    def get_warmup_length(self) -> int:
        return self._warmup_length

    def get_custom_params(self) -> Dict:
        return self._custom_params

    def get_contract_names(self) -> List[str]:
        return self._contract_names

    def get_contract_types(self) -> List[ContractType]:
        return self._contract_types

    def get_combo_definition(self) -> Dict[str, List[float]]:
        return self._combo_definition

    def get_strategy_signature(self) -> str:
        return self._strategy_signature
