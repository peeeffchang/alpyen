from datetime import datetime
from typing import List


class TimeDouble:
    """
    Basic timestamped data class.
    """

    def __init__(self,
                 data_name: str,
                 timestamp: datetime,
                 value: float) -> None:
        """
        Initialize time double.

        Parameters
        ----------
        data_name: str
            Name of the data represented.
        timestamp: datetime
            Time of the realization of the data point.
        value: float
            Numerical value of the data.
        """
        self._data_name = data_name
        self._timestamp = timestamp
        self._value = value

    def get_name(self) -> str:
        return self._data_name

    def get_time(self) -> datetime:
        return self._timestamp

    def get_value(self) -> float:
        return self._value


class PriceTimeSeries:
    """
    Class for price time series.
    """
    def __init__(self,
                 name: str,
                 timestamp: List[datetime],
                 price: List[float]) -> None:
        """
        Initialize price time series.

        Parameters
        ----------
        name: str
            Name of the data represented.
        timestamp: List[datetime]
            Time of the realization of the data point.
        price: List[float]
            Numerical value of the data.
        """
        self._name = name
        self._timestamp = timestamp
        self._price = price

    def get_price(self) -> List[float]:
        return self._price

    def get_timestamp(self) -> List[datetime]:
        return self._timestamp
