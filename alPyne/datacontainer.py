from datetime import datetime
import pandas as pd
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

    def get_name(self) -> str:
        return self._name


class DataUtils:
    @staticmethod
    def aggregate_yahoo_data(ticker_name_list: List[str],
                             path: str) -> List[PriceTimeSeries]:
        """
        Aggregate multiple ticker data (as Yahoo csv file).

        Parameters
        ----------
        ticker_name_list: List[str]
            A list of ticker names that are expected in the folder.
        path: str
            Path to files.

        Returns
        -------
            List[PriceTimeSeries]
                Aggregated data.
        """
        # TBD: Ideally this function should also perform timestamp matching, filling forward, NaN handling etc.
        output: List[PriceTimeSeries] = []

        # Read one by one
        for ticker in ticker_name_list:
            yahoo_data: PriceTimeSeries = DataUtils.read_yahoo_data(ticker, path)
            output.append(yahoo_data)
        return output

    @staticmethod
    def read_yahoo_data(ticker: str,
                        path: str) -> PriceTimeSeries:
        """
        Read Yahoo Finance csv file.

        Parameters
        ----------
        ticker: str
            Ticker name that are expected in the folder.
        path: str
            Path to files.

        Returns
        -------
            PriceTimeSeries
                Data read.
        """
        file_extension = '.csv'
        df = pd.read_csv(path + ticker + file_extension)
        yahoo_date_format = '%m/%d/%Y'
        datetime_list = [datetime.strptime(x, yahoo_date_format) for x in df['Date'].tolist()]
        return PriceTimeSeries(ticker, datetime_list, df['Adj Close'].tolist())
