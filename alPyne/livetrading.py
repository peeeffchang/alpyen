from . import brokerinterface


class EventContext:
    """
    Class for event context.
    TBD: Make it a singleton
    """

    def __init__(self,
                 broker_api: brokerinterface.BrokerAPIBase,
                 data_api: brokerinterface.BrokerAPIBase) -> None:
        """
        Initialize event context.

        Parameters
        ----------
        broker_api: brokerinterface.BrokerAPIBase
            Broker API.
        data_api: brokerinterface.BrokerAPIBase
            Data source API.
        """
        self._broker_handle = broker_api.get_handle()
        self._data_handle = data_api.get_handle()

    def get_broker(self):
        return self._broker_handle

    def get_data(self):
        return self._data_handle
