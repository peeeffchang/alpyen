======
alpyen
======


.. image:: https://img.shields.io/pypi/v/alpyen.svg
        :target: https://pypi.python.org/pypi/alpyen

.. image:: https://readthedocs.org/projects/alpyen/badge/?version=latest
        :target: https://alpyen.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://pepy.tech/badge/alpyen
        :target: https://pepy.tech/project/alpyen
        
.. image:: https://img.shields.io/github/repo-size/peeeffchang/alpyen   
        :alt: GitHub repo size


A lite-weight backtesting and live-trading algo engine for IB (Interactive Brokers) and other brokers.


* Free software: GNU General Public License v3
* Documentation: https://alpyen.readthedocs.io.

Features
--------

Providing a trading platform for IB that includes the functions of

* Data gathering
* Algo signal calculation
* Automatic trading
* Book management

Current Version
---------------
Able to perform backtesting. Live trading support in development.

Next Release
------------
Live trading support.

Support This Project
--------------------
* Use and discuss us
* Report a bug
* Submit a bug fix

Installation
------------
::

    pip install alpyen


Example
-------
.. code-block:: python

    from alpyen import datacontainer
    from alpyen import backtesting

    # Read data (assuming that BBH.csv from Yahoo Finance is in the Data folder)
    data_folder_path = 'Data\\'
    ticker_name = 'BBH'
    short_lookback = 5
    long_lookback = 200
    short_lookback_name = ticker_name + '_MA_' + str(short_lookback)
    long_lookback_name = ticker_name + '_MA_' + str(long_lookback)
    ticker_names = [ticker_name]
    all_input = datacontainer.DataUtils.aggregate_yahoo_data(ticker_names, data_folder_path)

    # Subscribe to signals
    signal_info_dict = {}
    signal_info_dict[short_lookback_name]\
        = backtesting.SignalInfo('MA', ticker_names, [short_lookback])
    signal_info_dict[long_lookback_name]\
        = backtesting.SignalInfo('MA', ticker_names, [long_lookback])

    # Subscribe to strategies
    strategy_info_dict = {}
    strategy_name = ticker_name + '_MACrossing_01'
    strategy_info_dict[strategy_name] = backtesting.StrategyInfo(
        'MACrossing',
        [short_lookback_name, long_lookback_name],
        1, {}, ticker_names, {'combo1': [1.0]})

    # Create backtester and run backtest
    number_path = 1000
    my_backtester = backtesting.Backtester(all_input, ticker_names, signal_info_dict, strategy_info_dict,
                                           number_path)
    my_backtester.run_backtest()
    backtest_results = my_backtester.get_results()
    
The

* moving average signal / MA-crossing trading strategy; and
* weighted momentum signal / VAA strategy

are intended to serve as examples. Users can use them as references and create their custom signals/strategies by deriving from the ``SignalBase`` class within the ``signal`` module, and the ``StrategyBase`` class within the ``strategy`` module. Note that the package needs a unique signature string for each derived signals/strategies for reflective object creation, so for example:

.. code-block:: python

    class MASignal(SignalBase):
        """
        Moving average signal.
        """

        _signal_signature = 'MA'
        
    class MACrossingStrategy(StrategyBase):
        """
        MA Crossing Strategy
        """

        _strategy_signature = 'MACrossing'

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
