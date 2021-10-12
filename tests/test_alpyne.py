#!/usr/bin/env python

"""Tests for `alpyen` package."""

import pytest
import statistics

from click.testing import CliRunner

from alpyen import datacontainer
from alpyen import backtesting
from alpyen import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'alpyen.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_backtesting_macrossing():
    # Read data
    data_folder_path = 'Data\\'
    ticker_name = 'BBH'
    short_lookback = 5
    long_lookback = 200
    ticker_names = [ticker_name]
    all_input = datacontainer.DataUtils.aggregate_yahoo_data(ticker_names, data_folder_path)

    # Subscribe to signals
    signal_info_dict = {}
    signal_info_dict[ticker_name + '_MA_' + str(short_lookback)]\
        = backtesting.SignalInfo(ticker_names, [short_lookback], 'MA')
    signal_info_dict[ticker_name + '_MA_' + str(long_lookback)]\
        = backtesting.SignalInfo(ticker_names, [long_lookback], 'MA')

    # Subscribe to strategies
    strategy_info_dict = {}
    strategy_name = ticker_name + '_MACrossing_01'
    strategy_info_dict[strategy_name] = backtesting.StrategyInfo(
        [ticker_name + '_MA_' + str(short_lookback), ticker_name + '_MA_' + str(long_lookback)],
        [1], [ticker_name], 'MACrossing')

    # Create backtester
    number_path = 1000
    my_backtester = backtesting.Backtester(all_input, ticker_names, signal_info_dict, strategy_info_dict,
                                           number_path)
    my_backtester.run_backtest()
    backtest_results = my_backtester.get_results()

    # Check
    # Actual historical path
    assert backtest_results[strategy_name][str(backtesting.MetricType.PoorMansSharpeRatio)][0]\
           == pytest.approx(0.09503, 0.0001)
    assert backtest_results[strategy_name][str(backtesting.MetricType.MaximumDrawDown)][0]\
           == pytest.approx(0.11913, 0.0001)
    assert backtest_results[strategy_name][str(backtesting.MetricType.Return)][0]\
           == pytest.approx(0.74978, 0.0001)
    # All (including simulated) paths
    assert statistics.mean(backtest_results[strategy_name][str(backtesting.MetricType.PoorMansSharpeRatio)])\
           == pytest.approx(0.105, 0.05)
    assert statistics.stdev(backtest_results[strategy_name][str(backtesting.MetricType.PoorMansSharpeRatio)])\
           == pytest.approx(0.0308, 0.05)
    assert statistics.mean(backtest_results[strategy_name][str(backtesting.MetricType.MaximumDrawDown)])\
           == pytest.approx(0.152, 0.05)
    assert statistics.stdev(backtest_results[strategy_name][str(backtesting.MetricType.MaximumDrawDown)])\
           == pytest.approx(0.0611, 0.05)
    assert statistics.mean(backtest_results[strategy_name][str(backtesting.MetricType.Return)])\
           == pytest.approx(0.865, 0.05)
    assert statistics.stdev(backtest_results[strategy_name][str(backtesting.MetricType.Return)])\
           == pytest.approx(0.326, 0.05)
