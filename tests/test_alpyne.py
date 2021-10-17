#!/usr/bin/env python

"""Tests for `alpyen` package."""

import pytest
import statistics

from click.testing import CliRunner

from alpyen import datacontainer
from alpyen import backtesting
from alpyen import cli
from alpyen import strategy


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


def test_backtesting_vaa():
    # Read data
    data_folder_path = 'Data\\'
    ticker_names = ['VOO', 'VWO', 'VEA', 'BND', 'SHY', 'IEF', 'LQD']
    all_input = datacontainer.DataUtils.aggregate_yahoo_data(ticker_names, data_folder_path)

    # Subscribe to signals
    signal_info_dict = {}
    lookback = 253
    for ticker in ticker_names:
        signal_info_dict[ticker + '_WM_1'] = backtesting.SignalInfo('WM', [ticker], [lookback])

    # Subscribe to strategies
    strategy_info_dict = {}
    strategy_name = '4ETF_VAA_01'
    strategy_info_dict[strategy_name] = backtesting.StrategyInfo(
        'VAA', [ticker + '_WM_1' for ticker in ticker_names],
        1, {'risk_on_size': 4, 'num_assets_to_hold': 2,
            'breadth_protection_threshold': 1, 'weighting_scheme': strategy.WeightingScheme.Equal},
        ticker_names, {'combo1': [1.0] * len(ticker_names)})

    # Create backtester
    number_path = 1000
    my_backtester = backtesting.Backtester(all_input, ticker_names, signal_info_dict, strategy_info_dict,
                                           number_path)
    my_backtester.run_backtest()
    backtest_results = my_backtester.get_results()

    # Check
    # Actual historical path
    assert backtest_results[strategy_name][str(backtesting.MetricType.PoorMansSharpeRatio)][0]\
           == pytest.approx(0.08549, 0.0001)
    assert backtest_results[strategy_name][str(backtesting.MetricType.MaximumDrawDown)][0]\
           == pytest.approx(0.08114, 0.0001)
    assert backtest_results[strategy_name][str(backtesting.MetricType.Return)][0]\
           == pytest.approx(1.01538, 0.0001)
    # All (including simulated) paths
    assert statistics.mean(backtest_results[strategy_name][str(backtesting.MetricType.PoorMansSharpeRatio)])\
           == pytest.approx(0.03377, 0.05)
    assert statistics.stdev(backtest_results[strategy_name][str(backtesting.MetricType.PoorMansSharpeRatio)])\
           == pytest.approx(0.01799, 0.05)
    assert statistics.mean(backtest_results[strategy_name][str(backtesting.MetricType.MaximumDrawDown)])\
           == pytest.approx(0.13297, 0.05)
    assert statistics.stdev(backtest_results[strategy_name][str(backtesting.MetricType.MaximumDrawDown)])\
           == pytest.approx(0.047857, 0.05)
    assert statistics.mean(backtest_results[strategy_name][str(backtesting.MetricType.Return)])\
           == pytest.approx(0.3629, 0.05)
    assert statistics.stdev(backtest_results[strategy_name][str(backtesting.MetricType.Return)])\
           == pytest.approx(0.2243, 0.05)
