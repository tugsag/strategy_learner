import RTLearner as rl
import BagLearner as bl
import StrategyLearner as sl
import ManualStrategy as ms
from marketsimcode import compute_portvals
from util import get_data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


def author():
    return 'gpark83'


def experiment2():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    price = get_data([symbol], pd.date_range(sd, ed), False)
    price = price.dropna()

    # Manual Strategy
    manual_strategy = ms.ManualStrategy()
    manual_orders = manual_strategy.testPolicy([symbol], sd, ed, sv)
    manual_orders = manual_orders.reset_index()
    manual_portvals = compute_portvals(manual_orders, price, sd, ed, sv, 9.95, 0.015)
    norm_manual = manual_portvals / manual_portvals[0]
    benchmark = manual_strategy.get_benchmark(symbol, price, sd, ed, sv)
    norm_benchmark = benchmark / benchmark[0]

    # Strategy Learner
    strategy_learner = sl.StrategyLearner()
    strategy_learner.addEvidence(symbol, sd, ed, sv)
    strategy_orders = strategy_learner.testPolicy(symbol, sd, ed, sv)
    strategy_orders = strategy_orders.reset_index()
    strategy_portvals = compute_portvals(strategy_orders, price, sd, ed, sv, 9.95, 0.015)
    norm_strategy = strategy_portvals / strategy_portvals[0]

    # Plotting
    fig1, ax = plt.subplots()
    norm_manual.plot(label='Manual Strategy', color='#d63729')
    norm_benchmark.plot(label='Benchmark', color='#1594CC')
    norm_strategy.plot(label='Strategy Learner', color='#26D279')
    plt.title('JPM Stock In Sample Impact Change')
    plt.ylabel('Normalized Portfolio Value')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    fig1.savefig("experiment2.png")
    plt.close(fig1)

if __name__ == "__main__":
	experiment2()