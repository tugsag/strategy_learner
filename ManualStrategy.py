"""MC2-P6: Manual Strategy

CS 4646/7646

Student Name: Grace Park
GT User ID: gpark83
GT ID: 903474899
"""

import pandas as pd
import numpy as np
from marketsimcode import compute_portvals, get_daily_ret
from util import get_data
from indicators import normalize, get_indicators
import datetime as dt
import matplotlib.pyplot as plt


def author():
    return 'gpark83'


class ManualStrategy:
    def __init__(self):
        pass

    def get_benchmark(self, symbol, price, sd, ed, sv):
        benchmark = pd.DataFrame.from_dict({'Date': [price.index[0]], symbol: [1000]})
        port_val = compute_portvals(benchmark, price, sd, ed, sv, 0, 0)
        return port_val

    # Strategy recommend by the Project Wiki
    def get_order(self, positions, symbol):
        holding_orders = pd.DataFrame(columns=['Date', symbol])
        current_holdings = 0

        for holding in positions.iterrows():
            date = holding[1]['Date']
            position = holding[1]['Position']

            if current_holdings == 0:
                if position == -1:
                    holding_orders = holding_orders.append({'Date': date, symbol: -1000}, ignore_index=True)
                    current_holdings -= 1000
                elif position == 1:
                    holding_orders = holding_orders.append({'Date': date, symbol: 1000}, ignore_index=True)
                    current_holdings += 1000
                else:
                    holding_orders = holding_orders.append({'Date': date, symbol: 0}, ignore_index=True)
            elif current_holdings == 1000:
                if position == -1:
                    holding_orders = holding_orders.append({'Date': date, symbol: -2000}, ignore_index=True)
                    current_holdings -= 2000
                else:
                    holding_orders = holding_orders.append({'Date': date, symbol: 0}, ignore_index=True)
            elif current_holdings == -1000:
                if position == 1:
                    holding_orders = holding_orders.append({'Date': date, symbol: 2000}, ignore_index=True)
                    current_holdings += 2000
                else:
                    holding_orders = holding_orders.append({'Date': date, symbol: 0}, ignore_index=True)
        return holding_orders

    # Manual Strategy Settings
    def check_value(self, sma_val, bbp_val, so_d):
        if (so_d < 22) and (sma_val < 0.95) or (bbp_val < 0):
            return 1
        elif (so_d > 78) and (sma_val > 1.05) or (bbp_val > 1):
            return -1
        else:
            return 0

    def testPolicy(self, symbols = ['JPM'], sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv = 100000):
        positions = pd.DataFrame(columns=['Date', 'Position'])

        sma, bbp, so, lookback, price = get_indicators(symbols, sd, ed)
        orders = price.copy()
        orders.iloc[:,:] = np.NaN

        prev = 0

        for day in range(lookback+1, price.shape[0]):
            date = price.iloc[day].name
            if prev == 0:
                positions = positions.append({'Date': date, 'Position': 0}, ignore_index=True)
                prev = date

            positions = positions.append({'Date': prev, 'Position': self.check_value(sma.iloc[day-1][0],
                                                                                     bbp.iloc[day-1][0],
                                                                                     so.iloc[day-1][5])},
                                         ignore_index=True)
            prev = date
            if date == ed:
                positions = positions.append({'Date': date, 'Position': self.check_value(sma.iloc[day][0],
                                                                                         bbp.iloc[day][0],
                                                                                         so.iloc[day][5])},
                                             ignore_index=True)

        holding_orders = self.get_order(positions, symbols[0])
        # manual_strategy = compute_portvals(holding_orders, price, sd, ed, sv, 9.95, 0.005)
        # norm_manual = manual_strategy / manual_strategy[0]
        #
        # benchmark = self.get_benchmark(symbols[0], price, sd, ed, sv)
        # norm_benchmark = benchmark / benchmark[0]

        # Plotting
        # fig1, ax = plt.subplots()
        # norm_manual.plot(label='Manual Strategy', color='#d63729')
        # for i, row in positions.iterrows():
        #     if row.values[1] == 1:
        #         ax.axvline(x=row.values[0], alpha=0.3, color='blue')
        #     elif row.values[1] == -1:
        #         ax.axvline(x=row.values[0], alpha=0.3, color='black')
        # norm_benchmark.plot(label='Benchmark', color='#20b049')
        # plt.title('Manual Strategy In Sample')
        # plt.ylabel('Normalized Portfolio Value')
        # plt.xlabel('Date')
        # plt.legend()
        # plt.grid()
        # fig1.savefig("MAN-InSample.png")
        # plt.close(fig1)

        holding_orders = holding_orders.set_index('Date')
        holding_orders = holding_orders[1:]
        return holding_orders


def test_code():
    symbols = ['JPM']
    start = dt.datetime(2008, 1, 1)
    end = dt.datetime(2009, 12, 31)
    sv = 100000
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbols, start, end, sv)
    return df_trades


if __name__ == "__main__":
    test_code()
