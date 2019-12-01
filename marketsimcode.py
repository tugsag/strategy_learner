"""MC2-P6: Manual Strategy

CS 4646/7646

Student Name: Grace Park
GT User ID: gpark83
GT ID: 903474899
"""

import pandas as pd
import numpy as np


def author():
    return 'gpark83'


def compute_portvals(orders_df, stock_val, start, end, start_val=100000, commission=0, impact=0):
    total_val = start_val

    # Take in data frame
    total_frame = pd.DataFrame(columns=['Date', 'Total Value'])

    # Values for pulling stock data
    stock_name = orders_df.columns[1]
    previous_daily_gain = 0
    total_shares = 0

    orders = stock_val.copy()

    for day in range(stock_val.shape[0]):
        orders.iloc[day,:] = 0

    for index, row in orders_df.iterrows():
        orders.at[row['Date']] = row[stock_name]

    orders.index.name = 'Date'
    orders.reset_index(inplace=True)

    for i, shares in orders.iterrows():
        date_time = orders.iloc[i]['Date']
        shares = abs(int(orders.iloc[i][stock_name]))
        stock_price = stock_val.loc[date_time][0]
        total_cost = stock_price * shares
        market_impact = total_cost * impact
        total_val -= commission + market_impact

        # Buy or Sell
        if int(orders.iloc[i][stock_name]) > 0:
            total_shares += shares
            total_val -= total_cost
        elif int(orders.iloc[i][stock_name]) < 0:
            total_shares -= shares
            total_val += total_cost

        daily_total = stock_price * total_shares
        todays_gain = daily_total
        total_val += daily_total - previous_daily_gain
        previous_daily_gain = todays_gain

        total_frame = total_frame.append({'Date': date_time, "Total Value": total_val}, ignore_index=True)

    total_frame = total_frame.set_index('Date')
    port_val = total_frame[total_frame.columns[0]]
    cum_ret = (port_val[-1] / port_val[0]) - 1
    daily_ret = (port_val / port_val.shift(1)) - 1
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret

    # print(f"Date Range: {start} to {end}")
    # print()
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    # print()
    # print(f"Cumulative Return of Fund: {cum_ret}")
    # print()
    # print(f"Standard Deviation of Fund: {std_daily_ret}")
    # print()
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")
    # print()
    # print(f"Final Portfolio Value: {port_val[-1]}")
    return port_val


def get_daily_ret(port_val):
    return (port_val / port_val.shift(1)) - 1


