"""MC2-P6: Manual Strategy

CS 4646/7646

Student Name: Grace Park
GT User ID: gpark83
GT ID: 903474899
"""

import pandas as pd
import numpy as np
from util import get_data
import matplotlib.pyplot as plt


def author():
    return 'gpark83'


def normalize(df):
    return df / df[df.first_valid_index()]


def get_indicators(symbols, sd, ed):
    lookback = 14

    price = get_data(symbols, pd.date_range(sd, ed), False)

    # Indicator 1: Simple Moving Average
    # Reference: Vectorize Me! Lecture David Byrd
    price = price.dropna()
    sma = price.cumsum()
    sma.values[lookback:,:] = (sma.values[lookback:,:] - sma.values[:-lookback,:]) / lookback
    sma.iloc[:lookback,:] = np.nan
    price_sma = price[symbols[0]] / sma[symbols[0]]
    normalized_price = normalize(price[symbols[0]])
    normalized_sma = normalize(sma[symbols[0]])

    fig1 = plt.figure()
    plt.subplot(211)
    plt.title('Indicator 1: Simple Moving Average')
    plt.ylabel('Normalized Value')
    plt.tick_params(axis='x', labelbottom=False)
    normalized_price.plot(label='JPM Price')
    normalized_sma.plot(label='SMA')
    plt.grid()
    plt.legend()
    plt.subplot(212)
    price_sma.plot(label='Price/SMA', color='green')
    plt.axhline(y=1, color='red', linestyle='-')
    plt.ylabel('Normalized Value')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    fig1.savefig("SMA.png")
    plt.close(fig1)

    # Indicator 2: Bollinger Bands %B
    # Reference: Vectorize Me! Lecture David Byrd
    bbp = price.copy()
    for day in range(price.shape[0]):
        bbp.iloc[day,:] = 0

    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (price - bottom_band) / (top_band - bottom_band)

    top_band = top_band[symbols[0]]
    bottom_band = bottom_band[symbols[0]]

    fig2 = plt.figure()
    plt.subplot(211)
    top_band.plot(label="Upper Band", color='#a2a3a2')
    bottom_band.plot(label="Lower Band", color='#a2a3a2')
    plt.fill_between(bbp.index.get_level_values(0), top_band, bottom_band, color='#cacccb')
    price[symbols[0]].plot(label='Price')
    sma[symbols[0]].plot(label="SMA")
    plt.title('Indicator 2: Bollinger Band Percent')
    plt.ylabel('Value')
    plt.tick_params(axis='x', labelbottom=False)
    plt.grid()
    plt.legend(fontsize='x-small', loc="lower right")
    plt.subplot(212)
    bbp[symbols[0]].plot(label="Bollinger Band %", color='green')
    plt.axhline(y=0, color='red', linestyle='-')
    plt.axhline(y=1, color='red', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid()
    plt.legend(fontsize='x-small', loc="lower right")
    fig2.savefig("BBP.png")
    plt.close(fig2)

    # Indicator 3: Stochastic Oscillator
    # References:
    # https://www.investopedia.com/terms/s/stochasticoscillator.asp
    # https://pythonforfinance.net/2017/10/10/stochastic-oscillator-trading-strategy-backtest-in-python/
    # http://www.andrewshamlet.net/2017/07/13/python-tutorial-stochastic-oscillator/

    high = get_data(symbols, pd.date_range(sd, ed), True, 'High')
    low = get_data(symbols, pd.date_range(sd, ed), True, 'Low')
    close = get_data(symbols, pd.date_range(sd, ed), True, 'Close')

    so = high.copy()

    for day in range(high.shape[0]):
        so.iloc[day,:] = 0

    so['16 Day Low'] = low[symbols[0]].rolling(window=16).min()
    so['16 Day High'] = high[symbols[0]].rolling(window=16).max()
    so['%K'] = ((close[symbols[0]] - so['16 Day Low']) / (so['16 Day High'] - so['16 Day Low'])) * 100
    so['%D'] = so['%K'].rolling(window=3).mean()

    fig3 = plt.figure()
    plt.subplot(211)
    plt.title('Indicator 3: Stochastic Oscillator')
    plt.ylabel('Value')
    plt.tick_params(axis='x', labelbottom=False)
    so['%K'].plot(label='%K')
    so['%D'].plot(label='%D')
    plt.grid()
    plt.legend(fontsize='x-small', loc="upper right")
    plt.subplot(212)
    close[symbols[0]].plot(label='Close', color='green')
    plt.xlabel('Date')
    plt.grid()
    plt.ylabel('Value')
    plt.legend()
    fig3.savefig('SO.png')

    return price/sma, bbp, so, lookback, price
