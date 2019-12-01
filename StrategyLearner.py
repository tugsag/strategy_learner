"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  

-----do not edit anything above this line---

Student Name: Grace Park
GT User ID: gpark83
GT ID: 903474899
"""  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut
import numpy as np
import random
import RTLearner as rt
import BagLearner as bl
from indicators import get_indicators, normalize
from marketsimcode import compute_portvals

def author():
    return 'gpark83'

class StrategyLearner(object):

    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.lookback = 14
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False, verbose=False)

    def get_daily_ret(self, df, day, symbol):
        daily_ret = df.copy()
        daily_ret[:-day] = (daily_ret[day:].values / daily_ret[: -day].values) - 1
        daily_ret.ix[-day:] = 0
        return daily_ret[symbol]

    def check_value(self, testingY):
        if testingY > 0:
            return 1
        elif testingY < 0:
            return -1
        else:
            return 0

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)

        normalized_prices = prices / prices.loc[prices.first_valid_index()]
        daily_ret = normalized_prices.copy()
        daily_ret[:-self.lookback] = (daily_ret[self.lookback:].values / daily_ret[: -self.lookback].values) - 1
        daily_ret.iloc[-self.lookback:] = 0
        daily_ret = daily_ret[symbol]

        rolling_mean = normalized_prices.rolling(window=self.lookback, min_periods=self.lookback).mean()
        price_sma = normalized_prices / rolling_mean

        rolling_std = normalized_prices.rolling(window=self.lookback, min_periods=self.lookback).std()
        top_band = rolling_mean + (2 * rolling_std)
        bottom_band = rolling_mean - (2 * rolling_std)
        bbp = (normalized_prices - bottom_band) / (top_band - bottom_band)

        high = normalized_prices.loc[sd:ed]
        low = normalized_prices.loc[sd:ed]
        close = normalized_prices.loc[sd:ed]

        so = high.copy()

        for day in range(high.shape[0]):
            so.iloc[day, :] = 0

        so['16 Day Low'] = low[symbol].rolling(window=16).min()
        so['16 Day High'] = high[symbol].rolling(window=16).max()
        so['%K'] = ((close[symbol] - so['16 Day Low']) / (so['16 Day High'] - so['16 Day Low'])) * 100
        so['%D'] = so['%K'].rolling(window=3).mean()

        so_d = so['%D']

        YBUY = 0.01
        YSELL = 0

        # price_sma, bbp, so, price, norm_price = get_indicators(syms, sd, ed, self.lookback)
        concat_frames = [price_sma, bbp, so_d]
        indicators = pd.concat(concat_frames, axis=1)
        indicators.columns = ['Price/SMA', 'BBP', 'SO']

        trainingX = indicators.values
        # daily_ret = self.get_daily_ret(normalized_prices, self.lookback, syms)

        trainingY = []
        # df = normalized_prices.copy()
        # df['Price/SMA'] = normalized_prices / rolling_mean
        # df['BBP'] = (normalized_prices - bottom_band) / (top_band - bottom_band)
        # df['SO'] = so_d
        # df['action'] = 0
        # df.loc[daily_ret > (YBUY + self.impact), 'action'] = 1
        # df.loc[daily_ret < (YSELL + self.impact), 'action'] = -1
        # df = df.fillna(value=0).copy()
        # data = df.values
        # X = data[:, :-1]
        # Y = data[:, -1].astype(dtype=int)
        # print(X)
        # print(Y)

        for index, row in normalized_prices.iterrows():
            if daily_ret.loc[index] > (YBUY + self.impact):
                trainingY.append(1)
            elif daily_ret.loc[index] < (YSELL + self.impact):
                trainingY.append(-1)
            else:
                trainingY.append(0)

        print(trainingY)

        trainingY = np.array(trainingY)

        self.learner.addEvidence(trainingX, trainingY)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)

        normalized_prices = prices / prices.loc[prices.first_valid_index()]
        daily_ret = normalized_prices.copy()
        daily_ret[:-self.lookback] = (daily_ret[self.lookback:].values / daily_ret[: -self.lookback].values) - 1
        daily_ret.iloc[-self.lookback:] = 0
        daily_ret = daily_ret[symbol]

        rolling_mean = normalized_prices.rolling(window=self.lookback, min_periods=self.lookback).mean()
        price_sma = normalized_prices / rolling_mean

        rolling_std = normalized_prices.rolling(window=self.lookback, min_periods=self.lookback).std()
        top_band = rolling_mean + (2 * rolling_std)
        bottom_band = rolling_mean - (2 * rolling_std)
        bbp = (normalized_prices - bottom_band) / (top_band - bottom_band)

        high = normalized_prices.loc[sd:ed]
        low = normalized_prices.loc[sd:ed]
        close = normalized_prices.loc[sd:ed]

        so = high.copy()

        for day in range(high.shape[0]):
            so.iloc[day, :] = 0

        so['16 Day Low'] = low[symbol].rolling(window=16).min()
        so['16 Day High'] = high[symbol].rolling(window=16).max()
        so['%K'] = ((close[symbol] - so['16 Day Low']) / (so['16 Day High'] - so['16 Day Low'])) * 100
        so['%D'] = so['%K'].rolling(window=3).mean()

        so_d = so['%D']

        YBUY = 0.01
        YSELL = 0

        # price_sma, bbp, so, price, norm_price = get_indicators(syms, sd, ed, self.lookback)
        concat_frames = [price_sma, bbp, so_d]
        indicators = pd.concat(concat_frames, axis=1)
        indicators.columns = ['Price/SMA', 'BBP', 'SO']

        testingX = indicators.values

        testingY = []
        # df = normalized_prices.copy()
        # df['Price/SMA'] = normalized_prices / rolling_mean
        # df['BBP'] = (normalized_prices - bottom_band) / (top_band - bottom_band)
        # df['SO'] = so_d
        # df['action'] = 0
        # df.loc[daily_ret > (YBUY + self.impact), 'action'] = 1
        # df.loc[daily_ret < (YSELL + self.impact), 'action'] = -1
        # df = df.fillna(value=0).copy()
        # data = df.values
        # X = data[:, :-1]
        # Y = data[:, -1].astype(dtype=int)
        # print(X)
        # print(Y)

        for index, row in normalized_prices.iterrows():
            if daily_ret.loc[index] > (YBUY + self.impact):
                testingY.append(1)
            elif daily_ret.loc[index] < (YSELL + self.impact):
                testingY.append(-1)
            else:
                testingY.append(0)

        answer = self.learner.query(testingX)
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol, ]]  # only portfolio symbols

        # print(testingY)
        YBUY = 0.01
        YSELL = 0

        trades.values[:, :] = 0  # set them all to nothing

        trades.values[answer > (YBUY + 2 * self.impact)] = 1000
        trades.values[answer < (YSELL - 2 * self.impact)] = -1000

        orders = trades.copy()
        orders = orders.diff()

        orders[symbol][0] = trades[symbol][0]

        if self.verbose: print
        type(orders)  # it better be a DataFrame!
        if self.verbose: print
        orders
        if self.verbose: print
        prices_all

        return orders
        # prev = 0
        # for day in range(lookback+1, price.shape[0]):
        #     date = price.iloc[day].name
        #     if prev == 0:
        #         positions = positions.append({'Date': date, 'Position': 0}, ignore_index=True)
        #         prev = date
        #
        #     positions = positions.append({'Date': prev, 'Position': self.check_value(testingY[day])}, ignore_index=True)
        #     prev = date
        #     if date == ed:
        #         positions = positions.append({'Date': date, 'Position': self.check_value(testingY[day])}, ignore_index=True)
        #
        # holding_orders = pd.DataFrame(columns=['Date', symbol])
        # current_holdings = 0
        #
        # for holding in positions.iterrows():
        #     date = holding[1]['Date']
        #     position = holding[1]['Position']
        #
        #     if current_holdings == 0:
        #         if position == -1:
        #             holding_orders = holding_orders.append({'Date': date, symbol: -1000}, ignore_index=True)
        #             current_holdings -= 1000
        #         elif position == 1:
        #             holding_orders = holding_orders.append({'Date': date, symbol: 1000}, ignore_index=True)
        #             current_holdings += 1000
        #         else:
        #             holding_orders = holding_orders.append({'Date': date, symbol: 0}, ignore_index=True)
        #     elif current_holdings == 1000:
        #         if position == -1:
        #             holding_orders = holding_orders.append({'Date': date, symbol: -2000}, ignore_index=True)
        #             current_holdings -= 2000
        #         else:
        #             holding_orders = holding_orders.append({'Date': date, symbol: 0}, ignore_index=True)
        #     elif current_holdings == -1000:
        #         if position == 1:
        #             holding_orders = holding_orders.append({'Date': date, symbol: 2000}, ignore_index=True)
        #             current_holdings += 2000
        #         else:
        #             holding_orders = holding_orders.append({'Date': date, symbol: 0}, ignore_index=True)
        #
        # strategy_learner = compute_portvals(holding_orders, price, sd, ed, sv, 9.95, 0.005)
        # holding_orders = holding_orders.set_index('Date')
        # holding_orders = holding_orders[1:]
        # return holding_orders

if __name__=="__main__":
    learner = StrategyLearner()
    learner.addEvidence(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    learner.testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    print("One does not simply think up a strategy")
