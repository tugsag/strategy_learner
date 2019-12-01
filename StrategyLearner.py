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
        YSELL = -0.02

        concat_frames = [price_sma, bbp, so_d]
        indicators = pd.concat(concat_frames, axis=1)
        indicators.columns = ['Price/SMA', 'BBP', 'SO']
        indicators = indicators.fillna(value=0).copy()
        trainingX = indicators.values

        trainingY = []

        for index, row in normalized_prices.iterrows():
            if daily_ret.loc[index] > (YBUY + self.impact):
                trainingY.append(1)
            elif daily_ret.loc[index] < (YSELL + self.impact):
                trainingY.append(-1)
            else:
                trainingY.append(0)

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

        concat_frames = [price_sma, bbp, so_d]
        indicators = pd.concat(concat_frames, axis=1)
        indicators.columns = ['Price/SMA', 'BBP', 'SO']
        indicators = indicators.fillna(value=0).copy()
        testingX = indicators.values

        testingY = []

        for index, row in normalized_prices.iterrows():
            if daily_ret.loc[index] > (YBUY + self.impact):
                testingY.append(1)
            elif daily_ret.loc[index] < (YSELL + self.impact):
                testingY.append(-1)
            else:
                testingY.append(0)

        predY = self.learner.query(testingX)
        trades = prices_all[[symbol, ]]

        YBUY = 0.02
        YSELL = -0.02

        trades.values[:, :] = 0
        trades.values[predY > (YBUY + 2 * self.impact)] = 1000
        trades.values[predY < (YSELL - 2 * self.impact)] = -1000

        holding_orders = trades.diff()

        holding_orders[symbol][0] = trades[symbol][0]

        return holding_orders


if __name__=="__main__":
    learner = StrategyLearner()
    learner.addEvidence(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    learner.testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    print("One does not simply think up a strategy")
