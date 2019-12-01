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
from indicators import get_indicators
from marketsimcode import compute_portvals

def author():
    return 'gpark83'

class StrategyLearner(object):

    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
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

        lookback = 14

        # norm_prices = prices.divide(prices.ix[0])
        # daily_rets = self.compute_daily_returns(norm_prices, lookback, symbol)

        # Get indicators
        # rolling_mean = norm_prices.rolling(window=lookback).mean()
        # rolling_std = norm_prices.rolling(window=lookback).std()
        # up = (rolling_std * 2) + rolling_mean
        # down = (rolling_std * -2) + rolling_mean

        # df = norm_prices.copy()
        # df['bbp'] = (norm_prices - down) / (up - down)
        # df['Price/SMA'] = norm_prices / rolling_mean
        # df['bollinger_up'] = (rolling_std * 2) + rolling_mean
        # df['bollinger_down'] = (rolling_std * -2) + rolling_mean
        # df['Price/EMA'] = norm_prices / norm_prices.ewm(span=lookback).mean()
        # df = df.drop(symbol, 1)
        YBUY = 0.02
        YSELL = -0.02

        price_sma, bbp, so, price, norm_price = get_indicators(syms, sd, ed, lookback)
        concat_frames = [price_sma, bbp, so['%D']]
        indicators = pd.concat(concat_frames, axis=1)
        indicators.columns = ['Price/SMA', 'BBP', 'SO']
        indicators.fillna(0, inplace=True)
        trainingX = indicators.values

        daily_rets = self.get_daily_ret(price, lookback, syms)

        trainingY = []
        # indicators.loc[daily_rets > (YBUY + self.impact), Y] = 1
        # indicators.loc[daily_rets < (YSELL + self.impact), Y] = -1

        for index, row in price.iterrows():
            if daily_rets.loc[index, symbol] > (YBUY + self.impact):
                trainingY.append(1)
            elif daily_rets.loc[index, symbol] < (YSELL + self.impact):
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

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)

        lookback = 14

        # norm_prices = prices.divide(prices.ix[0])
        # daily_rets = self.compute_daily_returns(norm_prices, lookback, symbol)

        # Get indicators
        # rolling_mean = norm_prices.rolling(window=lookback).mean()
        # rolling_std = norm_prices.rolling(window=lookback).std()
        # up = (rolling_std * 2) + rolling_mean
        # down = (rolling_std * -2) + rolling_mean

        # df = norm_prices.copy()
        # df['bbp'] = (norm_prices - down) / (up - down)
        # df['Price/SMA'] = norm_prices / rolling_mean
        # df['bollinger_up'] = (rolling_std * 2) + rolling_mean
        # df['bollinger_down'] = (rolling_std * -2) + rolling_mean
        # df['Price/EMA'] = norm_prices / norm_prices.ewm(span=lookback).mean()
        # df = df.drop(symbol, 1)
        YBUY = 0.02
        YSELL = -0.02

        price_sma, bbp, so, price, norm_price = get_indicators(syms, sd, ed, lookback)
        concat_frames = [price_sma, bbp, so['%D']]
        indicators = pd.concat(concat_frames, axis=1)
        indicators.columns = ['Price/SMA', 'BBP', 'SO']
        indicators.fillna(0, inplace=True)
        testingX = indicators.values
        testingY = self.learner.query(testingX)

        # print(testingY)
        positions = pd.DataFrame(columns=['Date', 'Position'])
        prev = 0
        for day in range(lookback+1, price.shape[0]):
            date = price.iloc[day].name
            if prev == 0:
                positions = positions.append({'Date': date, 'Position': 0}, ignore_index=True)
                prev = date

            positions = positions.append({'Date': prev, 'Position': self.check_value(testingY[day])}, ignore_index=True)
            prev = date
            if date == ed:
                positions = positions.append({'Date': date, 'Position': self.check_value(testingY[day])}, ignore_index=True)

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

        strategy_learner = compute_portvals(holding_orders, price, sd, ed, sv, 9.95, 0.005)
        holding_orders = holding_orders.set_index('Date')
        holding_orders = holding_orders[1:]
        return holding_orders

if __name__=="__main__":
    learner = StrategyLearner()
    learner.addEvidence(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    learner.testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    print("One does not simply think up a strategy")
