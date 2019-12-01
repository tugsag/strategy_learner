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
from indicators import get_indicators
from marketsimcode import compute_portvals, get_daily_ret
import datetime as dt
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut
import numpy as np
import QLearner as ql
import random

class StrategyLearner(object):

    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			  	 		  		  		    	 		 		   		 		  
        self.impact = impact

    def get_order(self, positions, symbol):
        holding_orders = pd.DataFrame(columns=['Date', symbol])
        current_holdings = 0

        for holding in positions.iterrows():
            date = holding[1]['Date']
            position = holding[1]['Position']

            if current_holdings == 0:
                if position == 2:
                    holding_orders = holding_orders.append({'Date': date, symbol: -1000}, ignore_index=True)
                    current_holdings -= 1000
                elif position == 1:
                    holding_orders = holding_orders.append({'Date': date, symbol: 1000}, ignore_index=True)
                    current_holdings += 1000
                else:
                    holding_orders = holding_orders.append({'Date': date, symbol: 0}, ignore_index=True)
            elif current_holdings == 1000:
                if position == 2:
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

        # add your code to do learning here
        window = 20
        sma, bbp, so, price, normalized_price = get_indicators(symbol, sd, ed, window)
        so = so['%D']
        indicators = pd.concat([sma, bbp, so], axis=1)
        indicators.columns = ['SMA', 'BBP', 'SO']

        # Discretize
        num_steps = 10

        sma_copy = sma[symbol].to_numpy()
        sma_out, sma_bins = pd.qcut(sma_copy, num_steps, retbins=True, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], duplicates='drop')

        bbp_copy = bbp[symbol].to_numpy()
        bbp_out, bbp_bins = pd.qcut(bbp_copy, num_steps, retbins=True, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    duplicates='drop')

        so_copy = so.values
        so_out, so_bins = pd.qcut(so_copy, num_steps, retbins=True, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    duplicates='drop')

        sma_state = pd.cut(indicators['SMA'], bins=sma_bins, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        bbp_state = pd.cut(indicators['BBP'], bins=bbp_bins, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        so_state = pd.cut(indicators['SO'], bins=so_bins, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        indicators_states = pd.DataFrame(np.zeros(len(sma_state)))

        for i in range(len(sma_state)):
            indicators_states.iloc[i] = (sma_state.iloc[i] * 100) + (bbp_state.iloc[i] * 10) + (so_state.iloc[i])

        indicators_states.dropna(inplace=True)
        # Initialize QLearner
        qlearner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0,
                            verbose=False)

        converged = False
        epoch_counter = 0

        while not converged:
            epoch_counter += 1

            if epoch_counter > 20 and prev_holding_orders.equals(holding_orders):
                converged = True

            action = qlearner.querysetstate(int(indicators_states.iloc[0]))
            positions = pd.DataFrame(columns=['Date', 'Position'])
            positions = positions.append({'Date': sma_state.index[0], 'Position': 0}, ignore_index=True)

            for day in range(1, indicators_states.shape[0]):
                holding = action
                date = sma_state.index[day]
                price_index = price.index.get_loc(date)

                daily_return = (price.iloc[price_index] / price.iloc[price_index-1]) - 1

                reward = holding * daily_return

                action = qlearner.query(int(float(indicators_states.iloc[day])), int(reward))
                positions = positions.append({'Date': date, 'Position': action}, ignore_index=True)
            holding_orders = self.get_order(positions, symbol)
            strategy_learner = compute_portvals(holding_orders, price, sd, ed, sv, 9.95, 0.005)
            prev_holding_orders = holding_orders.copy()

        print(strategy_learner)
    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):  		   	  			  	 		  		  		    	 		 		   		 		  

        # here we build a fake set of trades
        # your code should return the same sort of data  		   	  			  	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		   	  			  	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
        trades = prices_all[[symbol,]]  # only portfolio symbols  		   	  			  	 		  		  		    	 		 		   		 		  
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[:,:] = 0 # set them all to nothing  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[0,:] = 1000 # add a BUY at the start  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[40,:] = -1000 # add a SELL  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[41,:] = 1000 # add a BUY  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[60,:] = -2000 # go short from long  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[61,:] = 2000 # go long from short  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[-1,:] = -1000 #exit on the last day
        # if self.verbose: print(type(trades)) # it better be a DataFrame!
        # if self.verbose: print(trades)
        # if self.verbose: print(prices_all)
        print(trades)
        return trades  		   	  			  	 		  		  		    	 		 		   		 		  

if __name__=="__main__":
    learner = StrategyLearner()
    learner.addEvidence(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    learner.testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
    print("One does not simply think up a strategy")  		   	  			  	 		  		  		    	 		 		   		 		  
