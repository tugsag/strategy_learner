"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  

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

import numpy as np
import random as rand  		   	  			  	 		  		  		    	 		 		   		 		  


class QLearner(object):

    def __init__(self, num_states=100,\
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.98, \
        radr = 0.999, \
        dyna = 0, \
        verbose=False): #initialize the learner
            self.verbose = verbose
            self.num_actions = num_actions
            self.s = 0
            self.a = 0
            self.Q = np.zeros(shape=[num_states, num_actions])
            self.alpha = alpha
            self.gamma = gamma
            self.rar = rar
            self.radr = radr
            self.dyna = dyna

            if self.dyna > 0:
                self.T = []

    def author(self):
        return 'gpark83'

    def querysetstate(self, s):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        # for rar call a random generator for 0 - 1, and if it is less than rar, you take a random action.
        # the intent of dyna is to sample/hallucinate from experiences you've already had
        # constructing t & r is one way, but there are other ways
        # test is that your policy converges to a value in lower number of iterations
        if rand.uniform(0, 1) < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])

        self.s = s
        self.a = action

        if self.verbose: print(f"s = {s}, a = {action}")
        
        return self.a

    def query(self, s_prime, r):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        # Q Table Update
        self.Q[self.s, self.a] = ((1 - self.alpha)*self.Q[self.s, self.a]) + \
                                 (self.alpha * (r + self.gamma * (self.Q[s_prime, np.argmax(self.Q[s_prime, :])])))

        # Dyna - Reference Lecture: Navigation Project
        if self.dyna > 0:
            # Build T Model
            self.T.append([self.s, self.a, s_prime, r])

            # Hallucination
            random_select = np.random.randint(0, len(self.T), size=self.dyna)

            for index in random_select:
                dyna_s = self.T[index][0]
                dyna_a = self.T[index][1]
                dyna_s_prime = self.T[index][2]
                dyna_r = self.T[index][3]

                self.Q[dyna_s, dyna_a] += self.alpha * (dyna_r + np.max(self.Q[dyna_s_prime, :]) -
                                                        self.Q[dyna_s, dyna_a])

        action = self.querysetstate(s_prime)

        self.rar *= self.radr

        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}")
        
        return action  		   	  			  	 		  		  		    	 		 		   		 		  


if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
