import numpy as np
import DTLearner as dt

class BagLearner(object):

    def __init__(self, learner=dt.DTLearner, kwargs={}, bags=20, boost=False, verbose=False):
        learners = []
        for i in range(0, bags):
            learners.append(learner(**kwargs))

        self.learners = learners
        self.kwargs = kwargs
        self.leaf_size = kwargs.get("leaf_size", 1)
        self.verbose = verbose
        self.bags = bags
        self.boost = boost

    def author(self):
        return 'gpark83'

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner and build the forest
        :param dataX: X values of data to add
        :param dataY: Y training values
        """
        for i in range(len(self.learners)):
            random_subset = np.random.choice(dataX.shape[0], dataX.shape[0])
            self.learners[i].addEvidence(dataX[random_subset], dataY[random_subset])

    def query(self, dataX):
        """
        @summary: Estimate
        :param dataX:
        :return: Query Results
        """
        query_results = []

        for i in range(len(self.learners)):
            query_results.append(self.learners[i].query(dataX))

        return np.mean(np.array(query_results), axis=0)
