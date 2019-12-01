import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose = False):
        self.tree = np.array([])
        self.leaf_size = leaf_size

    def author(self):
        return 'gpark83'

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner and build the decision tree
        :param dataX: X values of data to add
        :param dataY: Y training values
        """
        self.tree = self.build_tree(dataX, dataY)

    def query(self, dataX):
        """
        @summary: Estimate
        Reference: https://gatech.instructure.com/courses/61304/pages/decision-trees-part-1?module_item_id=419818
        https://gatech.instructure.com/courses/61304/pages/decision-trees-part-2?module_item_id=419820
        :param dataX:
        :return:
        """
        query_results = []

        for i in range(len(dataX)):
            curr_node = 0
            factor = int(self.tree[0][0])
            factor_value = dataX[i][factor]
            leaf_found = False

            while leaf_found is False:
                check_value = self.tree[curr_node][1]
                if factor == -1:
                    query_results.append(check_value)
                    leaf_found = True
                else:
                    if factor_value > check_value:
                        curr_node = int(self.tree[curr_node][3] + curr_node)
                    else:
                        curr_node = int(self.tree[curr_node][2] + curr_node)
                    factor = int(self.tree[curr_node][0])
                    factor_value = dataX[i][factor]
        return np.asarray(query_results)

    def build_tree(self, dataX, dataY):
        """
        Reference: https://gatech.instructure.com/courses/61304/pages/decision-trees-part-1?module_item_id=419818
        https://gatech.instructure.com/courses/61304/pages/decision-trees-part-2?module_item_id=419820
        :param dataX:
        :param dataY:
        :return: Tree
        """
        # -1 is a leaf
        if dataX.shape[0] <= self.leaf_size:
            return np.array([[-1, dataY.mean(), -1, -1]])

        if np.all(dataY == dataY[0]):
            return np.array([[-1, dataY[0], -1, -1]])
        else:
            best_feat = self.get_split_feat(dataX)
            split_val = np.median(dataX[:, best_feat])

            if split_val >= max(dataX[:, best_feat]) or split_val <= min(dataX[:, best_feat]):
                return np.array([[-1, dataY[0], -1, -1]])

            left_dataX = dataX[dataX[:, best_feat] <= split_val]
            left_dataY = dataY[dataX[:, best_feat] <= split_val]
            left_tree = self.build_tree(left_dataX, left_dataY)

            right_dataX = dataX[dataX[:, best_feat] > split_val]
            right_dataY = dataY[dataX[:, best_feat] > split_val]
            right_tree = self.build_tree(right_dataX, right_dataY)

            root = np.array([[best_feat, split_val, 1, left_tree.shape[0] + 1]])
            return np.concatenate((root, left_tree, right_tree), axis=0)

    @staticmethod
    def get_split_feat(currX):
        random_index = np.random.randint(currX.shape[1])
        return random_index
