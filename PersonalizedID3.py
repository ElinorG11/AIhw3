# The file was empty. I've created signatures of the required classes\methods from the
# pdf for convenience.

from ID3 import ID3, ID3Tree
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


class PersonalizedID3:
    def __init__(self, cost_false_negative, cost_false_positive, data=None):
        self.cost_false_negative = cost_false_negative
        self.cost_false_positive = cost_false_positive
        self.data = data
        self.id3tree = None

    def fit(self, x, y):
        validation_data = pd.DataFrame(x.copy())
        validation_data["diagnosis"] = pd.DataFrame(y)
        id3tree = ID3Tree(data=validation_data)
        pruned_tree = self.prune(id3tree, self.data)
        self.id3tree = pruned_tree

    def predict(self, x, y):
        data = x.copy()
        data["diagnosis"] = y
        num_of_samples = len(data.index)
        predictions_array = np.ndarray([num_of_samples])

        for row in range(len(data.index)):
            id3 = ID3()
            prediction = id3.tree_traversal(self.id3tree, row, data)
            if prediction == "M":
                predictions_array[row] = 1
            else:
                predictions_array[row] = 0
        return predictions_array

    def fit_predict(self, x_train, x_test, y_train, y_test):
        """
        Classifier to utilize ID3 tree.
        fitting the data into ID3 tree and predicts the diagnosis for data set x.
        computes accuracy and loss for y. (?)
        :param x_train, x_test, y_train, y_test: dataset
        :type: array
        :return: boolean array of size (#number_f_samples) where '1' indicates "B" and '0' indicates "M"
        :rtype: numpy.ndarray
        """

        # fit the classifier
        self.fit(x_train, y_train)
        # predict using test dataset
        predictions = self.predict(x_test, y_test)

        return predictions

    def check_leaf(self, T: ID3Tree, data):
        """
        function to check whether current node is a leaf
        :param T:
        :param data:
        :return:
        """
        return len(data.index) == 0 or T.is_leaf()

    def prune(self, T: ID3Tree, data):
        """
        function to prune the tree for optimizing loss
        :param T: is an ID3 tree node
        :return: pruned tree
        :rtype: ID3Tree
        """

        # stop condition - check if node is leaf
        if self.check_leaf(T, data) == 0:
            return T

        # perform slicing of the data by feature values
        data_left = []
        data_right = []
        for value in self.data[T.feature]:
            if value <= T.slice_thresh:
                data_left.append(value)
            else:
                data_right.append(value)

        # recursively prune subtrees
        T.left = self.prune(T.left, data_left)
        T.right = self.prune(T.right, data_right)

        # get new root diagnosis by costs
        count_b, count_m = 0, 0
        for x in data["diagnosis"]:
            if x == "M":
                count_m += 1
            else:
                count_b += 1
        if count_b * self.cost_false_positive < count_m * self.cost_false_negative:
            majority_diagnosis = "M"
        else:
            majority_diagnosis = "B"

        # evaluate error pruning vs no pruning
        err_prune, err_no_prune = 0, 0
        for row in range(len(data.index)):
            id3 = ID3()
            prediction_classification = id3.tree_traversal(T, row, data)
            real_classification = data["diagnosis"].iloc[row]
            err_prune += self.Evaluate(majority_diagnosis, real_classification)
            err_no_prune += self.Evaluate(prediction_classification, real_classification)

        # it will be better to prune - leaf should have only a diagnosis so we can identify it
        if err_prune < err_no_prune:
            T.data = None
            T.feature = None
            T.left = None
            T.right = None
            T.slice_thresh = None
            T.diagnosis = majority_diagnosis
        return T

    def Evaluate(self, prediction_classification, real_classification):
        if prediction_classification != real_classification:
            return self.cost_false_negative if real_classification == "M" else self.cost_false_positive
        else:
            return 0


def experiment(train):
    """
    function to find the best size of test subgroup for optimizing loss.
    :param train: train dataset
    :type train: ndarray
    """
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(train)

    # setting up default params
    costs_negative, costs_positive = [], []
    avg_losses = []
    for cost_n, cost_p in zip(costs_negative, costs_positive):
        losses = []
        classifier = PersonalizedID3(cost_n, cost_p)
        kf = KFold(n_splits=5, random_state=314985664, shuffle=True)
        for train_index, test_index in kf.split(train):
            x_train, x_test = train_df.iloc[train_index], train_df.iloc[test_index]
            y_train, y_test = test_df.iloc[train_index], test_df.iloc[test_index]
            acc, loss = classifier.fit_predict(x_train, x_test, y_train, y_test)
            losses.append(loss)
        avg = sum(losses) / len(losses)
        avg_losses.append(avg)
        print(f"loss={avg}")
    # print shit
    # generate graph
