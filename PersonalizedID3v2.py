# The file was empty. I've created signatures of the required classes\methods from the
# pdf for convenience.

from ID3 import ID3, ID3Tree, get_data_from_df, log, DEFAULT_CLASSIFICATION
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt


class PersonalizedID3:
    def __init__(self, test_size, data=None):
        self.test_size = test_size
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

    def fit_predict(self, train, test):
        """
        Classifier to utilize ID3 tree.
        fitting the data into ID3 tree and predicts the diagnosis for data set x.
        computes accuracy and loss for y. (?)
        :param train: dataset
        :type: numpy.ndarray
        :param test: dataset
        :return: boolean array of size (#number_f_samples) where '1' indicates "B" and '0' indicates "M"
        :rtype: numpy.ndarray
        """
        # retrieve the data from the csv file
        df_train = pd.DataFrame(train)
        df_test = pd.DataFrame(test)
        train_x, train_y = get_data_from_df(df_train)
        test_x, test_y = get_data_from_df(df_test)
        # fit the classifier
        self.fit(train_x, train_y)
        # predict using test dataset
        predictions = self.predict(test_x, test_y)
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
        if count_b < count_m * 8:
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
            return 8 if real_classification == "M" else 1
        else:
            return 0

    def calculate_loss_and_accuracy(self, test_x, test_y, predictions):
        test_x = pd.DataFrame(test_x)
        test_y = pd.DataFrame(test_y)
        data = test_x.copy()
        data["diagnosis"] = test_y
        correct_predictions = 0
        false_negative = 0
        false_positive = 0
        num_of_samples = len(data.index)
        real_binary_diagnosis = []
        for row in range(len(data.index)):
            if data["diagnosis"].iloc[row] == "M":
                real_binary_diagnosis.append(1)
            else:
                real_binary_diagnosis.append(0)
        for row in range(len(data.index)):
            prediction = predictions[row]
            if prediction == real_binary_diagnosis[row]:  # pred is correct
                correct_predictions += 1
            else:
                if prediction == 1:  # person is healthy but we predicted sick
                    false_positive += 1
                else:  # person is sick but we predicted healthy
                    false_negative += 1

        accuracy = float(correct_predictions) / float(num_of_samples)
        loss = (false_positive + 8 * false_negative)
        return accuracy, loss


def experiment(train, graph=False):
    raise NotImplementedError