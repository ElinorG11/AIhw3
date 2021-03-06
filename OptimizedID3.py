# The file was empty. I've created signatures of the required classes\methods from the
# pdf for convenience.

from ID3 import ID3, ID3Tree, get_data_from_df, log, DEFAULT_CLASSIFICATION
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt


class PersonalizedID3(ID3):
    def __init__(self):
        self.id3tree = None

    def predict(self, x, y):
        data = x.copy()
        data["diagnosis"] = y
        num_of_samples = len(data.index)
        predictions_array = np.ndarray([num_of_samples])

        for row in range(len(data.index)):
            prediction = self.tree_traversal(self.id3tree, row, data)
            if prediction == "M":
                predictions_array[row] = 1
            else:
                predictions_array[row] = 0
        return predictions_array

    def fit_predict(self, train, test, validation_ratio=0.425):
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
        x, y = get_data_from_df(df_train)
        x_test, y_test = get_data_from_df(df_test)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=validation_ratio)
        # fit the classifier
        validation_data = pd.DataFrame(x_valid.copy())
        validation_data["diagnosis"] = pd.DataFrame(y_valid)
        train_data = pd.DataFrame(x_train.copy())
        train_data["diagnosis"] = pd.DataFrame(y_train)
        self.id3tree = ID3Tree(data=train_data)
        # prune the tree
        pruned_tree = self.prune(self.id3tree, validation_data)
        self.id3tree = pruned_tree
        # predict using test dataset
        predictions = self.predict(x_test, y_test)
        return predictions

    def check_leaf(self, Node: ID3Tree, data):
        """
        function to check whether current node is a leaf
        :param Node:
        :param data:
        :return:
        """
        return len(data.index) == 0 or Node.is_leaf()

    def prune(self, Node: ID3Tree, v_data):
        """
        function to prune the tree for optimizing loss
        :param Node: is an ID3 tree node
        param v_data: validation data used for pruning
        :return: pruned tree
        :rtype: ID3Tree
        """

        # stop condition - check if node is leaf
        if self.check_leaf(Node, v_data) == 0:
            return Node

        # perform slicing of the data by feature values
        data_left = []
        data_right = []
        for value in v_data[Node.feature]:
            if value <= Node.slice_thresh:
                data_left.append(value)
            else:
                data_right.append(value)

        # recursively prune subtrees
        Node.left = self.prune(Node.left, data_left)
        Node.right = self.prune(Node.right, data_right)

        # get new root diagnosis by costs
        count_b, count_m = 0, 0
        for x in v_data["diagnosis"]:
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
        for row in range(len(v_data.index)):
            prediction_classification = self.tree_traversal(Node, row, v_data)  # our trees prediction
            real_classification = v_data["diagnosis"].iloc[row]
            err_prune += self.Evaluate(majority_diagnosis, real_classification)  # error if pruned (with validation)
            err_no_prune += self.Evaluate(prediction_classification, real_classification)  # error without prune

        # it will be better to prune - leaf should have only a diagnosis so we can identify it
        if err_prune < err_no_prune:
            Node.data = None
            Node.feature = None
            Node.left = None
            Node.right = None
            Node.slice_thresh = None
            Node.diagnosis = majority_diagnosis
        return Node

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
        false_negative = 0
        false_positive = 0
        real_binary_diagnosis = []
        for row in range(len(data.index)):
            if data["diagnosis"].iloc[row] == "M":
                real_binary_diagnosis.append(1)
            else:
                real_binary_diagnosis.append(0)
        for row in range(len(data.index)):
            prediction = predictions[row]
            if prediction != real_binary_diagnosis[row]:  # pred is correct
                if prediction == 1:  # person is healthy but we predicted sick
                    false_positive += 1
                else:  # person is sick but we predicted healthy
                    false_negative += 1
        loss = (false_positive + 8 * false_negative)
        return loss


def experiment(all_data, graph=False):
    """
    # TODO in order to see accuracy value, please uncomment in main part the first "TODO"
    graph: option to plot graph
    """
    x, y = get_data_from_df(all_data)
    x = x.to_numpy()
    y = y.to_numpy()
    kf = KFold(n_splits=5, random_state=314985664, shuffle=True)
    avg_loss_list = []
    k_classifier = PersonalizedID3()
    v_ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for v in v_ratios:
        losses = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train = np.concatenate((y_train, x_train), axis=1)
            test = np.concatenate((y_test, x_test), axis=1)
            predictions = k_classifier.fit_predict(train, test, v)
            loss = k_classifier.calculate_loss_and_accuracy(x_test, y_test, predictions)
            losses.append(loss)
        avg_loss_list.append(sum(losses) / float(len(losses)))
    if graph:
        print(f"Loss list: {avg_loss_list}")
        print(f"Value of best loss is: {min(avg_loss_list)}")
        plt.plot(v_ratios, avg_loss_list)
        plt.xlabel("Validation ratio")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":
    # get numpy ndarray from csv
    train = genfromtxt('train.csv', delimiter=',', dtype="unicode")

    # we send only test dataset to experiment function
    data = pd.DataFrame(train)

    # TODO: to run the experiment, please uncomment the following line
    experiment(data, graph=True)
