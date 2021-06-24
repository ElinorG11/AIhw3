# The file was empty. I've created signatures of the required classes\methods from the
# pdf for convenience.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import log2
from numpy import genfromtxt
from sklearn.model_selection import KFold

DEFAULT_CLASSIFICATION = "M"


def log(x):
    return x if x == 0 else log2(x)


def get_data_from_df(df):
    """
    function parses the Dataframe and divides it into "samples" and "labels"
    :return: x holds all the features of all the patients. x is of the form:
                                                                            0 f1 f2 ... fn
                                                                            1 f'1 f'2 ... f'n
                                                                            ...
                                                                            n f''1 f''2 ... f''n
            y holds the diagnosis of all the patients. y is of the form:
                                                                            0 diag_0
                                                                            1 diag_1
                                                                            ...
                                                                            n diag_n
    :rtype: both x,y are pandas.core.frame.DataFrame
    """
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0:1]
    return x, y


class ID3Tree:
    """
    Tree class creates the ID3 tree
    """

    def __init__(self, prune_thresh=-1, data=None):
        """
        Init function generates the tree.
        :param data: the dataset
        :type data: dataframe
        """
        self.feature = None
        self.slice_thresh = None
        self.left = None
        self.right = None
        self.data = data
        self.diagnosis = None
        self.prune_thresh = prune_thresh

        # check if we reached a leaf and whether it is homogenous
        leaf, diagnosis = self.check_homogenous_leaves(self.prune_thresh)
        if leaf:
            self.diagnosis = diagnosis
        else:
            # this is not a leaf so we update feature for slicing and slicing val
            self.feature, _, self.slice_thresh = self.choose_feature()
            # slice the dataframe
            data_left = self.data[pd.to_numeric(self.data[self.feature]) <= self.slice_thresh]
            data_right = self.data[pd.to_numeric(self.data[self.feature]) > self.slice_thresh]
            # recursively create more nodes
            self.left = ID3Tree(prune_thresh, data=data_left)
            self.right = ID3Tree(prune_thresh, data=data_right)

    def is_leaf(self):
        """
        decide whether current node is a leaf.
        :return: True if leaf
        :rtype: bool
        """
        return self.diagnosis is not None

    def get_threshold_list(self, feature):
        """
        define the separators to be the mean value of each adjacent pair in the samples array
        :param feature: a specific feature for which we want to calculate the thresholds
        :return: tuple
        """
        values = self.data[feature]
        diagnosis = self.data["diagnosis"]
        values_list = values.tolist()
        # creating the separator's list
        sorted_values = sorted(values_list, key=lambda x: x)
        separators_list = [(float(x) + float(y)) / 2 for x, y in zip(sorted_values, sorted_values[1:])]
        return values, diagnosis, separators_list

    def calculate_information_gain(self, feature):
        """
        find best separator value.
        :return: (Best IG for this feature, separator value)
        :rtype: tuple
        """
        values, diagnosis, separators_list = self.get_threshold_list(feature)
        best_ig = (float("-inf"), None)

        for separator in separators_list:
            num_of_smaller_samples, num_of_smaller_positive_samples, num_of_larger_samples, num_of_larger_positive_samples = self.recalculate_thresh_values(
                values, diagnosis, separator)
            # calculate the root's IG
            root_pos_prob = (num_of_larger_positive_samples + num_of_smaller_positive_samples) / len(values)
            root_neg_prob = 1 - root_pos_prob
            entropy_root = -root_pos_prob * log(root_pos_prob) - (root_neg_prob * log(root_neg_prob))
            if num_of_smaller_samples == 0 or num_of_larger_samples == 0:
                # since this separator won't help us, it will create a node identical to the root
                ig = 0
                if ig >= best_ig[0]:
                    best_ig = (ig, separator)
                continue
            # calculate the left son's IG
            lchild_pos_prob = num_of_smaller_positive_samples / num_of_smaller_samples
            lchild_neg_prob = 1 - lchild_pos_prob
            entropy_left = -lchild_pos_prob * log(lchild_pos_prob) - (lchild_neg_prob * log(lchild_neg_prob))
            # calculate the right son's IG
            rchild_pos_prob = num_of_larger_positive_samples / num_of_larger_samples
            rchild_neg_prob = 1 - rchild_pos_prob
            entropy_right = -rchild_pos_prob * log(rchild_pos_prob) - (rchild_neg_prob * log(rchild_neg_prob))

            ig = entropy_root - entropy_left * num_of_smaller_samples / len(
                values) - entropy_right * num_of_larger_samples / len(values)
            if ig >= best_ig[0]:
                best_ig = (ig, separator)

        if best_ig[1] is None:
            raise ValueError("separator not found!")
        return best_ig

    def recalculate_thresh_values(self, values, diagnosis, separator):
        """
        updates the values of size_bigger, bigger_positive, size_smaller, smaller_positive
        :param values: The values of features
        :param diagnosis: The diagnosis
        :param separator: divider from threshold_list
        :return: num_of_smaller_samples, num_of_smaller_positive_samples, num_of_larger_samples, num_of_larger_positive_samples
        :rtype: tuple
        """
        num_of_smaller_samples, num_of_smaller_positive_samples = 0, 0
        num_of_larger_samples, num_of_larger_positive_samples = 0, 0
        for val, diag in zip(values, diagnosis):
            if float(val) <= separator:
                num_of_smaller_samples += 1
                if diag == "M":
                    num_of_smaller_positive_samples += 1
            else:
                num_of_larger_samples += 1
                if diag == "M":
                    num_of_larger_positive_samples += 1
        return num_of_smaller_samples, num_of_smaller_positive_samples, num_of_larger_samples, num_of_larger_positive_samples

    def choose_feature(self):
        """
        function used to choose a feature for slicing according to all feature's IG in current node.
        :return: (feature name, feature's ig, separator value)
        :rtype: tuple
        """
        features = self.data.keys().tolist()
        features = features[:-1]

        best_ig = None, float("-inf"), None

        for feature in features:
            ig, separator = self.calculate_information_gain(feature)
            if ig >= best_ig[1]:
                best_ig = feature, ig, separator
        if best_ig[0] is None:
            raise ValueError("feature to separate not found!")
        return best_ig

    def check_homogenous_leaves(self, prune_thresh):
        """
        function checks whether a node is homogenous. i.e., it reached a state where all of it's data is either "M" or "B".
        :return: (True, diagnosis) or (False, None)
        :rtype: tuple
        """
        # method index returns the index range of the array.
        # check if it's an empty leaf and assign it with default classification.
        if len(self.data.index) == 0:
            return True, DEFAULT_CLASSIFICATION
        # check if leaf is homogenous.
        # Method unique checks if all the labels of the samples in the node has the same attribute.
        # unique() returns a list with all the different elements in y ("M"\"B"). if there's only 1, leaf is homogenous.
        # we choose candidate for pruning by #samples in current note is below given threshold
        # if it does, we use the definition of loss to decide whether to prune or not
        # we decide the diagnosis of the node by the majority labels, such that the relation between M and B
        # will be 8:1 which will potentially minimize the loss, since it acts like the Evaluation function
        # we have seen in the tutorial, but instead of evaluating the while pruned tree, we only evaluate the
        # subtree which is the candidate for pruning.
        if len(self.data["diagnosis"].unique()) == 1 or len(self.data["diagnosis"]) <= prune_thresh:
            num_of_sick = len([x for x in self.data["diagnosis"] if x == "M"])
            num_of_healthy = len([x for x in self.data["diagnosis"] if x == "B"])
            if 8 * num_of_sick < num_of_healthy:
                result = (True, "B")
            else:
                result = (True, "M")
        # not a leaf
        else:
            result = (False, None)
        return result


class ID3:
    def __init__(self, prune_thresh=-1):
        self.id3tree = None
        self.prune_thresh = prune_thresh

    def fit(self, x, y):
        # arrange in "data" variable, so we can pass it to ID3Tree() class to create the tree.
        data = pd.DataFrame(x.copy())
        data["diagnosis"] = pd.DataFrame(y)
        self.id3tree = ID3Tree(self.prune_thresh, data=data)

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

    def fit_predict(self, train, test):
        """
        Classifier to utilize ID3 tree.
        fitting the data into ID3 tree and predicts the diagnosis for data set x.
        computes accuracy for y.
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

    def tree_traversal(self, node, row, data):
        """
        recursively traverse the tree until leaf is reached.
        :param node: node with diagnosis, slice_thresh members and is_leaf method.
        :type node: ID3Node
        :param row: row number in dataframe
        :type row: int
        :param data: the dataset
        :type data: dataframe
        :return: diagnosis
        """
        if node.is_leaf():
            return node.diagnosis
        else:
            feature = node.feature
            value = data[feature].iloc[row]
            if float(value) <= node.slice_thresh:
                return self.tree_traversal(node.left, row, data)
            else:
                return self.tree_traversal(node.right, row, data)


def experiment(all_data, graph=False, ):
    """
    # TODO in order to see accuracy value, please uncomment in main part the first "TODO"
    graph: option to plot graph
    """
    x, y = get_data_from_df(all_data)
    x = x.to_numpy()
    y = y.to_numpy()
    m_values = [i for i in range(1, 51, 10)]  # TODO: check what happens when m_value = 0
    kf = KFold(n_splits=5, random_state=314985664, shuffle=True)
    avg_loss_list = []
    for m in m_values:
        losses = []
        k_classifier = ID3(prune_thresh=m)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train = np.concatenate((y_train, x_train), axis=1)
            test = np.concatenate((y_test, x_test), axis=1)
            predictions = k_classifier.fit_predict(train, test)
            accuracy, loss = k_classifier.calculate_loss_and_accuracy(x_test, y_test, predictions)
            losses.append(loss)
        avg_loss_list.append(sum(losses)/float(len(losses)))
    if graph:
        print(f"values of losses are {sorted(avg_loss_list)}")
        plt.plot(m_values, sorted(avg_loss_list))
        plt.xlabel("Value of M")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":
    classifier = ID3(prune_thresh=-1)

    # get numpy ndarray from csv
    train = genfromtxt('train.csv', delimiter=',', dtype="unicode")
    test = genfromtxt('test1.csv', delimiter=',', dtype="unicode")

    # we send only test dataset to experiment function
    data = pd.DataFrame(train)
    # TODO: to run the experiment and print the graph, pleas uncomment the following line
    # experiment(data, graph=True)
