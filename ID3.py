# The file was empty. I've created signatures of the required classes\methods from the
# pdf for convenience.

import pandas as pd
from numpy import log2
from numpy import genfromtxt

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

    def __init__(self, data=None):
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

        # check if we reached a leaf and whether it is homogenous
        leaf, diagnosis = self.check_homogenous_leaves()
        if leaf:
            self.diagnosis = diagnosis
        else:
            # this is not a leaf so we update feature for slicing and slicing val
            self.feature, _, self.slice_thresh = self.choose_feature()
            # slice the dataframe
            data_left = self.data[pd.to_numeric(self.data[self.feature]) <= self.slice_thresh]
            data_right = self.data[pd.to_numeric(self.data[self.feature]) > self.slice_thresh]
            # recursively create more nodes
            self.left = ID3Tree(data=data_left)
            self.right = ID3Tree(data=data_right)

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
            root_fraction = (num_of_larger_positive_samples + num_of_smaller_positive_samples) / len(values)
            entropy_root = -root_fraction * log(root_fraction) - ((1 - root_fraction) * log(1 - root_fraction))
            # calculate the left son's IG
            fraction = num_of_smaller_positive_samples / num_of_smaller_samples if num_of_smaller_samples != 0 else root_fraction
            entropy_left = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))
            # calculate the right son's IG
            fraction = num_of_larger_positive_samples / num_of_larger_samples if num_of_larger_samples != 0 else root_fraction
            entropy_right = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))

            ig = entropy_root - entropy_left * num_of_smaller_samples / len(
                values) - entropy_right * num_of_larger_samples / len(values)
            if ig >= best_ig[0]:
                best_ig = ig, separator

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
        num_of_smaller_samples, num_of_smaller_positive_samples, num_of_larger_samples, num_of_larger_positive_samples = 0, 0, 0, 0
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

    def check_homogenous_leaves(self):
        """
        function checks whether a node is homogenous. i.e., it reached a state where all of it's data is either "M" or "B".
        :return: (True, diagnosis) or (False, None)
        :rtype: tuple
        """
        # method index returns the index range of the array.
        # check if it's an empty leaf and assign it with default classification.
        if len(self.data.index) == 0:
            return True, DEFAULT_CLASSIFICATION
        # check if leaf is homogenous. Method unique checks if all the labels of the samples in the node has the same attribute.
        # unique() returns a list with all the different elements in y ("M"\"B"). if there's only 1, leaf is homogenous.
        if len(self.data.diagnosis.unique()) == 1:
            result = (True, self.data["diagnosis"].iloc[0])
        # not a leaf
        else:
            result = (False, None)
        return result


class ID3:
    def __init__(self):
        self.id3tree = None

    def fit(self, x, y):
        # arrange in "data" variable, so we can pass it to ID3Tree() class to create the tree.
        data = x.copy()
        data["diagnosis"] = y
        self.id3tree = ID3Tree(data=data)

    def predict(self, x, y):
        data = x.copy()
        data["diagnosis"] = y
        right_predictions = 0
        false_negative = 0
        false_positive = 0
        num_of_samples = len(data.index)
        for row in range(len(data.index)):
            prediction = self.tree_traversal(self.id3tree, row, data)
            if prediction == data["diagnosis"].iloc[row]:
                right_predictions += 1
            else:
                if prediction == "M":
                    false_positive += 1
                else:
                    false_negative += 1
        acc = right_predictions / num_of_samples
        loss = (0.1 * false_positive + false_negative) / num_of_samples
        return acc, loss

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
        accuracy, loss = self.predict(test_x, test_y)

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

    def experiment(self):
        raise NotImplementedError


if __name__ == "__main__":
    classifier = ID3()

    # get numpy ndarray from csv
    train = genfromtxt('train.csv', delimiter=',', dtype="unicode")
    test = genfromtxt('train.csv', delimiter=',', dtype="unicode")

    # do we need the loss here? example in the tutorial & pdf instructions are a bit different
    res_accuracy, res_loss = classifier.fit_predict(train, test)
    print(res_accuracy)
