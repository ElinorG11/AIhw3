# The file was empty. I've created signatures of the required classes\methods from the
# pdf for convenience.
class Tree:
    """
    Tree class creates the ID3 tree
    """
    def __init__(self,data=None):
        """
        Init function generates the tree.
        :param data: ?
        :type data: ?
        """
        self.feature = None
        self.slicing_val = None
        self.left = None
        self.right = None
        self.data = data
        self.diagnosis = None

        # check if we reached a leaf and whether it is homogenous
        leaf, diagnosis = self.check_homogenous_leaves()

    def check_homogenous_leaves(self):
        data = []
        data.index


class ID3:
    def __init__(self):
        raise NotImplementedError

    def fit_predict(self, train, test):
        """
        A classifire the utilizes the ID3 tree to fit and predict.
        fitting the data into ID3 tree and predicts the diagnosis for data set x.
        computes accuracy and loss for y.
        :param train: dataset
        :param test:
        :return:
        """
        raise NotImplementedError

    def experiment(self):
        raise NotImplementedError