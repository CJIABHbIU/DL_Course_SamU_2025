from builtins import range
from builtins import object
import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                diff = X[i] - self.X_train[j]          # (D,)
                dists[i, j] = np.sqrt(np.sum(diff * diff))
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # разница между i-м тестовым и всеми train
            diff = self.X_train - X[i]                # (num_train, D)
            dists[i, :] = np.sqrt(np.sum(diff * diff, axis=1))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (num_test, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1)          # (num_train,)
        cross = X.dot(self.X_train.T)                         # (num_test, num_train)

        dists_sq = X_sq + train_sq - 2.0 * cross              # (num_test, num_train)
        dists_sq = np.maximum(dists_sq, 0.0)                  # численная стабильность
        dists = np.sqrt(dists_sq)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test,  dtype=int)
        for i in range(num_test):
            closest_y = []
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 1) индексы k ближайших соседей
            nearest = np.argsort(dists[i])[:k]
            closest_y = self.y_train[nearest]

            # 2) самый частый класс, при равенстве — меньший индекс
            counts = np.bincount(closest_y.astype(int))
            y_pred[i] = np.argmax(counts)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred
