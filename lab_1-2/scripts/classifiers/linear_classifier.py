from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from scripts.classifiers.linear_svm import *
from scripts.classifiers.softmax import *


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y in 0...K-1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]          # (batch_size, D)
            y_batch = y[batch_indices]          # (batch_size,)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # считаем loss и grad для текущего батча
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.W -= learning_rate * grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        """
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scores = X.dot(self.W)              # (N, C)
        y_pred = np.argmax(scores, axis=1)  # (N,)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Абстрактный метод — переопределяется в подклассах (SVM, Softmax).
        """
        raise NotImplementedError


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
