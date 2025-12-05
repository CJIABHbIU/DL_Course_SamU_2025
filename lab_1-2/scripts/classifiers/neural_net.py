from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        hidden = X.dot(W1) + b1

        hidden_relu = np.maximum(0, hidden)

        scores = hidden_relu.dot(W2) + b2

        if y is None:
            return scores

        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[np.arange(N), y])
        data_loss = np.mean(correct_logprobs)
        reg_loss = reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        grads = {}

        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1.0
        dscores /= N                      # усреднение по батчу

        # градиенты по W2 и b2
        dW2 = hidden_relu.T.dot(dscores)
        db2 = np.sum(dscores, axis=0)

        dW2 += 2 * reg * W2

        # градиент по скрытому слою после ReLU
        dhidden = dscores.dot(W2.T)      # (N, C) @ (C, H) -> (N, H)
        # ReLU back: зануляем там, где до ReLU было ≤ 0
        dhidden[hidden <= 0] = 0

        # градиенты по W1 и b1
        dW1 = X.T.dot(dhidden)
        db1 = np.sum(dhidden, axis=0)
        
        dW1 += 2 * reg * W1

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads


    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # считаем loss и градиенты
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            for param_name in self.params:
                self.params[param_name] -= learning_rate * grads[param_name]

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # каждые epoch считаем accuracy и уменьшаем lr
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }


    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden = X.dot(W1) + b1
        hidden_relu = np.maximum(0, hidden)
        scores = hidden_relu.dot(W2) + b2

        y_pred = np.argmax(scores, axis=1)
        return y_pred