import numpy as np

class Adaline:
    """ADAptive LInear NEuron classifier.
       Gradient Descent

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Error in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        self.errors_ = []

        for n in range(self.n_iter):
            train = y - self.net_input(X)
            update = self.eta * train.dot(X)
            print("---------" + str(n) + "-----------")
            print(self.w_)
            self.w_[1:] += update
            self.w_[0] += self.eta * train.sum()

            cost_ = np.power(train, 2)
            errors = np.sum(cost_)/(2 * len(X))
            print(errors)
            self.errors_.append(errors)
        return self

        # TODO: Put your code here!


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
