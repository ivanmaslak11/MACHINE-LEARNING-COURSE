import numpy as np
import scipy.sparse as sp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class LinearLoss(BaseLoss):
    """
    Loss function for linear regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        N = X.shape[0]

        y_pred = X @ w

        mse = np.sum((y_pred - y) ** 2) / N

        reg = self.l2_coef * np.sum(w[1:] ** 2)

        return mse + reg

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray
        Returns
        -------
        : 1d numpy.ndarray
        """
        N = X.shape[0]

        y_pred = X.dot(w)

        mse_grad = 2.0 * X.T.dot(y_pred - y) / N

        reg_grad = np.zeros_like(w)
        reg_grad[1:] = 2.0 * self.l2_coef * w[1:]

        return mse_grad + reg_grad
