import numpy as np
import time
from losses import LinearLoss

class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio. CRITERIO: np.linalg.norm(current_w - previous_w) < tolerance
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.w = None

    def _learning_rate(self, k):
        return self.step_alpha / (k ** self.step_beta) if k > 0 else self.step_alpha

    def _get_batch_indices(self, n_samples, epoch):
        if self.batch_size is None:
            return [np.arange(n_samples)]

        indices = self.rng.permutation(n_samples)
        batch_indices = []

        for i in range(0, n_samples, self.batch_size):
            batch_indices.append(indices[i:i + self.batch_size])

        return batch_indices

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector for initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        n_samples, n_features = X.shape

        if w_0 is None:
            self.w = np.ones(n_features)
        else:
            self.w = w_0

        if trace:
            history = {
                'time': [],
                'func': [],
                'func_val': [] if X_val is not None else None
            }

        previous_w = self.w.copy()

        for epoch in range(1, self.max_iter + 1):
            start_time = time.time()

            lr = self._learning_rate(epoch)

            batch_indices_list = self._get_batch_indices(n_samples, epoch)

            for batch_indices in batch_indices_list:
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                gradient = self.loss_function.grad(X_batch, y_batch, self.w)

                self.w -= lr * gradient

            diff = np.linalg.norm(previous_w - self.w)
            if diff < self.tolerance:
                break

            previous_w = self.w.copy()

            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(self.get_objective(X, y))
                if X_val is not None and y_val is not None:
                    history['func_val'].append(self.get_objective(X_val, y_val))

        return history if trace else None

    def predict(self, X):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        return X @ self.w


    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d model weights vector.
        """
        return self.w.copy() if self.w is not None else None

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.w)
