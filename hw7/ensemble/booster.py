from __future__ import annotations
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from .sampler import FeatureSampler
from scipy.optimize import minimize_scalar


class Booster:
    def __init__(self, base_estimator, feature_sampler, n_estimators=10, lr=.5, **params):
        """
        n_estimators : int
            number of base estimators
        base_estimator : class
            class for base_estimator with fit(), predict() and predict_proba() methods
        feature_sampler : instance of FeatureSampler
        n_estimators : int
            number of base_estimators
        lr : float
            learning rate for estimators
        params : kwargs
            kwargs for base_estimator init
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.feature_sampler = feature_sampler
        self.estimators = []
        self.indices = []
        self.weights = []
        self.lr = lr
        self.params = params

    def _fit_first_estimator(self, X, y) -> Booster:
        n_features = X.shape[1]

        feature_indices = self.feature_sampler.sample_indices(n_features)
        X_sampled = X[:, feature_indices]

        estimator = self.base_estimator(**self.params)
        estimator.fit(X_sampled, y)

        self.estimators.append(estimator)
        self.indices.append(feature_indices)
        self.weights.append(1.0)

        return self

    def _gradient(self, y_true, y_pred):
        raise NotImplementedError

    def _loss(self, y_true, y_pred):
        raise NotImplementedError

    def _fit_base_estimator(self, X, y, predictions):
        raise NotImplementedError

    def fit(self, X, y) -> Booster:
        """
        Calculate final predictions:
            1) fit first estimator
            2) fit next estimator based on previous predictions
            3) update predictions
            4) got to step 2
        Don't forget, that each estimator has its own feature indices for prediction
        """

        self.estimators = []
        self.indices = []
        self.weights = []

        self._fit_first_estimator(X, y)

        predictions = self.lr * self.weights[0] * self.estimators[0].predict(X[:, self.indices[0]])

        for i in range(1, self.n_estimators):
            self._fit_base_estimator(X, y, predictions)
            current_predictions = self.estimators[i].predict(X[:, self.indices[i]])
            weight = self.weights[i]
            
            predictions += current_predictions * self.lr * weight

        return self

    def predict(self, X) -> np.ndarray:
        """
        Returns
        -------
        predictions : numpy ndarrays of shape (n_objects, n_classes)

        Calculate final predictions:
            1) calculate first estimator predictions
            2) calculate updates from next estimator
            3) update predictions
            4) got to step 2
        Don't forget, that each estimator has its own feature indices for prediction
        """
        if not (0 < len(self.estimators) == len(self.indices) == len(self.weights)):
            raise RuntimeError('Booster is not fitted', (len(self.estimators), len(self.indices)))

        predictions = np.zeros(X.shape[0])
        
        for estimator, feature_indices, weight in zip(self.estimators, self.indices, self.weights):
            X_selected = X[:, feature_indices]
            predictions += self.lr * weight * estimator.predict(X_selected)
        
        return predictions


class GradientBoostingClassifier(Booster):
    def __init__(self, n_estimators=30, max_features_samples=0.8, lr=.5, max_depth=None, min_samples_leaf=1,
                 random_state=None, **params):
        base_estimator = DecisionTreeRegressor
        feature_sampler = FeatureSampler(max_samples=max_features_samples, random_state=random_state)

        super().__init__(
            base_estimator=base_estimator,
            feature_sampler=feature_sampler,
            n_estimators=n_estimators,
            lr=lr,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **params,
        )

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _gradient(self, y_true, y_pred) -> np.ndarray:
        """
        Calculate gradient for NLL
        """

        sigmoid = self._sigmoid(y_pred)

        return sigmoid - y_true

    def _loss(self, y_true, y_pred) -> np.ndarray:
        """
        Calculate average NLL
        """
        nll = np.log(1 + np.exp(y_pred)) - y_true * y_pred
        return np.mean(nll)

    def _fit_base_estimator(self, X, y, predictions) -> GradientBoostingClassifier:
        """
        Fits next estimator:
            1) calculate gradient
            2) select random indices of features for current estimator
            3) fit base_estimator (don't forget to remain only selected features)
            4) save base_estimator (self.estimators) and feature indices (self.indices)
            5) find optimal weight for estimator using one-dimensional optimization

        NOTE that self.base_estimator is class and you should init it with
        self.base_estimator(**self.params) before fitting
        
        For one-dimensional optimization you may use scipy.optimize.minimize_scalar
        """
        gradient = self._gradient(y, predictions)

        n_features = X.shape[1]
        feature_indices = self.feature_sampler.sample_indices(n_features)

        X_sampled = X[:, feature_indices]
        estimator = self.base_estimator(**self.params)
        estimator.fit(X_sampled, -gradient)

        def loss_func(weight):
            estimator_pred = estimator.predict(X_sampled)
            new_predictions = predictions + self.lr * weight * estimator_pred
            return self._loss(y, new_predictions)

        result = minimize_scalar(loss_func, bounds=(0, 10))
        optimal_weight = result.x

        self.estimators.append(estimator)
        self.indices.append(feature_indices)
        self.weights.append(optimal_weight)

        return self

    def predict_proba(self, X):
        return np.clip(super().predict(X), 0, 1)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
