"""Predictors module."""

# Author: Hamza Cherkaoui

import numpy as np


class OracleForecaster:
    def __init__(self, states, P, lbda=1e-6):
        """
        Oracle forecaster based on discrete Markov transitions and ridge regression.

        Parameters
        ----------
        states : list of np.ndarray
            List of discrete state vectors.
        P : np.ndarray
            State transition probability matrix.
        lbda : float, default=1e-6
            Ridge regularization parameter.
        """
        self.states = states
        self.P = P

        self.lbda = lbda
        self.reg = None

        self.X_local = None
        self.y_local = None

        self.X = None
        self.y = None

        self.XtX = None
        self.theta = None

        self.shared_sample = None

    def _check_fit(self):
        """Raise an error if the model has not been fitted."""
        if self.theta is None:
            raise RuntimeError("Model must be fitted before prediction."
                               "Call 'batch_fit' first.")

    def _get_state_index(self, x):
        """Return the index of the given state vector `x` in the known state list."""
        return next((i for i, v in enumerate(self.states) if np.array_equal(v, x)), -1)

    def _h(self, x):
        """Predict the next state vector based on the transition matrix."""
        idx = self._get_state_index(x)
        next_x_idx = np.argmax(self.P[idx])
        return self.states[next_x_idx]

    def _concatenate_samples(self, l_X_y):
        """Concatenate in the samples."""
        X_, y_ = [self.X], [self.y]

        for X_i, y_i in l_X_y:
            X_i = np.atleast_2d(X_i)
            y_i = np.atleast_1d(y_i)
            X_.append(X_i)
            y_.append(y_i)

        X = np.concatenate(X_, axis=0)
        y = np.concatenate(y_, axis=0)

        return X, y

    def _fit_theta(self):
        """Internally estimate the regression parameter."""
        self.XtX = self.X.T @ self.X
        self.theta = np.linalg.solve(self.XtX + self.reg, self.X.T @ self.y)

    def _ref_fit(self, x, y, l_x_y=None):
        """Re-fit the model with the given additional samples."""
        self._check_fit()

        self.X_local = np.vstack([self.X_local, x[None, :]])
        self.y_local = np.append(self.y_local, y)

        self.X = np.vstack([self.X, x[None, :]])
        self.y = np.append(self.y, y)

        if l_x_y is not None:
            self.X, self.y = self._concatenate_samples(l_x_y)

        self._fit_theta()

    def _predict(self, x):
        """Make a prediction for the given state x."""
        return float(self.theta.T @ self._h(x))

    def fetch_X_y(self):
        """
        Return the gathered samples (X, y).

        Parameters
        ----------
        samples : tuple of np.ndarray
            Gathered samples (X, y).
        """
        return self.X_local, self.y_local

    def fetch_x_y(self):
        """
        Return the last observed samples (x, y).

        Parameters
        ----------
        samples : tuple of np.ndarray
            Gathered samples (X, y).
        """
        return self.X_local[-1], self.y_local[-1]

    def batch_fit(self, X, y, l_X_y=None):
        """
        Fit the model using batch ridge regression.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, d)
            Design matrix.
        y : np.ndarray of shape (n_samples,)
            Target vector.
        l_X_y :  list of tuple
            Additional samples for colalboration [(X_1, y_1), ..., (X_n, y_n)]

        Returns
        -------
        self : OracleForecaster
        """
        self.reg = self.lbda * np.eye(X.shape[1])

        self.X_local = X
        self.y_local = y

        self.X = X
        self.y = y

        if l_X_y is not None:
            self.X, self.y = self._concatenate_samples(l_X_y)

        self.XtX = self.X.T @ self.X
        self.theta = np.linalg.solve(self.XtX + self.reg, self.X.T @ self.y)

        return self

    def online_fit_predict(self, x, y, l_x_y=None):
        """
        Predict and update the model with a new observation.

        Parameters
        ----------
        x : np.ndarray of shape (d,)
            Current state vector.
        y : float
            Target value to update the model.
        l_x_y :  list of tuple
            Additional samples for colalboration [(x_1, y_1), ..., (x_n, y_n)]

        Returns
        -------
        float
            Predicted value for the next state.
        """
        self._ref_fit(x, y, l_x_y=l_x_y)
        return self._predict(x)

    def online_predict(self, x):
        """
        Predict the next value given the current state (without updating).

        Parameters
        ----------
        x : np.ndarray of shape (d,)
            Current state vector.

        Returns
        -------
        float
            Predicted value for the next state.
        """
        self._check_fit()
        return self._predict(x)


class LearningForecaster(OracleForecaster):
    def __init__(self, states, lbda=1e-6):
        """
        Forecaster that learns the transition matrix from data.

        Parameters
        ----------
        states : list of np.ndarray
            List of discrete state vectors.
        lbda : float
            Ridge regularization parameter.
        """
        super().__init__(states, P=None, lbda=lbda)

    def _estimate_transition_matrix(self, X):
        """Estimate the transition matrix."""
        n_states = len(self.states)
        counts = np.zeros((n_states, n_states))

        for i in range(len(X) - 1):
            idx_from = self._get_state_index(X[i])
            idx_to = self._get_state_index(X[i + 1])
            if idx_from >= 0 and idx_to >= 0:
                counts[idx_from, idx_to] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            P = counts / counts.sum(axis=1, keepdims=True)
            P[np.isnan(P)] = 0.0

        return P

    def batch_fit(self, X, y, l_X_y=None):
        """Fit the model using batch ridge regression."""
        self.P = self._estimate_transition_matrix(X)
        return super().batch_fit(X, y, l_X_y=l_X_y)

    def online_fit_predict(self, x, y, l_x_y=None):
        """Predict and update the model with a new observation."""
        self._ref_fit(x, y, l_x_y=l_x_y)
        self.P = self._estimate_transition_matrix(self.X)
        return self._predict(x)
