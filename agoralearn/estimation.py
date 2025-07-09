"""Stats helpers module."""

# Authors: Hamza Cherkaoui

import numpy as np
import numba
from .stats import bias_square_norm, bias_square_B_norm, Id_like


@numba.jit(nopython=True, cache=True, fastmath=True)
def estimate_operator_norm(A: np.ndarray,
                           num_iter: int = 100,
                           tol: float = 1e-6,
                           ) -> float:
    """
    Estimate the operator (spectral) norm of matrix A using power iteration.

    Parameters
    ----------
    A : np.ndarray
        The input A.
    num_iter : int, optional
        Number of power iterations (default is 100).
    tol : float, optional
        Early stopping tolerance for convergence (default is 1e-6).

    Returns
    -------
    float
        Estimated operator norm (largest singular value).
    """
    n = A.shape[1]
    v = np.random.randn(n)
    norm_v = np.linalg.norm(v)

    if norm_v == 0.0:
        return 0.0
    v /= norm_v

    for _ in range(num_iter):

        Av = A.dot(v)
        v_next = A.T.dot(Av)

        v_next_norm = np.linalg.norm(v_next)
        if v_next_norm < tol:
            return 0.0

        v_next /= v_next_norm

        if np.linalg.norm(v - v_next) < tol:
            break

        v = v_next

    return np.linalg.norm(A.dot(v))


@numba.jit(nopython=True, cache=True, fastmath=True)
def estimate_ridge(X: np.ndarray,
                   y: np.ndarray,
                   lbda: float = -1.0,
                   ) -> np.ndarray:
    """Estimate regression weights using OLS or Ridge regression.

    Parameters
    ----------
        X (np.ndarray): Design matrix of shape (n_samples, d).
        y (np.ndarray): Target vector of shape (n_samples,).
        lbda (float, optional): Regularization strength. Defaults to 1e-6.

    Returns:
    --------
        np.ndarray: Estimated parameter vector of shape (d,).
    """
    if lbda < 0.0:
        lbda = 1e-6

    XtX = X.T.dot(X)
    Xty = X.T.dot(y)

    return np.linalg.solve(XtX + lbda * Id_like(XtX), Xty)


@numba.jit(nopython=True, cache=True, fastmath=True)
def estimate_james_stein_coef(theta_hat: np.ndarray,
                              sigma: float,
                              ) -> float:
    """Compute the James–Stein shrinkage coefficient.

    Parameters
    ----------
    theta_hat : np.ndarray
        The OLS estimator of shape (d,).
    sigma : float
        The standard deviation of the noise.

    Returns
    -------
    float
        The James–Stein shrinkage coefficient.
        If negative, returns 0 (positive-part James–Stein).
    """
    norm_sq = np.dot(theta_hat, theta_hat)

    if norm_sq == 0.0:
        return 0.0

    coef = 1.0 - ((len(theta_hat) - 2) * sigma ** 2) / norm_sq

    return coef if coef > 0.0 else 0.0


@numba.jit(nopython=True, cache=True, fastmath=True)
def estimate_sigma_squared_ridge(X: np.ndarray,
                                 y: np.ndarray,
                                 lbda: float = -1.0,
                                 ) -> float:
    """
    Estimate the noise variance sigma^2 in a linear model y = X theta + eta,
    using Ridge (or OLS) regression and an unbiased estimator from the residuals.

    Parameters
    ----------
    X (np.ndarray): Design matrix of shape (n_samples, d).
    y (np.ndarray): Target vector of shape (n_samples,).
    lbda (float): Regularization strength (if negative, defaults to small OLS regularization).

    Returns
    -------
    float: Estimated noise variance sigma^2.
    """
    n, d = X.shape
    theta_hat = estimate_ridge(X, y, lbda)
    residuals = y - X.dot(theta_hat)
    return np.dot(residuals, residuals) / (n - d)


@numba.jit(nopython=True, cache=True, fastmath=True)
def estimate_bias_square_norm(theta1_hat: np.ndarray,
                              theta2_hat: np.ndarray,
                              M: np.ndarray,
                              A1_inv: np.ndarray,
                              A2_inv: np.ndarray,
                              sigma: float,
                              ) -> float:
    """
    Estimate || M (theta2^* - theta1^*) ||_2^2 using an unbiased estimator.

    Parameters
    ----------
    theta1_hat, theta2_hat : np.ndarray
        Estimated theta vectors of shape (d,)
    M : np.ndarray
        Matrix M_t of shape (m, d)
    A1_inv, A2_inv : np.ndarray
        Inverse Gram matrices of shape (d, d)
    sigma : float
        Noise standard deviations

    Returns
    -------
    float
        Unbiased estimate of the squared bias term.
    """
    bias_norm_sq = bias_square_norm(M, theta2_hat, theta1_hat)
    correction = sigma**2 * np.trace(M.dot(A2_inv + A1_inv).dot(M.T))

    return bias_norm_sq - correction


@numba.jit(nopython=True, cache=True, fastmath=True)
def estimate_bias_square_B_norm(theta1_hat: np.ndarray,
                                theta2_hat: np.ndarray,
                                M: np.ndarray,
                                B: np.ndarray,
                                A1_inv: np.ndarray,
                                A2_inv: np.ndarray,
                                sigma: float,
                                ) -> float:
    """
    Estimate || M (theta2^* - theta1^*) ||_B^2 using an unbiased estimator.

    Parameters
    ----------
    theta1_hat, theta2_hat : np.ndarray
        Estimated theta vectors of shape (d,)
    M : np.ndarray
        Matrix M_t of shape (m, d)
    A1_inv, A2_inv : np.ndarray
        Inverse Gram matrices of shape (d, d)
    sigma : float
        Noise standard deviations

    Returns
    -------
    float
        Unbiased estimate of the squared bias term.
    """
    bias_B_norm_sq = bias_square_B_norm(M, B, theta2_hat, theta1_hat)
    correction = sigma**2 * np.trace(B.dot(M.dot(A1_inv + A2_inv).dot(M.T)))

    return bias_B_norm_sq - correction
