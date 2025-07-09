"""Criterions module."""

# Authors: Hamza Cherkaoui

import numpy as np
import numba
from .stats import XtX, bias_square_norm, bias_square_B_norm, _B, _psi
from .estimation import estimate_bias_square_norm, estimate_bias_square_B_norm


@numba.jit(nopython=True, cache=True, fastmath=True)
def criterion_expect(
    theta1: np.ndarray,
    theta2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: float = 1.0,
    use_estimate: bool = False,
) -> float:
    """A collaboration test to determine whether combining estimators yields a statistically safer prediction than using a single agent alone.

    Parameters
    ----------
    theta1 : np.ndarray
        Unknown/Estimate parameter for agent 1
    theta2 : np.ndarray
        Unknown/Estimate parameter for agent 2
    X1 : np.ndarray
        Design matrix for agent 1, shape (n1, d)
    X2 : np.ndarray
        Design matrix for agent 2, shape (n2, d)
    sigma : float
        Known variance of the Gaussian noise (same for both agents)
    use_estimate : bool:
        Boolean to flag if estimate of thetas are used.

    Returns
    -------
    Delta : float
        The value of the benefit-of-collaboration test statistic.
        Delta < 0 implies collaboration is likely beneficial.
    """
    A1 = XtX(X1)
    A2 = XtX(X2)
    Ac = A1 + A2

    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)
    Ac_inv = np.linalg.inv(Ac)
    M = Ac_inv.dot(A2)

    if use_estimate:
        bias_norm_sq = estimate_bias_square_norm(theta1, theta2, M, A1_inv, A2_inv, sigma)
    else:
        bias_norm_sq = bias_square_norm(M, theta2, theta1)

    variance_term = sigma**2 * np.trace(Ac_inv - A1_inv)

    return float(bias_norm_sq + variance_term)


@numba.jit(nopython=True, cache=True, fastmath=True)
def criterion_inst(
    theta1: np.ndarray,
    theta2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: float = 1.0,
    delta: float = 0.05,
    use_estimate: bool = False,
) -> float:
    """A collaboration test to determine whether combining estimators yields a statistically safer prediction than using a single agent alone.

    Parameters
    ----------
    theta1 : np.ndarray
        Unknown/Estimate parameter for agent 1
    theta2 : np.ndarray
        Unknown/Estimate parameter for agent 2
    X1 : np.ndarray
        Design matrix for agent 1, shape (n1, d)
    X2 : np.ndarray
        Design matrix for agent 2, shape (n2, d)
    sigma : float
        Known variance of the Gaussian noise (same for both agents)
    delta : float, optional
        Probabilistic confidence level parameter of the collaboration test.
    use_estimate : bool, optional
        Boolean to flag if estimate of thetas are used.

    Returns
    -------
    Delta : float
        The value of the benefit-of-collaboration test statistic.
        Delta < 0 implies collaboration is likely beneficial.
    """
    A1 = XtX(X1)
    A2 = XtX(X2)
    Ac = A1 + A2

    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)
    Ac_inv = np.linalg.inv(Ac)

    M = np.dot(Ac_inv, A2)
    B = _B(A1_inv, A2_inv, M)

    if use_estimate:
        bias_norm_sq = estimate_bias_square_norm(theta1, theta2, M, A1_inv, A2_inv, sigma)
        bias_B_norm_sq = estimate_bias_square_B_norm(theta1, theta2, M, B, A1_inv, A2_inv, sigma)
    else:
        bias_norm_sq = bias_square_norm(M, theta2, theta1)
        bias_B_norm_sq = bias_square_B_norm(M, B, theta2, theta1)

    tr_A1_inv = np.trace(A1_inv)
    tr_A1_inv2 = np.trace(np.dot(A1_inv, A1_inv))
    tr_B = np.trace(B)
    tr_B2 = np.trace(np.dot(B, B))

    eig_A1_min = np.min(np.linalg.eigvalsh(A1))
    eig_B_max = np.max(np.linalg.eigvalsh(B))

    lower_bound_single = sigma**2 * (tr_A1_inv
                                     - 2 * np.sqrt(np.log(1 / delta) * tr_A1_inv2)
                                     - 2 * np.log(1 / delta) / eig_A1_min)

    cross_term = 2 * sigma * np.sqrt(np.log(1 / delta) * bias_B_norm_sq)
    variance_term = sigma**2 * (tr_B
                                + 2 * np.sqrt(np.log(1 / delta) * tr_B2)
                                + 2 * np.log(1 / delta) * eig_B_max)
    upper_bound_collab = bias_norm_sq + cross_term + variance_term

    return float(upper_bound_collab - lower_bound_single)


@numba.jit(nopython=True, cache=True, fastmath=True)
def criterion_heuri(
    theta1: np.ndarray,
    theta2: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: float = 1.0,
    beta: float = 1.0,
    use_estimate: bool = False,
) -> float:
    """
    A collaboration test to determine whether combining estimators yields a statistically safer prediction than using a single agent alone.

    Parameters
    ----------
    theta1 : np.ndarray
        Unknown/Estimate parameter for agent 1
    theta2 : np.ndarray
        Unknown/Estimate parameter for agent 2
    X1 : np.ndarray
        Design matrix for agent 1, shape (n1, d)
    X2 : np.ndarray
        Design matrix for agent 2, shape (n2, d)
    sigma : float, optional
        Known variance of the Gaussian noise (same for both agents)
    beta : float, optional
        Heuristic confidence level parameter of the collaboration test
    use_estimate : bool, optional
        Boolean to flag if estimate of thetas are used.

    Returns
    -------
    float
        The value of the benefit-of-collaboration test statistic.
        Delta < 0 implies collaboration is likely beneficial.
    """
    A1 = XtX(X1)
    A2 = XtX(X2)
    Ac = A1 + A2

    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)
    Ac_inv = np.linalg.inv(Ac)

    M = Ac_inv @ A2

    if use_estimate:
        bias_norm_sq = estimate_bias_square_norm(theta1, theta2, M, A1_inv, A2_inv, sigma)
    else:
        bias_norm_sq = bias_square_norm(M, theta2, theta1)

    B = _B(A1_inv, A2_inv, M)

    e_single = sigma**2 * np.trace(A1_inv)
    std_single = np.sqrt(2 * sigma**4 * np.trace(A1_inv @ A1_inv))

    e_collab = bias_norm_sq + sigma**2 * np.trace(Ac_inv)
    std_collab = np.sqrt(2 * sigma**4 * np.trace(B @ B))

    return float(e_collab + beta * std_collab - (e_single - beta * std_single))


@numba.jit(nopython=True, cache=True, fastmath=True)
def criterion_expect_whp(
    X1: np.ndarray,
    X2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    sigma: float = 1.0,
    L: float = 1.0,
    delta: float = 0.05,
) -> float:
    """A collaboration test to determine whether combining estimators yields a statistically safer prediction than using a single agent alone.

    Parameters
    ----------
    X1 : np.ndarray
        Design matrix for agent 1, shape (n1, d)
    X2 : np.ndarray
        Design matrix for agent 2, shape (n2, d)
    y1 : np.ndarray
        Response vector for agent 1, shape (n1,)
    y2 : np.ndarray
        Response vector for agent 2, shape (n2,)
    sigma : float
        Known variance of the Gaussian noise (same for both agents)
    L : float
        Known upper bound of the L2-norm of the unknown parameter
    delta : float:
        Probabilistic confidence level parameter of the collaboration test.

    Returns
    -------
    Delta : float
        The value of the benefit-of-collaboration test statistic.
        Delta < 0 implies collaboration is likely beneficial.
    """
    A1 = XtX(X1)
    A2 = XtX(X2)
    Ac = A1 + A2

    A1_inv = np.linalg.inv(A1)
    A2_inv = np.linalg.inv(A2)
    Ac_inv = np.linalg.inv(Ac)

    theta1_hat = np.linalg.solve(A1, X1.T @ y1)
    theta2_hat = np.linalg.solve(A2, X2.T @ y2)

    M = Ac_inv @ A2
    B1 = M @ A1_inv @ M.T
    B2 = M @ A2_inv @ M.T
    B = M @ (A1_inv + A2_inv) @ M.T
    v = M @ (theta2_hat - theta1_hat)

    lambda_min_A1 = np.min(np.linalg.eigvalsh(A1))
    lambda_min_A2 = np.min(np.linalg.eigvalsh(A2))
    lambda_max_A2 = np.linalg.norm(A2, ord=2)

    sigma_ub_t = np.sqrt(
        _psi(B1, B2, sigma)
        + 8
        * L**2
        * (lambda_max_A2 / (lambda_min_A1 + lambda_min_A2)) ** 4
        * (sigma**2 / lambda_min_A1 + sigma**2 / lambda_min_A2 - 2 * L**2)
    )

    lhs = sigma**2 * (np.trace(A1_inv) + np.trace(B))
    rhs = np.dot(v, v) + sigma**2 * np.trace(Ac_inv) + sigma_ub_t / np.sqrt(delta)

    return float(lhs - rhs)
