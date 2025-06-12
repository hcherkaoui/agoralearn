"""Criterions module."""

# Authors: Hamza Cherkaoui

import numpy as np



def criterion_1(X_train_1,
                X_train_2,
                theta_star_1,
                theta_star_2,
                lbda,
                sigma):
    """Evaluate whether collaboration is beneficial using a trace-based criterion.

    Args:
        X_train_1 (np.ndarray): Task 1 design matrix of shape (n_1, d).
        X_train_2 (np.ndarray): Task 2 design matrix of shape (n_2, d).
        theta_star_1 (np.ndarray): Ground truth parameter for task 1 (shape d,).
        theta_star_2 (np.ndarray): Ground truth parameter for task 2 (shape d,).
        lbda (float): Ridge regularization strength.
        sigma (float): Standard deviation of the noise.

    Returns:
        int: 1 if collaboration satisfies the criterion, else 0.
    """
    reg = lbda * np.eye(len(theta_star_1))

    A_1 = X_train_1.T @ X_train_1 + reg
    A_1_inv = np.linalg.inv(A_1)

    A_2 = X_train_2.T @ X_train_2 + reg

    X_train_c = np.r_[X_train_1, X_train_2]
    A_c = X_train_c.T @ X_train_c + reg
    A_c_inv = np.linalg.inv(A_c)

    Delta = theta_star_2 - theta_star_1
    M = A_c_inv @ A_2

    threshold = Delta.T @ M.T @ M @ Delta
    criterion_value = sigma**2 * np.trace(A_1_inv - A_c_inv)

    return criterion_value >= threshold


def criterion_2(l_X_y, sigma=1.0, L=1.0):
    """Evaluate whether collaboration is beneficial using a trace-based criterion.

    Args:
        l_X_y (list of tuple): Task 1 design matrix of shape (n_1, d).
        L (float): Standard deviation of the noise.

    Returns:
        int: 1 if collaboration satisfies the criterion, else 0.
    """
    msg = ("This criterion is only valid if all the observation matrices are"
           "the same, the ridge regularization is nill and the regression "
           "parameter are bounded by L")
    X_1 = l_X_y[0][0]

    for X, _ in l_X_y:
        assert np.allclose(X_1, X), msg

    n = len(l_X_y)
    A_1_inv = np.linalg.inv(X_1.T @ X_1)

    threshold = ((n + 1) * L**2) / (n * sigma**2)
    criterion_value = np.trace(A_1_inv)

    return criterion_value >= threshold
