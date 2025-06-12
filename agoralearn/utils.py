"""Utilities module."""

# Authors: Hamza Cherkaoui

import numpy as np


def estimate_operator_norm(matrix, num_iter=100, tol=1e-6):
    """
    Estimate the operator (spectral) norm of a matrix using power iteration.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix.
    num_iter : int, optional
        Number of power iterations (default is 100).
    tol : float, optional
        Early stopping tolerance for convergence (default is 1e-6).

    Returns
    -------
    float
        Estimated operator norm (largest singular value).
    """
    n = matrix.shape[1]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    for _ in range(num_iter):
        v_next = matrix.T @ (matrix @ v)
        v_next_norm = np.linalg.norm(v_next)

        if v_next_norm < tol:
            return 0.0

        v_next /= v_next_norm

        if np.linalg.norm(v - v_next) < tol:
            break

        v = v_next

    return np.linalg.norm(matrix @ v)


def generate_markov_chain(n, d, P, states):
    """Generate a sequence of state vectors from a Markov chain.

    Parameters
    ----------
        n (int): Number of time steps.
        d (int): Dimension of each state.
        P (np.ndarray): Transition probability matrix of shape (n_states, n_states).
        states (list or np.ndarray): List or array of state vectors of shape (n_states, d).

    Returns
    -------
        np.ndarray: Matrix of shape (n, d) representing the state sequence.
    """
    X = np.zeros((n, d))
    current_state = np.random.choice(len(states))

    for t in range(n):
        X[t] = states[current_state]
        current_state = np.random.choice(len(states), p=P[current_state])

    return X


def simulate_hmm(n, d, theta_star, P, states, sigma):
    """Simulate a linear Hidden Markov Model.

    Parameters
    ----------
        n (int): Number of time steps.
        d (int): Dimension of each state.
        theta_star (np.ndarray): Ground truth parameter vector of shape (d,).
        P (np.ndarray): Transition matrix.
        states (list or np.ndarray): State vector definitions.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns
    -------
        tuple: (X, y) where X is (n, d) and y is (n,)
    """
    X = generate_markov_chain(n, d, P, states)
    noise = np.random.randn(n) * sigma
    y = X @ theta_star + noise

    return X, y
