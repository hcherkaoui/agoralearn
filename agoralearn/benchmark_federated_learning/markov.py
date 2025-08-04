"""Markov chain helpers module."""

# Authors: Hamza Cherkaoui

import numpy as np


def generate_transition_matrix(d, type="uniform", bias=None, stay_prob=0.9):
    """
    Generate a structured transition matrix for a Markov chain.

    Parameters
    ----------
        d (int): Number of states.
        type (str): Type of transition matrix. Options:
            - "uniform": uniform transitions
            - "sticky": high probability of staying in the same state
            - "cyclic": state i goes mostly to state i+1
            - "biased": one state has higher attraction
            - "block": clustered blocks of transitions
        bias (int): Target state for "biased" type.
        stay_prob (float): Probability of staying in same state (for "sticky").

    Returns
    -------
        np.ndarray: A (d, d) row-stochastic transition matrix.
    """
    P = np.zeros((d, d))

    if type == "uniform":
        P[:] = 1.0 / d

    elif type == "sticky":
        for i in range(d):
            P[i] = (1 - stay_prob) / (d - 1)
            P[i, i] = stay_prob

    elif type == "cyclic":
        for i in range(d):
            P[i, (i + 1) % d] = stay_prob
            P[i] += (1 - stay_prob) / d

    elif type == "biased":
        if bias is None:
            bias = 0
        for i in range(d):
            P[i] = (1 - stay_prob) / (d - 1)
            P[i, bias] = stay_prob

    elif type == "block":
        n_blocks = int(np.sqrt(d))
        block_size = d // n_blocks
        for i in range(d):
            P[i] = (1 - stay_prob) / (d - 1)
            block_id = i // block_size
            for j in range(block_id * block_size, min((block_id + 1) * block_size, d)):
                P[i, j] += stay_prob / block_size

    else:
        raise ValueError(f"Unknown type '{type}'")

    return P / P.sum(axis=1, keepdims=True)


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
