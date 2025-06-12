"""Transition matrices module."""

# Authors: Hamza Cherkaoui

import numpy as np


def generate_transition_matrix(d, type="uniform", bias=None, stay_prob=0.9):
    """
    Generate a structured transition matrix for a Markov chain.

    Args:
        d (int): Number of states.
        type (str): Type of transition matrix. Options:
            - "uniform": uniform transitions
            - "sticky": high probability of staying in the same state
            - "cyclic": state i goes mostly to state i+1
            - "biased": one state has higher attraction
            - "block": clustered blocks of transitions
        bias (int): Target state for "biased" type.
        stay_prob (float): Probability of staying in same state (for "sticky").

    Returns:
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
        p = np.random.uniform(low=0.75, high=0.95)
        for i in range(d):
            P[i, (i + 1) % d] = p
            P[i] += (1 - p) / d

    elif type == "biased":
        if bias is None:
            bias = 0
        for i in range(d):
            P[i] = (1 - 0.8) / (d - 1)
            P[i, bias] = 0.8

    elif type == "block":
        n_blocks = int(np.sqrt(d))
        block_size = d // n_blocks
        for i in range(d):
            P[i] = (1 - 0.9) / (d - 1)
            block_id = i // block_size
            for j in range(block_id * block_size, min((block_id + 1) * block_size, d)):
                P[i, j] += 0.9 / block_size

    else:
        raise ValueError(f"Unknown type '{type}'")

    return P / P.sum(axis=1, keepdims=True)
