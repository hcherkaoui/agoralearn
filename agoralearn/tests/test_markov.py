"""Markov test module."""

# Authors: Hamza Cherkaoui

import pytest
import numpy as np
from agoralearn.markov import generate_transition_matrix, generate_markov_chain, simulate_hmm


@pytest.mark.parametrize("matrix_type", ["uniform", "sticky", "cyclic", "biased", "block"])
def test_transition_matrix_validity(matrix_type):
    d = 5
    bias = 2 if matrix_type == "biased" else None
    P = generate_transition_matrix(d, type=matrix_type, bias=bias)

    assert P.shape == (d, d)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(d), atol=1e-8)
    assert np.all((P >= 0) & (P <= 1))


def test_generate_markov_chain_shape_and_values():
    n, d = 20, 3
    num_states = 4
    P = generate_transition_matrix(num_states, type="sticky")
    states = np.random.randn(num_states, d)

    X = generate_markov_chain(n, d, P, states)

    assert X.shape == (n, d)

    for row in X:
        diffs = np.linalg.norm(states - row, axis=1)
        assert np.min(diffs) < 1e-8


def test_simulate_hmm_output_shape_and_noise():
    n, d = 30, 4
    sigma = 0.5
    theta_star = np.random.randn(d)
    num_states = 5
    P = generate_transition_matrix(num_states, type="biased", bias=1)
    states = np.random.randn(num_states, d)

    X, y = simulate_hmm(n, d, theta_star, P, states, sigma)

    assert X.shape == (n, d)
    assert y.shape == (n,)

    y_denoised = X @ theta_star
    residual = y - y_denoised
    assert np.std(residual) > 0.1


def test_invalid_transition_type():
    with pytest.raises(ValueError, match="Unknown type"):
        generate_transition_matrix(4, type="nonsense")
