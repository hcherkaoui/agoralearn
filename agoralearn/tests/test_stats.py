"""Stats test module."""

# Authors: Hamza Cherkaoui

import pytest
import numpy as np
from agoralearn.stats import XXt, XtX, Id_like, bias_square_norm, bias_square_B_norm, _B, _psi


@pytest.fixture
def random_setup():
    np.random.seed(0)
    X = np.random.randn(5, 3)
    theta1 = np.random.randn(3)
    theta2 = np.random.randn(3)
    M = np.random.randn(3, 3)
    B = np.eye(3)
    sigma = 2.0
    A1_inv = np.linalg.inv(np.random.randn(5, 5) @ np.random.randn(5, 5).T + np.eye(5))
    A2_inv = np.linalg.inv(np.random.randn(5, 5) @ np.random.randn(5, 5).T + np.eye(5))
    M_large = np.random.rand(5, 5)
    return X, theta1, theta2, M, B, sigma, A1_inv, A2_inv, M_large

def test_XXt(random_setup):
    X, *_ = random_setup
    expected = X @ X.T
    result = XXt(X)
    np.testing.assert_allclose(result, expected)

def test_XtX(random_setup):
    X, *_ = random_setup
    expected = X.T @ X
    result = XtX(X)
    np.testing.assert_allclose(result, expected)

def test_Id_like():
    A = np.random.randn(7, 3)
    result = Id_like(A)
    expected = np.eye(7)
    np.testing.assert_allclose(result, expected)

def test_bias_square_norm(random_setup):
    _, theta1, theta2, M, *_ = random_setup
    expected = ((M.T @ (theta2 - theta1))**2).sum()
    result = bias_square_norm(M, theta2, theta1)
    assert np.isclose(result, expected)

def test_bias_square_B_norm(random_setup):
    _, theta1, theta2, M, B, *_ = random_setup
    diff = M.T @ (theta2 - theta1)
    expected = diff.T @ B @ diff
    result = bias_square_B_norm(M, B, theta2, theta1)
    assert np.isclose(result, expected)

def test_B(random_setup):
    _, _, _, _, _, _, A1_inv, A2_inv, M = random_setup
    B = _B(A1_inv, A2_inv, M)
    assert B.shape == (5, 5)
    assert np.all(np.isfinite(B))

def test_psi(random_setup):
    _, _, _, _, _, sigma, *_ = random_setup
    B1 = np.random.randn(5, 5)
    B2 = np.random.randn(5, 5)
    expected = sigma**4 * (
        2 * np.trace(B1) * np.trace(B2)
        + 2 * (np.trace(B1 @ B1) + np.trace(B2 @ B2))
        + 8 * np.trace(B1 @ B2)
    )
    result = _psi(B1, B2, sigma)
    assert np.isclose(result, expected)
