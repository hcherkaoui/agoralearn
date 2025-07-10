"""Estimation test module."""

# Authors: Hamza Cherkaoui

import numpy as np
import pytest
from numpy.testing import assert_allclose
from agoralearn.estimation import (estimate_operator_norm, estimate_ridge, estimate_james_stein_coef,
                                   estimate_sigma_ridge, estimate_bias_square_norm, estimate_bias_square_B_norm)
from agoralearn.stats import bias_square_norm, bias_square_B_norm


@pytest.fixture
def data():
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d)
    y = X @ np.random.randn(d) + np.random.randn(n) * 0.5
    theta1 = np.random.randn(d)
    theta2 = np.random.randn(d)
    M = np.random.randn(d, d)
    B = np.eye(d)
    A1_inv = np.linalg.inv(X.T @ X + 0.1 * np.eye(d))
    A2_inv = np.linalg.inv(X.T @ X + 0.2 * np.eye(d))
    sigma = 1.0
    return X, y, theta1, theta2, M, B, A1_inv, A2_inv, sigma


def test_estimate_operator_norm(data):
    X, *_ = data
    A = X.T @ X
    op_norm = estimate_operator_norm(A)
    expected = np.linalg.norm(A, 2)
    assert np.isclose(op_norm, expected, rtol=1e-2)


def test_estimate_ridge_vs_closed_form(data):
    X, y, *_ = data
    lbda = 0.1
    theta_hat = estimate_ridge(X, y, lbda)
    expected = np.linalg.solve(X.T @ X + lbda * np.eye(X.shape[1]), X.T @ y)
    assert_allclose(theta_hat, expected, rtol=1e-4)


def test_estimate_james_stein_coef_positive(data):
    _, _, theta1, _, _, _, _, _, sigma = data
    coef = estimate_james_stein_coef(theta1, sigma)
    assert coef >= 0.0
    assert coef <= 1.0


def test_estimate_sigma_ridge(data):
    X, y, *_ = data
    sigma_hat = estimate_sigma_ridge(X, y)
    assert sigma_hat >= 0.0


def test_estimate_bias_square_norm_unbiased(data):
    _, _, theta1, theta2, M, _, A1_inv, A2_inv, sigma = data
    est = estimate_bias_square_norm(theta1, theta2, M, A1_inv, A2_inv, sigma)
    naive = bias_square_norm(M, theta2, theta1)
    assert est <= naive


def test_estimate_bias_square_B_norm_unbiased(data):
    _, _, theta1, theta2, M, B, A1_inv, A2_inv, sigma = data
    est = estimate_bias_square_B_norm(theta1, theta2, M, B, A1_inv, A2_inv, sigma)
    naive = bias_square_B_norm(M, B, theta2, theta1)
    assert est <= naive
