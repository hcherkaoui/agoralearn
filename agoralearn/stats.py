"""Stats helpers module."""

# Authors: Hamza Cherkaoui

import numpy as np
import numba


@numba.jit(nopython=True, cache=True, fastmath=True)
def XXt(X: np.ndarray) -> np.ndarray:
    """Compute X X.T."""
    return X.dot(X.T)


@numba.jit(nopython=True, cache=True, fastmath=True)
def XtX(X: np.ndarray) -> np.ndarray:
    """Compute X.T X."""
    return X.T.dot(X)


@numba.jit(nopython=True, cache=True, fastmath=True)
def Id_like(A: np.ndarray) -> np.ndarray:
    """Return the Identity."""
    return np.eye(A.shape[0])


@numba.jit(nopython=True, cache=True, fastmath=True)
def bias_square_norm(M: np.ndarray,
                     theta2_star: np.ndarray,
                     theta1_star: np.ndarray,
                     ) -> float:
    """Compute the bias term L2-norm."""
    theta_diff = theta2_star - theta1_star
    Mt_theta_diff = M.T @ theta_diff

    return Mt_theta_diff.T.dot(Mt_theta_diff)


@numba.jit(nopython=True, cache=True, fastmath=True)
def bias_square_B_norm(M: np.ndarray,
                       B: np.ndarray,
                       theta2_star: np.ndarray,
                       theta1_star: np.ndarray,
                       ) -> float:
    """Compute the bias term B-norm."""
    theta_diff = theta2_star - theta1_star
    Mt_theta_diff = M.T @ theta_diff

    return Mt_theta_diff.T.dot(B).dot(Mt_theta_diff)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _B(A1_inv: np.ndarray,
       A2_inv: np.ndarray,
       M: np.ndarray,
       ) -> np.ndarray:
    """Compute the B matrix."""
    Id = Id_like(M)
    return M.dot(A2_inv).dot(M.T) + (Id - M).dot(A1_inv).dot((Id - M).T)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _psi(B1: np.ndarray,
         B2: np.ndarray,
         sigma: float,
         ) -> float:
    """Psi function."""
    return sigma**4 * (
        2 * np.trace(B1) * np.trace(B2)
        + 2 * (np.trace(np.dot(B1, B1)) + np.trace(np.dot(B2, B2)))
        + 8 * np.trace(np.dot(B1, B2))
    )
