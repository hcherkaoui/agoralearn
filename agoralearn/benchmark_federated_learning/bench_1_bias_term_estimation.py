"""Collaboration criterions comparison."""

# Authors: Hamza Cherkaoui

import os
import time
import argparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from agoralearn.utils import crop_pdf, format_duration
from estimation import estimate_james_stein_coef, estimate_ridge


t0 = time.time()


###############################################################################
# Functions
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lbda", type=float, default=0.0,
                        help="Regularization strength for Ridge regression (lambda).")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Uncertainty control parameter beta.")
    parser.add_argument("--delta", type=float, default=0.95,
                        help="Confidence level delta.")
    parser.add_argument("--d", type=int, default=10,
                        help="Dimensionality of the feature space.")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Standard deviation of the data noise.")
    parser.add_argument("--T", type=int, default=1000,
                        help="Total number of time steps (global horizon).")
    parser.add_argument("--min_T", type=int, default=50,
                        help="Minimum number of time steps per task.")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of samples steps (e.g. checkpoints).")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="Approximation threshold epsilon (e.g., for stopping or precision control).")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs.")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of repetitions for averaging.")
    parser.add_argument("--joblib_verbose", type=int, default=0,
                        help="Verbosity level for joblib.")
    return parser.parse_args()


def main(
    tt: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray,
    sigma: float,
    lbda: float,
) -> dict:
    """Main function."""
    y1 = np.array([x.T @ theta1 + sigma * np.random.randn() for x in X1])
    y2 = np.array([x.T @ theta2 + sigma * np.random.randn() for x in X2])

    err_est_nn_biased, err_est_shrinked, err_est_biased = [], [], []
    for t in tt:

        X1_t = X1[:t, :]
        y1_t = y1[:t]

        X2_t = X2[:t, :]
        y2_t = y2[:t]

        A1_t = X1_t.T @ X1_t
        A2_t = X2_t.T @ X2_t
        Ac_t = A1_t + A2_t

        A1_t_inv = np.linalg.inv(A1_t)
        A2_t_inv = np.linalg.inv(A2_t)
        Ac_t_inv = np.linalg.inv(Ac_t)

        M_t = Ac_t_inv @ A2_t
        B_t = M_t @ (A2_t_inv + A1_t_inv) @ M_t.T

        theta1_hat = estimate_ridge(X1_t, y1_t, lbda)
        theta2_hat = estimate_ridge(X2_t, y2_t, lbda)

        theta1_shrinked = estimate_james_stein_coef(theta1_hat, sigma) * theta1_hat
        theta2_shrinked = estimate_james_stein_coef(theta2_hat, sigma) * theta2_hat

        theta_diff = theta2 - theta1
        theta_hat_diff = theta2_hat - theta1_hat
        theta_shrinked_diff = theta2_shrinked - theta1_shrinked

        M_t_theta_diff = M_t.T @ theta_diff
        M_t_theta_hat_diff = M_t.T @ theta_hat_diff
        M_t_theta_shrinked_diff = M_t.T @ theta_shrinked_diff

        b = float(M_t_theta_diff.T @ M_t_theta_diff)
        b_hat_biased = float(M_t_theta_hat_diff.T @ M_t_theta_hat_diff)
        b_hat_shrinked = float(M_t_theta_shrinked_diff.T @ M_t_theta_shrinked_diff)
        b_hat = b_hat_biased - sigma**2 * np.trace(B_t)

        err_est_biased.append((b_hat_biased - b)**2)
        err_est_shrinked.append((b_hat_shrinked - b)**2)
        err_est_nn_biased.append((b_hat - b)**2)

    to_return = [
        ("Empirical estimation error (unbiased)",
         'tab:blue', np.array(err_est_nn_biased)),
        ("Empirical estimation error (shrinked)",
         'tab:red', np.array(err_est_shrinked)),
        ("Empirical estimation error (biased)",
         'tab:orange', np.array(err_est_biased)),
        ]

    return to_return


###############################################################################
# Globals
np.random.seed(0)

figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

fontsize = 18
lw = 3
alpha = 0.5

###############################################################################
# Main
if __name__ == '__main__':

    args = parse_args()

    theta1 = np.array([1.0] + [0.0] * (args.d - 1))
    theta2 = np.array([1.0, args.eps] + [0.0] * (args.d - 2))

    tt = np.arange(args.min_T, args.T, args.steps)

    X1 = np.array(list(np.eye(args.d)) + [np.random.randn(args.d) for _ in range(args.T - args.d)])
    X2 = np.array(list(np.eye(args.d)) + [np.random.randn(args.d) for _ in range(args.T - args.d)])

    print('[INFO] Running bias term estimation...')
    kwargs = dict(tt=tt, X1=X1, X2=X2, theta1=theta1, theta2=theta2,
                  sigma=args.sigma, lbda=args.lbda)
    results = Parallel(verbose=args.joblib_verbose, n_jobs=args.n_jobs)(
        delayed(main)(**kwargs) for _ in range(args.n_runs)
        )

###############################################################################
# Plotting 1
    figsize = (4, 5)

    fig, ax = plt.subplots(figsize=figsize)

    for result in zip(*results):  # iteration on line types
        name, color, stats = zip(*result)  # iteration on trials

        mean_stats = np.mean(np.atleast_2d(stats), axis=0)

        ax.plot(tt, mean_stats, lw=lw, color=color[0], label=name[0],
                alpha=alpha)

    ax.set_xlabel('Iterations', fontsize=fontsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='y', which='both', labelsize=int(0.7*fontsize))

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
                       ncol=1, frameon=False, fontsize=int(0.7*fontsize))

    for legline in legend.get_lines():
        legline.set_linewidth(lw)

    fig.tight_layout()

    filename = os.path.join(figures_dir, "1_err_bias_term.pdf")
    print("[INFO] Saving figure to", filename)
    fig.savefig(filename, dpi=300)
    crop_pdf(filename)

###############################################################################
# Timing
    print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}")

