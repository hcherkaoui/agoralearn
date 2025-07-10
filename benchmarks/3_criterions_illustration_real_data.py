"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
import argparse
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from agoralearn.datasets import fetch_datasets
from agoralearn.estimation import estimate_ridge, estimate_sigma_ridge
from agoralearn.criterions import criterion_expect, criterion_inst, criterion_heuri


###############################################################################
# Functions
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lbda", type=float, default=0.0,
                        help="Regularization strength for Ridge regression (lambda).")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Uncertainty control parameter beta.")
    parser.add_argument("--delta", type=float, default=0.95,
                        help="Confidence level delta (typically small, e.g. 0.05).")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs to run with joblib.")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of repetitions for statistical stability.")
    parser.add_argument("--joblib_verbose", type=int, default=0,
                        help="Verbosity level for joblib (0 = silent).")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset to load (e.g., 'ccpp', 'wine').")
    return parser.parse_args()


def main(
    theta_1: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    min_T: int,
    T: int,
    sigma: float,
    lbda: float,
    delta: float,
    beta: float,
) -> dict:
    """Main function."""
    idx = np.random.permutation(len(X1))
    X1, y1 = X1[idx], y1[idx]

    idx = np.random.permutation(len(X2))
    X2, y2 = X2[idx], y2[idx]

    crit0, crit1, crit2, crit3 = [], [], [], []
    for t in range(min_T, T):

        X1_t = X1[:t, :]
        y1_t = y1[:t]

        X2_t = X2[:t, :]
        y2_t = y2[:t]

        theta_1_hat = estimate_ridge(X1_t, y1_t, lbda)
        theta_2_hat = estimate_ridge(X2_t, y2_t, lbda)
        theta_c_hat = estimate_ridge(np.r_[X1_t, X2_t], np.r_[y1_t, y2_t], lbda)

        err_1 = np.sum(np.square(theta_1_hat - theta_1))
        err_collab = np.sum(np.square(theta_c_hat - theta_1))
        diff_err = err_collab - err_1

        crit0.append(diff_err)
        crit1.append(criterion_expect(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma, use_estimate=True))
        crit2.append(criterion_inst(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma, delta, use_estimate=True))
        crit3.append(criterion_heuri(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma, beta, use_estimate=True))

    to_return = [
            ("Empirical error difference", 'gray', 'solid', np.array(crit0)),
            # ("In expectation Test", 'tab:orange', 'solid', np.array(crit1)),
            # ("Instantaneous Test", 'tab:red', 'solid', np.array(crit2)),
            # ("Heuristic Test", 'tab:blue', 'solid', np.array(crit3)),
                 ]

    return to_return


###############################################################################
# Globals
np.random.seed(0)

figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

fontsize = 18
lw = 4.0
alpha = 0.5

###############################################################################
# Main
if __name__ == '__main__':

    args = parse_args()

    X1, X2, y1, y2 = fetch_datasets(dataset_name=args.dataset_name)

    theta_1 = estimate_ridge(X1, y1, args.lbda)
    sigma = estimate_sigma_ridge(np.r_[X1, X2], np.r_[y1, y2], lbda=args.lbda)

    d = X1.shape[1]

    T = min(len(X1), len(X2))
    min_T = int(T / 10)

    print('[INFO] Running criterions illustration...')
    print(f"[INFO] T: {T}, min_T: {min_T}, d: {d}, sigma: {sigma:.2e}, lbda: {args.lbda:.2e}, "
          f"len(X1): {len(X1)}, len(X2): {len(X2)}")
    kwargs = dict(theta_1=theta_1, X1=X1, X2=X2, y1=y1, y2=y2, min_T=min_T,
                  T=T, sigma=sigma, lbda=args.lbda, delta=args.delta, beta=args.beta)
    results = Parallel(verbose=args.joblib_verbose, n_jobs=args.n_jobs)(
        delayed(main)(**kwargs) for _ in range(args.n_runs)
        )

###############################################################################
# Plotting 1
    figsize = (4, 8)
    tt = np.arange(min_T, T)

    fig, ax = plt.subplots(figsize=figsize)

    for result in zip(*results):  # iteration on line types
        name, color, linestyle, test_value = zip(*result)  # iteration on trials

        mean_test_value = np.mean(np.atleast_2d(test_value), axis=0)

        if np.isnan(mean_test_value).any():
            print(f"[WARNING] {name[0]} contains NaN values, skipping plot.")
            continue

        ax.plot(tt, mean_test_value, lw=lw, color=color[0], linestyle=linestyle[0],
                label=name[0], alpha=alpha)

    ax.axhline(0.0, lw=lw/2, linestyle='dashed', color='black', alpha=alpha)

    ax.set_xlabel('Iterations', fontsize=fontsize)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1)
    ax.tick_params(axis='y', which='both', labelsize=int(0.7*fontsize))
    ax.grid(True, which='both', axis='y')

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55),
                       ncol=1, frameon=False, fontsize=int(0.7*fontsize))

    for legline in legend.get_lines():
        legline.set_linewidth(lw)

    fig.tight_layout()

    fig.show()

    filename = os.path.join(figures_dir, "criterions_illustration.pdf")

    print("[INFO] Saving figure to", filename)
    fig.savefig(filename, dpi=300)
