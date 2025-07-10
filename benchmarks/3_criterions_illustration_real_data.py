"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
import time
import argparse
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from agoralearn.datasets import fetch_datasets
from agoralearn.estimation import estimate_ridge, estimate_sigma_ridge
from agoralearn.criterions import criterion_expect, criterion_inst, criterion_heuri
from agoralearn.utils import check_random_state, experiment_suffix, format_duration


t0 = time.time()

################################################################################
# Functions
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset to load (e.g., 'ccpp', 'wine').")
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
    parser.add_argument("--seed", type=int, default=0,
                        help="Random state seed.")
    return parser.parse_args()


def main(
    pseudo_theta_1_star: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    min_T: int,
    T: int,
    sigma_1: float,
    sigma_2: float,
    lbda: float,
    delta: float,
    beta: float,
    random_state: None,
) -> dict:
    """Main function."""
    random_state = check_random_state(random_state)

    idx = random_state.permutation(len(X1))
    X1, y1 = X1[idx], y1[idx]

    idx = random_state.permutation(len(X2))
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

        err_1 = np.sum(np.square(theta_1_hat - pseudo_theta_1_star))
        err_collab = np.sum(np.square(theta_c_hat - pseudo_theta_1_star))
        diff_err = err_collab - err_1

        crit0.append(diff_err)
        crit1.append(criterion_expect(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma_1, sigma_2, use_estimate=True))
        crit2.append(criterion_inst(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma_1, sigma_2, delta, use_estimate=True))
        crit3.append(criterion_heuri(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma_1, sigma_2, beta, use_estimate=True))

    to_return = [
            ("Empirical error difference", 'gray', 'solid', np.array(crit0)),
            ("In expectation Test", 'tab:orange', 'solid', np.array(crit1)),
            ("Instantaneous Test", 'tab:red', 'solid', np.array(crit2)),
            ("Heuristic Test", 'tab:blue', 'solid', np.array(crit3)),
                 ]

    return to_return


################################################################################
# Globals
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

fontsize = 18
lw = 1.5
alpha = 0.5

################################################################################
# Main
if __name__ == '__main__':

    args = parse_args()

    filename_suffix = experiment_suffix(args)

    random_state = check_random_state(args.seed)

    X1, X2, y1, y2 = fetch_datasets(dataset_name=args.dataset_name)

    pseudo_theta_1_star = estimate_ridge(X1, y1, args.lbda)
    sigma_1 = estimate_sigma_ridge(X1, y1, lbda=args.lbda)
    sigma_2 = estimate_sigma_ridge(X2, y2, lbda=args.lbda)

    d = X1.shape[1]

    T = min(len(X1), len(X2))
    min_T = int(T / 30)

    print('[INFO] Running criterions illustration...')
    print(f"[INFO] T: {T}, min_T: {min_T}, d: {d}, sigma_1: {sigma_1:.2e}, sigma_2: {sigma_2:.2e}"
          f" lbda: {args.lbda:.2e}, len(X1): {len(X1)}, len(X2): {len(X2)}")

    kwargs = dict(pseudo_theta_1_star=pseudo_theta_1_star, X1=X1, X2=X2, y1=y1, y2=y2, min_T=min_T,
                  T=T, sigma_1=sigma_1, sigma_2=sigma_2, lbda=args.lbda, delta=args.delta,
                  beta=args.beta, random_state=random_state)
    results = Parallel(verbose=args.joblib_verbose, n_jobs=args.n_jobs)(
        delayed(main)(**kwargs) for _ in range(args.n_runs)
        )

################################################################################
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
    ax.tick_params(axis='y', which='both', labelsize=int(0.7*fontsize))
    ax.grid(True, which='both', axis='y')

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55),
                       ncol=1, frameon=False, fontsize=int(0.7*fontsize))

    for legline in legend.get_lines():
        legline.set_linewidth(lw)

    fig.tight_layout()

    filename = os.path.join(figures_dir, "criterions_illustration") + filename_suffix + '.pdf'

    print("[INFO] Saving figure to", filename)
    fig.savefig(filename, dpi=300)

################################################################################
# Timing
print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}")
