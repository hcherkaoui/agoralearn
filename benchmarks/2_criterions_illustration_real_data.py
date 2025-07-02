"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from agoralearn.datasets import fetch_datasets
from agoralearn.estimation import estimate_ridge, estimate_sigma_squared_ridge
from agoralearn.criterions import criterion_expect, criterion_inst, criterion_heuri


###############################################################################
# Functions
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
            ("In expectation Test", 'tab:orange', 'solid', np.array(crit1)),
            ("Instantaneous Test", 'tab:red', 'solid', np.array(crit2)),
            ("Heuristic Test", 'tab:blue', 'solid', np.array(crit3)),
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

verbose = 100
n_jobs = -2
n_runs = 5

###############################################################################
# Settings
lbda = 0.0
delta = 0.9  # to be adjusted
beta = 0.1  # to be adjusted

###############################################################################
# Data
X1, X2, y1, y2 = fetch_datasets(dataset_name='clip')

# Ensure data is Gaussian-like
X1 = QuantileTransformer(output_distribution='normal').fit_transform(X1)
X2 = QuantileTransformer(output_distribution='normal').fit_transform(X2)

###############################################################################
# Main
if __name__ == '__main__':

    theta_1 = estimate_ridge(X1, y1, lbda)
    sigma = estimate_sigma_squared_ridge(np.r_[X1, X2], np.r_[y1, y2], lbda=lbda)

    T = min(len(X1), len(X2))
    min_T = int(T / 10)

    print('[INFO] Running criterions illustration...')
    print(f"[INFO] T: {T}, min_T: {min_T}, sigma: {sigma:.2e}, lbda: {lbda:.2e}, "
          f"len(X1): {len(X1)}, len(X2): {len(X2)}")
    kwargs = dict(theta_1=theta_1, X1=X1, X2=X2, y1=y1, y2=y2, min_T=min_T,
                  T=T, sigma=sigma, lbda=lbda, delta=delta, beta=beta)
    results = Parallel(verbose=verbose, n_jobs=n_jobs)(delayed(main)(**kwargs)
                                                       for _ in range(n_runs))

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
