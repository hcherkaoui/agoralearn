"""Collaboration criterions comparison."""

# Authors: Hamza Cherkaoui

import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from agoralearn.estimation import estimate_ridge
from agoralearn.criterions import criterion_expect, criterion_inst, criterion_heuri


###############################################################################
# Functions
def first_pos_index(
        stat_run: np.ndarray,
) -> int:
    """Return the index of the first negative value in stat_run, or len(stat_run) if none."""
    indices = np.where(stat_run > 0.0)[0]
    return int(indices[0]) if indices.size > 0 else len(stat_run)


def main(
    X1: np.ndarray,
    X2: np.ndarray,
    min_T: int,
    T: int,
    theta1: np.ndarray,
    theta2: np.ndarray,
    sigma: float,
    lbda: float,
    delta: float,
    beta: float,
) -> dict:
    """Main function."""
    y1 = np.array([x.T @ theta1 + sigma * np.random.randn() for x in X1])
    y2 = np.array([x.T @ theta2 + sigma * np.random.randn() for x in X2])

    diff_err, crit1, crit2, crit3, crit4, crit5, crit6 = [], [], [], [], [], [], []
    for t in range(min_T, T):

        X1_t = X1[:t, :]
        y1_t = y1[:t]

        X2_t = X2[:t, :]
        y2_t = y2[:t]

        theta_1_hat = estimate_ridge(X1_t, y1_t, lbda)
        theta_2_hat = estimate_ridge(X2_t, y2_t, lbda)
        theta_collab_hat = estimate_ridge(np.r_[X1_t, X2_t], np.r_[y1_t, y2_t], lbda)

        _err_collab = np.sum(np.square(theta_collab_hat - theta1))
        _err_single = np.sum(np.square(theta_1_hat - theta1))

        diff_err.append(_err_collab - _err_single)

        crit1.append(criterion_expect(theta1, theta2, X1_t, X2_t, sigma))
        crit2.append(criterion_expect(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma, use_estimate=True))

        crit3.append(criterion_inst(theta1, theta2, X1_t, X2_t, sigma, delta))
        crit4.append(criterion_inst(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma, delta, use_estimate=True))

        crit5.append(criterion_heuri(theta1, theta2, X1_t, X2_t, sigma, beta))
        crit6.append(criterion_heuri(theta_1_hat, theta_2_hat, X1_t, X2_t, sigma, beta, use_estimate=True))

    to_return = [
            ("Empirical error difference", 'tab:blue', 'solid', np.array(diff_err)),
            ("Oracle in expectation Test", 'tab:orange', 'solid', np.array(crit1)),
            ("In expectation Test", 'tab:orange', 'dashed', np.array(crit2)),
            ("Oracle instantaneous Test", 'tab:red', 'solid', np.array(crit3)),
            ("Instantaneous Test", 'tab:red', 'dashed', np.array(crit4)),
            ("Oracle heuristic Test", 'tab:olive', 'solid', np.array(crit5)),
            ("Heuristic Test", 'tab:olive', 'dashed', np.array(crit6)),
                 ]

    return to_return


###############################################################################
# Globals
np.random.seed(0)

figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

fontsize = 18
lw = 1.5
alpha = 0.5

verbose = 100

n_jobs = -1
n_runs = 8 * 4

###############################################################################
# Settings
d = 50

min_T = d
T = 500

sigma = 1.0
lbda = 0.0

delta = 0.9  # to be adjusted
beta = 0.1  # to be adjusted

eps = 0.5

theta1 = np.array([1.0] + [0.0] * (d - 1))
theta2 = np.array([1.0, eps] + [0.0] * (d - 2))

X1 = np.array(list(np.eye(d)) + [np.random.randn(d) for _ in range(T - d)])
X2 = np.array([np.random.randn(d) for _ in range(T)])

###############################################################################
# Main
if __name__ == '__main__':

    print('[INFO] Running criterions comparison...')
    kwargs = dict(X1=X1, X2=X2, min_T=min_T, T=T, theta1=theta1, theta2=theta2,
                  sigma=sigma, lbda=lbda, delta=delta, beta=beta)
    results = Parallel(verbose=verbose, n_jobs=n_jobs)(delayed(main)(**kwargs)
                                                       for _ in range(n_runs))

###############################################################################
# Plotting 1
    figsize = (4, 7)
    tt = np.arange(min_T, T)

    fig, ax = plt.subplots(figsize=figsize)

    for result in zip(*results):  # iteration on line types
        name, color, linestyle, stats = zip(*result)  # iteration on trials

        mean_stats = np.mean(np.atleast_2d(stats), axis=0)
        std_stats = np.std(np.atleast_2d(stats), axis=0)
        ax.plot(tt, mean_stats, lw=lw, color=color[0], linestyle=linestyle[0],
                label=name[0], alpha=alpha)
        ax.fill_between(tt, mean_stats + std_stats, mean_stats - std_stats,
                        color=color[0], alpha=alpha/5)

        idx_first_pos = max(min_T, first_pos_index(mean_stats))
        ax.axvline(idx_first_pos, lw=lw/2, color=color[0],  linestyle=linestyle[0],
                   alpha=alpha)

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

    filename = os.path.join(figures_dir, "criterions_comparison.pdf")

    print("[INFO] Saving figure to", filename)
    fig.savefig(filename, dpi=300)

###############################################################################
# Plotting 2
    figsize = (4, 6)

    fig, ax = plt.subplots(figsize=figsize)

    to_plot = []
    for result in zip(*results):  # iteration on line types
        name, color, linestyle, stats = zip(*result)  # iteration on trials

        ss = [first_pos_index(stats_per_run)
              for stats_per_run in np.atleast_2d(stats)]

        for n, c, s in zip(name, color, ss):
            to_plot.append({'name': n, 'color': c, 'count': s})

    df = pd.DataFrame(to_plot)

    for name, group in df.groupby("name"):
        color = group["color"].iloc[0]
        sns.kdeplot(
            data=group,
            x="count",
            ax=ax,
            label=name,
            color=color,
            linewidth=lw,
            alpha=alpha,
        )

    ax.set_xlim(left=0)
    ax.set_xlabel('Stopping iteration density', fontsize=fontsize)
    ax.set_ylabel('', fontsize=fontsize)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
                       ncol=1, frameon=False, fontsize=int(0.7*fontsize))

    for legline in legend.get_lines():
        legline.set_linewidth(lw)

    fig.tight_layout()

    fig.show()

    filename = os.path.join(figures_dir, "stopping_times_comparison.pdf")

    print("[INFO] Saving figure to", filename)
    fig.savefig(filename, dpi=300)

# %%
