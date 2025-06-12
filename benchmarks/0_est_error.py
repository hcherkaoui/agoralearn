#%%
"""Estimation error display."""

import os
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from agoralearn.utils import simulate_hmm
from agoralearn.criterions import criterion


###############################################################################
# Functions
def estimate_theta(X, y, lbda=None):
    """Estimate regression weights using OLS or Ridge regression.

    Args:
        X (np.ndarray): Design matrix of shape (n_samples, d).
        y (np.ndarray): Target vector of shape (n_samples,).
        lbda (float, optional): Regularization strength. Defaults to 1e-6.

    Returns:
        np.ndarray: Estimated parameter vector of shape (d,).
    """
    d = X.shape[1]
    if lbda is None:
        lbda = 1e-6
    XtX = X.T @ X
    reg = lbda * np.eye(d)
    return np.linalg.solve(XtX + reg, X.T @ y)


def lauch_one_run(theta_star_1,
                  theta_star_2,
                  P1,
                  P2,
                  states1,
                  states2,
                  n_train_samples=500,
                  n_train_min_samples=10,
                  lbda = 1e-6,
                  sigma=1.0):
    """Launch one experiment run."""
    d = len(theta_star_1)

    full_train_X1, full_train_y1 = simulate_hmm(n_train_samples, d,
                                                theta_star_1, P1, states1,
                                                sigma)
    full_train_X2, full_train_y2 = simulate_hmm(n_train_samples, d,
                                                theta_star_2, P2, states2,
                                                sigma)

    errors_1_per_run, errors_c_per_run = [], []
    collab_condition_per_run = []

    for n in range(n_train_min_samples, n_train_samples):

        X_train_1, y_train_1 = full_train_X1[:n], full_train_y1[:n]
        X_train_2 = full_train_X2[:n]
        X_train_c = np.r_[full_train_X1[:n], full_train_X2[:n]]
        y_train_c = np.r_[full_train_y1[:n], full_train_y2[:n]]

        c = criterion(X_train_1, X_train_2, theta_star_1, theta_star_2, lbda,
                      sigma)
        theta_hat_1 = estimate_theta(X_train_1, y_train_1, lbda=lbda)
        theta_hat_c = estimate_theta(X_train_c, y_train_c, lbda=lbda)

        errors_1_per_run.append(np.mean((theta_hat_1 - theta_star_1)**2))
        errors_c_per_run.append(np.mean((theta_hat_c - theta_star_1)**2))
        collab_condition_per_run.append(c)

    return errors_1_per_run, errors_c_per_run, collab_condition_per_run


def gather_results(results):
    """Gather results form runs before plotting."""
    err_1, err_c, t_stop = [], [], []

    for err_1_per_run, err_c_per_run, collab_condition_per_run in results:

        err_1.append(err_1_per_run)
        err_c.append(err_c_per_run)

        try:
            t = collab_condition_per_run.index(0)
        except:
            t = len(collab_condition_per_run)
        t_stop.append(t)

    return err_1, err_c, t_stop

###############################################################################
# Globals
np.random.seed(42)

figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

figsize = (4, 3)
fontsize = 18
lw = 4.0

###############################################################################
# Settings

n_jobs = 4
n_trials = 10

d = 30
n_train_samples = 5000
n_train_min_samples = 100
sigma = 1.0
beta = 0.5
lbda = 0.1

theta_star_1 = np.array([1.0] + [0.0] * (d - 1))
theta_star_2 = theta_star_1 + beta * np.random.randn(d)

states_1 = np.eye(d)
states_2 = np.vstack([np.roll(np.eye(d), i, axis=0)[0] for i in range(d)])

P_1 = np.full((d, d), 1/d)
P_2 = np.eye(d) * 0.7 + np.full((d, d), 0.3 / d)

###############################################################################
# Main
if __name__ == '__main__':

    kwargs = dict(theta_star_1=theta_star_1, theta_star_2=theta_star_2, P1=P_1,
                  P2=P_2, states1=states_1, states2=states_2,
                  n_train_min_samples=n_train_min_samples,
                  n_train_samples=n_train_samples, lbda=lbda, sigma=sigma)

    results = Parallel(n_jobs=n_jobs)(delayed(lauch_one_run)(**kwargs)
                                      for i in range(n_trials))
    errors_1, errors_c, t_stop = gather_results(results)

    mean_errors_1 = np.mean(errors_1, axis=0)
    std_errors_1 = np.std(errors_1, axis=0)

    mean_errors_c = np.mean(errors_c, axis=0)
    std_errors_c = np.std(errors_c, axis=0)

    mean_t_stop = np.mean(t_stop)
    std_t_stop = np.std(t_stop)

    tt = np.arange(n_train_min_samples, n_train_samples)

###############################################################################
# Plotting
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(tt, mean_errors_1, lw=lw, label='Est. error', color='tab:blue',
            alpha=0.8)
    ax.fill_between(tt, mean_errors_1 - std_errors_1,
                    mean_errors_1 + std_errors_1, color='tab:blue', alpha=0.4)

    ax.plot(tt, mean_errors_c, lw=lw, label='Coll. est. error',
            color='tab:orange', alpha=0.8)
    ax.fill_between(tt, mean_errors_c - std_errors_c,
                    mean_errors_c + std_errors_c, color='tab:orange', alpha=0.4)

    ax.axvline(mean_t_stop + n_train_min_samples, label='Coll. stopping',
               color='gray', alpha=0.8)
    ax.axvspan(mean_t_stop + n_train_min_samples - std_t_stop,
               mean_t_stop + n_train_min_samples + std_t_stop,
               color='gray', alpha=0.4)

    ax.set_xscale('log')

    ax.set_xlabel('Iterations', fontsize=fontsize)
    ax.set_ylabel('Error', fontsize=fontsize)
    ax.legend(fontsize=int(0.5*fontsize))

    fig.tight_layout()

    fig.show()
    fig.savefig(os.path.join(figures_dir, "estimation_errors.pdf"), dpi=300)

# %%