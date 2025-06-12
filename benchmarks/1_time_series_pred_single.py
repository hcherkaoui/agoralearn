#%%
"""Prediction error display."""

import os
import numpy as np
import matplotlib.pyplot as plt
from agoralearn.forecasters import LearningForecaster
from agoralearn.utils import simulate_hmm
from agoralearn.transition_matrix import generate_transition_matrix


###############################################################################
# Globals
np.random.seed(42)

figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

figsize = (10, 4)
fontsize = 18
lw = 3.0

###############################################################################
# Settings
d = 30
n_states = 40

n_train = 5
n_test = 69
n_val = 30

sigma = 1e-1
lbda = 1e-1

theta_star = np.array([1.0] + [0.0] * (d - 1))
states = list(np.random.randn(n_states, d))
P = generate_transition_matrix(n_states, type="cyclic")

X, y = simulate_hmm(n=n_train+n_test+n_val,
                    d=d,
                    theta_star=theta_star,
                    P=P,
                    states=states,
                    sigma=sigma)

X_train = X[:n_train, :]
X_test = X[n_train:n_test, :]
X_val = X[n_train+n_test:, :]
y_train = y[:n_train]
y_test = y[n_train:n_test]
y_val = y[n_train+n_test:]

###############################################################################
# Main
if __name__ == '__main__':

    forecaster = LearningForecaster(states=states, lbda=lbda)

    forecaster.batch_fit(X_train, y_train)

    err_est = [np.linalg.norm(forecaster.theta - theta_star)]
    y_test_pred = [forecaster.online_fit_predict(X_train[-1], y_train[-1])]
    for x, y in zip(X_test[:-1], y_test[:-1]):
        y_test_pred.append(forecaster.online_fit_predict(x, y))
        err_est.append(np.linalg.norm(forecaster.theta - theta_star))

    err_test = [(y_test_ - y_test_pred_)**2
                for y_test_, y_test_pred_ in zip(y_test, y_test_pred)]

    y_val_pred = [forecaster.online_predict(X_test[-1])]
    for x in X_val[:-1]:
        y_val_pred.append(forecaster.online_predict(x))

    err_val = [(y_val_ - y_val_pred_)**2
               for y_val_, y_val_pred_ in zip(y_val, y_val_pred)]

###############################################################################
# Preparing plots
    tt_train = np.arange(stop=len(y_train))
    tt_test = np.arange(start=len(y_train), stop=len(y_train)+len(y_test))
    tt_val = np.arange(start=len(y_train)+len(y_test),
                       stop=len(y_train)+len(y_test)+len(y_val))

###############################################################################
# Plotting
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(tt_train, y_train, lw=lw, label='True', color='tab:blue',
            alpha=0.8)
    ax.plot(tt_test, y_test, lw=lw, color='tab:blue', alpha=0.8)
    ax.plot(tt_val, y_val, lw=lw, color='tab:blue', alpha=0.8)

    ax.plot(tt_test, y_test_pred, lw=lw, linestyle='--',
            label='Test pred.', color='tab:orange', alpha=0.8)
    ax.plot(tt_val, y_val_pred, lw=lw, linestyle='--',
            label='Val. pred.', color='tab:red', alpha=0.8)

    ax.axvline(len(y_train), lw=lw, linestyle='--', color='gray', alpha=0.8)
    ax.axvline(len(y_train)+len(y_test), lw=lw, linestyle='--', color='gray',
               alpha=0.8)

    ax_twin_1 = ax.twinx()
    ax_twin_1.tick_params(axis='y', colors='tab:red')
    ax_twin_2 = ax.twinx()
    ax_twin_2.tick_params(axis='y', colors='tab:green')
    ax_twin_2.spines['right'].set_position(('outward', 75))

    ax_twin_1.plot(tt_test, err_est, lw=lw, label='Est. err.', color='tab:red',
                   alpha=0.4)
    ax_twin_2.plot(tt_test, err_test, lw=lw, label='Test err.',
                   color='tab:green', alpha=0.4)
    ax_twin_2.plot(tt_val, err_val, lw=lw, label='Val err.',
                   color='tab:green', alpha=0.4)

    ax.set_xlabel('Iterations', fontsize=fontsize)
    ax.set_ylabel('Time-series', color='tab:orange', fontsize=fontsize)
    ax_twin_1.set_ylabel('Est. error', color='tab:red', fontsize=fontsize)
    ax_twin_2.set_ylabel('Pred. error', color='tab:green', fontsize=fontsize)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                       ncol=5, frameon=False, fontsize=int(0.75*fontsize))
    for legline in legend.get_lines():
        legline.set_linewidth(lw)

    fig.tight_layout()

    fig.show()
    fig.savefig(os.path.join(figures_dir, "time_series_forecasting_single.pdf"), dpi=300)

# %%