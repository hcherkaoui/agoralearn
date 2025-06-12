#%%
"""Collaborative prediction error display."""

import os
import numpy as np
import matplotlib.pyplot as plt
from agoralearn.forecasters import LearningForecaster
from agoralearn.proxy_forecasters import ProxySamplesSharing
from agoralearn.utils import simulate_hmm
from agoralearn.transition_matrix import generate_transition_matrix


###############################################################################
# Globals
np.random.seed(None)

figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

figsize = (10, 4)
fontsize = 18
lw = 3.0

###############################################################################
# Env. settings
d = 10
n_states = 20

n_train = 50
n_test = 200
n_valid = 150
n_samples = n_train + n_test + n_valid

sigma = 1e-3
lbda = 0.0

L = 1e-5
theta_star_1 = np.array([1.0] + [0.0] * (d - 1))
theta_star_2 = np.array([1.0, L] + [0.0] * (d - 2))

states = list(np.random.randn(n_states, d))
P_1 = generate_transition_matrix(n_states, type="cyclic")

###############################################################################
# Data settings
X_1, y_1 = simulate_hmm(n=n_samples, d=d, theta_star=theta_star_1, P=P_1,
                        states=states, sigma=sigma)

X_2, y_2 = simulate_hmm(n=n_samples, d=d, theta_star=theta_star_2, P=P_1,
                        states=states, sigma=sigma)

X_train_1, y_train_1 = X_1[:n_train, :], y_1[:n_train]
X_test_1, y_test_1 = X_1[n_train:n_train+n_test, :], y_1[n_train:n_train+n_test]
X_valid_1, y_valid_1 = X_1[n_train+n_test:, :], y_1[n_train+n_test:]

X_train_2, y_train_2 = X_2[:n_train, :], y_2[:n_train]
X_test_2, y_test_2 = X_2[n_train:n_train+n_test, :], y_2[n_train:n_train+n_test]
X_valid_2, y_valid_2 = X_2[n_train+n_test:, :], y_2[n_train+n_test:]

###############################################################################
# Agents settings
agent_1_single = LearningForecaster(states=states, lbda=lbda)
agent_2_single = LearningForecaster(states=states, lbda=lbda)

dummy_agent = LearningForecaster(states=states, lbda=lbda)
agent_1_collab = ProxySamplesSharing(main_agent=dummy_agent,
                                     agents=[agent_2_single],
                                     sigma=sigma,
                                     L=L)

agent_1_single.batch_fit(X_train_1, y_train_1)
agent_2_single.batch_fit(X_train_2, y_train_2)

agent_1_collab.batch_fit(X_train_1, y_train_1)

###############################################################################
# Main
if __name__ == '__main__':

    n_collab_of_agent_1_collab = [len(agent_1_collab.collaborators)]

    y_pred_test_1_single = [agent_1_single.online_fit_predict(X_train_1[-1],
                                                              y_train_1[-1])]
    y_pred_test_2_single = [agent_2_single.online_fit_predict(X_train_2[-1],
                                                              y_train_2[-1])]
    y_pred_test_1_collab = [agent_1_collab.online_fit_predict(X_train_1[-1],
                                                              y_train_1[-1])]

    for i in range(n_test-1):

        # single agent 1
        x = X_test_1[i]
        y = y_test_1[i]
        y_pred = agent_1_single.online_fit_predict(x, y)
        y_pred_test_1_single.append(y_pred)

        # single agent 2
        x = X_test_2[i]
        y = y_test_2[i]
        y_pred = agent_2_single.online_fit_predict(x, y)
        y_pred_test_2_single.append(y_pred)

        # collab agent 1
        x = X_test_1[i]
        y = y_test_1[i]
        y_pred = agent_1_collab.online_fit_predict(x, y)
        y_pred_test_1_collab.append(y_pred)
        n_collab_of_agent_1_collab.append(len(agent_1_collab.collaborators))

    y_pred_valid_1_single = [agent_1_single.online_predict(X_test_1[-1])]
    y_pred_valid_2_single = [agent_2_single.online_predict(X_test_2[-1])]
    y_pred_valid_1_collab = [agent_1_collab.online_predict(X_test_1[-1])]

    for i in range(n_valid-1):

        # single agent 1
        x = X_valid_1[i]
        y_pred = agent_1_single.online_predict(x)
        y_pred_valid_1_single.append(y_pred)

        # single agent 2
        x = X_valid_2[i]
        y_pred = agent_2_single.online_predict(x)
        y_pred_valid_2_single.append(y_pred)

        # collab agent 1
        x = X_valid_1[i]
        y_pred = agent_1_collab.online_predict(x)
        y_pred_valid_1_collab.append(y_pred)
        n_collab_of_agent_1_collab.append(len(agent_1_collab.collaborators))

###############################################################################
# Preparing plots
    tt_train = np.arange(start=0, stop=n_train)
    tt_test = np.arange(start=n_train, stop=n_train+n_test)
    tt_valid = np.arange(start=n_train+n_test, stop=n_train+n_test+n_valid)

    y_pred_test_1_single = np.array(y_pred_test_1_single)
    y_pred_valid_1_single = np.array(y_pred_valid_1_single)

    y_pred_test_2_single = np.array(y_pred_test_2_single)
    y_pred_valid_2_single = np.array(y_pred_valid_2_single)

    y_pred_test_1_collab = np.array(y_pred_test_1_collab)
    y_pred_valid_1_collab = np.array(y_pred_valid_1_collab)

    error_test_1_single = np.cumsum((y_test_1 - y_pred_test_1_single)**2)
    error_valid_1_single = error_test_1_single[-1]
    error_valid_1_single += np.cumsum((y_valid_1 - y_pred_valid_1_single)**2)

    error_test_2_single = np.cumsum((y_test_2 - y_pred_test_2_single)**2)
    error_valid_2_single = error_test_2_single[-1]
    error_valid_2_single += np.cumsum((y_valid_2 - y_pred_valid_2_single)**2)

    error_test_1_collab = np.cumsum((y_test_1 - y_pred_test_1_collab)**2)
    error_valid_1_collab = error_test_1_collab[-1]
    error_valid_1_collab += np.cumsum((y_valid_1 - y_pred_valid_1_collab)**2)

    try:
        t_stop = n_collab_of_agent_1_collab.index(0)
    except ValueError:
        t_stop = n_test + n_valid

###############################################################################
# Plotting
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(tt_test, error_test_1_single, lw=lw, ls='solid', color='tab:orange',
            label='Single 1', alpha=0.8)
    ax.plot(tt_valid, error_valid_1_single, lw=lw, ls='solid',
            color='tab:orange', alpha=0.8)

    ax.plot(tt_test, error_test_2_single, lw=lw, ls='solid', color='tab:blue',
            label='Single 2', alpha=0.8)
    ax.plot(tt_valid, error_valid_2_single, lw=lw, ls='solid',
            color='tab:blue', alpha=0.8)

    ax.plot(tt_test, error_test_1_collab, lw=lw, ls='dashed',
            color='tab:orange', label='Collab. 1', alpha=0.8)
    ax.plot(tt_valid, error_valid_1_collab, lw=lw, ls='dashed',
            color='tab:orange', alpha=0.8)

    ax.axvline(n_train, lw=lw, linestyle='dashed', color='gray', alpha=0.8)
    ax.axvline(n_train+n_test, lw=lw, linestyle='dashed', color='gray',
               alpha=0.8)
    ax.axvline(n_train+t_stop, lw=lw, linestyle='solid', color='gray',
               label='End of collab.', alpha=0.8)

    ax.set_xlabel('Iterations', fontsize=fontsize)
    ax.set_ylabel('Cumul. error', fontsize=fontsize)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                       ncol=5, frameon=False, fontsize=int(0.75*fontsize))
    for legline in legend.get_lines():
        legline.set_linewidth(lw)

    fig.tight_layout()

    fig.show()
    fig.savefig(os.path.join(figures_dir, "time_series_forecasting_collab.pdf"), dpi=300)

# %%