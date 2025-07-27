"""Collaboration criterion on gradient."""

# Authors: Hamza Cherkaoui

import os
import argparse
import numpy as np
from joblib import Parallel, delayed
import torch
import matplotlib.pyplot as plt


###############################################################################
# Globals
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

FIG_DIR = 'figures'
FIGSIZE = (9, 4)
FONTSIZE = 14
LW = 3.5

os.makedirs(FIG_DIR, exist_ok=True)

################################################################################
# Functions
def empirical_error_difference(theta_hat, theta1):
    """Computes the empirical squared error."""
    return (theta_hat - theta1).pow(2).sum().item()


def theoretical_error_difference(g_a, theta_hat, theta, nu):
    """Computes the theoretical difference in expected error between
    collaborative and single update."""
    return float((nu ** 2) * np.sum(g_a ** 2) - 2 * nu * np.dot(g_a, theta_hat - theta))


def mse(X, y, theta):
    """Computes the mean squared error."""
    return 0.5 * torch.sum((X @ theta - y) ** 2)


def generate_x(n, d):
    """Generates a random design matrix."""
    X = np.vstack([np.eye(d)] + [np.random.randn(d) for _ in range(n - d)])
    return torch.tensor(X, dtype=torch.float32)


def generate_y(X, theta, sigma):
    """Generates noisy target values."""
    return X @ theta + sigma * torch.randn(X.size(0))


def main(tt, X1_train, X2_train, X1_test, theta1, theta2, sigma, nu):
    """Linear model training with PyTorch SGD + autograd."""

    y1_train = generate_y(X1_train, theta1, sigma)
    y2_train = generate_y(X2_train, theta2, sigma)
    y1_test = generate_y(X1_test, theta1, sigma)

    theta_single = torch.zeros_like(theta1, requires_grad=True)
    theta_collab = torch.zeros_like(theta1, requires_grad=True)
    theta_hat = theta_collab.detach().clone()

    opt_single = torch.optim.SGD([theta_single], lr=nu)
    opt_collab = torch.optim.SGD([theta_collab], lr=nu)

    delta_empirical, delta_theory = [], []
    l_collab_error, l_single_error = [], []
    l_collab_loss, l_single_loss = [], []

    for t in tt:

        X1_train_t, y1_train_t = X1_train[:t], y1_train[:t]
        X2_train_t, y2_train_t = X2_train[:t], y2_train[:t]

        # single gradient step
        opt_single.zero_grad()
        loss_single = mse(X1_train_t, y1_train_t, theta_single)
        loss_single.backward()
        opt_single.step()
        l_single_loss.append(mse(X1_test, y1_test, theta_single).item())

        # collaborative gradient step
        opt_collab.zero_grad()
        loss_collab = mse(torch.cat([X1_train_t, X2_train_t]),
                          torch.cat([y1_train_t, y2_train_t]),
                          theta_collab)
        loss_collab.backward()
        opt_collab.step()
        l_collab_loss.append(mse(X1_test, y1_test, theta_collab).item())

        # empirical error difference
        single_err = empirical_error_difference(theta_single, theta1)
        collab_err = empirical_error_difference(theta_collab, theta1)
        l_single_error.append(single_err)
        l_collab_error.append(collab_err)
        delta_empirical.append(collab_err - single_err)

        # theoretical error difference
        with torch.no_grad():
            g_a = X2_train_t.T @ (X2_train_t @ theta_hat - y2_train_t)
            delta_theory_val = theoretical_error_difference(g_a.numpy(),
                                                            theta_hat.numpy(),
                                                            theta1.numpy(),
                                                            nu,
                                                            )
            delta_theory.append(delta_theory_val)

        theta_hat = theta_collab.detach().clone() if delta_theory_val < 0 else theta_single.detach().clone()

    return {
        "Empirical $\\Delta$": np.array(delta_empirical),
        "Theoretical $\\Delta$": np.array(delta_theory),
        "Single Loss": np.array(l_single_loss),
        "Collaborative Loss": np.array(l_collab_loss),
        "Single Error": np.array(l_single_error),
        "Collaborative Error": np.array(l_collab_error),
    }


################################################################################
# Main
if __name__ == "__main__":

    print("[INFO] Setting experiment parameters...")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--joblib_verbose", type=int, default=0)
    args = parser.parse_args()

    theta1 = torch.tensor([1.0] + [0.0] * (args.d - 1), dtype=torch.float32)
    theta2 = torch.tensor([1.0, args.eps] + [0.0] * (args.d - 2), dtype=torch.float32)

    X1_train = generate_x(args.T, args.d)
    X2_train = generate_x(args.T, args.d)
    X1_test = generate_x(args.T, args.d)

    tt = np.arange(args.d, args.T, args.steps)

    print("[INFO] Comparing empirical and theoretical Delta...")
    results = Parallel(n_jobs=args.n_jobs, verbose=args.joblib_verbose)(
        delayed(main)(tt, X1_train, X2_train, X1_test, theta1, theta2, args.sigma, args.nu)
        for _ in range(args.n_runs)
    )

################################################################################
# Plotting
    T_start = 10
    l_labels = [("Empirical $\\Delta$", "Theoretical $\\Delta$"),
                ("Collaborative Error", "Single Error"),
                ("Collaborative Loss", "Single Loss"),
                ]

    fig, axes = plt.subplots(1, len(l_labels), figsize=FIGSIZE, sharex=True)

    for i, (stat_name_1, stat_name_2) in enumerate(l_labels):

        for color, label in zip(['tab:blue', 'tab:orange'], [stat_name_1, stat_name_2]):

            stacked = np.stack([r[label] for r in results], axis=0)

            tt_to_plot = tt[T_start:]
            mean_stat_to_plot = stacked.mean(axis=0)[T_start:]
            std_stat_to_plot = stacked.std(axis=0)[T_start:]

            axes[i].plot(tt_to_plot, mean_stat_to_plot, label=label, lw=LW, color=color, alpha=0.6)
            axes[i].fill_between(tt_to_plot, mean_stat_to_plot - std_stat_to_plot,
                                 mean_stat_to_plot + std_stat_to_plot, color=color, alpha=0.3)

        if 'Loss' not in stat_name_1:
            axes[i].axhline(0, color='black', lw=0.5*LW, ls='--')

        if '$\\Delta$' not in stat_name_1:
                axes[i].set_yscale('log')

        axes[i].grid(True, linestyle='--', lw=0.3*LW, alpha=0.3)
        axes[i].set_xscale('log')
        axes[i].set_xlabel("Iterations", fontsize=FONTSIZE)
        axes[i].tick_params(labelsize=0.8*FONTSIZE)
        axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=1, fontsize=0.8*FONTSIZE)

    plt.tight_layout()

    filename = os.path.join(FIG_DIR, "delta_gd_comparison.pdf")
    print(f"[INFO] Plot saved as '{filename}'.")
    plt.savefig(filename, dpi=300)

