"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from agoralearn.utils import format_duration, crop_pdf
from utils import load_latest_run_per_mode
from constants import CACHE_DIR, FIGURES_DIR, L_LR, L_BATCH_SIZE, MODE_STYLE, LW, ALPHA, FONTSIZE


t0 = time.time()


####################################################################################################
# Functions
def smooth_curve(y, window=5):
    """Smooths a 1D array using a centered moving average."""
    if window < 2:
        return y
    y = np.asarray(y)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def plot_stats_evolution(latest,
                         stats_name,
                         figure_dir,
                         filename,
                         smooth_window=None,
                         ref_hline=None,
                         verbose=True,
                         ):
    """Plot the evolution of a given statistic (e.g., loss, grad norm) for all modes across learning
    rates and batch sizes."""

    any_mode = next(iter(latest))
    _, all_results_ref, meta_ref = latest[any_mode]
    lrs_ref = meta_ref.get("learning_rates", L_LR)
    bss_ref = meta_ref.get("batch_sizes", L_BATCH_SIZE)
    n_rows, n_cols = len(lrs_ref), len(bss_ref)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * (n_rows + 1)),
                             sharex=True, sharey=True)

    for mode, (path, all_results_m, meta_m) in latest.items():
        lrs_m = meta_m.get("learning_rates")
        bss_m = meta_m.get("batch_sizes")
        if lrs_m != lrs_ref or bss_m != bss_ref:
            if verbose:
                print(f"[WARN] Skipping '{path}' (mode={mode}): grid mismatch with reference.")
            continue

        style = MODE_STYLE.get(mode, dict(color=None, linestyle="-", label=mode))

        for i in range(n_rows):
            for j in range(n_cols):
                result = all_results_m[(i, j)]
                epochs = result.get("epoch")
                stats = result.get(stats_name)
                if epochs is not None and stats is not None:
                    if smooth_window is not None:
                        stats = savgol_filter(stats, window_length=smooth_window, polyorder=2)
                    axes[i, j].plot(epochs, stats, lw=LW, alpha=ALPHA, **style)
                    if ref_hline is not None:
                        axes[i, j].axhline(ref_hline, color='black', ls='--', lw=LW/2, alpha=ALPHA)

    for i in range(n_rows):
        for j in range(n_cols):
            lr = all_results_ref[(i, j)].get("lr")
            bs = all_results_ref[(i, j)].get("batch_size")
            if i == n_rows - 1:
                axes[i, j].set_xlabel("Epochs", fontsize=0.9 * FONTSIZE)
            if j == 0:
                axes[i, j].set_ylabel(stats_name.replace("_", " ").capitalize(), fontsize=0.9 * FONTSIZE)
            axes[i, j].set_title(f"lr={lr:.0e}, bs={bs}", fontsize=0.8 * FONTSIZE)

    all_y = []
    for i in range(n_rows):
        for j in range(n_cols):
            for line in axes[i, j].get_lines():
                y = line.get_ydata()
                if len(y) > 0:
                    all_y.append(np.asarray(y))
    if all_y:
        y_all = np.concatenate(all_y)
        y_all = y_all[~np.isnan(y_all)]
        if y_all.size > 0:
            lo, hi = np.percentile(y_all, [2.5, 97.5])
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            for i in range(n_rows):
                for j in range(n_cols):
                    axes[i, j].set_ylim(lo - pad, hi + pad)

    for ax in axes.ravel():
        ax.margins(y=0)
        ax.tick_params(axis='y', which='both', labelsize=0.8 * FONTSIZE)
        ax.grid(True, alpha=0.4)

    handles, labels = [], []
    for line in axes[0, 0].get_lines():
        label = line.get_label()
        if ('_grad' in label) and (label not in labels):
            handles.append(line)
            labels.append(label)

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, fontsize=0.8 * FONTSIZE,
                   frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.85))

    stacked_name = os.path.join(figure_dir, filename)
    if verbose:
        print(f"[INFO] Stacked plot saved as '{stacked_name}'")
    fig.savefig(stacked_name, dpi=300)
    crop_pdf(stacked_name)


####################################################################################################
# Main
latest = load_latest_run_per_mode(CACHE_DIR)

if not latest:
    print("[WARN] No cached runs found to plot.")

else:
    stats = 'test_loss_t'
    plot_stats_evolution(latest, stats, FIGURES_DIR, f'6_{stats}_evolution.pdf')

    stats = 'crit_t'
    plot_stats_evolution(latest, stats, FIGURES_DIR, f'6_{stats}_evolution.pdf', ref_hline=0.0)

    stats = 'cos_sim_t'
    plot_stats_evolution(latest, stats, FIGURES_DIR, f'6_{stats}_evolution.pdf', smooth_window=6, ref_hline=0.0)

####################################################################################################
# Timing
print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}")

