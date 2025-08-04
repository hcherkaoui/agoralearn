"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
import time
import argparse
import pickle
import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from agoralearn.utils import format_duration, experiment_filename_suffix
from utils import (load_cifar_10_and_adapted_100, SimpleCNN, evaluate, compute_per_sample_gradients,
                   collaboration_criterion, set_random_torch_numpy, compute_g_mean, grads_alignement)
from constants import CACHE_DIR, L_LR, L_BATCH_SIZE, GRAD_MODES


t0 = time.time()


####################################################################################################
# Functions
def _atomic_save(path, data):
    """Save 'data' under the Pickle file 'path'."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, path)

####################################################################################################
# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="Nb of training iterations")
parser.add_argument("--gradient_mode", type=str, default="no_additional_grad", choices=GRAD_MODES, help="Gradient augmentation mode.")
parser.add_argument("--train_size", type=int, default=None, help="Nb of train samples (None for full dataset).")
parser.add_argument("--additional_train_size", type=int, default=None, help="Nb of the additional train samples (None for full dataset).")
parser.add_argument("--test_size", type=int, default=None, help="Nb of test samples (None for full dataset).")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--device", type=str, default='cpu', help="Torch device")
args = parser.parse_args()

print("[INFO] Setting training loop.")

device = torch.device(args.device)
set_random_torch_numpy(args.seed)

train_10, train_100, test_10 = load_cifar_10_and_adapted_100(root='./_data',
                                                             train_size=args.train_size,
                                                             test_size=args.test_size,
                                                             additional_train_size=args.additional_train_size,
                                                             seed=args.seed,
                                                             )

if len(train_10) > len(train_100):
    raise ValueError(f"The massive additional dataset has less samples than the main dataset,"
                     f" got len(train_10) = {len(train_10)} and len(train_100) = {len(train_100)}")

suffix = experiment_filename_suffix(args)
results_file = os.path.join(CACHE_DIR, f"_6_test_losses{suffix}.pkl")

meta = {"n_epochs": args.n_epochs,
        "device": str(device),
        "learning_rates": L_LR,
        "batch_sizes": L_BATCH_SIZE,
        "gradient_mode": args.gradient_mode,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if device.type=="cuda" else "cpu",
        "script_start_time": t0,
        }

all_results = {(i, j): {"lr": L_LR[i], "batch_size": L_BATCH_SIZE[j],
                        "epoch": [-1], "test_loss_t": [np.nan],
                        "crit_t": [np.nan], "cos_sim_t": [np.nan]}
                for i in range(len(L_LR)) for j in range(len(L_BATCH_SIZE))}
_atomic_save(results_file, (all_results, meta))

####################################################################################################
# Main
for i in range(len(L_LR)):

    for j in range(len(L_BATCH_SIZE)):

        lr = L_LR[i]
        batch_size = L_BATCH_SIZE[j]

        model = SimpleCNN().to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        test_loader = DataLoader(test_10, batch_size=256, shuffle=False)

        desc = f"[INFO] Training model with lr={lr:.0e}, bs={batch_size}, mode={args.gradient_mode}"
        for epoch in tqdm.tqdm(range(args.n_epochs), total=args.n_epochs, desc=desc, dynamic_ncols=True):

            train_ref_loader = DataLoader(train_10, batch_size=batch_size, shuffle=True, drop_last=True)
            train_add_loader = DataLoader(train_100, batch_size=batch_size, shuffle=True, drop_last=True)

            for (x10, y10), (x100, y100) in zip(train_ref_loader, train_add_loader):

                x10 = x10.to(device)
                x100 = x100.to(device)
                y10 = y10.to(device)
                y100 = y100.to(device)

                grads_s, grads_a = compute_per_sample_gradients(model, loss_fn, x10, y10, x100, y100)
                crit = collaboration_criterion(grads_s, grads_a, eta=lr)
                g_mean = compute_g_mean(args.gradient_mode, grads_s, grads_a, crit)

                optimizer.zero_grad(set_to_none=True)
                offset = 0
                for p in model.parameters():
                    n = p.numel()
                    p.grad = g_mean[offset:offset + n].view_as(p).detach().clone()
                    offset += n
                optimizer.step()

            all_results[i, j]["epoch"].append(epoch + 1)
            all_results[i, j]["test_loss_t"].append(evaluate(loss_fn, model, test_loader, device))
            all_results[i, j]["crit_t"].append(crit if args.gradient_mode == 'dynamic_additional_grad' else np.nan)
            all_results[i, j]["cos_sim_t"].append(grads_alignement(grads_s, grads_a))

            _atomic_save(results_file, (all_results, meta))

print(f"[INFO] Training results saved to '{results_file}'.")

####################################################################################################
# Timing
print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}")

