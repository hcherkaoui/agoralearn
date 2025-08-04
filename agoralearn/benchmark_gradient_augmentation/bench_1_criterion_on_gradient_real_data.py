"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from agoralearn.utils import crop_pdf, format_duration
from utils import SimpleCNN, gradient_alignment, compute_per_sample_gradients, collaboration_criterion, evaluate, set_random_torch_numpy
from constants import FIGURES_DIR, LW, ALPHA, FONTSIZE


t0 = time.time()


####################################################################################################
# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--n_iters", type=int, default=100, help="Nb of training iterations")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=0, help="Random seed for subsetting (reproducible sampling).")
args = parser.parse_args()

print("[INFO] Loading datasets.")
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

cifar10_train = datasets.CIFAR10(root='./_data', train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(root='./_data', train=False, download=True, transform=transform)
loader_cifar10_train = DataLoader(cifar10_train, batch_size=64, shuffle=True)
loader_cifar10_test = DataLoader(cifar10_test, batch_size=256, shuffle=False)

add_train = datasets.CIFAR10(root='./_data', train=True, download=True, transform=transform)
loader_add_train = DataLoader(add_train, batch_size=64, shuffle=True)

print("[INFO] Setting model.")

set_random_torch_numpy(args.seed)

model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("[INFO] Setting training loop.")
loss_diff, criterion, cosine_sim = [], [], []
model.train()

iter_cifar10_train = iter(loader_cifar10_train)
iter_add_train = iter(loader_add_train)

####################################################################################################
# Main
for step in tqdm(range(args.n_iters), desc="[INFO] Training model"):

    try:
        x10, y10 = next(iter_cifar10_train)
        x_add, y_add = next(iter_add_train)

    except StopIteration:
        break

    cos_sim_t = gradient_alignment(model, loss_fn, x10, y10, x_add, y_add)
    cosine_sim.append(cos_sim_t)

    grads_s, grads_a = compute_per_sample_gradients(model, loss_fn, x10, y10, x_add, y_add)
    crit_t = collaboration_criterion(grads_s, grads_a, args.lr)
    criterion.append(crit_t)

    model_state_backup = {k: v.clone() for k, v in model.state_dict().items()}

    grad_vec = grads_s.mean(dim=0) + grads_a.mean(dim=0) if crit_t < 0 else grads_s.mean(dim=0)
    optimizer.zero_grad()
    offset = 0
    for param in model.parameters():
        n = param.numel()
        param.grad = grad_vec[offset:offset + n].view_as(param).clone()
        offset += n
    optimizer.step()
    test_loss_with = evaluate(loss_fn, model, loader_cifar10_test)

    model.load_state_dict(model_state_backup)
    optimizer.zero_grad()
    offset = 0
    grad_vec_no_collab = grads_s.mean(dim=0)
    for param in model.parameters():
        n = param.numel()
        param.grad = grad_vec_no_collab[offset:offset + n].view_as(param).clone()
        offset += n
    optimizer.step()
    test_loss_without = evaluate(loss_fn, model, loader_cifar10_test)

    loss_diff.append(test_loss_without - test_loss_with)

####################################################################################################
# Plotting
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.plot(cosine_sim, lw=LW, label="Cosine similarity", color="tab:blue", alpha=ALPHA)
plt.axhline(0, linestyle='--', color='black', linewidth=1)
plt.xlabel("Step", fontsize=0.8 * FONTSIZE)
plt.tick_params(labelsize=0.8 * FONTSIZE)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=1, fontsize=0.8 * FONTSIZE)

plt.subplot(1, 3, 2)
plt.plot(criterion, lw=LW, label="$\\Delta$ criterion", color="tab:orange", alpha=ALPHA)
plt.axhline(0, linestyle='--', color='black', linewidth=1)
plt.xlabel("Step", fontsize=0.8 * FONTSIZE)
plt.tick_params(labelsize=0.8 * FONTSIZE)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=1, fontsize=0.8 * FONTSIZE)

plt.subplot(1, 3, 3)
plt.plot(criterion, lw=LW, label="Test loss difference", color="tab:red", alpha=ALPHA)
plt.xlabel("Step", fontsize=0.8 * FONTSIZE)
plt.tick_params(labelsize=0.8 * FONTSIZE)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=1, fontsize=0.8 * FONTSIZE)

plt.tight_layout()

filename = os.path.join(FIGURES_DIR, "5_gradient_alignement.pdf")
print(f"[INFO] Plot saved as '{filename}'.")
plt.savefig(filename, dpi=300)
crop_pdf(filename)

####################################################################################################
# Timing
print(f"[INFO] Experiment duration: {format_duration(time.time() - t0)}")

