"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
import glob
import pickle
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset
from torchvision import datasets, transforms
from constants import FINE_IDX_TO_COARSE


def set_random_torch_numpy(seed):
    """Fix randomness for the current experiment."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class CIFAR100To10(Dataset):
    """Wrapper to the adapted CIFAR 100 dataset."""
    def __init__(self, base_dataset, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []

        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            if label in FINE_IDX_TO_COARSE:
                self.data.append(img)
                self.targets.append(FINE_IDX_TO_COARSE[label])

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def load_cifar100_to_10(root, train=True, download=True, transform=None):
    """Helper to load the adapted CIFAR 100 dataset."""
    base = datasets.CIFAR100(root=root, train=train, download=download)
    return CIFAR100To10(base_dataset=base, transform=transform)


def load_cifar_10_and_adapted_100(root='./_data', train_size=None, test_size=None, additional_train_size=None,  transform=None, seed=0):
    """Load and return the CIFAR 10 train, the CIFAR 1000 adapted target, CIFAR 10 test."""
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010)),
                                        ])

    train_10_full = datasets.CIFAR10(root=root, train=True,  download=True, transform=transform)
    test_10_full  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    train_100_full = load_cifar100_to_10(root=root, train=True,  download=True, transform=transform)

    train_10  = limit_dataset(train_10_full,  train_size, seed=seed)
    test_10   = limit_dataset(test_10_full,   test_size,  seed=seed)
    train_100 = limit_dataset(train_100_full, additional_train_size, seed=seed)

    return train_10, train_100, test_10


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 10),
        )

    def forward(self, x):
        return self.net(x)


def limit_dataset(ds, n: int | None, seed: int = 0) -> torch.utils.data.Dataset:
    """Return a random Subset of ds with n samples (or ds itself if n is None/invalid)."""
    if n is None or n <= 0 or n >= len(ds):
        return ds

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]

    return Subset(ds, idx.tolist())


def evaluate(loss_fn, model, loader, device='cpu'):
    """Evaluate the 'model' on 'loader' dataset with the 'loss_fn'."""
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * x.size(0)
    model.train()

    return total_loss / len(loader.dataset)


def load_latest_run_per_mode(cache_dir: str, pattern: str = "_6_test_losses*.pkl"):
    """Return a dict: mode -> (path, all_results, meta) using newest file per mode."""
    paths = glob.glob(os.path.join(cache_dir, pattern))
    by_mode = defaultdict(list)

    for p in paths:
        try:
            with open(p, "rb") as f:
                all_results_p, meta_p = pickle.load(f)
            mode = meta_p.get("gradient_mode", "unknown")
            mtime = os.path.getmtime(p)
            by_mode[mode].append((mtime, p, all_results_p, meta_p))
        except Exception as e:
            print(f"[WARN] Skipping '{p}': {e}")

    latest = {}
    for mode, entries in by_mode.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        latest[mode] = entries[0][1:]

    return latest


def compute_gradients(model, loss_fn, x, y):
    """Flattened gradient for a single (x, y) batch."""
    model.zero_grad(set_to_none=True)

    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

    return torch.cat([g.reshape(-1) for g in grads])


def compute_per_sample_gradients(model, loss_fn, x1, y1, x2, y2):
    """Per-sample flattened grads for two batches of equal size."""
    g_s, g_a = [], []
    for i in range(x1.size(0)):
        g_s.append(compute_gradients(model, loss_fn, x1[i:i+1], y1[i:i+1]))
        g_a.append(compute_gradients(model, loss_fn, x2[i:i+1], y2[i:i+1]))

    return torch.stack(g_s, dim=0), torch.stack(g_a, dim=0)


def compute_g_mean(gradient_mode, grads_s, grads_a, crit_t=None):
    """Return the gradient for the descent step following the 'gradient_mode' logic."""
    g_mean_s = grads_s.mean(dim=0)
    g_mean_a = grads_a.mean(dim=0)

    if gradient_mode == "dynamic_additional_grad":
        return g_mean_s + g_mean_a if crit_t is not None and crit_t < 0.0 else g_mean_s
    elif gradient_mode == "additional_grad":
        return g_mean_s + g_mean_a
    elif gradient_mode == "no_additional_grad":
        return g_mean_s
    elif gradient_mode == "only_additional_grad":
        return g_mean_a
    else:
        raise ValueError(f"Unknown gradient mode option: {gradient_mode}")


def compute_grads_and_grads_alignement(model, loss_fn, x1, y1, x2, y2):
    """Compute the gradient cosine similarity."""
    g1 = compute_gradients(model, loss_fn, x1, y1)
    g2 = compute_gradients(model, loss_fn, x2, y2)

    return F.cosine_similarity(g1, g2, dim=0).item()


def grads_alignement(grads_s, grads_a):
    """Compute cosine similarity between two gradients."""
    def flatten_grad_list(grad_list):
        if isinstance(grad_list, (list, tuple)):
            return torch.cat([g.view(-1) for g in grad_list if g is not None])
        elif isinstance(grad_list, torch.Tensor):
            return grad_list.view(-1)
        else:
            raise ValueError("Unsupported gradient format.")

    flat_s = flatten_grad_list(grads_s)
    flat_a = flatten_grad_list(grads_a)

    if flat_s.norm() == 0 or flat_a.norm() == 0:
        return torch.tensor(0.0)

    return F.cosine_similarity(flat_s.unsqueeze(0), flat_a.unsqueeze(0)).item()


def collaboration_criterion(
    grads_s: torch.Tensor,
    grads_a: torch.Tensor,
    eta: float,
    *,
    unbiased: bool = True,
    eps: float = 0.0,
) -> float:
    """
    Δ ≈ -η‖μ‖² + η² μᵀ H μ + (η²/2) Tr(H Σ_a),
    with μ = mean(grads_s), H ≈ diag(Var(grads_s)), Σ_a ≈ Cov(grads_a) (diag).
    Returns Δ (float). Negative => collaboration helpful.
    """
    if grads_s.shape != grads_a.shape or grads_s.ndim != 2:
        raise ValueError("grads_s and grads_a must have shape (N, d) and match.")

    with torch.no_grad():
        mu = grads_s.mean(dim=0)                                # (d,)
        h_diag = grads_s.var(dim=0, unbiased=unbiased) + eps    # (d,)
        sigma_a_diag = grads_a.var(dim=0, unbiased=unbiased)    # (d,)

        first_order  = -eta * (mu * mu).sum()
        second_order = (eta ** 2) * (h_diag * (mu * mu)).sum()
        trace_term   = (h_diag * sigma_a_diag).sum()
        noise_penalty = 0.5 * (eta ** 2) * trace_term
        total = first_order + second_order + noise_penalty

        return float(total.item())

