"""Collaboration criterions illustration."""

# Authors: Hamza Cherkaoui

import os
from torchvision.datasets import CIFAR100


FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

CACHE_DIR = '_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

FONTSIZE = 18
LW = 3.5
ALPHA = 0.4

L_LR = [5e-3, 1e-2]
L_BATCH_SIZE = [128, 256]

GRAD_MODES = ["no_additional_grad", "only_additional_grad", "additional_grad", "dynamic_additional_grad"]

MODE_STYLE = {
    "no_additional_grad": dict(color="tab:blue", linestyle="-", label="no_additional_grad"),
    "additional_grad": dict(color="tab:green", linestyle="-", label="additional_grad"),
    "dynamic_additional_grad": dict(color="tab:orange", linestyle="-", label="dynamic_additional_grad"),
}

COARSE_TO_FINE = {
    0: ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    1: ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    2: ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
    3: ['bottles', 'bowls', 'cans', 'cups', 'plates'],
    4: ['apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers'],
    5: ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
    6: ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    7: ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    8: ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    9: ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    # 10-19 omitted (not in CIFAR-10)
}

FINE_TO_COARSE = {}
for coarse_idx, fine_names in COARSE_TO_FINE.items():
    for name in fine_names:
        FINE_TO_COARSE[name] = coarse_idx

ALL_FINE_LABELS = CIFAR100(root='./_data', train=True, download=True).classes

FINE_IDX_TO_COARSE = {
    i: FINE_TO_COARSE[name]
    for i, name in enumerate(ALL_FINE_LABELS)
    if name in FINE_TO_COARSE
}

