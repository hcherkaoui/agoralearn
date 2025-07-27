"""Datasets module."""

# Authors: Hamza Cherkaoui

import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import torch
from torchvision.datasets import CIFAR10, STL10
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_VIT_WEIGHTS_FILENAME = "openai/clip-vit-base-patch32"
CIFAR_DIR = os.path.join(DATA_DIR, 'data', 'cifar')
STL10_DIR = os.path.join(DATA_DIR, 'data', 'stl10')


def fetch_datasets(dataset_name: str):
    """
    Load a dataset by name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    """
    if dataset_name == 'wine':
        (X1, y1), (X2, y2) = _load_wine_dataset()

    elif dataset_name == 'clip':
        (X1, y1), (X2, y2) = _load_clip_regression_dataset()

    elif dataset_name == 'power':
        (X1, y1), (X2, y2) = _load_power_plant_dataset()

    elif dataset_name == 'boston':
        (X1, y1), (X2, y2) = _load_boston_dataset()

    elif dataset_name == 'medical':
        (X1, y1), (X2, y2) = _load_synthea_dataset()

    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    idx1 = np.random.permutation(len(X1))
    X1, y1 = X1[idx1], y1[idx1]

    idx2 = np.random.permutation(len(X2))
    X2, y2 = X2[idx2], y2[idx2]

    return X1.astype(float), X2.astype(float), y1.astype(float), y2.astype(float)


def _load_wine_dataset():
    """
    Load the wine quality dataset from a semicolon-separated CSV file.
    """
    def _load_wine(filepath):
        df = pd.read_csv(filepath, sep=';')
        X = df.drop(columns=['quality']).values.astype(np.float64)
        y = df['quality'].values.astype(np.float64)
        return X, y

    X1, y1 = _load_wine(os.path.join(DATA_DIR, 'data', 'winequality', 'winequality-red.csv'))
    X2, y2 = _load_wine(os.path.join(DATA_DIR, 'data', 'winequality', 'winequality-white.csv'))

    return (X1, y1), (X2, y2)


def _load_power_plant_dataset():
    """
    Load Combined Cycle Power Plant dataset from .xlsx format.
    Returns two (X, y) datasets from a split.
    """
    filepath = os.path.join(DATA_DIR, 'data', 'ccpp', 'Folds5x2_pp.xlsx')
    df = pd.read_excel(filepath)

    df.columns = [col.strip() for col in df.columns]

    X = df[['AT', 'AP', 'RH', 'V']].to_numpy(dtype=np.float64)
    y = df['PE'].to_numpy(dtype=np.float64)

    idx = len(X) // 2
    return (X[:idx], y[:idx]), (X[idx:], y[idx:])


def _load_boston_dataset():
    """
    Load Boston Housing dataset from scikit-learn.
    Returns two splits of (X, y) for collaborative forecasting.
    """
    data = fetch_california_housing()
    X_full = data.data.astype(np.float64)
    y_full = data.target.astype(np.float64)

    n = len(y_full)
    split = n // 2
    return (X_full[:split], y_full[:split]), (X_full[split:], y_full[split:])


def _load_synthea_dataset():
    """
    Load Synthea patient dataset from local CSV file.

    Returns two splits of (X, y) for collaborative modeling.
    X includes: age, is_male.
    y is binary: whether patient died during simulation.
    """
    filepath = os.path.join(DATA_DIR, "data", "synthea", "synthea_patients.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Synthea file not found at: {filepath}")

    patients = pd.read_csv(filepath)

    patients.columns = [col.lower() for col in patients.columns]
    patients['birthdate'] = pd.to_datetime(patients['birthdate'])
    patients['deathdate'] = pd.to_datetime(patients['deathdate'], errors='coerce')
    patients['age'] = 2020 - patients['birthdate'].dt.year
    patients['is_male'] = (patients['gender'] == 'male').astype(float)
    patients['died'] = (~patients['deathdate'].isna()).astype(float)

    X = patients[['age', 'is_male']].to_numpy(dtype=np.float64)
    y = patients['died'].to_numpy(dtype=np.float64)

    n = len(y)
    split = n // 2
    return (X[:split], y[:split]), (X[split:], y[split:])


def _load_clip_regression_dataset(n_samples=1000, device="cpu"):
    """
    Load CIFAR10 and STL10 datasets (CLIP features, brightness target).
    """
    model = CLIPModel.from_pretrained(CLIP_VIT_WEIGHTS_FILENAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_VIT_WEIGHTS_FILENAME)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    cifar = CIFAR10(root=CIFAR_DIR, train=True, download=True, transform=transform)
    stl = STL10(root=STL10_DIR, split='train', download=True, transform=transform)

    def _extract_samples(dataset, n):
        imgs, targets = [], []
        for i in range(n):
            img_tensor, _ = dataset[i]
            brightness = img_tensor.mean().item()
            imgs.append(img_tensor)
            targets.append(brightness)
        return imgs, np.array(targets)

    X1_imgs, y1 = _extract_samples(cifar, n_samples)
    X2_imgs, y2 = _extract_samples(stl, n_samples)

    def _extract_features(imgs):
        features = []
        for i in range(0, len(imgs), 8):
            batch = imgs[i:i + 8]
            pil_batch = [transforms.ToPILImage()(img) for img in batch]
            inputs = processor(images=pil_batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            features.append(feats.cpu())
        return torch.cat(features).numpy()

    X1 = _extract_features(X1_imgs)
    X2 = _extract_features(X2_imgs)

    return (X1, y1), (X2, y2)
