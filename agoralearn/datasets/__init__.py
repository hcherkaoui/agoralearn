"""Datasets module."""

# Authors: Hamza Cherkaoui

import os
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10, STL10
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo


data_dir = os.path.dirname(os.path.abspath(__file__))
clip_vit_weights_filename = "openai/clip-vit-base-patch32"
cifar_dir = os.path.join(data_dir, "cifar")
stl10_dir = os.path.join(data_dir, "stl10")


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
        (X1, y1), (X2, y2) =_load_wine_dataset()

    elif dataset_name == 'bike':
        (X1, y1), (X2, y2) = _load_seoul_bike_agents_dataset()

    elif dataset_name == 'clip':
        (X1, y1), (X2, y2) = _load_clip_regression_dataset()
    
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
    
    X1, y1 = _load_wine(os.path.join(data_dir, 'winequality', 'winequality-red.csv'))
    X2, y2 = _load_wine(os.path.join(data_dir, 'winequality', 'winequality-white.csv'))

    return (X1, y1), (X2, y2)


def _load_seoul_bike_agents_dataset():
    """
    Load the Seoul Bike Sharing Demand dataset,
    splits into two agent datasets, and returns (X1,y1), (X2,y2).
    """
    dataset = fetch_ucirepo(id=560)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df = df.rename(columns={"Rented Bike Count": "cnt"})

    numeric_df = df.select_dtypes(include=[np.number])

    features = [col for col in numeric_df.columns if col != 'cnt']
    X = numeric_df[features].values.astype(np.float64)
    y = numeric_df['cnt'].values.astype(np.float64)

    mask1 = df['Hour'].isin([7, 8, 9, 17, 18, 19])

    X1, y1 = X[mask1], y[mask1]
    X2, y2 = X[~mask1], y[~mask1]

    return (X1, y1), (X2, y2)


def _load_clip_regression_dataset(n_samples=1000, device="cpu"):
    """
    Load CIFAR10 and STL10 datasets (CLIP features, brightness target).
    """
    model = CLIPModel.from_pretrained(clip_vit_weights_filename).to(device)
    processor = CLIPProcessor.from_pretrained(clip_vit_weights_filename)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    cifar = CIFAR10(root=cifar_dir, train=True, download=True, transform=transform)
    stl = STL10(root=stl10_dir, split='train', download=True, transform=transform)

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
        for i in tqdm(range(0, len(imgs), 8)):
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
