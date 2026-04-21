"""
src/dataset.py
==============
Chargement du dataset et définition des transformations.
Ce module est utilisé par les notebooks ET par l'API pour garantir
que le preprocessing en production est identique à l'entraînement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, MEAN, STD, BATCH_SIZE, SEED,
)


# ─────────────────────────────────────────────────────────────
# TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """
    Transformations appliquées uniquement sur le split train.
    L'augmentation artificielle augmente la diversité des données
    et réduit l'overfitting.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        # Normalisation ImageNet — obligatoire pour transfer learning
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transform() -> transforms.Compose:
    """
    Transformations appliquées sur val et test.
    PAS d'augmentation — on veut évaluer sur des images "naturelles".
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ─────────────────────────────────────────────────────────────
# GESTION DU DÉSÉQUILIBRE DE CLASSES
# ─────────────────────────────────────────────────────────────

def get_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Crée un sampler qui sur-échantillonne la classe minoritaire.
    Indispensable pour les datasets médicaux souvent déséquilibrés.
    """
    targets = dataset.targets
    class_counts = Counter(targets)
    # Poids inversement proportionnels à la fréquence de chaque classe
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Calcule les poids de classes pour la CrossEntropyLoss.
    Classe rare → poids plus élevé → pénalité plus forte si mal classée.
    """
    targets = dataset.targets
    class_counts = Counter(targets)
    total = len(targets)
    n_classes = len(class_counts)
    # Formule standard : total / (n_classes * count_i)
    weights = [total / (n_classes * class_counts[i]) for i in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Charge les trois splits et retourne les DataLoaders + noms de classes.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Datasets
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_train_transform())
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=get_val_transform())
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=get_val_transform())

    class_names = train_dataset.classes

    # Sampler pour compenser le déséquilibre sur le train
    sampler = get_weighted_sampler(train_dataset) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),   # shuffle=False si sampler défini
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Classes : {class_names}")
    print(f"Train : {len(train_dataset)} images | Val : {len(val_dataset)} | Test : {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names
