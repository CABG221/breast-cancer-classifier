"""
src/model.py
============
Définition du modèle de classification basé sur EfficientNet-B3
avec transfer learning depuis ImageNet.

Stratégie :
  1. Charger EfficientNet-B3 pré-entraîné (ImageNet)
  2. Geler toutes les couches existantes
  3. Remplacer le classifier final par une tête custom (2 classes)
  4. Option : dégeler les dernières couches pour le fine-tuning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from config import NUM_CLASSES, DROPOUT


def build_model(
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Construit le modèle EfficientNet-B3 adapté à la classification binaire.

    Args:
        num_classes:      Nombre de classes (2 : Benign / Malignant)
        dropout:          Taux de dropout avant la couche de sortie
        freeze_backbone:  Si True, seule la tête est entraînable au départ

    Returns:
        Modèle PyTorch prêt à l'entraînement
    """
    # Chargement des poids ImageNet (transfer learning)
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    # ── Gel du backbone ──────────────────────────────────────
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ── Remplacement du classifier ───────────────────────────
    # EfficientNet-B3 : in_features = 1536
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    # Les paramètres du nouveau classifier sont entraînables par défaut

    return model


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 3) -> None:
    """
    Dégèle les n derniers blocs de features pour le fine-tuning.
    À appeler après quelques époques d'entraînement de la tête seule.

    Args:
        model:    Le modèle EfficientNet
        n_blocks: Nombre de blocs à dégeler (depuis la fin)
    """
    features = list(model.features.children())
    for block in features[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")


def count_parameters(model: nn.Module) -> dict:
    """Retourne le nombre de paramètres total et entraînable."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_model(model_path: str | Path, device: torch.device) -> nn.Module:
    """
    Charge un modèle sauvegardé depuis un fichier .pth.

    Args:
        model_path: Chemin vers le fichier best_model.pth
        device:     Device cible (cpu / cuda)

    Returns:
        Modèle en mode évaluation, prêt pour l'inférence
    """
    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        dropout=checkpoint.get("dropout", DROPOUT),
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Modèle chargé : {model_path}")
    print(f"Époque sauvegardée : {checkpoint.get('epoch', '?')}")
    print(f"Val accuracy : {checkpoint.get('val_acc', '?'):.4f}")

    return model
