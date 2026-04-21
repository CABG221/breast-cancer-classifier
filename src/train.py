"""
src/train.py
============
Boucle d'entraînement complète avec :
  - Early stopping sur val_loss
  - Sauvegarde du meilleur modèle
  - Scheduler cosinus pour le learning rate
  - Logging des métriques époque par époque
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    NUM_EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    MODEL_PATH, CLASS_NAMES_PATH, NUM_CLASSES, DROPOUT, SEED,
)
from src.dataset import get_dataloaders, compute_class_weights
from src.model import build_model, unfreeze_last_blocks


# ─────────────────────────────────────────────────────────────
# REPRODUCTIBILITÉ
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# UNE ÉPOQUE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Effectue une passe complète sur le set d'entraînement.
    Returns: (loss_moyenne, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
# ÉVALUATION SUR VAL / TEST
# ─────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Évalue le modèle sans gradient (inférence).
    Returns: (loss_moyenne, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Arrête l'entraînement si la val_loss ne s'améliore pas
    pendant `patience` époques consécutives.
    """
    def __init__(self, patience: int = PATIENCE, delta: float = 1e-4):
        self.patience = patience
        self.delta    = delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# ─────────────────────────────────────────────────────────────
def train(
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    unfreeze_after: int = 5,    # Dégeler backbone après N époques
) -> dict:
    """
    Lance l'entraînement complet du modèle.

    Returns:
        Historique des métriques {train_loss, val_loss, train_acc, val_acc}
    """
    set_seed()

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Données ──────────────────────────────────────────────
    train_loader, val_loader, _, class_names = get_dataloaders()

    # ── Modèle ───────────────────────────────────────────────
    model = build_model(freeze_backbone=True).to(device)
    print(f"Modèle : EfficientNet-B3 | Classes : {class_names}")

    # ── Loss avec pondération des classes ────────────────────
    from src.dataset import compute_class_weights
    from torchvision import datasets
    from src.dataset import get_train_transform
    from config import TRAIN_DIR
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_train_transform())
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiseur (sur les paramètres entraînables seulement) ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    early_stopping = EarlyStopping(patience=PATIENCE)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss = float("inf")

    print(f"\n{'Époque':>6} | {'Train Loss':>10} | {'Val Loss':>9} | "
          f"{'Train Acc':>9} | {'Val Acc':>8} | {'LR':>8} | {'Temps':>6}")
    print("─" * 72)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Dégel du backbone après N époques pour le fine-tuning
        if epoch == unfreeze_after:
            print(f"\n[Époque {epoch}] Fine-tuning : dégel des 3 derniers blocs")
            unfreeze_last_blocks(model, n_blocks=3)
            # Réinitialise l'optimiseur pour inclure les nouveaux paramètres
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr / 10,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - epoch
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | "
              f"{train_acc:>9.4f} | {val_acc:>8.4f} | {current_lr:>8.2e} | {elapsed:>5.1f}s")

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "val_acc":          val_acc,
                "num_classes":      NUM_CLASSES,
                "dropout":          DROPOUT,
                "class_names":      class_names,
            }, MODEL_PATH)
            print(f"  → Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping déclenché à l'époque {epoch}.")
            break

    # Sauvegarde des noms de classes
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nEntraînement terminé. Meilleur modèle : {MODEL_PATH}")
    return history


if __name__ == "__main__":
    train()
