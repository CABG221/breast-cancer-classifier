"""
src/evaluate.py
===============
Évaluation complète du modèle sur le set de test avec :
  - Matrice de confusion
  - Rapport de classification (precision, recall, F1)
  - Courbe ROC + AUC
  - Analyse des faux positifs / faux négatifs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay,
)

from config import MODEL_PATH, CLASS_NAMES


def get_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les prédictions sur un DataLoader complet.

    Returns:
        all_labels  : vraies étiquettes (int)
        all_preds   : prédictions (int)
        all_probs   : probabilités softmax pour la classe positive (float)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob classe Malignant

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list = CLASS_NAMES,
    save_path: str | None = None,
):
    """Affiche et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion — Set de test", fontsize=13, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Matrice sauvegardée : {save_path}")
    plt.show()


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str | None = None,
) -> float:
    """
    Trace la courbe ROC et retourne l'AUC-ROC.
    L'AUC est la métrique principale en contexte médical.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (TPR)", fontsize=12)
    ax.set_title("Courbe ROC — Classification cancer du sein", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Courbe ROC sauvegardée : {save_path}")
    plt.show()
    return roc_auc


def plot_training_curves(history: dict, save_path: str | None = None):
    """Trace les courbes loss et accuracy train vs val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Courbes de loss
    axes[0].plot(history["train_loss"], label="Train", color="#7F77DD", lw=2)
    axes[0].plot(history["val_loss"],   label="Val",   color="#D85A30", lw=2)
    axes[0].set_title("Loss", fontsize=13)
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Courbes d'accuracy
    axes[1].plot(history["train_acc"], label="Train", color="#7F77DD", lw=2)
    axes[1].plot(history["val_acc"],   label="Val",   color="#D85A30", lw=2)
    axes[1].set_title("Accuracy", fontsize=13)
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Courbes d'apprentissage", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Courbes sauvegardées : {save_path}")
    plt.show()


def print_classification_report(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list = CLASS_NAMES,
):
    """Affiche le rapport sklearn : precision, recall, F1 par classe."""
    print("\n" + "=" * 60)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=class_names, digits=4))


def find_misclassified(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list = CLASS_NAMES,
    n_show: int = 8,
):
    """
    Identifie et affiche les images mal classées avec leur score de confiance.
    Crucial pour l'analyse médicale : faux négatifs (Malignant prédit Benign)
    sont plus dangereux que les faux positifs.
    """
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    misclassified = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs   = F.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1).cpu()

            wrong = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in wrong:
                misclassified.append({
                    "image":      images[idx],
                    "true_label": labels[idx].item(),
                    "pred_label": preds[idx].item(),
                    "confidence": probs[idx, preds[idx]].item(),
                })
            if len(misclassified) >= n_show:
                break

    if not misclassified:
        print("Aucune image mal classée trouvée.")
        return

    n = min(n_show, len(misclassified))
    fig, axes = plt.subplots(2, n // 2, figsize=(16, 7))
    axes = axes.flat

    for ax, item in zip(axes, misclassified[:n]):
        img = item["image"].permute(1, 2, 0).numpy()
        img = (img * std + mean).clip(0, 1)   # dénormalisation
        ax.imshow(img)
        true_name = class_names[item["true_label"]]
        pred_name = class_names[item["pred_label"]]
        color = "#A32D2D" if true_name == "Malignant" else "#0C447C"
        ax.set_title(
            f"Vrai: {true_name}\nPrédit: {pred_name} ({item['confidence']:.1%})",
            fontsize=8, color=color
        )
        ax.axis("off")

    plt.suptitle("Images mal classées (rouge = faux négatif dangereux)", fontsize=13)
    plt.tight_layout()
    plt.show()


def full_evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    history: dict | None = None,
    save_dir: str | None = None,
):
    """
    Lance l'évaluation complète : rapport + confusion + ROC + erreurs.

    Args:
        model:       Modèle chargé en mode eval
        test_loader: DataLoader du set de test
        device:      Device
        history:     Historique d'entraînement (optionnel pour les courbes)
        save_dir:    Dossier où sauvegarder les figures (optionnel)
    """
    sd = Path(save_dir) if save_dir else None

    print("Calcul des prédictions sur le set de test...")
    labels, preds, probs = get_predictions(model, test_loader, device)

    print_classification_report(labels, preds)

    roc_auc = plot_roc_curve(
        labels, probs,
        save_path=str(sd / "roc_curve.png") if sd else None,
    )
    print(f"AUC-ROC : {roc_auc:.4f}")

    plot_confusion_matrix(
        labels, preds,
        save_path=str(sd / "confusion_matrix.png") if sd else None,
    )

    if history:
        plot_training_curves(
            history,
            save_path=str(sd / "training_curves.png") if sd else None,
        )

    find_misclassified(model, test_loader, device)

    return {"auc_roc": roc_auc, "labels": labels, "preds": preds, "probs": probs}
