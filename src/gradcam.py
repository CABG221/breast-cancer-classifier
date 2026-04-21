"""
src/gradcam.py
==============
Implémentation de Grad-CAM pour visualiser les zones de l'image
qui ont influencé la décision du modèle.

Grad-CAM (Gradient-weighted Class Activation Mapping) :
  1. Passe forward → activation de la dernière couche conv
  2. Passe backward → gradients par rapport à ces activations
  3. Poids = moyenne globale des gradients (GAP)
  4. Heatmap = ReLU(somme pondérée des feature maps)
  5. Superposition sur l'image originale

Dépendance : pip install grad-cam
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from config import CLASS_NAMES, IMG_SIZE, MEAN, STD


# ─────────────────────────────────────────────────────────────
# GRAD-CAM VIA pytorch-grad-cam
# ─────────────────────────────────────────────────────────────

def get_gradcam_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Calcule la heatmap Grad-CAM pour une image.

    Args:
        model:         Modèle EfficientNet en mode eval
        image_tensor:  Image normalisée [1, 3, H, W]
        device:        Device
        target_class:  Classe cible (None = classe prédite)

    Returns:
        heatmap      : np.ndarray [H, W] valeurs [0, 1]
        pred_class   : indice de la classe prédite
        confidence   : score de confiance (softmax)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError("Installe pytorch-grad-cam : pip install grad-cam")

    # Couche cible = dernier bloc de features d'EfficientNet-B3
    target_layers = [model.features[-1]]

    image_tensor = image_tensor.to(device)

    # Prédiction pour connaître la classe
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    if target_class is None:
        target_class = pred_class

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        heatmap = grayscale_cam[0]   # [H, W] float32 [0, 1]

    return heatmap, pred_class, confidence


def overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superpose la heatmap Grad-CAM sur l'image originale.

    Args:
        image_np: Image RGB normalisée [0, 1] float32 (H, W, 3)
        heatmap:  Heatmap [H, W] float32 [0, 1]
        alpha:    Opacité de la heatmap (0=invisible, 1=opaque)
        colormap: Colormap OpenCV (JET par défaut)

    Returns:
        Image superposée RGB [0, 1] float32
    """
    # Redimensionne la heatmap à la taille de l'image
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    overlay = alpha * heatmap_rgb + (1 - alpha) * image_np
    return np.clip(overlay, 0, 1)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Dénormalise un tenseur image pour l'affichage.
    Inverse la normalisation ImageNet.
    """
    mean = np.array(MEAN)
    std  = np.array(STD)
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1).astype(np.float32)


def visualize_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    class_names: list = CLASS_NAMES,
    title: str = "",
    save_path: str | None = None,
) -> np.ndarray:
    """
    Affiche côte à côte : image originale | heatmap seule | superposition.

    Returns:
        overlay : image superposée np.ndarray [H, W, 3]
    """
    heatmap, pred_class, confidence = get_gradcam_heatmap(
        model, image_tensor, device
    )

    image_np = denormalize(image_tensor)
    overlay  = overlay_heatmap(image_np, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Image originale", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap Grad-CAM", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    pred_name = class_names[pred_class]
    color = "#A32D2D" if pred_name == "Malignant" else "#085041"
    axes[2].set_title(
        f"Superposition\nPrédit : {pred_name} ({confidence:.1%})",
        fontsize=11, color=color,
    )
    axes[2].axis("off")

    if title:
        plt.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM sauvegardé : {save_path}")
    plt.show()

    return overlay


def batch_gradcam(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list = CLASS_NAMES,
    n_samples: int = 6,
    save_path: str | None = None,
):
    """
    Affiche une grille Grad-CAM sur N échantillons du set de test.
    Montre des exemples de chaque classe (Benign + Malignant).
    """
    model.eval()
    mean = np.array(MEAN)
    std  = np.array(STD)

    samples = []
    seen_classes = set()

    for images, labels in test_loader:
        for i in range(len(labels)):
            cls = labels[i].item()
            if cls not in seen_classes or len(samples) < n_samples:
                samples.append((images[i:i+1], labels[i].item()))
                seen_classes.add(cls)
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    cols = 3   # colonnes : original | heatmap | overlay
    rows = len(samples)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    for row, (img_tensor, true_label) in enumerate(samples):
        heatmap, pred_class, confidence = get_gradcam_heatmap(
            model, img_tensor.to(device), device
        )
        image_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        image_np = (image_np * std + mean).clip(0, 1).astype(np.float32)
        overlay  = overlay_heatmap(image_np, heatmap)

        true_name = class_names[true_label]
        pred_name = class_names[pred_class]
        ok = "✓" if true_label == pred_class else "✗"
        color = "#085041" if true_label == pred_class else "#A32D2D"

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"Vrai : {true_name}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title("Grad-CAM", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"{ok} Prédit : {pred_name} ({confidence:.1%})",
            fontsize=9, color=color,
        )
        axes[row, 2].axis("off")

    plt.suptitle(
        "Analyse Grad-CAM — Zones décisives pour la classification",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grille Grad-CAM sauvegardée : {save_path}")
    plt.show()


def gradcam_from_path(
    image_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    """
    Charge une image depuis son chemin et calcule le Grad-CAM.
    Utilisé par l'API pour générer la heatmap en production.

    Returns:
        overlay_uint8 : image superposée [H, W, 3] uint8 (pour PIL/base64)
        pred_class    : indice de classe
        confidence    : score de confiance
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    heatmap, pred_class, confidence = get_gradcam_heatmap(
        model, img_tensor, device
    )

    img_np  = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    overlay = overlay_heatmap(img_np.astype(np.float32), heatmap)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    return overlay_uint8, pred_class, confidence
