"""
src/predict.py
==============
Inférence sur une image unique.
Utilisé par l'API FastAPI pour les prédictions en production.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import MODEL_PATH, CLASS_NAMES_PATH, TRANSFORM_CONFIG_PATH, IMG_SIZE, MEAN, STD
from src.model import load_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transform_config() -> dict:
    """Charge la config de preprocessing depuis le fichier JSON sauvegardé."""
    if TRANSFORM_CONFIG_PATH.exists():
        with open(TRANSFORM_CONFIG_PATH) as f:
            return json.load(f)
    return {"img_size": IMG_SIZE, "mean": MEAN, "std": STD}


def get_inference_transform() -> transforms.Compose:
    """Transformations identiques au val_transform (sans augmentation)."""
    cfg = load_transform_config()
    return transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    class_names: list | None = None,
) -> dict:
    """
    Prédit la classe d'une image PIL.

    Args:
        image:       Image PIL (RGB)
        model:       Modèle chargé
        device:      Device
        class_names: Liste des noms de classes

    Returns:
        dict avec predicted_class, confidence, probabilities
    """
    if class_names is None:
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH) as f:
                class_names = json.load(f)
        else:
            class_names = ["Benign", "Malignant"]

    transform = get_inference_transform()
    image_rgb = image.convert("RGB")
    tensor = transform(image_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1).squeeze(0)

    pred_idx    = probs.argmax().item()
    confidence  = probs[pred_idx].item()
    pred_class  = class_names[pred_idx]

    return {
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence":      round(confidence, 4),
        "probabilities":   {
            cls: round(probs[i].item(), 4)
            for i, cls in enumerate(class_names)
        },
    }


def image_to_base64(image_np: np.ndarray) -> str:
    """Convertit un np.ndarray [H, W, 3] uint8 en string base64 PNG."""
    pil_img = Image.fromarray(image_np)
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict_with_gradcam(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    class_names: list | None = None,
) -> dict:
    """
    Prédit la classe ET génère la heatmap Grad-CAM.
    Retourne un dict complet utilisable directement par l'API.

    Returns:
        {predicted_class, confidence, probabilities, gradcam_base64}
    """
    from src.gradcam import gradcam_from_path
    import tempfile, os

    # Sauvegarde temporaire pour gradcam_from_path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        overlay_uint8, pred_class_idx, confidence = gradcam_from_path(
            tmp_path, model, device
        )
    finally:
        os.unlink(tmp_path)

    if class_names is None:
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH) as f:
                class_names = json.load(f)
        else:
            class_names = ["Benign", "Malignant"]

    # Calcul des probabilités complètes
    transform = get_inference_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1).squeeze(0)

    return {
        "predicted_class":  class_names[pred_class_idx],
        "predicted_index":  pred_class_idx,
        "confidence":       round(confidence, 4),
        "probabilities":    {
            cls: round(probs[i].item(), 4)
            for i, cls in enumerate(class_names)
        },
        "gradcam_base64":   image_to_base64(overlay_uint8),
    }
