"""
api/main.py
===========
API FastAPI pour la prédiction de classification du cancer du sein.

Routes :
    GET  /health   → Statut de l'API et du modèle
    POST /predict  → Prédiction + Grad-CAM sur une image uploadée

Usage local :
    cd api && uvicorn main:app --reload --port 8000

Usage Docker :
    docker build -t breast-api . && docker run -p 8000:8000 breast-api
"""

import sys
import json
import time
import logging
from pathlib import Path

# Ajout du répertoire racine au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from config import MODEL_PATH, CLASS_NAMES_PATH
from src.model import load_model
from src.predict import predict_with_gradcam, get_device

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Cancer Classifier API",
    description=(
        "API de classification d'images histologiques du cancer du sein. "
        "Retourne la classe prédite (Benign / Malignant), le score de confiance "
        "et une heatmap Grad-CAM en base64."
    ),
    version="1.0.0",
)

# CORS — autorise l'app Streamlit à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE (au démarrage de l'API)
# ─────────────────────────────────────────────────────────────
model = None
device = None
class_names = None
model_loaded = False
startup_time = None


@app.on_event("startup")
async def startup_event():
    global model, device, class_names, model_loaded, startup_time
    startup_time = time.time()
    device = get_device()
    logger.info(f"Device : {device}")

    try:
        model = load_model(MODEL_PATH, device)
        model_loaded = True
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(f"Modèle non trouvé : {MODEL_PATH}. Lance d'abord l'entraînement.")
        model_loaded = False

    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    else:
        class_names = ["Benign", "Malignant"]

    logger.info(f"Classes : {class_names}")


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", summary="Accueil")
def root():
    return {
        "message": "Breast Cancer Classifier API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health", summary="Statut de l'API")
def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    uptime = round(time.time() - startup_time, 1) if startup_time else 0
    return {
        "status":       "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path":   str(MODEL_PATH),
        "device":       str(device),
        "class_names":  class_names,
        "uptime_s":     uptime,
    }


@app.post("/predict", summary="Prédiction + Grad-CAM")
async def predict(file: UploadFile = File(...)):
    """
    Accepte une image (JPG, PNG, BMP) et retourne :
    - `predicted_class`  : "Benign" ou "Malignant"
    - `confidence`       : score de confiance [0, 1]
    - `probabilities`    : probabilités par classe
    - `gradcam_base64`   : heatmap Grad-CAM encodée en base64 (PNG)

    L'image est redimensionnée à 224×224 et normalisée automatiquement.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Vérifiez que l'entraînement a été effectué.",
        )

    # Validation du type de fichier
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Type de fichier non supporté : {file.content_type}. "
                   f"Formats acceptés : JPEG, PNG, BMP, TIFF",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire l'image : {e}")

    try:
        t0     = time.time()
        result = predict_with_gradcam(image, model, device, class_names)
        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
        result["filename"] = file.filename

        logger.info(
            f"Prédiction : {result['predicted_class']} "
            f"({result['confidence']:.1%}) — {result['inference_time_ms']}ms"
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
