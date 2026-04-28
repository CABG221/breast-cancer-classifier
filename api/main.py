"""
api/main.py
===========
API FastAPI pour la prédiction de classification du cancer du sein.

Routes :
    GET  /health   → Statut de l'API et du modèle
    POST /predict  → Prédiction + Grad-CAM sur une image uploadée
"""

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from config import MODEL_PATH, CLASS_NAMES_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Breast Cancer Classifier API",
    description="API de classification d'images histologiques du cancer du sein.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Variables globales — UN SEUL jeu de variables ─────────────
model        = None
device       = None
class_names  = None
model_loaded = False
startup_time = None


@app.on_event("startup")
async def startup_event():
    global model, device, class_names, model_loaded, startup_time
    startup_time = time.time()
    device = torch.device("cpu")
    logger.info(f"Device : {device}")

    # Chargement des noms de classes
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    else:
        class_names = ["Benign", "Malignant"]

    # Chargement du modèle
    try:
        from src.model import build_model
        from config import NUM_CLASSES, DROPOUT

        logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        model = build_model(
            num_classes=checkpoint.get("num_classes", NUM_CLASSES),
            dropout=checkpoint.get("dropout", DROPOUT),
            freeze_backbone=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model_loaded = True

        logger.info(f"Modèle chargé — époque {checkpoint.get('epoch','?')} "
                    f"| val_acc {checkpoint.get('val_acc', 0):.4f}")
        logger.info(f"Classes : {class_names}")

    except Exception as e:
        logger.error(f"Erreur chargement modèle : {e}")
        model_loaded = False


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Classifier API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    uptime = round(time.time() - startup_time, 1) if startup_time else 0
    return {
        "status":       "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path":   str(MODEL_PATH),
        "device":       str(device),
        "class_names":  class_names,
        "uptime_s":     uptime,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Type non supporté : {file.content_type}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {e}")

    try:
        from src.predict import predict_with_gradcam
        t0     = time.time()
        result = predict_with_gradcam(image, model, device, class_names)
        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
        result["filename"]          = file.filename

        logger.info(
            f"{result['predicted_class']} ({result['confidence']:.1%}) "
            f"— {result['inference_time_ms']}ms"
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))