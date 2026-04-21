"""
config.py
=========
Fichier de configuration central du projet.
Tous les hyperparamètres, chemins et constantes sont définis ici.
Importer ce fichier dans tous les modules pour garantir la cohérence.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CHEMINS DU PROJET
# ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"
MODELS_DIR  = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ["Benign", "Malignant"]
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # Taille d'entrée d'EfficientNet-B3
# Moyennes et écarts-types ImageNet (standard pour transfer learning)
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-4          # Learning rate initial (AdamW)
WEIGHT_DECAY = 1e-4         # Régularisation L2
DROPOUT     = 0.4           # Dropout avant le classifier final
PATIENCE    = 7             # Early stopping : arrêt si val_loss ne baisse plus

# ─────────────────────────────────────────────────────────────
# ARTEFACTS SAUVEGARDÉS
# ─────────────────────────────────────────────────────────────
MODEL_PATH           = MODELS_DIR / "best_model.pth"
CLASS_NAMES_PATH     = MODELS_DIR / "class_names.json"
TRANSFORM_CONFIG_PATH = MODELS_DIR / "transform_config.json"

# ─────────────────────────────────────────────────────────────
# SPLITS DATASET
# ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
