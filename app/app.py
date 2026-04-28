"""
app.py — Interface Streamlit de Classification du Cancer du Sein
================================================================
Interface web connectée à l'API FastAPI pour :
  - Upload d'images histopathologiques
  - Affichage des prédictions (Bénin/Malin) avec design premium
  - Visualisation Grad-CAM pour l'explicabilité
  - Historique des analyses de la session
  - Statistiques de session en temps réel

Auteur : Groupe Projet 1 — Université de Thiès 2025-2026
"""

import os
import io
import base64
import time
from datetime import datetime

import streamlit as st
import requests
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OncoScan AI — Détection Cancer du Sein",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60

# ─────────────────────────────────────────────
# CSS PREMIUM — Dark Medical Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #0a0e1a;
        color: #e2e8f0;
    }

    /* ── Header principal ── */
    .oncoscan-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .oncoscan-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #63b3ed, #90cdf4, #63b3ed, transparent);
    }
    .oncoscan-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .oncoscan-subtitle {
        color: #63b3ed;
        font-size: 0.95rem;
        margin: 0.3rem 0 0;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .oncoscan-badge {
        display: inline-block;
        background: rgba(99,179,237,0.15);
        border: 1px solid rgba(99,179,237,0.3);
        color: #90cdf4;
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-top: 0.8rem;
        letter-spacing: 1px;
    }

    /* ── Cards de résultat ── */
    .result-malignant {
        background: linear-gradient(135deg, rgba(254,178,178,0.08) 0%, rgba(245,101,101,0.05) 100%);
        border: 1px solid rgba(245,101,101,0.4);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-malignant::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #f56565, #fc8181, #f56565);
    }
    .result-benign {
        background: linear-gradient(135deg, rgba(154,230,180,0.08) 0%, rgba(72,187,120,0.05) 100%);
        border: 1px solid rgba(72,187,120,0.4);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-benign::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #48bb78, #68d391, #48bb78);
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin: 0.3rem 0;
    }
    .result-confidence {
        font-family: 'DM Mono', monospace;
        font-size: 2.8rem;
        font-weight: 500;
        margin: 0.2rem 0;
    }
    .result-desc {
        font-size: 0.85rem;
        opacity: 0.7;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: rgba(26,39,68,0.6);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'DM Mono', monospace;
        font-size: 1.4rem;
        font-weight: 500;
        color: #90cdf4;
        display: block;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* ── Prob bars ── */
    .prob-container { margin: 0.4rem 0; }
    .prob-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill-mal {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #f56565, #fc8181);
        transition: width 0.8s ease;
    }
    .prob-bar-fill-ben {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #48bb78, #68d391);
        transition: width 0.8s ease;
    }

    /* ── Warning box ── */
    .warning-box {
        background: rgba(237,137,54,0.08);
        border: 1px solid rgba(237,137,54,0.3);
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        color: #fbd38d;
        line-height: 1.6;
    }

    /* ── Grad-CAM section ── */
    .gradcam-header {
        background: rgba(26,39,68,0.5);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0 1rem;
        font-size: 0.85rem;
        color: #a0aec0;
        line-height: 1.7;
    }

    /* ── Historique ── */
    .history-item {
        background: rgba(26,39,68,0.4);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(26,39,68,0.4);
        border-radius: 10px;
        padding: 0.3rem;
        gap: 0.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #718096;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,179,237,0.15) !important;
        color: #90cdf4 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1420;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: rgba(26,39,68,0.4);
        border: 2px dashed rgba(99,179,237,0.25);
        border-radius: 12px;
        padding: 0.5rem;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99,179,237,0.5);
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2b6cb0, #3182ce);
        border: none;
        border-radius: 10px;
        color: white;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        letter-spacing: 0.3px;
        transition: all 0.2s;
        box-shadow: 0 4px 15px rgba(49,130,206,0.3);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(49,130,206,0.4);
    }

    /* ── Status indicators ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #48bb78; box-shadow: 0 0 6px #48bb78; }
    .status-offline { background: #f56565; box-shadow: 0 0 6px #f56565; }
    .status-warn { background: #ed8936; box-shadow: 0 0 6px #ed8936; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.78rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 2rem;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.3px;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 1rem 0 !important;
    }

    /* ── Sections label ── */
    .section-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #4a5568;
        margin-bottom: 0.6rem;
        font-family: 'DM Mono', monospace;
    }

    /* ── Interprétation box ── */
    .interp-box {
        background: rgba(26,39,68,0.5);
        border-left: 3px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.83rem;
        color: #a0aec0;
        line-height: 1.7;
        margin-top: 0.8rem;
    }
    .interp-box.mal { border-left-color: #f56565; }
    .interp-box.ben { border-left-color: #48bb78; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────
def check_api_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, {}


def call_predict_api(image_bytes, filename, content_type):
    resp = requests.post(
        f"{API_URL}/predict",
        files={"file": (filename, image_bytes, content_type)},
        timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def base64_to_pil(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def prob_bar(label, prob, is_malignant=False):
    color_class = "prob-bar-fill-mal" if is_malignant else "prob-bar-fill-ben"
    color_text  = "#fc8181" if is_malignant else "#68d391"
    pct = prob * 100
    return f"""
    <div class="prob-container">
        <div class="prob-header">
            <span style="color:#a0aec0">{'🔴 Malin' if is_malignant else '🟢 Bénin'}</span>
            <span style="color:{color_text};font-family:'DM Mono',monospace;font-weight:500">
                {pct:.1f}%
            </span>
        </div>
        <div class="prob-bar-bg">
            <div class="{color_class}" style="width:{pct}%"></div>
        </div>
    </div>"""


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="oncoscan-header">
    <div style="display:flex;align-items:center;gap:1rem">
        <div style="font-size:2.5rem">🔬</div>
        <div>
            <p class="oncoscan-title">OncoScan AI</p>
            <p class="oncoscan-subtitle">Système de classification histopathologique du cancer du sein</p>
            <span class="oncoscan-badge">EfficientNet-B3 · Transfer Learning · Grad-CAM XAI</span>
        </div>
        <div style="margin-left:auto;text-align:right">
            <div style="font-size:0.72rem;color:#4a5568;font-family:'DM Mono',monospace;
                        letter-spacing:1px;text-transform:uppercase">Université de Thiès</div>
            <div style="font-size:0.72rem;color:#4a5568;font-family:'DM Mono',monospace">
                UFR SET · MaRT2 · 2025-2026</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">Statut système</p>', unsafe_allow_html=True)

    api_ok, health_data = check_api_health()
    model_ok = health_data.get("model_loaded", False) if api_ok else False

    if api_ok and model_ok:
        st.markdown(
            '<span class="status-dot status-online"></span>'
            '<span style="color:#68d391;font-size:0.85rem">API opérationnelle</span>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<span class="status-dot status-online"></span>'
            '<span style="color:#68d391;font-size:0.85rem">Modèle chargé</span>',
            unsafe_allow_html=True
        )
        device = health_data.get("device", "cpu")
        st.markdown(
            f'<span class="status-dot status-online"></span>'
            f'<span style="color:#68d391;font-size:0.85rem">Device : {device}</span>',
            unsafe_allow_html=True
        )
    elif api_ok and not model_ok:
        st.markdown(
            '<span class="status-dot status-warn"></span>'
            '<span style="color:#fbd38d;font-size:0.85rem">API OK — modèle absent</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span>'
            '<span style="color:#fc8181;font-size:0.85rem">API inaccessible</span>',
            unsafe_allow_html=True
        )
        st.code(f"uvicorn api.main:app\n--port 8000", language="bash")

    st.divider()

    st.markdown('<p class="section-label">Options</p>', unsafe_allow_html=True)
    generate_cam  = st.checkbox("Générer Grad-CAM", value=True)
    show_details  = st.checkbox("Détails techniques", value=False)

    st.divider()

    # Stats session
    st.markdown('<p class="section-label">Session</p>', unsafe_allow_html=True)
    n_total   = len(st.session_state.history)
    n_mal     = sum(1 for h in st.session_state.history if h["prediction"] == "Malignant")
    n_ben     = n_total - n_mal

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <span class="metric-value">{n_total}</span>
            <div class="metric-label">Total</div>
        </div>
        <div class="metric-card">
            <span class="metric-value" style="color:#fc8181">{n_mal}</span>
            <div class="metric-label">Malins</div>
        </div>
        <div class="metric-card">
            <span class="metric-value" style="color:#68d391">{n_ben}</span>
            <div class="metric-label">Bénins</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<p class="section-label" style="margin-top:0.8rem">Dernières analyses</p>',
                    unsafe_allow_html=True)
        for entry in reversed(st.session_state.history[-5:]):
            icon  = "🔴" if entry["prediction"] == "Malignant" else "🟢"
            fname = entry["filename"][:18] + "…" if len(entry["filename"]) > 18 else entry["filename"]
            conf  = entry["confidence"] * 100
            st.markdown(
                f'<div class="history-item">'
                f'<span>{icon} {fname}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;color:#718096;font-size:0.78rem">'
                f'{conf:.0f}%</span></div>',
                unsafe_allow_html=True
            )
        if st.button("Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.markdown('<p class="section-label">Classes</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.82rem;color:#a0aec0;line-height:2">'
        '🟢 <b>Bénin</b> — tissu mammaire sain<br>'
        '🔴 <b>Malin</b> — carcinome infiltrant'
        '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# AVERTISSEMENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="warning-box">
    ⚠️ <strong>Avertissement médical :</strong>
    Cet outil est un <em>prototype académique</em> destiné à l'aide au diagnostic.
    Il ne se substitue en aucun cas au jugement d'un médecin pathologiste qualifié.
    Tout résultat doit être confirmé par un spécialiste.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ZONE PRINCIPALE
# ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown('<p class="section-label">Image histopathologique</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Déposer ou sélectionner une image",
        type=["jpg", "jpeg", "png", "tiff"],
        help="Formats : JPEG, PNG, TIFF — Taille max : 10 Mo",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        # Métadonnées de l'image
        size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="metric-row" style="margin-top:0.6rem">
            <div class="metric-card">
                <span class="metric-value" style="font-size:1rem">{image.size[0]}×{image.size[1]}</span>
                <div class="metric-label">Pixels</div>
            </div>
            <div class="metric-card">
                <span class="metric-value" style="font-size:1rem">{size_kb:.0f} Ko</span>
                <div class="metric-label">Taille</div>
            </div>
            <div class="metric-card">
                <span class="metric-value" style="font-size:1rem">{image.mode}</span>
                <div class="metric-label">Mode</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if not api_ok:
            st.markdown(
                '<div class="warning-box">API non disponible — analyse impossible</div>',
                unsafe_allow_html=True
            )

        if st.button(
            "🧠  Analyser l'image",
            type="primary",
            disabled=not api_ok,
            use_container_width=True
        ):
            with st.spinner("Analyse en cours…"):
                try:
                    t0 = time.time()
                    uploaded_file.seek(0)
                    result = call_predict_api(
                        image_bytes=uploaded_file.read(),
                        filename=uploaded_file.name,
                        content_type=uploaded_file.type or "image/jpeg"
                    )
                    elapsed = (time.time() - t0) * 1000
                    result["_client_ms"] = elapsed

                    st.session_state.last_result = result
                    st.session_state.last_image  = image

                    st.session_state.history.append({
                        "filename":   uploaded_file.name,
                        "prediction": result.get("predicted_class", result.get("prediction", "?")),
                        "confidence": result.get("confidence", 0),
                        "timestamp":  datetime.now().strftime("%H:%M:%S"),
                    })
                    st.rerun()

                except requests.exceptions.Timeout:
                    st.error("⏱️ Délai dépassé — réessayez.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Erreur API {e.response.status_code}")
                    try:
                        st.error(e.response.json().get("detail", str(e)))
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")

    else:
        # Placeholder upload
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;
                    background:rgba(26,39,68,0.3);border-radius:12px;
                    border:2px dashed rgba(99,179,237,0.15)">
            <div style="font-size:3rem;margin-bottom:1rem">🔬</div>
            <div style="color:#4a5568;font-size:0.85rem;line-height:2">
                Déposez une image histopathologique<br>
                JPEG · PNG · TIFF
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RÉSULTATS
# ─────────────────────────────────────────────
with col_result:
    st.markdown('<p class="section-label">Diagnostic IA</p>', unsafe_allow_html=True)

    if "last_result" in st.session_state and "last_image" in st.session_state:
        result = st.session_state.last_result
        pred   = result.get("predicted_class", result.get("prediction", "?"))
        conf   = result.get("confidence", 0)
        probs  = result.get("probabilities", {})

        is_mal = (pred == "Malignant")

        # ── Résultat principal ──────────────────────────────────
        if is_mal:
            st.markdown(f"""
            <div class="result-malignant">
                <div style="color:#fc8181;font-size:0.78rem;letter-spacing:3px;
                            text-transform:uppercase;margin-bottom:0.3rem">Résultat</div>
                <div class="result-label" style="color:#fc8181">MALIN</div>
                <div class="result-confidence" style="color:#ffffff">{conf*100:.1f}%</div>
                <div class="result-desc">Tissu potentiellement cancéreux détecté</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-benign">
                <div style="color:#68d391;font-size:0.78rem;letter-spacing:3px;
                            text-transform:uppercase;margin-bottom:0.3rem">Résultat</div>
                <div class="result-label" style="color:#68d391">BÉNIN</div>
                <div class="result-confidence" style="color:#ffffff">{conf*100:.1f}%</div>
                <div class="result-desc">Tissu mammaire non cancéreux</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Métriques ──────────────────────────────────────────
        proc_ms    = result.get("processing_time_ms", result.get("inference_time_ms", result.get("_client_ms", 0)))
        pred_idx   = result.get("predicted_index", 1 if is_mal else 0)

        st.markdown(f"""
        <div class="metric-row" style="margin-top:0.8rem">
            <div class="metric-card">
                <span class="metric-value">{conf*100:.1f}%</span>
                <div class="metric-label">Confiance</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">{proc_ms:.0f} ms</span>
                <div class="metric-label">Inférence</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">{'CPU' if 'cpu' in str(health_data.get('device','cpu')) else 'GPU'}</span>
                <div class="metric-label">Device</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probabilités ───────────────────────────────────────
        st.markdown('<p class="section-label" style="margin-top:1rem">Probabilités</p>',
                    unsafe_allow_html=True)

        bars_html = ""
        for cls, prob in probs.items():
            bars_html += prob_bar(cls, prob, is_malignant=(cls == "Malignant"))
        st.markdown(bars_html, unsafe_allow_html=True)

        # ── Grad-CAM ───────────────────────────────────────────
        gradcam_b64 = result.get("gradcam_base64", "")
        if gradcam_b64:
            st.markdown('<p class="section-label" style="margin-top:1rem">Explicabilité — Grad-CAM</p>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="gradcam-header">
                La heatmap colorie les régions ayant influencé la décision :
                <span style="color:#f56565">■ rouge</span> = forte activation,
                <span style="color:#63b3ed">■ bleu</span> = faible activation.
            </div>
            """, unsafe_allow_html=True)

            col_orig, col_cam = st.columns(2)
            with col_orig:
                st.image(
                    st.session_state.last_image.resize((224, 224)),
                    caption="Originale (224×224)",
                    use_column_width=True
                )
            with col_cam:
                st.image(
                    base64_to_pil(gradcam_b64),
                    caption="Superposition Grad-CAM",
                    use_column_width=True
                )

            # Bouton téléchargement
            buf = io.BytesIO()
            base64_to_pil(gradcam_b64).save(buf, format="PNG")
            st.download_button(
                "⬇️  Télécharger la heatmap",
                data=buf.getvalue(),
                file_name=f"gradcam_{datetime.now().strftime('%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )

            # Interprétation contextuelle
            if is_mal:
                st.markdown("""
                <div class="interp-box mal">
                    🔬 Les zones rouges correspondent aux régions présentant des anomalies
                    morphologiques — densité cellulaire élevée, mitoses atypiques —
                    caractéristiques d'un tissu malin. <strong>Validation pathologiste requise.</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="interp-box ben">
                    🔬 Les zones activées correspondent aux structures tissulaires saines
                    identifiées par le modèle comme caractéristiques d'un tissu mammaire normal.
                </div>
                """, unsafe_allow_html=True)

        # ── Détails techniques ─────────────────────────────────
        if show_details:
            with st.expander("🔧 Détails techniques JSON"):
                st.json({
                    "prediction":      pred,
                    "confidence":      f"{conf:.6f}",
                    "probabilities":   {k: f"{v:.6f}" for k, v in probs.items()},
                    "inference_ms":    f"{proc_ms:.2f}",
                    "gradcam":         bool(gradcam_b64),
                    "api_endpoint":    f"{API_URL}/predict",
                    "device":          health_data.get("device", "?"),
                })

    else:
        # Placeholder résultats
        st.markdown("""
        <div style="text-align:center;padding:4rem 1rem;
                    background:rgba(26,39,68,0.2);border-radius:12px;
                    border:1px solid rgba(255,255,255,0.04)">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.3">📊</div>
            <div style="color:#4a5568;font-size:0.85rem;line-height:2">
                Le diagnostic apparaîtra ici<br>après l'analyse de l'image
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ONGLETS D'INFORMATION
# ─────────────────────────────────────────────
st.divider()
tab1, tab2, tab3 = st.tabs([
    "🤖 Modèle",
    "📊 Interprétation",
    "🏥 Contexte médical"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### Architecture
        **EfficientNet-B3** pré-entraîné sur ImageNet (14M images), adapté à la classification histopathologique via transfer learning.

        - Backbone : 12M paramètres
        - Tête custom : `Dropout → Linear(1536→2)`
        - Input : 224×224 px, normalisé ImageNet
        """)
    with c2:
        st.markdown("""
        #### Entraînement
        Stratégie en **2 phases** :

        1. **Feature extraction** (époques 1→5) : backbone gelé, seule la tête est entraînée
        2. **Fine-tuning** (époques 6→30) : 3 derniers blocs dégelés, LR réduit ×10

        **Optimiseur** : AdamW · **Scheduler** : CosineAnnealingLR · **Early stopping** : patience 7
        """)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### Lire le résultat

        | Indicateur | Signification |
        |---|---|
        | **MALIN** 🔴 | Anomalies malignes détectées |
        | **BÉNIN** 🟢 | Pas d'anomalie détectée |
        | **Confiance** | Certitude du modèle [0–100%] |
        | **Grad-CAM** | Zones décisives de l'image |
        """)
    with c2:
        st.markdown("""
        #### Métriques clés en oncologie

        - **Sensibilité** : détecter tous les vrais cas malins (minimiser faux négatifs ⚠️)
        - **Spécificité** : identifier correctement les cas bénins
        - **AUC-ROC** : performance globale — ce modèle : **~0.93**
        - **Faux négatif** : cancer non détecté — erreur cliniquement critique
        """)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### Épidémiologie
        Le cancer du sein est le cancer le plus fréquent chez la femme :
        - 2,3 millions de nouveaux cas/an dans le monde
        - 12% de tous les nouveaux cancers
        - L'histopathologie est l'examen de référence pour le diagnostic
        """)
    with c2:
        st.markdown("""
        #### Limitations & éthique
        - Dataset spécifique → généralisation limitée
        - Résolution Grad-CAM : 14×14 upscalée
        - Données de santé → RGPD applicable
        - **Responsabilité médicale** : le médecin reste seul responsable du diagnostic final
        """)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    OncoScan AI — Projet Deep Learning · Université de Thiès · UFR SET · MaRT2 · 2025-2026<br>
    Pr. Cheikh SARR &nbsp;|&nbsp; Groupe A : Cheikh Awa Balla GUEYE & Nafissata THIAM
    &nbsp;|&nbsp; API : <code>{API_URL}</code>
</div>
""", unsafe_allow_html=True)
