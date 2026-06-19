# =============================================================================
# TARV-Score — Application de scoring clinique | Bilingual FR/EN
# Mémoire ISE3 | ISSEA-CEMAC 2025-2026 | CNLS / GTC / Cameroun
# Auteur : AZONFACK MYRIAM DOLVIANNE
# Modèle : XGBoost — Rappel 83.8 % | AUC-ROC 0.704
# =============================================================================

import base64
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from fpdf import FPDF

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TARV-Score | CNLS Cameroun",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# TRADUCTIONS FR / EN
# ─────────────────────────────────────────────────────────────────────────────
T = {
    # ── Sidebar ──────────────────────────────────────────────────────────────
    "lang_label":    {"fr": "🌐 Langue / Language", "en": "🌐 Language / Langue"},
    "lang_fr":       {"fr": "🇫🇷 Français",          "en": "🇫🇷 French"},
    "lang_en":       {"fr": "🇬🇧 Anglais",           "en": "🇬🇧 English"},

    "sidebar_title": {"fr": "🎗️ TARV-Score",         "en": "🎗️ TARV-Score"},
    "sidebar_sub":   {"fr": "CNLS Cameroun · ISSEA-CEMAC 2025-2026",
                      "en": "CNLS Cameroon · ISSEA-CEMAC 2025-2026"},
    "perf_title":    {"fr": "📊 Performances du modèle",
                      "en": "📊 Model Performance"},
    "recall_lbl":    {"fr": "Rappel",                "en": "Recall"},
    "algo_lbl":      {"fr": "Algorithme",            "en": "Algorithm"},
    "seuil_lbl":     {"fr": "Seuil",                 "en": "Threshold"},
    "smote_lbl":     {"fr": "SMOTE ✓",               "en": "SMOTE ✓"},
    "train_lbl":     {"fr": "Entraînement",          "en": "Training"},
    "test_lbl":      {"fr": "Test",                  "en": "Test"},

    "grid_title":    {"fr": "🚦 Grille d'interprétation",
                      "en": "🚦 Risk Interpretation Grid"},
    "risk_low":      {"fr": "Faible",                "en": "Low"},
    "risk_mod":      {"fr": "Modéré",                "en": "Moderate"},
    "risk_high":     {"fr": "Élevé",                 "en": "High"},
    "risk_low_desc": {"fr": "probabilité < 30 %",    "en": "probability < 30%"},
    "risk_mod_desc": {"fr": "probabilité 30 % – 50 %", "en": "probability 30%–50%"},
    "risk_high_desc":{"fr": "probabilité ≥ 50 %",   "en": "probability ≥ 50%"},

    "ctx_title":     {"fr": "ℹ️ Contexte de l'étude",
                      "en": "ℹ️ Study Context"},
    "ctx_body":      {
        "fr": "Données CNLS 2024 · Enquête nationale auprès de **2 720 PvVIH** dans les 10 régions du Cameroun.\n\n4 modèles comparés : Régression Logistique, Random Forest, SVM, **XGBoost** (retenu).",
        "en": "CNLS 2024 data · National survey of **2,720 PLHIV** across Cameroon's 10 regions.\n\n4 models compared: Logistic Regression, Random Forest, SVM, **XGBoost** (selected).",
    },
    "disclaimer":    {
        "fr": "⚠️ Outil d'aide à la décision.<br>Ne remplace pas le jugement clinique.",
        "en": "⚠️ Decision-support tool.<br>Does not replace clinical judgment.",
    },

    # ── En-tête ──────────────────────────────────────────────────────────────
    "main_title":    {
        "fr": "OUTIL DE DÉTECTION DU RISQUE D'INTERRUPTION AU TARV<br>CHEZ LES PVVIH AU CAMEROUN",
        "en": "RISK OF ART INTERRUPTION DETECTION TOOL<br>AMONG PLHIV IN CAMEROON",
    },
    "main_sub":      {
        "fr": "Prédiction du risque d'interruption du Traitement Antirétroviral",
        "en": "Prediction of the risk of Antiretroviral Treatment interruption",
    },
    "badge_cnls":    {"fr": "CNLS Cameroun",         "en": "NACC Cameroon"},
    "badge_xgb":     {"fr": "XGBoost Champion",      "en": "XGBoost Champion"},
    "badge_recall":  {"fr": "Rappel : 83,8 %",       "en": "Recall: 83.8%"},
    "badge_auc":     {"fr": "AUC-ROC : 0,704",       "en": "AUC-ROC: 0.704"},

    # ── Formulaire ───────────────────────────────────────────────────────────
    "form_intro":    {
        "fr": "📋 Profil du patient — 12 variables du modèle XGBoost",
        "en": "📋 Patient Profile — 12 variables of the XGBoost model",
    },
    "form_sub":      {
        "fr": "Renseignez toutes les caractéristiques, puis cliquez sur <strong>Calculer le Score</strong>.",
        "en": "Fill in all characteristics, then click <strong>Calculate Score</strong>.",
    },
    "sec_loc":       {"fr": "📍 Localisation & Structure",
                      "en": "📍 Location & Facility"},
    "sec_socio":     {"fr": "👤 Profil socio-démographique",
                      "en": "👤 Socio-demographic Profile"},
    "sec_thera":     {"fr": "💊 Suivi thérapeutique",
                      "en": "💊 Therapeutic Follow-up"},

    # Variables
    "region":        {"fr": "Région",                "en": "Region"},
    "type_fosa":     {"fr": "Type de FOSA",          "en": "Health Facility Type"},
    "pepfar":        {"fr": "Soutien PEPFAR",        "en": "PEPFAR Support"},
    "milieu":        {"fr": "Milieu de résidence",   "en": "Residence Setting"},
    "sexe":          {"fr": "Sexe",                  "en": "Sex"},
    "tranche_age":   {"fr": "Tranche d'âge",         "en": "Age Group"},
    "niveau_etude":  {"fr": "Niveau d'étude",        "en": "Education Level"},
    "statut_mat":    {"fr": "Statut matrimonial",    "en": "Marital Status"},
    "activite":      {"fr": "Activité rémunérée",    "en": "Paid Activity"},
    "observance":    {"fr": "Observance (4 derniers jours)",
                      "en": "Adherence (last 4 days)"},
    "dsd":           {"fr": "Mode de dispensation (DSD)",
                      "en": "Dispensation Mode (DSD)"},
    "protocole":     {"fr": "Protocole ARV",         "en": "ART Protocol"},

    "oui":           {"fr": "Oui",                   "en": "Yes"},
    "non":           {"fr": "Non",                   "en": "No"},
    "urbain":        {"fr": "Urbain",                "en": "Urban"},
    "rural":         {"fr": "Rural",                 "en": "Rural"},
    "masculin":      {"fr": "Masculin",              "en": "Male"},
    "feminin":       {"fr": "Féminin",               "en": "Female"},

    "btn_calc":      {"fr": "🔍  Calculer le Score de Risque",
                      "en": "🔍  Calculate Risk Score"},
    "spinner":       {"fr": "Calcul du score en cours…",
                      "en": "Computing score…"},

    # ── Résultats ────────────────────────────────────────────────────────────
    "res_divider":   {"fr": "📊 Résultat du Scoring Clinique",
                      "en": "📊 Clinical Scoring Result"},
    "prob_label":    {"fr": "probabilité d'interruption",
                      "en": "interruption probability"},
    "seuil_txt":     {"fr": "Seuil de décision",     "en": "Decision threshold"},
    "reco_lbl":      {"fr": "Recommandation clinique :",
                      "en": "Clinical Recommendation:"},
    "risk_low_lbl":  {"fr": "FAIBLE",                "en": "LOW"},
    "risk_mod_lbl":  {"fr": "MODÉRÉ",                "en": "MODERATE"},
    "risk_high_lbl": {"fr": "ÉLEVÉ",                 "en": "HIGH"},

    "reco_low": {
        "fr": ("Profil stable. Maintenir le suivi standard. Renforcer les messages "
               "d'observance lors de la prochaine dispensation."),
        "en": ("Stable profile. Maintain standard follow-up. Reinforce adherence "
               "counselling at next dispensation visit."),
    },
    "reco_mod": {
        "fr": ("Suivi renforcé recommandé. Identifier les freins à l'observance "
               "(transport, soutien familial, effets indésirables) et proposer "
               "un accompagnement individualisé dès cette consultation."),
        "en": ("Enhanced follow-up recommended. Identify barriers to adherence "
               "(transport, family support, side effects) and offer individualised "
               "support starting from this visit."),
    },
    "reco_high": {
        "fr": ("Intervention prioritaire requise. Déclencher un plan de rétention : "
               "contact téléphonique dans les 48 h, visite à domicile (VAD), "
               "orientation vers un accompagnement psychosocial et révision "
               "du mode de dispensation (DSD) si possible."),
        "en": ("Priority intervention required. Launch a retention plan: "
               "phone contact within 48 h, home visit (VAD), referral for "
               "psychosocial support, and review of the dispensation mode (DSD) if possible."),
    },

    "imp_sub":       {
        "fr": "Importances globales du modèle XGBoost — plus la barre est longue, plus la variable pèse dans la prédiction.",
        "en": "Global XGBoost feature importances — longer bar = more weight in the prediction.",
    },
    "imp_title":     {"fr": "🔬 Top {} facteurs les plus contributifs",
                      "en": "🔬 Top {} most contributing factors"},
    "imp_xlabel":    {"fr": "Importance XGBoost (feature_importances_)",
                      "en": "XGBoost Importance (feature_importances_)"},
    "imp_legend":    {
        "fr": "🔴 Importance forte &nbsp;|&nbsp; 🟠 Modérée &nbsp;|&nbsp; 🟢 Modérée-faible &nbsp;|&nbsp; 🔵 Faible",
        "en": "🔴 High importance &nbsp;|&nbsp; 🟠 Moderate &nbsp;|&nbsp; 🟢 Moderate-low &nbsp;|&nbsp; 🔵 Low",
    },
    "recap_title":   {"fr": "🗂️ Récapitulatif du profil saisi",
                      "en": "🗂️ Summary of entered profile"},
    "recap_var":     {"fr": "Variable",              "en": "Variable"},
    "recap_val":     {"fr": "Valeur",                "en": "Value"},

    # ── Pied de page ─────────────────────────────────────────────────────────
    "footer": {
        "fr": ("🎗️ <strong>TARV-Score</strong> — Mémoire ISE3 ISSEA-CEMAC 2025-2026 &nbsp;|&nbsp; "
               "CNLS / GTC Cameroun &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Rappel : 83,8 % &nbsp;·&nbsp; "
               "AUC-ROC : 0,704 &nbsp;·&nbsp; N entraînement : 3 172 &nbsp;·&nbsp; N test : 544<br>"
               "<em>⚠️ Outil d'aide à la décision — Ne remplace pas le jugement clinique du prestataire de santé.</em>"),
        "en": ("🎗️ <strong>TARV-Score</strong> — ISE3 Thesis ISSEA-CEMAC 2025-2026 &nbsp;|&nbsp; "
               "NACC / GTC Cameroon &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Recall: 83.8% &nbsp;·&nbsp; "
               "AUC-ROC: 0.704 &nbsp;·&nbsp; Train N: 3,172 &nbsp;·&nbsp; Test N: 544<br>"
               "<em>⚠️ Decision-support tool — Does not replace the clinical judgment of the healthcare provider.</em>"),
    },

    # ── Options variables ─────────────────────────────────────────────────────
    "tranche_age_opts": {
        "fr": ["25 à 49 Ans","18 à 20 Ans","21 à 24 Ans","50 ans et plus"],
        "en": ["25 to 49 years","18 to 20 years","21 to 24 years","50 years and above"],
    },
    "niveau_etude_opts": {
        "fr": ["Primaire","Jamais fréquenté","Secondaire Premier Cycle",
               "Secondaire Second Cycle","Supérieur"],
        "en": ["Primary","Never attended school","Lower Secondary",
               "Upper Secondary","Higher Education"],
    },
    "statut_mat_opts": {
        "fr": ["Marié(e) en monogamie","Célibataire","En union libre/concubinage",
               "Marié(e) en polygamie","En séparation de corps / Divorcée","Veuf (ve)"],
        "en": ["Married (monogamous)","Single","Cohabiting/Common-law",
               "Married (polygamous)","Separated / Divorced","Widowed"],
    },
    "type_fosa_opts": {
        "fr": ["Public","Privé confessionnel","Privé laïc"],
        "en": ["Public","Faith-based Private","Secular Private"],
    },
    "observance_opts": {
        "fr": ["Bonne","Modérée","Médiocre"],
        "en": ["Good","Moderate","Poor"],
    },
    "dsd_opts": {
        "fr": ["Standard (Suivi classique)","Dispensation communautaire / VAD"],
        "en": ["Standard (Regular follow-up)","Community dispensation / Home visit"],
    },
    "protocole_opts": {
        "fr": ["TDF+3TC+DTG (TLD)","TDF+3TC+EFV (TELE/TLE)",
               "Protocoles avec IP (ATV/r)","Autre / Non spécifié"],
        "en": ["TDF+3TC+DTG (TLD)","TDF+3TC+EFV (TELE/TLE)",
               "PI-based regimens (ATV/r)","Other / Not specified"],
    },
    "region_opts": {
        "fr": ["Centre","Adamaoua","Est","Extrême-Nord","Littoral",
               "Nord","Nord-Ouest","Ouest","Sud","Sud-Ouest"],
        "en": ["Centre","Adamawa","East","Far North","Littoral",
               "North","North-West","West","South","South-West"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #eef2f7; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2d52 0%, #1a5e8a 55%, #0d7a5c 100%);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #ddeeff; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }
section[data-testid="stSidebar"] [data-testid="metric-container"] {
    background: rgba(255,255,255,0.12); border-radius: 8px; padding: 8px 12px;
}
section[data-testid="stSidebar"] [data-testid="metric-container"] label { color: #a8d0f0 !important; }
section[data-testid="stSidebar"] [data-testid="stMetricValue"]
    { color: #fff !important; font-size: 1.3em !important; font-weight: 700 !important; }

div[data-testid="stForm"] button[kind="primaryFormSubmit"],
button[kind="primary"] {
    background: linear-gradient(135deg, #0b2d52 0%, #1a6fa8 100%) !important;
    border: none !important; color: #fff !important;
    font-weight: 700 !important; font-size: 1.05em !important;
    border-radius: 10px !important; padding: 14px !important;
    box-shadow: 0 4px 16px rgba(11,45,82,0.35) !important;
    letter-spacing: 0.3px !important; transition: all 0.2s !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(11,45,82,0.45) !important;
}
.stSelectbox label, .stRadio label {
    font-size: 0.87em !important; font-weight: 600 !important; color: #1a2d4a !important;
}
.res-divider {
    display: flex; align-items: center; gap: 14px; margin: 24px 0 18px 0;
}
.res-divider hr { flex: 1; border: none; border-top: 2px solid #ccd8e8; }
.res-divider span {
    background: linear-gradient(135deg, #0b2d52, #1a6fa8);
    color: #fff; border-radius: 25px; padding: 5px 22px;
    font-size: 0.86em; font-weight: 700; white-space: nowrap;
    box-shadow: 0 3px 10px rgba(11,45,82,0.25);
}
.footer {
    text-align: center; font-size: 0.75em; color: #8a9bb0;
    margin-top: 30px; padding: 14px 0;
    border-top: 1px solid #ccd8e8; line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS & RESSOURCES
# ─────────────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
ASSETS  = BASE / "assets"
MODELES = BASE / "modeles"


@st.cache_resource
def load_resources():
    model    = joblib.load(MODELES / "XGBoost_v1.pkl")
    scaler   = joblib.load(MODELES / "scaler_prepro.pkl")
    colonnes = joblib.load(MODELES / "colonnes_train.pkl")
    with open(MODELES / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, colonnes, meta


def img_b64(path: Path) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# SÉLECTEUR DE LANGUE (sidebar, en premier)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {T['lang_label']['fr']}")
    lang_choice = st.radio(
        label="",
        options=["fr", "en"],
        format_func=lambda x: T["lang_fr"][x] if x == "fr" else T["lang_en"][x],
        horizontal=True,
        key="lang",
        label_visibility="collapsed",
    )

L = lang_choice  # "fr" ou "en"


def t(key: str) -> str:
    return T[key][L]


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, scaler, colonnes_train, meta = load_resources()
except Exception as exc:
    st.error(f"❌ {'Impossible de charger les ressources' if L=='fr' else 'Cannot load resources'}: {exc}")
    st.stop()

SEUIL = meta.get("seuil", 0.50)

# ─────────────────────────────────────────────────────────────────────────────
# RESTE DE LA SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {t('sidebar_title')}")
    st.caption(t("sidebar_sub"))
    st.divider()

    st.markdown(f"### {t('perf_title')}")
    c1, c2 = st.columns(2)
    c1.metric(t("recall_lbl"),  "83,8 %" if L == "fr" else "83.8%")
    c2.metric("AUC-ROC",        "0,704"  if L == "fr" else "0.704")
    c1.metric(t("train_lbl"),   "3 172")
    c2.metric(t("test_lbl"),    "544")
    st.caption(f"{t('algo_lbl')} : XGBoost  |  {t('seuil_lbl')} : {SEUIL:.0%}  |  {t('smote_lbl')}")

    st.divider()
    st.markdown(f"### {t('grid_title')}")
    grid_intro = ("Le score est interprété selon 3 niveaux de risque :"
                  if L == "fr" else
                  "The score is interpreted across 3 risk levels:")
    st.markdown(f'<p style="font-size:0.82em;opacity:0.8;margin-bottom:8px;">{grid_intro}</p>',
                unsafe_allow_html=True)
    for emoji, key_lbl, key_desc, border_col, bg_col, txt_col in [
        ("🟢", "risk_low",  "risk_low_desc",  "#27ae60", "rgba(39,174,96,0.18)",  "#90ffb0"),
        ("🟠", "risk_mod",  "risk_mod_desc",  "#f39c12", "rgba(243,156,18,0.18)", "#ffd580"),
        ("🔴", "risk_high", "risk_high_desc", "#e74c3c", "rgba(231,76,60,0.18)",  "#ffaaaa"),
    ]:
        st.markdown(
            f'<div style="border-left:4px solid {border_col};background:{bg_col};'
            f'border-radius:0 8px 8px 0;padding:9px 12px;margin-bottom:7px;">'
            f'<div style="color:{txt_col};font-weight:700;font-size:0.9em;">{emoji} {t(key_lbl)}</div>'
            f'<div style="color:rgba(255,255,255,0.75);font-size:0.8em;margin-top:2px;">{t(key_desc)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(f"### {t('ctx_title')}")
    st.markdown(t("ctx_body"))
    st.divider()
    st.markdown(
        f'<div style="font-size:0.74em;opacity:0.65;text-align:center;line-height:1.7;">'
        f'{t("disclaimer")}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
cnls_b64  = img_b64(ASSETS / "cnls_logo.png")
issea_b64 = img_b64(ASSETS / "issea_logo.png")

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0b2d52 0%, #1a5e8a 50%, #0d7a5c 100%);
    border-radius: 18px; padding: 22px 36px; margin-bottom: 26px;
    box-shadow: 0 10px 36px rgba(11,45,82,0.30);
    display: flex; align-items: center; justify-content: space-between; gap: 20px;">
    <img src="data:image/png;base64,{cnls_b64}" height="130"
         style="border-radius:12px; box-shadow:0 6px 20px rgba(0,0,0,0.40); flex-shrink:0; object-fit:contain;">
    <div style="text-align:center; color:#fff; flex:1;">
        <div style="font-size:3.8em; line-height:1; margin-bottom:8px;">
            <span style="color:#ff2222; filter:drop-shadow(0 0 10px #ff000099) drop-shadow(0 0 4px #ff0000cc);">🎗️</span>
        </div>
        <div style="font-size:1.22em; font-weight:900; letter-spacing:1px; text-shadow:0 2px 8px rgba(0,0,0,0.35); line-height:1.3; text-transform:uppercase;">{t("main_title")}</div>
        <div style="font-size:0.88em; font-weight:300; opacity:0.88; margin-top:9px;">{t("main_sub")}</div>
        <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:13px;">
            <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);border-radius:20px;padding:3px 14px;font-size:0.78em;font-weight:600;">{t("badge_cnls")}</span>
            <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);border-radius:20px;padding:3px 14px;font-size:0.78em;font-weight:600;">{t("badge_xgb")}</span>
            <span style="background:rgba(39,174,96,0.35);border:1px solid rgba(39,174,96,0.6);border-radius:20px;padding:3px 14px;font-size:0.78em;font-weight:600;">{t("badge_recall")}</span>
            <span style="background:rgba(52,152,219,0.35);border:1px solid rgba(52,152,219,0.6);border-radius:20px;padding:3px 14px;font-size:0.78em;font-weight:600;">{t("badge_auc")}</span>
        </div>
    </div>
    <img src="data:image/png;base64,{issea_b64}" height="130"
         style="border-radius:12px; box-shadow:0 6px 20px rgba(0,0,0,0.40); flex-shrink:0; object-fit:contain;">
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAPPINGS INTERNES (valeurs d'entraînement — NE PAS MODIFIER)
# ─────────────────────────────────────────────────────────────────────────────
FOSA_INTERNAL = {
    "Public": "Public",
    "Privé confessionnel": "Privé confessionnel",
    "Privé laïc": "Privé laic",            # pas de ï dans les données
    "Faith-based Private": "Privé confessionnel",
    "Secular Private": "Privé laic",
}
OBSERVANCE_INTERNAL = {
    "Bonne": "Bonne",  "Modérée": "Modérée",  "Médiocre": "Mediocre",  # pas de é
    "Good": "Bonne",   "Moderate": "Modérée",  "Poor": "Mediocre",
}

REGION_EN2FR  = dict(zip(T["region_opts"]["en"],       T["region_opts"]["fr"]))
AGE_EN2FR     = dict(zip(T["tranche_age_opts"]["en"],  T["tranche_age_opts"]["fr"]))
ETUDE_EN2FR   = dict(zip(T["niveau_etude_opts"]["en"], T["niveau_etude_opts"]["fr"]))
STATUT_EN2FR  = dict(zip(T["statut_mat_opts"]["en"],   T["statut_mat_opts"]["fr"]))
DSD_EN2FR     = dict(zip(T["dsd_opts"]["en"],          T["dsd_opts"]["fr"]))
PROTO_EN2FR   = dict(zip(T["protocole_opts"]["en"],    T["protocole_opts"]["fr"]))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE (identique au notebook)
# ─────────────────────────────────────────────────────────────────────────────
def get_dummies_fixed(df: pd.DataFrame) -> pd.DataFrame:
    cats = ["Tranche_Age","Sexe","Niveau_Etude","Statut_Matrimonial",
            "Region","Type_FOSA","Soutien_PEPFAR","Milieu_Residence",
            "Observance_4j","Activite_Remuneree","DSD_Recode","Protocole_Recode"]
    df_dum = pd.get_dummies(df, columns=cats, dtype=int)
    refs   = ["Tranche_Age_25 à 49 Ans","Sexe_Féminin","Niveau_Etude_Primaire",
              "Statut_Matrimonial_Marié(e) en monogamie","Region_Centre",
              "Type_FOSA_Public","Soutien_PEPFAR_Non","Milieu_Residence_Urbain",
              "Observance_4j_Bonne","Activite_Remuneree_Non",
              "DSD_Recode_Standard (Suivi classique)","Protocole_Recode_TDF+3TC+DTG (TLD)"]
    return df_dum.drop(columns=[c for c in refs if c in df_dum.columns])


def predict(raw_fr: dict) -> float:
    df     = pd.DataFrame([raw_fr])
    dum    = get_dummies_fixed(df).reindex(columns=colonnes_train, fill_value=0)
    scaled = pd.DataFrame(scaler.transform(dum), columns=colonnes_train)
    return float(model.predict_proba(scaled)[0][1])


# ─────────────────────────────────────────────────────────────────────────────
# GAUGE SVG SEMI-CIRCULAIRE
# ─────────────────────────────────────────────────────────────────────────────
def svg_gauge(prob: float, color: str) -> str:
    W, H   = 360, 218
    cx, cy = 180, 182
    R, r   = 138, 98

    def pt(deg, radius):
        rad = math.radians(deg)
        return cx + radius * math.cos(rad), cy - radius * math.sin(rad)

    def ring(d1, d2, fill, stroke):
        ox1, oy1 = pt(d1, R); ox2, oy2 = pt(d2, R)
        ix1, iy1 = pt(d1, r); ix2, iy2 = pt(d2, r)
        la = 1 if abs(d1 - d2) > 180 else 0
        d = (f"M{ox1:.2f} {oy1:.2f} A{R} {R} 0 {la} 0 {ox2:.2f} {oy2:.2f} "
             f"L{ix2:.2f} {iy2:.2f} A{r} {r} 0 {la} 1 {ix1:.2f} {iy1:.2f}Z")
        return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="1.8" stroke-linejoin="round"/>'

    d_g = 180 - 0.30 * 180   # 126°
    d_o = 180 - 0.50 * 180   # 90°
    d_n = 180 - prob * 180    # aiguille

    arcs = (ring(180, d_g, "#d5f5e3", "#27ae60")
          + ring(d_g, d_o,  "#fef3e2", "#f39c12")
          + ring(d_o, 0,    "#fadbd8", "#e74c3c"))

    ticks = ""
    for d in range(0, 181, 18):
        mx1, my1 = pt(180 - d, R + 3)
        mx2, my2 = pt(180 - d, R + 10)
        ticks += f'<line x1="{mx1:.1f}" y1="{my1:.1f}" x2="{mx2:.1f}" y2="{my2:.1f}" stroke="#bbb" stroke-width="1.5"/>'

    def zone_label(mid_deg, txt, col):
        lx, ly = pt(mid_deg, R + 22)
        return (f'<text x="{lx:.0f}" y="{ly:.0f}" '
                f'font-family="Inter,sans-serif" font-size="11.5" font-weight="700" '
                f'fill="{col}" text-anchor="middle">{txt}</text>')

    labels = (zone_label(153, t("risk_low"),  "#27ae60")
            + zone_label(108, t("risk_mod"),  "#d68910")
            + zone_label(45,  t("risk_high"), "#e74c3c"))

    tick_labels = ""
    for deg, lbl, anchor in [(180,"0%","end"), (90,"50%","middle"), (0,"100%","start")]:
        tx, ty = pt(deg, R + 36)
        tick_labels += (f'<text x="{tx:.0f}" y="{ty:.0f}" '
                        f'font-family="Inter,sans-serif" font-size="9.5" '
                        f'fill="#aaa" text-anchor="{anchor}">{lbl}</text>')

    nx, ny   = pt(d_n, r - 6)
    nxb, nyb = pt(d_n + 90, 12)
    nxc, nyc = pt(d_n - 90, 12)
    needle = (f'<polygon points="{nx:.1f},{ny:.1f} {nxb:.1f},{nyb:.1f} {nxc:.1f},{nyc:.1f}" '
              f'fill="{color}" opacity="0.95"/>')
    hub    = (f'<circle cx="{cx}" cy="{cy}" r="11" fill="{color}"/>'
              f'<circle cx="{cx}" cy="{cy}" r="5.5" fill="white"/>')

    pct = (f'<text x="{cx}" y="{cy+36}" font-family="Inter,sans-serif" '
           f'font-size="34" font-weight="800" fill="{color}" text-anchor="middle">{prob:.1%}</text>')
    sub = (f'<text x="{cx}" y="{cy+54}" font-family="Inter,sans-serif" '
           f'font-size="10" fill="#999" text-anchor="middle">{t("prob_label")}</text>')

    return (f'<div style="text-align:center">'
            f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;max-width:390px;display:inline-block;">'
            f'{arcs}{ticks}{tick_labels}{labels}{needle}{hub}{pct}{sub}</svg></div>')


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
LABELS_MAP = {
    "Observance_4j_Mediocre":                       {"fr":"Observance médiocre (4j)",        "en":"Poor adherence (4 days)"},
    "Observance_4j_Modérée":                        {"fr":"Observance modérée (4j)",          "en":"Moderate adherence (4 days)"},
    "Type_FOSA_Privé laic":                         {"fr":"FOSA — Privé laïc",                "en":"Facility — Secular Private"},
    "Type_FOSA_Privé confessionnel":                {"fr":"FOSA — Privé confessionnel",       "en":"Facility — Faith-based"},
    "Soutien_PEPFAR_Oui":                           {"fr":"Soutien PEPFAR",                   "en":"PEPFAR support"},
    "Milieu_Residence_Rural":                       {"fr":"Milieu rural",                     "en":"Rural setting"},
    "Sexe_Masculin":                                {"fr":"Sexe masculin",                    "en":"Male sex"},
    "Activite_Remuneree_Oui":                       {"fr":"Activité rémunérée",               "en":"Paid activity"},
    "DSD_Recode_Dispensation communautaire / VAD":  {"fr":"Dispensation communautaire",       "en":"Community dispensation"},
    "Protocole_Recode_TDF+3TC+EFV (TELE/TLE)":     {"fr":"Protocole TELE/TLE",               "en":"TELE/TLE protocol"},
    "Protocole_Recode_Protocoles avec IP (ATV/r)":  {"fr":"Protocole IP (ATV/r)",             "en":"PI-based (ATV/r)"},
    "Protocole_Recode_Autre / Non spécifié":        {"fr":"Protocole autre",                  "en":"Other protocol"},
    "Statut_Matrimonial_Célibataire":               {"fr":"Célibataire",                      "en":"Single"},
    "Statut_Matrimonial_Veuf (ve)":                 {"fr":"Veuf(ve)",                         "en":"Widowed"},
    "Statut_Matrimonial_En union libre/concubinage":{"fr":"Union libre",                      "en":"Cohabiting"},
    "Statut_Matrimonial_Marié(e) en polygamie":     {"fr":"Polygamie",                        "en":"Polygamous"},
    "Niveau_Etude_Supérieur":                       {"fr":"Niveau supérieur",                 "en":"Higher education"},
    "Niveau_Etude_Jamais fréquenté":                {"fr":"Jamais scolarisé",                 "en":"Never schooled"},
    "Niveau_Etude_Secondaire Premier Cycle":         {"fr":"Secondaire 1er cycle",             "en":"Lower secondary"},
    "Niveau_Etude_Secondaire Second Cycle":          {"fr":"Secondaire 2ᵉ cycle",              "en":"Upper secondary"},
    "Tranche_Age_18 à 20 Ans":                      {"fr":"Âge 18–20 ans",                    "en":"Age 18–20 yrs"},
    "Tranche_Age_21 à 24 Ans":                      {"fr":"Âge 21–24 ans",                    "en":"Age 21–24 yrs"},
    "Tranche_Age_50 ans et plus":                   {"fr":"Âge ≥ 50 ans",                     "en":"Age ≥ 50 yrs"},
}


def draw_importance(top_n: int = 10) -> plt.Figure:
    imp  = model.feature_importances_
    feat = pd.DataFrame({"Variable": colonnes_train, "Importance": imp})
    feat = feat.sort_values("Importance", ascending=False).head(top_n).sort_values("Importance")

    def get_label(var):
        entry = LABELS_MAP.get(var)
        if entry:
            return entry[L]
        return var.replace("Region_", ("Région : " if L=="fr" else "Region: ")).replace("_", " ")

    feat["Label"] = feat["Variable"].apply(get_label)
    n      = len(feat)
    COLORS = ["#2980b9","#2980b9","#27ae60","#27ae60","#f39c12",
               "#f39c12","#e67e22","#e74c3c","#c0392b","#922b21"]
    colors = COLORS[::-1][:n][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7faff")
    bars = ax.barh(feat["Label"], feat["Importance"],
                   color=colors, height=0.60, edgecolor="#fff", linewidth=0.6)
    for bar, val in zip(bars, feat["Importance"]):
        ax.text(bar.get_width() + 0.0007, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8.5, color="#444", fontweight="500")
    ax.set_xlabel(t("imp_xlabel"), fontsize=8.5, color="#666")
    ax.set_title(t("imp_title").format(top_n), fontsize=10.5,
                 fontweight="700", color="#0b2d52", pad=12)
    ax.set_xlim(0, feat["Importance"].max() * 1.22)
    ax.tick_params(axis="y", labelsize=9, colors="#333", length=0)
    ax.tick_params(axis="x", labelsize=8, colors="#888")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#e0e8f4")
    ax.spines["bottom"].set_color("#e0e8f4")
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#cce")
    plt.tight_layout(pad=0.9)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION PDF
# ─────────────────────────────────────────────────────────────────────────────
def safe(text: str) -> str:
    """Encode le texte en latin-1 pour fpdf2 (gère les accents français)."""
    return text.encode('latin-1', 'replace').decode('latin-1')


def generate_pdf(patient_vals: list, prob: float, niveau: str,
                 reco: str, lang: str, cnls_path: Path, issea_path: Path) -> bytes:
    """Génère une fiche PDF patient prête à imprimer."""

    if prob < 0.30:
        r, g, b       = 39, 174, 96
        bg_r, bg_g, bg_b = 213, 245, 227
    elif prob < 0.50:
        r, g, b       = 230, 126, 34
        bg_r, bg_g, bg_b = 254, 243, 226
    else:
        r, g, b       = 231, 76, 60
        bg_r, bg_g, bg_b = 253, 237, 236

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── En-tête ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(11, 45, 82)
    pdf.rect(0, 0, 210, 44, 'F')

    # Logos
    try:
        pdf.image(str(cnls_path),  x=8,  y=4,  h=36)
        pdf.image(str(issea_path), x=172, y=4, h=36)
    except Exception:
        pass

    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_xy(50, 7)
    pdf.cell(110, 7, safe('TARV-Score — Fiche de Scoring Clinique'), align='C', ln=True)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(50, 17)
    lbl_sub = ("Outil de Detection du Risque d'Interruption au TARV chez les PVVIH au Cameroun"
               if lang == 'fr' else
               "Risk of ART Interruption Detection Tool among PLHIV in Cameroon")
    pdf.cell(110, 5, safe(lbl_sub), align='C', ln=True)
    pdf.set_xy(50, 25)
    pdf.cell(110, 5, safe('CNLS / GTC Cameroun | ISSEA-CEMAC 2025-2026'), align='C', ln=True)
    pdf.set_xy(50, 33)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.cell(110, 5, safe('XGBoost | Rappel : 83,8 % | AUC-ROC : 0,704 | Seuil : 50 %'), align='C')

    # ── Date d'évaluation ────────────────────────────────────────────────────
    pdf.set_text_color(100, 100, 100)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(10, 50)
    date_lbl = "Date d'evaluation" if lang == 'fr' else "Evaluation date"
    pdf.cell(0, 5, safe(f'{date_lbl} : {datetime.now().strftime("%d/%m/%Y  %H:%M")}'))

    # ── Score de risque ───────────────────────────────────────────────────────
    pdf.set_xy(10, 60)
    pdf.set_fill_color(bg_r, bg_g, bg_b)
    pdf.set_draw_color(r, g, b)
    pdf.set_line_width(0.8)
    pdf.rect(10, 60, 190, 30, 'FD')

    pdf.set_text_color(r, g, b)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_xy(10, 64)
    pdf.cell(95, 12, f'{prob:.1%}', align='C')

    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_xy(105, 64)
    risk_word = 'RISQUE' if lang == 'fr' else 'RISK'
    pdf.cell(95, 12, safe(f'{risk_word} {niveau}'), align='C')

    pdf.set_text_color(120, 120, 120)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(10, 80)
    seuil_txt = ('Seuil de classification : 50 % — Modele XGBoost'
                 if lang == 'fr' else
                 'Classification threshold: 50% — XGBoost model')
    pdf.cell(190, 5, safe(seuil_txt), align='C')

    # ── Profil du patient ─────────────────────────────────────────────────────
    pdf.set_text_color(11, 45, 82)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_xy(10, 96)
    pdf.cell(0, 7, safe('Profil du Patient' if lang == 'fr' else 'Patient Profile'), ln=True)

    # En-tête tableau
    pdf.set_fill_color(11, 45, 82)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_xy(10, 104)
    pdf.cell(85, 7, safe('Variable'), border=1, fill=True, align='C')
    pdf.cell(105, 7, safe('Valeur' if lang == 'fr' else 'Value'), border=1, fill=True, align='C', ln=True)

    # Lignes tableau
    var_labels = {
        'fr': ['Region', "Type de FOSA", 'Soutien PEPFAR', 'Milieu de residence',
               'Sexe', "Tranche d'age", "Niveau d'etude", 'Statut matrimonial',
               'Activite remuneree', 'Observance (4 jours)', 'Mode DSD', 'Protocole ARV'],
        'en': ['Region', 'Health Facility Type', 'PEPFAR Support', 'Residence Setting',
               'Sex', 'Age Group', 'Education Level', 'Marital Status',
               'Paid Activity', 'Adherence (4 days)', 'DSD Mode', 'ART Protocol'],
    }
    pdf.set_line_width(0.3)
    for i, (var, val) in enumerate(zip(var_labels[lang], patient_vals)):
        pdf.set_fill_color(240, 245, 255) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font('Helvetica', '', 8.5)
        pdf.cell(85, 6, safe(str(var)), border=1, fill=True)
        pdf.cell(105, 6, safe(str(val)), border=1, fill=True, ln=True)

    # ── Recommandation ────────────────────────────────────────────────────────
    pdf.ln(5)
    pdf.set_text_color(11, 45, 82)
    pdf.set_font('Helvetica', 'B', 10)
    reco_lbl = 'Recommandation clinique' if lang == 'fr' else 'Clinical Recommendation'
    pdf.cell(0, 7, safe(reco_lbl), ln=True)

    pdf.set_fill_color(bg_r, bg_g, bg_b)
    pdf.set_draw_color(r, g, b)
    pdf.set_line_width(0.8)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.multi_cell(190, 5.5, safe(reco), border=1, fill=True)

    # ── Pied de page ──────────────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(160, 160, 160)
    disc = ("Outil d'aide a la decision. Ne remplace pas le jugement clinique du prestataire de sante."
            if lang == 'fr' else
            "Decision-support tool. Does not replace the clinical judgment of the healthcare provider.")
    pdf.cell(0, 5, safe(disc), align='C')

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS IMPORT BATCH
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes attendues dans le fichier (valeurs internes FR)
COLS_REQUIS = [
    "Region", "Type_FOSA", "Soutien_PEPFAR", "Milieu_Residence",
    "Sexe", "Tranche_Age", "Niveau_Etude", "Statut_Matrimonial",
    "Activite_Remuneree", "Observance_4j", "DSD_Recode", "Protocole_Recode",
]

VALEURS_VALIDES = {
    "Region":             ["Centre","Adamaoua","Est","Extrême-Nord","Littoral","Nord","Nord-Ouest","Ouest","Sud","Sud-Ouest"],
    "Type_FOSA":          ["Public","Privé confessionnel","Privé laïc"],
    "Soutien_PEPFAR":     ["Oui","Non"],
    "Milieu_Residence":   ["Urbain","Rural"],
    "Sexe":               ["Féminin","Masculin"],
    "Tranche_Age":        ["25 à 49 Ans","18 à 20 Ans","21 à 24 Ans","50 ans et plus"],
    "Niveau_Etude":       ["Primaire","Jamais fréquenté","Secondaire Premier Cycle","Secondaire Second Cycle","Supérieur"],
    "Statut_Matrimonial": ["Marié(e) en monogamie","Célibataire","En union libre/concubinage","Marié(e) en polygamie","En séparation de corps / Divorcée","Veuf (ve)"],
    "Activite_Remuneree": ["Oui","Non"],
    "Observance_4j":      ["Bonne","Modérée","Médiocre"],
    "DSD_Recode":         ["Standard (Suivi classique)","Dispensation communautaire / VAD"],
    "Protocole_Recode":   ["TDF+3TC+DTG (TLD)","TDF+3TC+EFV (TELE/TLE)","Protocoles avec IP (ATV/r)","Autre / Non spécifié"],
}


def generate_template() -> bytes:
    """Génère un fichier Excel modèle avec les colonnes et valeurs valides."""
    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()

    # ── Feuille 1 : données ──────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Données patients"

    header_fill  = PatternFill("solid", fgColor="0B2D52")
    header_font  = Font(color="FFFFFF", bold=True, size=10)
    border_side  = Side(style="thin", color="CCCCCC")
    cell_border  = Border(left=border_side, right=border_side, top=border_side, bottom=border_side)

    # Colonne ID patient + 12 variables
    headers = ["ID_Patient"] + COLS_REQUIS
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = cell_border
        ws.column_dimensions[get_column_letter(c)].width = 22

    # Exemple de ligne
    exemple = ["P001", "Centre", "Public", "Oui", "Urbain", "Féminin",
               "25 à 49 Ans", "Primaire", "Marié(e) en monogamie", "Non",
               "Bonne", "Standard (Suivi classique)", "TDF+3TC+DTG (TLD)"]
    for c, val in enumerate(exemple, 1):
        cell = ws.cell(row=2, column=c, value=val)
        cell.fill = PatternFill("solid", fgColor="EEF5FF")
        cell.border = cell_border
        cell.alignment = Alignment(horizontal="center")

    ws.row_dimensions[1].height = 32

    # ── Feuille 2 : valeurs valides ──────────────────────────────────────────
    ws2 = wb.create_sheet("Valeurs valides")
    ws2.cell(1, 1, "Variable").font = Font(bold=True, color="FFFFFF", size=10)
    ws2.cell(1, 1).fill = header_fill
    ws2.cell(1, 2, "Valeurs acceptées").font = Font(bold=True, color="FFFFFF", size=10)
    ws2.cell(1, 2).fill = header_fill
    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 80

    row = 2
    for var, vals in VALEURS_VALIDES.items():
        ws2.cell(row, 1, var).font = Font(bold=True, color="0B2D52")
        ws2.cell(row, 2, " | ".join(vals))
        ws2.cell(row, 2).alignment = Alignment(wrap_text=True)
        ws2.row_dimensions[row].height = 20
        row += 1

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


def normalize_fosa(val: str) -> str:
    mapping = {"Privé laïc": "Privé laic", "Prive laic": "Privé laic",
               "Prive confessionnel": "Privé confessionnel"}
    return mapping.get(str(val).strip(), str(val).strip())


def normalize_observance(val: str) -> str:
    mapping = {"Médiocre": "Mediocre", "Mediocre": "Mediocre",
               "Modérée": "Modérée", "Bonne": "Bonne",
               "Poor": "Mediocre", "Moderate": "Modérée", "Good": "Bonne"}
    return mapping.get(str(val).strip(), str(val).strip())


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Score tous les patients d'un DataFrame et retourne df enrichi."""
    results = []
    for _, row in df.iterrows():
        try:
            raw = {
                "Tranche_Age":        str(row.get("Tranche_Age", "")).strip(),
                "Sexe":               str(row.get("Sexe", "")).strip(),
                "Niveau_Etude":       str(row.get("Niveau_Etude", "")).strip(),
                "Statut_Matrimonial": str(row.get("Statut_Matrimonial", "")).strip(),
                "Region":             str(row.get("Region", "")).strip(),
                "Type_FOSA":          normalize_fosa(row.get("Type_FOSA", "")),
                "Soutien_PEPFAR":     str(row.get("Soutien_PEPFAR", "")).strip(),
                "Milieu_Residence":   str(row.get("Milieu_Residence", "")).strip(),
                "Observance_4j":      normalize_observance(row.get("Observance_4j", "")),
                "Activite_Remuneree": str(row.get("Activite_Remuneree", "")).strip(),
                "DSD_Recode":         str(row.get("DSD_Recode", "")).strip(),
                "Protocole_Recode":   str(row.get("Protocole_Recode", "")).strip(),
            }
            prob = predict(raw)
            if prob < 0.30:
                niv = "Faible / Low"
            elif prob < 0.50:
                niv = "Modéré / Moderate"
            else:
                niv = "Élevé / High"
            results.append({"Probabilité (%)": round(prob * 100, 1), "Niveau de risque": niv, "Erreur": ""})
        except Exception as e:
            results.append({"Probabilité (%)": None, "Niveau de risque": "—", "Erreur": str(e)})

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


def generate_batch_pdf(df_res: pd.DataFrame, cnls_path: Path, issea_path: Path) -> bytes:
    """PDF récapitulatif de tous les patients scorés."""
    pdf = FPDF(orientation='L')   # paysage pour plus de colonnes
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    # En-tête
    pdf.set_fill_color(11, 45, 82)
    pdf.rect(0, 0, 297, 30, 'F')
    try:
        pdf.image(str(cnls_path),  x=4,  y=2, h=26)
        pdf.image(str(issea_path), x=263, y=2, h=26)
    except Exception:
        pass
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_xy(40, 5)
    pdf.cell(217, 8, safe('TARV-Score — Rapport de Scoring par Lot'), align='C', ln=True)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(40, 16)
    pdf.cell(217, 5, safe(f'CNLS / GTC Cameroun | {datetime.now().strftime("%d/%m/%Y %H:%M")} | {len(df_res)} patients'), align='C')

    # Statistiques résumé
    pdf.set_xy(10, 36)
    pdf.set_text_color(11, 45, 82)
    pdf.set_font('Helvetica', 'B', 10)
    n_tot  = len(df_res)
    n_low  = (df_res["Niveau de risque"].str.startswith("Faible")).sum()
    n_mod  = (df_res["Niveau de risque"].str.startswith("Mod")).sum()
    n_high = (df_res["Niveau de risque"].str.startswith("El")).sum()
    pdf.cell(0, 6, safe(f'Resume : {n_tot} patients | Faible : {n_low} | Modere : {n_mod} | Eleve : {n_high}'), ln=True)

    # Tableau
    pdf.ln(2)
    cols_show = (["ID_Patient"] if "ID_Patient" in df_res.columns else []) + \
                COLS_REQUIS[:6] + ["Probabilité (%)", "Niveau de risque"]
    col_widths = [22] * len(cols_show)
    if "ID_Patient" in cols_show:
        col_widths[0] = 18
    col_widths[-1] = 28
    col_widths[-2] = 22

    # En-tête tableau
    pdf.set_fill_color(11, 45, 82)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 7)
    for col, w in zip(cols_show, col_widths):
        pdf.cell(w, 7, safe(col[:18]), border=1, fill=True, align='C')
    pdf.ln()

    # Lignes
    pdf.set_font('Helvetica', '', 7)
    for i, (_, row) in enumerate(df_res.iterrows()):
        niv = str(row.get("Niveau de risque", ""))
        if niv.startswith("El"):
            pdf.set_fill_color(253, 237, 236)
            pdf.set_text_color(180, 40, 30)
        elif niv.startswith("Mod"):
            pdf.set_fill_color(254, 243, 226)
            pdf.set_text_color(160, 80, 10)
        else:
            pdf.set_fill_color(213, 245, 227) if i % 2 == 0 else pdf.set_fill_color(240, 255, 245)
            pdf.set_text_color(30, 100, 60)

        for col, w in zip(cols_show, col_widths):
            val = str(row.get(col, ""))[:20]
            pdf.cell(w, 6, safe(val), border=1, fill=True, align='C')
        pdf.ln()

    # Pied de page
    pdf.set_y(-12)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 5, safe("Outil d'aide a la decision — Ne remplace pas le jugement clinique | TARV-Score | ISSEA-CEMAC 2025-2026"), align='C')

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────────────────────────────────────
tab_lbl1 = "📝 Saisie individuelle" if L == "fr" else "📝 Individual entry"
tab_lbl2 = "📂 Import fichier (Excel / CSV)" if L == "fr" else "📂 File import (Excel / CSV)"
tab1, tab2 = st.tabs([tab_lbl1, tab_lbl2])

with tab1:

    st.markdown(f"""
<div style="background:#fff;border-radius:14px;padding:18px 22px 6px 22px;
     margin-bottom:4px;box-shadow:0 2px 10px rgba(0,0,0,0.06);border-top:4px solid #1a5e8a;">
  <div style="font-size:1.02em;font-weight:700;color:#0b2d52;">{t("form_intro")}</div>
  <div style="font-size:0.84em;color:#777;margin-top:3px;">{t("form_sub")}</div>
</div>
""", unsafe_allow_html=True)

    with st.form("patient_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3, gap="medium")
    
        with col1:
            st.markdown(
                f'<div style="font-size:0.82em;font-weight:700;color:#1a5e8a;'
                f'text-transform:uppercase;letter-spacing:1px;padding:8px 0 6px 0;'
                f'border-bottom:2px solid #d0e4f7;margin-bottom:12px;">'
                f'{t("sec_loc")}</div>', unsafe_allow_html=True)
            region    = st.selectbox(t("region"),    T["region_opts"][L],    key=f"{L}_region")
            type_fosa = st.selectbox(t("type_fosa"), T["type_fosa_opts"][L], key=f"{L}_fosa")
            pepfar    = st.radio(t("pepfar"),  [t("non"), t("oui")], horizontal=True, key=f"{L}_pepfar")
            milieu    = st.radio(t("milieu"),  [t("urbain"), t("rural")], horizontal=True, key=f"{L}_milieu")
    
        with col2:
            st.markdown(
                f'<div style="font-size:0.82em;font-weight:700;color:#0d7a5c;'
                f'text-transform:uppercase;letter-spacing:1px;padding:8px 0 6px 0;'
                f'border-bottom:2px solid #c3e8da;margin-bottom:12px;">'
                f'{t("sec_socio")}</div>', unsafe_allow_html=True)
            sexe         = st.radio(t("sexe"),         [t("feminin"), t("masculin")], horizontal=True, key=f"{L}_sexe")
            tranche_age  = st.selectbox(t("tranche_age"),  T["tranche_age_opts"][L],  key=f"{L}_age")
            niveau_etude = st.selectbox(t("niveau_etude"), T["niveau_etude_opts"][L], key=f"{L}_etude")
            statut_mat   = st.selectbox(t("statut_mat"),   T["statut_mat_opts"][L],   key=f"{L}_statut")
    
        with col3:
            st.markdown(
                f'<div style="font-size:0.82em;font-weight:700;color:#6c3483;'
                f'text-transform:uppercase;letter-spacing:1px;padding:8px 0 6px 0;'
                f'border-bottom:2px solid #e8d5f5;margin-bottom:12px;">'
                f'{t("sec_thera")}</div>', unsafe_allow_html=True)
            activite   = st.radio(t("activite"),   [t("non"), t("oui")], horizontal=True, key=f"{L}_activite")
            observance = st.selectbox(t("observance"),  T["observance_opts"][L], key=f"{L}_observance")
            dsd        = st.selectbox(t("dsd"),         T["dsd_opts"][L],        key=f"{L}_dsd")
            protocole  = st.selectbox(t("protocole"),   T["protocole_opts"][L],  key=f"{L}_protocole")
    
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(t("btn_calc"), use_container_width=True, type="primary")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # RÉSULTATS
    # ─────────────────────────────────────────────────────────────────────────────
    if submitted:
        # Convertir en valeurs internes (FR identiques à l'entraînement)
        region_fr    = REGION_EN2FR.get(region, region)              if L == "en" else region
        type_fosa_fr = FOSA_INTERNAL[type_fosa]
        pepfar_fr    = "Oui" if pepfar == t("oui") else "Non"
        milieu_fr    = "Urbain" if milieu == t("urbain") else "Rural"
        sexe_fr      = "Masculin" if sexe == t("masculin") else "Féminin"
        tranche_fr   = AGE_EN2FR.get(tranche_age, tranche_age)       if L == "en" else tranche_age
        etude_fr     = ETUDE_EN2FR.get(niveau_etude, niveau_etude)   if L == "en" else niveau_etude
        statut_fr    = STATUT_EN2FR.get(statut_mat, statut_mat)      if L == "en" else statut_mat
        activite_fr  = "Oui" if activite == t("oui") else "Non"
        observ_fr    = OBSERVANCE_INTERNAL[observance]
        dsd_fr       = DSD_EN2FR.get(dsd, dsd)                       if L == "en" else dsd
        proto_fr     = PROTO_EN2FR.get(protocole, protocole)         if L == "en" else protocole
    
        raw_fr = {
            "Tranche_Age":        tranche_fr,
            "Sexe":               sexe_fr,
            "Niveau_Etude":       etude_fr,
            "Statut_Matrimonial": statut_fr,
            "Region":             region_fr,
            "Type_FOSA":          type_fosa_fr,
            "Soutien_PEPFAR":     pepfar_fr,
            "Milieu_Residence":   milieu_fr,
            "Observance_4j":      observ_fr,
            "Activite_Remuneree": activite_fr,
            "DSD_Recode":         dsd_fr,
            "Protocole_Recode":   proto_fr,
        }
    
        with st.spinner(t("spinner")):
            prob = predict(raw_fr)
    
        if prob < 0.30:
            niveau, color        = t("risk_low_lbl"),  "#27ae60"
            emoji_r              = "🟢"
            bg_card, bd_card     = "#eafaf1", "#27ae60"
            bg_reco, bd_reco, tc = "#eafaf1", "#27ae60", "#1e5f3a"
            reco = t("reco_low")
        elif prob < SEUIL:
            niveau, color        = t("risk_mod_lbl"),  "#e67e22"
            emoji_r              = "🟠"
            bg_card, bd_card     = "#fef5e7", "#f39c12"
            bg_reco, bd_reco, tc = "#fef9e7", "#f39c12", "#784212"
            reco = t("reco_mod")
        else:
            niveau, color        = t("risk_high_lbl"), "#e74c3c"
            emoji_r              = "🔴"
            bg_card, bd_card     = "#fdedec", "#e74c3c"
            bg_reco, bd_reco, tc = "#fdedec", "#e74c3c", "#7b241c"
            reco = t("reco_high")
    
        risk_word = "RISQUE" if L == "fr" else "RISK"
    
        st.markdown(
            f'<div class="res-divider"><hr><span>{t("res_divider")}</span><hr></div>',
            unsafe_allow_html=True,
        )
    
        col_g, col_r = st.columns([1, 1.45], gap="large")
    
        with col_g:
            st.markdown(svg_gauge(prob, color), unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:{bg_card};border-radius:14px;border:2px solid {bd_card};
                 padding:18px 20px;text-align:center;margin-top:12px;
                 box-shadow:0 4px 16px {bd_card}33;">
                <div style="font-size:0.78em;font-weight:800;letter-spacing:2.5px;
                     text-transform:uppercase;color:{color};margin-bottom:4px;">
                    {emoji_r} &nbsp; {risk_word} {niveau}
                </div>
                <div style="font-size:3em;font-weight:900;color:{color};line-height:1.1;">
                    {prob:.1%}
                </div>
                <div style="font-size:0.8em;color:#777;margin-top:4px;">
                    {t("seuil_txt")} : {SEUIL:.0%} — XGBoost
                </div>
            </div>
            <div style="background:{bg_reco};border-left:5px solid {bd_reco};
                 border-radius:0 10px 10px 0;padding:14px 16px;
                 margin-top:10px;font-size:0.88em;line-height:1.6;color:{tc};">
                <strong>{t("reco_lbl")}</strong><br>{reco}
            </div>
            """, unsafe_allow_html=True)
    
        with col_r:
            st.markdown(
                f'<div style="font-size:0.83em;color:#777;margin-bottom:6px;">{t("imp_sub")}</div>',
                unsafe_allow_html=True,
            )
            st.pyplot(draw_importance(10), use_container_width=True)
            st.markdown(
                f'<div style="font-size:0.78em;color:#888;margin-top:2px;text-align:center;">'
                f'{t("imp_legend")}</div>',
                unsafe_allow_html=True,
            )
    
            with st.expander(t("recap_title")):
                var_labels = {
                    "fr": ["Région","Type de FOSA","Soutien PEPFAR","Milieu de résidence",
                           "Sexe","Tranche d'âge","Niveau d'étude","Statut matrimonial",
                           "Activité rémunérée","Observance (4j)","Mode DSD","Protocole ARV"],
                    "en": ["Region","Health Facility Type","PEPFAR Support","Residence Setting",
                           "Sex","Age Group","Education Level","Marital Status",
                           "Paid Activity","Adherence (4 days)","DSD Mode","ART Protocol"],
                }
                recap = pd.DataFrame({
                    t("recap_var"): var_labels[L],
                    t("recap_val"): [region, type_fosa, pepfar, milieu,
                                     sexe, tranche_age, niveau_etude, statut_mat,
                                     activite, observance, dsd, protocole],
                })
                st.dataframe(recap, use_container_width=True, hide_index=True)
    
        # ── Bouton export PDF ─────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:#fff;border-radius:12px;padding:16px 20px;'
            f'border:2px dashed #1a5e8a;text-align:center;margin-top:8px;">'
            f'<div style="font-size:0.9em;font-weight:600;color:#0b2d52;margin-bottom:10px;">'
            f'{"📄 Télécharger la fiche patient (PDF imprimable)" if L=="fr" else "📄 Download patient report (printable PDF)"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        patient_vals = [region, type_fosa, pepfar, milieu, sexe, tranche_age,
                        niveau_etude, statut_mat, activite, observance, dsd, protocole]
        try:
            pdf_bytes = generate_pdf(
                patient_vals=patient_vals,
                prob=prob,
                niveau=niveau,
                reco=reco,
                lang=L,
                cnls_path=ASSETS / "cnls_logo.png",
                issea_path=ASSETS / "issea_logo.png",
            )
            btn_lbl = "⬇️ Télécharger le PDF" if L == "fr" else "⬇️ Download PDF"
            st.download_button(
                label=btn_lbl,
                data=pdf_bytes,
                file_name=f"fiche_TARV_Score_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF non disponible : {e}")
    
with tab2:

    # ── Introduction ─────────────────────────────────────────────────────────
    intro_txt = ("Uploadez un fichier Excel ou CSV contenant les données de plusieurs patients. "
                 "L'application les scorera automatiquement et vous pourrez télécharger les résultats."
                 if L == "fr" else
                 "Upload an Excel or CSV file with multiple patient records. "
                 "The app will score them automatically and you can download the results.")
    st.markdown(f"""
    <div style="background:#fff;border-radius:14px;padding:16px 22px;
         margin-bottom:16px;box-shadow:0 2px 10px rgba(0,0,0,0.06);border-top:4px solid #0d7a5c;">
      <div style="font-size:1em;font-weight:700;color:#0b2d52;">
          {"📂 Scoring par lot — Import de fichier" if L=="fr" else "📂 Batch Scoring — File Import"}
      </div>
      <div style="font-size:0.84em;color:#777;margin-top:4px;">{intro_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Téléchargement du modèle ──────────────────────────────────────────────
    st.markdown(
        f'<div style="font-weight:600;color:#0b2d52;margin-bottom:6px;">'
        f'{"📋 Étape 1 — Téléchargez le modèle Excel à remplir" if L=="fr" else "📋 Step 1 — Download the Excel template to fill in"}'
        f'</div>', unsafe_allow_html=True)

    try:
        tpl_bytes = generate_template()
        st.download_button(
            label="⬇️ Télécharger le modèle Excel" if L == "fr" else "⬇️ Download Excel template",
            data=tpl_bytes,
            file_name="modele_TARV_Score.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )
    except Exception as e:
        st.error(f"Erreur génération modèle : {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload du fichier ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-weight:600;color:#0b2d52;margin-bottom:6px;">'
        f'{"📤 Étape 2 — Uploadez votre fichier rempli" if L=="fr" else "📤 Step 2 — Upload your filled file"}'
        f'</div>', unsafe_allow_html=True)

    upload_label = "Choisir un fichier Excel ou CSV" if L == "fr" else "Choose an Excel or CSV file"
    uploaded = st.file_uploader(upload_label, type=["xlsx", "xls", "csv"], key="batch_upload")

    if uploaded is not None:
        # Lecture du fichier
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_up = pd.read_csv(uploaded, sep=None, engine="python")
            else:
                df_up = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
            df_up = None

        if df_up is not None:
            # Vérification des colonnes
            missing = [c for c in COLS_REQUIS if c not in df_up.columns]
            if missing:
                st.error(
                    ("❌ Colonnes manquantes dans le fichier : **" + "**, **".join(missing) + "**\n\n"
                     "Utilisez le modèle Excel fourni à l'étape 1."
                     if L == "fr" else
                     "❌ Missing columns in file: **" + "**, **".join(missing) + "**\n\n"
                     "Please use the Excel template from step 1.")
                )
            else:
                st.success(f"✅ {'Fichier valide' if L=='fr' else 'Valid file'} — {len(df_up)} {'patients détectés' if L=='fr' else 'patients detected'}")

                # Aperçu
                with st.expander("👁️ Aperçu du fichier" if L == "fr" else "👁️ File preview"):
                    st.dataframe(df_up.head(5), use_container_width=True)

                # Bouton de scoring
                btn_score = "🔍 Scorer tous les patients" if L == "fr" else "🔍 Score all patients"
                if st.button(btn_score, type="primary", use_container_width=True, key="batch_score_btn"):
                    with st.spinner("Calcul en cours…" if L == "fr" else "Computing…"):
                        df_res = score_dataframe(df_up)

                    # Résultats
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="font-weight:700;font-size:1em;color:#0b2d52;margin-bottom:8px;">'
                        f'{"📊 Résultats du scoring" if L=="fr" else "📊 Scoring Results"}'
                        f'</div>', unsafe_allow_html=True)

                    # Tableau coloré
                    def color_row(row):
                        niv = str(row.get("Niveau de risque", ""))
                        if niv.startswith("El"):
                            return ["background-color:#fdedec"] * len(row)
                        elif niv.startswith("Mod"):
                            return ["background-color:#fef9e7"] * len(row)
                        else:
                            return ["background-color:#eafaf1"] * len(row)

                    st.dataframe(
                        df_res.style.apply(color_row, axis=1),
                        use_container_width=True,
                        height=min(400, 40 + len(df_res) * 35),
                    )

                    # Statistiques rapides
                    n_tot  = len(df_res)
                    n_low  = (df_res["Niveau de risque"].str.startswith("Faible")).sum()
                    n_mod  = (df_res["Niveau de risque"].str.startswith("Mod")).sum()
                    n_high = (df_res["Niveau de risque"].str.startswith("El")).sum()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total", n_tot)
                    c2.metric("🟢 Faible / Low",   n_low)
                    c3.metric("🟠 Modéré / Mod",   n_mod)
                    c4.metric("🔴 Élevé / High",   n_high)

                    st.markdown("<br>", unsafe_allow_html=True)
                    dl1, dl2 = st.columns(2)

                    # Export Excel
                    with dl1:
                        import io as _io
                        buf = _io.BytesIO()
                        df_res.to_excel(buf, index=False, engine="openpyxl")
                        buf.seek(0)
                        st.download_button(
                            label="⬇️ Télécharger Excel (avec scores)" if L == "fr" else "⬇️ Download Excel (with scores)",
                            data=buf.read(),
                            file_name=f"TARV_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

                    # Export PDF récapitulatif
                    with dl2:
                        try:
                            pdf_batch = generate_batch_pdf(
                                df_res,
                                cnls_path=ASSETS / "cnls_logo.png",
                                issea_path=ASSETS / "issea_logo.png",
                            )
                            st.download_button(
                                label="⬇️ Télécharger PDF récapitulatif" if L == "fr" else "⬇️ Download PDF summary",
                                data=pdf_batch,
                                file_name=f"TARV_rapport_lot_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.warning(f"PDF non disponible : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PIED DE PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="footer">{t("footer")}</div>', unsafe_allow_html=True)
