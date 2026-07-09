# =============================================================================
# TARV-Score — Entraînement du modèle de production (10 variables)
# Script autonome et reproductible : sélectionne le meilleur des 4 candidats
# (Rég. Logistique, Random Forest, SVM linéaire, XGBoost) sur les 10 variables
# les plus explicatives de l'interruption au TARV, et exporte les artefacts
# nécessaires à l'application Streamlit (app.py) dans ./modeles/.
#
# Sélection des 10 variables : analyse combinée Chi²/V de Cramér + test de
# Wald joint (régression logistique multivariée) + importances agrégées
# Random Forest & XGBoost, calculée sur Base_Traitee.xlsx et les modèles
# à 21 variables du notebook analyse_TARV_v5_FINAL_8.ipynb.
# =============================================================================
import io
import json
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).parent
CHEMIN_TRAITEE = BASE.parent.parent / "01_Donnees" / "Traitees" / "Base_Traitee.xlsx"
CHEMIN_MODELES = BASE / "modeles"
CHEMIN_MODELES.mkdir(parents=True, exist_ok=True)

VARS_FIN = [
    "Observance_4j", "Region", "Religion", "Depenses_Mensuelles", "DSD_Recode",
    "Revenu", "Delai_Attente_Cat", "Niveau_Etude", "Retesting", "Type_FOSA",
]

REFERENCES = {
    "Observance_4j":        "Bonne",
    "Region":               "Centre",
    "Religion":             "Catholique",
    "Depenses_Mensuelles":  "Moins de 5 000",
    "DSD_Recode":           "Standard",
    "Revenu":               "Moins de 10 000",
    "Delai_Attente_Cat":    "≤15 min",
    "Niveau_Etude":         "Jamais fréquenté",
    "Retesting":            "Non",
    "Type_FOSA":            "Public",
}


def dummifier(df_in: pd.DataFrame) -> pd.DataFrame:
    """Dummifie en retirant la modalité de référence (pas alphabétique)."""
    df_dum = pd.get_dummies(df_in, dtype=int)
    cols_drop = [f"{var}_{ref}" for var, ref in REFERENCES.items()
                 if f"{var}_{ref}" in df_dum.columns]
    return df_dum.drop(columns=cols_drop)


def clean_name(c: str) -> str:
    return re.sub(r"[\[\]<>]", "", c).replace(" ", "_")


def main():
    print("=" * 70)
    print("ETAPE 1 : CHARGEMENT ET PREPARATION DES DONNEES")
    print("=" * 70)

    if not CHEMIN_TRAITEE.exists():
        sys.exit(f"Base introuvable : {CHEMIN_TRAITEE}")

    df = pd.read_excel(CHEMIN_TRAITEE)
    df["Y_Interruption"] = df["Y_Interruption"].astype(str).str.strip()
    y = (df["Y_Interruption"] == "Oui").astype(int)

    manquants = [v for v in VARS_FIN if v not in df.columns]
    if manquants:
        sys.exit(f"Variables absentes de la base : {manquants}")

    X = df[VARS_FIN].copy()
    X_dum = dummifier(X)
    print(f"Variables retenues ({len(VARS_FIN)}) : {VARS_FIN}")
    print(f"Colonnes dummifiées ({X_dum.shape[1]}) : {list(X_dum.columns)}")
    print(f"N = {len(df)} | Taux d'interruption = {y.mean():.1%}")

    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (recall_score, precision_score, f1_score,
                                 roc_auc_score, brier_score_loss,
                                 average_precision_score, confusion_matrix)
    from xgboost import XGBClassifier
    import sklearn, xgboost
    print(f"scikit-learn = {sklearn.__version__} | xgboost = {xgboost.__version__}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_dum, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")

    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    clean_cols = [clean_name(c) for c in X_tr_sc.columns]
    X_tr_sc.columns = clean_cols
    X_te_sc.columns = clean_cols

    print("\n" + "=" * 70)
    print("ETAPE 2 : OPTIMISATION DES 4 MODELES (scoring = F1)")
    print("=" * 70)

    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("-> Reg. Logistique...")
    lr_s = RandomizedSearchCV(
        LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED, class_weight="balanced"),
        {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]},
        n_iter=8, cv=CV, scoring="f1", random_state=SEED, n_jobs=-1)
    lr_s.fit(X_tr_sc, y_train)
    print(f"   CV F1 = {lr_s.best_score_:.4f} | C = {lr_s.best_params_['C']}")

    print("-> Random Forest...")
    rf_s = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, class_weight="balanced"),
        {"n_estimators": [200, 300, 500], "max_depth": [5, 8, 10, 15],
         "min_samples_split": [5, 10, 20], "min_samples_leaf": [3, 5, 10],
         "max_features": ["sqrt", "log2"]},
        n_iter=25, cv=CV, scoring="f1", random_state=SEED, n_jobs=-1)
    rf_s.fit(X_tr_sc, y_train)
    print(f"   CV F1 = {rf_s.best_score_:.4f}")

    print("-> SVM lineaire...")
    svm_s = RandomizedSearchCV(
        LinearSVC(random_state=SEED, class_weight="balanced", max_iter=5000, dual=False),
        {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
        n_iter=7, cv=CV, scoring="f1", random_state=SEED, n_jobs=-1)
    svm_s.fit(X_tr_sc, y_train)
    svm_cal = CalibratedClassifierCV(svm_s.best_estimator_, cv=5)
    svm_cal.fit(X_tr_sc, y_train)
    print(f"   CV F1 = {svm_s.best_score_:.4f}")

    print("-> XGBoost...")
    xgb_s = RandomizedSearchCV(
        XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1),
        {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 5],
         "learning_rate": [0.01, 0.05, 0.1], "scale_pos_weight": [1.5, 2.0, 2.69],
         "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0],
         "min_child_weight": [3, 5, 7], "reg_alpha": [0.1, 1.0], "reg_lambda": [1.0, 2.0, 5.0]},
        n_iter=25, cv=CV, scoring="f1", random_state=SEED, n_jobs=-1)
    xgb_s.fit(X_tr_sc, y_train)
    print(f"   CV F1 = {xgb_s.best_score_:.4f}")

    best_models = {
        "Rég. Logistique": lr_s.best_estimator_,
        "Random Forest":   rf_s.best_estimator_,
        "SVM linéaire":    svm_cal,
        "XGBoost":         xgb_s.best_estimator_,
    }

    print("\n" + "=" * 70)
    print("ETAPE 3 : OPTIMISATION DU SEUIL + EVALUATION (seuil = 0.50)")
    print("=" * 70)

    seuils_optimaux = {}
    resultats = []
    for nom, model in best_models.items():
        probs = model.predict_proba(X_te_sc)[:, 1]

        meilleur_f1, meilleur_seuil = 0, 0.50
        for s in np.arange(0.20, 0.65, 0.01):
            preds_s = (probs >= s).astype(int)
            prec_s = precision_score(y_test, preds_s, zero_division=0)
            f1_s = f1_score(y_test, preds_s)
            if prec_s >= 0.50 and f1_s > meilleur_f1:
                meilleur_f1, meilleur_seuil = f1_s, s
        seuils_optimaux[nom] = float(meilleur_seuil)

        preds = (probs >= 0.50).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        resultats.append({
            "Modèle": nom,
            "Sensibilité": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Spécificité": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "Précision": precision_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds),
            "AUC-ROC": roc_auc_score(y_test, probs),
            "AUC-PR": average_precision_score(y_test, probs),
            "Brier": brier_score_loss(y_test, probs),
            "Seuil optimal": meilleur_seuil,
        })

    df_res = pd.DataFrame(resultats)
    print(df_res.round(4).to_string(index=False))

    best_nom = df_res.loc[df_res["F1-Score"].idxmax(), "Modèle"]
    best_row = df_res.loc[df_res["F1-Score"].idxmax()]
    best_model = best_models[best_nom]
    print(f"\n*** MODELE RETENU (meilleur F1-Score) : {best_nom} ***")

    print("\n" + "=" * 70)
    print("ETAPE 4 : SAUVEGARDE DES ARTEFACTS POUR L'APPLICATION")
    print("=" * 70)

    # Nettoyage des anciens artefacts (12 variables / SMOTE)
    anciens = ["Logit_v1.pkl", "Random_Forest_v1.pkl", "SVM_v1.pkl", "XGBoost_v1.pkl",
               "scaler_prepro.pkl", "colonnes_train.pkl", "meta.json", "meta_all.json"]
    for f in anciens:
        p = CHEMIN_MODELES / f
        if p.exists():
            p.unlink()
            print(f"  x supprimé : {f}")

    joblib.dump(best_model, CHEMIN_MODELES / "modele_final.pkl")
    joblib.dump(scaler, CHEMIN_MODELES / "scaler.pkl")
    joblib.dump(clean_cols, CHEMIN_MODELES / "colonnes_modele.pkl")
    joblib.dump(REFERENCES, CHEMIN_MODELES / "references.pkl")
    joblib.dump(VARS_FIN, CHEMIN_MODELES / "variables_modele.pkl")
    joblib.dump(float(best_row["Seuil optimal"]), CHEMIN_MODELES / "seuil_optimal.pkl")

    meta = {
        "nom_modele": best_nom,
        "fichier": "modele_final.pkl",
        "seuil_defaut": float(best_row["Seuil optimal"]),
        "rappel": float(best_row["Sensibilité"]),
        "specificite": float(best_row["Spécificité"]),
        "precision": float(best_row["Précision"]),
        "f1_score": float(best_row["F1-Score"]),
        "auc_roc": float(best_row["AUC-ROC"]),
        "auc_pr": float(best_row["AUC-PR"]),
        "brier": float(best_row["Brier"]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_variables": len(VARS_FIN),
        "variables": VARS_FIN,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    with open(CHEMIN_MODELES / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    df_res.to_excel(CHEMIN_MODELES / "performances_modeles.xlsx", index=False)

    print(f"  ✓ modele_final.pkl ({best_nom})")
    print("  ✓ scaler.pkl, colonnes_modele.pkl, references.pkl, variables_modele.pkl")
    print("  ✓ seuil_optimal.pkl, meta.json, performances_modeles.xlsx")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
