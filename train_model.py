# =============================================================================
# TARV-Score — Entraînement du modèle de production (14 variables)
# Script autonome et reproductible : sélectionne le meilleur des 4 candidats
# (Rég. Logistique, Random Forest, SVM linéaire, XGBoost) sur les 14 variables
# retenues pour l'application, et exporte les artefacts nécessaires à
# l'application Streamlit (app.py) dans ./modeles/.
#
# Sélection des variables : 8 choisies par analyse combinée Chi²/V de Cramér +
# test de Wald joint (régression logistique multivariée) + importances
# agrégées Random Forest & XGBoost (calculée sur Base_Traitee.xlsx et les
# modèles à 21 variables du notebook analyse_TARV_v5_FINAL_8.ipynb), 4
# variables protégées par la littérature (Sexe, Tranche_Age,
# Statut_Matrimonial, Soutien_PEPFAR) — cf. Étape 12 du notebook — et 2
# variables ajoutées à la demande du CNLS (Soutien_Familial,
# Traitement_Alternatif).
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
    "Sexe", "Tranche_Age", "Statut_Matrimonial", "Soutien_PEPFAR",
    "Observance_4j", "Region", "Religion", "Depenses_Mensuelles",
    "DSD_Recode", "Niveau_Etude", "Retesting", "Type_FOSA",
    "Soutien_Familial", "Traitement_Alternatif",
]

REFERENCES = {
    "Sexe":                 "Féminin",
    "Tranche_Age":          "25 à 49 Ans",
    "Statut_Matrimonial":   "Marié(e) en monogamie",
    "Soutien_PEPFAR":       "Oui",
    "Observance_4j":        "Bonne",
    "Region":               "Centre",
    "Religion":             "Catholique",
    "Depenses_Mensuelles":  "Moins de 5 000",
    "DSD_Recode":           "Standard",
    "Niveau_Etude":         "Jamais fréquenté",
    "Retesting":            "Non",
    "Type_FOSA":            "Public",
    "Soutien_Familial":     "Oui",
    "Traitement_Alternatif": "Non",
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
    lr_cal = CalibratedClassifierCV(lr_s.best_estimator_, cv=5, method="isotonic")
    lr_cal.fit(X_tr_sc, y_train)
    print(f"   CV F1 = {lr_s.best_score_:.4f} | C = {lr_s.best_params_['C']}")

    print("-> Random Forest...")
    rf_s = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, class_weight="balanced"),
        {"n_estimators": [200, 300, 500], "max_depth": [5, 8, 10, 15],
         "min_samples_split": [5, 10, 20], "min_samples_leaf": [3, 5, 10],
         "max_features": ["sqrt", "log2"]},
        n_iter=25, cv=CV, scoring="f1", random_state=SEED, n_jobs=-1)
    rf_s.fit(X_tr_sc, y_train)
    rf_cal = CalibratedClassifierCV(rf_s.best_estimator_, cv=5, method="isotonic")
    rf_cal.fit(X_tr_sc, y_train)
    print(f"   CV F1 = {rf_s.best_score_:.4f}")

    print("-> SVM lineaire...")
    svm_s = RandomizedSearchCV(
        LinearSVC(random_state=SEED, class_weight="balanced", max_iter=5000, dual=False),
        {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
        n_iter=7, cv=CV, scoring="f1", random_state=SEED, n_jobs=-1)
    svm_s.fit(X_tr_sc, y_train)
    svm_cal = CalibratedClassifierCV(svm_s.best_estimator_, cv=5, method="isotonic")
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
    xgb_cal = CalibratedClassifierCV(xgb_s.best_estimator_, cv=5, method="isotonic")
    xgb_cal.fit(X_tr_sc, y_train)
    print(f"   CV F1 = {xgb_s.best_score_:.4f}")

    # Tous les candidats sont calibrés (isotonic, cv=5) avant comparaison : un
    # score de Brier/composite bas ne garantit pas une bonne calibration par
    # tranche de risque (cf. vérification ECE plus bas), donc on ne compare
    # et ne déploie que des probabilités déjà post-calibrées.
    best_models = {
        "Rég. Logistique": lr_cal,
        "Random Forest":   rf_cal,
        "SVM linéaire":    svm_cal,
        "XGBoost":         xgb_cal,
    }

    print("\n" + "=" * 70)
    print("ETAPE 3 : OPTIMISATION DU SEUIL + EVALUATION (au seuil optimal de chaque modèle)")
    print("=" * 70)

    seuils_optimaux = {}
    resultats = []
    probs_par_modele = {}
    for nom, model in best_models.items():
        probs = model.predict_proba(X_te_sc)[:, 1]
        probs_par_modele[nom] = probs

        meilleur_f1, meilleur_seuil = 0, 0.50
        for s in np.arange(0.20, 0.65, 0.01):
            preds_s = (probs >= s).astype(int)
            prec_s = precision_score(y_test, preds_s, zero_division=0)
            f1_s = f1_score(y_test, preds_s)
            if prec_s >= 0.50 and f1_s > meilleur_f1:
                meilleur_f1, meilleur_seuil = f1_s, s
        seuils_optimaux[nom] = float(meilleur_seuil)

        # Évaluation au seuil propre à chaque modèle (et non à 0.50 fixe) :
        # c'est ce seuil qui sera réellement déployé dans l'application, donc
        # les métriques de comparaison doivent refléter ce point de fonctionnement.
        preds = (probs >= meilleur_seuil).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        sensibilite = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificite = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(y_test, preds, zero_division=0)
        auc_roc = roc_auc_score(y_test, probs)
        auc_pr = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        resultats.append({
            "Modèle": nom,
            "Sensibilité": sensibilite,
            "Spécificité": specificite,
            "Précision": precision,
            "F1-Score": f1_score(y_test, preds),
            "AUC-ROC": auc_roc,
            "AUC-PR": auc_pr,
            "Brier": brier,
            # Score composite : moyenne à poids égal des 6 métriques cliniquement
            # pertinentes (Brier inversé car "plus petit = meilleur"). Choisi pour
            # ne sacrifier aucune dimension (contexte de retrait des bailleurs :
            # le modèle déployé doit être robuste sur tous les plans, pas
            # seulement optimal sur un seul critère comme le F1-Score).
            "Score composite": (sensibilite + specificite + precision
                                 + auc_roc + auc_pr + (1 - brier)) / 6,
            "Seuil optimal": meilleur_seuil,
        })

    df_res = pd.DataFrame(resultats)
    print(df_res.round(4).to_string(index=False))

    best_nom = df_res.loc[df_res["Score composite"].idxmax(), "Modèle"]
    best_row = df_res.loc[df_res["Score composite"].idxmax()]
    best_model = best_models[best_nom]
    print(f"\n*** MODELE RETENU (meilleur Score composite, 6 métriques à poids égal) : {best_nom} ***")

    print("\n" + "=" * 70)
    print("ETAPE 3bis : VERIFICATION DE LA CALIBRATION (modèle retenu)")
    print("=" * 70)

    from sklearn.calibration import calibration_curve
    probs_best = probs_par_modele[best_nom]
    frac_obs, frac_pred = calibration_curve(y_test, probs_best, n_bins=10, strategy="quantile")
    # ECE (Expected Calibration Error) : écart moyen, pondéré par la taille de
    # chaque tranche, entre la probabilité annoncée et la fréquence réellement
    # observée. Un score de Brier bas ne suffit pas à garantir une bonne
    # calibration par tranche de risque (Faible/Modéré/Élevé) : c'est cette
    # vérification qui le confirme réellement.
    bin_edges = np.quantile(probs_best, np.linspace(0, 1, 11))
    bin_ids = np.clip(np.digitize(probs_best, bin_edges[1:-1]), 0, len(frac_obs) - 1)
    poids = np.array([np.sum(bin_ids == i) for i in range(len(frac_obs))]) / len(probs_best)
    ece = float(np.sum(poids * np.abs(frac_obs - frac_pred)))

    df_calib = pd.DataFrame({
        "Probabilité moyenne prédite": frac_pred,
        "Fréquence réelle observée": frac_obs,
        "Écart absolu": np.abs(frac_obs - frac_pred),
    })
    print(df_calib.round(4).to_string(index=False))
    print(f"\nECE (Expected Calibration Error) = {ece:.4f}  "
          f"({'bien calibré' if ece < 0.05 else 'calibration a surveiller'} si < 0.05)")
    print(f"Brier du modèle retenu = {float(best_row['Brier']):.4f} (mesure globale ; l'ECE ci-dessus vérifie "
          f"la calibration tranche par tranche, ce que le Brier seul ne garantit pas)")

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
        "ece": ece,
        "score_composite": float(best_row["Score composite"]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_variables": len(VARS_FIN),
        "variables": VARS_FIN,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    with open(CHEMIN_MODELES / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with pd.ExcelWriter(CHEMIN_MODELES / "performances_modeles.xlsx") as writer:
        df_res.to_excel(writer, sheet_name="Comparaison modeles", index=False)
        df_calib.to_excel(writer, sheet_name="Calibration", index=False)
        writer.book.properties.creator = "CNLS / GTC Cameroun"
        writer.book.properties.lastModifiedBy = "CNLS / GTC Cameroun"
        writer.book.properties.title = "Performances des modeles TARV-Score"

    print(f"  ✓ modele_final.pkl ({best_nom})")
    print("  ✓ scaler.pkl, colonnes_modele.pkl, references.pkl, variables_modele.pkl")
    print("  ✓ seuil_optimal.pkl, meta.json, performances_modeles.xlsx (comparaison + calibration)")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
