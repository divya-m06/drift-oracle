"""
German Credit — XGBoost (Champion) vs SVM (Challenger) + PSI-Triggered Model Switching
Uses RandomizedSearchCV + feature engineering to maximize accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

GERMAN_PATH         = "data/german_data.csv"
PSI_DRIFT_THRESHOLD = 0.2
AGREE_THRESHOLD     = 0.90

CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
]


def find_best_threshold(y_true, y_proba):
    """Scan thresholds 0.1–0.9, return the one with best accuracy."""
    best_thresh, best_acc = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        acc = (preds == y_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return round(best_thresh, 2)


def evaluate_model(name, model, X_te, y_te, threshold=None):
    """Evaluate a fitted model on test set, return metrics dict."""
    y_proba = model.predict_proba(X_te)[:, 1]
    thresh  = threshold if threshold is not None else find_best_threshold(y_te, y_proba)
    y_pred  = (y_proba >= thresh).astype(int)

    auc      = roc_auc_score(y_te, y_proba)
    f1       = f1_score(y_te, y_pred, zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    accuracy = (y_pred == y_te).mean()

    print(f"\n{name} (threshold={thresh})")
    print(f"  AUC: {auc:.4f}  Accuracy: {accuracy:.4f}  F1: {f1:.4f}  F1 Macro: {f1_macro:.4f}")
    print(classification_report(y_te, y_pred, target_names=['Good Credit', 'Bad Credit'], zero_division=0))

    return {
        "name": name, "model": model,
        "auc": auc, "f1": f1, "f1_macro": f1_macro,
        "accuracy": accuracy, "y_pred": y_pred, "y_proba": y_proba,
    }


def print_comparison(results, title="Model Comparison"):
    """Print a side-by-side comparison table."""
    print(f"\n{title}")
    df = pd.DataFrame([{
        "Model":    r["name"],
        "AUC":      round(r["auc"], 4),
        "Accuracy": f"{r['accuracy']*100:.1f}%",
        "F1":       round(r["f1"], 4),
        "F1 Macro": round(r["f1_macro"], 4),
    } for r in results])
    print(df.to_string(index=False))


def compare_predictions(champion_preds, challenger_preds):
    """Compare champion vs challenger predictions. Returns (agreement_ratio, verdict)."""
    total = len(champion_preds)
    agree = int((champion_preds == challenger_preds).sum())
    ratio = agree / total

    print(f"\nPrediction Agreement: {agree}/{total} ({ratio:.1%})")
    print(f"  Threshold: {AGREE_THRESHOLD:.0%}")

    if ratio >= AGREE_THRESHOLD:
        verdict = "STABLE"
        print(f"  Verdict: STABLE — champion and challenger agree, champion stays")
    else:
        verdict = "INVALID"
        print(f"  Verdict: INVALID — models diverge, switch to challenger")

    return ratio, verdict


# ── Load and preprocess German Credit data ──

try:
    df_german = pd.read_csv(GERMAN_PATH)

    if df_german.shape[1] == 21 and "checking_status" not in df_german.columns:
        df_german.columns = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment", "installment_commitment", "personal_status",
            "other_parties", "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class"
        ]

    print(f"Loaded German Credit: {df_german.shape}")

    target_col = next((c for c in df_german.columns if c.lower() == "class"), None)
    if target_col is None:
        raise ValueError(f"No 'class' column found. Columns: {df_german.columns.tolist()}")

    # Remap 1→0 (Good), 2→1 (Bad) if original encoding
    if df_german[target_col].isin([1, 2]).all():
        df_german[target_col] = (df_german[target_col] == 2).astype(int)

    print(f"Target distribution:\n{df_german[target_col].value_counts()}\n")

    X_g = df_german.drop(target_col, axis=1).copy()
    y_g = df_german[target_col]

    # Feature engineering — meaningful financial ratios
    if 'credit_amount' in X_g.columns and 'duration' in X_g.columns:
        X_g['credit_per_month'] = X_g['credit_amount'] / X_g['duration'].clip(lower=1)
    if 'credit_amount' in X_g.columns and 'age' in X_g.columns:
        X_g['credit_to_age'] = X_g['credit_amount'] / X_g['age'].clip(lower=1)
    if 'installment_commitment' in X_g.columns and 'duration' in X_g.columns:
        X_g['total_commitment'] = X_g['installment_commitment'] * X_g['duration']
    if 'age' in X_g.columns and 'duration' in X_g.columns:
        X_g['age_duration_ratio'] = X_g['age'] / X_g['duration'].clip(lower=1)

    print(f"After feature engineering: {X_g.shape}")

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X_g.columns]
    if len(cat_cols_present) == 0:
        cat_cols_present = X_g.select_dtypes(include="object").columns.tolist()
    else:
        for col in cat_cols_present:
            X_g[col] = X_g[col].astype(str)

    X_g_encoded = pd.get_dummies(X_g, columns=cat_cols_present, drop_first=True).astype(float)
    print(f"After OHE: {X_g_encoded.shape}")

    if X_g_encoded.shape[1] == 0:
        raise ValueError("OHE produced 0 columns — check categorical column names.")

    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_g_encoded, y_g, test_size=0.2, random_state=42, stratify=y_g
    )

    scaler_g     = StandardScaler()
    Xg_train_sc  = scaler_g.fit_transform(Xg_train)
    Xg_test_sc   = scaler_g.transform(Xg_test)
    yg_train_arr = yg_train.values
    yg_test_arr  = yg_test.values

    neg_g = (yg_train_arr == 0).sum()
    pos_g = (yg_train_arr == 1).sum()
    print(f"Train: {Xg_train_sc.shape}, Test: {Xg_test_sc.shape}, Imbalance: {neg_g/pos_g:.1f}:1\n")

except FileNotFoundError:
    print(f"File not found: '{GERMAN_PATH}' — place german_data.csv in data/ and rerun.")
    raise SystemExit(1)
except ValueError as e:
    print(f"Data error: {e}")
    raise SystemExit(1)


# ── Load PSI drift results ──

try:
    with open("drift_data.pkl", "rb") as f:
        drift_data = pickle.load(f)

    psi_results      = drift_data["psi_results"]
    any_drift        = drift_data["any_drift"]
    drifted_features = drift_data["drifted_features"]

    print("PSI Results:")
    for r in psi_results:
        if r["psi"] > 0:
            print(f"  {r['feature']:<40} {r['psi']:>8.4f}  {r['status']}")
    print(f"  Drift detected: {'YES' if any_drift else 'NO'}\n")

except FileNotFoundError:
    print("drift_data.pkl not found — run drift_detection.py first.")
    raise SystemExit(1)


# ── Champion model (XGBoost) — old model with pre-drift parameters ──

imbalance_g = neg_g / pos_g

xgb_champion = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=imbalance_g,
    use_label_encoder=False, eval_metric="auc",
    random_state=42, n_jobs=-1,
)
xgb_champion.fit(Xg_train_sc, yg_train_arr)
champion_res = evaluate_model("XGBoost (Champion)", xgb_champion, Xg_test_sc, yg_test_arr, threshold=0.5)


# ── PSI-based model selection ──

selected    = champion_res
verdict     = "STABLE (no drift)"
ratio       = 1.0
all_results = [champion_res]

if not any_drift:
    print("\nNo drift detected (PSI < 0.2) — champion XGBoost stays in production.")

else:
    print(f"\nDrift detected on: {drifted_features}")
    print("Tuning challenger SVM (RandomizedSearchCV)...\n")

    svm_search = RandomizedSearchCV(
        SVC(probability=True, class_weight='balanced', random_state=42),
        param_distributions={
            'kernel': ['rbf', 'poly'],
            'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
            'gamma': ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            'degree': [2, 3, 4],
        },
        n_iter=100, cv=5, scoring='accuracy',
        random_state=42, n_jobs=-1,
    )
    svm_search.fit(Xg_train_sc, yg_train_arr)
    print(f"Best params: {svm_search.best_params_}")
    print(f"Best CV accuracy: {svm_search.best_score_:.4f}")

    challenger_res = evaluate_model("SVM (Challenger)", svm_search.best_estimator_, Xg_test_sc, yg_test_arr)

    ratio, verdict = compare_predictions(champion_res["y_pred"], challenger_res["y_pred"])
    all_results = [champion_res, challenger_res]

    if verdict == "STABLE":
        selected = champion_res
    else:
        selected = challenger_res


# ── Final output ──

print_comparison(all_results, title="Final Comparison" + (" (Post-Drift)" if any_drift else ""))
print(f"\nSelected model: {selected['name']} | Verdict: {verdict}")

with open("german_credit_results.pkl", "wb") as f:
    pickle.dump({
        "all_results":    all_results,
        "selected_model": selected["name"],
        "verdict":        verdict,
        "agree_ratio":    ratio,
        "any_drift":      any_drift,
    }, f)

print("Saved german_credit_results.pkl")
