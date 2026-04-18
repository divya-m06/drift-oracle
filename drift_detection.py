"""
PSI Drift Detection — simulates an economic shock and measures
feature distribution shift using Population Stability Index.
Computes PSI on both numeric and one-hot encoded features.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

df           = data["df"]
NUM_FEATURES = data["NUM_FEATURES"]
TARGET       = "TARGET"


def compute_psi(expected, actual, bins=10):
    """PSI between two distributions. Handles both continuous and binary features."""
    n_unique = len(np.unique(expected))

    if n_unique <= 2:
        # Binary feature: compare proportions directly
        p_exp = np.clip(expected.mean(), 1e-4, 1 - 1e-4)
        p_act = np.clip(actual.mean(),   1e-4, 1 - 1e-4)
        expected_perc = np.array([1 - p_exp, p_exp])
        actual_perc   = np.array([1 - p_act, p_act])
    else:
        breakpoints = np.linspace(0, 100, bins + 1)
        bin_edges = np.unique(np.percentile(expected, breakpoints))
        if len(bin_edges) < 2:
            return 0.0
        expected_perc = np.histogram(expected, bins=bin_edges)[0] / len(expected)
        actual_perc   = np.histogram(actual,   bins=bin_edges)[0] / len(actual)

    expected_perc = np.where(expected_perc == 0, 1e-4, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-4, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def psi_status(psi):
    if psi > 0.2:  return "DRIFT"
    if psi > 0.1:  return "WARN"
    return "STABLE"


# Prepare original and shocked data (before OHE)
original = df.drop(TARGET, axis=1, errors="ignore").copy()
shocked  = original.copy()
shocked["AMT_INCOME_TOTAL"] *= 0.7   # income drops 30%
shocked["AMT_CREDIT"]       *= 1.3   # loan amounts increase 30%
shocked["AMT_ANNUITY"]      *= 1.2   # EMI increases 20%

# Apply one-hot encoding to both
cat_cols = original.select_dtypes(include="object").columns.tolist()
original_ohe = pd.get_dummies(original, columns=cat_cols, drop_first=True).astype(float)
shocked_ohe  = pd.get_dummies(shocked,  columns=cat_cols, drop_first=True).astype(float)

# Align columns
common_cols = sorted(original_ohe.columns.intersection(shocked_ohe.columns))
original_ohe = original_ohe[common_cols]
shocked_ohe  = shocked_ohe[common_cols]

# Compute PSI on all features (numeric + OHE)
psi_results = []
any_drift   = False

print(f"{'Feature':<40} {'PSI':>8}  Status")
print("-" * 58)

# Numeric features first
for col in NUM_FEATURES:
    psi    = compute_psi(original_ohe[col].values, shocked_ohe[col].values)
    status = psi_status(psi)
    print(f"{col:<40} {psi:>8.4f}  {status}")
    psi_results.append({"feature": col, "psi": round(psi, 4), "status": status, "drift": psi > 0.2})
    if psi > 0.2:
        any_drift = True

# OHE (categorical) features
ohe_cols = [c for c in common_cols if c not in NUM_FEATURES]
for col in ohe_cols:
    psi    = compute_psi(original_ohe[col].values, shocked_ohe[col].values)
    status = psi_status(psi)
    print(f"{col:<40} {psi:>8.4f}  {status}")
    psi_results.append({"feature": col, "psi": round(psi, 4), "status": status, "drift": psi > 0.2})
    if psi > 0.2:
        any_drift = True

drifted_features = [r["feature"] for r in psi_results if r["drift"]]
print(f"\nTotal features checked: {len(common_cols)} (numeric + OHE)")
print(f"Drifted features: {drifted_features}")
print(f"Overall drift: {'YES' if any_drift else 'NO'}")

# Save
psi_df = pd.DataFrame(psi_results)
psi_df.to_csv("psi_metrics.csv", index=False)

with open("drift_data.pkl", "wb") as f:
    pickle.dump({
        "psi_results":      psi_results,
        "psi_df":           psi_df,
        "drift_df":         shocked_ohe,
        "original_df":      original_ohe,
        "any_drift":        any_drift,
        "drifted_features": drifted_features,
    }, f)

print("Saved psi_metrics.csv and drift_data.pkl")