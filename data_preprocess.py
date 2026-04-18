import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings("ignore")

HOME_CREDIT_PATH = "data/application_train.csv"

NUM_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
]
CAT_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]
TARGET = "TARGET"

df_hc = pd.read_csv(HOME_CREDIT_PATH)
print(f"Loaded Home Credit: {df_hc.shape}")
print(f"Target distribution:\n{df_hc[TARGET].value_counts()}\n")

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
df = df_hc[ALL_FEATURES + [TARGET]].copy()

df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

num_imputer = SimpleImputer(strategy="median")
df[NUM_FEATURES] = num_imputer.fit_transform(df[NUM_FEATURES])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[CAT_FEATURES] = cat_imputer.fit_transform(df[CAT_FEATURES])

print(f"After imputation: {df.shape}, missing: {df.isnull().sum().sum()}")

df_encoded = pd.get_dummies(df, columns=CAT_FEATURES, drop_first=True)
print(f"After OHE: {df_encoded.shape}")

feature_cols = [c for c in df_encoded.columns if c != TARGET]
X = df_encoded[feature_cols].astype(float)
y = df_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Imbalance ratio: {y_train.value_counts()[0] / y_train.value_counts()[1]:.1f}:1")

with open("processed_data.pkl", "wb") as f:
    pickle.dump({
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "y_train":        y_train,
        "y_test":         y_test,
        "df":             df,
        "NUM_FEATURES":   NUM_FEATURES,
        "feature_cols":   feature_cols,
    }, f)

print("Saved processed_data.pkl")
