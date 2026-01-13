from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler

# =========================
# 0) AYARLAR
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "ml_out" / "logistic_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Threshold 0-1 aralığında olmalı
THRESHOLD = 0.85

# LR baseline parametreleri
LR_PARAMS = dict(
    penalty="l2",
    C=1.0,
    solver="saga",           # büyük veri + sparse-friendly
    max_iter=5000,
    n_jobs=-1,
    class_weight="balanced", # dengesiz sınıfı dengeler
    random_state=RANDOM_STATE,
)

# =========================
# 1) YARDIMCI FONKSİYONLAR
# =========================
def find_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "y", "target", "anomaly", "is_anomaly", "class"]
    for c in candidates:
        if c in df.columns:
            return c

    best = None
    best_score = -1
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            uniq = set(pd.Series(s.dropna().unique()).tolist())
            if uniq.issubset({0, 1}) and len(uniq) >= 1:
                score = 100 - len(s.dropna().unique())
                if score > best_score:
                    best_score = score
                    best = c

    if best is None:
        raise ValueError(
            "Label kolonu bulunamadı. CSV içinde 'label'/'y' gibi bir kolon olmalı "
            "veya 0/1 değerli bir hedef kolon olmalı."
        )
    return best


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str):
    y_train = train_df[label_col].astype(int).values
    y_test  = test_df[label_col].astype(int).values

    X_train_raw = train_df.drop(columns=[label_col])
    X_test_raw  = test_df.drop(columns=[label_col])

    X_train = X_train_raw.select_dtypes(include=[np.number]).copy()
    X_test  = X_test_raw.select_dtypes(include=[np.number]).copy()

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test  = X_test.fillna(med)

    feature_names = X_train.columns.tolist()
    return X_train.values.astype(np.float32), y_train, X_test.values.astype(np.float32), y_test, feature_names


def scale_train_test(X_train: np.ndarray, X_test: np.ndarray):
    # Zaman sızıntısı yok: scaler sadece train ile fit edilir
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def eval_binary(y_true: np.ndarray, y_proba: np.ndarray, threshold: float):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc = roc_auc_score(y_true, y_proba)
    pr  = average_precision_score(y_true, y_proba)

    err = 1.0 - acc

    return {
        "threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "error_rate": float(err),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }

# =========================
# 2) ANA AKIŞ
# =========================
def main():
    if not (0.0 <= THRESHOLD <= 1.0):
        raise ValueError("THRESHOLD 0 ile 1 arasında olmalı!")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    label_col = find_label_column(train_df)
    if label_col not in test_df.columns:
        raise ValueError(f"Test CSV içinde label kolonu yok: {label_col}")

    X_train, y_train, X_test, y_test, feat_names = prepare_xy(train_df, test_df, label_col)

    # Logistic Regression için ölçekleme önemli
    X_train_s, X_test_s, scaler = scale_train_test(X_train, X_test)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_s, y_train)

    proba_test = clf.predict_proba(X_test_s)[:, 1]
    metrics = eval_binary(y_test, proba_test, THRESHOLD)

    print("\n=== TEST RESULTS (Logistic Regression) ===")
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(
        f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | ErrorRate={metrics['error_rate']:.4f}"
    )
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC : {metrics['pr_auc']:.4f}")

    # Kayıt
    out = {
        "model": "LogisticRegression",
        "label_col": label_col,
        "lr_params": LR_PARAMS,
        "n_features": int(len(feat_names)),
        "metrics": metrics,
        "scaler": {
            "type": "StandardScaler",
            "fit_on": "train_only"
        },
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Katsayılar (yorumlanabilirlik)
    coef = clf.coef_.reshape(-1)
    coef_df = pd.DataFrame({
        "feature": feat_names,
        "coef": coef.astype(float),
        "abs_coef": np.abs(coef).astype(float),
    }).sort_values("abs_coef", ascending=False)

    coef_df.to_csv(OUT_DIR / "coefficients.csv", index=False, encoding="utf-8")
    coef_df.head(30).to_csv(OUT_DIR / "coefficients_top30.csv", index=False, encoding="utf-8")

    print(f"\n✅ Kayıt edildi: {OUT_DIR}")


if __name__ == "__main__":
    main()
