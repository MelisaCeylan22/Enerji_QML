from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# =========================
# 0) AYARLAR
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "ml_out" / "random_forest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Threshold 0-1 aralığında olmalı
THRESHOLD = 0.85

# RF hiperparametreleri (baseline)
RF_PARAMS = dict(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",  # dengesiz sınıf için iyi baseline
)

# =========================
# 1) YARDIMCI FONKSİYONLAR
# =========================
def find_label_column(df: pd.DataFrame) -> str:
    """
    Label kolonunu otomatik bulmaya çalışır.
    Önce bilinen isimler, yoksa {0,1} değerli en iyi adayı seçer.
    """
    candidates = ["label", "y", "target", "anomaly", "is_anomaly", "class"]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: 0/1 içeren kolon ara (en az benzersiz değerli olanlardan)
    best = None
    best_score = -1
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            uniq = set(pd.Series(s.dropna().unique()).tolist())
            if uniq.issubset({0, 1}) and len(uniq) >= 1:
                # daha "label gibi" olanı seçmek için basit skor:
                # 0/1 ve az sayıda unique -> iyi
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

    # Sadece sayısal kolonlar (projenizde zaten çoğu numeric olmalı)
    X_train = X_train_raw.select_dtypes(include=[np.number]).copy()
    X_test  = X_test_raw.select_dtypes(include=[np.number]).copy()

    # Eksik değerleri train median ile doldur (leakage yok)
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test  = X_test.fillna(med)

    # float32 bellek/donanım açısından daha rahat
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    feature_names = X_train.columns.tolist()
    return X_train.values, y_train, X_test.values, y_test, feature_names


def eval_binary(y_true: np.ndarray, y_proba: np.ndarray, threshold: float):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC metrikleri threshold'dan bağımsız (skor üzerinden)
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

    # Model
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)

    # Olasılık (pozitif sınıf = 1)
    proba_test = clf.predict_proba(X_test)[:, 1]

    # Değerlendirme
    metrics = eval_binary(y_test, proba_test, THRESHOLD)

    print("\n=== TEST RESULTS (Random Forest) ===")
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(
        f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | ErrorRate={metrics['error_rate']:.4f}"
    )
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC : {metrics['pr_auc']:.4f}")

    # Kayıt: metrikler + model paramları
    out = {
        "model": "RandomForestClassifier",
        "label_col": label_col,
        "rf_params": RF_PARAMS,
        "n_features": int(len(feat_names)),
        "metrics": metrics,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Feature importance (varsa)
    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feat_names,
            "importance": clf.feature_importances_.astype(float),
        }).sort_values("importance", ascending=False)
        imp.to_csv(OUT_DIR / "feature_importances.csv", index=False, encoding="utf-8")
        imp.head(30).to_csv(OUT_DIR / "feature_importances_top30.csv", index=False, encoding="utf-8")

    print(f"\n✅ Kayıt edildi: {OUT_DIR}")


if __name__ == "__main__":
    main()
