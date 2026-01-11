from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import joblib


# =========================
# 0) AYARLAR (BURAYI DÜZENLE)
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "svm_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_RATIO_WITHIN_TRAIN = 0.20

# SVM params (başlangıç için iyi)
C = 5.0
GAMMA = "scale"   # veya 0.1 gibi sabit deneyebilirsin
KERNEL = "rbf"

# Threshold tarama (decision_function skoru üzerinde)
# SVM skorları -/+ geniş aralıkta olabilir; percentil tabanlı tarama yapacağız.
N_THRESH = 60

SEED = 42

FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
LABEL_COL = "label"
TIME_COL = "timestamp"


# =========================
# 1) Yardımcılar
# =========================
def load_dataset(path: Path) -> pd.DataFrame:
    usecols = [TIME_COL] + FEATURE_COLS + [LABEL_COL]
    df = pd.read_csv(path, usecols=usecols)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    for c in FEATURE_COLS + [LABEL_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def time_split_train_val(df_train: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df_train)
    cut = int(n * (1 - val_ratio))
    return df_train.iloc[:cut].copy(), df_train.iloc[cut:].copy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    denom = tp + tn + fp + fn
    err  = float((fp + fn) / denom) if denom > 0 else 0.0

    return tp, tn, fp, fn, acc, prec, rec, f1, err


def scan_thresholds(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray):
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp, tn, fp, fn, acc, prec, rec, f1, err = compute_metrics(y_true, y_pred)
        rows.append((float(t), tp, tn, fp, fn, acc, prec, rec, f1, err))
    return rows


# =========================
# 2) MAIN
# =========================
def main():
    print("Loading datasets...")
    df_train_full = load_dataset(TRAIN_CSV)
    df_test_full  = load_dataset(TEST_CSV)

    df_tr, df_val = time_split_train_val(df_train_full, VAL_RATIO_WITHIN_TRAIN)

    X_tr_raw = df_tr[FEATURE_COLS].values.astype(np.float32)
    y_tr = df_tr[LABEL_COL].values.astype(np.int32)

    X_val_raw = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val = df_val[LABEL_COL].values.astype(np.int32)

    X_te_raw = df_test_full[FEATURE_COLS].values.astype(np.float32)
    y_te = df_test_full[LABEL_COL].values.astype(np.int32)

    print("Fitting scaler on TRAIN only...")
    scaler = MinMaxScaler()
    X_tr  = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_te  = scaler.transform(X_te_raw)

    print("Training SVM...")
    svm = SVC(
        kernel=KERNEL,
        C=C,
        gamma=GAMMA,
        class_weight="balanced",
        probability=False,   # decision_function kullanacağız
        random_state=SEED,
    )
    svm.fit(X_tr, y_tr)

    # validation scores
    val_scores = svm.decision_function(X_val)

    # threshold candidates: val score dağılımından percentil ile (daha sağlam)
    lo, hi = np.percentile(val_scores, 1), np.percentile(val_scores, 99)
    thresholds = np.linspace(lo, hi, N_THRESH)

    # AUC metrics (score ile)
    try:
        val_roc = float(roc_auc_score(y_val, val_scores))
    except Exception:
        val_roc = None
    try:
        val_pr = float(average_precision_score(y_val, val_scores))
    except Exception:
        val_pr = None

    results = scan_thresholds(y_val, val_scores, thresholds)
    best = max(results, key=lambda r: r[8])  # F1 max
    best_t = best[0]

    print("\n=== VAL THRESHOLD SCAN (top 5 by F1) ===")
    for r in sorted(results, key=lambda x: x[8], reverse=True)[:5]:
        t, tp, tn, fp, fn, acc, prec, rec, f1, err = r
        print(f"t={t:.4f} | TP={tp} TN={tn} FP={fp} FN={fn} | Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    print(f"\nBest threshold (VAL by F1): {best_t:.4f}")
    if val_roc is not None:
        print(f"VAL ROC-AUC: {val_roc:.4f}")
    if val_pr is not None:
        print(f"VAL PR-AUC : {val_pr:.4f}")

    # test
    print("\nPredicting on TEST...")
    te_scores = svm.decision_function(X_te)
    y_te_pred = (te_scores >= best_t).astype(int)

    tp, tn, fp, fn, acc, prec, rec, f1, err = compute_metrics(y_te, y_te_pred)

    try:
        test_roc = float(roc_auc_score(y_te, te_scores))
    except Exception:
        test_roc = None
    try:
        test_pr = float(average_precision_score(y_te, te_scores))
    except Exception:
        test_pr = None

    print("\n=== TEST RESULTS (SVM) ===")
    print(f"Threshold: {best_t:.4f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | ErrorRate={err:.4f}")
    if test_roc is not None:
        print(f"Test ROC-AUC: {test_roc:.4f}")
    if test_pr is not None:
        print(f"Test PR-AUC : {test_pr:.4f}")

    # save
    model_path = OUT_DIR / "svm_model.joblib"
    scaler_path = OUT_DIR / "scaler.joblib"
    eval_path = OUT_DIR / "eval_summary.json"

    joblib.dump(svm, model_path)
    joblib.dump(scaler, scaler_path)

    eval_payload = {
        "config": {
            "train_csv": str(TRAIN_CSV),
            "test_csv": str(TEST_CSV),
            "val_ratio_within_train": VAL_RATIO_WITHIN_TRAIN,
            "kernel": KERNEL,
            "C": C,
            "gamma": str(GAMMA),
            "class_weight": "balanced",
            "threshold_scan": {"method": "linspace(percentile 1..99 of val scores)", "n_thresh": N_THRESH},
            "seed": SEED,
        },
        "best_threshold_val_f1": float(best_t),
        "val_metrics": {"roc_auc": val_roc, "pr_auc": val_pr},
        "test_metrics": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "error_rate": err,
            "roc_auc": test_roc, "pr_auc": test_pr,
        },
        "threshold_scan_val": results,
    }
    eval_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", model_path)
    print(" -", scaler_path)
    print(" -", eval_path)


if __name__ == "__main__":
    main()
