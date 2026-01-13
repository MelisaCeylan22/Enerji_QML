from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_sample_weight


# =========================
# 0) AYARLAR
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "ml_out" / "adaboost_tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TIME_COL = "timestamp"
LABEL_COL = "label"   # sende "label" kesin, istersen otomatik buldururuz

VAL_RATIO_WITHIN_TRAIN = 0.20   # train'in SON %20'si validation (time-based)

# Threshold seçimi
THRESHOLDS = np.arange(0.01, 1.00, 0.01)  # 0..1 aralığında tarama
BETA = 1.0  # 1.0=F1, 0.5=precision odaklı, 2.0=recall odaklı

# AdaBoost parametreleri (buradan oynayabilirsin)
ADA_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.08,
    random_state=RANDOM_STATE,
)

# Base estimator: stump yerine biraz daha güçlü ağaç çoğu zaman Recall'ı iyileştirir
BASE_ESTIMATOR = DecisionTreeClassifier(
    max_depth=3,              # 2-4 arası dene
    min_samples_leaf=50,      # aşırı ezberlemeyi azaltır
    random_state=RANDOM_STATE,
)


# =========================
# 1) Yardımcılar
# =========================
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    # numeric cast
    for c in df.columns:
        if c == TIME_COL:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # label 0/1
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def time_split_train_val(df: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * (1 - val_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def prepare_xy(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    # numeric features (timestamp hariç)
    drop_cols = [LABEL_COL]
    if TIME_COL in df_tr.columns:
        drop_cols.append(TIME_COL)

    Xtr = df_tr.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()
    Xva = df_val.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()
    Xte = df_te.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()

    # aynı feature setini garanti et
    common = Xtr.columns
    Xva = Xva[common]
    Xte = Xte[common]

    # impute (TRAIN medyanı ile)
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xva = Xva.fillna(med)
    Xte = Xte.fillna(med)

    ytr = df_tr[LABEL_COL].values.astype(np.int32)
    yva = df_val[LABEL_COL].values.astype(np.int32)
    yte = df_te[LABEL_COL].values.astype(np.int32)

    return Xtr.values.astype(np.float32), ytr, Xva.values.astype(np.float32), yva, Xte.values.astype(np.float32), yte, common.tolist()


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * (p * r) / denom if denom > 0 else 0.0


def scan_thresholds(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray, beta: float):
    rows = []
    best = None
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        fb   = fbeta(prec, rec, beta)
        err  = 1.0 - acc

        row = (float(t), int(tp), int(tn), int(fp), int(fn), float(acc), float(prec), float(rec), float(f1), float(fb), float(err))
        rows.append(row)
        if best is None or fb > best[9]:
            best = row
    return rows, best


def eval_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, t: float):
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    err  = float(1.0 - acc)

    try:
        roc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc = None
    try:
        pr = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr = None

    return dict(
        threshold=float(t),
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
        accuracy=acc, precision=prec, recall=rec, f1=f1, error_rate=err,
        roc_auc=roc, pr_auc=pr
    )


# =========================
# 2) MAIN
# =========================
def main():
    print("Loading datasets...")
    df_train_full = load_df(TRAIN_CSV)
    df_test = load_df(TEST_CSV)

    # train içinde time-based val
    df_tr, df_val = time_split_train_val(df_train_full, VAL_RATIO_WITHIN_TRAIN)

    Xtr, ytr, Xva, yva, Xte, yte, feat_names = prepare_xy(df_tr, df_val, df_test)

    # imbalance için sample_weight (AdaBoost class_weight almaz)
    wtr = compute_sample_weight(class_weight="balanced", y=ytr)

    # sklearn sürüm uyumu (estimator vs base_estimator)
    try:
        clf = AdaBoostClassifier(estimator=BASE_ESTIMATOR, **ADA_PARAMS)
    except TypeError:
        clf = AdaBoostClassifier(base_estimator=BASE_ESTIMATOR, **ADA_PARAMS)

    print("Training AdaBoost...")
    clf.fit(Xtr, ytr, sample_weight=wtr)

    print("Scoring VAL for threshold tuning...")
    va_proba = clf.predict_proba(Xva)[:, 1]

    rows, best = scan_thresholds(yva, va_proba, THRESHOLDS, beta=BETA)
    best_t, tp, tn, fp, fn, acc, prec, rec, f1, fb, err = best

    print(f"\nBest threshold on VAL by F{BETA}: {best_t:.2f}")
    print(f"VAL: TP={tp} TN={tn} FP={fp} FN={fn} | Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} F{BETA}={fb:.4f}")

    print("\nPredicting on TEST...")
    te_proba = clf.predict_proba(Xte)[:, 1]
    test_metrics = eval_at_threshold(yte, te_proba, best_t)

    print("\n=== TEST RESULTS (AdaBoost - tuned) ===")
    print(f"Threshold: {test_metrics['threshold']:.2f}  | selected by F{BETA} on VAL")
    print(f"TP={test_metrics['TP']}, TN={test_metrics['TN']}, FP={test_metrics['FP']}, FN={test_metrics['FN']}")
    print(
        f"Accuracy={test_metrics['accuracy']:.4f} | Precision={test_metrics['precision']:.4f} | "
        f"Recall={test_metrics['recall']:.4f} | F1={test_metrics['f1']:.4f} | ErrorRate={test_metrics['error_rate']:.4f}"
    )
    if test_metrics["roc_auc"] is not None:
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    if test_metrics["pr_auc"] is not None:
        print(f"Test PR-AUC : {test_metrics['pr_auc']:.4f}")

    # kayıtlar
    scan_df = pd.DataFrame(rows, columns=[
        "threshold","TP","TN","FP","FN","accuracy","precision","recall","f1",f"f{BETA}","error_rate"
    ])
    scan_df.to_csv(OUT_DIR / "threshold_scan_val.csv", index=False, encoding="utf-8")

    out = {
        "model": "AdaBoostClassifier(tuned)",
        "features": feat_names,
        "val_ratio_within_train": VAL_RATIO_WITHIN_TRAIN,
        "beta": BETA,
        "best_threshold_val": float(best_t),
        "ada_params": ADA_PARAMS,
        "base_estimator": {
            "type": "DecisionTreeClassifier",
            "max_depth": int(getattr(BASE_ESTIMATOR, "max_depth", -1)),
            "min_samples_leaf": int(getattr(BASE_ESTIMATOR, "min_samples_leaf", -1)),
        },
        "test_metrics": test_metrics,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # feature importance
    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feat_names,
            "importance": np.asarray(clf.feature_importances_, dtype=float),
        }).sort_values("importance", ascending=False)
        imp.to_csv(OUT_DIR / "feature_importances.csv", index=False, encoding="utf-8")

    print(f"\n✅ Kayıt edildi: {OUT_DIR}")


if __name__ == "__main__":
    main()
