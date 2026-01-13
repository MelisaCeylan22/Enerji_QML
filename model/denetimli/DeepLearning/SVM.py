from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
import joblib


# =========================
# 0) AYARLAR
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "svm_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_RATIO_WITHIN_TRAIN = 0.20
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

# Time features
LAGS = [1, 5, 15, 60]
ROLL_FEATS = ["Global_active_power", "Voltage", "Global_intensity"]
ROLL_WINDOWS = [5, 15, 60]

# Streaming / batch
CHUNK_SIZE = 200_000
BATCH_SIZE = 50_000
EPOCHS = 3

# Logistic SGD (base classifier)
LOSS = "log_loss"
PENALTY = "elasticnet"
L1_RATIO = 0.10
ALPHA = 1e-4
LEARNING_RATE = "adaptive"
ETA0 = 0.01

# Threshold tuning (0..1 probability)
N_THRESH = 101
FBETA = 0.5
THRESH_MIN = 0.001
THRESH_MAX = 0.999

EPS = 1e-6


# =========================
# 1) Yardımcılar
# =========================
def load_csv_basic(path: Path) -> pd.DataFrame:
    dtypes = {c: "float32" for c in FEATURE_COLS}
    dtypes[LABEL_COL] = "int8"

    df = pd.read_csv(path, usecols=[TIME_COL] + FEATURE_COLS + [LABEL_COL], dtype=dtypes)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype("int8")

    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).reset_index(drop=True)
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(TIME_COL).reset_index(drop=True).copy()

    # lag
    for col in FEATURE_COLS:
        base = df[col].astype("float32")
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = base.shift(lag).astype("float32")

    # rolling (past-only)
    for col in ROLL_FEATS:
        s = df[col].astype("float32").shift(1)
        for w in ROLL_WINDOWS:
            mu = s.rolling(w, min_periods=w).mean().astype("float32")
            sd = s.rolling(w, min_periods=w).std().astype("float32")

            df[f"{col}_roll{w}_mean"] = mu
            df[f"{col}_roll{w}_std"]  = sd

            cur = df[col].astype("float32")
            delta = (cur - mu).astype("float32")
            z = (delta / (sd + EPS)).astype("float32")

            df[f"{col}_roll{w}_delta"] = delta
            df[f"{col}_roll{w}_z"]     = z

    return df


def time_split_train_val(df_train: pd.DataFrame, val_ratio: float):
    n = len(df_train)
    cut = int(n * (1 - val_ratio))
    return df_train.iloc[:cut], df_train.iloc[cut:]


def fbeta(prec: float, rec: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * prec + rec
    return (1 + b2) * (prec * rec) / denom if denom > 0 else 0.0


def confusion_2x2(y_true: np.ndarray, y_pred: np.ndarray):
    # labels sabit: [0,1] -> ravel güvenli
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()  # tn, fp, fn, tp


def scan_thresholds_prob(y_true: np.ndarray, prob: np.ndarray, beta: float, n_thresh: int):
    thresholds = np.linspace(THRESH_MIN, THRESH_MAX, n_thresh)

    best = None
    rows = []
    for t in thresholds:
        y_pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_2x2(y_true, y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fb = fbeta(prec, rec, beta)
        err = (fp + fn) / (tp + tn + fp + fn)

        row = (float(t), int(tp), int(tn), int(fp), int(fn),
               float(acc), float(prec), float(rec), float(f1), float(fb), float(err))
        rows.append(row)
        if best is None or fb > best[9]:
            best = row

    return rows, best


def stream_scores(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler, clf: SGDClassifier, batch_size: int):
    """Sadece 1D score + label döndürür (RAM-safe)."""
    scores = []
    ys = []
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        part = df.iloc[start:end]
        X = scaler.transform(part[feature_cols].to_numpy(dtype=np.float32, copy=False))
        y = part[LABEL_COL].to_numpy(dtype=np.int32, copy=False)
        s = clf.decision_function(X).astype(np.float32)
        scores.append(s)
        ys.append(y)
    return np.concatenate(ys), np.concatenate(scores)


def platt_calibrate(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Platt scaling: p = sigmoid(a*score + b)
    VAL üzerinde fit edilir.
    """
    cal = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None,
    )
    cal.fit(scores.reshape(-1, 1), y)
    return cal


def apply_calibrator(cal: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    prob = cal.predict_proba(scores.reshape(-1, 1))[:, 1].astype(np.float32)
    # (opsiyonel) uçları kırp
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return prob


# =========================
# 2) MAIN
# =========================
def main():
    np.random.seed(SEED)

    print("Loading datasets...")
    df_train = load_csv_basic(TRAIN_CSV)
    df_test  = load_csv_basic(TEST_CSV)

    warmup = max(max(LAGS), max(ROLL_WINDOWS))

    # -------------------------
    # A) TRAIN features
    # -------------------------
    print("Computing TRAIN lag/rolling features...")
    df_train_f = compute_time_features(df_train)
    df_train_f = df_train_f.iloc[warmup:]  # ilk warmup satırları at

    # -------------------------
    # B) TEST features (train tail + test)
    # -------------------------
    print("Computing TEST lag/rolling features (using train tail as history)...")
    tail = df_train.tail(warmup).copy()
    df_test_ext = pd.concat([tail, df_test], axis=0, ignore_index=True)
    df_test_ext = compute_time_features(df_test_ext)
    df_test_f = df_test_ext.iloc[warmup:]  # tail kısmı çöpe, kalan test

    feature_cols = [c for c in df_train_f.columns if c not in [TIME_COL, LABEL_COL]]

    # train içinde val ayır
    df_tr, df_val = time_split_train_val(df_train_f, VAL_RATIO_WITHIN_TRAIN)

    # class weights (train)
    classes = np.array([0, 1], dtype=int)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=df_tr[LABEL_COL].to_numpy())
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print("class_weight:", class_weight)

    scaler = StandardScaler(with_mean=True, with_std=True)

    clf = SGDClassifier(
        loss=LOSS,
        penalty=PENALTY,
        l1_ratio=L1_RATIO,
        alpha=ALPHA,
        learning_rate=LEARNING_RATE,
        eta0=ETA0,
        max_iter=1,
        tol=None,
        random_state=SEED,
        average=True,
    )

    # ---------- 1) scaler partial_fit (TRAIN only) ----------
    print("Fitting StandardScaler incrementally on TRAIN only...")
    n_tr = len(df_tr)
    for start in range(0, n_tr, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_tr)
        X_chunk = df_tr.iloc[start:end][feature_cols].to_numpy(dtype=np.float32, copy=False)
        scaler.partial_fit(X_chunk)

    # ---------- 2) model partial_fit ----------
    print("Training SGD (streaming, logistic)...")
    for ep in range(1, EPOCHS + 1):
        for start in range(0, n_tr, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_tr)
            part = df_tr.iloc[start:end]

            Xb = scaler.transform(part[feature_cols].to_numpy(dtype=np.float32, copy=False))
            yb = part[LABEL_COL].to_numpy(dtype=np.int32, copy=False)
            wb = np.where(yb == 1, class_weight[1], class_weight[0]).astype(np.float32)

            if ep == 1 and start == 0:
                clf.partial_fit(Xb, yb, classes=classes, sample_weight=wb)
            else:
                clf.partial_fit(Xb, yb, sample_weight=wb)

        print(f"Epoch {ep}/{EPOCHS} done.")

    # ---------- 3) VAL: score -> calibration -> prob threshold ----------
    print("Scoring VAL (decision scores) + calibrating to probabilities...")
    y_val, val_score = stream_scores(df_val, feature_cols, scaler, clf, batch_size=BATCH_SIZE)

    # Calibration (VAL üstünden)
    calibrator = platt_calibrate(val_score, y_val)
    val_prob = apply_calibrator(calibrator, val_score)  # 0..1

    # AUC (prob üzerinden)
    try:
        val_roc = float(roc_auc_score(y_val, val_prob))
    except Exception:
        val_roc = None
    try:
        val_pr = float(average_precision_score(y_val, val_prob))
    except Exception:
        val_pr = None

    rows, best = scan_thresholds_prob(y_val, val_prob, beta=FBETA, n_thresh=N_THRESH)
    best_t, tp, tn, fp, fn, acc, prec, rec, f1, fb, err = best

    print("\n=== VAL THRESHOLD SCAN (top 5 by Fbeta, PROB 0..1) ===")
    for r in sorted(rows, key=lambda r: r[9], reverse=True)[:5]:
        t, tp, tn, fp, fn, acc, prec, rec, f1, fb, err = r
        print(f"t={t:.4f} | TP={tp} TN={tn} FP={fp} FN={fn} | Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} F{FBETA}={fb:.4f}")

    print(f"\nBest threshold (VAL by F{FBETA}, PROB): {best_t:.4f}")
    if val_roc is not None:
        print(f"VAL ROC-AUC: {val_roc:.4f}")
    if val_pr is not None:
        print(f"VAL PR-AUC : {val_pr:.4f}")

    # ---------- 4) TEST ----------
    print("Scoring TEST (decision scores -> calibrated probabilities)...")
    y_te, te_score = stream_scores(df_test_f, feature_cols, scaler, clf, batch_size=BATCH_SIZE)
    te_prob = apply_calibrator(calibrator, te_score)

    y_te_pred = (te_prob >= best_t).astype(int)

    tn, fp, fn, tp = confusion_2x2(y_te, y_te_pred)
    acc  = float(accuracy_score(y_te, y_te_pred))
    prec = float(precision_score(y_te, y_te_pred, zero_division=0))
    rec  = float(recall_score(y_te, y_te_pred, zero_division=0))
    f1   = float(f1_score(y_te, y_te_pred, zero_division=0))
    fb   = float(fbeta(prec, rec, FBETA))
    err  = float((fp + fn) / (tp + tn + fp + fn))

    try:
        test_roc = float(roc_auc_score(y_te, te_prob))
    except Exception:
        test_roc = None
    try:
        test_pr = float(average_precision_score(y_te, te_prob))
    except Exception:
        test_pr = None

    print("\n=== TEST RESULTS (Streaming Logistic-SGD, PROB threshold 0..1) ===")
    print(f"Threshold(prob): {best_t:.4f}  | selected by F{FBETA}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | F{FBETA}={fb:.4f} | ErrorRate={err:.4f}")
    if test_roc is not None:
        print(f"Test ROC-AUC: {test_roc:.4f}")
    if test_pr is not None:
        print(f"Test PR-AUC : {test_pr:.4f}")

    # Save
    model_path = OUT_DIR / "sgd_logistic_streaming_ram_safe.joblib"
    scaler_path = OUT_DIR / "scaler_streaming_ram_safe.joblib"
    cal_path = OUT_DIR / "platt_calibrator.joblib"
    eval_path = OUT_DIR / "sgd_logistic_streaming_ram_safe_eval.json"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(calibrator, cal_path)

    eval_payload = {
        "config": {
            "train_csv": str(TRAIN_CSV),
            "test_csv": str(TEST_CSV),
            "val_ratio_within_train": VAL_RATIO_WITHIN_TRAIN,
            "time_features": {"lags": LAGS, "roll_feats": ROLL_FEATS, "roll_windows": ROLL_WINDOWS},
            "model": {
                "type": "SGDClassifier(log_loss)",
                "penalty": PENALTY,
                "l1_ratio": L1_RATIO,
                "alpha": ALPHA,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "eta0": ETA0,
                "average": True,
            },
            "threshold": {"selection": f"F{FBETA} on VAL (calibrated PROB 0..1)", "n_thresh": N_THRESH, "range": [THRESH_MIN, THRESH_MAX]},
            "seed": SEED,
        },
        "best_threshold_prob": float(best_t),
        "val_metrics": {"roc_auc": val_roc, "pr_auc": val_pr},
        "test_metrics": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, f"f{FBETA}": fb,
            "error_rate": err, "roc_auc": test_roc, "pr_auc": test_pr,
        },
    }
    eval_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", model_path)
    print(" -", scaler_path)
    print(" -", cal_path)
    print(" -", eval_path)


if __name__ == "__main__":
    main()
