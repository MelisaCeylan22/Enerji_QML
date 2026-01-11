from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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


# =========================
# 0) AYARLAR (BURAYI DÜZENLE)
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "cnn1d_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_RATIO_WITHIN_TRAIN = 0.20

# Sequence length (dakika)
LOOKBACK = 60

# Training
EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
SEED = 42

SHUFFLE_TRAIN_WINDOWS = True

THRESHOLDS = np.arange(0.05, 0.96, 0.05)

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

EPS = 1e-9


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


def fit_minmax(X: np.ndarray) -> dict:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return {"mins": mins, "maxs": maxs}


def transform_minmax(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = (maxs - mins)
    denom = np.where(denom < EPS, 1.0, denom)
    return (X - mins) / denom


def make_ts_dataset(X_scaled: np.ndarray, y: np.ndarray, lookback: int, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    """
    sequence i -> X[i : i+lookback]
    target i   -> y[i+lookback-1]
    """
    if len(X_scaled) < lookback:
        raise ValueError(f"Segment çok kısa: len={len(X_scaled)} < lookback={lookback}")

    targets = y[lookback - 1:]
    end_index = len(X_scaled) - lookback

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X_scaled,
        targets=targets,
        sequence_length=lookback,
        sequence_stride=1,
        sampling_rate=1,
        start_index=0,
        end_index=end_index,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    return ds


def scan_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        err = (fp + fn) / (tp + tn + fp + fn)
        rows.append((float(t), int(tp), int(tn), int(fp), int(fn), float(acc), float(prec), float(rec), float(f1), float(err)))
    return rows


def collect_preds(ds: tf.data.Dataset, model: tf.keras.Model) -> tuple[np.ndarray, np.ndarray]:
    y_true_list = []
    y_prob_list = []
    for batch_x, batch_y in ds:
        prob = model.predict(batch_x, verbose=0).ravel()
        y_prob_list.append(prob.astype(np.float32))
        y_true_list.append(batch_y.numpy().astype(np.int32))
    return np.concatenate(y_true_list), np.concatenate(y_prob_list)


# =========================
# 2) 1D CNN Model
# =========================
def build_cnn1d(lookback: int, n_features: int) -> tf.keras.Model:
    """
    Basit ve güçlü bir 1D CNN:
      - Conv -> Conv -> Pool
      - Conv -> GlobalAvgPool
      - Dense -> Sigmoid
    """
    inp = layers.Input(shape=(lookback, n_features))

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name="CNN1D")
    return model


# =========================
# 3) MAIN
# =========================
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading datasets...")
    df_train_full = load_dataset(TRAIN_CSV)
    df_test_full  = load_dataset(TEST_CSV)

    df_tr, df_val = time_split_train_val(df_train_full, VAL_RATIO_WITHIN_TRAIN)

    X_tr_raw = df_tr[FEATURE_COLS].values.astype(np.float32)
    y_tr_raw = df_tr[LABEL_COL].values.astype(np.int32)

    X_val_raw = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val_raw = df_val[LABEL_COL].values.astype(np.int32)

    X_te_raw = df_test_full[FEATURE_COLS].values.astype(np.float32)
    y_te_raw = df_test_full[LABEL_COL].values.astype(np.int32)

    print("Fitting MinMax on TRAIN only...")
    mm = fit_minmax(X_tr_raw)
    mins = mm["mins"].astype(np.float32)
    maxs = mm["maxs"].astype(np.float32)

    X_tr  = transform_minmax(X_tr_raw,  mins, maxs).astype(np.float32)
    X_val = transform_minmax(X_val_raw, mins, maxs).astype(np.float32)
    X_te  = transform_minmax(X_te_raw,  mins, maxs).astype(np.float32)

    print("Creating time-series datasets...")
    ds_tr  = make_ts_dataset(X_tr,  y_tr_raw,  LOOKBACK, BATCH_SIZE, shuffle=SHUFFLE_TRAIN_WINDOWS)
    ds_val = make_ts_dataset(X_val, y_val_raw, LOOKBACK, BATCH_SIZE, shuffle=False)
    ds_te  = make_ts_dataset(X_te,  y_te_raw,  LOOKBACK, BATCH_SIZE, shuffle=False)

    # class_weight: sequence hedefleri y[lookback-1:]
    y_tr_seq = y_tr_raw[LOOKBACK - 1:]
    classes = np.unique(y_tr_seq)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr_seq)
    class_weight = {int(c): float(wi) for c, wi in zip(classes, w)}
    print("class_weight:", class_weight)

    model = build_cnn1d(LOOKBACK, len(FEATURE_COLS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("\nTraining 1D-CNN...")
    history = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    print("\nPredicting on VAL for threshold tuning...")
    y_val_true, y_val_prob = collect_preds(ds_val, model)

    try:
        val_roc = float(roc_auc_score(y_val_true, y_val_prob))
    except Exception:
        val_roc = None
    try:
        val_pr = float(average_precision_score(y_val_true, y_val_prob))
    except Exception:
        val_pr = None

    results = scan_thresholds(y_val_true, y_val_prob, THRESHOLDS)
    best = max(results, key=lambda r: r[8])
    best_t = best[0]

    print("\n=== VAL THRESHOLD SCAN (top 5 by F1) ===")
    for r in sorted(results, key=lambda x: x[8], reverse=True)[:5]:
        t, tp, tn, fp, fn, acc, prec, rec, f1, err = r
        print(f"t={t:.2f} | TP={tp} TN={tn} FP={fp} FN={fn} | Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    print(f"\nBest threshold (VAL by F1): {best_t:.2f}")
    if val_roc is not None:
        print(f"VAL ROC-AUC: {val_roc:.4f}")
    if val_pr is not None:
        print(f"VAL PR-AUC : {val_pr:.4f}")

    print("\nPredicting on TEST...")
    y_te_true, y_te_prob = collect_preds(ds_te, model)
    y_te_pred = (y_te_prob >= best_t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te_true, y_te_pred).ravel()
    acc  = float(accuracy_score(y_te_true, y_te_pred))
    prec = float(precision_score(y_te_true, y_te_pred, zero_division=0))
    rec  = float(recall_score(y_te_true, y_te_pred, zero_division=0))
    f1   = float(f1_score(y_te_true, y_te_pred, zero_division=0))
    err  = float((fp + fn) / (tp + tn + fp + fn))

    try:
        test_roc = float(roc_auc_score(y_te_true, y_te_prob))
    except Exception:
        test_roc = None
    try:
        test_pr = float(average_precision_score(y_te_true, y_te_prob))
    except Exception:
        test_pr = None

    print("\n=== TEST RESULTS (1D-CNN) ===")
    print(f"Threshold: {best_t:.2f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | ErrorRate={err:.4f}")
    if test_roc is not None:
        print(f"Test ROC-AUC: {test_roc:.4f}")
    if test_pr is not None:
        print(f"Test PR-AUC : {test_pr:.4f}")

    # save artifacts
    model_path = OUT_DIR / "cnn1d_model.keras"
    model.save(model_path)

    scaler_path = OUT_DIR / "minmax_params.json"
    eval_path = OUT_DIR / "eval_summary.json"

    minmax_payload = {
        "feature_cols": FEATURE_COLS,
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "eps": EPS,
        "lookback": LOOKBACK,
    }
    scaler_path.write_text(json.dumps(minmax_payload, indent=2), encoding="utf-8")

    eval_payload = {
        "config": {
            "train_csv": str(TRAIN_CSV),
            "test_csv": str(TEST_CSV),
            "val_ratio_within_train": VAL_RATIO_WITHIN_TRAIN,
            "lookback": LOOKBACK,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "shuffle_train_windows": SHUFFLE_TRAIN_WINDOWS,
            "cnn1d": {
                "conv1": {"filters": 64, "kernel_size": 5},
                "conv2": {"filters": 64, "kernel_size": 5},
                "pool": {"type": "MaxPooling1D", "pool_size": 2},
                "conv3": {"filters": 128, "kernel_size": 3},
                "head": {"dense": 64},
                "dropout": 0.2,
            },
        },
        "class_weight": class_weight,
        "best_threshold_val_f1": best_t,
        "val_metrics": {"roc_auc": val_roc, "pr_auc": val_pr},
        "test_metrics": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
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
