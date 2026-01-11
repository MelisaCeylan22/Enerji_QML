from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model

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

OUT_DIR = DATA_DIR / "deepsvdd_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_RATIO_WITHIN_TRAIN = 0.20
LOOKBACK = 60

EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
SEED = 42

SHUFFLE_TRAIN_WINDOWS = True

# Threshold taraması
N_THRESH = 60

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

# Center (c) hesaplamak için kaç batch?
CENTER_BATCHES = 200


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
    return {"mins": X.min(axis=0), "maxs": X.max(axis=0)}


def transform_minmax(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = (maxs - mins)
    denom = np.where(denom < EPS, 1.0, denom)
    return (X - mins) / denom


def make_ts_dataset(
    X_scaled: np.ndarray,
    y: np.ndarray,
    lookback: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    sequence i -> X[i : i+lookback]
    target i   -> y[i+lookback-1]
    """
    if len(X_scaled) < lookback:
        raise ValueError(f"Segment çok kısa: len={len(X_scaled)} < lookback={lookback}")

    targets = y[lookback - 1:]
    end_index = len(X_scaled) - lookback

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X_scaled.astype(np.float32),
        targets=targets.astype(np.int32),
        sequence_length=lookback,
        sequence_stride=1,
        sampling_rate=1,
        start_index=0,
        end_index=end_index,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    return ds


def scan_thresholds(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray):
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)  # score yüksek => anomali
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        err = (fp + fn) / (tp + tn + fp + fn)
        rows.append((float(t), int(tp), int(tn), int(fp), int(fn), float(acc), float(prec), float(rec), float(f1), float(err)))
    return rows


# =========================
# 2) FIX: normal filtreleme (batch BEFORE değil AFTER)
# =========================
def normal_only_x(ds_xy_batched: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    """
    ds_xy_batched: (batch_x, batch_y)
    -> unbatch -> filter(y==0 scalar) -> map(x) -> batch -> prefetch
    """
    ds = ds_xy_batched.unbatch()
    ds = ds.filter(lambda x, y: tf.equal(y, 0))  # burada y scalar -> bool scalar döner
    ds = ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def normal_only_xy(ds_xy_batched: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    """
    Val loss için: normal-only (x,y) dataset
    """
    ds = ds_xy_batched.unbatch()
    ds = ds.filter(lambda x, y: tf.equal(y, 0))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =========================
# 3) DeepSVDD Encoder
# =========================
def build_encoder(lookback: int, n_features: int, rep_dim: int = 32) -> tf.keras.Model:
    inp = layers.Input(shape=(lookback, n_features))

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    z = layers.Dense(rep_dim, activation=None, name="rep")(x)

    return Model(inp, z, name="DeepSVDD_Encoder")


def compute_center_c(encoder: tf.keras.Model, ds_x: tf.data.Dataset, max_batches: int | None) -> np.ndarray:
    n = 0
    mean = None

    for i, batch_x in enumerate(ds_x):
        if (max_batches is not None) and (i >= max_batches):
            break
        z = encoder(batch_x, training=False).numpy().astype(np.float32)

        if mean is None:
            mean = z.mean(axis=0)
            n = z.shape[0]
        else:
            b = z.shape[0]
            mean = (mean * n + z.sum(axis=0)) / (n + b)
            n += b

    if mean is None:
        raise RuntimeError("Center hesaplanamadı: normal pencere yok gibi görünüyor.")
    return mean.astype(np.float32)


def train_deepsvdd(
    encoder: tf.keras.Model,
    ds_train_x: tf.data.Dataset,
    ds_val_xy_normal: tf.data.Dataset,
    center_c: np.ndarray,
    epochs: int,
    lr: float,
    patience: int = 3,
) -> tuple[list[float], list[float]]:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    c = tf.convert_to_tensor(center_c, dtype=tf.float32)

    best_val = np.inf
    best_weights = None
    wait = 0

    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        # train
        tl = []
        for batch_x in ds_train_x:
            with tf.GradientTape() as tape:
                z = encoder(batch_x, training=True)
                dist = tf.reduce_sum(tf.square(z - c), axis=1)
                loss = tf.reduce_mean(dist)
                if encoder.losses:
                    loss = loss + tf.add_n(encoder.losses)

            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
            tl.append(float(loss.numpy()))
        train_loss = float(np.mean(tl)) if tl else float("nan")
        train_losses.append(train_loss)

        # val (normal only)
        vl = []
        for batch_x, _ in ds_val_xy_normal:
            z = encoder(batch_x, training=False)
            dist = tf.reduce_sum(tf.square(z - c), axis=1)
            vl.append(float(tf.reduce_mean(dist).numpy()))
        val_loss = float(np.mean(vl)) if vl else float("nan")
        val_losses.append(val_loss)

        print(f"Epoch {ep:02d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_weights = encoder.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"EarlyStopping: best_val={best_val:.6f}")
                break

    if best_weights is not None:
        encoder.set_weights(best_weights)

    return train_losses, val_losses


def collect_scores(ds_xy: tf.data.Dataset, encoder: tf.keras.Model, center_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_list, scores_list = [], []
    c = tf.convert_to_tensor(center_c, dtype=tf.float32)

    for batch_x, batch_y in ds_xy:
        z = encoder(batch_x, training=False)
        s = tf.reduce_sum(tf.square(z - c), axis=1)
        scores_list.append(s.numpy().astype(np.float32))
        y_true_list.append(batch_y.numpy().astype(np.int32))

    return np.concatenate(y_true_list), np.concatenate(scores_list)


# =========================
# 4) MAIN
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
    ds_tr_xy  = make_ts_dataset(X_tr,  y_tr_raw,  LOOKBACK, BATCH_SIZE, shuffle=SHUFFLE_TRAIN_WINDOWS)
    ds_val_xy = make_ts_dataset(X_val, y_val_raw, LOOKBACK, BATCH_SIZE, shuffle=False)
    ds_te_xy  = make_ts_dataset(X_te,  y_te_raw,  LOOKBACK, BATCH_SIZE, shuffle=False)

    # ✅ FIX'li normal filtre
    ds_tr_x = normal_only_x(ds_tr_xy, batch_size=BATCH_SIZE)
    ds_val_xy_norm = normal_only_xy(ds_val_xy, batch_size=BATCH_SIZE)

    encoder = build_encoder(LOOKBACK, len(FEATURE_COLS), rep_dim=32)

    print("Computing center c from NORMAL train windows...")
    center_c = compute_center_c(encoder, ds_tr_x, max_batches=CENTER_BATCHES)
    print("center_c (first 5):", center_c[:5])

    print("\nTraining DeepSVDD...")
    train_losses, val_losses = train_deepsvdd(
        encoder=encoder,
        ds_train_x=ds_tr_x,
        ds_val_xy_normal=ds_val_xy_norm,
        center_c=center_c,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=3,
    )

    print("\nScoring VAL for threshold tuning...")
    y_val_true, val_scores = collect_scores(ds_val_xy, encoder, center_c)

    try:
        val_roc = float(roc_auc_score(y_val_true, val_scores))
    except Exception:
        val_roc = None
    try:
        val_pr = float(average_precision_score(y_val_true, val_scores))
    except Exception:
        val_pr = None

    lo, hi = np.percentile(val_scores, 1), np.percentile(val_scores, 99)
    thresholds = np.linspace(lo, hi, N_THRESH)

    results = scan_thresholds(y_val_true, val_scores, thresholds)
    best = max(results, key=lambda r: r[8])
    best_t = best[0]

    print("\n=== VAL THRESHOLD SCAN (top 5 by F1) ===")
    for r in sorted(results, key=lambda x: x[8], reverse=True)[:5]:
        t, tp, tn, fp, fn, acc, prec, rec, f1, err = r
        print(f"t={t:.6f} | TP={tp} TN={tn} FP={fp} FN={fn} | Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    print(f"\nBest threshold (VAL by F1): {best_t:.6f}")
    if val_roc is not None:
        print(f"VAL ROC-AUC: {val_roc:.4f}")
    if val_pr is not None:
        print(f"VAL PR-AUC : {val_pr:.4f}")

    print("\nScoring TEST...")
    y_te_true, te_scores = collect_scores(ds_te_xy, encoder, center_c)
    y_te_pred = (te_scores >= best_t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te_true, y_te_pred).ravel()
    acc  = float(accuracy_score(y_te_true, y_te_pred))
    prec = float(precision_score(y_te_true, y_te_pred, zero_division=0))
    rec  = float(recall_score(y_te_true, y_te_pred, zero_division=0))
    f1   = float(f1_score(y_te_true, y_te_pred, zero_division=0))
    err  = float((fp + fn) / (tp + tn + fp + fn))

    try:
        test_roc = float(roc_auc_score(y_te_true, te_scores))
    except Exception:
        test_roc = None
    try:
        test_pr = float(average_precision_score(y_te_true, te_scores))
    except Exception:
        test_pr = None

    print("\n=== TEST RESULTS (DeepSVDD) ===")
    print(f"Threshold: {best_t:.6f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | ErrorRate={err:.4f}")
    if test_roc is not None:
        print(f"Test ROC-AUC: {test_roc:.4f}")
    if test_pr is not None:
        print(f"Test PR-AUC : {test_pr:.4f}")

    # Save
    encoder_path = OUT_DIR / "deepsvdd_encoder.keras"
    encoder.save(encoder_path)

    scaler_path = OUT_DIR / "minmax_params.json"
    eval_path = OUT_DIR / "eval_summary.json"

    minmax_payload = {
        "feature_cols": FEATURE_COLS,
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "eps": EPS,
        "lookback": LOOKBACK,
        "center_c": center_c.tolist(),
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
            "center_batches": CENTER_BATCHES,
            "encoder": {"type": "1D-CNN", "rep_dim": 32},
            "threshold_scan": {"method": "linspace(p1..p99 val scores)", "n_thresh": N_THRESH},
        },
        "best_threshold_val_f1": float(best_t),
        "val_metrics": {"roc_auc": val_roc, "pr_auc": val_pr},
        "test_metrics": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "error_rate": err,
            "roc_auc": test_roc, "pr_auc": test_pr,
        },
        "threshold_scan_val": results,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    eval_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", encoder_path)
    print(" -", scaler_path)
    print(" -", eval_path)


if __name__ == "__main__":
    main()
