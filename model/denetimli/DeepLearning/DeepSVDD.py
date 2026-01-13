from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

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

# AE pretrain
DO_AE_PRETRAIN = True
AE_EPOCHS = 5
AE_LR = 1e-3

# DeepSVDD train
EPOCHS = 15
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
SEED = 42
SHUFFLE_TRAIN_WINDOWS = True

# Threshold tuning (0..1)
N_THRESH = 101
FBETA = 0.5  # precision odaklı (FP pahalıysa 0.3 da deneyebilirsin)

# Score normalize (VAL score quantile range)
# Robust min-max için (uç outlier’ları bastırmak)
SCORE_Q_LOW  = 0.001
SCORE_Q_HIGH = 0.999

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
CENTER_EPS = 1e-6
CENTER_BATCHES = 400

WEIGHT_DECAY = 1e-6

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


def time_split_train_val(df_train: pd.DataFrame, val_ratio: float):
    n = len(df_train)
    cut = int(n * (1 - val_ratio))
    return df_train.iloc[:cut].copy(), df_train.iloc[cut:].copy()


def fit_minmax(X: np.ndarray) -> dict:
    return {"mins": X.min(axis=0), "maxs": X.max(axis=0)}


def transform_minmax(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = (maxs - mins)
    denom = np.where(denom < EPS, 1.0, denom)
    Xs = (X - mins) / denom
    # stabilite için clip (test’te scale dışına taşmasın)
    return np.clip(Xs, 0.0, 1.0)


def _window_max_targets(y: np.ndarray, lookback: int) -> np.ndarray:
    s = pd.Series(y.astype(np.int32))
    t = s.rolling(lookback).max().to_numpy()[lookback - 1 :]
    return t.astype(np.int32)


def make_ts_dataset(
    X_scaled: np.ndarray,
    y: np.ndarray,
    lookback: int,
    batch_size: int,
    shuffle: bool,
    target_mode: str = "last",   # "last" veya "window_max"
) -> tf.data.Dataset:
    if len(X_scaled) < lookback:
        raise ValueError(f"Segment çok kısa: len={len(X_scaled)} < lookback={lookback}")

    if target_mode == "last":
        targets = y[lookback - 1 :].astype(np.int32)
    elif target_mode == "window_max":
        targets = _window_max_targets(y, lookback)
    else:
        raise ValueError("target_mode must be 'last' or 'window_max'")

    end_index = len(X_scaled) - lookback

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X_scaled.astype(np.float32),
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


def normal_only_x(ds_xy_batched: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    ds = ds_xy_batched.unbatch()
    ds = ds.filter(lambda x, y: tf.equal(y, 0))
    ds = ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


def fbeta(prec: float, rec: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * prec + rec
    return (1 + b2) * (prec * rec) / denom if denom > 0 else 0.0


def safe_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(tn), int(fp), int(fn)


def scan_thresholds_01(y_true: np.ndarray, score01: np.ndarray, beta: float, n_thresh: int):
    thresholds = np.linspace(0.0, 1.0, n_thresh)

    rows = []
    best = None
    for t in thresholds:
        y_pred = (score01 >= t).astype(int)

        tp, tn, fp, fn = safe_confusion(y_true, y_pred)
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        fb = float(fbeta(prec, rec, beta))
        err = float((fp + fn) / (tp + tn + fp + fn))

        row = (float(t), tp, tn, fp, fn, acc, prec, rec, f1, fb, err)
        rows.append(row)
        if best is None or fb > best[9]:
            best = row

    return rows, best


def robust_fit_score_norm(val_scores: np.ndarray, q_low: float, q_high: float):
    lo = float(np.quantile(val_scores, q_low))
    hi = float(np.quantile(val_scores, q_high))
    if hi - lo < 1e-12:
        # skorlar neredeyse sabit -> degenerate
        hi = lo + 1e-6
    return lo, hi


def robust_score_to_01(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = (scores - lo) / (hi - lo)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def maybe_fix_direction_by_auc(y_true: np.ndarray, score01: np.ndarray):
    """
    Eğer ROC-AUC < 0.5 ise skor yönü ters demektir -> 1-score yap.
    Bu sayede "yüksek skor = daha anomali" konvansiyonu korunur.
    """
    try:
        auc = float(roc_auc_score(y_true, score01))
    except Exception:
        # AUC hesaplanamazsa (tek sınıf vb.), yön düzeltme yok
        return score01, False, None

    if auc < 0.5:
        return (1.0 - score01).astype(np.float32), True, auc
    return score01, False, auc


# =========================
# 2) Model: Encoder + AE Decoder
# =========================
def build_encoder(lookback: int, n_features: int, rep_dim: int = 32) -> tf.keras.Model:
    inp = layers.Input(shape=(lookback, n_features))

    x = layers.Conv1D(
        64, 5, padding="same", activation="relu",
        use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(inp)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(
        64, 3, padding="same", activation="relu",
        use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # 60 -> 30

    x = layers.Conv1D(
        128, 3, padding="same", activation="relu",
        use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(
        64, activation="relu", use_bias=False,
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.Dropout(0.2)(x)

    z = layers.Dense(
        rep_dim, activation=None, use_bias=False,
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="rep"
    )(x)

    return Model(inp, z, name="DeepSVDD_Encoder")


def build_autoencoder(encoder: tf.keras.Model, lookback: int, n_features: int) -> tf.keras.Model:
    inp = encoder.input
    z = encoder.output

    x = layers.Dense(30 * 128, activation="relu")(z)
    x = layers.Reshape((30, 128))(x)
    x = layers.UpSampling1D(size=2)(x)  # 30 -> 60
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(n_features, 3, padding="same", activation=None)(x)

    return Model(inp, x, name="AE")


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

    c = mean.astype(np.float32)
    c = np.where((np.abs(c) < CENTER_EPS) & (c >= 0), CENTER_EPS, c)
    c = np.where((np.abs(c) < CENTER_EPS) & (c < 0), -CENTER_EPS, c)
    return c


def train_deepsvdd(
    encoder: tf.keras.Model,
    ds_train_x: tf.data.Dataset,
    ds_val_x: tf.data.Dataset,
    center_c: np.ndarray,
    epochs: int,
    lr: float,
    patience: int = 3,
):
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    c = tf.convert_to_tensor(center_c, dtype=tf.float32)

    best_val = np.inf
    best_weights = None
    wait = 0

    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        tl = []
        for bx in ds_train_x:
            with tf.GradientTape() as tape:
                z = encoder(bx, training=True)
                dist = tf.reduce_sum(tf.square(z - c), axis=1)
                loss = tf.reduce_mean(dist)
                if encoder.losses:
                    loss = loss + tf.add_n(encoder.losses)

            grads = tape.gradient(loss, encoder.trainable_variables)
            opt.apply_gradients(zip(grads, encoder.trainable_variables))
            tl.append(float(loss.numpy()))
        tr_loss = float(np.mean(tl)) if tl else float("nan")
        train_losses.append(tr_loss)

        vl = []
        for bx in ds_val_x:
            z = encoder(bx, training=False)
            dist = tf.reduce_sum(tf.square(z - c), axis=1)
            vl.append(float(tf.reduce_mean(dist).numpy()))
        va_loss = float(np.mean(vl)) if vl else float("nan")
        val_losses.append(va_loss)

        print(f"Epoch {ep:02d}/{epochs} | train_loss={tr_loss:.6e} | val_loss={va_loss:.6e}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_weights = encoder.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"EarlyStopping: best_val={best_val:.6e}")
                break

    if best_weights is not None:
        encoder.set_weights(best_weights)

    return train_losses, val_losses


def collect_raw_scores(ds_xy: tf.data.Dataset, encoder: tf.keras.Model, center_c: np.ndarray):
    y_true_list, scores_list = [], []
    c = tf.convert_to_tensor(center_c, dtype=tf.float32)

    for bx, by in ds_xy:
        z = encoder(bx, training=False)
        s = tf.reduce_sum(tf.square(z - c), axis=1)  # distance (0..inf)
        scores_list.append(s.numpy().astype(np.float32))
        y_true_list.append(by.numpy().astype(np.int32))

    return np.concatenate(y_true_list), np.concatenate(scores_list)


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

    print("Creating datasets...")
    # (1) normal seçimi için temiz hedef: window_max
    ds_tr_xy_max  = make_ts_dataset(X_tr,  y_tr_raw,  LOOKBACK, BATCH_SIZE, shuffle=SHUFFLE_TRAIN_WINDOWS, target_mode="window_max")
    ds_val_xy_max = make_ts_dataset(X_val, y_val_raw, LOOKBACK, BATCH_SIZE, shuffle=False, target_mode="window_max")

    ds_tr_x = normal_only_x(ds_tr_xy_max, batch_size=BATCH_SIZE)
    ds_val_x = normal_only_x(ds_val_xy_max, batch_size=BATCH_SIZE)

    # (2) metrik için: last-label
    ds_val_xy_last = make_ts_dataset(X_val, y_val_raw, LOOKBACK, BATCH_SIZE, shuffle=False, target_mode="last")
    ds_te_xy_last  = make_ts_dataset(X_te,  y_te_raw,  LOOKBACK, BATCH_SIZE, shuffle=False, target_mode="last")

    encoder = build_encoder(LOOKBACK, len(FEATURE_COLS), rep_dim=32)

    # ---------- AE PRETRAIN ----------
    if DO_AE_PRETRAIN:
        print("\n[AE PRETRAIN] training autoencoder on NORMAL windows...")
        ae = build_autoencoder(encoder, LOOKBACK, len(FEATURE_COLS))
        ae.compile(optimizer=tf.keras.optimizers.Adam(AE_LR), loss="mse")

        ds_tr_ae = ds_tr_x.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        ds_val_ae = ds_val_x.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        cb = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
        ]
        ae.fit(ds_tr_ae, validation_data=ds_val_ae, epochs=AE_EPOCHS, callbacks=cb, verbose=1)

    # ---------- Center c ----------
    print("\nComputing center c from NORMAL train windows...")
    center_c = compute_center_c(encoder, ds_tr_x, max_batches=CENTER_BATCHES)
    print("center_c (first 5):", center_c[:5])

    # ---------- DeepSVDD train ----------
    print("\nTraining DeepSVDD...")
    train_losses, val_losses = train_deepsvdd(
        encoder=encoder,
        ds_train_x=ds_tr_x,
        ds_val_x=ds_val_x,
        center_c=center_c,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=3,
    )

    # ---------- VAL: raw scores ----------
    print("\nScoring VAL (raw distance) ...")
    y_val_true, val_scores_raw = collect_raw_scores(ds_val_xy_last, encoder, center_c)

    # ---------- Normalize score to 0..1 using VAL distribution ----------
    score_lo, score_hi = robust_fit_score_norm(val_scores_raw, SCORE_Q_LOW, SCORE_Q_HIGH)
    val_score01 = robust_score_to_01(val_scores_raw, score_lo, score_hi)

    # ---------- Auto-fix direction (ROC-AUC < 0.5) ----------
    val_score01_fixed, inverted, auc_before = maybe_fix_direction_by_auc(y_val_true, val_score01)

    # Metrics after fix
    try:
        val_roc = float(roc_auc_score(y_val_true, val_score01_fixed))
    except Exception:
        val_roc = None
    try:
        val_pr = float(average_precision_score(y_val_true, val_score01_fixed))
    except Exception:
        val_pr = None

    print(f"VAL score norm params: lo(q{SCORE_Q_LOW})={score_lo:.6e}, hi(q{SCORE_Q_HIGH})={score_hi:.6e}")
    if auc_before is not None:
        print(f"VAL ROC-AUC before direction-fix: {auc_before:.4f} | inverted={inverted}")

    # ---------- Threshold scan in [0,1] ----------
    results, best = scan_thresholds_01(y_val_true, val_score01_fixed, beta=FBETA, n_thresh=N_THRESH)
    best_t = best[0]

    print("\n=== VAL THRESHOLD SCAN (top 5 by Fbeta) ===")
    for r in sorted(results, key=lambda x: x[9], reverse=True)[:5]:
        t, tp, tn, fp, fn, acc, prec, rec, f1, fb, err = r
        print(f"t={t:.3f} | TP={tp} TN={tn} FP={fp} FN={fn} | Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} F{FBETA}={fb:.4f}")

    print(f"\nBest threshold (VAL by F{FBETA}): {best_t:.3f}")
    if val_roc is not None:
        print(f"VAL ROC-AUC: {val_roc:.4f}")
    if val_pr is not None:
        print(f"VAL PR-AUC : {val_pr:.4f}")

    # ---------- TEST ----------
    print("\nScoring TEST...")
    y_te_true, te_scores_raw = collect_raw_scores(ds_te_xy_last, encoder, center_c)

    te_score01 = robust_score_to_01(te_scores_raw, score_lo, score_hi)
    if inverted:
        te_score01 = (1.0 - te_score01).astype(np.float32)

    y_te_pred = (te_score01 >= best_t).astype(int)

    tp, tn, fp, fn = safe_confusion(y_te_true, y_te_pred)
    acc  = float(accuracy_score(y_te_true, y_te_pred))
    prec = float(precision_score(y_te_true, y_te_pred, zero_division=0))
    rec  = float(recall_score(y_te_true, y_te_pred, zero_division=0))
    f1   = float(f1_score(y_te_true, y_te_pred, zero_division=0))
    fb   = float(fbeta(prec, rec, FBETA))
    err  = float((fp + fn) / (tp + tn + fp + fn))

    try:
        test_roc = float(roc_auc_score(y_te_true, te_score01))
    except Exception:
        test_roc = None
    try:
        test_pr = float(average_precision_score(y_te_true, te_score01))
    except Exception:
        test_pr = None

    print("\n=== TEST RESULTS (DeepSVDD | score in [0,1]) ===")
    print(f"Threshold: {best_t:.3f}  | selected by F{FBETA} on VAL")
    print(f"(score normalized by VAL quantiles; inverted={inverted})")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | F{FBETA}={fb:.4f} | ErrorRate={err:.4f}")
    if test_roc is not None:
        print(f"Test ROC-AUC: {test_roc:.4f}")
    if test_pr is not None:
        print(f"Test PR-AUC : {test_pr:.4f}")

    # ---------- Save ----------
    encoder_path = OUT_DIR / "deepsvdd_encoder.keras"
    encoder.save(encoder_path)

    params_path = OUT_DIR / "deepsvdd_params.json"
    eval_path = OUT_DIR / "eval_summary.json"

    params_payload = {
        "feature_cols": FEATURE_COLS,
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "eps": EPS,
        "lookback": LOOKBACK,
        "center_c": center_c.tolist(),
        "ae_pretrain": DO_AE_PRETRAIN,
        "score_norm": {
            "method": "robust_minmax_on_VAL_scores",
            "q_low": SCORE_Q_LOW,
            "q_high": SCORE_Q_HIGH,
            "lo": score_lo,
            "hi": score_hi,
            "inverted": bool(inverted),
        },
        "threshold_01": float(best_t),
        "threshold_selected_by": f"F{FBETA} on VAL (score01)",
    }
    params_path.write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    eval_payload = {
        "config": {
            "train_csv": str(TRAIN_CSV),
            "test_csv": str(TEST_CSV),
            "val_ratio_within_train": VAL_RATIO_WITHIN_TRAIN,
            "lookback": LOOKBACK,
            "ae_pretrain": DO_AE_PRETRAIN,
            "ae_epochs": AE_EPOCHS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "shuffle_train_windows": SHUFFLE_TRAIN_WINDOWS,
            "center_batches": CENTER_BATCHES,
            "score_norm": {
                "q_low": SCORE_Q_LOW,
                "q_high": SCORE_Q_HIGH,
                "inverted": bool(inverted),
            },
            "threshold_scan": {"range": "[0,1]", "n_thresh": N_THRESH, "beta": FBETA},
        },
        "best_threshold_val_fbeta_01": float(best_t),
        "val_metrics": {"roc_auc": val_roc, "pr_auc": val_pr},
        "test_metrics": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, f"f{FBETA}": fb,
            "error_rate": err,
            "roc_auc": test_roc, "pr_auc": test_pr,
        },
        "threshold_scan_val": results,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    eval_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", encoder_path)
    print(" -", params_path)
    print(" -", eval_path)


if __name__ == "__main__":
    main()
