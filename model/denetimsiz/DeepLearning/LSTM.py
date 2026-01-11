import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)

# =========================
# 0) Ayarlar
# =========================
TRAIN_PATH = r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed\train_labeled_balanced.csv"
TEST_PATH  = r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed\test_labeled_full.csv"

FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]
LABEL_COL = "label"
TIME_COL = "timestamp"

LOOKBACK = 60      # örn: son 60 dakika -> 60 satır
VAL_RATIO = 0.2    # train içinden validation
BATCH_SIZE = 128
EPOCHS = 200
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 1) Yardımcı fonksiyonlar
# =========================
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # timestamp parse
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    # numeric cast (virgül nokta karmaşası vs için güvenli)
    for c in FEATURE_COLS + [LABEL_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # eksikler
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    # label int
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # time sort
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df

def make_sequences(features_2d: np.ndarray, labels_1d: np.ndarray, lookback: int):
    """
    features_2d: (N, n_features)
    labels_1d:   (N,)
    returns:
      X: (N-lookback+1, lookback, n_features)
      y: (N-lookback+1,)
    Label olarak pencerenin SON satırının label'ını alıyoruz.
    """
    X, y = [], []
    for i in range(lookback - 1, len(features_2d)):
        X.append(features_2d[i - lookback + 1:i + 1])
        y.append(labels_1d[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def scan_thresholds(y_true, y_prob, thresholds=np.arange(0.1, 0.9, 0.1)):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        err = (fp + fn) / (tp + tn + fp + fn)
        rows.append((t, tp, tn, fp, fn, acc, prec, rec, f1, err))
    return rows

# =========================
# 2) Veriyi oku
# =========================
train_df = load_and_clean(TRAIN_PATH)
test_df  = load_and_clean(TEST_PATH)

# =========================
# 3) Train/Val split (time-based, shuffle yok)
# =========================
n = len(train_df)
split = int(n * (1 - VAL_RATIO))

train_part = train_df.iloc[:split].copy()
val_part   = train_df.iloc[split:].copy()

X_train_raw = train_part[FEATURE_COLS].values
y_train_raw = train_part[LABEL_COL].values

X_val_raw = val_part[FEATURE_COLS].values
y_val_raw = val_part[LABEL_COL].values

X_test_raw = test_df[FEATURE_COLS].values
y_test_raw = test_df[LABEL_COL].values

# =========================
# 4) Ölçekleme (Scaler sadece train_part ile fit!)
# =========================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled   = scaler.transform(X_val_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# =========================
# 5) Sequence üretimi
# =========================
X_train, y_train = make_sequences(X_train_scaled, y_train_raw, LOOKBACK)
X_val,   y_val   = make_sequences(X_val_scaled,   y_val_raw,   LOOKBACK)
X_test,  y_test  = make_sequences(X_test_scaled,  y_test_raw,  LOOKBACK)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# =========================
# 6) Model
# =========================
model = Sequential([
    LSTM(64, activation="tanh", input_shape=(LOOKBACK, len(FEATURE_COLS))),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =========================
# 7) Eğitim grafiği
# =========================
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Val Loss")
plt.legend()
plt.show()

# =========================
# 8) Validation threshold taraması
# =========================
val_prob = model.predict(X_val).ravel()

# ROC-AUC (opsiyonel ama güzel)
try:
    val_auc = roc_auc_score(y_val, val_prob)
    print(f"Val ROC-AUC: {val_auc:.4f}")
except Exception:
    pass

results = scan_thresholds(y_val, val_prob, thresholds=np.arange(0.1, 0.9, 0.1))

print("\nThreshold | TP | TN | FP | FN | Acc   | Prec  | Rec   | F1    | Err")
for r in results:
    t, tp, tn, fp, fn, acc, prec, rec, f1, err = r
    print(f"{t:0.2f}      | {tp:3d}| {tn:3d}| {fp:3d}| {fn:3d}| {acc:0.4f}| {prec:0.4f}| {rec:0.4f}| {f1:0.4f}| {err:0.4f}")

best = max(results, key=lambda x: x[8])  # F1'e göre
best_t = best[0]
print(f"\n✅ En iyi threshold (Val F1): {best_t:0.2f} | F1={best[8]:0.4f}")

# =========================
# 9) Test performansı (best threshold ile)
# =========================
test_prob = model.predict(X_test).ravel()
test_pred = (test_prob >= best_t).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
acc  = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec  = recall_score(y_test, test_pred, zero_division=0)
f1   = f1_score(y_test, test_pred, zero_division=0)

print("\n=== TEST RESULTS ===")
print(f"Threshold: {best_t:0.2f}")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Accuracy={acc:0.4f} | Precision={prec:0.4f} | Recall={rec:0.4f} | F1={f1:0.4f}")

try:
    test_auc = roc_auc_score(y_test, test_prob)
    print(f"Test ROC-AUC: {test_auc:.4f}")
except Exception:
    pass
