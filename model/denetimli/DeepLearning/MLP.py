import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score
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

VAL_RATIO = 0.2
BATCH_SIZE = 256
EPOCHS = 200
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 1) Yardımcılar
# =========================
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    for c in FEATURE_COLS + [LABEL_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # zaman sırasına diz
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df

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
# 3) Train/Val split (time-based)
# =========================
n = len(train_df)
split = int(n * (1 - VAL_RATIO))

train_part = train_df.iloc[:split].copy()
val_part   = train_df.iloc[split:].copy()

X_train_raw = train_part[FEATURE_COLS].values
y_train     = train_part[LABEL_COL].values

X_val_raw = val_part[FEATURE_COLS].values
y_val     = val_part[LABEL_COL].values

X_test_raw = test_df[FEATURE_COLS].values
y_test     = test_df[LABEL_COL].values

# =========================
# 4) Ölçekleme (Scaler sadece train_part ile fit)
# =========================
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# =========================
# 5) MLP Modeli
# =========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(len(FEATURE_COLS),)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

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
# 6) Loss grafiği
# =========================
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Train vs Val Loss")
plt.legend()
plt.show()

# =========================
# 7) Validation threshold taraması
# =========================
val_prob = model.predict(X_val).ravel()

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
# 8) Test performansı (best threshold ile)
# =========================
test_prob = model.predict(X_test).ravel()
test_pred = (test_prob >= best_t).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
acc  = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec  = recall_score(y_test, test_pred, zero_division=0)
f1   = f1_score(y_test, test_pred, zero_division=0)

print("\n=== TEST RESULTS (MLP) ===")
print(f"Threshold: {best_t:0.2f}")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Accuracy={acc:0.4f} | Precision={prec:0.4f} | Recall={rec:0.4f} | F1={f1:0.4f}")

try:
    test_auc = roc_auc_score(y_test, test_prob)
    print(f"Test ROC-AUC: {test_auc:.4f}")
except Exception:
    pass