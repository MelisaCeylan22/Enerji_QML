# qsvc_qsvm.py (MemoryError fix: batch scoring + max_circuits_per_job + ML-format + logs)
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    fbeta_score,
)

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap

# =========================
# 0) AYARLAR
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "qml_out" / "qsvc_qsvm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_QUBITS = 8

MAX_TRAIN = 300   # ✅ öneri: 600 yerine 200-400 arası (kernel O(n^2))
MAX_TEST  = 600   # ✅ öneri: 2000 yerine 300-800 arası

VAL_SIZE = 0.2
DEFAULT_THRESHOLD = 0.85

# Threshold seçimi
THRESHOLD_SELECTION = "f0_5"  # "f0_5" | "f1" | "fpr_cap"
BETA = 0.5
FPR_TARGET = 0.01

# QSVC params
C = 1.0
MAX_ITER = 2000

# ✅ Memory fix ayarları
BATCH_SIZE_SCORE = 20              # decision_function batch boyutu (10-30 arası iyi)
KERNEL_MAX_CIRCUITS_PER_JOB = 256  # job bölme (128/256/512 denenebilir)

LABEL_CANDIDATES = ["label", "y", "target", "anomaly", "is_anomaly"]

# =========================
# 1) YARDIMCI FONKSİYONLAR
# =========================
def log(msg: str):
    print(msg, flush=True)

def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Label sütunu bulunamadı. Adaylar: {LABEL_CANDIDATES}. "
        f"Mevcut sütunlar (ilk 20): {list(df.columns)[:20]}"
    )

def make_binary_labels(y: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(y).ravel())

    if s.dtype == bool:
        return s.astype(int).to_numpy()

    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().all():
        uniq = pd.unique(s_num)
        if len(uniq) != 2:
            raise ValueError(f"Binary bekleniyordu, sınıf sayısı={len(uniq)} -> {uniq.tolist()}")
        u = np.unique(s_num.to_numpy())
        if set(u.tolist()) == {0, 1}:
            return s_num.astype(int).to_numpy()
        if set(u.tolist()) == {-1, 1}:
            return (s_num == 1).astype(int).to_numpy()
        hi = float(np.max(u))
        return np.where(s_num.to_numpy() == hi, 1, 0).astype(int)

    s_str = s.astype(str).str.strip().str.lower()
    uniq = pd.unique(s_str)
    if len(uniq) != 2:
        raise ValueError(f"Binary bekleniyordu, sınıf sayısı={len(uniq)} -> {uniq.tolist()}")
    mapping = {uniq[0]: 0, uniq[1]: 1}
    return s_str.map(mapping).astype(int).to_numpy()

def downsample(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    if max_n is None or max_n <= 0 or len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed).sort_index()

def safe_minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel().astype(float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if np.isclose(mx, mn):
        return np.full_like(x, 0.5, dtype=float)
    return (x - mn) / (mx - mn + 1e-12)

def batch_decision_function(model: QSVC, X: np.ndarray, batch_size: int) -> np.ndarray:
    """✅ Memory fix: decision_function'ı batch batch çalıştır."""
    X = np.asarray(X)
    outs = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        log(f"     decision_function batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size} | size={len(xb)}")
        dec_b = np.asarray(model.decision_function(xb)).ravel()
        outs.append(dec_b)
    return np.concatenate(outs, axis=0)

def get_score01_from_qsvc(model: QSVC, X: np.ndarray, batch_size: int) -> np.ndarray:
    dec = batch_decision_function(model, X, batch_size=batch_size)
    return safe_minmax01(dec)

def eval_at_threshold(y_true: np.ndarray, score01: np.ndarray, thr: float):
    y_true = np.asarray(y_true).astype(int).ravel()
    score01 = np.asarray(score01).astype(float).ravel()
    y_pred = (score01 >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    fbeta = fbeta_score(y_true, y_pred, beta=BETA, zero_division=0)
    fpr = fp / (fp + tn + 1e-12)

    return {
        "thr": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1v),
        f"f{BETA}": float(fbeta),
        "fpr": float(fpr),
    }

def find_best_threshold(y_true: np.ndarray, score01: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    score01 = np.asarray(score01).astype(float).ravel()

    if np.allclose(score01, score01[0]):
        return eval_at_threshold(y_true, score01, DEFAULT_THRESHOLD)

    candidates = np.unique(np.clip(score01, 0.0, 1.0))
    grid = np.linspace(0.0, 1.0, 201)
    thrs = np.unique(np.concatenate([candidates, grid]))

    best = None
    for thr in thrs:
        cur = eval_at_threshold(y_true, score01, float(thr))

        if THRESHOLD_SELECTION == "fpr_cap":
            if cur["fpr"] > FPR_TARGET:
                continue
            if best is None or (cur["recall"] > best["recall"]) or (
                np.isclose(cur["recall"], best["recall"]) and cur["precision"] > best["precision"]
            ):
                best = cur

        elif THRESHOLD_SELECTION == "f1":
            if best is None or (cur["f1"] > best["f1"]) or (
                np.isclose(cur["f1"], best["f1"]) and cur["precision"] > best["precision"]
            ):
                best = cur

        else:
            key = f"f{BETA}"
            if best is None or (cur[key] > best[key]) or (
                np.isclose(cur[key], best[key]) and cur["precision"] > best["precision"]
            ) or (
                np.isclose(cur[key], best[key]) and np.isclose(cur["precision"], best["precision"]) and cur["recall"] > best["recall"]
            ):
                best = cur

    if best is None and THRESHOLD_SELECTION == "fpr_cap":
        log("[WARN] FPR_TARGET çok sıkı, val’da hiç threshold uymadı. f0.5 moduna düşüyorum.")
        old = THRESHOLD_SELECTION
        try:
            globals()["THRESHOLD_SELECTION"] = "f0_5"
            best = find_best_threshold(y_true, score01)
        finally:
            globals()["THRESHOLD_SELECTION"] = old

    return best

def print_ml_style(name: str, tn, fp, fn, tp, thr, acc, prec, rec, f1v, err):
    print(f"{name}\tSonuçlar")
    print(f"True Positive (TP): \t{tp:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    print(f"True Negative (TN): \t{tn:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    print(f"False Positive (FP): \t{fp:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    print(f"False Negative (FN):\t{fn:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    print(f"Doğruluk (Accuracy): \t{acc:.4f}")
    print(f"Kesinlik (Precision):\t{prec:.4f}")
    print(f"Duyarlılık (Recall):\t{rec:.4f}")
    print(f"F Ölçümü (F1 Score):\t{f1v:.4f}")
    print(f"Hata Oranı (Error Rate):\t{err:.4f}")
    print(f"En İyi Eşik Değeri:\t{thr:.2f}")

# =========================
# 2) VERİYİ OKU
# =========================
log("[1/8] CSV'ler okunuyor...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

label_col = find_label_column(train_df)
if label_col not in test_df.columns:
    raise ValueError(f"Test dosyasında label sütunu yok: {label_col}")
log(f"     Label sütunu: {label_col}")

log("[2/8] Downsample uygulanıyor...")
train_df = downsample(train_df, MAX_TRAIN, RANDOM_STATE)
test_df  = downsample(test_df,  MAX_TEST,  RANDOM_STATE)
log(f"     Train size: {len(train_df)} | Test size: {len(test_df)}")

y_train = make_binary_labels(train_df[label_col].values)
y_test  = make_binary_labels(test_df[label_col].values)
log(f"     Train class dist: {np.bincount(y_train, minlength=2).tolist()}")
log(f"     Test  class dist: {np.bincount(y_test, minlength=2).tolist()}")

X_train = train_df.drop(columns=[label_col])
X_test  = test_df.drop(columns=[label_col])

log("[3/8] Sadece sayısal feature'lar seçiliyor...")
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train = X_train[num_cols].copy()
X_test  = X_test[num_cols].copy()
log(f"     Num features (raw): {X_train.shape[1]}")
if X_train.shape[1] == 0:
    raise ValueError("Feature bulunamadı (sayısal kolon yok).")

# =========================
# 3) PREPROCESS
# =========================
log("[4/8] Preprocess (impute+scale+PCA+[0,2pi])...")
algorithm_globals.random_seed = RANDOM_STATE
np.random.seed(RANDOM_STATE)

preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(N_QUBITS, X_train.shape[1]), random_state=RANDOM_STATE)),
        ("minmax", MinMaxScaler(feature_range=(0.0, 2.0 * np.pi))),
    ]
)

Xtr_all = preprocess.fit_transform(X_train)
Xte = preprocess.transform(X_test)
log(f"     Features after PCA (=qubits): {Xtr_all.shape[1]}")

log("[5/8] Train/Val split...")
Xtr, Xval, ytr, yval = train_test_split(
    Xtr_all, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
)
log(f"     Train used: {len(Xtr)} | Val: {len(Xval)}")

n_features = Xtr.shape[1]

# =========================
# 4) QSVC (QSVM)
# =========================
log("[6/8] QSVC kuruluyor (quantum kernel) ve eğitiliyor...")
feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2, entanglement="full")

qkernel = FidelityQuantumKernel(feature_map=feature_map)

# ✅ job bölme (sürüm farkı için hem param hem attribute deniyoruz)
try:
    # bazı sürümlerde attribute var
    qkernel.max_circuits_per_job = KERNEL_MAX_CIRCUITS_PER_JOB
except Exception:
    pass

qsvc = QSVC(
    quantum_kernel=qkernel,
    C=C,
    class_weight="balanced",
    max_iter=MAX_ITER,
)

t0 = time.time()
qsvc.fit(Xtr, ytr)
train_time_s = time.time() - t0
log(f"     Eğitim süresi: {train_time_s:.2f} s")

# =========================
# 5) VAL'da threshold seç
# =========================
log("[7/8] Validation üzerinde en iyi threshold seçiliyor...")
val_score01 = get_score01_from_qsvc(qsvc, Xval, batch_size=BATCH_SIZE_SCORE)
best_val = find_best_threshold(yval, val_score01)
best_thr = best_val["thr"]
log(f"     Seçilen threshold: {best_thr:.4f} | mode={THRESHOLD_SELECTION} | val={best_val}")

# =========================
# 6) TEST EVAL (✅ batch)
# =========================
log("[8/8] Test değerlendiriliyor (batch) + dosyalar kaydediliyor...")
test_score01 = get_score01_from_qsvc(qsvc, Xte, batch_size=BATCH_SIZE_SCORE)
y_pred = (test_score01 >= best_thr).astype(int)

# referans: QSVC'nin kendi predict'i (o da kernel ister -> batchlemek için kullanmıyoruz)
# y_pred_raw = qsvc.predict(Xte)  # ❌ büyükte yine RAM yakabilir
y_pred_raw = np.full_like(y_pred, -1, dtype=int)  # placeholder

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1v = f1_score(y_test, y_pred, zero_division=0)
err = (fp + fn) / (tp + tn + fp + fn + 1e-12)

try:
    roc = roc_auc_score(y_test, test_score01)
except Exception:
    roc = None
try:
    ap = average_precision_score(y_test, test_score01)
except Exception:
    ap = None

print_ml_style("QSVC (QSVM)", tn, fp, fn, tp, best_thr, acc, prec, rec, f1v, err)
log(f"ROC-AUC: {roc} | AP: {ap}")
log(f"Output: {OUT_DIR}")

# =========================
# 7) Kaydet
# =========================
cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
cm_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=True)

pred_df = pd.DataFrame(
    {
        "y_true": y_test,
        "score01": np.asarray(test_score01).ravel(),
        "y_pred_threshold": y_pred,
        "y_pred_raw": y_pred_raw,
    }
)
pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

metrics = {
    "model": "QSVC (QSVM)",
    "random_state": RANDOM_STATE,
    "n_train_total": int(len(Xtr_all)),
    "n_train_used": int(len(Xtr)),
    "n_val": int(len(Xval)),
    "n_test": int(len(Xte)),
    "n_features_after_pca": int(n_features),
    "threshold_selection": THRESHOLD_SELECTION,
    "beta": float(BETA),
    "fpr_target": float(FPR_TARGET),
    "best_threshold_on_val": float(best_thr),
    "val_at_best_threshold": best_val,
    "train_time_s": float(train_time_s),
    "svc_params": {"C": float(C), "max_iter": int(MAX_ITER), "class_weight": "balanced"},
    "memory_fixes": {"batch_size_score": int(BATCH_SIZE_SCORE), "kernel_max_circuits_per_job": int(KERNEL_MAX_CIRCUITS_PER_JOB)},
    "metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1v),
        "error_rate": float(err),
        "roc_auc": None if roc is None else float(roc),
        "avg_precision": None if ap is None else float(ap),
    },
    "confusion_matrix": cm.tolist(),
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# Model + preprocess kaydı
try:
    qsvc.to_dill(str(OUT_DIR / "qsvc_model.dill"))
except Exception as e:
    with open(OUT_DIR / "save_warning.txt", "w", encoding="utf-8") as f:
        f.write(f"qsvc.to_dill failed: {repr(e)}\n")

try:
    import joblib
    joblib.dump(preprocess, OUT_DIR / "preprocess.joblib")
except Exception as e:
    with open(OUT_DIR / "save_warning.txt", "a", encoding="utf-8") as f:
        f.write(f"joblib.dump(preprocess) failed: {repr(e)}\n")

log("✅ QSVC (QSVM) bitti.")
