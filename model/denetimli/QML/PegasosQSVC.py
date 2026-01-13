# pegasos_qsvc.py
from __future__ import annotations

import json
import time
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# Qiskit ML
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap

try:
    from qiskit_machine_learning.algorithms import PegasosQSVC
except Exception:
    from qiskit_machine_learning.algorithms.classifiers import PegasosQSVC

# =========================
# 0) AYARLAR
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "qml_out" / "pegasos_qsvc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

N_QUBITS = 8
MAX_TRAIN = 600
MAX_TEST  = 2000

THRESHOLD = 0.85

# Pegasos ayarları (imza sürüme göre değişebiliyor; aşağıda otomatik uyarlıyoruz)
C = 1.0
NUM_STEPS = 1000  # artırırsan daha iyi öğrenebilir ama daha uzun sürer

LABEL_CANDIDATES = ["label", "y", "target", "anomaly", "is_anomaly"]

# =========================
# 1) YARDIMCI FONKSİYONLAR
# =========================
def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Label sütunu bulunamadı. Adaylar: {LABEL_CANDIDATES}. "
        f"Mevcut sütunlar (ilk 20): {list(df.columns)[:20]}"
    )

def make_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Basit ama sağlam:
    - 0/1, -1/1, "0"/"1", True/False, string iki sınıf (ilk görülen 0, ikinci 1)
    """
    s = pd.Series(y)

    # bool
    if s.dtype == bool:
        return s.astype(int).to_numpy()

    # numeric'e çevirmeyi dene
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
        # başka iki değer: max->1, diğer->0
        hi = float(np.max(u))
        return np.where(s_num.to_numpy() == hi, 1, 0).astype(int)

    # string/categorical fallback
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
    x = np.asarray(x).ravel()
    mn, mx = float(np.min(x)), float(np.max(x))
    return (x - mn) / (mx - mn + 1e-12)

# =========================
# 2) VERİYİ OKU
# =========================
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

label_col = find_label_column(train_df)
if label_col not in test_df.columns:
    raise ValueError(f"Test dosyasında label sütunu yok: {label_col}")

train_df = downsample(train_df, MAX_TRAIN, RANDOM_STATE)
test_df  = downsample(test_df,  MAX_TEST,  RANDOM_STATE)

y_train = make_binary_labels(train_df[label_col].values)
y_test  = make_binary_labels(test_df[label_col].values)

X_train = train_df.drop(columns=[label_col])
X_test  = test_df.drop(columns=[label_col])

# sayısal olmayanları düş
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train = X_train[num_cols].copy()
X_test  = X_test[num_cols].copy()

if X_train.shape[1] == 0:
    raise ValueError("Feature bulunamadı (sayısal kolon yok).")

# =========================
# 3) PREPROCESS
# =========================
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

Xtr = preprocess.fit_transform(X_train)
Xte = preprocess.transform(X_test)
n_features = Xtr.shape[1]

# =========================
# 4) PEGASOS QSVC
# =========================
feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2, entanglement="full")
qkernel = FidelityQuantumKernel(feature_map=feature_map)

# sürüm uyumluluğu için imzaya göre kwargs hazırla
sig = inspect.signature(PegasosQSVC)
kwargs = {}

if "quantum_kernel" in sig.parameters:
    kwargs["quantum_kernel"] = qkernel
elif "kernel" in sig.parameters:
    kwargs["kernel"] = qkernel

# C / num_steps farklı isimlerle gelebilir
if "C" in sig.parameters:
    kwargs["C"] = C
elif "c" in sig.parameters:
    kwargs["c"] = C

if "num_steps" in sig.parameters:
    kwargs["num_steps"] = NUM_STEPS
elif "steps" in sig.parameters:
    kwargs["steps"] = NUM_STEPS

# seed paramı varsa ver
for seed_name in ["seed", "random_seed", "random_state"]:
    if seed_name in sig.parameters:
        kwargs[seed_name] = RANDOM_STATE
        break

model = PegasosQSVC(**kwargs)

t0 = time.time()
# bazı sürümlerde sample_weight desteklenmez; dene/geri düş
try:
    model.fit(Xtr, y_train, sample_weight=None)
except TypeError:
    model.fit(Xtr, y_train)
train_time_s = time.time() - t0

# =========================
# 5) SCORE + PRED
# =========================
score01 = None

if hasattr(model, "predict_proba"):
    try:
        proba = np.asarray(model.predict_proba(Xte))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            score01 = proba[:, 1]
        elif proba.ndim == 1:
            score01 = proba
    except Exception:
        score01 = None

if score01 is None and hasattr(model, "decision_function"):
    try:
        dec = np.asarray(model.decision_function(Xte)).ravel()
        score01 = safe_minmax01(dec)
    except Exception:
        score01 = None

y_pred_raw = np.asarray(model.predict(Xte)).astype(int).ravel()
if score01 is None:
    score01 = y_pred_raw.astype(float)

y_pred = (np.asarray(score01).ravel() >= THRESHOLD).astype(int)

# =========================
# 6) METRİKLER
# =========================
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

try:
    roc = roc_auc_score(y_test, score01)
except Exception:
    roc = None

try:
    ap = average_precision_score(y_test, score01)
except Exception:
    ap = None

# =========================
# 7) KAYDET
# =========================
cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
cm_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=True)

pred_df = pd.DataFrame(
    {
        "y_true": y_test,
        "y_pred_threshold": y_pred,
        "y_pred_raw": y_pred_raw,
        "score01": np.asarray(score01).ravel(),
    }
)
pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

metrics = {
    "model": "PegasosQSVC",
    "random_state": RANDOM_STATE,
    "n_train": int(len(Xtr)),
    "n_test": int(len(Xte)),
    "n_features_after_pca": int(n_features),
    "threshold": float(THRESHOLD),
    "params_requested": {"C": float(C), "num_steps": int(NUM_STEPS)},
    "init_kwargs_used": {k: (str(v) if k in ["quantum_kernel", "kernel"] else v) for k, v in kwargs.items()},
    "train_time_s": float(train_time_s),
    "metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": None if roc is None else float(roc),
        "avg_precision": None if ap is None else float(ap),
    },
    "confusion_matrix": cm.tolist(),
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# model + preprocess kaydı
try:
    model.to_dill(str(OUT_DIR / "pegasos_qsvc_model.dill"))
except Exception as e:
    with open(OUT_DIR / "save_warning.txt", "w", encoding="utf-8") as f:
        f.write(f"model.to_dill failed: {repr(e)}\n")

try:
    import joblib
    joblib.dump(preprocess, OUT_DIR / "preprocess.joblib")
except Exception as e:
    with open(OUT_DIR / "save_warning.txt", "a", encoding="utf-8") as f:
        f.write(f"joblib.dump(preprocess) failed: {repr(e)}\n")

print("✅ PegasosQSVC bitti.")
print("Output:", OUT_DIR)
print("Accuracy:", acc, "| F1:", f1, "| ROC-AUC:", roc, "| AP:", ap)
print("Confusion matrix:\n", cm_df)
