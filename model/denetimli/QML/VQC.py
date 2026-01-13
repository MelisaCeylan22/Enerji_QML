# vqc_classifier.py (ML-format çıktı + precision-odaklı threshold seçimi + adım adım terminal log)
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
from sklearn.utils.class_weight import compute_sample_weight

from qiskit_machine_learning.utils import algorithm_globals
try:
    from qiskit_machine_learning.algorithms import VQC
except Exception:
    from qiskit_machine_learning.algorithms.classifiers import VQC

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# =========================
# 0) AYARLAR
# =========================
DATA_DIR = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")
TRAIN_CSV = DATA_DIR / "train_labeled_full.csv"
TEST_CSV  = DATA_DIR / "test_labeled_full.csv"

OUT_DIR = DATA_DIR / "qml_out" / "vqc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_QUBITS = 8
MAX_TRAIN = 800
MAX_TEST  = 3000

# ---- Threshold seçimi modu ----
# "f0_5"  -> precision ağırlıklı (önerilen)
# "fpr_cap" -> önce FPR limiti (FP azalır), sonra recall'u mümkün olduğunca yüksek tut
THRESHOLD_SELECTION = "f0_5"
BETA = 0.5
FPR_TARGET = 0.01

DEFAULT_THRESHOLD = 0.85

# ---- VQC hiperparametreleri ----
REPS_FEATURE_MAP = 2
REPS_ANSATZ = 2
MAXITER = 300

USE_SPSA = True
SPSA_LR = 0.05
SPSA_PERT = 0.1

VAL_SIZE = 0.2

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
    y = np.asarray(y).ravel()
    s = pd.Series(y)

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

def get_score01(model, X) -> np.ndarray:
    # 1) predict_proba varsa
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1].astype(float)
            if proba.ndim == 1:
                return proba.astype(float)
        except Exception:
            pass

    # 2) decision_function varsa
    if hasattr(model, "decision_function"):
        try:
            dec = np.asarray(model.decision_function(X)).ravel()
            return safe_minmax01(dec).astype(float)
        except Exception:
            pass

    # 3) yoksa predict (0/1)
    yhat = np.asarray(model.predict(X)).astype(int).ravel()
    return yhat.astype(float)

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
        else:
            key = f"f{BETA}"
            if best is None or (cur[key] > best[key]) or (
                np.isclose(cur[key], best[key]) and cur["precision"] > best["precision"]
            ) or (
                np.isclose(cur[key], best[key]) and np.isclose(cur["precision"], best["precision"]) and cur["recall"] > best["recall"]
            ):
                best = cur

    if best is None:
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

def build_sampler():
    try:
        from qiskit_aer.primitives import Sampler as AerSampler
        return AerSampler()
    except Exception:
        try:
            from qiskit.primitives import Sampler
            return Sampler()
        except Exception:
            return None

def build_quantum_instance(seed: int):
    try:
        from qiskit.utils import QuantumInstance
        try:
            from qiskit_aer import Aer
            backend = Aer.get_backend("aer_simulator_statevector")
            return QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        except Exception:
            from qiskit import BasicAer
            backend = BasicAer.get_backend("qasm_simulator")
            return QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed, shots=1024)
    except Exception:
        return None

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

log("[5/8] Train/Val split yapılıyor (threshold tuning için)...")
Xtr, Xval, ytr, yval = train_test_split(
    Xtr_all, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
)
log(f"     Train used: {len(Xtr)} | Val: {len(Xval)}")

n_features = Xtr.shape[1]

# =========================
# 4) VQC KURULUMU
# =========================
log("[6/8] VQC kuruluyor ve eğitiliyor...")
feature_map = ZZFeatureMap(feature_dimension=n_features, reps=REPS_FEATURE_MAP, entanglement="full")
ansatz = RealAmplitudes(num_qubits=n_features, reps=REPS_ANSATZ, entanglement="full")
log(f"     feature_map reps={REPS_FEATURE_MAP} | ansatz reps={REPS_ANSATZ} | qubits={n_features}")

if USE_SPSA:
    try:
        from qiskit_algorithms.optimizers import SPSA
    except Exception:
        from qiskit.algorithms.optimizers import SPSA
    optimizer = SPSA(maxiter=MAXITER, learning_rate=SPSA_LR, perturbation=SPSA_PERT)
    log(f"     Optimizer: SPSA | maxiter={MAXITER} | lr={SPSA_LR} | pert={SPSA_PERT}")
else:
    try:
        from qiskit_algorithms.optimizers import COBYLA
    except Exception:
        from qiskit.algorithms.optimizers import COBYLA
    optimizer = COBYLA(maxiter=MAXITER)
    log(f"     Optimizer: COBYLA | maxiter={MAXITER}")

loss_history: list[float] = []
def _callback(*args):
    for a in reversed(args):
        if isinstance(a, (float, np.floating)):
            loss_history.append(float(a))
            break

sig = inspect.signature(VQC)
kwargs = dict(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
if "callback" in sig.parameters:
    kwargs["callback"] = _callback

sampler = build_sampler()
if sampler is not None and "sampler" in sig.parameters:
    kwargs["sampler"] = sampler
    log("     Sampler: OK")

qi = build_quantum_instance(RANDOM_STATE)
if qi is not None and "quantum_instance" in sig.parameters:
    kwargs["quantum_instance"] = qi
    log("     QuantumInstance: OK")

vqc = VQC(**kwargs)

sample_weight = compute_sample_weight(class_weight="balanced", y=ytr)

t0 = time.time()
try:
    vqc.fit(Xtr, ytr, sample_weight=sample_weight)
except TypeError:
    vqc.fit(Xtr, ytr)
train_time_s = time.time() - t0
log(f"     Eğitim süresi: {train_time_s:.2f} s")

# =========================
# 5) THRESHOLD SEÇ (VAL)
# =========================
log("[7/8] Validation üzerinde en iyi threshold seçiliyor...")
val_score01 = get_score01(vqc, Xval)
best_val = find_best_threshold(yval, val_score01)
best_thr = best_val["thr"]
log(f"     Seçilen threshold: {best_thr:.4f} | mode={THRESHOLD_SELECTION} | val={best_val}")

# =========================
# 6) TEST EVAL
# =========================
log("[8/8] Test değerlendiriliyor + dosyalar kaydediliyor...")
test_score01 = get_score01(vqc, Xte)
y_pred = (test_score01 >= best_thr).astype(int)

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

# Kaydet
cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
cm_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=True)

pred_df = pd.DataFrame(
    {
        "y_true": y_test,
        "score01": np.asarray(test_score01).ravel(),
        "y_pred": y_pred,
    }
)
pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

metrics = {
    "model": "VQC",
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
    "optimizer": ("SPSA" if USE_SPSA else "COBYLA"),
    "optimizer_params": ({"maxiter": int(MAXITER), "learning_rate": float(SPSA_LR), "perturbation": float(SPSA_PERT)}
                         if USE_SPSA else {"maxiter": int(MAXITER)}),
    "params": {
        "reps_feature_map": int(REPS_FEATURE_MAP),
        "reps_ansatz": int(REPS_ANSATZ),
        "n_qubits": int(n_features),
    },
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
    "loss_history": loss_history[:],
}
with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# ML-style çıktı
print_ml_style("VQC", tn, fp, fn, tp, best_thr, acc, prec, rec, f1v, err)
log(f"ROC-AUC: {roc} | AP: {ap}")
log(f"Output: {OUT_DIR}")
log("✅ VQC bitti.")
