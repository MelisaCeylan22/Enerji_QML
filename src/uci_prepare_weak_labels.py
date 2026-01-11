from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# 0) SABİT AYARLAR (BURAYI DÜZENLE)
# =========================
INPUT_TXT = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\raw\household_power_consumption.txt")
OUT_DIR   = Path(r"C:\Users\Melisa\Desktop\Enerji_QML\data\processed")

TRAIN_RATIO   = 0.80
DRIFT_WINDOW  = 60    # dakika
MIN_PERSIST   = 10    # dakika (süren sinyaller için)

# Quantiles (eşikler train'den öğrenilecek)
Q_PEAK   = 0.99
Q_STEP   = 0.995
Q_RESID  = 0.995
Q_PF_LOW = 0.01
Q_V_LOW  = 0.005
Q_V_HIGH = 0.995

USE_VOLTAGE_RULE = True  # False yaparsan voltage kuralı kapalı
DROP_DEBUG_COLS  = True # True yaparsan ara kolonlar silinir

FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# =========================
# 1) Yükleme + Temizleme
# =========================
def load_uci_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?", "NA", "NaN", ""],
        low_memory=False,
    )

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # time-series friendly missing handling
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(limit_direction="both")
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

    df = df.drop(columns=["Date", "Time"], errors="ignore")
    return df

def time_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def run_length_filter(flag: pd.Series, min_run: int) -> pd.Series:
    flag = flag.fillna(False).astype(bool)
    if min_run <= 1:
        return flag
    grp = (flag != flag.shift(1)).cumsum()
    run_len = flag.groupby(grp).transform("sum")
    return flag & (run_len >= min_run)

# =========================
# 2) Threshold Fit (sadece train)
# =========================
def fit_thresholds(train_df: pd.DataFrame) -> dict:
    P = train_df["Global_active_power"].astype(float)
    Q = train_df["Global_reactive_power"].astype(float)
    V = train_df["Voltage"].astype(float)

    thr_peak = float(P.quantile(Q_PEAK))

    dP = P.diff().abs()
    thr_step = float(dP.dropna().quantile(Q_STEP))

    rollP = P.rolling(window=DRIFT_WINDOW, min_periods=DRIFT_WINDOW).mean()
    thr_drift = float(rollP.dropna().quantile(0.95))

    # Residual mismatch (Wh/min)
    energy_wh = (P * 1000.0) / 60.0
    sub_sum = train_df["Sub_metering_1"] + train_df["Sub_metering_2"] + train_df["Sub_metering_3"]
    resid = energy_wh - sub_sum
    thr_resid = float(resid.abs().dropna().quantile(Q_RESID))

    # PF low
    denom = np.sqrt((P ** 2) + (Q ** 2))
    pf = (P / denom).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 1.0)
    thr_pf_low = float(pf.quantile(Q_PF_LOW))

    thr_v_low = float(V.quantile(Q_V_LOW))
    thr_v_high = float(V.quantile(Q_V_HIGH))

    return {
        "input_txt": str(INPUT_TXT),
        "train_ratio": TRAIN_RATIO,
        "drift_window": DRIFT_WINDOW,
        "min_persist": MIN_PERSIST,

        "q_peak": Q_PEAK,
        "q_step": Q_STEP,
        "q_resid": Q_RESID,
        "q_pf_low": Q_PF_LOW,
        "q_v_low": Q_V_LOW,
        "q_v_high": Q_V_HIGH,

        "thr_peak_gap": thr_peak,
        "thr_step_abs_dgap": thr_step,
        "thr_drift_rollmean_gap": thr_drift,
        "thr_resid_abs_wh": thr_resid,
        "thr_pf_low": thr_pf_low,
        "thr_voltage_low": thr_v_low,
        "thr_voltage_high": thr_v_high,
        "use_voltage_rule": USE_VOLTAGE_RULE,
    }

# =========================
# 3) Kuralları uygula + label üret
# =========================
def apply_rules(df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    out = df.copy()

    P = out["Global_active_power"].astype(float)
    Q = out["Global_reactive_power"].astype(float)
    V = out["Voltage"].astype(float)

    out["gap_rollmean"] = P.rolling(window=DRIFT_WINDOW, min_periods=DRIFT_WINDOW).mean()
    out["gap_abs_diff"] = P.diff().abs()

    energy_wh = (P * 1000.0) / 60.0
    sub_sum = out["Sub_metering_1"] + out["Sub_metering_2"] + out["Sub_metering_3"]
    out["resid_wh"] = energy_wh - sub_sum

    denom = np.sqrt((P ** 2) + (Q ** 2))
    pf = (P / denom).replace([np.inf, -np.inf], np.nan)
    out["power_factor"] = pf.fillna(1.0).clip(0.0, 1.0)

    # raw flags
    out["r_peak_p99"] = (P >= thr["thr_peak_gap"])
    out["r_step"] = (out["gap_abs_diff"] >= thr["thr_step_abs_dgap"])
    out["r_drift"] = (out["gap_rollmean"] >= thr["thr_drift_rollmean_gap"])
    out["r_resid"] = (out["resid_wh"].abs() >= thr["thr_resid_abs_wh"])
    out["r_pf_low"] = (out["power_factor"] <= thr["thr_pf_low"])

    if USE_VOLTAGE_RULE:
        out["r_voltage_out"] = (V <= thr["thr_voltage_low"]) | (V >= thr["thr_voltage_high"])
    else:
        out["r_voltage_out"] = False

    # persistence
    out["r_drift_persist"] = run_length_filter(out["r_drift"], min_run=MIN_PERSIST)
    out["r_resid_persist"] = run_length_filter(out["r_resid"], min_run=MIN_PERSIST)
    out["r_pf_persist"] = run_length_filter(out["r_pf_low"], min_run=MIN_PERSIST)

    # score
    rule_cols = ["r_peak_p99", "r_step", "r_drift_persist", "r_resid_persist", "r_pf_persist", "r_voltage_out"]
    out["weak_score"] = out[rule_cols].astype(int).sum(axis=1)

    # FINAL LABEL
    out["label"] = (
        out["r_drift_persist"]
        | out["r_resid_persist"]
        | out["r_pf_persist"]
        | (out["r_step"] & out["r_voltage_out"])
        | out["r_peak_p99"]
    ).astype(int)

    return out

# =========================
# 4) MAIN
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading:", INPUT_TXT)
    df = load_uci_txt(INPUT_TXT)

    raw_train, raw_test = time_split(df, TRAIN_RATIO)

    print("Fitting thresholds on TRAIN only...")
    thr = fit_thresholds(raw_train)

    print("Applying rules + label...")
    train_l = apply_rules(raw_train, thr)
    test_l  = apply_rules(raw_test, thr)

    # Optionally drop debug cols
    if DROP_DEBUG_COLS:
        drop_cols = [
            "gap_rollmean","gap_abs_diff","resid_wh","power_factor",
            "r_peak_p99","r_step","r_drift","r_resid","r_pf_low","r_voltage_out",
            "r_drift_persist","r_resid_persist","r_pf_persist","weak_score",
        ]
        train_l = train_l.drop(columns=[c for c in drop_cols if c in train_l.columns])
        test_l  = test_l.drop(columns=[c for c in drop_cols if c in test_l.columns])

    train_path = OUT_DIR / "train_labeled_full.csv"
    test_path  = OUT_DIR / "test_labeled_full.csv"
    thr_path   = OUT_DIR / "weak_thresholds.json"

    train_l.to_csv(train_path, index=False)
    test_l.to_csv(test_path, index=False)
    thr_path.write_text(json.dumps(thr, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print("total rows:", len(df))
    print("train rows:", len(train_l), "pos_ratio(label):", float(train_l["label"].mean()))
    print("test  rows:", len(test_l),  "pos_ratio(label):", float(test_l["label"].mean()))
    print("saved:", train_path)
    print("saved:", test_path)
    print("saved:", thr_path)

if __name__ == "__main__":
    main()
