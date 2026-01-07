from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_base_freq_minutes(dt_index: pd.DatetimeIndex) -> dict:
    """
    Zaman serisinin temel örnekleme aralığını (dakika) tahmin eder.
    """
    diffs = dt_index.to_series().diff().dropna().dt.total_seconds().astype(int)
    if diffs.empty:
        return {"base_freq_min": None, "diffs_summary": {}}

    # en sık görülen fark (mode)
    mode_sec = int(diffs.mode().iloc[0])
    base_min = mode_sec / 60

    diffs_summary = {
        "mode_seconds": mode_sec,
        "mode_minutes": base_min,
        "min_seconds": int(diffs.min()),
        "max_seconds": int(diffs.max()),
        "p50_seconds": int(diffs.quantile(0.50)),
        "p95_seconds": int(diffs.quantile(0.95)),
        "p99_seconds": int(diffs.quantile(0.99)),
    }
    return {"base_freq_min": base_min, "diffs_summary": diffs_summary}


def load_dataset(path: Path) -> pd.DataFrame:
    """
    UCI veri setini okur, datetime index oluşturur, numeric kolonları sayısala çevirir.
    """
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?", "NA", "NaN", ""],
        low_memory=False,
    )

    required = [
        "Date", "Time",
        "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik kolon(lar) var: {missing_cols}. Bulunan kolonlar: {list(df.columns)}")

    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )
    df.insert(0, "datetime", dt)
    df = df.drop(columns=["Date", "Time"])

    # datetime bozuk satırlar
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # numeric parse
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------------------
# 1) Veri bütünlüğü & zaman yapısı
# ---------------------------
def analyze_integrity_and_time(df: pd.DataFrame, expected_freq: str = "1min") -> dict:
    out = {}

    out["rows_total"] = int(df.shape[0])
    out["cols_total"] = int(df.shape[1])
    out["start"] = str(df.index.min())
    out["end"] = str(df.index.max())

    # duplicate timestamps
    dup_mask = df.index.duplicated(keep=False)
    out["duplicate_timestamps"] = int(dup_mask.sum())

    # monotonic
    out["is_monotonic_increasing"] = bool(df.index.is_monotonic_increasing)

    # freq detection
    freq_info = detect_base_freq_minutes(df.index)
    out.update(freq_info)

    # missing intervals based on expected frequency
    # NOTE: expected_freq "1min" ise full timeline oluşturur.
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    missing_ts = full_idx.difference(df.index.unique())
    out["expected_freq"] = expected_freq
    out["expected_total_points"] = int(len(full_idx))
    out["missing_timestamps_count"] = int(len(missing_ts))
    out["missing_timestamps_ratio"] = float(len(missing_ts) / len(full_idx)) if len(full_idx) else None

    # en büyük gap (dakika)
    diffs = df.index.to_series().diff().dropna()
    if not diffs.empty:
        max_gap = diffs.max()
        out["max_gap"] = str(max_gap)
        out["max_gap_minutes"] = float(max_gap.total_seconds() / 60)
    else:
        out["max_gap"] = None
        out["max_gap_minutes"] = None

    return out, missing_ts


# ---------------------------
# 2) Eksik değer analizi
# ---------------------------
def analyze_missingness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    - sütun bazlı eksik oran
    - gün bazlı eksik oran (zaman içinde kümelenme var mı?)
    """
    col_missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_ratio": df.isna().mean(),
        "non_missing_count": df.notna().sum(),
    }).sort_values("missing_ratio", ascending=False)

    # Günlük missing pattern (hangi günlerde eksik artıyor)
    daily = df.isna().resample("1D").mean()
    daily["rows_that_day"] = df.resample("1D").size()
    return col_missing, daily


# ---------------------------
# 3) Geçersiz/bozuk değer kontrolü
# ---------------------------
def analyze_invalid_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fiziksel/mantıksal olarak bariz bozuk değerleri raporlar.
    NOT: Üst sınırlar tartışmalı olabilir. Bu yüzden iki yaklaşım:
    - Kesin bozuklar: negatifler, Voltage<=0 gibi
    - Aşırı uçlar: %99.9 üstü gibi (outlier adayı)
    """

    rules = {
        "Global_active_power": {"min": 0.0, "max": None},      # kW, negatif bariz bozuk
        "Global_reactive_power": {"min": 0.0, "max": None},    # kW, negatif bariz bozuk (genelde >=0)
        "Voltage": {"min": 1.0, "max": None},                  # V, 0/negatif bozuk
        "Global_intensity": {"min": 0.0, "max": None},         # A, negatif bozuk
        "Sub_metering_1": {"min": 0.0, "max": None},           # Wh, negatif bozuk
        "Sub_metering_2": {"min": 0.0, "max": None},
        "Sub_metering_3": {"min": 0.0, "max": None},
    }

    rows = []
    details = {}

    for col, lim in rules.items():
        s = df[col]

        invalid_min = (s < lim["min"]) if lim["min"] is not None else pd.Series(False, index=s.index)
        invalid_max = (s > lim["max"]) if lim["max"] is not None else pd.Series(False, index=s.index)

        invalid = invalid_min | invalid_max
        invalid_count = int(invalid.sum())
        invalid_ratio = float(invalid.mean())

        # aşırı uç: 99.9 percentile üstü (NaN hariç)
        q999 = float(s.quantile(0.999)) if s.notna().any() else np.nan
        extreme = (s > q999) & s.notna()
        extreme_count = int(extreme.sum())
        extreme_ratio = float(extreme.mean())

        rows.append({
            "column": col,
            "invalid_count": invalid_count,
            "invalid_ratio": invalid_ratio,
            "min_rule": lim["min"],
            "max_rule": lim["max"],
            "p0_min": float(s.min()) if s.notna().any() else np.nan,
            "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
            "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
            "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
            "p999": q999,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "extreme_count_p999": extreme_count,
            "extreme_ratio_p999": extreme_ratio,
        })

        # örnek timestamp listesi (ilk 20)
        details[col] = {
            "invalid_examples": [str(x) for x in s[invalid].head(20).index],
            "extreme_examples_p999": [str(x) for x in s[extreme].head(20).index],
        }

    summary_df = pd.DataFrame(rows).sort_values(["invalid_ratio", "extreme_ratio_p999"], ascending=False)
    return summary_df, details


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Raw dataset path (txt/csv). Örn: data/raw/household_power_consumption.txt")
    ap.add_argument("--expected_freq", type=str, default="1min", help="Beklenen örnekleme frekansı (default: 1min)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    ensure_dir(out_dir)

    df = load_dataset(path)

    # 1) integrity & time
    integrity, missing_ts = analyze_integrity_and_time(df, expected_freq=args.expected_freq)
    (out_dir / "integrity_time_summary.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    # missing timestamp list (çoksa sadece ilk 50)
    missing_preview = pd.Series(missing_ts[:50].astype(str), name="missing_timestamps_preview")
    missing_preview.to_csv(out_dir / "missing_timestamps_preview.csv", index=False)

    # 2) missingness
    col_missing, daily_missing = analyze_missingness(df)
    col_missing.to_csv(out_dir / "missing_by_column.csv")
    daily_missing.to_csv(out_dir / "missing_daily_ratio.csv")

    # 3) invalid values
    invalid_summary, invalid_details = analyze_invalid_values(df)
    invalid_summary.to_csv(out_dir / "invalid_values_summary.csv", index=False)
    (out_dir / "invalid_values_examples.json").write_text(json.dumps(invalid_details, indent=2), encoding="utf-8")

    # terminale kısa özet
    print("\n=== 1) Veri Bütünlüğü & Zaman Yapısı ===")
    print(json.dumps(integrity, indent=2))

    print("\n=== 2) Eksik Değer (Sütun Bazlı) İlk 10 ===")
    print(col_missing.head(10))

    print("\n=== 3) Geçersiz/Bozuk Değer Özeti (İlk 10) ===")
    print(invalid_summary.head(10))

    print(f"\n[OK] Tüm çıktılar yazıldı: {out_dir}")


if __name__ == "__main__":
    main()
