from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_uci(path: Path) -> pd.DataFrame:
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
        raise ValueError(f"Eksik kolon(lar): {missing_cols}. Bulunan kolonlar: {list(df.columns)}")

    # datetime parse
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df.insert(0, "datetime", dt)
    df = df.drop(columns=["Date", "Time"])

    # drop invalid datetime rows and set index
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # numeric conversion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop duplicate timestamps if any (senin verinde 0 ama garanti)
    df = df[~df.index.duplicated(keep="first")]

    return df


def basic_physical_filters(df: pd.DataFrame) -> pd.DataFrame:
    # Senin analizinde invalid yoktu, ama yine de güvenli filtre
    if "Global_active_power" in df.columns:
        df.loc[df["Global_active_power"] < 0, "Global_active_power"] = np.nan
    if "Global_reactive_power" in df.columns:
        df.loc[df["Global_reactive_power"] < 0, "Global_reactive_power"] = np.nan
    if "Global_intensity" in df.columns:
        df.loc[df["Global_intensity"] < 0, "Global_intensity"] = np.nan
    if "Voltage" in df.columns:
        df.loc[df["Voltage"] <= 0, "Voltage"] = np.nan

    for c in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    return df


def split_indices(df: pd.DataFrame, test_size: float, split: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    if n < 10:
        raise ValueError("Veri çok az; split yapılamaz.")

    if split == "chronological":
        cut = int(round((1.0 - test_size) * n))
        cut = max(1, min(cut, n - 1))
        train_idx = np.arange(0, cut)
        test_idx = np.arange(cut, n)
        return train_idx, test_idx

    if split == "random":
        rng = np.random.default_rng(seed)
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        test_n = int(round(test_size * n))
        test_n = max(1, min(test_n, n - 1))
        test_idx = all_idx[:test_n]
        train_idx = all_idx[test_n:]
        # train/test sırası fark etmez ama çıktı daha okunur olsun:
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
        return train_idx, test_idx

    raise ValueError("split parametresi 'chronological' veya 'random' olmalı.")


def compute_threshold(train_gap: pd.Series, q: float) -> float:
    train_gap = train_gap.dropna()
    if train_gap.empty:
        raise ValueError("Train setinde Global_active_power boş; threshold hesaplanamaz.")
    return float(train_gap.quantile(q))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.endswith(".gz"):
        df.to_csv(path, index=True, compression="gzip")
    else:
        df.to_csv(path, index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw txt path. Örn: data/raw/household_power_consumption.txt")
    ap.add_argument("--out_dir", default="data/processed", help="Çıktı klasörü (default: data/processed)")
    ap.add_argument("--q", type=float, default=0.99, help="Quantile (default: 0.99 => p99)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test oranı (default: 0.2)")
    ap.add_argument("--split", choices=["chronological", "random"], default="chronological",
                    help="Split tipi (default: chronological)")
    ap.add_argument("--seed", type=int, default=42, help="Random split için seed")
    ap.add_argument("--dropna", choices=["gap_only", "all_numeric"], default="all_numeric",
                    help="Eksik veri satırlarını eleme stratejisi")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("outputs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    df = load_uci(in_path)
    df = basic_physical_filters(df)

    # Eksik satırları eleme politikası
    if args.dropna == "gap_only":
        df = df.dropna(subset=["Global_active_power"])
    else:
        numeric_cols = [
            "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        ]
        df = df.dropna(subset=numeric_cols)

    df = df.sort_index()
    n = len(df)

    train_idx, test_idx = split_indices(df, test_size=args.test_size, split=args.split, seed=args.seed)

    # Threshold sadece train'den (leakage yok)
    train_gap = df["Global_active_power"].iloc[train_idx]
    thr = compute_threshold(train_gap, q=args.q)

    # Label üretimi: dakikalık GAP >= p99(train) => 1
    df_l = df.copy()
    df_l["label"] = (df_l["Global_active_power"] >= thr).astype(int)

    # Train/Test flag (rapor/analiz için faydalı)
    df_l["split"] = "train"
    df_l.iloc[test_idx, df_l.columns.get_loc("split")] = "test"

    # Save outputs (gzip önerilir; çok büyük)
    train_df = df_l.iloc[train_idx].copy()
    test_df = df_l.iloc[test_idx].copy()

    train_path = out_dir / "uci_minute_p99_train.csv.gz"
    test_path = out_dir / "uci_minute_p99_test.csv.gz"
    full_path = out_dir / "uci_minute_p99_full.csv.gz"

    save_csv(train_df, train_path)
    save_csv(test_df, test_path)
    save_csv(df_l, full_path)

    summary = {
        "input_rows_after_dropna": int(n),
        "split": args.split,
        "test_size": args.test_size,
        "quantile_q": args.q,
        "threshold_pq_train_global_active_power": thr,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_pos_ratio_train": float(train_df["label"].mean()),
        "label_pos_ratio_test": float(test_df["label"].mean()),
        "label_pos_ratio_full": float(df_l["label"].mean()),
        "time_start": str(df_l.index.min()),
        "time_end": str(df_l.index.max()),
    }
    (logs_dir / "labeling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] Threshold (train):", thr)
    print("[OK] Train pos ratio:", summary["label_pos_ratio_train"])
    print("[OK] Test  pos ratio:", summary["label_pos_ratio_test"])
    print("[OK] Saved:")
    print(" -", train_path)
    print(" -", test_path)
    print(" -", full_path)
    print("[OK] Summary:", logs_dir / "labeling_summary.json")


if __name__ == "__main__":
    main()
