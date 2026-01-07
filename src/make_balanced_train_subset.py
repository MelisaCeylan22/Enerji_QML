# src/make_balanced_train_subset.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_uci(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    # timestamp
    ts = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df.insert(0, "timestamp", ts)
    df = df.drop(columns=["Date", "Time"])

    # numeric cast (UCI sometimes comes as object because of '?')
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort & clean timestamp
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, test_size: float):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="UCI txt/csv path (e.g., data/raw/household_power_consumption.txt)")
    ap.add_argument("--outdir", default="data/processed", help="output directory")
    ap.add_argument("--q", type=float, default=0.99, help="quantile threshold (train-based)")
    ap.add_argument("--test-size", type=float, default=0.2, help="chronological test ratio")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_uci(args.raw)

    # Drop NA for the columns you relied on in your quality analysis (aligns with your earlier stats)
    required_cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]
    before = len(df)
    df = df.dropna(subset=required_cols).copy()
    after = len(df)

    train_df, test_df = chronological_split(df, test_size=args.test_size)

    # Threshold computed ONLY on train
    threshold = float(train_df["Global_active_power"].quantile(args.q))

    # Labels using the same threshold for train/test
    train_df["label"] = (train_df["Global_active_power"] > threshold).astype(np.int8)
    test_df["label"] = (test_df["Global_active_power"] > threshold).astype(np.int8)

    # Balanced subset from TRAIN: take all anomalies + same number of normals
    pos = train_df[train_df["label"] == 1].copy()
    neg = train_df[train_df["label"] == 0].copy()

    n_pos = len(pos)
    if n_pos == 0:
        raise RuntimeError("Train içinde hiç anomali yok. q değerini düşürmeyi dene (örn. 0.98).")

    neg_sample = neg.sample(n=n_pos, random_state=args.seed, replace=False)

    train_bal = pd.concat([pos, neg_sample], axis=0)

    # (Opsiyonel) shuffle: model eğitiminde sıra önemli değilse açabilirsin
    train_bal = train_bal.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Save
    train_df.to_csv(outdir / "train_labeled_full.csv", index=False)
    test_df.to_csv(outdir / "test_labeled_full.csv", index=False)
    train_bal.to_csv(outdir / "train_labeled_balanced.csv", index=False)

    # Quick report to terminal
    print("=== SUMMARY ===")
    print(f"input_rows_before_dropna: {before}")
    print(f"input_rows_after_dropna : {after}")
    print(f"split: chronological | test_size={args.test_size}")
    print(f"quantile_q: {args.q}")
    print(f"threshold_pq_train_global_active_power: {threshold:.6f}")
    print(f"train_rows: {len(train_df)} | test_rows: {len(test_df)}")
    print(f"label_pos_ratio_train: {train_df['label'].mean():.6f}")
    print(f"label_pos_ratio_test : {test_df['label'].mean():.6f}")
    print("--- balanced train ---")
    print(f"train_bal_rows: {len(train_bal)} (pos={n_pos}, neg={n_pos})")


if __name__ == "__main__":
    main()
