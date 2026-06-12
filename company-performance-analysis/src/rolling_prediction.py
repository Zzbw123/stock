"""Rolling out-of-sample predictions and latest 5-day direction signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lstm_predict import (
    FIGURE_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    LSTMConfig,
    TARGET_DIRECTION_TEMPLATE,
    TARGET_RETURN_TEMPLATE,
    load_or_build_model_data,
    predict_model,
    restrict_to_common_fusion_period,
    select_feature_columns,
    train_model,
)
from rolling_validation import run_rolling_validation, save_outputs


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _make_train_sequences(
    feature_values: np.ndarray,
    targets: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for pos in range(window, len(targets)):
        x.append(feature_values[pos - window : pos])
        y.append(targets[pos])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def latest_signal(
    df: pd.DataFrame,
    model_type: str,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    threshold: float,
    seed: int,
) -> pd.DataFrame:
    """Train on all labeled history and predict the latest available feature row."""
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=horizon)
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=horizon)
    feature_cols = select_feature_columns(df, model_type, horizon)
    keep = list(dict.fromkeys(["date", "close", target_return, target_direction] + feature_cols))
    work = df[keep].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in ["close", target_return, target_direction] + feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).sort_values("date").reset_index(drop=True)

    labeled = work.dropna(subset=[target_direction] + feature_cols).reset_index(drop=True)
    latest_features = work.dropna(subset=feature_cols).reset_index(drop=True)
    if len(labeled) <= window + 30 or len(latest_features) <= window:
        raise ValueError("Not enough rows for latest rolling prediction.")

    split = max(int((len(labeled) - window) * 0.85), 1)
    sequence_count = len(labeled) - window
    train_idx = np.arange(split)
    valid_idx = np.arange(split, sequence_count)
    if len(valid_idx) == 0:
        valid_idx = train_idx[-max(1, len(train_idx) // 10) :]
        train_idx = train_idx[: -len(valid_idx)]

    train_target_end = int(window + train_idx[-1])
    scaler = StandardScaler()
    scaler.fit(labeled.loc[:train_target_end, feature_cols])
    scaled_labeled = scaler.transform(labeled[feature_cols])
    x, y = _make_train_sequences(scaled_labeled, labeled[target_direction].to_numpy(dtype=float), window)

    model, _loss = train_model(
        x,
        y,
        train_idx,
        valid_idx,
        task="classification",
        config=LSTMConfig(),
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        class_weight="balanced",
    )

    latest_scaled = scaler.transform(latest_features[feature_cols])
    latest_x = latest_scaled[-window:][None, :, :].astype(np.float32)
    pred, prob = predict_model(model, latest_x, task="classification", threshold=threshold)
    latest_row = latest_features.iloc[-1]
    return pd.DataFrame(
        [
            {
                "date": latest_row["date"],
                "model": model_type,
                "window": window,
                "horizon": horizon,
                "close": latest_row["close"],
                "predicted_direction": int(pred[0]),
                "predicted_probability": float(prob[0]),
                "threshold": threshold,
                "signal": "hold_long" if int(pred[0]) == 1 else "stay_flat",
                "note": "Predicted from latest available feature row; future return is not yet known.",
            }
        ]
    )


def plot_rolling_signals(predictions: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, group in predictions.groupby("model"):
        if model_name == "naive_momentum":
            continue
        group = group.sort_values("date")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        colors = np.where(group["predicted_direction"].astype(int) == 1, "#16a34a", "#dc2626")
        ax1.plot(pd.to_datetime(group["date"]), group["close"], color="#2563eb", linewidth=1.5, label="close")
        ax1.scatter(pd.to_datetime(group["date"]), group["close"], c=colors, s=18, alpha=0.7, label="signal")
        ax1.set_title(f"Rolling prediction signals - {model_name}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close")
        ax1.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"rolling_prediction_signals_{model_name}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling predictions and latest future-5-day signal.")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "lstm_model_data.csv"))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--latest-model", choices=["base", "fusion"], default="fusion")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    predictions, metrics = run_rolling_validation(
        data_path=args.data,
        window=args.window,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        first_test_year=args.first_test_year,
        transaction_cost=args.transaction_cost,
        seed=args.seed,
    )
    save_outputs(predictions, metrics)
    predictions.to_csv(TABLE_DIR / "rolling_prediction_signals.csv", index=False, encoding="utf-8-sig")
    plot_rolling_signals(predictions)

    df = load_or_build_model_data(args.data, args.horizon)
    signal = latest_signal(
        df,
        model_type=args.latest_model,
        window=args.window,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold=args.threshold,
        seed=args.seed,
    )
    signal.to_csv(TABLE_DIR / "rolling_latest_signal.csv", index=False, encoding="utf-8-sig")
    print(f"Saved rolling prediction signals: {TABLE_DIR / 'rolling_prediction_signals.csv'}")
    print(f"Saved latest signal: {TABLE_DIR / 'rolling_latest_signal.csv'}")
    print(signal.to_string(index=False))


if __name__ == "__main__":
    main()
