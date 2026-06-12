"""Model metrics and simple trading backtest helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mape),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
        "DirectionHitRate": float((y_true == y_pred).mean()) if len(y_true) else np.nan,
    }


def max_drawdown(cumulative_return: pd.Series) -> float:
    curve = 1 + cumulative_return.fillna(0)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return float(drawdown.min())


def backtest_long_flat(
    actual_return: np.ndarray | pd.Series,
    predicted_direction: np.ndarray | pd.Series,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Hold the stock when the model predicts an up move, otherwise stay flat.

    transaction_cost is a single-side cost applied when the position changes
    between flat and long. A 0.001 value means 0.1% per buy or sell.
    """
    actual = pd.Series(actual_return, dtype=float).fillna(0)
    signal = pd.Series(predicted_direction, dtype=float).fillna(0).clip(lower=0, upper=1)
    turnover = signal.diff().abs().fillna(signal.abs())
    cost = turnover * max(transaction_cost, 0.0)
    strategy_return = actual * signal - cost
    out = pd.DataFrame(
        {
            "strategy_return": strategy_return,
            "buy_hold_return": actual,
            "position": signal,
            "turnover": turnover,
            "transaction_cost": cost,
        }
    )
    out["strategy_cum_return"] = (1 + out["strategy_return"]).cumprod() - 1
    out["buy_hold_cum_return"] = (1 + out["buy_hold_return"]).cumprod() - 1
    out["max_drawdown"] = max_drawdown(out["strategy_cum_return"])
    return out


def confusion_matrix_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
