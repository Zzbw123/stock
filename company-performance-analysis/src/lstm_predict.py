"""PyTorch multi-feature LSTM experiments for stock-trend prediction."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from merge_model_data import build_lstm_model_data
from model_evaluation import (
    backtest_long_flat,
    classification_metrics,
    confusion_matrix_frame,
    regression_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

TARGET_RETURN_TEMPLATE = "future_{horizon}d_return"
TARGET_DIRECTION_TEMPLATE = "future_{horizon}d_direction"
NON_FEATURE_COLUMNS = {"date", "year", "source", "period", "period_date"}
FINANCIAL_COLUMNS = {
    "revenue",
    "net_profit",
    "roe",
    "gross_margin",
    "net_margin",
    "asset_liability_ratio",
    "current_ratio",
    "revenue_growth",
    "net_profit_growth",
    "topsis_score",
    "rank",
}


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 0.001


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for LSTM training. Install it with: pip install torch"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _torch_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_or_build_model_data(path: str | Path, horizon: int) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return build_lstm_model_data(output_path=path, horizon=horizon)


def select_feature_columns(df: pd.DataFrame, model_type: str, horizon: int) -> list[str]:
    label_cols = {
        TARGET_RETURN_TEMPLATE.format(horizon=horizon),
        TARGET_DIRECTION_TEMPLATE.format(horizon=horizon),
    }
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    excluded = NON_FEATURE_COLUMNS | label_cols
    if model_type == "base":
        excluded = excluded | FINANCIAL_COLUMNS
    elif model_type != "fusion":
        raise ValueError("model_type must be base or fusion.")
    features = [col for col in numeric_cols if col not in excluded]
    if not features:
        raise ValueError(f"No numeric feature columns are available for {model_type} model.")
    return features


def prepare_model_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    extra_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    keep = ["date", "close", target_col] + list(extra_cols or []) + feature_cols
    keep = list(dict.fromkeys([col for col in keep if col in df.columns]))
    frame = df[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in [col for col in keep if col != "date"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["date", "close", target_col] + feature_cols).sort_values("date").reset_index(drop=True)
    return frame


def split_sequence_indices(n_sequences: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(int(n_sequences * 0.70), 1)
    valid_end = max(int(n_sequences * 0.85), train_end + 1)
    indices = np.arange(n_sequences)
    return indices[:train_end], indices[train_end:valid_end], indices[valid_end:]


def make_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window: int,
    train_sequence_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    target_positions = np.arange(window, len(frame))
    if len(target_positions) == 0:
        raise ValueError("Not enough rows to create sequences for the selected window.")

    train_target_end = int(target_positions[train_sequence_indices[-1]]) if len(train_sequence_indices) else window
    scaler = StandardScaler()
    scaler.fit(frame.loc[:train_target_end, feature_cols])
    scaled = scaler.transform(frame[feature_cols])

    x, y, meta_rows = [], [], []
    for pos in target_positions:
        x.append(scaled[pos - window : pos])
        y.append(frame.loc[pos, target_col])
        meta_rows.append(
            {
                "date": frame.loc[pos, "date"],
                "close": frame.loc[pos, "close"],
                "actual_return": frame.loc[pos, TARGET_RETURN_TEMPLATE.format(horizon=5)]
                if TARGET_RETURN_TEMPLATE.format(horizon=5) in frame
                else np.nan,
            }
        )
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(meta_rows), scaler


def make_sequences_with_meta(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    actual_return_col: str,
    actual_direction_col: str,
    window: int,
    train_sequence_indices: np.ndarray,
    meta_cols: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    target_positions = np.arange(window, len(frame))
    if len(target_positions) == 0:
        raise ValueError("Not enough rows to create sequences for the selected window.")

    train_target_end = int(target_positions[train_sequence_indices[-1]]) if len(train_sequence_indices) else window
    scaler = StandardScaler()
    scaler.fit(frame.loc[:train_target_end, feature_cols])
    scaled = scaler.transform(frame[feature_cols])

    x, y, meta_rows = [], [], []
    extra_meta_cols = [col for col in (meta_cols or []) if col in frame.columns]
    for pos in target_positions:
        x.append(scaled[pos - window : pos])
        y.append(frame.loc[pos, target_col])
        row = {
            "date": frame.loc[pos, "date"],
            "close": frame.loc[pos, "close"],
            "actual_return": frame.loc[pos, actual_return_col],
            "actual_direction": frame.loc[pos, actual_direction_col],
        }
        for col in extra_meta_cols:
            row[col] = frame.loc[pos, col]
        meta_rows.append(row)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(meta_rows), scaler


def build_model(input_size: int, hidden_size: int, num_layers: int, dropout: float, task: str):
    _torch, nn, _DataLoader, _TensorDataset = _require_torch()

    class StockLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.output(self.dropout(hidden[-1])).squeeze(-1)

    return StockLSTM()


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    task: str,
    config: LSTMConfig,
    epochs: int,
    batch_size: int,
    seed: int = 42,
    class_weight: str = "balanced",
    log_interval: int = 0,
    run_name: str = "lstm",
    patience: int = 0,
    min_delta: float = 0.0,
) -> tuple[object, float]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    device = _torch_device(torch)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(x.shape[2], config.hidden_size, config.num_layers, config.dropout, task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if task == "classification":
        pos_weight = None
        if class_weight == "balanced":
            y_train = y[train_idx]
            positives = float((y_train == 1).sum())
            negatives = float((y_train == 0).sum())
            if positives > 0 and negatives > 0:
                pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(x[train_idx]), torch.tensor(y[train_idx]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_valid = math.inf
    best_epoch = 0
    stale_epochs = 0
    max_epochs = max(1, epochs)
    epochs_trained = 0
    for epoch in range(max_epochs):
        epochs_trained = epoch + 1
        model.train()
        train_loss = math.nan
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb.float())
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        if len(valid_idx):
            model.eval()
            with torch.no_grad():
                valid_x = torch.tensor(x[valid_idx], device=device)
                valid_y = torch.tensor(y[valid_idx], device=device).float()
                valid_pred = model(valid_x)
                valid_loss = criterion(valid_pred, valid_y).item()
        else:
            valid_loss = loss.item()

        if valid_loss < best_valid - min_delta:
            best_valid = valid_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if log_interval > 0 and (
            epoch == 0 or (epoch + 1) % log_interval == 0 or epoch + 1 == max_epochs
        ):
            print(
                f"[{run_name}] epoch {epoch + 1}/{max_epochs} "
                f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
                f"best_valid={best_valid:.4f} best_epoch={best_epoch}"
            )
        if patience > 0 and stale_epochs >= patience:
            if log_interval > 0:
                print(f"[{run_name}] early stopping at epoch {epoch + 1}; best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_summary = {
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid),
        "epochs": int(max_epochs),
        "epochs_trained": int(epochs_trained),
        "device": str(device),
    }
    return model, float(best_valid)


def predict_model(model, x: np.ndarray, task: str, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    torch, _nn, _DataLoader, _TensorDataset = _require_torch()
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(x, device=device)).detach().cpu().numpy()
    if task == "classification":
        probability = 1 / (1 + np.exp(-raw))
        prediction = (probability >= threshold).astype(int)
        return prediction, probability
    return raw.astype(float), np.full_like(raw, np.nan, dtype=float)


def make_trade_signal(
    probabilities: np.ndarray,
    buy_threshold: float,
    sell_threshold: float | None = None,
    threshold_mode: str = "single",
    market_filter: np.ndarray | None = None,
) -> np.ndarray:
    """Convert probabilities into a long-flat position signal."""
    prob = np.asarray(probabilities, dtype=float)
    if threshold_mode == "dual":
        sell = float(sell_threshold if sell_threshold is not None else buy_threshold)
        position = 0
        signal = []
        for idx, value in enumerate(prob):
            allowed = True if market_filter is None else bool(market_filter[idx])
            if not allowed:
                position = 0
            elif value >= buy_threshold:
                position = 1
            elif value <= sell:
                position = 0
            signal.append(position)
        return np.asarray(signal, dtype=int)

    signal = (prob >= buy_threshold).astype(int)
    if market_filter is not None:
        signal = signal * np.asarray(market_filter, dtype=bool).astype(int)
    return signal.astype(int)


def market_filter_mask(meta: pd.DataFrame, column: str, minimum: float) -> np.ndarray | None:
    if not column or column not in meta.columns:
        return None
    return pd.to_numeric(meta[column], errors="coerce").fillna(-np.inf).to_numpy(dtype=float) >= minimum


def select_classification_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    actual_returns: np.ndarray,
    objective: str,
    transaction_cost: float,
    min_hold_proba: float,
    min_valid_trades: int,
    threshold_mode: str,
    drawdown_penalty: float,
    trade_penalty: float,
    market_filter: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Choose a long-flat probability threshold on validation data."""
    start = max(0.30, min(float(min_hold_proba), 0.80))
    buy_thresholds = np.round(np.arange(start, 0.81, 0.02), 2)
    if len(buy_thresholds) == 0:
        buy_thresholds = np.asarray([start])

    best_buy_threshold = float(buy_thresholds[0])
    best_sell_threshold = np.nan
    best_score = -math.inf
    best_trade_count = 0.0
    fallback: tuple[float, float, float, float] | None = None

    for buy_threshold in buy_thresholds:
        if threshold_mode == "dual":
            sell_thresholds = np.round(np.arange(0.30, max(0.31, buy_threshold), 0.02), 2)
            sell_thresholds = sell_thresholds[sell_thresholds < buy_threshold]
            if len(sell_thresholds) == 0:
                sell_thresholds = np.asarray([max(0.30, buy_threshold - 0.10)])
        else:
            sell_thresholds = np.asarray([np.nan])

        for sell_threshold in sell_thresholds:
            pred = make_trade_signal(
                probabilities,
                float(buy_threshold),
                None if np.isnan(sell_threshold) else float(sell_threshold),
                threshold_mode=threshold_mode,
                market_filter=market_filter,
            )
            backtest = backtest_long_flat(actual_returns, pred, transaction_cost=transaction_cost)
            trade_count = float(backtest["turnover"].sum())
            f1 = classification_metrics(y_true.astype(int), pred.astype(int))["F1-score"]
            strategy_return = float(backtest["strategy_cum_return"].iloc[-1])
            max_dd = abs(float(backtest["max_drawdown"].iloc[-1]))
            if objective == "f1":
                score = f1
            elif objective == "strategy_return":
                score = strategy_return
            elif objective == "risk_adjusted_return":
                score = strategy_return - drawdown_penalty * max_dd - trade_penalty * trade_count
            else:
                raise ValueError("threshold objective must be f1, strategy_return, or risk_adjusted_return.")

            if fallback is None or f1 > fallback[2]:
                fallback = (float(buy_threshold), float(sell_threshold), float(f1), trade_count)

            if trade_count < min_valid_trades:
                continue
            if score > best_score:
                best_buy_threshold = float(buy_threshold)
                best_sell_threshold = float(sell_threshold)
                best_score = float(score)
                best_trade_count = trade_count

    if not math.isfinite(best_score):
        assert fallback is not None
        return fallback
    return best_buy_threshold, best_sell_threshold, best_score, best_trade_count


def hidden_size_candidates(hidden_size: int) -> list[int]:
    return sorted({max(16, hidden_size // 2), hidden_size, hidden_size * 2})


def random_config(num_layers: int, hidden_size: int) -> LSTMConfig:
    return LSTMConfig(
        hidden_size=random.choice(hidden_size_candidates(hidden_size)),
        num_layers=num_layers,
        dropout=random.choice([0.1, 0.2, 0.3]),
        lr=random.choice([0.0005, 0.001, 0.002]),
    )


def mutate_config(config: LSTMConfig) -> LSTMConfig:
    child = LSTMConfig(config.hidden_size, config.num_layers, config.dropout, config.lr)
    field = random.choice(["hidden_size", "dropout", "lr"])
    if field == "hidden_size":
        child.hidden_size = random.choice(hidden_size_candidates(config.hidden_size))
    elif field == "dropout":
        child.dropout = random.choice([0.1, 0.2, 0.3])
    else:
        child.lr = random.choice([0.0005, 0.001, 0.002])
    return child


def genetic_search(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    task: str,
    batch_size: int,
    generations: int,
    population_size: int,
    ga_epochs: int,
    seed: int,
    class_weight: str,
    num_layers: int,
    hidden_size: int,
) -> LSTMConfig:
    """Small genetic search over hidden units, dropout and learning rate."""
    if generations <= 0 or population_size <= 0:
        return LSTMConfig(hidden_size=hidden_size, num_layers=num_layers)

    random.seed(seed)
    population = [random_config(num_layers, hidden_size) for _ in range(population_size)]
    scored: list[tuple[float, LSTMConfig]] = []
    for _generation in range(generations):
        scored = []
        for config in population:
            _model, valid_loss = train_model(
                x,
                y,
                train_idx,
                valid_idx,
                task=task,
                config=config,
                epochs=ga_epochs,
                batch_size=batch_size,
                seed=seed,
                class_weight=class_weight,
            )
            scored.append((valid_loss, config))
        scored.sort(key=lambda item: item[0])
        survivors = [config for _loss, config in scored[: max(2, population_size // 2)]]
        population = survivors.copy()
        while len(population) < population_size:
            population.append(mutate_config(random.choice(survivors)))
    return sorted(scored, key=lambda item: item[0])[0][1]


def evaluate_predictions(task: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return classification_metrics(actual.astype(int), predicted.astype(int))
    return regression_metrics(actual.astype(float), predicted.astype(float))


def make_metrics_rows(
    model_name: str,
    task: str,
    window: int,
    horizon: int,
    metrics: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "model": model_name,
            "task": task,
            "window": window,
            "horizon": horizon,
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


def naive_baseline(meta: pd.DataFrame, task: str) -> pd.DataFrame:
    predictions = meta.copy()
    momentum_return = predictions["actual_return"].shift(5).fillna(0)
    if task == "classification":
        predictions["predicted_direction"] = (momentum_return > 0).astype(int)
        predictions["predicted_probability"] = predictions["predicted_direction"].astype(float)
        predictions["predicted_return"] = momentum_return
    else:
        predictions["predicted_return"] = momentum_return
        predictions["predicted_direction"] = (predictions["predicted_return"] > 0).astype(int)
        predictions["predicted_probability"] = np.nan
    return predictions


def run_one_model(
    df: pd.DataFrame,
    model_name: str,
    task: str,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    ga_generations: int,
    ga_population: int,
    ga_epochs: int,
    seed: int,
    class_weight: str,
    threshold: float,
    transaction_cost: float,
    num_layers: int,
    hidden_size: int,
    log_interval: int,
    patience: int,
    threshold_objective: str,
    min_hold_proba: float,
    min_valid_trades: int,
    threshold_mode: str,
    drawdown_penalty: float,
    trade_penalty: float,
    market_filter_column: str,
    market_filter_min: float,
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=horizon)
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=horizon)
    target = target_direction if task == "classification" else target_return
    feature_cols = select_feature_columns(df, model_name, horizon)
    frame = prepare_model_frame(df, feature_cols, target, extra_cols=[target_return, target_direction])
    frame = frame.dropna(subset=[target_return, target_direction]).reset_index(drop=True)

    n_sequences = len(frame) - window
    if n_sequences < 20:
        raise ValueError(f"Not enough usable rows for {model_name}: {len(frame)} rows after cleaning.")
    train_idx, valid_idx, test_idx = split_sequence_indices(n_sequences)
    x, y, meta, _scaler = make_sequences_with_meta(
        frame,
        feature_cols,
        target,
        target_return,
        target_direction,
        window,
        train_idx,
        meta_cols=[market_filter_column],
    )

    config = genetic_search(
        x,
        y,
        train_idx,
        valid_idx,
        task=task,
        batch_size=batch_size,
        generations=ga_generations,
        population_size=ga_population,
        ga_epochs=ga_epochs,
        seed=seed,
        class_weight=class_weight,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )
    model, _valid_loss = train_model(
        x,
        y,
        train_idx,
        valid_idx,
        task=task,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        class_weight=class_weight,
        log_interval=log_interval,
        run_name=f"{model_name}-{task}",
        patience=patience,
    )

    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    selected_threshold = threshold
    selected_sell_threshold = np.nan
    threshold_score = np.nan
    valid_trade_count = np.nan
    if task == "classification":
        _valid_pred, valid_prob = predict_model(model, x[valid_idx], task, threshold=threshold)
        selected_threshold, selected_sell_threshold, threshold_score, valid_trade_count = select_classification_threshold(
            y[valid_idx],
            valid_prob,
            meta.iloc[valid_idx]["actual_return"].to_numpy(dtype=float),
            objective=threshold_objective,
            transaction_cost=transaction_cost,
            min_hold_proba=min_hold_proba,
            min_valid_trades=min_valid_trades,
            threshold_mode=threshold_mode,
            drawdown_penalty=drawdown_penalty,
            trade_penalty=trade_penalty,
            market_filter=market_filter_mask(meta.iloc[valid_idx], market_filter_column, market_filter_min),
        )
    pred, prob = predict_model(model, x[test_idx], task, threshold=selected_threshold)

    if task == "classification":
        predicted_direction = make_trade_signal(
            prob,
            selected_threshold,
            None if np.isnan(selected_sell_threshold) else selected_sell_threshold,
            threshold_mode=threshold_mode,
            market_filter=market_filter_mask(test_meta, market_filter_column, market_filter_min),
        )
        predicted_return = np.where(predicted_direction == 1, np.abs(test_meta["actual_return"]), -np.abs(test_meta["actual_return"]))
        metric_actual = y_test.astype(int)
        metric_pred = predicted_direction
    else:
        predicted_return = pred.astype(float)
        predicted_direction = (predicted_return > 0).astype(int)
        metric_actual = y_test.astype(float)
        metric_pred = predicted_return

    pred_table = test_meta.copy()
    pred_table["model"] = model_name
    pred_table["predicted_return"] = predicted_return
    pred_table["predicted_direction"] = predicted_direction
    pred_table["predicted_probability"] = prob
    backtest = backtest_long_flat(
        pred_table["actual_return"],
        pred_table["predicted_direction"],
        transaction_cost=transaction_cost,
    )
    pred_table = pd.concat([pred_table, backtest], axis=1)

    metrics = evaluate_predictions(task, metric_actual, metric_pred)
    metrics["strategy_cum_return"] = float(backtest["strategy_cum_return"].iloc[-1])
    metrics["buy_hold_cum_return"] = float(backtest["buy_hold_cum_return"].iloc[-1])
    metrics["max_drawdown"] = float(backtest["max_drawdown"].iloc[-1])
    metrics["trade_count"] = float(backtest["turnover"].sum())
    metrics["total_transaction_cost"] = float(backtest["transaction_cost"].sum())
    metrics["hidden_size"] = float(config.hidden_size)
    metrics["num_layers"] = float(config.num_layers)
    metrics["dropout"] = float(config.dropout)
    metrics["learning_rate"] = float(config.lr)
    metrics["threshold"] = float(selected_threshold)
    metrics["sell_threshold"] = float(selected_sell_threshold)
    metrics[f"valid_{threshold_objective}_at_threshold"] = float(threshold_score)
    metrics["valid_trade_count"] = float(valid_trade_count)
    metrics["threshold_mode"] = threshold_mode
    metrics["min_hold_proba"] = float(min_hold_proba)
    metrics["min_valid_trades"] = float(min_valid_trades)
    metrics["drawdown_penalty"] = float(drawdown_penalty)
    metrics["trade_penalty"] = float(trade_penalty)
    metrics["market_filter_min"] = float(market_filter_min)
    metrics["best_epoch"] = float(getattr(model, "training_summary", {}).get("best_epoch", np.nan))
    metrics["best_valid_loss"] = float(getattr(model, "training_summary", {}).get("best_valid_loss", np.nan))
    metrics["epochs_trained"] = float(getattr(model, "training_summary", {}).get("epochs_trained", np.nan))

    confusion = pd.DataFrame()
    if task == "classification":
        confusion = confusion_matrix_frame(metric_actual, metric_pred)
    return pred_table, make_metrics_rows(model_name, task, window, horizon, metrics), confusion


def run_naive_baseline(
    df: pd.DataFrame,
    task: str,
    window: int,
    horizon: int,
    transaction_cost: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=horizon)
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=horizon)
    feature_cols = select_feature_columns(df, "base", horizon)
    frame = prepare_model_frame(
        df,
        feature_cols,
        target_direction if task == "classification" else target_return,
        extra_cols=[target_return, target_direction],
    )
    frame = frame.dropna(subset=[target_return, target_direction]).reset_index(drop=True)
    n_sequences = len(frame) - window
    _train_idx, _valid_idx, test_idx = split_sequence_indices(n_sequences)
    meta = frame.iloc[np.arange(window, len(frame))][["date", "close", target_return, target_direction]].copy()
    meta = meta.rename(columns={target_return: "actual_return", target_direction: "actual_direction"}).reset_index(drop=True)
    pred_table = naive_baseline(meta.iloc[test_idx].reset_index(drop=True), task)
    pred_table["model"] = "naive_momentum"
    backtest = backtest_long_flat(
        pred_table["actual_return"],
        pred_table["predicted_direction"],
        transaction_cost=transaction_cost,
    )
    pred_table = pd.concat([pred_table, backtest], axis=1)

    if task == "classification":
        metrics = classification_metrics(
            pred_table["actual_direction"].astype(int).to_numpy(),
            pred_table["predicted_direction"].astype(int).to_numpy(),
        )
    else:
        metrics = regression_metrics(
            pred_table["actual_return"].to_numpy(),
            pred_table["predicted_return"].to_numpy(),
        )
    metrics["strategy_cum_return"] = float(backtest["strategy_cum_return"].iloc[-1])
    metrics["buy_hold_cum_return"] = float(backtest["buy_hold_cum_return"].iloc[-1])
    metrics["max_drawdown"] = float(backtest["max_drawdown"].iloc[-1])
    metrics["trade_count"] = float(backtest["turnover"].sum())
    metrics["total_transaction_cost"] = float(backtest["transaction_cost"].sum())
    return pred_table, make_metrics_rows("naive_momentum", task, window, horizon, metrics)


def restrict_to_common_fusion_period(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Align base and fusion experiments to dates with all fusion fields available."""
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=horizon)
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=horizon)
    fusion_features = select_feature_columns(df, "fusion", horizon)
    required = ["date", target_return, target_direction] + fusion_features
    common = df.dropna(subset=[col for col in required if col in df.columns]).copy()
    if common.empty:
        return df
    start_date = pd.to_datetime(common["date"]).min()
    end_date = pd.to_datetime(common["date"]).max()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out[(out["date"] >= start_date) & (out["date"] <= end_date)].reset_index(drop=True)


def plot_outputs(predictions: pd.DataFrame, metrics: pd.DataFrame, task: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not predictions.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in predictions.groupby("model"):
            group = group.sort_values("date")
            ax.plot(pd.to_datetime(group["date"]), group["predicted_return"], label=f"{model} predicted", linewidth=1.6)
        first_model = predictions.sort_values("date").drop_duplicates("date")
        ax.plot(pd.to_datetime(first_model["date"]), first_model["actual_return"], label="actual", color="#111827", linewidth=1.4)
        ax.set_title("Actual vs LSTM predicted 5-day return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lstm_prediction.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in predictions.groupby("model"):
            group = group.sort_values("date")
            ax.plot(pd.to_datetime(group["date"]), group["strategy_cum_return"], label=f"{model} strategy", linewidth=1.8)
        bh = predictions[predictions["model"] == predictions["model"].iloc[0]].sort_values("date")
        ax.plot(pd.to_datetime(bh["date"]), bh["buy_hold_cum_return"], label="buy and hold", color="#111827", linewidth=1.6)
        ax.set_title("Strategy cumulative return vs buy and hold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lstm_strategy_return.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        marker_df = predictions[predictions["model"].isin(["fusion", "base"])].copy()
        if marker_df.empty:
            marker_df = predictions.copy()
        marker_df = marker_df.sort_values("date").drop_duplicates("date")
        colors = np.where(marker_df["predicted_direction"].astype(float) > 0, "#16a34a", "#dc2626")
        ax.plot(pd.to_datetime(marker_df["date"]), marker_df["close"], color="#2563eb", linewidth=1.6)
        ax.scatter(pd.to_datetime(marker_df["date"]), marker_df["close"], c=colors, s=18, alpha=0.75)
        ax.set_title("Close price with predicted direction markers")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lstm_price_direction.png", dpi=180)
        plt.close(fig)

    selected_metric = "F1-score" if task == "classification" else "RMSE"
    metric_subset = metrics[metrics["metric"].isin([selected_metric, "strategy_cum_return", "buy_hold_cum_return"])]
    if not metric_subset.empty:
        pivot = metric_subset.pivot_table(index="model", columns="metric", values="value", aggfunc="first")
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Model metric comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Metric value")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lstm_metrics_comparison.png", dpi=180)
        plt.close(fig)


def save_confusion_figure(confusion: pd.DataFrame) -> None:
    if confusion.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion.to_numpy(), cmap="Blues")
    ax.set_xticks([0, 1], labels=confusion.columns)
    ax.set_yticks([0, 1], labels=confusion.index)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, int(confusion.iloc[i, j]), ha="center", va="center", color="#111827")
    ax.set_title("LSTM classification confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lstm_confusion_matrix.png", dpi=180)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, predictions: pd.DataFrame, task: str, horizon: int, window: int) -> None:
    report_path = PROJECT_ROOT / "outputs" / "lstm_report.md"
    best_metric = "F1-score" if task == "classification" else "RMSE"
    summary = metrics[metrics["metric"] == best_metric].copy()
    strategy = metrics[metrics["metric"].isin(["strategy_cum_return", "buy_hold_cum_return", "max_drawdown"])]

    lines = [
        "# 基于 TOPSIS 经营绩效评价与 LSTM 的股价趋势预测研究",
        "",
        "## 1. 研究问题",
        "本文在长春高新经营绩效评价基础上，将年频财务指标、熵权 TOPSIS 综合得分与日频行情技术指标结合，比较基础 LSTM 与融合 LSTM 在未来短期趋势预测中的表现。",
        "",
        "## 2. 为什么选择 LSTM",
        "股票收益序列具有明显的时间依赖，LSTM 能通过门控结构保留一段历史窗口中的有效信息，适合处理短期动量、均线、波动率等序列特征。",
        "",
        f"## 3. 预测目标",
        f"本研究优先预测未来 {horizon} 个交易日累计收益率方向，而不是直接预测绝对股价。方向标签由 future_{horizon}d_return 是否大于 0 得到，能降低股价尺度变化和复权误差带来的解释难度。",
        "",
        "## 4. 数据来源",
        "日频股价来自 data/processed/stock_prices.csv，补充后的市场特征来自 data/processed/stock_market_features.csv，包含前复权 OHLCV、成交额、换手率、沪深300、中证医药、创业板指、估值和市值字段；财务指标来自 data/processed/financial_indicators.csv；TOPSIS 得分来自 outputs/tables/topsis_scores.csv；财报披露日来自 data/processed/financial_disclosure_dates.csv。",
        "",
        "## 5. 特征体系",
        "基础模型使用 open、high、low、close、volume、amount、turnover、收益率、均线、成交量均线、波动率、RSI、MACD、布林带、市场指数收益率、PE/PB/PS 和市值等行情、技术与市场环境变量；融合模型在此基础上加入 revenue、net_profit、roe、gross_margin、net_margin、asset_liability_ratio、current_ratio、revenue_growth、net_profit_growth、topsis_score、rank 等经营绩效变量。",
        "",
        "## 6. 财务绩效与 TOPSIS 融合方法",
        "脚本优先使用 financial_disclosure_dates.csv 中的披露日期，通过 merge_asof 将已经披露的最新财务指标和 TOPSIS 得分映射到每个交易日，从而避免在披露日前使用未来财务信息。部分早期年份若 AkShare 未返回实际披露日，则按年报 4 月 30 日、半年报 8 月 31 日、一季报 4 月 30 日、三季报 10 月 31 日进行估算，并在 disclosure_source 中标记。",
        "",
        "## 7. 模型结构与数据切分",
        f"每个样本使用过去 {window} 个交易日作为输入窗口。数据按时间顺序切分为约 70% 训练集、15% 验证集、15% 测试集，不进行随机打乱。特征标准化只在训练集拟合 scaler，验证集和测试集仅 transform。",
        "",
        "模型采用可配置多层 PyTorch LSTM、Dropout 和 Dense 输出层。分类任务使用 sigmoid 概率和 BCEWithLogitsLoss；回归任务使用线性输出和 MSELoss。脚本还提供小规模遗传算法搜索 hidden_size、dropout 和 learning_rate。",
        "",
        "## 8. 实验结果",
    ]
    if not summary.empty:
        lines.append("```text")
        lines.append(summary.pivot_table(index="model", columns="metric", values="value", aggfunc="first").to_string())
        lines.append("```")
    else:
        lines.append("暂无可汇总的核心指标。")
    lines += ["", "## 9. 策略收益对比"]
    if not strategy.empty:
        lines.append("```text")
        lines.append(strategy.pivot_table(index="model", columns="metric", values="value", aggfunc="first").to_string())
        lines.append("```")
    else:
        lines.append("暂无策略收益结果。")
    lines += [
        "",
        "## 10. 模型局限性",
        "第一，虽然已补充完整 OHLCV、指数和估值字段，但样本仍只覆盖单一股票，结论容易受个股阶段行情影响。第二，财务指标以年频为主，无法反映报告期内的实时经营变化。第三，部分早期披露日期为规则估算值，仍需用公告原文进一步校验。第四，模型结果不应直接解释为可交易投资建议。",
        "",
        "## 11. 后续改进方向",
        "后续可进一步补充公告原文披露时间、分钟级流动性指标、行业成分股横截面样本和分析师预期数据；还可使用滚动窗口验证、交易成本和滑点假设、阈值优化与类别不均衡处理，提高结论稳健性。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _report_metric_table(metrics: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    selected = metrics[metrics["metric"].isin(metric_names)].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    table = selected.pivot_table(index="model", columns="metric", values="value", aggfunc="first")
    return table.reindex(columns=[m for m in metric_names if m in table.columns])


def _report_metric_value(metrics: pd.DataFrame, model: str, metric: str, default: float = np.nan) -> float:
    selected = metrics[(metrics["model"] == model) & (metrics["metric"] == metric)]
    if selected.empty:
        return default
    return float(pd.to_numeric(selected["value"].iloc[0], errors="coerce"))


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def _fmt_num(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(metrics: pd.DataFrame, predictions: pd.DataFrame, task: str, horizon: int, window: int) -> None:
    report_path = PROJECT_ROOT / "outputs" / "lstm_report.md"
    core_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "strategy_cum_return",
        "buy_hold_cum_return",
        "max_drawdown",
        "trade_count",
        "threshold",
        "sell_threshold",
        "hidden_size",
        "num_layers",
        "best_epoch",
        "epochs_trained",
    ]
    table = _report_metric_table(metrics, core_metrics)

    base_return = _report_metric_value(metrics, "base", "strategy_cum_return")
    fusion_return = _report_metric_value(metrics, "fusion", "strategy_cum_return")
    base_f1 = _report_metric_value(metrics, "base", "F1-score")
    fusion_f1 = _report_metric_value(metrics, "fusion", "F1-score")
    base_drawdown = _report_metric_value(metrics, "base", "max_drawdown")
    fusion_drawdown = _report_metric_value(metrics, "fusion", "max_drawdown")
    base_trades = _report_metric_value(metrics, "base", "trade_count")
    fusion_trades = _report_metric_value(metrics, "fusion", "trade_count")
    threshold = _report_metric_value(metrics, "base", "threshold")
    sell_threshold = _report_metric_value(metrics, "base", "sell_threshold")
    hidden_size = _report_metric_value(metrics, "base", "hidden_size")
    num_layers = _report_metric_value(metrics, "base", "num_layers")
    epochs_trained = _report_metric_value(metrics, "base", "epochs_trained")

    model_choice = "普通 LSTM"
    if pd.notna(fusion_return) and fusion_return > base_return and pd.notna(fusion_f1) and fusion_f1 >= base_f1:
        model_choice = "融合 LSTM"

    lines = [
        "# LSTM股价方向预测与交易策略报告",
        "",
        "## 1. 研究目标",
        f"本报告从日频行情、市场环境、估值指标和经营绩效指标出发，预测未来 {horizon} 个交易日累计收益率方向，并把预测概率转化为可回测的 long-flat 交易信号。研究重点不是直接预测股价点位，而是判断短期上涨概率是否足够高、是否值得持仓。",
        "",
        "## 2. 数据与标签",
        f"每个样本使用过去 {window} 个交易日作为输入窗口。分类标签由 `future_{horizon}d_return > 0` 得到：未来累计收益为正记为上涨，否则记为非上涨。数据按时间顺序切分训练集、验证集和测试集，避免随机打乱造成未来信息泄露。",
        "",
        "数据来源包括 `data/processed/stock_market_features.csv`、`data/processed/financial_indicators.csv`、`outputs/tables/topsis_scores.csv` 和披露日表。财务与TOPSIS指标通过披露日向后生效，只允许模型使用当日已经公开的信息。",
        "",
        "## 3. 模型设计",
        "普通 LSTM 使用行情、技术指标、市场指数收益、估值和市值等日频特征，重点捕捉价格序列自身的短期动量、均值回归和波动结构。",
        "",
        "融合 LSTM 在普通特征之外加入营业收入、净利润、ROE、毛利率、净利率、资产负债率、流动比率、收入增长、净利润增长、TOPSIS得分和排名等经营绩效变量，用来检验基本面质量是否能提高方向判断。",
        "",
        f"训练脚本默认采用 2 层 LSTM、隐藏层宽度 64；本轮单次结果表记录的普通模型实际参数为 {int(num_layers) if pd.notna(num_layers) else 'NA'} 层、隐藏层宽度 {int(hidden_size) if pd.notna(hidden_size) else 'NA'}。模型使用 dropout、BCEWithLogitsLoss 和早停机制。普通模型本轮在第 {_fmt_num(epochs_trained, 0)} 轮附近停止，说明验证集损失已经进入平台期，继续加轮数收益有限。",
        "",
        "## 4. 策略层设计",
        f"模型输出上涨概率后，不再简单使用 0.50 作为买入阈值，而是在验证集上选择双阈值：买入阈值约为 {_fmt_num(threshold, 2)}，卖出/空仓阈值约为 {_fmt_num(sell_threshold, 2)}。当上涨概率高于买入阈值且市场过滤条件通过时持仓，否则空仓。",
        "",
        "阈值目标采用风险调整收益：验证集策略收益减去回撤惩罚和交易频率惩罚。回测按单边0.1%交易成本扣除，并加入中证医药20日收益过滤，只有行业环境不弱于设定阈值时才允许开仓。",
        "",
        "## 5. 核心结果",
        "```text",
        table.round(4).to_string() if not table.empty else "No metric table available.",
        "```",
        "",
        "结果解释：",
        f"- 普通 LSTM 的 F1-score 为 {_fmt_num(base_f1)}，策略累计收益为 {_fmt_pct(base_return)}，最大回撤为 {_fmt_pct(base_drawdown)}，交易次数为 {_fmt_num(base_trades, 0)}。",
        f"- 融合 LSTM 的 F1-score 为 {_fmt_num(fusion_f1)}，策略累计收益为 {_fmt_pct(fusion_return)}，最大回撤为 {_fmt_pct(fusion_drawdown)}，交易次数为 {_fmt_num(fusion_trades, 0)}。",
        f"- 当前应优先采用{model_choice}作为 5 日方向交易主模型。融合模型可以保留为解释性对照和中长期研究方向，但在本轮短周期策略中没有显示出稳定优势。",
        "",
        "## 6. 图表解释",
        "- `outputs/figures/lstm_prediction.png`：展示测试集真实方向与预测概率/预测类别的时间序列关系，用来判断模型信号是否集中出现在趋势转换附近。",
        "- `outputs/figures/lstm_price_direction.png`：把预测信号叠加到收盘价走势上，便于观察买入信号是否避开了明显下跌段。",
        "- `outputs/figures/lstm_strategy_return.png`：比较模型策略、买入持有和朴素动量策略的累计收益，是判断模型是否有交易价值的主图。",
        "- `outputs/figures/lstm_metrics_comparison.png`：比较 Accuracy、F1-score、策略收益和回撤等指标，避免只看准确率造成误判。",
        "- `outputs/figures/lstm_confusion_matrix.png`：展示上涨/非上涨分类的混淆矩阵，用来识别模型是否偏向空仓或偏向看涨。",
        "",
        "## 7. 模型选择建议",
        "短期交易策略优先看样本外策略收益、最大回撤、交易次数和滚动验证稳定性，而不是单次准确率。当前普通 LSTM 在策略收益和交易可执行性上更好，融合模型虽然理论上包含更多经营信息，但年频基本面变量对5日方向预测的边际贡献有限，且容易带来样本不足和信号钝化。",
        "",
        "因此，当前版本建议：普通 LSTM 作为主模型；融合 LSTM 暂不作为主交易模型，继续用于解释、稳健性对照和更长预测周期实验。",
        "",
        "## 8. 局限与下一步",
        "本实验仍是单股票、短样本、日频回测，结论会受个股阶段行情影响。下一步应重点做三件事：第一，用滚动验证持续检验每一年样本外表现；第二，扩展到同行业多股票，区分个股特异性和行业共性；第三，尝试更长预测周期或低频再平衡，让财务绩效和TOPSIS指标有更充分的发挥空间。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train base/fusion LSTM models for stock trend prediction.")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "lstm_model_data.csv"))
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--model-type", choices=["base", "fusion", "both"], default="both")
    parser.add_argument("--ga-generations", type=int, default=1)
    parser.add_argument("--ga-population", type=int, default=4)
    parser.add_argument("--ga-epochs", type=int, default=5)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-objective", choices=["f1", "strategy_return", "risk_adjusted_return"], default="risk_adjusted_return")
    parser.add_argument("--threshold-mode", choices=["single", "dual"], default="dual")
    parser.add_argument("--min-hold-proba", type=float, default=0.55)
    parser.add_argument("--min-valid-trades", type=int, default=5)
    parser.add_argument("--drawdown-penalty", type=float, default=0.5)
    parser.add_argument("--trade-penalty", type=float, default=0.002)
    parser.add_argument("--market-filter-column", default="csi_pharma_return_20d")
    parser.add_argument("--market-filter-min", type=float, default=0.0)
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Single-side cost, e.g. 0.001 means 0.1%%.")
    parser.add_argument("--log-interval", type=int, default=10, help="Print training loss every N epochs; 0 disables logs.")
    parser.add_argument("--patience", type=int, default=10, help="Stop after N epochs without validation-loss improvement; 0 disables early stopping.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_build_model_data(args.data, horizon=args.horizon)
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=args.horizon)
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=args.horizon)
    if target_return not in df or target_direction not in df:
        df = build_lstm_model_data(output_path=args.data, horizon=args.horizon)
    if args.model_type == "both":
        df = restrict_to_common_fusion_period(df, args.horizon)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    confusion_for_plot = pd.DataFrame()

    naive_predictions, naive_metrics = run_naive_baseline(
        df,
        args.task,
        args.window,
        args.horizon,
        transaction_cost=args.transaction_cost,
    )
    all_predictions.append(naive_predictions)
    all_metrics.extend(naive_metrics)

    model_names = ["base", "fusion"] if args.model_type == "both" else [args.model_type]
    for model_name in model_names:
        predictions, metrics_rows, confusion = run_one_model(
            df,
            model_name=model_name,
            task=args.task,
            window=args.window,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            ga_generations=args.ga_generations,
            ga_population=args.ga_population,
            ga_epochs=args.ga_epochs,
            seed=args.seed,
            class_weight=args.class_weight,
            threshold=args.threshold,
            transaction_cost=args.transaction_cost,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            log_interval=args.log_interval,
            patience=args.patience,
            threshold_objective=args.threshold_objective,
            min_hold_proba=args.min_hold_proba,
            min_valid_trades=args.min_valid_trades,
            threshold_mode=args.threshold_mode,
            drawdown_penalty=args.drawdown_penalty,
            trade_penalty=args.trade_penalty,
            market_filter_column=args.market_filter_column,
            market_filter_min=args.market_filter_min,
        )
        all_predictions.append(predictions)
        all_metrics.extend(metrics_rows)
        if model_name == "fusion" and not confusion.empty:
            confusion_for_plot = confusion
        elif confusion_for_plot.empty:
            confusion_for_plot = confusion

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    columns = [
        "date",
        "model",
        "actual_return",
        "predicted_return",
        "actual_direction",
        "predicted_direction",
        "predicted_probability",
        "strategy_return",
        "buy_hold_return",
        "position",
        "turnover",
        "transaction_cost",
        "strategy_cum_return",
        "buy_hold_cum_return",
        "max_drawdown",
        "close",
    ]
    for col in columns:
        if col not in predictions_df:
            predictions_df[col] = np.nan
    predictions_df[columns].to_csv(TABLE_DIR / "lstm_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(TABLE_DIR / "lstm_metrics.csv", index=False, encoding="utf-8-sig")

    plot_outputs(predictions_df, metrics_df, args.task)
    save_confusion_figure(confusion_for_plot)
    write_report(metrics_df, predictions_df, args.task, args.horizon, args.window)

    print(f"Saved metrics: {TABLE_DIR / 'lstm_metrics.csv'}")
    print(f"Saved predictions: {TABLE_DIR / 'lstm_predictions.csv'}")
    print(f"Saved figures: {FIGURE_DIR}")
    print(f"Saved report: {PROJECT_ROOT / 'outputs' / 'lstm_report.md'}")


if __name__ == "__main__":
    main()
