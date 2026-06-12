"""Dual-branch LSTM + MLP model for panel stock-direction prediction."""

from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from model_evaluation import classification_metrics, max_drawdown
from panel_modeling import (
    FIGURE_DIR,
    FINANCIAL_FEATURES,
    PROCESSED_DIR,
    TABLE_DIR,
    TARGET_DIRECTION,
    TARGET_RETURN,
    VALUATION_FEATURES,
    feature_sets,
    load_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class DualBranchConfig:
    lstm_hidden: int = 32
    static_hidden: int = 32
    dropout: float = 0.2
    lr: float = 0.001


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install it with: pip install torch") from exc
    return torch, nn, DataLoader, TensorDataset


def _existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def branch_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return sequence and static feature columns."""
    sets = feature_sets(df)
    sequence_cols = sets["valuation"]
    static_candidates = FINANCIAL_FEATURES + [
        "revenue_growth",
        "net_profit_growth",
        "topsis_score",
        "rank",
    ]
    static_cols = _existing(df, static_candidates)

    # Keep valuation in the sequence branch because valuation changes daily;
    # financial/TOPSIS variables are slow-moving and enter the static branch.
    sequence_cols = [col for col in sequence_cols if col not in static_cols]
    return list(dict.fromkeys(sequence_cols)), list(dict.fromkeys(static_cols))


def split_by_date(df: pd.DataFrame, valid_start: str, test_start: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    valid_date = pd.Timestamp(valid_start)
    test_date = pd.Timestamp(test_start)
    if df["date"].min() >= valid_date or df["date"].max() < test_date:
        raise ValueError("Invalid date split for the available data.")
    return valid_date, test_date


def fit_preprocessors(
    df: pd.DataFrame,
    sequence_cols: list[str],
    static_cols: list[str],
    train_mask: pd.Series,
) -> tuple[SimpleImputer, StandardScaler, SimpleImputer, StandardScaler, list[str]]:
    seq_imputer = SimpleImputer(strategy="median")
    seq_scaler = StandardScaler()
    seq_train = seq_imputer.fit_transform(df.loc[train_mask, sequence_cols])
    seq_scaler.fit(seq_train)

    identity = pd.get_dummies(df[["symbol", "sector"]].fillna("unknown"), columns=["symbol", "sector"], dtype=float)
    static_design = pd.concat([df[static_cols].apply(pd.to_numeric, errors="coerce"), identity], axis=1)
    static_feature_cols = list(static_design.columns)
    static_imputer = SimpleImputer(strategy="median")
    static_scaler = StandardScaler()
    static_train = static_imputer.fit_transform(static_design.loc[train_mask, static_feature_cols])
    static_scaler.fit(static_train)
    return seq_imputer, seq_scaler, static_imputer, static_scaler, static_feature_cols


def transform_features(
    df: pd.DataFrame,
    sequence_cols: list[str],
    static_cols: list[str],
    seq_imputer: SimpleImputer,
    seq_scaler: StandardScaler,
    static_imputer: SimpleImputer,
    static_scaler: StandardScaler,
    static_feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    seq_values = seq_scaler.transform(seq_imputer.transform(df[sequence_cols]))
    identity = pd.get_dummies(df[["symbol", "sector"]].fillna("unknown"), columns=["symbol", "sector"], dtype=float)
    static_design = pd.concat([df[static_cols].apply(pd.to_numeric, errors="coerce"), identity], axis=1)
    static_design = static_design.reindex(columns=static_feature_cols, fill_value=0)
    static_values = static_scaler.transform(static_imputer.transform(static_design))
    return seq_values.astype(np.float32), static_values.astype(np.float32)


def make_panel_sequences(
    df: pd.DataFrame,
    seq_values: np.ndarray,
    static_values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_seq, x_static, y, meta = [], [], [], []
    indexed = df.reset_index(drop=True)
    for _symbol, group in indexed.groupby("symbol", sort=False):
        positions = group.index.to_numpy()
        for offset in range(window, len(positions)):
            pos = positions[offset]
            history = positions[offset - window : offset]
            x_seq.append(seq_values[history])
            x_static.append(static_values[pos])
            y.append(indexed.loc[pos, TARGET_DIRECTION])
            meta.append(
                {
                    "date": indexed.loc[pos, "date"],
                    "symbol": indexed.loc[pos, "symbol"],
                    "name": indexed.loc[pos, "name"],
                    "sector": indexed.loc[pos, "sector"],
                    "close": indexed.loc[pos, "close"],
                    "actual_return": indexed.loc[pos, TARGET_RETURN],
                    "actual_direction": indexed.loc[pos, TARGET_DIRECTION],
                }
            )
    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(x_static, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DataFrame(meta),
    )


def sequence_indices_by_dates(meta: pd.DataFrame, valid_start: str, test_start: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_date = pd.Timestamp(valid_start)
    test_date = pd.Timestamp(test_start)
    dates = pd.to_datetime(meta["date"], errors="coerce")
    train_idx = np.where(dates < valid_date)[0]
    valid_idx = np.where((dates >= valid_date) & (dates < test_date))[0]
    test_idx = np.where(dates >= test_date)[0]
    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Sequence split produced an empty subset.")
    return train_idx, valid_idx, test_idx


def build_model(seq_dim: int, static_dim: int, config: DualBranchConfig):
    torch, nn, _DataLoader, _TensorDataset = _require_torch()

    class DualBranchLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=config.lstm_hidden, batch_first=True)
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, config.static_hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(config.lstm_hidden + config.static_hidden, config.lstm_hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.lstm_hidden, 1),
            )

        def forward(self, seq_x, static_x):
            _, (hidden, _) = self.lstm(seq_x)
            seq_repr = hidden[-1]
            static_repr = self.static_net(static_x)
            return self.head(torch.cat([seq_repr, static_repr], dim=1)).squeeze(-1)

    return DualBranchLSTM()


def train_dual_branch(
    x_seq: np.ndarray,
    x_static: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: DualBranchConfig,
    epochs: int,
    batch_size: int,
    seed: int,
    patience: int,
) -> tuple[object, float, int]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(x_seq.shape[2], x_static.shape[1], config)
    positives = float((y[train_idx] == 1).sum())
    negatives = float((y[train_idx] == 0).sum())
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_ds = TensorDataset(
        torch.tensor(x_seq[train_idx]),
        torch.tensor(x_static[train_idx]),
        torch.tensor(y[train_idx]),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_valid = np.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(max(1, epochs)):
        model.train()
        for seq_batch, static_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(seq_batch, static_batch)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(torch.tensor(x_seq[valid_idx]), torch.tensor(x_static[valid_idx]))
            valid_loss = criterion(valid_logits, torch.tensor(y[valid_idx]).float()).item()
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_valid), int(best_epoch)


def predict_proba(model, x_seq: np.ndarray, x_static: np.ndarray) -> np.ndarray:
    torch, _nn, _DataLoader, _TensorDataset = _require_torch()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_seq), torch.tensor(x_static)).numpy()
    return 1 / (1 + np.exp(-logits))


def best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    objective: str,
    min_hold_proba: float,
    transaction_cost: float,
    min_valid_trades: int,
    meta: pd.DataFrame | None = None,
) -> tuple[float, float]:
    best_t, best_score = 0.5, -1.0
    start = max(0.20, min_hold_proba)
    for threshold in np.round(np.arange(start, 0.81, 0.02), 2):
        pred = (proba >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y_true.astype(int), pred, zero_division=0)
        elif objective == "strategy_return":
            if meta is None:
                raise ValueError("meta is required when threshold objective is strategy_return.")
            valid_pred = meta.reset_index(drop=True).copy()
            valid_pred["predicted_direction"] = pred
            valid_pred = add_strategy_returns(valid_pred, transaction_cost)
            strategy_metrics = aggregate_strategy(valid_pred)
            if strategy_metrics["trade_count"] < min_valid_trades:
                score = -np.inf
            else:
                score = strategy_metrics["strategy_cum_return"]
        else:
            raise ValueError("threshold objective must be f1 or strategy_return.")
        if score > best_score:
            best_t, best_score = float(threshold), float(score)
    return best_t, best_score


def add_strategy_returns(predictions: pd.DataFrame, transaction_cost: float) -> pd.DataFrame:
    parts = []
    for _symbol, group in predictions.sort_values("date").groupby("symbol", sort=False):
        group = group.copy()
        signal = group["predicted_direction"].astype(float).clip(lower=0, upper=1)
        turnover = signal.diff().abs().fillna(signal.abs())
        group["position"] = signal
        group["turnover"] = turnover
        group["transaction_cost"] = turnover * transaction_cost
        group["strategy_return"] = group["actual_return"] * signal - group["transaction_cost"]
        group["buy_hold_return"] = group["actual_return"]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def aggregate_strategy(predictions: pd.DataFrame) -> dict[str, float]:
    daily = predictions.groupby("date", as_index=False)[["strategy_return", "buy_hold_return"]].mean().sort_values("date")
    daily["strategy_cum_return"] = (1 + daily["strategy_return"]).cumprod() - 1
    daily["buy_hold_cum_return"] = (1 + daily["buy_hold_return"]).cumprod() - 1
    return {
        "strategy_cum_return": float(daily["strategy_cum_return"].iloc[-1]),
        "buy_hold_cum_return": float(daily["buy_hold_cum_return"].iloc[-1]),
        "max_drawdown": max_drawdown(daily["strategy_cum_return"]),
        "trade_count": float(predictions["turnover"].sum()),
        "total_transaction_cost": float(predictions["transaction_cost"].sum()),
    }


def config_candidates(search: bool) -> list[DualBranchConfig]:
    if not search:
        return [DualBranchConfig()]
    candidates = []
    for lstm_hidden, static_hidden, dropout, lr in itertools.product(
        [32, 64],
        [16, 32],
        [0.2, 0.3],
        [0.001, 0.0005],
    ):
        candidates.append(
            DualBranchConfig(
                lstm_hidden=lstm_hidden,
                static_hidden=static_hidden,
                dropout=dropout,
                lr=lr,
            )
        )
    return candidates


def select_best_model(
    x_seq: np.ndarray,
    x_static: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    meta: pd.DataFrame,
    epochs: int,
    batch_size: int,
    seed: int,
    patience: int,
    threshold_objective: str,
    min_hold_proba: float,
    transaction_cost: float,
    min_valid_trades: int,
    search: bool,
) -> tuple[object, DualBranchConfig, float, int, float, float]:
    best = None
    for i, config in enumerate(config_candidates(search)):
        model, valid_loss, best_epoch = train_dual_branch(
            x_seq,
            x_static,
            y,
            train_idx,
            valid_idx,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + i,
            patience=patience,
        )
        valid_proba = predict_proba(model, x_seq[valid_idx], x_static[valid_idx])
        threshold, threshold_score = best_threshold(
            y[valid_idx],
            valid_proba,
            objective=threshold_objective,
            min_hold_proba=min_hold_proba,
            transaction_cost=transaction_cost,
            min_valid_trades=min_valid_trades,
            meta=meta.iloc[valid_idx],
        )
        candidate = (threshold_score, -valid_loss, model, config, valid_loss, best_epoch, threshold, threshold_score)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        print(
            "[SEARCH]" if search else "[TRAIN]",
            f"lstm={config.lstm_hidden}",
            f"static={config.static_hidden}",
            f"dropout={config.dropout}",
            f"lr={config.lr}",
            f"threshold={threshold:.2f}",
            f"score={threshold_score:.4f}",
            f"loss={valid_loss:.4f}",
            f"epoch={best_epoch}",
        )
    assert best is not None
    _score, _neg_loss, model, config, valid_loss, best_epoch, threshold, threshold_score = best
    return model, config, float(valid_loss), int(best_epoch), float(threshold), float(threshold_score)


def run_dual_branch(
    data_path: str | Path,
    valid_start: str,
    test_start: str,
    window: int,
    epochs: int,
    batch_size: int,
    transaction_cost: float,
    seed: int,
    patience: int,
    threshold_objective: str,
    min_hold_proba: float,
    min_valid_trades: int,
    search: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_panel(data_path)
    df = df.dropna(subset=[TARGET_RETURN, TARGET_DIRECTION]).sort_values(["symbol", "date"]).reset_index(drop=True)
    split_by_date(df, valid_start, test_start)
    sequence_cols, static_cols = branch_columns(df)
    train_mask = df["date"] < pd.Timestamp(valid_start)
    preprocessors = fit_preprocessors(df, sequence_cols, static_cols, train_mask)
    seq_values, static_values = transform_features(df, sequence_cols, static_cols, *preprocessors)
    x_seq, x_static, y, meta = make_panel_sequences(df, seq_values, static_values, window)
    train_idx, valid_idx, test_idx = sequence_indices_by_dates(meta, valid_start, test_start)

    model, config, valid_loss, best_epoch, threshold, threshold_score = select_best_model(
        x_seq,
        x_static,
        y,
        train_idx,
        valid_idx,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        patience=patience,
        threshold_objective=threshold_objective,
        min_hold_proba=min_hold_proba,
        transaction_cost=transaction_cost,
        min_valid_trades=min_valid_trades,
        search=search,
        meta=meta,
    )
    test_proba = predict_proba(model, x_seq[test_idx], x_static[test_idx])
    test_pred = (test_proba >= threshold).astype(int)

    predictions = meta.iloc[test_idx].reset_index(drop=True).copy()
    predictions["model"] = "dual_branch_lstm"
    predictions["predicted_probability"] = test_proba
    predictions["threshold"] = threshold
    predictions["predicted_direction"] = test_pred
    predictions = add_strategy_returns(predictions, transaction_cost)

    metric_values = classification_metrics(y[test_idx].astype(int), test_pred.astype(int))
    metric_values.update(aggregate_strategy(predictions))
    metric_values[f"valid_{threshold_objective}_at_threshold"] = threshold_score
    metric_values["valid_loss"] = valid_loss
    metric_values["best_epoch"] = float(best_epoch)
    metric_values["threshold"] = threshold
    metric_values["threshold_objective"] = threshold_objective
    metric_values["min_hold_proba"] = min_hold_proba
    metric_values["min_valid_trades"] = float(min_valid_trades)
    metric_values["lstm_hidden"] = float(config.lstm_hidden)
    metric_values["static_hidden"] = float(config.static_hidden)
    metric_values["dropout"] = float(config.dropout)
    metric_values["learning_rate"] = float(config.lr)
    metric_values["sequence_feature_count"] = float(len(sequence_cols))
    metric_values["static_feature_count"] = float(len(preprocessors[-1]))
    metrics = pd.DataFrame(
        [
            {
                "model": "dual_branch_lstm",
                "metric": metric,
                "value": value,
                "valid_start": valid_start,
                "test_start": test_start,
                "window": window,
                "transaction_cost": transaction_cost,
            }
            for metric, value in metric_values.items()
        ]
    )
    return predictions, metrics


def plot_outputs(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    daily = predictions.groupby("date", as_index=False)[["strategy_return", "buy_hold_return"]].mean().sort_values("date")
    daily["strategy_cum_return"] = (1 + daily["strategy_return"]).cumprod() - 1
    daily["buy_hold_cum_return"] = (1 + daily["buy_hold_return"]).cumprod() - 1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily["date"], daily["strategy_cum_return"], label="dual_branch_strategy", linewidth=1.8)
    ax.plot(daily["date"], daily["buy_hold_cum_return"], label="equal_weight_buy_hold", linewidth=1.5, color="#111827")
    ax.set_title("Dual-branch LSTM strategy return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dual_branch_strategy_return.png", dpi=180)
    plt.close(fig)

    by_symbol = predictions.groupby("symbol", as_index=False).agg(
        name=("name", "first"),
        accuracy=("predicted_direction", lambda s: np.nan),
    )
    rows = []
    for symbol, group in predictions.groupby("symbol"):
        score = classification_metrics(group["actual_direction"].astype(int), group["predicted_direction"].astype(int))
        rows.append({"symbol": symbol, "name": group["name"].iloc[0], **score})
    symbol_metrics = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
    symbol_metrics.to_csv(TABLE_DIR / "dual_branch_symbol_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(symbol_metrics["name"], symbol_metrics["F1-score"], color="#2563eb")
    ax.set_title("Dual-branch LSTM F1-score by company")
    ax.set_xlabel("F1-score")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dual_branch_symbol_f1.png", dpi=180)
    plt.close(fig)


def write_report(metrics: pd.DataFrame) -> None:
    report_path = PROJECT_ROOT / "outputs" / "dual_branch_lstm_report.md"
    lines = [
        "# 双分支 LSTM + MLP 模型报告",
        "",
        "## 模型结构",
        "日频行情、技术指标、指数和估值变量进入 LSTM 分支；财务指标、TOPSIS 指标以及公司/行业哑变量进入 MLP 静态分支。两个分支的隐表示拼接后，通过全连接层输出未来 5 日上涨概率。",
        "",
        "训练阶段支持 early stopping；阈值可按验证集 F1-score 或验证集策略收益选择；也可设置最低持仓概率形成不交易区间，减少低置信度交易。",
        "",
        "## 结果",
        "```text",
        metrics.to_string(index=False),
        "```",
        "",
        "## 解释",
        "该结构避免将低频财务信息简单复制为日频序列，从建模结构上区分了快变量和慢变量。它适合作为单分支 LSTM 与 LightGBM 面板模型之间的深度学习增强版本。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-branch LSTM + MLP on panel data.")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "panel_model_data.csv"))
    parser.add_argument("--valid-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threshold-objective", choices=["f1", "strategy_return"], default="f1")
    parser.add_argument("--min-hold-proba", type=float, default=0.2)
    parser.add_argument("--min-valid-trades", type=int, default=20)
    parser.add_argument("--search", action="store_true", help="Run a small hyperparameter grid search.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    predictions, metrics = run_dual_branch(
        data_path=args.data,
        valid_start=args.valid_start,
        test_start=args.test_start,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        transaction_cost=args.transaction_cost,
        seed=args.seed,
        patience=args.patience,
        threshold_objective=args.threshold_objective,
        min_hold_proba=args.min_hold_proba,
        min_valid_trades=args.min_valid_trades,
        search=args.search,
    )
    predictions.to_csv(TABLE_DIR / "dual_branch_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "dual_branch_metrics.csv", index=False, encoding="utf-8-sig")
    plot_outputs(predictions, metrics)
    write_report(metrics)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
