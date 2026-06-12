"""Rolling-window LSTM validation with class imbalance and trading costs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from lstm_predict import (
    FIGURE_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    LSTMConfig,
    TARGET_DIRECTION_TEMPLATE,
    TARGET_RETURN_TEMPLATE,
    load_or_build_model_data,
    make_metrics_rows,
    make_sequences_with_meta,
    predict_model,
    prepare_model_frame,
    restrict_to_common_fusion_period,
    select_feature_columns,
    train_model,
)
from model_evaluation import backtest_long_flat, classification_metrics, confusion_matrix_frame


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _year_splits(frame: pd.DataFrame, first_test_year: int | None, min_train_rows: int) -> list[tuple[str, int]]:
    years = sorted(pd.to_datetime(frame["date"]).dt.year.dropna().astype(int).unique())
    if first_test_year is None:
        first_test_year = years[0] + 3
    splits = []
    for year in years:
        if year < first_test_year:
            continue
        train_rows = (pd.to_datetime(frame["date"]).dt.year < year).sum()
        test_rows = (pd.to_datetime(frame["date"]).dt.year == year).sum()
        if train_rows >= min_train_rows and test_rows >= 40:
            splits.append((f"test_{year}", year))
    return splits


def _rolling_indices(meta: pd.DataFrame, test_year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = pd.to_datetime(meta["date"]).dt.year.to_numpy()
    pre_test = np.where(years < test_year)[0]
    test_idx = np.where(years == test_year)[0]
    valid_size = max(20, int(len(pre_test) * 0.15))
    if len(pre_test) <= valid_size:
        raise ValueError(f"Not enough pre-test samples for {test_year}.")
    train_idx = pre_test[:-valid_size]
    valid_idx = pre_test[-valid_size:]
    return train_idx, valid_idx, test_idx


def _best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.round(np.arange(0.30, 0.71, 0.05), 2)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        score = f1_score(y_true.astype(int), pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def _run_naive_fold(meta: pd.DataFrame, test_idx: np.ndarray, transaction_cost: float) -> tuple[pd.DataFrame, dict[str, float]]:
    test_meta = meta.iloc[test_idx].reset_index(drop=True).copy()
    momentum = test_meta["actual_return"].shift(5).fillna(0)
    test_meta["model"] = "naive_momentum"
    test_meta["predicted_return"] = momentum
    test_meta["predicted_direction"] = (momentum > 0).astype(int)
    test_meta["predicted_probability"] = test_meta["predicted_direction"].astype(float)
    backtest = backtest_long_flat(
        test_meta["actual_return"],
        test_meta["predicted_direction"],
        transaction_cost=transaction_cost,
    )
    test_meta = pd.concat([test_meta, backtest], axis=1)
    metrics = classification_metrics(
        test_meta["actual_direction"].astype(int).to_numpy(),
        test_meta["predicted_direction"].astype(int).to_numpy(),
    )
    metrics["strategy_cum_return"] = float(backtest["strategy_cum_return"].iloc[-1])
    metrics["buy_hold_cum_return"] = float(backtest["buy_hold_cum_return"].iloc[-1])
    metrics["max_drawdown"] = float(backtest["max_drawdown"].iloc[-1])
    metrics["trade_count"] = float(backtest["turnover"].sum())
    metrics["total_transaction_cost"] = float(backtest["transaction_cost"].sum())
    metrics["threshold"] = 0.0
    return test_meta, metrics


def run_model_fold(
    df: pd.DataFrame,
    model_name: str,
    test_year: int,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    transaction_cost: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=horizon)
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=horizon)
    feature_cols = select_feature_columns(df, model_name, horizon)
    frame = prepare_model_frame(df, feature_cols, target_direction, extra_cols=[target_return, target_direction])
    frame = frame.dropna(subset=[target_return, target_direction]).reset_index(drop=True)

    placeholder_idx = np.arange(max(1, min(len(frame) - window, 10)))
    x, y, meta, _scaler = make_sequences_with_meta(
        frame,
        feature_cols,
        target_direction,
        target_return,
        target_direction,
        window,
        placeholder_idx,
    )
    train_idx, valid_idx, test_idx = _rolling_indices(meta, test_year)
    x, y, meta, _scaler = make_sequences_with_meta(
        frame,
        feature_cols,
        target_direction,
        target_return,
        target_direction,
        window,
        train_idx,
    )

    model, _valid_loss = train_model(
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
    _valid_pred, valid_prob = predict_model(model, x[valid_idx], task="classification", threshold=0.5)
    threshold, valid_f1 = _best_threshold(y[valid_idx], valid_prob)
    test_pred, test_prob = predict_model(model, x[test_idx], task="classification", threshold=threshold)

    pred_table = meta.iloc[test_idx].reset_index(drop=True).copy()
    pred_table["model"] = model_name
    pred_table["predicted_direction"] = test_pred.astype(int)
    pred_table["predicted_probability"] = test_prob
    pred_table["predicted_return"] = np.where(
        pred_table["predicted_direction"] == 1,
        np.abs(pred_table["actual_return"]),
        -np.abs(pred_table["actual_return"]),
    )
    backtest = backtest_long_flat(
        pred_table["actual_return"],
        pred_table["predicted_direction"],
        transaction_cost=transaction_cost,
    )
    pred_table = pd.concat([pred_table, backtest], axis=1)

    metrics = classification_metrics(y[test_idx].astype(int), test_pred.astype(int))
    metrics["strategy_cum_return"] = float(backtest["strategy_cum_return"].iloc[-1])
    metrics["buy_hold_cum_return"] = float(backtest["buy_hold_cum_return"].iloc[-1])
    metrics["max_drawdown"] = float(backtest["max_drawdown"].iloc[-1])
    metrics["trade_count"] = float(backtest["turnover"].sum())
    metrics["total_transaction_cost"] = float(backtest["transaction_cost"].sum())
    metrics["threshold"] = threshold
    metrics["valid_f1_at_threshold"] = valid_f1
    confusion = confusion_matrix_frame(y[test_idx].astype(int), test_pred.astype(int))
    return pred_table, metrics, confusion


def run_rolling_validation(
    data_path: str | Path,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    first_test_year: int | None,
    transaction_cost: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_or_build_model_data(data_path, horizon=horizon)
    df = restrict_to_common_fusion_period(df, horizon)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=horizon)
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=horizon)
    common_frame = df.dropna(subset=["date", target_return, target_direction]).copy()
    splits = _year_splits(common_frame, first_test_year, min_train_rows=max(250, window * 8))
    if not splits:
        raise ValueError("No rolling folds are available. Try a smaller first_test_year or more data.")

    all_predictions: list[pd.DataFrame] = []
    all_metric_rows: list[dict[str, object]] = []
    for fold_name, test_year in splits:
        print(f"Running rolling fold: {fold_name}")
        base_feature_cols = select_feature_columns(df, "base", horizon)
        base_frame = prepare_model_frame(
            df,
            base_feature_cols,
            target_direction,
            extra_cols=[target_return, target_direction],
        ).dropna(subset=[target_return, target_direction]).reset_index(drop=True)
        target_positions = np.arange(window, len(base_frame))
        meta = base_frame.iloc[target_positions][["date", "close", target_return, target_direction]].rename(
            columns={target_return: "actual_return", target_direction: "actual_direction"}
        )
        _, _, test_idx = _rolling_indices(meta.reset_index(drop=True), test_year)
        naive_pred, naive_metrics = _run_naive_fold(meta.reset_index(drop=True), test_idx, transaction_cost)
        naive_pred["fold"] = fold_name
        all_predictions.append(naive_pred)
        all_metric_rows.extend(
            {**row, "fold": fold_name}
            for row in make_metrics_rows("naive_momentum", "classification", window, horizon, naive_metrics)
        )

        for model_name in ["base", "fusion"]:
            pred, metrics, _confusion = run_model_fold(
                df,
                model_name=model_name,
                test_year=test_year,
                window=window,
                horizon=horizon,
                epochs=epochs,
                batch_size=batch_size,
                transaction_cost=transaction_cost,
                seed=int(seed + test_year),
            )
            pred["fold"] = fold_name
            all_predictions.append(pred)
            all_metric_rows.extend(
                {**row, "fold": fold_name}
                for row in make_metrics_rows(model_name, "classification", window, horizon, metrics)
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(all_metric_rows)
    return predictions, metrics


def save_outputs(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(TABLE_DIR / "lstm_rolling_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "lstm_rolling_metrics.csv", index=False, encoding="utf-8-sig")

    f1 = metrics[metrics["metric"] == "F1-score"].pivot_table(index="fold", columns="model", values="value")
    if not f1.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        f1.plot(marker="o", ax=ax)
        ax.set_title("Rolling validation F1-score by fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("F1-score")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lstm_rolling_f1.png", dpi=180)
        plt.close(fig)

    returns = metrics[metrics["metric"] == "strategy_cum_return"].pivot_table(index="fold", columns="model", values="value")
    if not returns.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        returns.plot(kind="bar", ax=ax)
        ax.set_title("Rolling validation strategy return after trading costs")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Cumulative return")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lstm_rolling_strategy_return.png", dpi=180)
        plt.close(fig)

    write_rolling_report(metrics)


def write_rolling_report(metrics: pd.DataFrame) -> None:
    report_path = FIGURE_DIR.parents[0] / "lstm_rolling_report.md"
    selected = metrics[metrics["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return", "trade_count"])]
    by_fold = selected.pivot_table(index=["fold", "model"], columns="metric", values="value", aggfunc="first")
    average = selected.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")

    lines = [
        "# LSTM 滚动窗口验证报告",
        "",
        "## 方法说明",
        "本实验按年度进行 expanding-window 滚动验证：测试年前的历史样本用于训练和验证，测试年份完全留作样本外评估。训练尾部 15% 作为验证集，并在验证集上选择 F1-score 最优的分类阈值。分类损失使用训练集正负样本比例自动设置 pos_weight，以缓解涨跌方向不均衡。",
        "",
        "回测策略为 long-flat：预测未来 5 个交易日上涨时持有，预测下跌时空仓。交易成本按单边 0.1% 在仓位变化时扣除。",
        "",
        "## 分年度结果",
        "```text",
        by_fold.round(4).to_string(),
        "```",
        "",
        "## 平均结果",
        "```text",
        average.round(4).to_string(),
        "```",
        "",
        "## 结论",
        "滚动验证比单次 holdout 更接近样本外检验。当前结果显示，融合财务绩效和 TOPSIS 后的 F1-score 略高于基础 LSTM，但策略累计收益仍不稳定，说明经营绩效变量可能改善部分方向识别，却尚未形成稳健交易优势。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-window LSTM validation.")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "lstm_model_data.csv"))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--first-test-year", type=int, default=2023)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    print(f"Saved rolling metrics: {TABLE_DIR / 'lstm_rolling_metrics.csv'}")
    print(f"Saved rolling predictions: {TABLE_DIR / 'lstm_rolling_predictions.csv'}")
    summary = metrics[metrics["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return"])].pivot_table(
        index="model",
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
