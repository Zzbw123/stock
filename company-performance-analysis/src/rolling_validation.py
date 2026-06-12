"""Rolling-window LSTM validation with class imbalance and trading costs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lstm_predict import (
    FIGURE_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    LSTMConfig,
    TARGET_DIRECTION_TEMPLATE,
    TARGET_RETURN_TEMPLATE,
    load_or_build_model_data,
    make_trade_signal,
    make_metrics_rows,
    make_sequences_with_meta,
    market_filter_mask,
    predict_model,
    prepare_model_frame,
    restrict_to_common_fusion_period,
    select_classification_threshold,
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
        meta_cols=[market_filter_column],
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
        meta_cols=[market_filter_column],
    )

    config = LSTMConfig(hidden_size=hidden_size)
    model, _valid_loss = train_model(
        x,
        y,
        train_idx,
        valid_idx,
        task="classification",
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        class_weight="balanced",
        log_interval=log_interval,
        run_name=f"rolling-{model_name}-{test_year}",
        patience=patience,
    )
    _valid_pred, valid_prob = predict_model(model, x[valid_idx], task="classification", threshold=0.5)
    threshold, sell_threshold, threshold_score, valid_trade_count = select_classification_threshold(
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
    test_pred, test_prob = predict_model(model, x[test_idx], task="classification", threshold=threshold)
    test_signal = make_trade_signal(
        test_prob,
        threshold,
        None if np.isnan(sell_threshold) else sell_threshold,
        threshold_mode=threshold_mode,
        market_filter=market_filter_mask(meta.iloc[test_idx], market_filter_column, market_filter_min),
    )

    pred_table = meta.iloc[test_idx].reset_index(drop=True).copy()
    pred_table["model"] = model_name
    pred_table["predicted_direction"] = test_signal.astype(int)
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

    metrics = classification_metrics(y[test_idx].astype(int), test_signal.astype(int))
    metrics["strategy_cum_return"] = float(backtest["strategy_cum_return"].iloc[-1])
    metrics["buy_hold_cum_return"] = float(backtest["buy_hold_cum_return"].iloc[-1])
    metrics["max_drawdown"] = float(backtest["max_drawdown"].iloc[-1])
    metrics["trade_count"] = float(backtest["turnover"].sum())
    metrics["total_transaction_cost"] = float(backtest["transaction_cost"].sum())
    metrics["threshold"] = threshold
    metrics["sell_threshold"] = float(sell_threshold)
    metrics[f"valid_{threshold_objective}_at_threshold"] = threshold_score
    metrics["valid_trade_count"] = valid_trade_count
    metrics["threshold_objective"] = threshold_objective
    metrics["threshold_mode"] = threshold_mode
    metrics["min_hold_proba"] = min_hold_proba
    metrics["min_valid_trades"] = float(min_valid_trades)
    metrics["drawdown_penalty"] = float(drawdown_penalty)
    metrics["trade_penalty"] = float(trade_penalty)
    metrics["market_filter_min"] = float(market_filter_min)
    metrics["num_layers"] = float(config.num_layers)
    metrics["hidden_size"] = float(config.hidden_size)
    metrics["best_epoch"] = float(getattr(model, "training_summary", {}).get("best_epoch", np.nan))
    metrics["best_valid_loss"] = float(getattr(model, "training_summary", {}).get("best_valid_loss", np.nan))
    metrics["epochs_trained"] = float(getattr(model, "training_summary", {}).get("epochs_trained", np.nan))
    confusion = confusion_matrix_frame(y[test_idx].astype(int), test_signal.astype(int))
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
    hidden_size: int = 64,
    log_interval: int = 10,
    patience: int = 10,
    threshold_objective: str = "strategy_return",
    min_hold_proba: float = 0.55,
    min_valid_trades: int = 5,
    threshold_mode: str = "dual",
    drawdown_penalty: float = 0.5,
    trade_penalty: float = 0.002,
    market_filter_column: str = "csi_pharma_return_20d",
    market_filter_min: float = 0.0,
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
                hidden_size=hidden_size,
                log_interval=log_interval,
                patience=patience,
                threshold_objective=threshold_objective,
                min_hold_proba=min_hold_proba,
                min_valid_trades=min_valid_trades,
                threshold_mode=threshold_mode,
                drawdown_penalty=drawdown_penalty,
                trade_penalty=trade_penalty,
                market_filter_column=market_filter_column,
                market_filter_min=market_filter_min,
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


def _rolling_metric_table(metrics: pd.DataFrame, metric_names: list[str], by_fold: bool = True) -> pd.DataFrame:
    selected = metrics[metrics["metric"].isin(metric_names)].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    index = ["fold", "model"] if by_fold else "model"
    aggfunc = "first" if by_fold else "mean"
    table = selected.pivot_table(index=index, columns="metric", values="value", aggfunc=aggfunc)
    return table.reindex(columns=[m for m in metric_names if m in table.columns])


def _rolling_avg_value(metrics: pd.DataFrame, model: str, metric: str, default: float = np.nan) -> float:
    selected = metrics[(metrics["model"] == model) & (metrics["metric"] == metric)].copy()
    if selected.empty:
        return default
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    return float(selected["value"].mean())


def _rolling_fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def _rolling_fmt_num(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_rolling_report(metrics: pd.DataFrame) -> None:
    report_path = FIGURE_DIR.parents[0] / "lstm_rolling_report.md"
    key_metrics = [
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
        "best_epoch",
        "epochs_trained",
    ]
    by_fold = _rolling_metric_table(metrics, key_metrics, by_fold=True)
    average = _rolling_metric_table(metrics, key_metrics, by_fold=False)

    base_return = _rolling_avg_value(metrics, "base", "strategy_cum_return")
    fusion_return = _rolling_avg_value(metrics, "fusion", "strategy_cum_return")
    naive_return = _rolling_avg_value(metrics, "naive_momentum", "strategy_cum_return")
    base_f1 = _rolling_avg_value(metrics, "base", "F1-score")
    fusion_f1 = _rolling_avg_value(metrics, "fusion", "F1-score")
    base_trades = _rolling_avg_value(metrics, "base", "trade_count")
    fusion_trades = _rolling_avg_value(metrics, "fusion", "trade_count")
    base_drawdown = _rolling_avg_value(metrics, "base", "max_drawdown")
    fusion_drawdown = _rolling_avg_value(metrics, "fusion", "max_drawdown")

    lines = [
        "# LSTM滚动验证与策略稳健性报告",
        "",
        "## 1. 为什么要做滚动验证",
        "单次训练/测试切分只能说明某一段样本外表现，容易被特定年份行情放大或掩盖。滚动验证采用 expanding-window：每次只使用测试年份之前的历史样本训练和调参，再把完整测试年份留作样本外检验，更接近真实投资中逐年更新模型的过程。",
        "",
        "## 2. 验证流程",
        "每个年度折叠中，训练期尾部15%作为验证集。模型在验证集上早停，并用风险调整收益选择交易阈值。测试年完全不参与训练、标准化拟合和阈值选择。",
        "",
        "交易策略为 long-flat：预测上涨概率高于买入阈值且行业市场过滤通过时持仓，否则空仓。回测扣除单边0.1%交易成本，并使用双阈值降低频繁反复交易。",
        "",
        "## 3. 分年度结果",
        "```text",
        by_fold.round(4).to_string() if not by_fold.empty else "No fold table available.",
        "```",
        "",
        "## 4. 平均表现",
        "```text",
        average.round(4).to_string() if not average.empty else "No average table available.",
        "```",
        "",
        "## 5. 结论",
        f"滚动验证均值显示，普通 LSTM 的 F1-score 为 {_rolling_fmt_num(base_f1)}，平均策略收益为 {_rolling_fmt_pct(base_return)}，平均最大回撤为 {_rolling_fmt_pct(base_drawdown)}，平均交易次数为 {_rolling_fmt_num(base_trades, 1)}。",
        f"融合 LSTM 的 F1-score 为 {_rolling_fmt_num(fusion_f1)}，平均策略收益为 {_rolling_fmt_pct(fusion_return)}，平均最大回撤为 {_rolling_fmt_pct(fusion_drawdown)}，平均交易次数为 {_rolling_fmt_num(fusion_trades, 1)}。朴素动量策略平均收益为 {_rolling_fmt_pct(naive_return)}。",
        "",
        "当前滚动结果支持把普通 LSTM 作为主策略模型。融合模型加入了更多经营绩效信息，但在5日短周期上没有形成稳定的收益优势，且交易次数偏少时容易出现“准确率不低但实际不开仓”的情况。",
        "",
        "## 6. 图表解释",
        "- `outputs/figures/lstm_rolling_f1.png`：按年度展示不同模型的 F1-score，用来观察方向识别能力是否跨年份稳定。",
        "- `outputs/figures/lstm_rolling_strategy_return.png`：按年度展示扣除交易成本后的策略收益，是判断模型是否可交易的核心稳健性图。",
        "- `outputs/figures/rolling_prediction_signals_base.png`：展示普通 LSTM 最新滚动模型在价格图上的买入/空仓信号，适合解释最终策略行为。",
        "- `outputs/figures/rolling_prediction_signals_fusion.png`：展示融合 LSTM 最新信号，可作为普通模型的对照图，不建议单独作为主结论。",
        "",
        "## 7. 后续优化方向",
        "后续不建议盲目继续增加 LSTM 层数。更有价值的方向是扩展股票横截面、加入更严格的滑点和停牌/涨跌停约束、按不同市场阶段分别调参，并测试10日或20日预测周期，看基本面融合模型是否在更长周期上发挥作用。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-window LSTM validation.")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "lstm_model_data.csv"))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--first-test-year", type=int, default=2023)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--threshold-objective", choices=["f1", "strategy_return", "risk_adjusted_return"], default="risk_adjusted_return")
    parser.add_argument("--threshold-mode", choices=["single", "dual"], default="dual")
    parser.add_argument("--min-hold-proba", type=float, default=0.55)
    parser.add_argument("--min-valid-trades", type=int, default=5)
    parser.add_argument("--drawdown-penalty", type=float, default=0.5)
    parser.add_argument("--trade-penalty", type=float, default=0.002)
    parser.add_argument("--market-filter-column", default="csi_pharma_return_20d")
    parser.add_argument("--market-filter-min", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=10, help="Print training loss every N epochs; 0 disables logs.")
    parser.add_argument("--patience", type=int, default=10, help="Stop after N epochs without validation-loss improvement; 0 disables early stopping.")
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
