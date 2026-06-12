"""Walk-forward LightGBM panel prediction and TopK portfolio backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_sectional_backtest import run_variant_backtest, summarize_backtest
from model_evaluation import classification_metrics
from panel_modeling import (
    FIGURE_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    TARGET_DIRECTION,
    TARGET_RETURN,
    best_threshold,
    build_model,
    feature_sets,
    load_panel,
    make_design_matrix,
    predict_probability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "walk_forward_panel_backtest_report.md"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def year_split(df: pd.DataFrame, test_year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use all data before the validation year for training, previous year for validation."""
    valid_year = test_year - 1
    year = df["date"].dt.year
    train_idx = np.where(year < valid_year)[0]
    valid_idx = np.where(year == valid_year)[0]
    test_idx = np.where(year == test_year)[0]
    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    return train_idx, valid_idx, test_idx


def run_walk_forward_predictions(
    df: pd.DataFrame,
    feature_set_names: list[str],
    first_test_year: int,
    last_test_year: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = feature_sets(df)
    available_sets = [name for name in feature_set_names if name in sets]
    if not available_sets:
        raise ValueError(f"No valid feature sets selected. Available: {sorted(sets)}")

    max_year = int(df["date"].dt.year.max()) if last_test_year is None else int(last_test_year)
    y = df[TARGET_DIRECTION].astype(int).to_numpy()
    all_predictions: list[pd.DataFrame] = []
    all_fold_metrics: list[dict[str, object]] = []

    for test_year in range(first_test_year, max_year + 1):
        train_idx, valid_idx, test_idx = year_split(df, test_year)
        if len(train_idx) == 0:
            print(f"Skip {test_year}: empty train/valid/test split.")
            continue

        for feature_set in available_sets:
            features = sets[feature_set]
            x = make_design_matrix(df, features)
            model = build_model("lightgbm", seed)
            model.fit(x.iloc[train_idx], y[train_idx])

            valid_prob = predict_probability(model, x.iloc[valid_idx])
            threshold, valid_f1 = best_threshold(y[valid_idx], valid_prob)
            test_prob = predict_probability(model, x.iloc[test_idx])
            test_pred = (test_prob >= threshold).astype(int)

            pred = df.iloc[test_idx][["date", "symbol", "name", "sector", "close", TARGET_RETURN, TARGET_DIRECTION]].copy()
            pred = pred.rename(columns={TARGET_RETURN: "actual_return", TARGET_DIRECTION: "actual_direction"})
            pred["source"] = "walk_forward"
            pred["feature_set"] = feature_set
            pred["model"] = "lightgbm"
            pred["variant"] = "lightgbm_" + feature_set
            pred["test_year"] = test_year
            pred["train_end_year"] = test_year - 2
            pred["valid_year"] = test_year - 1
            pred["predicted_probability"] = test_prob
            pred["threshold"] = threshold
            pred["predicted_direction"] = test_pred
            all_predictions.append(pred)

            fold_scores = classification_metrics(y[test_idx], test_pred)
            fold_scores.update(
                {
                    "source": "walk_forward",
                    "variant": "lightgbm_" + feature_set,
                    "feature_set": feature_set,
                    "model": "lightgbm",
                    "test_year": test_year,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "test_rows": len(test_idx),
                    "threshold": threshold,
                    "valid_f1_at_threshold": valid_f1,
                }
            )
            all_fold_metrics.append(fold_scores)
            print(
                f"[OK] test_year={test_year} lightgbm/{feature_set}: "
                f"valid_f1={valid_f1:.3f}, test_f1={fold_scores['F1-score']:.3f}, threshold={threshold:.2f}"
            )

    if not all_predictions:
        raise RuntimeError("No walk-forward predictions were generated.")
    return pd.concat(all_predictions, ignore_index=True), pd.DataFrame(all_fold_metrics)


def run_topk_grid(
    predictions: pd.DataFrame,
    top_k_values: list[int],
    min_probability_values: list[float],
    rebalance_step: int,
    transaction_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backtests: list[pd.DataFrame] = []
    details: list[pd.DataFrame] = []

    for (source, variant), group in predictions.groupby(["source", "variant"], sort=False):
        for top_k in top_k_values:
            for min_probability in min_probability_values:
                backtest, detail = run_variant_backtest(
                    group,
                    source=source,
                    variant=variant,
                    top_k=top_k,
                    rebalance_step=rebalance_step,
                    transaction_cost=transaction_cost,
                    min_probability=min_probability,
                )
                if not backtest.empty:
                    backtests.append(backtest)
                if not detail.empty:
                    details.append(detail)

    if not backtests:
        raise RuntimeError("No valid TopK walk-forward backtests were generated.")

    backtest_df = pd.concat(backtests, ignore_index=True)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    metrics_df = summarize_backtest(backtest_df, detail_df, rebalance_step)
    return backtest_df, detail_df, metrics_df


def summarize_yearly(backtest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source, variant, top_k, min_probability, year), group in backtest.assign(
        year=backtest["rebalance_date"].dt.year
    ).groupby(["source", "variant", "top_k", "min_probability", "year"], sort=False):
        group = group.sort_values("rebalance_date")
        total_return = float((1 + group["net_return"]).prod() - 1)
        benchmark_return = float((1 + group["benchmark_return"]).prod() - 1)
        rows.append(
            {
                "source": source,
                "variant": variant,
                "top_k": top_k,
                "min_probability": min_probability,
                "year": year,
                "periods": len(group),
                "total_return": total_return,
                "benchmark_total_return": benchmark_return,
                "excess_total_return": total_return - benchmark_return,
                "max_drawdown": float((1 + group["net_return"]).cumprod().pipe(lambda curve: curve / curve.cummax() - 1).min()),
                "win_rate": float((group["net_return"] > 0).mean()),
                "avg_turnover": float(group["turnover"].mean()),
                "avg_selected_count": float(group["selected_count"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "total_return"], ascending=[True, False])


def plot_walk_forward(backtest: pd.DataFrame, metrics: pd.DataFrame, top_n: int = 6) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    label_cols = ["source", "variant", "top_k", "min_probability"]
    best = metrics.head(top_n)[label_cols].apply(tuple, axis=1).tolist()
    plot_df = backtest[backtest[label_cols].apply(tuple, axis=1).isin(best)].copy()
    if plot_df.empty:
        return

    plot_df["label"] = (
        plot_df["variant"]
        + " K="
        + plot_df["top_k"].astype(str)
        + " P>="
        + plot_df["min_probability"].map(lambda value: f"{value:.2f}")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, group in plot_df.groupby("label", sort=False):
        group = group.sort_values("rebalance_date")
        ax.plot(group["rebalance_date"], group["cumulative_return"], label=label, linewidth=1.8)
    benchmark = backtest.sort_values("rebalance_date").drop_duplicates("rebalance_date")
    ax.plot(
        benchmark["rebalance_date"],
        benchmark["benchmark_cumulative_return"],
        label="equal_weight_benchmark",
        linestyle="--",
        color="black",
        linewidth=1.5,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Walk-forward non-overlapping TopK portfolio return")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "walk_forward_topk_return.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, group in plot_df.groupby("label", sort=False):
        group = group.sort_values("rebalance_date")
        ax.plot(group["rebalance_date"], group["excess_cumulative_return"], label=label, linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Walk-forward TopK excess return")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Excess cumulative return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "walk_forward_topk_excess_return.png", dpi=180)
    plt.close(fig)

    best_rows = metrics.head(12).copy()
    best_rows["label"] = best_rows["variant"] + " K=" + best_rows["top_k"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    best_rows.sort_values("total_return").plot.barh(x="label", y="total_return", ax=ax, color="#2563eb", legend=False)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Walk-forward TopK total return ranking")
    ax.set_xlabel("Total return")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "walk_forward_topk_ranking.png", dpi=180)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    yearly_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Walk-forward 面板 LightGBM + TopK 回测",
        "",
        "## 设置",
        "",
        f"- 测试年份：{args.first_test_year} 至 {args.last_test_year or '数据最新年份'}",
        f"- 验证方式：测试年前一年选分类阈值，训练集为验证年前全部历史数据",
        f"- 特征集：{', '.join(args.feature_sets)}",
        f"- TopK：{', '.join(map(str, args.top_k_values))}",
        f"- 调仓间隔：每 {args.rebalance_step} 个交易日",
        f"- 单边交易成本：{args.transaction_cost:.4f}",
        "",
    ]
    if not metrics.empty:
        best = metrics.iloc[0]
        lines.extend(
            [
                "## 最优组合",
                "",
                f"- 模型：{best['variant']}",
                f"- 参数：Top{int(best['top_k'])}, 最低概率 {best['min_probability']:.3f}",
                f"- 总收益：{best['total_return']:.4f}",
                f"- 等权基准总收益：{best['benchmark_total_return']:.4f}",
                f"- 超额收益：{best['excess_total_return']:.4f}",
                f"- 最大回撤：{best['max_drawdown']:.4f}",
                f"- Sharpe：{best['sharpe']:.4f}" if pd.notna(best["sharpe"]) else "- Sharpe：NA",
                "",
            ]
        )

    if not fold_metrics.empty:
        fold_view = fold_metrics[
            ["variant", "test_year", "Accuracy", "Precision", "Recall", "F1-score", "threshold", "valid_f1_at_threshold"]
        ].copy()
        lines.extend(
            [
                "## 年度分类表现",
                "",
                "```text",
                fold_view.round(4).to_string(index=False),
                "```",
                "",
            ]
        )

    if not yearly_metrics.empty and not metrics.empty:
        best = metrics.iloc[0]
        best_yearly = yearly_metrics[
            (yearly_metrics["variant"] == best["variant"])
            & (yearly_metrics["top_k"] == best["top_k"])
            & (yearly_metrics["min_probability"] == best["min_probability"])
        ]
        lines.extend(
            [
                "## 最优组合年度拆解",
                "",
                "```text",
                best_yearly.round(4).to_string(index=False),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 结果文件",
            "",
            "- outputs/tables/walk_forward_panel_predictions.csv",
            "- outputs/tables/walk_forward_fold_metrics.csv",
            "- outputs/tables/walk_forward_topk_backtest.csv",
            "- outputs/tables/walk_forward_topk_metrics.csv",
            "- outputs/tables/walk_forward_topk_yearly_metrics.csv",
            "- outputs/tables/walk_forward_topk_selection_detail.csv",
            "- outputs/figures/walk_forward_topk_return.png",
            "- outputs/figures/walk_forward_topk_excess_return.png",
            "- outputs/figures/walk_forward_topk_ranking.png",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROCESSED_DIR / "panel_model_data.csv"))
    parser.add_argument("--feature-sets", nargs="+", default=["market", "valuation", "financial"])
    parser.add_argument("--first-test-year", type=int, default=2022)
    parser.add_argument("--last-test-year", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-grid", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--min-probability", type=float, default=0.0)
    parser.add_argument("--min-probability-grid", nargs="+", type=float, default=None)
    parser.add_argument("--rebalance-step", type=int, default=5)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.top_k_values = args.top_k_grid or [args.top_k]
    args.min_probability_values = args.min_probability_grid or [args.min_probability]
    if any(value <= 0 for value in args.top_k_values):
        raise ValueError("All TopK values must be positive.")
    if args.rebalance_step <= 0:
        raise ValueError("--rebalance-step must be positive.")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel(args.data)
    predictions, fold_metrics = run_walk_forward_predictions(
        df=df,
        feature_set_names=args.feature_sets,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        seed=args.seed,
    )
    backtest, detail, topk_metrics = run_topk_grid(
        predictions=predictions,
        top_k_values=args.top_k_values,
        min_probability_values=args.min_probability_values,
        rebalance_step=args.rebalance_step,
        transaction_cost=args.transaction_cost,
    )
    yearly_metrics = summarize_yearly(backtest)

    predictions.to_csv(TABLE_DIR / "walk_forward_panel_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(TABLE_DIR / "walk_forward_fold_metrics.csv", index=False, encoding="utf-8-sig")
    backtest.to_csv(TABLE_DIR / "walk_forward_topk_backtest.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(TABLE_DIR / "walk_forward_topk_selection_detail.csv", index=False, encoding="utf-8-sig")
    topk_metrics.to_csv(TABLE_DIR / "walk_forward_topk_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_metrics.to_csv(TABLE_DIR / "walk_forward_topk_yearly_metrics.csv", index=False, encoding="utf-8-sig")
    plot_walk_forward(backtest, topk_metrics)
    write_report(topk_metrics, fold_metrics, yearly_metrics, args)

    print("Walk-forward panel TopK backtest finished.")
    print(topk_metrics.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
