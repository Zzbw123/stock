"""Non-overlapping TopK cross-sectional backtest from panel prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_evaluation import max_drawdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_PATH = PROJECT_ROOT / "outputs" / "cross_sectional_backtest_report.md"

PREDICTION_FILES = {
    "dual_branch": TABLE_DIR / "dual_branch_predictions.csv",
    "panel": TABLE_DIR / "panel_model_predictions.csv",
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_predictions(path: Path, source: str) -> pd.DataFrame:
    """Load prediction rows and build a stable model variant name."""
    df = pd.read_csv(path, dtype={"symbol": str})
    required = {"date", "symbol", "actual_return", "predicted_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["actual_return"] = pd.to_numeric(df["actual_return"], errors="coerce")
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")

    if "feature_set" in df.columns:
        df["variant"] = df["model"].astype(str) + "_" + df["feature_set"].astype(str)
    else:
        df["variant"] = df.get("model", source).astype(str)
    df["source"] = source

    keep = [
        "source",
        "variant",
        "date",
        "symbol",
        "name",
        "sector",
        "close",
        "actual_return",
        "actual_direction",
        "predicted_probability",
    ]
    keep = [col for col in keep if col in df.columns]
    return df[keep].dropna(subset=["date", "actual_return", "predicted_probability"])


def select_rebalance_dates(dates: pd.Series, step: int) -> list[pd.Timestamp]:
    unique_dates = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values().reset_index(drop=True)
    if unique_dates.empty:
        return []
    return list(unique_dates.iloc[::step])


def weight_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(previous) | set(current)
    return float(sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))


def run_variant_backtest(
    df: pd.DataFrame,
    source: str,
    variant: str,
    top_k: int,
    rebalance_step: int,
    transaction_cost: float,
    min_probability: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = select_rebalance_dates(df["date"], rebalance_step)
    previous_weights: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for date in dates:
        daily = df[df["date"] == date].copy()
        daily = daily.dropna(subset=["actual_return", "predicted_probability"])
        if daily.empty:
            continue

        candidates = daily[daily["predicted_probability"] >= min_probability].sort_values(
            "predicted_probability", ascending=False
        )
        selected = candidates.head(top_k)
        selected_count = len(selected)
        if selected_count:
            current_weights = dict(zip(selected["symbol"], np.repeat(1 / selected_count, selected_count)))
            gross_return = float(selected["actual_return"].mean())
            selected_symbols = ",".join(selected["symbol"].astype(str).tolist())
            selected_names = ",".join(selected.get("name", selected["symbol"]).astype(str).tolist())
            avg_probability = float(selected["predicted_probability"].mean())
        else:
            current_weights = {}
            gross_return = 0.0
            selected_symbols = ""
            selected_names = ""
            avg_probability = np.nan

        turnover = weight_turnover(previous_weights, current_weights)
        cost = max(transaction_cost, 0.0) * turnover
        net_return = gross_return - cost
        benchmark_return = float(daily["actual_return"].mean())
        rows.append(
            {
                "source": source,
                "variant": variant,
                "rebalance_date": date,
                "holding_days": rebalance_step,
                "top_k": top_k,
                "min_probability": min_probability,
                "available_count": len(daily),
                "selected_count": selected_count,
                "selected_symbols": selected_symbols,
                "selected_names": selected_names,
                "avg_probability": avg_probability,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": cost,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "excess_return": net_return - benchmark_return,
            }
        )

        for _, item in selected.iterrows():
            detail_rows.append(
                {
                    "source": source,
                    "variant": variant,
                    "rebalance_date": date,
                    "top_k": top_k,
                    "min_probability": min_probability,
                    "symbol": item["symbol"],
                    "name": item.get("name", ""),
                    "sector": item.get("sector", ""),
                    "predicted_probability": item["predicted_probability"],
                    "actual_return": item["actual_return"],
                    "actual_direction": item.get("actual_direction", np.nan),
                }
            )
        previous_weights = current_weights

    result = pd.DataFrame(rows)
    if not result.empty:
        result["cumulative_return"] = (1 + result["net_return"]).cumprod() - 1
        result["benchmark_cumulative_return"] = (1 + result["benchmark_return"]).cumprod() - 1
        result["excess_cumulative_return"] = (
            (1 + result["net_return"]).cumprod() / (1 + result["benchmark_return"]).cumprod() - 1
        )
    return result, pd.DataFrame(detail_rows)


def summarize_backtest(backtest: pd.DataFrame, detail: pd.DataFrame, rebalance_step: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods_per_year = 252 / rebalance_step

    group_cols = ["source", "variant", "top_k", "min_probability"]
    for keys, group in backtest.groupby(group_cols, sort=False):
        source, variant, top_k, min_probability = keys
        group = group.sort_values("rebalance_date")
        periods = len(group)
        total_return = float((1 + group["net_return"]).prod() - 1)
        benchmark_total = float((1 + group["benchmark_return"]).prod() - 1)
        ann_return = np.nan
        benchmark_ann_return = np.nan
        if periods and total_return > -1:
            ann_return = float((1 + total_return) ** (periods_per_year / periods) - 1)
        if periods and benchmark_total > -1:
            benchmark_ann_return = float((1 + benchmark_total) ** (periods_per_year / periods) - 1)
        period_std = float(group["net_return"].std(ddof=1)) if periods > 1 else np.nan
        sharpe = np.nan
        if period_std and not np.isnan(period_std) and period_std > 1e-12:
            sharpe = float(group["net_return"].mean() / period_std * np.sqrt(periods_per_year))

        selected = detail[
            (detail["source"] == source)
            & (detail["variant"] == variant)
            & (detail["top_k"] == top_k)
            & (detail["min_probability"] == min_probability)
        ]
        rows.append(
            {
                "source": source,
                "variant": variant,
                "top_k": top_k,
                "min_probability": min_probability,
                "periods": periods,
                "total_return": total_return,
                "benchmark_total_return": benchmark_total,
                "excess_total_return": total_return - benchmark_total,
                "annualized_return": ann_return,
                "benchmark_annualized_return": benchmark_ann_return,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown(group["cumulative_return"]),
                "win_rate": float((group["net_return"] > 0).mean()) if periods else np.nan,
                "avg_period_return": float(group["net_return"].mean()) if periods else np.nan,
                "avg_excess_return": float(group["excess_return"].mean()) if periods else np.nan,
                "avg_turnover": float(group["turnover"].mean()) if periods else np.nan,
                "avg_selected_count": float(group["selected_count"].mean()) if periods else np.nan,
                "selected_positive_rate": float((selected["actual_return"] > 0).mean()) if len(selected) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["total_return", "sharpe"], ascending=False)


def plot_backtest(backtest: pd.DataFrame, metrics: pd.DataFrame, top_n: int = 6) -> None:
    if backtest.empty or metrics.empty:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    label_cols = ["source", "variant", "top_k", "min_probability"]
    best = metrics.head(top_n)[label_cols].apply(tuple, axis=1).tolist()
    plot_df = backtest[backtest[label_cols].apply(tuple, axis=1).isin(best)].copy()
    plot_df["label"] = (
        plot_df["source"]
        + ":"
        + plot_df["variant"]
        + " K="
        + plot_df["top_k"].astype(str)
        + " P>="
        + plot_df["min_probability"].map(lambda value: f"{value:.2f}")
    )

    plt.figure(figsize=(11, 6))
    for label, group in plot_df.groupby("label", sort=False):
        group = group.sort_values("rebalance_date")
        plt.plot(group["rebalance_date"], group["cumulative_return"], label=label, linewidth=1.8)
    benchmark = backtest.sort_values("rebalance_date").drop_duplicates("rebalance_date")
    plt.plot(
        benchmark["rebalance_date"],
        benchmark["benchmark_cumulative_return"],
        label="equal_weight_benchmark",
        linestyle="--",
        color="black",
        linewidth=1.5,
    )
    plt.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    plt.title("Non-overlapping 5-day TopK portfolio return")
    plt.xlabel("Rebalance date")
    plt.ylabel("Cumulative return")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cross_sectional_topk_return.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for label, group in plot_df.groupby("label", sort=False):
        group = group.sort_values("rebalance_date")
        plt.plot(group["rebalance_date"], group["excess_cumulative_return"], label=label, linewidth=1.8)
    plt.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    plt.title("TopK excess return versus equal-weight benchmark")
    plt.xlabel("Rebalance date")
    plt.ylabel("Excess cumulative return")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cross_sectional_topk_excess_return.png", dpi=180)
    plt.close()


def write_report(metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 截面 TopK 非重叠调仓回测",
        "",
        f"- 调仓间隔：每 {args.rebalance_step} 个交易日",
        f"- TopK：{', '.join(map(str, args.top_k_values))}",
        f"- 最低预测概率：{', '.join(f'{value:.3f}' for value in args.min_probability_values)}",
        f"- 单边交易成本：{args.transaction_cost:.4f}",
        "",
    ]
    if metrics.empty:
        lines.append("未生成有效回测结果。")
    else:
        best = metrics.iloc[0]
        lines.extend(
            [
                "## 最优组合",
                "",
                f"- 来源/模型：{best['source']} / {best['variant']}",
                f"- 参数：Top{int(best['top_k'])}, 最低概率 {best['min_probability']:.3f}",
                f"- 总收益：{best['total_return']:.4f}",
                f"- 等权基准总收益：{best['benchmark_total_return']:.4f}",
                f"- 超额收益：{best['excess_total_return']:.4f}",
                f"- 最大回撤：{best['max_drawdown']:.4f}",
                f"- Sharpe：{best['sharpe']:.4f}" if pd.notna(best["sharpe"]) else "- Sharpe：NA",
                "",
                "## 结果文件",
                "",
                "- outputs/tables/cross_sectional_backtest.csv",
                "- outputs/tables/cross_sectional_selection_detail.csv",
                "- outputs/tables/cross_sectional_metrics.csv",
                "- outputs/figures/cross_sectional_topk_return.png",
                "- outputs/figures/cross_sectional_topk_excess_return.png",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3, help="Number of stocks selected on each rebalance date.")
    parser.add_argument(
        "--top-k-grid",
        nargs="+",
        type=int,
        default=None,
        help="Optional TopK values to evaluate in one run.",
    )
    parser.add_argument("--rebalance-step", type=int, default=5, help="Non-overlapping rebalance interval in trading days.")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Single-side transaction cost.")
    parser.add_argument("--min-probability", type=float, default=0.0, help="Minimum predicted probability for selection.")
    parser.add_argument(
        "--min-probability-grid",
        nargs="+",
        type=float,
        default=None,
        help="Optional probability floors to evaluate in one run.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["dual_branch", "panel"],
        choices=sorted(PREDICTION_FILES),
        help="Prediction sources to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.rebalance_step <= 0:
        raise ValueError("--rebalance-step must be positive.")
    args.top_k_values = args.top_k_grid or [args.top_k]
    args.min_probability_values = args.min_probability_grid or [args.min_probability]
    if any(value <= 0 for value in args.top_k_values):
        raise ValueError("All TopK values must be positive.")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    for source in args.sources:
        path = PREDICTION_FILES[source]
        if not path.exists():
            print(f"Skip missing prediction file: {path}")
            continue
        all_predictions.append(load_predictions(path, source))
    if not all_predictions:
        raise FileNotFoundError("No prediction files were found. Run panel_modeling.py or dual_branch_lstm.py first.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    backtests = []
    details = []
    for (source, variant), group in predictions.groupby(["source", "variant"], sort=False):
        for top_k in args.top_k_values:
            for min_probability in args.min_probability_values:
                backtest, detail = run_variant_backtest(
                    group,
                    source=source,
                    variant=variant,
                    top_k=top_k,
                    rebalance_step=args.rebalance_step,
                    transaction_cost=args.transaction_cost,
                    min_probability=min_probability,
                )
                if not backtest.empty:
                    backtests.append(backtest)
                if not detail.empty:
                    details.append(detail)

    if not backtests:
        raise RuntimeError("No valid backtest periods were generated.")

    backtest_df = pd.concat(backtests, ignore_index=True)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    metrics_df = summarize_backtest(backtest_df, detail_df, args.rebalance_step)

    backtest_df.to_csv(TABLE_DIR / "cross_sectional_backtest.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(TABLE_DIR / "cross_sectional_selection_detail.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(TABLE_DIR / "cross_sectional_metrics.csv", index=False, encoding="utf-8-sig")
    plot_backtest(backtest_df, metrics_df)
    write_report(metrics_df, args)

    print("Cross-sectional TopK backtest finished.")
    print(metrics_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
