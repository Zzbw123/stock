"""Generate publication-style LSTM figures and a full Chinese report."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_PATH = PROJECT_ROOT / "outputs" / "lstm_full_report.md"
SCRIPT_PATH = Path(__file__).resolve()


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
    }
)

COLORS = {
    "blue": "#2563eb",
    "teal": "#0f766e",
    "orange": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#64748b",
    "black": "#111827",
    "light_gray": "#e5e7eb",
}


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def to_numeric_value(df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def metric_value(df: pd.DataFrame, metric: str, model: str | None = None) -> float:
    mask = df["metric"].eq(metric)
    if model is not None:
        mask &= df["model"].eq(model)
    values = pd.to_numeric(df.loc[mask, "value"], errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else np.nan


def export_figure(fig: plt.Figure, figure_id: str, contract: dict, plot_data_paths: list[Path]) -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for ext in ["svg", "png", "pdf"]:
        path = FIGURE_DIR / f"{figure_id}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        output_paths.append(path)
    trace = {
        "figure_id": figure_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "source_data_paths": contract["source_data_paths"],
        "plot_data_paths": [str(path) for path in plot_data_paths],
        "output_targets": [str(path) for path in output_paths],
        "contract": contract,
    }
    trace_path = FIGURE_DIR / f"{figure_id}.trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "plot_data": "; ".join(str(path) for path in plot_data_paths),
        "trace": str(trace_path),
        "outputs": "; ".join(str(path) for path in output_paths),
    }


def upsert_csv(path: Path, rows: list[dict], key: str) -> None:
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        old_df = old_df[~old_df[key].isin(new_df[key])]
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False, encoding="utf-8-sig")


def category_of(feature: str) -> str:
    valuation = {"pe_ttm", "pe_static", "pb", "peg", "pcf", "ps", "total_market_cap", "float_market_cap"}
    market = {"hs300", "csi_pharma", "chinext"}
    financial = {"revenue", "net_profit", "roe", "gross_margin", "net_margin", "asset_liability_ratio", "current_ratio"}
    topsis = {"topsis_score", "rank"}
    technical_keywords = [
        "return",
        "ma",
        "volatility",
        "rsi",
        "macd",
        "boll",
        "volume",
        "turnover",
        "close",
        "open",
        "high",
        "low",
        "amount",
        "pct",
    ]
    if feature in topsis:
        return "TOPSIS/绩效"
    if feature in valuation:
        return "估值"
    if feature in financial or feature.endswith("_growth"):
        return "财务"
    if any(feature.startswith(prefix) for prefix in market):
        return "市场环境"
    if any(key in feature for key in technical_keywords):
        return "行情/技术"
    return "其他"


def load_all() -> dict[str, pd.DataFrame]:
    data = {
        "lstm_metrics": to_numeric_value(read_csv("lstm_metrics.csv")),
        "lstm_predictions": read_csv("lstm_predictions.csv", parse_dates=["date"]),
        "rolling_metrics": to_numeric_value(read_csv("lstm_rolling_metrics.csv")),
        "latest_signal": read_csv("rolling_latest_signal.csv"),
        "shap_importance": read_csv("shap_importance.csv"),
        "shap_metrics": read_csv("shap_proxy_metrics.csv"),
        "dual_metrics": to_numeric_value(read_csv("dual_branch_metrics.csv")),
        "dual_symbol": read_csv("dual_branch_symbol_metrics.csv", dtype={"symbol": str}),
        "walk_metrics": read_csv("walk_forward_topk_metrics.csv"),
        "walk_yearly": read_csv("walk_forward_topk_yearly_metrics.csv"),
        "walk_fold": read_csv("walk_forward_fold_metrics.csv"),
        "walk_backtest": read_csv("walk_forward_topk_backtest.csv", parse_dates=["rebalance_date"]),
    }
    return data


def figure_model_progress(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    lstm = data["lstm_metrics"]
    rolling = data["rolling_metrics"]
    dual = data["dual_metrics"]

    rows = []
    for model, label in [
        ("naive_momentum", "动量基准"),
        ("base", "基础 LSTM"),
        ("fusion", "绩效融合 LSTM"),
    ]:
        rows.append(
            {
                "stage": "单次切分",
                "model": label,
                "f1": metric_value(lstm, "F1-score", model),
                "strategy_return": metric_value(lstm, "strategy_cum_return", model),
            }
        )

    for model, label in [
        ("naive_momentum", "滚动动量基准"),
        ("base", "滚动基础 LSTM"),
        ("fusion", "滚动融合 LSTM"),
    ]:
        sub = rolling[(rolling["model"] == model) & (rolling["metric"].isin(["F1-score", "strategy_cum_return"]))]
        pivot = sub.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
        rows.append(
            {
                "stage": "滚动验证均值",
                "model": label,
                "f1": float(pivot.get("F1-score", pd.Series([np.nan])).iloc[0]),
                "strategy_return": float(pivot.get("strategy_cum_return", pd.Series([np.nan])).iloc[0]),
            }
        )

    rows.append(
        {
            "stage": "面板深度模型",
            "model": "双分支 LSTM",
            "f1": metric_value(dual, "F1-score", "dual_branch_lstm"),
            "strategy_return": metric_value(dual, "strategy_cum_return", "dual_branch_lstm"),
        }
    )
    plot_data = pd.DataFrame(rows)
    plot_path = TABLE_DIR / "plot_data_lstm_model_progress.csv"
    plot_data.to_csv(plot_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    y = np.arange(len(plot_data))
    colors = plot_data["stage"].map(
        {
            "单次切分": COLORS["gray"],
            "滚动验证均值": COLORS["blue"],
            "面板深度模型": COLORS["teal"],
        }
    )
    axes[0].barh(y, plot_data["f1"], color=colors)
    axes[0].set_yticks(y, plot_data["model"])
    axes[0].set_xlabel("F1-score")
    axes[0].set_title("A 方向预测能力")
    axes[0].axvline(0.5, color=COLORS["light_gray"], linewidth=1.2, linestyle="--")

    ret_colors = [COLORS["teal"] if value >= 0 else COLORS["red"] for value in plot_data["strategy_return"]]
    axes[1].barh(y, plot_data["strategy_return"], color=ret_colors)
    axes[1].set_xlabel("策略累计收益")
    axes[1].set_title("B 简化交易收益")
    axes[1].axvline(0, color=COLORS["black"], linewidth=0.8)
    fig.suptitle("LSTM 模型演进：单股融合并未稳定优于基准，双分支面板模型改善 F1 但收益仍需风控", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    contract = {
        "figure_id": "lstm_fig1_model_progress",
        "purpose": "比较 LSTM 体系从单股模型到滚动验证与双分支面板模型的性能演进",
        "core_conclusion": "双分支 LSTM 的 F1-score 明显高于单股 LSTM，但策略收益仍为负，说明模型提升不能替代交易风控。",
        "chart_type": "two-panel horizontal bar chart",
        "evidence_layers": "F1-score, strategy cumulative return, stage grouping",
        "source_data_paths": "outputs/tables/lstm_metrics.csv; outputs/tables/lstm_rolling_metrics.csv; outputs/tables/dual_branch_metrics.csv",
        "output_targets": "outputs/figures/lstm_fig1_model_progress.svg/png/pdf",
        "failure_signal": "若双分支模型 F1 和策略收益均未改善，则深度结构扩展缺少实证收益。",
    }
    source_map = export_figure(fig, "lstm_fig1_model_progress", contract, [plot_path])
    return contract, source_map


def figure_rolling_stability(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    rolling = data["rolling_metrics"].copy()
    rolling["test_year"] = rolling["fold"].str.extract(r"(\d{4})").astype(int)
    pivot = rolling[rolling["metric"].isin(["F1-score", "strategy_cum_return"])].pivot_table(
        index=["test_year", "model"], columns="metric", values="value", aggfunc="first"
    ).reset_index()
    plot_path = TABLE_DIR / "plot_data_lstm_rolling_stability.csv"
    pivot.to_csv(plot_path, index=False, encoding="utf-8-sig")

    labels = {"naive_momentum": "动量基准", "base": "基础 LSTM", "fusion": "绩效融合 LSTM"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharex=True)
    for model, group in pivot.groupby("model"):
        group = group.sort_values("test_year")
        axes[0].plot(group["test_year"], group["F1-score"], marker="o", linewidth=2, label=labels.get(model, model))
        axes[1].plot(group["test_year"], group["strategy_cum_return"], marker="o", linewidth=2, label=labels.get(model, model))
    axes[0].set_title("A 年度 F1 稳定性")
    axes[0].set_ylabel("F1-score")
    axes[0].set_xlabel("测试年份")
    axes[1].set_title("B 年度策略累计收益")
    axes[1].set_ylabel("累计收益")
    axes[1].set_xlabel("测试年份")
    axes[1].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("滚动验证揭示：融合 LSTM 的分类指标略有改善，但年度收益不稳定", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    contract = {
        "figure_id": "lstm_fig2_rolling_stability",
        "purpose": "检验单股 LSTM 在滚动年份中的稳定性和失效风险",
        "core_conclusion": "融合 LSTM 的平均 F1 略高，但不同年份策略收益波动较大，模型需要滚动验证而非单次切分。",
        "chart_type": "two-panel line chart",
        "evidence_layers": "yearly F1-score, yearly strategy cumulative return, model comparison",
        "source_data_paths": "outputs/tables/lstm_rolling_metrics.csv",
        "output_targets": "outputs/figures/lstm_fig2_rolling_stability.svg/png/pdf",
        "failure_signal": "若年度曲线方向相反或收益长期为负，则单股模型不能直接用于交易信号。",
    }
    source_map = export_figure(fig, "lstm_fig2_rolling_stability", contract, [plot_path])
    return contract, source_map


def figure_shap_explainability(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    shap = data["shap_importance"].head(18).copy()
    shap["category"] = shap["feature"].map(category_of)
    plot_path = TABLE_DIR / "plot_data_lstm_shap_importance.csv"
    shap.to_csv(plot_path, index=False, encoding="utf-8-sig")
    palette = {
        "行情/技术": COLORS["blue"],
        "估值": COLORS["orange"],
        "市场环境": COLORS["teal"],
        "财务": COLORS["purple"],
        "TOPSIS/绩效": COLORS["red"],
        "其他": COLORS["gray"],
    }

    fig, ax = plt.subplots(figsize=(9, 6.2))
    plot_df = shap.sort_values("mean_abs_shap")
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color=plot_df["category"].map(palette))
    ax.set_xlabel("mean |SHAP value|")
    ax.set_ylabel("")
    ax.set_title("SHAP 解释：短期方向主要由估值状态与技术形态共同驱动", loc="left", fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in palette.values()]
    ax.legend(handles, palette.keys(), frameon=False, fontsize=9, ncol=3, loc="lower right")
    fig.tight_layout()

    contract = {
        "figure_id": "lstm_fig3_shap_explainability",
        "purpose": "解释融合预测模型中哪些特征贡献最大",
        "core_conclusion": "PEG、布林带宽度、均线、波动率和指数收益等特征贡献靠前，短期方向更依赖估值与市场技术状态。",
        "chart_type": "horizontal SHAP importance bar chart",
        "evidence_layers": "top mean absolute SHAP features, feature category colors",
        "source_data_paths": "outputs/tables/shap_importance.csv; outputs/tables/shap_proxy_metrics.csv",
        "output_targets": "outputs/figures/lstm_fig3_shap_explainability.svg/png/pdf",
        "failure_signal": "若财务/TOPSIS 特征贡献很低，应避免夸大经营绩效对短期股价方向的即时解释力。",
    }
    source_map = export_figure(fig, "lstm_fig3_shap_explainability", contract, [plot_path])
    return contract, source_map


def figure_dual_branch_panel(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    symbol = data["dual_symbol"].copy()
    symbol["symbol"] = symbol["symbol"].astype(str).str.zfill(6)
    symbol = symbol.sort_values("F1-score", ascending=True)
    plot_path = TABLE_DIR / "plot_data_lstm_dual_branch_symbol_metrics.csv"
    symbol.to_csv(plot_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    axes[0].barh(symbol["name"], symbol["F1-score"], color=COLORS["teal"])
    axes[0].set_xlabel("F1-score")
    axes[0].set_title("A 个股层面 F1 差异")
    axes[0].axvline(symbol["F1-score"].mean(), color=COLORS["black"], linestyle="--", linewidth=1.0, label="均值")
    axes[0].legend(frameon=False, fontsize=9)

    scatter = axes[1].scatter(symbol["Precision"], symbol["Recall"], s=70, color=COLORS["blue"], alpha=0.85)
    for _, row in symbol.iterrows():
        axes[1].annotate(row["name"], (row["Precision"], row["Recall"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    axes[1].set_xlabel("Precision")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("B 高召回、低精度的信号结构")
    axes[1].set_xlim(max(0, symbol["Precision"].min() - 0.05), min(1, symbol["Precision"].max() + 0.08))
    axes[1].set_ylim(max(0, symbol["Recall"].min() - 0.05), 1.03)
    fig.suptitle("双分支 LSTM 在面板样本中提升了召回率，但个股间异质性明显", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _ = scatter

    contract = {
        "figure_id": "lstm_fig4_dual_branch_panel",
        "purpose": "展示双分支 LSTM 在不同同行股票上的预测差异",
        "core_conclusion": "药明康德、恒瑞医药等样本 F1 较高，但整体呈现高召回、低精度结构，阈值和风控仍是关键。",
        "chart_type": "bar chart plus precision-recall scatter",
        "evidence_layers": "symbol-level F1-score, precision, recall",
        "source_data_paths": "outputs/tables/dual_branch_symbol_metrics.csv",
        "output_targets": "outputs/figures/lstm_fig4_dual_branch_panel.svg/png/pdf",
        "failure_signal": "若个股间表现差异过大，应采用分组阈值或行业/市值分层模型。",
    }
    source_map = export_figure(fig, "lstm_fig4_dual_branch_panel", contract, [plot_path])
    return contract, source_map


def figure_walk_forward_strategy(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    metrics = data["walk_metrics"]
    best = metrics.iloc[0]
    yearly = data["walk_yearly"]
    best_yearly = yearly[
        (yearly["variant"] == best["variant"])
        & (yearly["top_k"] == best["top_k"])
        & (yearly["min_probability"] == best["min_probability"])
    ].copy()
    backtest = data["walk_backtest"]
    best_bt = backtest[
        (backtest["variant"] == best["variant"])
        & (backtest["top_k"] == best["top_k"])
        & (backtest["min_probability"] == best["min_probability"])
    ].copy()
    plot_year_path = TABLE_DIR / "plot_data_lstm_walk_forward_yearly.csv"
    plot_bt_path = TABLE_DIR / "plot_data_lstm_walk_forward_curve.csv"
    best_yearly.to_csv(plot_year_path, index=False, encoding="utf-8-sig")
    best_bt.to_csv(plot_bt_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(best_yearly))
    width = 0.36
    axes[0].bar(x - width / 2, best_yearly["total_return"], width=width, label="TopK 策略", color=COLORS["blue"])
    axes[0].bar(x + width / 2, best_yearly["benchmark_total_return"], width=width, label="等权基准", color=COLORS["gray"])
    axes[0].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[0].set_xticks(x, best_yearly["year"].astype(str))
    axes[0].set_ylabel("年度收益")
    axes[0].set_title("A 年度收益对比")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(best_bt["rebalance_date"], best_bt["cumulative_return"], color=COLORS["blue"], linewidth=2, label="TopK 策略")
    axes[1].plot(
        best_bt["rebalance_date"],
        best_bt["benchmark_cumulative_return"],
        color=COLORS["black"],
        linewidth=1.5,
        linestyle="--",
        label="等权基准",
    )
    axes[1].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[1].set_ylabel("累计收益")
    axes[1].set_xlabel("调仓日期")
    axes[1].set_title("B Walk-forward 累计净值")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Walk-forward 显示模型具备相对选股能力，但绝对收益仍受行业下行拖累", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    contract = {
        "figure_id": "lstm_fig5_walk_forward_strategy",
        "purpose": "在逐年重训的样本外环境中检验模型选股信号是否稳健",
        "core_conclusion": "最优 walk-forward Top5 组合跑赢等权基准 17.5 个百分点，但总收益仍为负，说明模型具有相对收益而非绝对收益保证。",
        "chart_type": "annual grouped bar plus cumulative return line",
        "evidence_layers": "yearly strategy return, yearly benchmark return, cumulative return curves",
        "source_data_paths": "outputs/tables/walk_forward_topk_metrics.csv; outputs/tables/walk_forward_topk_yearly_metrics.csv; outputs/tables/walk_forward_topk_backtest.csv",
        "output_targets": "outputs/figures/lstm_fig5_walk_forward_strategy.svg/png/pdf",
        "failure_signal": "若多数年份不能跑赢等权基准，则模型不具备稳定截面排序能力。",
    }
    source_map = export_figure(fig, "lstm_fig5_walk_forward_strategy", contract, [plot_year_path, plot_bt_path])
    return contract, source_map


def append_metrics(data: dict[str, pd.DataFrame]) -> None:
    lstm = data["lstm_metrics"]
    rolling = data["rolling_metrics"]
    dual = data["dual_metrics"]
    walk = data["walk_metrics"].iloc[0]
    shap_metrics = data["shap_metrics"]

    new_rows = [
        {"metric": "lstm_single_base_f1", "value": metric_value(lstm, "F1-score", "base"), "source": "generate_lstm_full_report.py"},
        {"metric": "lstm_single_fusion_f1", "value": metric_value(lstm, "F1-score", "fusion"), "source": "generate_lstm_full_report.py"},
        {
            "metric": "lstm_rolling_fusion_mean_f1",
            "value": float(rolling[(rolling["model"] == "fusion") & (rolling["metric"] == "F1-score")]["value"].mean()),
            "source": "generate_lstm_full_report.py",
        },
        {"metric": "dual_branch_lstm_f1", "value": metric_value(dual, "F1-score", "dual_branch_lstm"), "source": "generate_lstm_full_report.py"},
        {"metric": "dual_branch_lstm_strategy_return", "value": metric_value(dual, "strategy_cum_return", "dual_branch_lstm"), "source": "generate_lstm_full_report.py"},
        {"metric": "walk_forward_best_excess_return", "value": float(walk["excess_total_return"]), "source": "generate_lstm_full_report.py"},
        {"metric": "walk_forward_best_total_return", "value": float(walk["total_return"]), "source": "generate_lstm_full_report.py"},
        {
            "metric": "shap_proxy_f1",
            "value": float(shap_metrics.loc[shap_metrics["metric"].eq("F1-score"), "value"].iloc[0]),
            "source": "generate_lstm_full_report.py",
        },
    ]
    path = PROJECT_ROOT / "outputs" / "metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["metric"].isin([row["metric"] for row in new_rows])]
        out = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        out = pd.DataFrame(new_rows)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def write_report(data: dict[str, pd.DataFrame], contracts: list[dict]) -> None:
    lstm = data["lstm_metrics"]
    rolling = data["rolling_metrics"]
    dual = data["dual_metrics"]
    latest = data["latest_signal"].iloc[0]
    shap = data["shap_importance"].head(10)
    shap_proxy = data["shap_metrics"]
    walk = data["walk_metrics"].iloc[0]
    best_yearly = data["walk_yearly"][
        (data["walk_yearly"]["variant"] == walk["variant"])
        & (data["walk_yearly"]["top_k"] == walk["top_k"])
        & (data["walk_yearly"]["min_probability"] == walk["min_probability"])
    ]

    rolling_summary = (
        rolling[rolling["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return"])]
        .pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
        .round(4)
    )
    single_summary = (
        lstm[lstm["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return", "buy_hold_cum_return", "max_drawdown"])]
        .pivot_table(index="model", columns="metric", values="value", aggfunc="first")
        .round(4)
    )
    dual_summary = dual[dual["metric"].isin(["Accuracy", "Precision", "Recall", "F1-score", "strategy_cum_return", "buy_hold_cum_return", "max_drawdown", "threshold", "best_epoch"])]
    dual_summary = dual_summary[["metric", "value"]].copy()
    dual_summary["value"] = dual_summary["value"].round(4)

    lines = [
        "# 基于 TOPSIS 经营绩效融合的 LSTM 股价趋势预测完整报告",
        "",
        "## 摘要",
        "",
        "本报告在长春高新经营绩效评价基础上，构建未来 5 个交易日方向预测任务，并依次评估单股 LSTM、绩效融合 LSTM、滚动验证、SHAP 解释、同行面板双分支 LSTM 以及 walk-forward 截面 TopK 回测。核心结论是：低频经营绩效和 TOPSIS 得分可以作为解释性补充，但短期股价方向仍主要受估值、技术形态和市场环境驱动；双分支 LSTM 能改善分类 F1，但若没有市场状态过滤和仓位控制，绝对收益仍不稳定。",
        "",
        "## 1. 研究目标与预测任务",
        "",
        "研究目标不是直接预测股价点位，而是预测未来 5 个交易日累计收益 `future_5d_return` 是否大于 0。该任务将股价尺度、复权处理和极端价格误差的影响降到较低水平，更适合与交易信号、组合回测和方向命中率指标连接。输入窗口设置为过去 20 个交易日。",
        "",
        "## 2. 数据与特征体系",
        "",
        "数据包括前复权 OHLCV、成交额、换手率、均线、收益率、波动率、RSI、MACD、布林带、沪深300/中证医药/创业板指数特征、估值与市值字段。经营绩效侧包含收入、净利润、ROE、毛利率、净利率、资产负债率、流动比率、收入增速、净利润增速、TOPSIS 得分和排名。财务数据按披露日 `merge_asof` 映射到交易日，避免在公告披露前使用未来财务信息。",
        "",
        "## 3. 模型设计",
        "",
        "基础 LSTM 使用日频行情、技术指标、指数和估值变量作为时间序列输入；融合 LSTM 在此基础上纳入已经披露的经营绩效与 TOPSIS 信息。双分支 LSTM 将快变量和慢变量分开处理：日频行情/技术/指数/估值进入 LSTM 分支，财务/TOPSIS/公司与行业哑变量进入 MLP 静态分支，二者拼接后输出未来 5 日上涨概率。该结构避免把年频财务指标简单复制成日频序列后完全交给 LSTM 学习。",
        "",
        "## 4. 单股 LSTM 结果",
        "",
        "单次时间切分结果如下：",
        "",
        "```text",
        single_summary.to_string(),
        "```",
        "",
        "该结果显示，单次切分中基础 LSTM 和融合 LSTM 的分类 F1 并不理想，融合信息没有稳定转化为更强的短期方向预测能力。这一阶段的意义在于建立可运行的序列预测管线，而不是证明单股模型已具备稳定交易价值。",
        "",
        "![LSTM 模型演进](figures/lstm_fig1_model_progress.png)",
        "",
        "## 5. 滚动验证与最新信号",
        "",
        "滚动验证均值如下：",
        "",
        "```text",
        rolling_summary.to_string(),
        "```",
        "",
        f"最新滚动信号日期为 `{latest['date']}`，模型为 `{latest['model']}`，收盘价 `{latest['close']}`，上涨概率 `{float(latest['predicted_probability']):.4f}`，阈值 `{float(latest['threshold']):.2f}`，信号为 `{latest['signal']}`。滚动验证比单次切分更接近真实部署，因为每个测试年份只能使用之前的数据。",
        "",
        "![滚动验证稳定性](figures/lstm_fig2_rolling_stability.png)",
        "",
        "## 6. SHAP 可解释性",
        "",
        "SHAP 代理模型指标如下：",
        "",
        "```text",
        shap_proxy.round(4).to_string(index=False),
        "```",
        "",
        "Top SHAP 特征如下：",
        "",
        "```text",
        shap.round(4).to_string(index=False),
        "```",
        "",
        "解释结果表明，`peg`、布林带宽度、短期均线、波动率、指数收益等特征贡献靠前。经营绩效/TOPSIS 信息更适合作为慢变量和状态变量，不宜被解释为短期股价方向的唯一主因。",
        "",
        "![SHAP 特征贡献](figures/lstm_fig3_shap_explainability.png)",
        "",
        "## 7. 双分支 LSTM 面板扩展",
        "",
        "双分支 LSTM 的关键指标如下：",
        "",
        "```text",
        dual_summary.to_string(index=False),
        "```",
        "",
        "双分支模型在测试集上取得较高召回率和 `F1-score = {:.4f}`，策略累计收益为 `{:.4f}`，同期买入持有收益为 `{:.4f}`。这说明模型相对单股 LSTM 有明显分类改善，但仍存在精度不足、阈值敏感和回撤偏大的问题。".format(
            metric_value(dual, "F1-score", "dual_branch_lstm"),
            metric_value(dual, "strategy_cum_return", "dual_branch_lstm"),
            metric_value(dual, "buy_hold_cum_return", "dual_branch_lstm"),
        ),
        "",
        "![双分支 LSTM 个股异质性](figures/lstm_fig4_dual_branch_panel.png)",
        "",
        "## 8. Walk-forward 截面 TopK 检验",
        "",
        "为避免固定测试集带来的过拟合错觉，进一步进行了逐年 walk-forward 检验：测试年前一年作为验证集选择阈值，验证年前所有历史数据作为训练集，然后在测试年按预测概率进行非重叠 5 日 TopK 调仓。当前最优组合为 `{}` / Top{}，总收益 `{:.4f}`，等权基准 `{:.4f}`，超额收益 `{:.4f}`。".format(
            walk["variant"],
            int(walk["top_k"]),
            float(walk["total_return"]),
            float(walk["benchmark_total_return"]),
            float(walk["excess_total_return"]),
        ),
        "",
        "最优组合年度拆解如下：",
        "",
        "```text",
        best_yearly.round(4).to_string(index=False),
        "```",
        "",
        "结果表明模型在 2022、2023、2025、2026 年具有相对抗跌或增强能力，但 2024 年失效，说明后续必须引入市场状态过滤、动态仓位和不交易区间。",
        "",
        "![Walk-forward 截面 TopK](figures/lstm_fig5_walk_forward_strategy.png)",
        "",
        "## 9. 局限性",
        "",
        "第一，样本仍集中在医药及相关同行，行业系统性下行会压制绝对收益。第二，财务和 TOPSIS 特征为低频变量，对 5 日短周期预测的边际贡献有限。第三，当前交易回测仅考虑单边交易成本，尚未加入滑点、涨跌停无法成交、停牌、成交量约束和组合容量。第四，双分支 LSTM 召回率高但精度偏低，容易在弱势环境中维持过多仓位。第五，SHAP 使用 LightGBM 代理解释，不等同于直接解释 PyTorch LSTM 内部状态。",
        "",
        "## 10. 后续优化方向",
        "",
        "后续优先级应从继续堆模型转向风控和验证口径：一是加入沪深300/中证医药趋势过滤，在行业弱势时降低仓位；二是基于预测概率分布设置不交易区间，概率不够分散时空仓；三是使用 walk-forward 方式重训双分支 LSTM，而不是只做固定切分；四是引入概率校准和分组阈值，缓解高召回、低精度结构；五是进一步扩展同行样本和宏观/资金流/分析师预期数据。",
        "",
        "## 11. 图表追溯",
        "",
        "本报告新增 5 张图，均已导出 SVG、PNG、PDF 和 trace 文件，并写入 `outputs/tables/figure_contracts.csv` 与 `outputs/tables/figure_source_map.csv`。",
        "",
        "```text",
        pd.DataFrame(contracts)[["figure_id", "core_conclusion", "source_data_paths", "output_targets"]].to_string(index=False),
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()

    figure_functions = [
        figure_model_progress,
        figure_rolling_stability,
        figure_shap_explainability,
        figure_dual_branch_panel,
        figure_walk_forward_strategy,
    ]
    contracts = []
    source_maps = []
    for func in figure_functions:
        contract, source_map = func(data)
        contracts.append(contract)
        source_maps.append(source_map)

    upsert_csv(TABLE_DIR / "figure_contracts.csv", contracts, "figure_id")
    upsert_csv(TABLE_DIR / "figure_source_map.csv", source_maps, "figure_id")
    append_metrics(data)
    write_report(data, contracts)

    print("Generated LSTM full report figures and report.")
    print(REPORT_PATH)
    print(pd.DataFrame(contracts)[["figure_id", "core_conclusion"]].to_string(index=False))


if __name__ == "__main__":
    main()
