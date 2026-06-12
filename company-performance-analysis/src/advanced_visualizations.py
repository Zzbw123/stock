"""Additional robustness and explanatory figures for the performance report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from entropy_topsis import entropy_weights, topsis_score
from indicators import INDICATOR_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
METRICS_CSV = PROJECT_ROOT / "outputs" / "metrics.csv"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _stage(year: int) -> str:
    if year <= 2005:
        return "早期基础期"
    if year <= 2015:
        return "稳步扩张期"
    if year <= 2020:
        return "加速增长期"
    if year <= 2023:
        return "高位表现期"
    return "近期调整期"


def _indicator_label(indicator: str) -> str:
    return INDICATOR_LABELS.get(indicator, indicator)


def _export_figure(fig: plt.Figure, stem: str, figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": figure_dir / f"{stem}.png",
        "svg": figure_dir / f"{stem}.svg",
        "pdf": figure_dir / f"{stem}.pdf",
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    return {key: str(path) for key, path in outputs.items()}


def _write_trace(stem: str, contract: dict, metrics: dict, source_data_paths: list[str], figure_dir: Path) -> Path:
    trace_path = figure_dir / f"{stem}.trace.json"
    payload = {
        "figure_id": contract["figure_id"],
        "contract": contract,
        "metrics": metrics,
        "source_data_paths": source_data_paths,
        "generation_script": "src/advanced_visualizations.py",
    }
    trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace_path


def _append_contract_tables(contracts: list[dict], source_map: list[dict], table_dir: Path) -> None:
    contract_path = table_dir / "figure_contracts.csv"
    source_map_path = table_dir / "figure_source_map.csv"
    new_contracts = pd.DataFrame(contracts)
    new_source_map = pd.DataFrame(source_map)

    if contract_path.exists():
        old = pd.read_csv(contract_path)
        new_contracts = pd.concat([old, new_contracts], ignore_index=True)
        new_contracts = new_contracts.drop_duplicates("figure_id", keep="last")
    if source_map_path.exists():
        old = pd.read_csv(source_map_path)
        new_source_map = pd.concat([old, new_source_map], ignore_index=True)
        new_source_map = new_source_map.drop_duplicates("figure_id", keep="last")

    new_contracts.to_csv(contract_path, index=False, encoding="utf-8-sig")
    new_source_map.to_csv(source_map_path, index=False, encoding="utf-8-sig")


def _append_metrics(metrics: dict[str, float | int | str], metrics_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {"metric": key, "value": value, "source": "advanced_visualizations.py"}
            for key, value in metrics.items()
        ]
    )
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        rows = pd.concat([old, rows], ignore_index=True)
        rows = rows.drop_duplicates("metric", keep="last")
    rows.to_csv(metrics_path, index=False, encoding="utf-8-sig")


def _save_figure_artifacts(
    fig: plt.Figure,
    stem: str,
    contract: dict,
    plot_data_path: Path | str,
    metrics: dict,
    source_data_paths: list[str],
    figure_dir: Path,
) -> tuple[dict[str, str], Path, dict]:
    outputs = _export_figure(fig, stem, figure_dir)
    plt.close(fig)
    trace = _write_trace(stem, contract, metrics, source_data_paths, figure_dir)
    source_row = {
        "figure_id": contract["figure_id"],
        "plot_data": str(plot_data_path),
        "trace": str(trace),
        "outputs": "; ".join(outputs.values()),
    }
    return outputs, trace, source_row


def run_advanced_visualizations(
    standardized: pd.DataFrame,
    weights: pd.DataFrame,
    scores: pd.DataFrame,
    pca_loadings: pd.DataFrame | None = None,
    figure_dir: str | Path = FIGURE_DIR,
    table_dir: str | Path = TABLE_DIR,
    metrics_path: str | Path = METRICS_CSV,
) -> dict[str, object]:
    """Generate recommended robustness, explanation and phase-structure figures."""
    figure_dir = Path(figure_dir)
    table_dir = Path(table_dir)
    metrics_path = Path(metrics_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    id_cols = [col for col in ["period", "year"] if col in standardized.columns]
    indicator_cols = [col for col in standardized.columns if col not in id_cols]
    data = standardized[id_cols + indicator_cols].copy()
    data["year"] = data["year"].astype(int)
    for col in indicator_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    score_by_year = scores[["year", "topsis_score", "rank"]].copy()
    score_by_year["year"] = score_by_year["year"].astype(int)
    latest_year = int(data["year"].max())
    baseline_latest = score_by_year.loc[score_by_year["year"] == latest_year].iloc[0]
    baseline_top5 = set(score_by_year.nsmallest(5, "rank")["year"])
    weight_lookup = weights.set_index("indicator")["weight"]

    contracts: list[dict] = []
    source_map: list[dict] = []
    metrics: dict[str, float | int | str] = {}
    figure_outputs: dict[str, dict[str, str]] = {}

    # 1. Leave-one-indicator sensitivity
    sensitivity_rows = []
    for removed in indicator_cols:
        kept = [col for col in indicator_cols if col != removed]
        sub_standardized = data[id_cols + kept].copy()
        sub_weights = entropy_weights(sub_standardized, id_columns=id_cols)
        sub_scores = topsis_score(sub_standardized, sub_weights, id_columns=id_cols)
        sub_scores["year"] = sub_scores["year"].astype(int)
        latest_sub = sub_scores.loc[sub_scores["year"] == latest_year].iloc[0]
        joined = score_by_year.merge(sub_scores[["year", "topsis_score", "rank"]], on="year", suffixes=("_base", "_loo"))
        sensitivity_rows.append(
            {
                "removed_indicator": removed,
                "removed_indicator_label": _indicator_label(removed),
                "latest_rank_after_removal": int(latest_sub["rank"]),
                "latest_rank_change": int(latest_sub["rank"] - baseline_latest["rank"]),
                "latest_score_after_removal": float(latest_sub["topsis_score"]),
                "latest_score_change": float(latest_sub["topsis_score"] - baseline_latest["topsis_score"]),
                "spearman_vs_baseline": float(joined["topsis_score_base"].corr(joined["topsis_score_loo"], method="spearman")),
                "top5_overlap": len(baseline_top5 & set(sub_scores.nsmallest(5, "rank")["year"])),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values("latest_rank_change", ascending=False)
    sensitivity_path = table_dir / "plot_data_leave_one_indicator_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False, encoding="utf-8-sig")
    metrics.update(
        {
            "sensitivity_min_spearman": float(sensitivity["spearman_vs_baseline"].min()),
            "sensitivity_latest_rank_max_abs_change": int(sensitivity["latest_rank_change"].abs().max()),
            "sensitivity_min_top5_overlap": int(sensitivity["top5_overlap"].min()),
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    plot_df = sensitivity.sort_values("latest_rank_change")
    colors = np.where(plot_df["latest_rank_change"] > 0, "#d62728", "#2a9d8f")
    ax.barh(plot_df["removed_indicator_label"], plot_df["latest_rank_change"], color=colors, alpha=0.85)
    zero_rows = plot_df.loc[plot_df["latest_rank_change"] == 0]
    ax.scatter(
        np.zeros(len(zero_rows)),
        zero_rows["removed_indicator_label"],
        s=36,
        color="#555555",
        zorder=4,
        label="排名不变",
    )
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel(f"去除指标后 {latest_year} 年排名变化（正值=变差）")
    ax.set_title("留一指标敏感性：最新年度排名对单项指标不敏感")
    for row in plot_df.itertuples(index=False):
        ax.text(row.latest_rank_change + (0.08 if row.latest_rank_change >= 0 else -0.08), row.removed_indicator_label, f"{row.spearman_vs_baseline:.2f}", va="center", ha="left" if row.latest_rank_change >= 0 else "right", fontsize=8)
    ax.text(0.02, 0.02, "条形为排名变化，数字为与原 TOPSIS 得分的 Spearman 相关", transform=ax.transAxes, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    contract = {
        "figure_id": "fig_leave_one_indicator_sensitivity",
        "purpose": "检验 TOPSIS 结论是否被某一个指标主导",
        "core_conclusion": f"去除任一指标后，{latest_year} 年排名最大变化为 {int(sensitivity['latest_rank_change'].abs().max())} 位，排序结论较稳健",
        "chart_type": "horizontal bar sensitivity plot",
        "evidence_layers": "leave-one-indicator rank change, Spearman correlation labels",
        "source_data_paths": "data/processed/standardized_matrix.csv; outputs/tables/topsis_scores.csv",
        "output_targets": "outputs/figures/leave_one_indicator_sensitivity.png/svg/pdf",
        "failure_signal": "若去掉某一指标后最新年度排名大幅改善或相关性显著降低，则模型被单一指标主导",
    }
    outputs, _, source_row = _save_figure_artifacts(fig, "leave_one_indicator_sensitivity", contract, sensitivity_path, metrics, ["data/processed/standardized_matrix.csv", "outputs/tables/topsis_scores.csv"], figure_dir)
    contracts.append(contract)
    source_map.append(source_row)
    figure_outputs["leave_one_indicator_sensitivity"] = outputs

    # 2. Indicator heatmap
    heatmap = data[["year"] + indicator_cols].copy()
    heatmap_path = table_dir / "plot_data_indicator_heatmap.csv"
    heatmap.to_csv(heatmap_path, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    matrix = heatmap[indicator_cols].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(indicator_cols)))
    ax.set_xticklabels([_indicator_label(col) for col in indicator_cols], rotation=35, ha="right")
    ax.set_yticks(range(len(heatmap)))
    ax.set_yticklabels(heatmap["year"].astype(str), fontsize=8)
    ax.set_title("指标热力图：高位年份呈现多指标同步改善")
    ax.set_xlabel("标准化指标（数值越高越好）")
    ax.set_ylabel("年份")
    fig.colorbar(im, ax=ax, shrink=0.78, label="标准化值")
    contract = {
        "figure_id": "fig_indicator_heatmap",
        "purpose": "展示各年度在所有标准化指标上的强弱结构",
        "core_conclusion": "2021-2023 年多项指标同步处于高位，2025 年主要在净利润、ROE 和净利率上明显转弱",
        "chart_type": "heatmap",
        "evidence_layers": "year by indicator standardized matrix",
        "source_data_paths": "data/processed/standardized_matrix.csv",
        "output_targets": "outputs/figures/indicator_heatmap.png/svg/pdf",
        "failure_signal": "若高绩效年份没有形成成片高值区域，则综合得分缺少多指标共振证据",
    }
    outputs, _, source_row = _save_figure_artifacts(fig, "indicator_heatmap", contract, heatmap_path, metrics, ["data/processed/standardized_matrix.csv"], figure_dir)
    contracts.append(contract)
    source_map.append(source_row)
    figure_outputs["indicator_heatmap"] = outputs

    # 3. 2023 to 2025 degradation contribution
    compare_years = [2023, latest_year]
    compare = data.loc[data["year"].isin(compare_years), ["year"] + indicator_cols].set_index("year")
    if set(compare_years).issubset(compare.index):
        contribution = pd.DataFrame(
            {
                "indicator": indicator_cols,
                "indicator_label": [_indicator_label(col) for col in indicator_cols],
                "standardized_2023": compare.loc[2023, indicator_cols].to_numpy(dtype=float),
                f"standardized_{latest_year}": compare.loc[latest_year, indicator_cols].to_numpy(dtype=float),
                "weight": weight_lookup.reindex(indicator_cols).fillna(0).to_numpy(dtype=float),
            }
        )
        contribution["standardized_change"] = contribution[f"standardized_{latest_year}"] - contribution["standardized_2023"]
        contribution["weighted_contribution"] = contribution["standardized_change"] * contribution["weight"]
    else:
        contribution = pd.DataFrame(columns=["indicator", "indicator_label", "weighted_contribution"])
    contribution = contribution.sort_values("weighted_contribution")
    contribution_path = table_dir / "plot_data_2023_2025_degradation_contribution.csv"
    contribution.to_csv(contribution_path, index=False, encoding="utf-8-sig")
    metrics["degradation_total_weighted_change_2023_to_latest"] = float(contribution["weighted_contribution"].sum())

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = np.where(contribution["weighted_contribution"] < 0, "#d62728", "#2a9d8f")
    ax.barh(contribution["indicator_label"], contribution["weighted_contribution"], color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel(f"{latest_year} 相对 2023 的加权标准化变化")
    ax.set_title("2023 到 2025 的绩效退化贡献：利润与回报指标是主要拖累")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    contract = {
        "figure_id": "fig_degradation_contribution",
        "purpose": "拆解 2023 高点到最新年度的综合绩效下降来源",
        "core_conclusion": "净利润、ROE、净利率和营收规模回落是 2025 年相对 2023 年退化的主要来源",
        "chart_type": "diverging horizontal bar",
        "evidence_layers": "standardized indicator change weighted by entropy weights",
        "source_data_paths": "data/processed/standardized_matrix.csv; outputs/tables/entropy_weights.csv",
        "output_targets": "outputs/figures/degradation_contribution_2023_2025.png/svg/pdf",
        "failure_signal": "若贡献分散且无主要负向指标，则退化结论不能归因到利润与回报指标",
    }
    outputs, _, source_row = _save_figure_artifacts(fig, "degradation_contribution_2023_2025", contract, contribution_path, metrics, ["data/processed/standardized_matrix.csv", "outputs/tables/entropy_weights.csv"], figure_dir)
    contracts.append(contract)
    source_map.append(source_row)
    figure_outputs["degradation_contribution"] = outputs

    # 4. Entropy weight vs PCA loading
    if pca_loadings is not None and not pca_loadings.empty:
        pca_compare = weights[["indicator", "weight"]].merge(
            pca_loadings[["indicator", "pc1_loading", "loading_norm"]],
            on="indicator",
            how="inner",
        )
        pca_compare["indicator_label"] = pca_compare["indicator"].map(_indicator_label)
        pca_compare["abs_pc1_loading"] = pca_compare["pc1_loading"].abs()
        method_corr = pca_compare["weight"].corr(pca_compare["abs_pc1_loading"], method="spearman")
    else:
        pca_compare = weights[["indicator", "weight"]].copy()
        pca_compare["pc1_loading"] = np.nan
        pca_compare["loading_norm"] = np.nan
        pca_compare["indicator_label"] = pca_compare["indicator"].map(_indicator_label)
        pca_compare["abs_pc1_loading"] = np.nan
        method_corr = np.nan
    pca_compare_path = table_dir / "plot_data_entropy_vs_pca_loading.csv"
    pca_compare.to_csv(pca_compare_path, index=False, encoding="utf-8-sig")
    metrics["entropy_weight_vs_abs_pc1_loading_spearman"] = float(method_corr) if pd.notna(method_corr) else "NA"

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.scatter(pca_compare["weight"], pca_compare["abs_pc1_loading"], s=70, color="#4c78a8", edgecolor="white", linewidth=0.8)
    for row in pca_compare.itertuples(index=False):
        ax.annotate(row.indicator_label, (row.weight, row.abs_pc1_loading), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("熵权法权重")
    ax.set_ylabel("|PC1 载荷|")
    ax.set_title("熵权与 PCA 载荷对比：规模利润指标在两种方法中都靠前")
    ax.text(0.02, 0.97, f"Spearman={method_corr:.3f}" if pd.notna(method_corr) else "Spearman=NA", transform=ax.transAxes, va="top", fontsize=9, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"})
    ax.grid(axis="both", color="#dddddd", linewidth=0.8)
    contract = {
        "figure_id": "fig_entropy_vs_pca_loading",
        "purpose": "比较熵权法和 PCA 对关键指标重要性的判断是否一致",
        "core_conclusion": f"熵权与 PC1 载荷的秩相关为 {method_corr:.3f}，两种方法均强调规模与利润指标" if pd.notna(method_corr) else "PCA 载荷缺失，无法比较两种方法的重要性判断",
        "chart_type": "scatter with labels",
        "evidence_layers": "entropy weights and absolute PC1 loadings by indicator",
        "source_data_paths": "outputs/tables/entropy_weights.csv; outputs/tables/plot_data_pca_loadings.csv",
        "output_targets": "outputs/figures/entropy_vs_pca_loading.png/svg/pdf",
        "failure_signal": "若两种方法的重要性排序完全背离，则综合评价解释需要谨慎",
    }
    outputs, _, source_row = _save_figure_artifacts(fig, "entropy_vs_pca_loading", contract, pca_compare_path, metrics, ["outputs/tables/entropy_weights.csv", "outputs/tables/plot_data_pca_loadings.csv"], figure_dir)
    contracts.append(contract)
    source_map.append(source_row)
    figure_outputs["entropy_vs_pca_loading"] = outputs

    # 5. Stage distribution boxplot
    stage_scores = score_by_year.copy()
    stage_scores["stage"] = stage_scores["year"].map(_stage)
    stage_order = ["早期基础期", "稳步扩张期", "加速增长期", "高位表现期", "近期调整期"]
    stage_scores["stage"] = pd.Categorical(stage_scores["stage"], categories=stage_order, ordered=True)
    stage_path = table_dir / "plot_data_stage_score_distribution.csv"
    stage_scores.to_csv(stage_path, index=False, encoding="utf-8-sig")
    stage_median = stage_scores.groupby("stage", observed=True)["topsis_score"].median()
    metrics["stage_high_period_median_score"] = float(stage_median.get("高位表现期", np.nan))
    metrics["stage_recent_period_median_score"] = float(stage_median.get("近期调整期", np.nan))

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    ordered_data = [stage_scores.loc[stage_scores["stage"] == stage, "topsis_score"].dropna() for stage in stage_order]
    box = ax.boxplot(ordered_data, labels=stage_order, patch_artist=True, widths=0.55, showfliers=False)
    palette = ["#7f7f7f", "#4c78a8", "#59a14f", "#f28e2b", "#e15759"]
    for patch, color in zip(box["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)
    for idx, values in enumerate(ordered_data, start=1):
        x = np.full(len(values), idx) + np.linspace(-0.08, 0.08, len(values)) if len(values) else []
        ax.scatter(x, values, s=38, color=palette[idx - 1], edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_ylabel("TOPSIS 综合绩效得分")
    ax.set_title("分阶段绩效分布：高位表现期显著高于近期调整期")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    contract = {
        "figure_id": "fig_stage_score_distribution",
        "purpose": "比较不同发展阶段的综合绩效得分分布",
        "core_conclusion": "2021-2023 高位表现期得分整体最高，2024-2025 近期调整期明显回落",
        "chart_type": "boxplot with jittered observations",
        "evidence_layers": "stage-level TOPSIS score distribution",
        "source_data_paths": "outputs/tables/topsis_scores.csv",
        "output_targets": "outputs/figures/stage_score_distribution.png/svg/pdf",
        "failure_signal": "若阶段分布高度重叠，则阶段性叙述证据不足",
    }
    outputs, _, source_row = _save_figure_artifacts(fig, "stage_score_distribution", contract, stage_path, metrics, ["outputs/tables/topsis_scores.csv"], figure_dir)
    contracts.append(contract)
    source_map.append(source_row)
    figure_outputs["stage_score_distribution"] = outputs

    # 6. Rank bump chart
    rank_curve = score_by_year.sort_values("year").copy()
    rank_curve_path = table_dir / "plot_data_rank_bump.csv"
    rank_curve.to_csv(rank_curve_path, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(rank_curve["year"], rank_curve["rank"], color="#0f766e", linewidth=2.2, marker="o", markersize=4.5)
    for year in [2020, 2021, 2022, 2023, 2024, latest_year]:
        if year in set(rank_curve["year"]):
            row = rank_curve.loc[rank_curve["year"] == year].iloc[0]
            ax.annotate(str(year), (row["year"], row["rank"]), xytext=(4, -10), textcoords="offset points", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("年份")
    ax.set_ylabel("TOPSIS 排名（越高越好）")
    ax.set_title("排名轨迹：2023 达到样本第一后转入回落")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    ax.grid(axis="both", color="#dddddd", linewidth=0.8)
    contract = {
        "figure_id": "fig_rank_bump",
        "purpose": "展示年度 TOPSIS 排名随时间的变化轨迹",
        "core_conclusion": "公司在 2021-2023 年排名快速上行并于 2023 年居首，2024-2025 年排名回落",
        "chart_type": "bump/rank line chart",
        "evidence_layers": "annual TOPSIS rank series",
        "source_data_paths": "outputs/tables/topsis_scores.csv",
        "output_targets": "outputs/figures/topsis_rank_bump.png/svg/pdf",
        "failure_signal": "若排名轨迹没有明显转折，则阶段性回落叙事不足",
    }
    outputs, _, source_row = _save_figure_artifacts(fig, "topsis_rank_bump", contract, rank_curve_path, metrics, ["outputs/tables/topsis_scores.csv"], figure_dir)
    contracts.append(contract)
    source_map.append(source_row)
    figure_outputs["topsis_rank_bump"] = outputs

    _append_contract_tables(contracts, source_map, table_dir)
    _append_metrics(metrics, metrics_path)

    return {
        "sensitivity": sensitivity,
        "contribution": contribution,
        "stage_scores": stage_scores,
        "rank_curve": rank_curve,
        "entropy_pca_compare": pca_compare,
        "metrics": metrics,
        "figures": figure_outputs,
    }


def main() -> None:
    standardized = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "standardized_matrix.csv")
    weights = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "entropy_weights.csv")
    scores = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "topsis_scores.csv")
    loadings_path = PROJECT_ROOT / "outputs" / "tables" / "plot_data_pca_loadings.csv"
    pca_loadings = pd.read_csv(loadings_path) if loadings_path.exists() else None
    outputs = run_advanced_visualizations(standardized, weights, scores, pca_loadings)
    print("Advanced visualizations completed.")
    print(outputs["metrics"])


if __name__ == "__main__":
    main()
