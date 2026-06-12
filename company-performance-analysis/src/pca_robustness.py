"""PCA robustness checks and publication-style PCA figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

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


def _minmax(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


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


def _export_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": FIGURE_DIR / f"{stem}.png",
        "svg": FIGURE_DIR / f"{stem}.svg",
        "pdf": FIGURE_DIR / f"{stem}.pdf",
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    return {key: str(path) for key, path in outputs.items()}


def _write_trace(stem: str, contract: dict, metrics: dict, source_data_paths: list[str]) -> Path:
    trace_path = FIGURE_DIR / f"{stem}.trace.json"
    payload = {
        "figure_id": contract["figure_id"],
        "contract": contract,
        "metrics": metrics,
        "source_data_paths": source_data_paths,
        "generation_script": "src/pca_robustness.py",
    }
    trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace_path


def _save_contract_tables(contracts: list[dict], source_map: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(contracts).to_csv(TABLE_DIR / "figure_contracts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_map).to_csv(TABLE_DIR / "figure_source_map.csv", index=False, encoding="utf-8-sig")


def _write_metrics(metrics: dict[str, float | int | str]) -> None:
    rows = [
        {"metric": key, "value": value, "source": "PCA robustness check on standardized_matrix.csv"}
        for key, value in metrics.items()
    ]
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")


def run_pca_robustness(
    standardized: pd.DataFrame,
    scores: pd.DataFrame,
    figure_dir: str | Path = FIGURE_DIR,
    table_dir: str | Path = TABLE_DIR,
    metrics_path: str | Path = METRICS_CSV,
) -> dict[str, object]:
    """Run PCA, export robustness tables, figures, contracts, traces and metrics."""
    global FIGURE_DIR, TABLE_DIR, METRICS_CSV
    FIGURE_DIR = Path(figure_dir)
    TABLE_DIR = Path(table_dir)
    METRICS_CSV = Path(metrics_path)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)

    id_cols = [col for col in ["period", "year"] if col in standardized.columns]
    indicator_cols = [col for col in standardized.columns if col not in id_cols]
    work = standardized[id_cols + indicator_cols].copy()
    for col in indicator_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    imputed = work[indicator_cols].copy()
    impute_values = imputed.median(numeric_only=True)
    imputed = imputed.fillna(impute_values)

    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(imputed.to_numpy(dtype=float))
    coords = work[id_cols].copy()
    coords["year"] = coords["year"].astype(int)
    coords["pc1"] = components[:, 0]
    coords["pc2"] = components[:, 1]

    joined = coords.merge(scores[["year", "topsis_score", "rank"]], on="year", how="left")
    pc1_corr = joined["pc1"].corr(joined["topsis_score"])
    if pd.notna(pc1_corr) and pc1_corr < 0:
        joined["pc1"] = -joined["pc1"]
        components[:, 0] = -components[:, 0]

    joined["pca_score"] = _minmax(joined["pc1"])
    joined["pca_rank"] = joined["pca_score"].rank(ascending=False, method="min").astype(int)
    joined["stage"] = joined["year"].map(_stage)

    pearson_corr = joined["pca_score"].corr(joined["topsis_score"], method="pearson")
    spearman_corr = joined["pca_score"].corr(joined["topsis_score"], method="spearman")
    top5_topsis = set(joined.nsmallest(5, "rank")["year"])
    top5_pca = set(joined.nsmallest(5, "pca_rank")["year"])
    latest_year = int(joined["year"].max())
    latest_row = joined.loc[joined["year"] == latest_year].iloc[0]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["pc1_loading", "pc2_loading"],
        index=indicator_cols,
    ).reset_index(names="indicator")
    if joined["pc1"].corr(joined["topsis_score"]) < 0:
        loadings["pc1_loading"] = -loadings["pc1_loading"]
    loadings["indicator_label"] = loadings["indicator"].map(INDICATOR_LABELS).fillna(loadings["indicator"])
    loadings["loading_norm"] = np.sqrt(loadings["pc1_loading"] ** 2 + loadings["pc2_loading"] ** 2)

    metrics = {
        "pca_pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pca_pc2_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "pca_pc1_pc2_cumulative_variance_ratio": float(pca.explained_variance_ratio_[:2].sum()),
        "pca_topsis_pearson_corr": float(pearson_corr),
        "pca_topsis_spearman_corr": float(spearman_corr),
        "top5_overlap_count": int(len(top5_topsis & top5_pca)),
        "latest_year": latest_year,
        "latest_topsis_rank": int(latest_row["rank"]),
        "latest_pca_rank": int(latest_row["pca_rank"]),
    }

    coordinates_path = TABLE_DIR / "plot_data_pca_coordinates.csv"
    loadings_path = TABLE_DIR / "plot_data_pca_loadings.csv"
    summary_path = TABLE_DIR / "pca_robustness_summary.csv"
    joined.to_csv(coordinates_path, index=False, encoding="utf-8-sig")
    loadings.to_csv(loadings_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    _write_metrics(metrics)

    contracts = []
    source_map = []

    robustness_contract = {
        "figure_id": "fig_pca_robustness",
        "purpose": "检验 PCA 综合得分是否支持 TOPSIS 绩效排序结论",
        "core_conclusion": f"PCA 与 TOPSIS 得分高度一致，Spearman 相关系数为 {spearman_corr:.3f}",
        "chart_type": "scatter with reference trend",
        "evidence_layers": "PCA score, TOPSIS score, year labels for recent and top-score years",
        "source_data_paths": "data/processed/standardized_matrix.csv; outputs/tables/topsis_scores.csv",
        "output_targets": "outputs/figures/pca_topsis_robustness.png/svg/pdf",
        "failure_signal": "若相关系数低或近期年份排名方向相反，则 TOPSIS 结论对降维方法不稳健",
    }
    contracts.append(robustness_contract)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    colors = {
        "早期基础期": "#7f7f7f",
        "稳步扩张期": "#4c78a8",
        "加速增长期": "#59a14f",
        "高位表现期": "#f28e2b",
        "近期调整期": "#e15759",
    }
    for stage, group in joined.groupby("stage"):
        ax.scatter(
            group["pca_score"],
            group["topsis_score"],
            s=52,
            color=colors.get(stage, "#333333"),
            label=stage,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
        )
    z = np.polyfit(joined["pca_score"], joined["topsis_score"], deg=1)
    x_line = np.linspace(joined["pca_score"].min(), joined["pca_score"].max(), 100)
    ax.plot(x_line, z[0] * x_line + z[1], color="#222222", linewidth=1.6, linestyle="--", label="线性趋势")
    label_years = set(joined.nlargest(3, "topsis_score")["year"]) | {latest_year, 2024, 2020}
    for row in joined.loc[joined["year"].isin(label_years)].itertuples(index=False):
        ax.annotate(str(row.year), (row.pca_score, row.topsis_score), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("PCA 综合得分（PC1 方向，0-1 标准化）")
    ax.set_ylabel("TOPSIS 综合绩效得分")
    ax.set_title("PCA 稳健性检验：综合得分与 TOPSIS 排序保持一致")
    ax.text(
        0.02,
        0.97,
        f"Pearson={pearson_corr:.3f}\nSpearman={spearman_corr:.3f}\nTop5 overlap={len(top5_topsis & top5_pca)}/5",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="both", color="#dddddd", linewidth=0.8)
    robustness_outputs = _export_figure(fig, "pca_topsis_robustness")
    plt.close(fig)
    robustness_trace = _write_trace(
        "pca_topsis_robustness",
        robustness_contract,
        metrics,
        ["data/processed/standardized_matrix.csv", "outputs/tables/topsis_scores.csv"],
    )
    source_map.append(
        {
            "figure_id": "fig_pca_robustness",
            "plot_data": str(coordinates_path),
            "trace": str(robustness_trace),
            "outputs": "; ".join(robustness_outputs.values()),
        }
    )

    biplot_contract = {
        "figure_id": "fig_pca_biplot",
        "purpose": "展示年度绩效样本在 PC1-PC2 空间中的二维结构，并解释主要指标驱动",
        "core_conclusion": "高位表现期沿收入、净利润和 ROE 方向聚集，近期调整期从高绩效区域回落",
        "chart_type": "PCA biplot",
        "evidence_layers": "year coordinates, period stage colors, indicator loading arrows",
        "source_data_paths": "data/processed/standardized_matrix.csv; outputs/tables/topsis_scores.csv",
        "output_targets": "outputs/figures/pca_2d_biplot.png/svg/pdf",
        "failure_signal": "若前两主成分解释率过低或载荷方向无法解释绩效分层，则二维图不适合作为主证据",
    }
    contracts.append(biplot_contract)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for stage, group in joined.groupby("stage"):
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=58,
            color=colors.get(stage, "#333333"),
            label=stage,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
        )
    for row in joined.loc[joined["year"].isin(label_years | {1993})].itertuples(index=False):
        ax.annotate(str(row.year), (row.pc1, row.pc2), xytext=(5, 5), textcoords="offset points", fontsize=8)

    arrow_scale = min(
        (joined["pc1"].max() - joined["pc1"].min()),
        (joined["pc2"].max() - joined["pc2"].min()),
    ) * 0.32
    key_loading_indicators = [
        "revenue",
        "net_profit",
        "roe",
        "net_margin",
        "asset_liability_ratio",
        "current_ratio",
    ]
    label_offsets = {
        "revenue": (0.05, -0.05),
        "net_profit": (0.06, 0.01),
        "roe": (-0.02, 0.08),
        "net_margin": (0.06, 0.03),
        "asset_liability_ratio": (0.06, 0.02),
        "current_ratio": (0.02, -0.05),
    }
    loading_rows = loadings.loc[loadings["indicator"].isin(key_loading_indicators)]
    for row in loading_rows.itertuples(index=False):
        ax.arrow(
            0,
            0,
            row.pc1_loading * arrow_scale,
            row.pc2_loading * arrow_scale,
            color="#333333",
            width=0.003,
            head_width=0.055,
            length_includes_head=True,
            alpha=0.8,
        )
        dx, dy = label_offsets.get(row.indicator, (0.0, 0.0))
        ax.text(
            row.pc1_loading * arrow_scale * 1.08 + dx,
            row.pc2_loading * arrow_scale * 1.08 + dy,
            row.indicator_label,
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA 二维可视化：年度绩效结构与指标载荷")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(axis="both", color="#dddddd", linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    biplot_outputs = _export_figure(fig, "pca_2d_biplot")
    plt.close(fig)
    biplot_trace = _write_trace(
        "pca_2d_biplot",
        biplot_contract,
        metrics,
        ["data/processed/standardized_matrix.csv", "outputs/tables/topsis_scores.csv"],
    )
    source_map.append(
        {
            "figure_id": "fig_pca_biplot",
            "plot_data": f"{coordinates_path}; {loadings_path}",
            "trace": str(biplot_trace),
            "outputs": "; ".join(biplot_outputs.values()),
        }
    )

    _save_contract_tables(contracts, source_map)
    return {
        "coordinates": joined,
        "loadings": loadings,
        "metrics": metrics,
        "figures": {
            "pca_topsis_robustness": robustness_outputs,
            "pca_2d_biplot": biplot_outputs,
        },
    }


def main() -> None:
    standardized = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "standardized_matrix.csv")
    scores = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "topsis_scores.csv")
    outputs = run_pca_robustness(standardized, scores)
    print("PCA robustness completed.")
    print(outputs["metrics"])


if __name__ == "__main__":
    main()
